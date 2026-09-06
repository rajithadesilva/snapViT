"""Preflight checks and held-out orchestration for fusion ablations.

The training driver intentionally owns optimization and model checkpointing.  This
module keeps the two operationally sensitive edges of the unattended campaign in
one place:

* a read-only preflight which validates the machine, checkpoint, datasets, cache,
  and focused test suite before hours of work are scheduled; and
* resumable subprocess orchestration for the run1 confirmation evaluations.

The functions are independent of :mod:`src.training.fusion_ablation`, so the
campaign driver can import them without introducing a circular import.
"""

from __future__ import annotations

import gc
import hashlib
import importlib
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


# Deterministic CUDA GEMM kernels require this to be set before the first CUDA
# context is created.  The shell entrypoint exports the same value, while this
# default also protects direct invocations of the Python campaign driver.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HELDOUT_ROOT = REPO_ROOT / "datasets/tempovine/dataset_tempovine_mar_run1"
DEFAULT_STAGE2_SEEDS = (42, 7, 123)
DEFAULT_FOCUSED_TESTS = (
    "tests.test_ground_encoder_fusion",
    "tests.test_dataset_multiroot",
    "tests.test_fusion_cache",
    "tests.test_fusion_ablation",
    "tests.test_fusion_ablation_driver",
    "tests.test_fusion_campaign_helpers",
    "tests.test_fusion_evaluation",
)
REQUIRED_IMPORTS = ("torch", "torchvision", "numpy", "timm", "tqdm", "matplotlib")

_FUSION_PREFIXES = (
    "ground_encoder.feature_norm.",
    "ground_encoder.token_mlp.",
    "ground_encoder.column_mlp.",
    "ground_encoder.token_attention.",
    "ground_encoder.view_attention.",
)
_FUSION_EXACT_KEYS = {"ground_encoder.column_residual_scale"}
_FULL_STATE_KEYS = ("full_model_state_dict", "model_state_dict", "state_dict")


class PreflightError(RuntimeError):
    """Raised after all preflight checks have run when one or more are fatal."""

    def __init__(self, report: Mapping[str, Any]):
        failed = [check["name"] for check in report.get("checks", []) if check.get("status") == "error"]
        super().__init__(f"Fusion campaign preflight failed: {', '.join(failed)}")
        self.report = dict(report)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(_jsonable(dict(payload)), stream, indent=2, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return payload


def _sha256(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


_SHA256_CACHE: dict[tuple[str, int, int], str] = {}


def _sha256_cached(path: Path) -> str:
    resolved = path.expanduser().resolve()
    stat = resolved.stat()
    key = (str(resolved), stat.st_size, stat.st_mtime_ns)
    if key not in _SHA256_CACHE:
        _SHA256_CACHE[key] = _sha256(resolved)
    return _SHA256_CACHE[key]


def _stable_payload_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        _jsonable(dict(payload)),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_fusion_key(name: str) -> bool:
    return name in _FUSION_EXACT_KEYS or name.startswith(_FUSION_PREFIXES)


def validate_imports(module_names: Sequence[str] = REQUIRED_IMPORTS) -> dict[str, str]:
    """Import every runtime dependency and return its reported version."""
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/snapvit-matplotlib")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp/snapvit-cache")
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
    versions: dict[str, str] = {}
    for name in module_names:
        module = importlib.import_module(name)
        versions[name] = str(getattr(module, "__version__", "available"))
    # These imports also catch project-local syntax/import regressions.
    importlib.import_module("src.models.snapvit")
    importlib.import_module("src.training.fusion_cache")
    importlib.import_module("src.evaluation.evaluation_pipeline")
    importlib.import_module("src.evaluation.fusion_ablation_report")
    return versions


def validate_device(device: str, *, allocate: bool = True) -> dict[str, Any]:
    """Validate a device and probe the fusion kernels in deterministic mode."""
    import torch

    requested = torch.device(device)
    details: dict[str, Any] = {"requested": str(requested)}
    if requested.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device {device!r} was requested, but CUDA is unavailable")
        index = torch.cuda.current_device() if requested.index is None else requested.index
        if index < 0 or index >= torch.cuda.device_count():
            raise RuntimeError(
                f"CUDA index {index} is invalid; this process can see {torch.cuda.device_count()} device(s)"
            )
        details.update(
            index=index,
            device_count=torch.cuda.device_count(),
            name=torch.cuda.get_device_name(index),
            capability=list(torch.cuda.get_device_capability(index)),
        )
        if allocate:
            probe_device = torch.device("cuda", index=index)
            deterministic_before = torch.are_deterministic_algorithms_enabled()
            warn_only_before = torch.is_deterministic_algorithms_warn_only_enabled()
            try:
                torch.use_deterministic_algorithms(True)

                # Exercise the actual operation families used by N0--N9:
                # learned projections, indexed sum/max reductions, and their
                # backward passes.  An unsupported deterministic CUDA kernel
                # therefore fails preflight instead of failing hours later.
                source = torch.randn(6, 4, device=probe_device, requires_grad=True)
                group_index = torch.tensor([0, 0, 1, 1, 2, 2], device=probe_device)
                expanded_index = group_index[:, None].expand_as(source)
                sums = torch.zeros(3, 4, device=probe_device).scatter_add(
                    0, expanded_index, source
                )
                maxima = torch.full(
                    (3, 4), -torch.inf, device=probe_device
                ).scatter_reduce(
                    0,
                    expanded_index,
                    source,
                    reduce="amax",
                    include_self=True,
                )
                projection = torch.nn.Linear(4, 4).to(probe_device)
                objective = sums.square().mean() + maxima.mean() + projection(source).mean()
                objective.backward()
                torch.cuda.synchronize(index)

                details["allocation"] = True
                details["deterministic_fusion_probe"] = True
                del objective, projection, maxima, sums, expanded_index, group_index, source
            finally:
                torch.use_deterministic_algorithms(
                    deterministic_before,
                    warn_only=warn_only_before,
                )
    elif requested.type == "cpu":
        details["allocation"] = True
    else:
        raise ValueError(f"Unsupported campaign device type: {requested.type!r}")
    return details


def _nearest_existing_parent(path: Path) -> Path:
    candidate = path.resolve()
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    if not candidate.exists():
        raise FileNotFoundError(f"No existing parent found for {path}")
    return candidate


def validate_disk_space(run_dir: Path, minimum_free_gb: float = 8.0) -> dict[str, Any]:
    target = _nearest_existing_parent(run_dir)
    usage = shutil.disk_usage(target)
    free_gb = usage.free / (1024**3)
    if free_gb < minimum_free_gb:
        raise RuntimeError(
            f"Only {free_gb:.2f} GiB is free at {target}; at least {minimum_free_gb:.2f} GiB is required"
        )
    return {
        "filesystem_path": str(target),
        "free_gib": free_gb,
        "required_gib": float(minimum_free_gb),
        "total_gib": usage.total / (1024**3),
    }


def _extract_state_dict(payload: Any, checkpoint: Path) -> Mapping[str, Any]:
    if isinstance(payload, Mapping):
        for key in _FULL_STATE_KEYS:
            state = payload.get(key)
            if isinstance(state, Mapping) and state:
                return state
        if payload and all(isinstance(key, str) and hasattr(value, "shape") for key, value in payload.items()):
            return payload
    raise TypeError(f"No full model state dict found in {checkpoint}")


def _load_checkpoint_cpu(path: Path) -> Any:
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except (TypeError, RuntimeError):
        # ``mmap`` is unavailable in older torch releases and for a few legacy
        # serialization layouts.  Falling back preserves checkpoint support.
        return torch.load(path, map_location="cpu", weights_only=False)


def validate_base_checkpoint(
    checkpoint: Path,
    *,
    verify_state_compatibility: bool = True,
    expected_model_name: str = "convnext_base",
    expected_feature_dim: int = 128,
    expected_num_views: int = 8,
) -> dict[str, Any]:
    """Validate the base config and every non-fusion state tensor used by a variant."""
    checkpoint = checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Base checkpoint not found: {checkpoint}")
    config_path = checkpoint.parent / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Base checkpoint config not found: {config_path}")
    config = _read_json(config_path)
    model_name = config.get("model_name", config.get("vit_model"))
    expected = {
        "model_name": expected_model_name,
        "feature_dim": expected_feature_dim,
        "num_ugv_views": expected_num_views,
        "ground_fusion_mode": "mlp",
    }
    actual = {
        "model_name": model_name,
        "feature_dim": config.get("feature_dim"),
        "num_ugv_views": config.get("num_ugv_views"),
        "ground_fusion_mode": config.get("ground_fusion_mode", "avg"),
    }
    mismatches = {key: {"expected": value, "actual": actual[key]} for key, value in expected.items() if actual[key] != value}
    if mismatches:
        raise RuntimeError(f"Base checkpoint configuration is incompatible: {mismatches}")

    details: dict[str, Any] = {
        "checkpoint": str(checkpoint),
        "config": str(config_path),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "checkpoint_sha256": _sha256(checkpoint),
        "model": actual,
        "state_compatibility": "deferred",
    }
    if not verify_state_compatibility:
        return details

    from src.models.snapvit import SnapViT

    resolved = dict(config)
    resolved.update(
        pretrained_backbones=False,
        ground_fusion_mode="mlp",
        ground_fusion_variant="N0",
        use_height_positional_encoding=False,
        num_ugv_views=expected_num_views,
        device="cpu",
    )
    model = SnapViT(resolved)
    convnext = getattr(model.ground_encoder.feature_extractor, "convnext", None)
    if convnext is None or not hasattr(convnext, "stages_2"):
        raise RuntimeError(
            "Current ground backbone does not expose the required ConvNeXT stages_2 module"
        )
    payload = _load_checkpoint_cpu(checkpoint)
    state = _extract_state_dict(payload, checkpoint)
    target = model.state_dict()
    missing: list[str] = []
    mismatched: list[str] = []
    for name, tensor in target.items():
        if _is_fusion_key(name):
            continue
        source = state.get(name)
        if source is None:
            missing.append(name)
        elif tuple(source.shape) != tuple(tensor.shape):
            mismatched.append(f"{name}: {tuple(source.shape)} != {tuple(tensor.shape)}")
    if missing or mismatched:
        raise RuntimeError(
            "Base checkpoint is not compatible with the current model: "
            f"missing={missing[:10]}, mismatched={mismatched[:10]}"
        )
    details.update(
        state_compatibility="ok",
        state_tensor_count=len(state),
        non_fusion_tensor_count=sum(not _is_fusion_key(name) for name in target),
        stage2_module="ground_encoder.feature_extractor.convnext.stages_2",
    )
    del state, payload, model, target
    gc.collect()
    return details


def _metadata_scene_dirs(root: Path) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {root}")
    return sorted(
        path.resolve()
        for path in root.iterdir()
        if path.is_dir() and (path / "metadata.json").is_file()
    )


def _metadata_references(metadata: Mapping[str, Any]) -> Iterable[tuple[str, str]]:
    uav_path = metadata.get("uav_image_path")
    if not isinstance(uav_path, str) or not uav_path:
        raise ValueError("uav_image_path must be a non-empty string")
    yield "uav_image_path", uav_path
    views = metadata.get("ugv_images")
    if not isinstance(views, list) or not views:
        raise ValueError("ugv_images must be a non-empty list")
    for index, view in enumerate(views):
        if not isinstance(view, Mapping):
            raise TypeError(f"ugv_images[{index}] must be an object")
        image_path = view.get("image_path")
        if not isinstance(image_path, str) or not image_path:
            raise ValueError(f"ugv_images[{index}].image_path must be a non-empty string")
        yield f"ugv_images[{index}].image_path", image_path
        if "depth_path" in view:
            depth_path = view["depth_path"]
            if not isinstance(depth_path, str) or not depth_path:
                raise ValueError(f"ugv_images[{index}].depth_path must be a non-empty string")
            yield f"ugv_images[{index}].depth_path", depth_path


def resolve_metadata_reference(scene: Path, owning_root: Path, value: str) -> Path:
    """Resolve one reference without falling through into a different run root."""
    raw = Path(value).expanduser()
    candidates = [raw] if raw.is_absolute() else [scene / raw, owning_root / raw]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(str(candidates[-1]))


def validate_dataset_root(root: Path, *, expected_scene_count: int | None = None) -> dict[str, Any]:
    """Parse every scene metadata file and verify every image/depth reference."""
    root = root.expanduser().resolve()
    scenes = _metadata_scene_dirs(root)
    if not scenes:
        raise ValueError(f"Dataset root contains no scene metadata: {root}")
    if expected_scene_count is not None and len(scenes) != expected_scene_count:
        raise RuntimeError(
            f"Expected {expected_scene_count} scenes in {root}, found {len(scenes)}"
        )
    reference_count = 0
    unique_sources: set[str] = set()
    failures: list[str] = []
    for scene in scenes:
        metadata_path = scene / "metadata.json"
        try:
            metadata = _read_json(metadata_path)
            for field, value in _metadata_references(metadata):
                reference_count += 1
                try:
                    unique_sources.add(str(resolve_metadata_reference(scene, root, value)))
                except FileNotFoundError as error:
                    failures.append(f"{metadata_path}:{field} -> {value!r} ({error})")
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            failures.append(f"{metadata_path}: {error}")
    if failures:
        preview = "\n".join(failures[:20])
        suffix = f"\n... and {len(failures) - 20} more" if len(failures) > 20 else ""
        raise RuntimeError(
            f"{len(failures)} invalid metadata reference(s) under {root}:\n{preview}{suffix}"
        )
    return {
        "root": str(root),
        "scene_count": len(scenes),
        "reference_count": reference_count,
        "unique_source_count": len(unique_sources),
    }


def validate_feature_cache(
    cache_dir: Path,
    *,
    expected_fingerprint: str | None = None,
    required_ground_paths: Sequence[str | os.PathLike[str]] | None = None,
    required_overhead_paths: Sequence[str | os.PathLike[str]] | None = None,
) -> dict[str, Any]:
    """Inspect an existing FP32 mmap cache without mutating or rebuilding it."""
    import numpy as np

    cache_dir = cache_dir.expanduser().resolve()
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.is_file():
        return {"cache_dir": str(cache_dir), "valid": False, "rebuild_required": True, "reason": "absent"}
    try:
        manifest = _read_json(manifest_path)
        if manifest.get("version") != 1:
            raise RuntimeError(f"unsupported cache version {manifest.get('version')!r}")
        if expected_fingerprint and manifest.get("fingerprint") != expected_fingerprint:
            raise RuntimeError("fingerprint mismatch")
        if manifest.get("dtype") != "float32":
            raise RuntimeError(f"manifest dtype is {manifest.get('dtype')!r}, expected 'float32'")
        ground_paths = [str(Path(path).resolve()) for path in manifest.get("ground_paths", [])]
        overhead_paths = [str(Path(path).resolve()) for path in manifest.get("overhead_paths", [])]
        if not ground_paths or not overhead_paths:
            raise RuntimeError("cache path index is empty")
        ground = np.load(cache_dir / "ground_features.npy", mmap_mode="r")
        overhead = np.load(cache_dir / "overhead_features.npy", mmap_mode="r")
        if ground.dtype != np.float32 or overhead.dtype != np.float32:
            raise RuntimeError("feature arrays are not FP32")
        if ground.flags.writeable or overhead.flags.writeable:
            raise RuntimeError("feature arrays did not open read-only")
        if ground.ndim != 4 or overhead.ndim != 4:
            raise RuntimeError(f"expected rank-4 arrays, found {ground.shape} and {overhead.shape}")
        if len(ground) != len(ground_paths) or len(overhead) != len(overhead_paths):
            raise RuntimeError("feature-array length does not match the manifest path index")
        missing_sources = [path for path in ground_paths + overhead_paths if not Path(path).is_file()]
        if missing_sources:
            raise RuntimeError(f"{len(missing_sources)} indexed source image(s) are missing")
        required_ground = {str(Path(path).resolve()) for path in required_ground_paths or ()}
        required_overhead = {str(Path(path).resolve()) for path in required_overhead_paths or ()}
        missing_ground = required_ground - set(ground_paths)
        missing_overhead = required_overhead - set(overhead_paths)
        if missing_ground or missing_overhead:
            raise RuntimeError(
                f"cache lacks {len(missing_ground)} ground and {len(missing_overhead)} overhead source(s)"
            )
        return {
            "cache_dir": str(cache_dir),
            "valid": True,
            "rebuild_required": False,
            "fingerprint": manifest.get("fingerprint"),
            "ground_shape": list(ground.shape),
            "overhead_shape": list(overhead.shape),
        }
    except (OSError, ValueError, TypeError, RuntimeError, json.JSONDecodeError) as error:
        return {
            "cache_dir": str(cache_dir),
            "valid": False,
            "rebuild_required": True,
            "reason": str(error),
        }


def run_focused_unit_tests(
    *,
    python_bin: str = sys.executable,
    test_modules: Sequence[str] = DEFAULT_FOCUSED_TESTS,
    log_path: Path | None = None,
    timeout_seconds: float = 600.0,
) -> dict[str, Any]:
    """Run the focused campaign tests in an isolated Python subprocess."""
    command = [python_bin, "-m", "unittest", *test_modules]
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )
    output = (completed.stdout or "") + (completed.stderr or "")
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(output, encoding="utf-8")
    if completed.returncode:
        tail = "\n".join(output.splitlines()[-30:])
        raise RuntimeError(
            f"Focused unit tests exited with status {completed.returncode}:\n{tail}"
        )
    return {"command": command, "modules": list(test_modules), "returncode": 0}


def run_preflight(
    *,
    run_dir: Path,
    base_checkpoint: Path,
    data_roots: Sequence[Path],
    heldout_root: Path = DEFAULT_HELDOUT_ROOT,
    cache_dir: Path | None = None,
    expected_cache_fingerprint: str | None = None,
    device: str = "cuda:0",
    minimum_free_gb: float = 25.0,
    expected_heldout_scenes: int | None = 553,
    python_bin: str = sys.executable,
    focused_tests: Sequence[str] = DEFAULT_FOCUSED_TESTS,
    run_unit_tests: bool = True,
    dry_run: bool = False,
    raise_on_error: bool = True,
) -> dict[str, Any]:
    """Run every campaign preflight check and persist ``preflight.json``.

    All independent checks run even after one fails, giving unattended launches a
    single actionable report.  Dry runs still validate CUDA availability, paths,
    metadata, and tests, but defer the memory-heavy state/model shape comparison.
    An absent or stale cache is recorded as ``rebuild_required`` rather than a
    fatal error because the cache builder is expected to replace it atomically.
    """
    run_dir = run_dir.expanduser().resolve()
    started = time.time()
    checks: list[dict[str, Any]] = []

    def check(name: str, operation: Callable[[], Any], *, status: str = "ok") -> None:
        check_started = time.time()
        try:
            details = operation()
            checks.append(
                {
                    "name": name,
                    "status": status,
                    "details": _jsonable(details),
                    "duration_seconds": time.time() - check_started,
                }
            )
        except Exception as error:  # collect independent failures before raising
            checks.append(
                {
                    "name": name,
                    "status": "error",
                    "error": f"{type(error).__name__}: {error}",
                    "duration_seconds": time.time() - check_started,
                }
            )

    check("imports", validate_imports)
    check("device", lambda: validate_device(device, allocate=not dry_run))
    check("disk_space", lambda: validate_disk_space(run_dir, minimum_free_gb))
    check(
        "base_checkpoint",
        lambda: validate_base_checkpoint(
            base_checkpoint, verify_state_compatibility=not dry_run
        ),
    )
    for index, root in enumerate(data_roots):
        check(f"training_dataset_{index + 1}", lambda root=Path(root): validate_dataset_root(root))
    check(
        "heldout_dataset",
        lambda: validate_dataset_root(
            Path(heldout_root), expected_scene_count=expected_heldout_scenes
        ),
    )
    if cache_dir is None:
        checks.append(
            {
                "name": "feature_cache",
                "status": "rebuild_required",
                "details": {"reason": "cache directory was not supplied"},
                "duration_seconds": 0.0,
            }
        )
    else:
        cache_details = validate_feature_cache(
            Path(cache_dir), expected_fingerprint=expected_cache_fingerprint
        )
        checks.append(
            {
                "name": "feature_cache",
                "status": "ok" if cache_details["valid"] else "rebuild_required",
                "details": cache_details,
                "duration_seconds": 0.0,
            }
        )
    if run_unit_tests:
        check(
            "focused_unit_tests",
            lambda: run_focused_unit_tests(
                python_bin=python_bin,
                test_modules=focused_tests,
                log_path=run_dir / "preflight_tests.log",
            ),
        )
    else:
        checks.append(
            {
                "name": "focused_unit_tests",
                "status": "skipped",
                "details": {"reason": "run_unit_tests=False"},
                "duration_seconds": 0.0,
            }
        )

    report = {
        "version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dry_run": bool(dry_run),
        "ok": not any(item["status"] == "error" for item in checks),
        "cache_rebuild_required": any(item["status"] == "rebuild_required" for item in checks),
        "duration_seconds": time.time() - started,
        "checks": checks,
    }
    _atomic_json(run_dir / "preflight.json", report)
    if not report["ok"] and raise_on_error:
        raise PreflightError(report)
    return report


@dataclass(frozen=True)
class HeldoutEvaluationSpec:
    """One independently resumable held-out evaluation subprocess."""

    name: str
    variant: str
    seed: int
    checkpoint: Path
    base_checkpoint: Path
    scene_manifest: Path
    scene_manifest_sha256: str
    output_dir: Path
    localization_samples: int
    expected_retrieval_scenes: int
    expected_localization_scenes: int
    canonical_full: bool = False

    def as_json(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


def _selection_values(selection: Mapping[str, Any]) -> list[str]:
    for key in ("finalists", "top_three", "top3", "top_variants"):
        value = selection.get(key)
        if isinstance(value, list) and all(isinstance(item, str) for item in value):
            return list(dict.fromkeys(item.upper() for item in value))
    return []


def _canonical_identity(selection: Mapping[str, Any]) -> tuple[str, int]:
    run = selection.get("canonical_run")
    if not isinstance(run, Mapping):
        raise RuntimeError("stage2_selection.json does not contain canonical_run")
    variant = run.get("variant") or selection.get("winner_variant") or selection.get("winner")
    seed = run.get("seed")
    if not isinstance(variant, str) or seed is None:
        raise RuntimeError("Canonical winner is missing its variant or seed")
    return variant.upper(), int(seed)


def load_or_create_heldout_manifest(
    *,
    run_dir: Path,
    heldout_root: Path,
    expected_scene_count: int = 553,
) -> tuple[Path, str]:
    """Persist and verify the immutable run1 scene identity manifest.

    The manifest is intentionally stored in the campaign directory rather than
    regenerated by each evaluator.  A changed root or scene set is rejected so a
    count-preserving dataset change cannot silently reuse stale per-scene results.
    """
    campaign_root = run_dir.expanduser().resolve()
    data_root = heldout_root.expanduser().resolve()
    scenes = _metadata_scene_dirs(data_root)
    if len(scenes) != int(expected_scene_count):
        raise RuntimeError(
            f"Expected {expected_scene_count} held-out scenes in {data_root}, found {len(scenes)}"
        )
    scene_ids = [f"{data_root.name}/{scene.name}" for scene in scenes]
    payload: dict[str, Any] = {
        "version": 1,
        "data_root": str(data_root),
        "expected_scene_count": int(expected_scene_count),
        "splits": {"test": scene_ids},
    }
    payload["fingerprint"] = _stable_payload_sha256(payload)
    manifest_path = campaign_root / "heldout_run1_manifest.json"
    if manifest_path.is_file():
        existing = _read_json(manifest_path)
        embedded_fingerprint = existing.get("fingerprint")
        unsigned = dict(existing)
        unsigned.pop("fingerprint", None)
        actual_fingerprint = _stable_payload_sha256(unsigned)
        if embedded_fingerprint != actual_fingerprint:
            raise RuntimeError(
                f"Held-out scene manifest fingerprint is invalid: {manifest_path}"
            )
        if existing != payload:
            raise RuntimeError(
                "Held-out run1 scene identities changed for this RUN_DIR; "
                "use a new RUN_DIR rather than mixing evaluation records."
            )
    else:
        _atomic_json(manifest_path, payload)
    return manifest_path, _sha256_cached(manifest_path)


def build_heldout_evaluation_specs(
    *,
    run_dir: Path,
    base_checkpoint: Path,
    heldout_root: Path = DEFAULT_HELDOUT_ROOT,
    seeds: Sequence[int] = DEFAULT_STAGE2_SEEDS,
    expected_run1_scenes: int = 553,
    localization_subset_size: int = 100,
) -> list[HeldoutEvaluationSpec]:
    """Resolve the nine finalist evaluations and canonical full evaluation."""
    root = run_dir.expanduser().resolve()
    stage1_path = root / "stage1_selection.json"
    stage2_path = root / "stage2_selection.json"
    if not stage1_path.is_file() or not stage2_path.is_file():
        raise FileNotFoundError("Held-out evaluation requires stage1_selection.json and stage2_selection.json")
    finalists = _selection_values(_read_json(stage1_path))
    if len(finalists) != 3:
        raise RuntimeError(f"Expected exactly three stage-1 finalists, found {finalists}")
    canonical_variant, canonical_seed = _canonical_identity(_read_json(stage2_path))
    if canonical_variant not in finalists or canonical_seed not in {int(seed) for seed in seeds}:
        raise RuntimeError(
            f"Canonical run {canonical_variant}/seed_{canonical_seed} is not one of the finalist runs"
        )
    heldout_root = heldout_root.expanduser().resolve()
    base_checkpoint = base_checkpoint.expanduser().resolve()
    scene_manifest, scene_manifest_sha256 = load_or_create_heldout_manifest(
        run_dir=root,
        heldout_root=heldout_root,
        expected_scene_count=expected_run1_scenes,
    )
    specs: list[HeldoutEvaluationSpec] = []
    expected_subset = (
        expected_run1_scenes
        if localization_subset_size <= 0
        else min(localization_subset_size, expected_run1_scenes)
    )
    for variant in finalists:
        for seed_value in seeds:
            seed = int(seed_value)
            experiment = root / "stage2" / variant / f"seed_{seed}"
            specs.append(
                HeldoutEvaluationSpec(
                    name=f"{variant}_seed_{seed}",
                    variant=variant,
                    seed=seed,
                    checkpoint=experiment / "best_adapter.pth",
                    base_checkpoint=base_checkpoint,
                    scene_manifest=scene_manifest,
                    scene_manifest_sha256=scene_manifest_sha256,
                    output_dir=experiment / "evaluation",
                    localization_samples=localization_subset_size,
                    expected_retrieval_scenes=expected_run1_scenes,
                    expected_localization_scenes=expected_subset,
                )
            )

    # Prefer the materialized full checkpoint.  Falling back to the canonical
    # adapter makes this helper useful if materialization is deliberately delayed.
    materialized = root / "winner" / "best_model.pth"
    canonical_adapter = root / "stage2" / canonical_variant / f"seed_{canonical_seed}" / "best_adapter.pth"
    canonical_checkpoint = materialized if materialized.is_file() else canonical_adapter
    specs.append(
        HeldoutEvaluationSpec(
            name=f"canonical_{canonical_variant}_seed_{canonical_seed}_full",
            variant=canonical_variant,
            seed=canonical_seed,
            checkpoint=canonical_checkpoint,
            base_checkpoint=base_checkpoint,
            scene_manifest=scene_manifest,
            scene_manifest_sha256=scene_manifest_sha256,
            output_dir=root / "winner" / "evaluation_full",
            localization_samples=0,
            expected_retrieval_scenes=expected_run1_scenes,
            expected_localization_scenes=expected_run1_scenes,
            canonical_full=True,
        )
    )
    return specs


def evaluation_command(
    spec: HeldoutEvaluationSpec,
    *,
    heldout_root: Path,
    python_bin: str = sys.executable,
    device: str = "cuda:0",
    num_workers: int = 4,
    subset_seed: int = 42,
) -> list[str]:
    """Build the exact strict/resumable evaluation command for one spec."""
    return [
        python_bin,
        "-m",
        "src.evaluation.evaluation_pipeline",
        "--data-root",
        str(heldout_root.expanduser().resolve()),
        "--checkpoint",
        str(spec.checkpoint),
        "--base-checkpoint",
        str(spec.base_checkpoint),
        "--scene-manifest",
        str(spec.scene_manifest),
        "--manifest-split",
        "test",
        "--output-dir",
        str(spec.output_dir),
        "--device",
        device,
        "--num-workers",
        str(int(num_workers)),
        "--num-ugv-views",
        "8",
        "--seed",
        str(spec.seed),
        "--localization-samples",
        str(spec.localization_samples),
        "--subset-seed",
        str(int(subset_seed)),
        "--top-k",
        "5",
        "--grid-range-m",
        "5.0",
        "--grid-resolution-m",
        "0.25",
        "--yaw-search-angles-deg=-90,-45,0,45,90,135,180",
        "--localization-thresholds-m",
        "1",
        "2",
        "3",
        "--resume",
    ]


def _expected_evaluation_context(
    spec: HeldoutEvaluationSpec,
    *,
    heldout_root: Path,
    subset_seed: int,
) -> dict[str, Any]:
    checkpoint = spec.checkpoint.expanduser().resolve()
    base_checkpoint = spec.base_checkpoint.expanduser().resolve()
    adapter_checkpoint = checkpoint.name.endswith("adapter.pth")
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256_cached(checkpoint),
        "base_checkpoint": str(base_checkpoint) if adapter_checkpoint else None,
        "base_checkpoint_sha256": (
            _sha256_cached(base_checkpoint) if adapter_checkpoint else None
        ),
        "data_roots": [str(heldout_root.expanduser().resolve())],
        "scene_manifest": str(spec.scene_manifest.expanduser().resolve()),
        "scene_manifest_sha256": spec.scene_manifest_sha256,
        "manifest_split": "test",
        "num_ugv_views": 8,
        "seed": spec.seed,
        "subset_seed": int(subset_seed),
        "localization_samples": spec.localization_samples,
        "top_k": 5,
        "softmax_temp": 0.07,
        "grid_range_m": 5.0,
        "grid_resolution_m": 0.25,
        "yaw_search_angles_deg": "-90,-45,0,45,90,135,180",
        "localization_thresholds_m": [1.0, 2.0, 3.0],
    }


def _valid_evaluation_summary(
    spec: HeldoutEvaluationSpec,
    *,
    heldout_root: Path,
    subset_seed: int,
) -> tuple[bool, dict[str, Any]]:
    summary_path = spec.output_dir / "evaluation_summary.json"
    context_path = spec.output_dir / "evaluation_context.json"
    if not summary_path.is_file() or not context_path.is_file():
        return False, {}
    try:
        summary = _read_json(summary_path)
        context = _read_json(context_path)
        expected_context = _expected_evaluation_context(
            spec, heldout_root=heldout_root, subset_seed=subset_seed
        )
        if context != expected_context:
            return False, {}
        retrieval = summary.get("retrieval")
        localization = summary.get("localization")
        valid = (
            summary.get("num_retrieval_scenes") == spec.expected_retrieval_scenes
            and summary.get("num_localization_scenes") == spec.expected_localization_scenes
            and isinstance(retrieval, Mapping)
            and all(metric in retrieval for metric in ("recall@1", "recall@5", "mrr"))
            and isinstance(localization, Mapping)
        )
        return bool(valid), summary
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False, {}


def _archive_stale_evaluation_output(
    spec: HeldoutEvaluationSpec,
    *,
    heldout_root: Path,
    subset_seed: int,
) -> Path | None:
    """Quarantine an incompatible resume directory instead of deleting records."""
    output_dir = spec.output_dir
    if not output_dir.exists():
        return None
    context_path = output_dir / "evaluation_context.json"
    if context_path.is_file():
        try:
            if _read_json(context_path) == _expected_evaluation_context(
                spec, heldout_root=heldout_root, subset_seed=subset_seed
            ):
                return None
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass
    else:
        durable_outputs = (
            output_dir / "per_scene",
            output_dir / "evaluation_summary.json",
            output_dir / "localization_summary.json",
        )
        if not any(path.exists() for path in durable_outputs):
            return None

    archive_root = output_dir.parent / f"{output_dir.name}_stale"
    archive_root.mkdir(parents=True, exist_ok=True)
    token = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{time.time_ns()}"
    destination = archive_root / token
    os.replace(output_dir, destination)
    output_dir.mkdir(parents=True, exist_ok=True)
    return destination


def _default_command_runner(command: Sequence[str], cwd: Path, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{datetime.now(timezone.utc).isoformat()}] {' '.join(command)}\n")
        log.flush()
        completed = subprocess.run(
            list(command),
            cwd=cwd,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    return int(completed.returncode)


def run_heldout_evaluations(
    *,
    run_dir: Path,
    base_checkpoint: Path,
    heldout_root: Path = DEFAULT_HELDOUT_ROOT,
    python_bin: str = sys.executable,
    device: str = "cuda:0",
    num_workers: int = 4,
    seeds: Sequence[int] = DEFAULT_STAGE2_SEEDS,
    expected_run1_scenes: int = 553,
    localization_subset_size: int = 100,
    subset_seed: int = 42,
    max_attempts: int = 2,
    dry_run: bool = False,
    command_runner: Callable[[Sequence[str], Path, Path], int] | None = None,
    report_builder: Callable[[Path], Any] | None = None,
) -> dict[str, Any]:
    """Run all finalist confirmations sequentially, resuming and isolating failures.

    A nonzero evaluation does not abort the campaign.  It is retried once by
    default, recorded, and the next seed continues.  A valid summary is the
    completion marker, allowing recovery even if the parent process stopped just
    before it wrote ``status.json``.
    """
    root = run_dir.expanduser().resolve()
    specs = build_heldout_evaluation_specs(
        run_dir=root,
        base_checkpoint=base_checkpoint,
        heldout_root=heldout_root,
        seeds=seeds,
        expected_run1_scenes=expected_run1_scenes,
        localization_subset_size=localization_subset_size,
    )
    commands = [
        evaluation_command(
            spec,
            heldout_root=heldout_root,
            python_bin=python_bin,
            device=device,
            num_workers=num_workers,
            subset_seed=subset_seed,
        )
        for spec in specs
    ]
    if dry_run:
        plan = {
            "state": "planned",
            "evaluations": [
                {"spec": spec.as_json(), "command": command}
                for spec, command in zip(specs, commands)
            ],
        }
        _atomic_json(root / "heldout_plan.json", plan)
        return plan

    if max_attempts < 1:
        raise ValueError("max_attempts must be at least one")
    runner = command_runner or _default_command_runner
    if report_builder is None:
        from src.evaluation.fusion_ablation_report import build_report

        report_builder = build_report

    results: list[dict[str, Any]] = []
    campaign_started = time.time()

    def persist_progress(state: str) -> dict[str, Any]:
        failures = [result for result in results if result["state"] == "failed"]
        payload = {
            "state": state,
            "status": state,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "runtime_seconds": time.time() - campaign_started,
            "total": len(specs),
            "completed": sum(result["state"] == "complete" for result in results),
            "failed": len(failures),
            "pending": len(specs) - len(results),
            "evaluations": list(results),
        }
        _atomic_json(root / "heldout_status.json", payload)
        return payload

    def refresh_report(result: dict[str, Any]) -> None:
        persist_progress("running")
        try:
            report_builder(root)
        except Exception as error:
            result["report_error"] = f"{type(error).__name__}: {error}"
            persist_progress("running")

    persist_progress("running")
    for spec, command in zip(specs, commands):
        spec.output_dir.mkdir(parents=True, exist_ok=True)
        # Do not call this ``status.json``: the campaign report intentionally
        # discovers training experiments by that filename.  A distinct name
        # avoids double-counting each stage-2 run after evaluation.
        status_path = spec.output_dir / "evaluation_status.json"
        archived_output = None
        adapter_checkpoint = spec.checkpoint.name.endswith("adapter.pth")
        context_inputs_exist = spec.checkpoint.is_file() and (
            not adapter_checkpoint or spec.base_checkpoint.is_file()
        )
        if context_inputs_exist:
            archived = _archive_stale_evaluation_output(
                spec, heldout_root=heldout_root, subset_seed=subset_seed
            )
            archived_output = str(archived) if archived is not None else None
        valid, summary = _valid_evaluation_summary(
            spec, heldout_root=heldout_root, subset_seed=subset_seed
        )
        if valid:
            previous_status = (
                _read_json(status_path) if status_path.is_file() else {}
            )
            result = {
                "state": "complete",
                "status": "complete",
                "name": spec.name,
                "variant": spec.variant,
                "seed": spec.seed,
                "canonical_full": spec.canonical_full,
                "resumed": True,
                "attempts_this_run": 0,
                "runtime_seconds": previous_status.get("runtime_seconds"),
                "archived_stale_output": archived_output,
                "summary": summary,
            }
            _atomic_json(status_path, result)
            results.append(result)
            refresh_report(result)
            continue

        if not spec.checkpoint.is_file():
            result = {
                "state": "failed",
                "status": "failed",
                "name": spec.name,
                "variant": spec.variant,
                "seed": spec.seed,
                "canonical_full": spec.canonical_full,
                "attempts_this_run": 0,
                "archived_stale_output": archived_output,
                "error": f"Checkpoint not found: {spec.checkpoint}",
            }
            _atomic_json(status_path, result)
            results.append(result)
            refresh_report(result)
            continue

        evaluation_started = time.time()
        return_codes: list[int] = []
        runner_errors: list[str] = []
        for attempt in range(1, max_attempts + 1):
            running = {
                "state": "running",
                "status": "running",
                "name": spec.name,
                "variant": spec.variant,
                "seed": spec.seed,
                "canonical_full": spec.canonical_full,
                "attempt": attempt,
                "max_attempts": max_attempts,
                "command": command,
            }
            _atomic_json(status_path, running)
            try:
                return_code = int(runner(command, REPO_ROOT, spec.output_dir / "evaluation.log"))
            except Exception as error:
                return_code = -1
                runner_errors.append(f"attempt {attempt}: {type(error).__name__}: {error}")
            return_codes.append(return_code)
            valid, summary = _valid_evaluation_summary(
                spec, heldout_root=heldout_root, subset_seed=subset_seed
            )
            # The count-validated summary is the durable completion marker.  It
            # may have been atomically written immediately before a wrapper or
            # logging failure changed the subprocess return code.
            if valid:
                break

        if valid:
            result = {
                "state": "complete",
                "status": "complete",
                "name": spec.name,
                "variant": spec.variant,
                "seed": spec.seed,
                "canonical_full": spec.canonical_full,
                "attempts_this_run": len(return_codes),
                "return_codes": return_codes,
                "runner_errors": runner_errors,
                "runtime_seconds": time.time() - evaluation_started,
                "archived_stale_output": archived_output,
                "summary": summary,
            }
        else:
            result = {
                "state": "failed",
                "status": "failed",
                "name": spec.name,
                "variant": spec.variant,
                "seed": spec.seed,
                "canonical_full": spec.canonical_full,
                "attempts_this_run": len(return_codes),
                "return_codes": return_codes,
                "runner_errors": runner_errors,
                "runtime_seconds": time.time() - evaluation_started,
                "archived_stale_output": archived_output,
                "error": "Evaluation did not produce a complete, count-validated summary",
            }
        _atomic_json(status_path, result)
        results.append(result)
        refresh_report(result)

    failures = [result for result in results if result["state"] != "complete"]
    final_state = "complete" if not failures else "complete_with_failures"
    campaign_status = persist_progress(final_state)
    try:
        report_builder(root)
    except Exception as error:
        campaign_status["report_error"] = f"{type(error).__name__}: {error}"
        _atomic_json(root / "heldout_status.json", campaign_status)
    return campaign_status


__all__ = [
    "DEFAULT_FOCUSED_TESTS",
    "DEFAULT_HELDOUT_ROOT",
    "DEFAULT_STAGE2_SEEDS",
    "HeldoutEvaluationSpec",
    "PreflightError",
    "build_heldout_evaluation_specs",
    "evaluation_command",
    "load_or_create_heldout_manifest",
    "resolve_metadata_reference",
    "run_focused_unit_tests",
    "run_heldout_evaluations",
    "run_preflight",
    "validate_base_checkpoint",
    "validate_dataset_root",
    "validate_device",
    "validate_disk_space",
    "validate_feature_cache",
    "validate_imports",
]
