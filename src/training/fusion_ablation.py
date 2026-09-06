"""Resumable unattended fusion-ablation campaign for SnapViT.

The module deliberately keeps orchestration, checkpointing and ranking independent
from the generic training loop.  The public entry point is::

    python -m src.training.fusion_ablation campaign --run-dir ablation_results/fusion

Each experiment is an isolated directory and can also be executed with ``run-one``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.data.dataset import VineyardDataset
from src.evaluation.common import compute_retrieval_metrics, pool_bev_embeddings
from src.models.snapvit import SnapViT
from src.training.fusion_cache import (
    FP32FeatureCache,
    cache_fingerprint,
    scene_source_records,
    selected_scene_sources,
)
from src.training.losses import masked_info_nce_loss, symmetric_info_nce_loss_masked
from src.training.train_loop import _select_loss, normalize_config


# Required by deterministic CUDA GEMM kernels when deterministic algorithms are
# enabled. Set it before the campaign creates a CUDA context.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE_CHECKPOINT = REPO_ROOT / "models/provvisiorio/t1/best_model.pth"
DEFAULT_DATA_ROOTS = (
    REPO_ROOT / "datasets/tempovine/dataset_tempovine_mar_run2",
    REPO_ROOT / "datasets/tempovine/dataset_tempovine_mar_run3",
)
VARIANTS = tuple(f"N{i}" for i in range(10))
STAGE1_SEED = 42
STAGE2_SEEDS = (42, 7, 123)
TRANSFORM_DESCRIPTION = (
    "torchvision.Resize(train_img_size,antialias=True);"
    "ConvertImageDtype(float32);ImageNetNormalize-v1"
)
FUSION_PREFIXES = (
    "ground_encoder.feature_norm.",
    "ground_encoder.token_mlp.",
    "ground_encoder.column_mlp.",
    "ground_encoder.token_attention.",
    "ground_encoder.view_attention.",
)
FUSION_EXACT_NAMES = {"ground_encoder.column_residual_scale"}


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    return value


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(_jsonable(dict(payload)), stream, indent=2, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_torch_save(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def load_base_config(checkpoint: Path) -> dict[str, Any]:
    config_path = checkpoint.parent / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Base checkpoint config not found: {config_path}")
    with config_path.open(encoding="utf-8") as stream:
        config = normalize_config(json.load(stream))
    config["pretrained_backbones"] = False
    config["ground_fusion_mode"] = "mlp"
    config["num_ugv_views"] = 8
    return config


def resolved_campaign_config(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = Path(args.base_checkpoint).resolve()
    config = load_base_config(checkpoint)
    config.update(
        data_root=[str(Path(root).resolve()) for root in args.data_root],
        device=args.device,
        num_workers=args.num_workers,
        ground_fusion_mode="mlp",
        pretrained_backbones=False,
        batch_size=1 if args.smoke_test else int(config.get("batch_size", 2)),
    )
    if args.smoke_test:
        config["mixed_loss_delay"] = 0
        config["fusion_finetune_start_epoch"] = 1
        config["fusion_smoke_test"] = True
        config["fusion_smoke_train_scenes"] = 8
        config["fusion_smoke_validation_scenes"] = 4
    else:
        config["fusion_finetune_start_epoch"] = 20
    config["deterministic_algorithms"] = True
    return config


def _scene_paths(root: Path) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {root}")
    return sorted(
        path.resolve()
        for path in root.iterdir()
        if path.is_dir() and (path / "metadata.json").is_file()
    )


def create_split_manifest(
    roots: Sequence[Path],
    output_path: Path,
    split_seed: int = 42,
    val_ratio: float = 0.2,
    eligible_scenes: Iterable[str | Path] | None = None,
) -> dict[str, Any]:
    """Create a deterministic split independently inside every dataset root.

    ``VineyardDataset`` may reject scenes whose views all violate the configured
    edge margin.  Passing its eligible scene list keeps those unusable scenes out
    of the persisted split instead of failing later during cache preparation.
    """
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between zero and one")
    eligible = (
        {str(Path(scene).resolve()) for scene in eligible_scenes}
        if eligible_scenes is not None
        else None
    )
    records: list[dict[str, str]] = []
    all_scene_count = 0
    for root_index, root in enumerate(map(Path, roots)):
        all_scenes = _scene_paths(root)
        all_scene_count += len(all_scenes)
        scenes = [
            scene
            for scene in all_scenes
            if eligible is None or str(scene.resolve()) in eligible
        ]
        if not scenes:
            raise RuntimeError(f"No eligible training scenes remain in {root.resolve()}")
        order = list(range(len(scenes)))
        random.Random(split_seed + root_index).shuffle(order)
        val_count = max(1, int(len(scenes) * val_ratio)) if len(scenes) > 1 else len(scenes)
        validation = set(order[:val_count])
        for index, scene in enumerate(scenes):
            records.append(
                {
                    "root": str(root.resolve()),
                    "scene": str(scene),
                    "scene_id": f"{root.name}/{scene.name}",
                    "split": "validation" if index in validation else "train",
                }
            )
    if eligible is not None:
        recorded = {row["scene"] for row in records}
        unowned = eligible - recorded
        if unowned:
            raise RuntimeError(
                f"{len(unowned)} eligible scene(s) are outside the configured roots"
            )
    manifest = {
        "version": 1,
        "split_seed": split_seed,
        "val_ratio": val_ratio,
        "roots": [str(Path(root).resolve()) for root in roots],
        "eligible_scene_count": len(records),
        "excluded_scene_count": all_scene_count - len(records),
        "records": records,
    }
    atomic_json(output_path, manifest)
    return manifest


def load_or_create_manifest(
    roots: Sequence[Path],
    output_path: Path,
    split_seed: int = 42,
    val_ratio: float = 0.2,
    eligible_scenes: Iterable[str | Path] | None = None,
) -> dict[str, Any]:
    eligible = (
        {str(Path(scene).resolve()) for scene in eligible_scenes}
        if eligible_scenes is not None
        else None
    )
    if output_path.is_file():
        with output_path.open(encoding="utf-8") as stream:
            manifest = json.load(stream)
        expected = [str(Path(root).resolve()) for root in roots]
        recorded = {
            str(Path(row["scene"]).resolve())
            for row in manifest.get("records", [])
            if isinstance(row, Mapping) and "scene" in row
        }
        if (
            manifest.get("roots") != expected
            or manifest.get("split_seed") != split_seed
            or float(manifest.get("val_ratio", -1.0)) != float(val_ratio)
            or (eligible is not None and recorded != eligible)
        ):
            raise RuntimeError(
                "Existing split manifest does not match the current roots, seed, "
                "validation ratio, or eligible scene set; choose a new RUN_DIR or "
                "remove the manifest intentionally."
            )
        return manifest
    return create_split_manifest(
        roots,
        output_path,
        split_seed,
        val_ratio,
        eligible_scenes=eligible,
    )


def eligible_scene_paths(config: Mapping[str, Any]) -> list[str]:
    """Return exactly the scenes accepted by the configured training dataset."""
    image_transform, depth_transform = build_transforms(config)
    dataset = VineyardDataset(
        root_dir=config["data_root"],
        config=dict(config),
        transforms=image_transform,
        depth_transforms=depth_transform,
        consecutive_frames=bool(config.get("consecutive_frames", True)),
    )
    return [str(Path(scene).resolve()) for scene in dataset.scene_folders]


class DeterministicSceneSubset(Dataset):
    """Select manifest scenes and make per-scene view sampling repeatable."""

    def __init__(self, dataset: VineyardDataset, scene_paths: Iterable[str], seed: int):
        self.dataset = dataset
        lookup = {str(Path(path).resolve()): i for i, path in enumerate(dataset.scene_folders)}
        missing = [path for path in scene_paths if str(Path(path).resolve()) not in lookup]
        if missing:
            raise RuntimeError(f"{len(missing)} manifest scenes were filtered from the dataset")
        self.indices = [lookup[str(Path(path).resolve())] for path in scene_paths]
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        # VineyardDataset uses Python's module-level RNG to pick consecutive views.
        state = random.getstate()
        random.seed(self.seed * 1_000_003 + self.indices[index])
        try:
            item = self.dataset[self.indices[index]]
        finally:
            random.setstate(state)
        item["scene_id"] = str(Path(self.dataset.scene_folders[self.indices[index]]).resolve())
        sources = selected_scene_sources(
            self.dataset, self.indices[index], self.seed
        )
        item["cache_ground_paths"] = sources["ground_paths"]
        item["cache_overhead_path"] = sources["overhead_path"]
        return item


def build_transforms(config: Mapping[str, Any]):
    image_transform = transforms.Compose(
        [
            transforms.Resize(config["train_img_size"], antialias=True),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    depth_transform = transforms.Compose(
        [
            transforms.Resize(config["train_img_size"], antialias=True),
            transforms.ConvertImageDtype(torch.float),
        ]
    )
    return image_transform, depth_transform


def build_datasets(config: Mapping[str, Any], manifest: Mapping[str, Any], seed: int):
    image_transform, depth_transform = build_transforms(config)
    base = VineyardDataset(
        root_dir=config["data_root"],
        config=dict(config),
        transforms=image_transform,
        depth_transforms=depth_transform,
        consecutive_frames=bool(config.get("consecutive_frames", True)),
    )
    train_scenes = [row["scene"] for row in manifest["records"] if row["split"] == "train"]
    val_scenes = [row["scene"] for row in manifest["records"] if row["split"] == "validation"]
    if config.get("fusion_smoke_test"):
        train_scenes = train_scenes[: int(config["fusion_smoke_train_scenes"])]
        val_scenes = val_scenes[: int(config["fusion_smoke_validation_scenes"])]
    return (
        DeterministicSceneSubset(base, train_scenes, seed),
        DeterministicSceneSubset(base, val_scenes, seed),
    )


def build_loaders(config: Mapping[str, Any], manifest: Mapping[str, Any], seed: int):
    train, validation = build_datasets(config, manifest, seed)
    generator = torch.Generator().manual_seed(seed)
    kwargs = dict(
        batch_size=int(config["batch_size"]),
        num_workers=int(config["num_workers"]),
        pin_memory=bool(config.get("pin_memory", True)),
        worker_init_fn=seed_worker,
    )
    if kwargs["num_workers"]:
        kwargs["prefetch_factor"] = 2
    return (
        DataLoader(train, shuffle=True, generator=generator, **kwargs),
        DataLoader(validation, shuffle=False, **kwargs),
        generator,
    )


def feature_cache_source_paths(
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    seeds: Iterable[int] = (STAGE1_SEED, *STAGE2_SEEDS),
) -> tuple[list[str], list[str], int, list[int]]:
    """Resolve every unique-image cache input for the requested view seeds."""
    seed_list = sorted({int(seed) for seed in seeds})
    train_subset, validation_subset = build_datasets(
        config, manifest, STAGE1_SEED
    )
    base_dataset = train_subset.dataset
    indices = sorted(set(train_subset.indices + validation_subset.indices))
    source_records = []
    for view_seed in seed_list:
        source_records.extend(
            scene_source_records(base_dataset, indices, view_seed)
        )
    ground_paths = [
        path for record in source_records for path in record["ground_paths"]
    ]
    overhead_paths = [record["overhead_path"] for record in source_records]
    return ground_paths, overhead_paths, len(indices), seed_list


def prepare_feature_cache(
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    run_dir: Path,
    checkpoint: Path,
    *,
    seeds: Iterable[int] = (STAGE1_SEED, *STAGE2_SEEDS),
) -> FP32FeatureCache:
    """Build or validate the campaign's unique-image FP32 feature cache."""
    ground_paths, overhead_paths, scene_count, seed_list = feature_cache_source_paths(
        config, manifest, seeds=seeds
    )
    base_hash = sha256_file(checkpoint)
    fingerprint = cache_fingerprint(
        base_checkpoint_sha256=base_hash,
        config=config,
        transform_description=TRANSFORM_DESCRIPTION,
        ground_paths=ground_paths,
        overhead_paths=overhead_paths,
    )
    cache_dir = run_dir / "feature_cache"
    context_path = run_dir / "feature_cache_context.json"

    try:
        cache = FP32FeatureCache(
            cache_dir, expected_fingerprint=fingerprint
        )
    except (FileNotFoundError, RuntimeError, OSError, ValueError):
        seed_everything(STAGE1_SEED)
        cache_config = dict(config)
        cache_config.pop("ground_fusion_variant", None)
        cache_config["ground_fusion_mode"] = "mlp"
        cache_config["use_height_positional_encoding"] = False
        model = SnapViT(cache_config).to(config["device"])
        load_base_whitelist(model, checkpoint)
        model.eval()
        image_transform, _ = build_transforms(config)
        cache_batch_size = int(
            config.get(
                "fusion_cache_batch_size",
                max(1, int(config.get("batch_size", 2)) * int(config["num_ugv_views"])),
            )
        )
        cache = FP32FeatureCache.build(
            cache_dir,
            fingerprint=fingerprint,
            ground_paths=ground_paths,
            overhead_paths=overhead_paths,
            ground_encoder=model.ground_encoder.feature_extractor,
            overhead_encoder=model.overhead_encoder,
            image_transform=image_transform,
            device=config["device"],
            batch_size=cache_batch_size,
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    atomic_json(
        context_path,
        {
            "fingerprint": fingerprint,
            "base_checkpoint_sha256": base_hash,
            "seeds": seed_list,
            "scene_count": scene_count,
            "unique_ground_images": len(cache.ground_index),
            "unique_overhead_images": len(cache.overhead_index),
            "ground_feature_shape": list(cache.ground.shape),
            "overhead_feature_shape": list(cache.overhead.shape),
            "dtype": "float32",
            "read_only": True,
        },
    )
    return cache


def open_prepared_feature_cache(run_dir: Path) -> FP32FeatureCache:
    context_path = run_dir / "feature_cache_context.json"
    if not context_path.is_file():
        raise FileNotFoundError(
            f"Prepared feature-cache context is missing: {context_path}"
        )
    context = json.loads(context_path.read_text(encoding="utf-8"))
    return FP32FeatureCache(
        run_dir / "feature_cache",
        expected_fingerprint=context["fingerprint"],
    )


def resume_config_signature(config: Mapping[str, Any]) -> dict[str, Any]:
    """Fields that must remain identical for an exact epoch-boundary resume."""
    keys = (
        "vit_model",
        "model_name",
        "feature_dim",
        "ground_fusion_mode",
        "ground_fusion_variant",
        "use_height_positional_encoding",
        "train_img_size",
        "grid_size",
        "grid_resolution",
        "num_ugv_views",
        "use_depth",
        "depth_range",
        "ground_tile_size",
        "consecutive_frames",
        "edge_margin_m",
        "data_root",
        "batch_size",
        "num_workers",
        "pin_memory",
        "device",
        "pixel_loss_weight",
        "mixed_loss_delay",
        "use_pixel_loss",
        "use_global_loss",
        "fusion_finetune_start_epoch",
        "fusion_ablation_stage",
        "fusion_ablation_seed",
        "fusion_smoke_test",
        "deterministic_algorithms",
    )
    return {key: _jsonable(config.get(key)) for key in keys}


def is_fusion_name(name: str) -> bool:
    return name in FUSION_EXACT_NAMES or name.startswith(FUSION_PREFIXES)


def fusion_parameter_names(model: torch.nn.Module) -> list[str]:
    helper = getattr(model.ground_encoder, "fusion_named_parameters", None)
    if callable(helper):
        raw = list(helper())
        names = [
            name if name.startswith("ground_encoder.") else f"ground_encoder.{name}"
            for name, _ in raw
        ]
    else:
        names = [name for name, _ in model.named_parameters() if is_fusion_name(name)]
    existing = dict(model.named_parameters())
    names = [name for name in names if name in existing]
    if not names:
        raise RuntimeError("No fusion parameters were discovered for this variant")
    return sorted(set(names))


def checkpoint_state_dict(path: Path) -> Mapping[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, Mapping) and isinstance(payload.get("state_dict"), Mapping):
        payload = payload["state_dict"]
    if not isinstance(payload, Mapping):
        raise TypeError(f"Unsupported checkpoint format: {path}")
    return payload


def load_base_whitelist(model: torch.nn.Module, checkpoint: Path) -> dict[str, list[str]]:
    """Load all compatible non-fusion tensors; all fusion tensors remain fresh."""
    source = checkpoint_state_dict(checkpoint)
    target = model.state_dict()
    accepted: dict[str, torch.Tensor] = {}
    mismatched: list[str] = []
    for name, tensor in source.items():
        if is_fusion_name(name):
            continue
        if name in target and target[name].shape == tensor.shape:
            accepted[name] = tensor
        elif name in target:
            mismatched.append(name)
    required = [name for name in target if not is_fusion_name(name)]
    missing = sorted(set(required) - set(accepted))
    if missing or mismatched:
        raise RuntimeError(
            f"Base checkpoint is incompatible: missing={missing[:8]}, mismatched={mismatched[:8]}"
        )
    result = model.load_state_dict(accepted, strict=False)
    unexpected = list(result.unexpected_keys)
    bad_missing = [name for name in result.missing_keys if not is_fusion_name(name)]
    if unexpected or bad_missing:
        raise RuntimeError(f"Unsafe base load: missing={bad_missing}, unexpected={unexpected}")
    return {"loaded": sorted(accepted), "fresh": sorted(result.missing_keys)}


def configure_trainable_parameters(
    model: torch.nn.Module, stage: int, epoch: int, fine_tune_start: int = 20
):
    fusion_names = set(fusion_parameter_names(model))
    unfreeze_ground_stage = stage == 2 and epoch >= fine_tune_start
    groups: dict[str, list[torch.nn.Parameter]] = {"fusion": [], "ground_stage": []}
    for name, parameter in model.named_parameters():
        if name in fusion_names:
            parameter.requires_grad_(True)
            groups["fusion"].append(parameter)
        elif unfreeze_ground_stage and name.startswith(
            "ground_encoder.feature_extractor.convnext.stages_2."
        ):
            parameter.requires_grad_(True)
            groups["ground_stage"].append(parameter)
        else:
            parameter.requires_grad_(False)
    return groups


def keep_frozen_modules_in_eval(
    model: torch.nn.Module, stage: int, epoch: int, fine_tune_start: int = 20
) -> None:
    model.train()
    model.overhead_encoder.eval()
    model.ground_encoder.feature_extractor.eval()
    if stage == 2 and epoch >= fine_tune_start:
        model.ground_encoder.feature_extractor.convnext.stages_2.train()


def make_optimizer(groups: Mapping[str, Sequence[torch.nn.Parameter]]) -> torch.optim.Optimizer:
    parameter_groups = [{"params": groups["fusion"], "lr": 1e-4, "name": "fusion"}]
    if groups["ground_stage"]:
        parameter_groups.append(
            {"params": groups["ground_stage"], "lr": 1e-5, "name": "ground_stage"}
        )
    return torch.optim.Adam(parameter_groups)


def _device_dict(values: Mapping[str, Any], device: str, include_images: bool = True):
    result = {}
    for key, value in values.items():
        if not include_images and key in {"ugv_images", "uav_image"}:
            continue
        result[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return result


def _uncollate_ground_paths(collated: Sequence[Sequence[str]]) -> list[list[str]]:
    """Convert default-collated per-view strings back to batch-major paths."""
    if not collated:
        return []
    return [list(paths) for paths in zip(*collated)]


def forward_loss(
    model: SnapViT,
    batch: Mapping[str, Any],
    config: Mapping[str, Any],
    epoch: int,
    *,
    feature_cache: FP32FeatureCache | None = None,
    cache_ground: bool = False,
):
    use_cache = feature_cache is not None
    ugv = _device_dict(
        batch["ugv_data"], config["device"], include_images=not cache_ground
    )
    if cache_ground:
        batch_paths = _uncollate_ground_paths(batch["cache_ground_paths"])
        encoded_ground = feature_cache.encoded_ground(
            batch_paths,
            image_size=tuple(config["train_img_size"]),
            device=config["device"],
        )
        ground, validity = model.ground_encoder.project(
            encoded_ground=encoded_ground, ugv_data=ugv
        )
    else:
        ground, validity = model.ground_encoder(**ugv)

    if use_cache:
        overhead_paths = list(batch["cache_overhead_path"])
        overhead = feature_cache.overhead_tensor(
            overhead_paths, device=config["device"]
        )
    else:
        uav = _device_dict(batch["uav_data"], config["device"])
        overhead = model.overhead_encoder(**uav)
    overhead = F.interpolate(overhead, ground.shape[-2:], mode="bilinear", align_corners=False)
    pixel = masked_info_nce_loss(ground, overhead, validity, model.temperature)
    global_loss = symmetric_info_nce_loss_masked(ground, overhead, validity, model.temperature)
    return _select_loss(config, pixel, global_loss, epoch), pixel, global_loss, ground, overhead, validity


def require_finite_metrics(metrics: Mapping[str, Any], context: str) -> None:
    invalid = {}
    for name, value in metrics.items():
        if isinstance(value, (int, float, np.number)) and not np.isfinite(value):
            invalid[name] = value
    if invalid:
        raise FloatingPointError(f"Non-finite metrics during {context}: {invalid}")


def validate(
    model: SnapViT,
    loader: DataLoader,
    config: Mapping[str, Any],
    epoch: int,
    *,
    feature_cache: FP32FeatureCache | None = None,
    cache_ground: bool = False,
):
    model.eval()
    totals = np.zeros(3, dtype=np.float64)
    ground_embeddings, overhead_embeddings = [], []
    with torch.no_grad():
        for batch in loader:
            loss, pixel, global_loss, ground, overhead, validity = forward_loss(
                model,
                batch,
                config,
                epoch,
                feature_cache=feature_cache,
                cache_ground=cache_ground,
            )
            totals += [loss.item(), pixel.item(), global_loss.item()]
            ground_embeddings.append(pool_bev_embeddings(ground, validity).cpu())
            overhead_embeddings.append(pool_bev_embeddings(overhead).cpu())
    if not ground_embeddings:
        raise RuntimeError("Validation loader is empty")
    metrics = compute_retrieval_metrics(
        torch.cat(ground_embeddings), torch.cat(overhead_embeddings), ks=(1, 5)
    )
    totals /= len(loader)
    metrics.update(val_loss=float(totals[0]), val_pixel_loss=float(totals[1]), val_global_loss=float(totals[2]))
    require_finite_metrics(metrics, "validation")
    return metrics


def _write_history(path: Path, history: Sequence[Mapping[str, Any]]) -> None:
    if not history:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(history[0])
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(history)
    os.replace(temporary, path)


def _adapter_payload(
    model: SnapViT,
    optimizer: torch.optim.Optimizer,
    config: Mapping[str, Any],
    checkpoint: Path,
    epoch: int,
    history: Sequence[Mapping[str, Any]],
    best: Mapping[str, Any],
    no_improvement: int,
    base_checkpoint_sha256: str,
    dataloader_generator: torch.Generator | None = None,
) -> dict[str, Any]:
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    adapter = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items() if name in trainable}
    return {
        "format_version": 1,
        "adapter_state_dict": adapter,
        "trainable_state_dict": adapter,
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "history": list(history),
        "best_metrics": dict(best),
        "early_stopping": {"epochs_without_improvement": int(no_improvement)},
        "rng_state": capture_rng_state(),
        "dataloader_generator_state": (
            dataloader_generator.get_state()
            if dataloader_generator is not None
            else None
        ),
        "base_checkpoint_path": str(checkpoint.resolve()),
        "base_checkpoint_sha256": base_checkpoint_sha256,
        "resolved_config": _jsonable(dict(config)),
    }


def _load_adapter(
    model: SnapViT,
    optimizer: torch.optim.Optimizer | None,
    path: Path,
    payload: Mapping[str, Any] | None = None,
):
    if payload is None:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    result = model.load_state_dict(payload["adapter_state_dict"], strict=False)
    unexpected = result.unexpected_keys
    if unexpected:
        raise RuntimeError(f"Unexpected adapter keys: {unexpected}")
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    restore_rng_state(payload["rng_state"])
    return payload


def run_directory(run_dir: Path, stage: int, variant: str, seed: int) -> Path:
    return run_dir / f"stage{stage}" / variant / f"seed_{seed}"


def run_one(args: argparse.Namespace) -> dict[str, Any]:
    if args.variant not in VARIANTS:
        raise ValueError(f"Unknown variant {args.variant}; expected one of {VARIANTS}")
    stage, seed = int(args.stage), int(args.seed)
    root = Path(args.run_dir).resolve()
    destination = run_directory(root, stage, args.variant, seed)
    destination.mkdir(parents=True, exist_ok=True)
    status_path = destination / "status.json"
    existing_status = None
    if status_path.is_file():
        existing_status = json.loads(status_path.read_text())

    config = resolved_campaign_config(args)
    config["ground_fusion_variant"] = args.variant
    config["use_height_positional_encoding"] = args.variant == "N5"
    config["fusion_ablation_stage"] = stage
    config["fusion_ablation_seed"] = seed
    config["output_model_path"] = str(destination)
    checkpoint = Path(args.base_checkpoint).resolve()
    base_checkpoint_sha256 = sha256_file(checkpoint)
    if existing_status and (
        existing_status.get("state") == "complete"
        or existing_status.get("status") == "complete"
    ):
        completed_adapter = destination / "best_adapter.pth"
        if not completed_adapter.is_file():
            raise RuntimeError(
                f"Completed run is missing its best adapter: {completed_adapter}"
            )
        completed_payload = torch.load(
            completed_adapter, map_location="cpu", weights_only=False
        )
        if completed_payload.get("base_checkpoint_sha256") != base_checkpoint_sha256:
            raise RuntimeError(
                "Completed run belongs to a different base checkpoint; use a new RUN_DIR."
            )
        if resume_config_signature(
            completed_payload.get("resolved_config", {})
        ) != resume_config_signature(config):
            raise RuntimeError(
                "Completed run belongs to a different resolved configuration; "
                "use a new RUN_DIR."
            )
        return existing_status
    atomic_json(destination / "config.json", config)
    if args.dry_run:
        status = {
            "state": "planned",
            "status": "planned",
            "stage": f"stage{stage}",
            "variant": args.variant,
            "seed": seed,
        }
        atomic_json(status_path, status)
        return status

    seed_everything(seed)
    manifest = load_or_create_manifest(
        [Path(path) for path in config["data_root"]],
        root / "split_manifest.json",
        eligible_scenes=eligible_scene_paths(config),
    )
    try:
        if not getattr(args, "cache_prepared", False):
            raise FileNotFoundError("Direct run-one validates the cache fingerprint")
        feature_cache = open_prepared_feature_cache(root)
    except (FileNotFoundError, RuntimeError, OSError, ValueError):
        feature_cache = prepare_feature_cache(
            config,
            manifest,
            root,
            checkpoint,
            seeds=(STAGE1_SEED, *STAGE2_SEEDS, seed),
        )
    # Cache construction uses its own fixed seed and may consume RNG state.
    # Restore the experiment seed immediately before fresh fusion initialization.
    seed_everything(seed)
    model = SnapViT(config)
    load_base_whitelist(model, checkpoint)
    model.to(config["device"])
    train_loader, val_loader, dataloader_generator = build_loaders(
        config, manifest, seed
    )
    max_epochs = (2 if args.smoke_test else 40) if stage == 1 else (3 if args.smoke_test else 100)
    patience = 2 if args.smoke_test else (8 if stage == 1 else 12)
    fine_tune_start = int(config["fusion_finetune_start_epoch"])
    start_epoch, history, best = 0, [], {"val_loss": float("inf")}
    no_improvement = 0
    latest_path = destination / "latest_adapter.pth"
    if latest_path.is_file():
        # Inspect the saved epoch before constructing the optimizer, because a
        # stage-2 checkpoint after epoch 20 has a second parameter group.
        header = torch.load(latest_path, map_location="cpu", weights_only=False)
        if header.get("base_checkpoint_sha256") != base_checkpoint_sha256:
            raise RuntimeError(
                "Cannot resume: adapter base-checkpoint SHA-256 does not match the current base."
            )
        if resume_config_signature(header.get("resolved_config", {})) != resume_config_signature(config):
            raise RuntimeError(
                "Cannot resume: the resolved experiment configuration changed. "
                "Use a new RUN_DIR for a different experiment."
        )
        resume_epoch = int(header["epoch"]) + 1
        optimizer_epoch = (
            resume_epoch - 1
            if stage == 2 and resume_epoch == fine_tune_start
            else resume_epoch
        )
        groups = configure_trainable_parameters(
            model, stage, optimizer_epoch, fine_tune_start=fine_tune_start
        )
        optimizer = make_optimizer(groups)
        payload = _load_adapter(
            model,
            optimizer,
            latest_path,
            payload=header,
        )
        start_epoch = resume_epoch
        history = list(payload.get("history", []))
        best = dict(payload.get("best_metrics", best))
        no_improvement = int(
            payload.get("early_stopping", {}).get(
                "epochs_without_improvement", 0
            )
        )
        if payload.get("dataloader_generator_state") is not None:
            dataloader_generator.set_state(payload["dataloader_generator_state"])
    else:
        groups = configure_trainable_parameters(
            model, stage, start_epoch, fine_tune_start=fine_tune_start
        )
        optimizer = make_optimizer(groups)

    atomic_json(status_path, {"state": "running", "status": "running", "stage": f"stage{stage}", "variant": args.variant, "seed": seed, "start_epoch": start_epoch})
    started = time.time()
    elapsed_offset = (
        float(history[-1].get("elapsed_seconds", 0.0)) if history else 0.0
    )
    for epoch in range(start_epoch, max_epochs):
        # Add the live ground stage without discarding the fusion head's Adam
        # moments accumulated during the cached phase.
        if stage == 2 and epoch == fine_tune_start:
            groups = configure_trainable_parameters(
                model, stage, epoch, fine_tune_start=fine_tune_start
            )
            if not any(
                group.get("name") == "ground_stage"
                for group in optimizer.param_groups
            ):
                optimizer.add_param_group(
                    {
                        "params": groups["ground_stage"],
                        "lr": 1e-5,
                        "name": "ground_stage",
                    }
                )
            no_improvement = 0
            best = {"val_loss": float("inf")}
        else:
            configure_trainable_parameters(
                model, stage, epoch, fine_tune_start=fine_tune_start
            )
        keep_frozen_modules_in_eval(
            model, stage, epoch, fine_tune_start=fine_tune_start
        )
        cache_ground = stage == 1 or epoch < fine_tune_start
        total = np.zeros(3, dtype=np.float64)
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss, pixel, global_loss, *_ = forward_loss(
                model,
                batch,
                config,
                epoch,
                feature_cache=feature_cache,
                cache_ground=cache_ground,
            )
            if not all(
                torch.isfinite(value).item()
                for value in (loss, pixel, global_loss)
            ):
                raise FloatingPointError(
                    f"Non-finite training loss at epoch {epoch + 1}"
                )
            loss.backward()
            optimizer.step()
            total += [loss.item(), pixel.item(), global_loss.item()]
        total /= len(train_loader)
        require_finite_metrics(
            {
                "train_loss": total[0],
                "train_pixel_loss": total[1],
                "train_global_loss": total[2],
            },
            "training",
        )
        validation = validate(
            model,
            val_loader,
            config,
            epoch,
            feature_cache=feature_cache,
            cache_ground=cache_ground,
        )
        row = {
            "epoch": epoch + 1,
            "train_loss": float(total[0]),
            "train_pixel_loss": float(total[1]),
            "train_global_loss": float(total[2]),
            **validation,
            "elapsed_seconds": elapsed_offset + time.time() - started,
        }
        history.append(row)
        monitor_start = int(config.get("mixed_loss_delay", 0)) if stage == 1 else 0
        eligible = epoch >= monitor_start
        improved = eligible and validation["val_loss"] < best.get("val_loss", float("inf"))
        if improved:
            best = {**validation, "epoch": epoch + 1}
            no_improvement = 0
        elif eligible:
            no_improvement += 1
        payload = _adapter_payload(
            model,
            optimizer,
            config,
            checkpoint,
            epoch,
            history,
            best,
            no_improvement,
            base_checkpoint_sha256,
            dataloader_generator,
        )
        if improved:
            atomic_torch_save(destination / "best_adapter.pth", payload)
        _write_history(destination / "history.csv", history)
        # latest_adapter is the epoch commit marker: all other epoch artifacts
        # are durable before it is atomically replaced.
        atomic_torch_save(latest_path, payload)
        atomic_json(status_path, {"state": "running", "status": "running", "stage": f"stage{stage}", "variant": args.variant, "seed": seed, "epoch": epoch + 1, "best_metrics": best, "runtime_seconds": elapsed_offset + time.time() - started})
        patience_active = epoch >= monitor_start if stage == 1 else epoch >= fine_tune_start
        if patience_active and no_improvement >= patience:
            break

    result = {
        "state": "complete",
        "status": "complete",
        "stage": f"stage{stage}",
        "variant": args.variant,
        "seed": seed,
        "best_val_loss": best["val_loss"],
        "recall@1": best.get("recall@1", 0.0),
        "recall@5": best.get("recall@5", 0.0),
        "mrr": best.get("mrr", 0.0),
        "best_epoch": best.get("epoch"),
        "epochs_completed": len(history),
        "runtime_seconds": elapsed_offset + time.time() - started,
        "run_dir": str(destination),
        "best_metrics": {
            "val_loss": best["val_loss"],
            "recall@1": best.get("recall@1", 0.0),
            "recall@5": best.get("recall@5", 0.0),
            "mrr": best.get("mrr", 0.0),
        },
        "trainable_parameters": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
    }
    atomic_json(destination / "result.json", result)
    atomic_json(status_path, result)
    return result


def rank_results(results: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Rank complete runs by loss, Recall@1 and MRR, with loss as tie-breaker."""
    required_metrics = ("best_val_loss", "recall@1", "mrr")
    complete = [
        dict(row)
        for row in results
        if row.get("state") == "complete"
        and all(
            isinstance(row.get(metric), (int, float, np.number))
            and np.isfinite(row[metric])
            for metric in required_metrics
        )
    ]
    if not complete:
        return []
    criteria = (("best_val_loss", False), ("recall@1", True), ("mrr", True))
    rank_sums = [0] * len(complete)
    for field, descending in criteria:
        order = sorted(range(len(complete)), key=lambda i: complete[i][field], reverse=descending)
        previous_value = None
        tied_rank = 1
        for position, index in enumerate(order, start=1):
            value = complete[index][field]
            if previous_value is None or value != previous_value:
                tied_rank = position
                previous_value = value
            rank_sums[index] += tied_rank
    for row, score in zip(complete, rank_sums):
        row["rank_sum"] = score
    return sorted(complete, key=lambda row: (row["rank_sum"], row["best_val_loss"]))


def select_stage(run_dir: Path, stage: int) -> dict[str, Any]:
    paths = sorted((run_dir / f"stage{stage}").glob("*/seed_*/result.json"))
    rows = [json.loads(path.read_text()) for path in paths]
    if stage == 1:
        ranking = rank_results(rows)
        top_variants = [row["variant"] for row in ranking[:3]]
        selection = {
            "stage": "stage1",
            "ranking": ranking,
            "top_variants": top_variants,
            "top_three": top_variants,
            "top3": top_variants,
            "finalists": top_variants,
        }
    else:
        stage1_selection_path = run_dir / "stage1_selection.json"
        if not stage1_selection_path.is_file():
            raise FileNotFoundError(
                "Stage-2 selection requires the current stage1_selection.json"
            )
        stage1_selection = json.loads(
            stage1_selection_path.read_text(encoding="utf-8")
        )
        eligible_variants = set(
            stage1_selection.get(
                "finalists", stage1_selection.get("top_variants", [])
            )
        )
        if len(eligible_variants) != 3:
            raise RuntimeError(
                f"Expected three current stage-1 finalists, found {sorted(eligible_variants)}"
            )
        by_variant: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            if row["variant"] not in eligible_variants:
                continue
            by_variant.setdefault(row["variant"], []).append(row)
        aggregates = []
        for variant, variant_rows in by_variant.items():
            if {row["seed"] for row in variant_rows} != set(STAGE2_SEEDS):
                continue
            aggregates.append(
                {
                    "state": "complete",
                    "variant": variant,
                    "best_val_loss": float(np.mean([r["best_val_loss"] for r in variant_rows])),
                    "recall@1": float(np.mean([r["recall@1"] for r in variant_rows])),
                    "recall@5": float(np.mean([r["recall@5"] for r in variant_rows])),
                    "mrr": float(np.mean([r["mrr"] for r in variant_rows])),
                    "seeds": sorted(row["seed"] for row in variant_rows),
                }
            )
        ranking = rank_results(aggregates)
        winner = ranking[0]["variant"] if ranking else None
        canonical = rank_results(by_variant.get(winner, []))[0] if winner else None
        selection = {"stage": "stage2", "ranking": ranking, "winner_variant": winner, "winner": winner, "canonical_run": canonical}
    atomic_json(run_dir / f"stage{stage}_selection.json", selection)
    return selection


def materialize_winner(args: argparse.Namespace) -> dict[str, str]:
    root = Path(args.run_dir).resolve()
    selection = select_stage(root, 2)
    canonical = selection.get("canonical_run")
    if not canonical:
        raise RuntimeError("Stage 2 has no eligible three-seed winner")
    source = Path(canonical["run_dir"])
    config = json.loads((source / "config.json").read_text())
    checkpoint = Path(args.base_checkpoint).resolve()
    seed_everything(int(canonical["seed"]))
    model = SnapViT(config)
    load_base_whitelist(model, checkpoint)
    adapter = torch.load(source / "best_adapter.pth", map_location="cpu", weights_only=False)
    expected_base_hash = sha256_file(checkpoint)
    if adapter.get("base_checkpoint_sha256") != expected_base_hash:
        raise RuntimeError(
            "Winner adapter base-checkpoint SHA-256 does not match the requested base."
        )
    if resume_config_signature(adapter.get("resolved_config", {})) != resume_config_signature(config):
        raise RuntimeError(
            "Winner adapter resolved configuration does not match its run config."
        )
    expected_adapter_names = set(fusion_parameter_names(model))
    expected_adapter_names.update(
        name
        for name, _ in model.named_parameters()
        if name.startswith(
            "ground_encoder.feature_extractor.convnext.stages_2."
        )
    )
    actual_adapter_names = set(adapter.get("adapter_state_dict", {}))
    missing_adapter = expected_adapter_names - actual_adapter_names
    extra_adapter = actual_adapter_names - expected_adapter_names
    if missing_adapter or extra_adapter:
        raise RuntimeError(
            "Winner adapter does not contain exactly the fusion and stages_2 "
            f"parameters: missing={sorted(missing_adapter)[:10]}, "
            f"unexpected={sorted(extra_adapter)[:10]}"
        )
    result = model.load_state_dict(adapter["adapter_state_dict"], strict=False)
    if result.unexpected_keys:
        raise RuntimeError(f"Winner adapter has unexpected keys: {result.unexpected_keys}")
    destination = root / "winner"
    destination.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(destination / "best_model.pth", model.state_dict())
    atomic_json(destination / "config.json", config)
    payload = {"checkpoint": str(destination / "best_model.pth"), "config": str(destination / "config.json")}
    atomic_json(destination / "winner.json", payload)
    return payload


def _common_cli_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-dir", default=str(REPO_ROOT / "ablation_results/fusion"))
    parser.add_argument("--base-checkpoint", default=str(DEFAULT_BASE_CHECKPOINT))
    parser.add_argument("--data-root", action="append", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")


def normalize_cli_args(args: argparse.Namespace) -> argparse.Namespace:
    if hasattr(args, "data_root") and args.data_root is None:
        args.data_root = [str(path) for path in DEFAULT_DATA_ROOTS]
    return args


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    from src.training.fusion_campaign_helpers import run_preflight

    root = Path(args.run_dir).resolve()
    expected_fingerprint = None
    cache_manifest_path = root / "feature_cache" / "manifest.json"
    split_manifest_path = root / "split_manifest.json"
    if cache_manifest_path.is_file():
        if split_manifest_path.is_file():
            config = resolved_campaign_config(args)
            manifest = json.loads(
                split_manifest_path.read_text(encoding="utf-8")
            )
            expected_roots = [
                str(Path(path).resolve()) for path in config["data_root"]
            ]
            if manifest.get("roots") == expected_roots and manifest.get(
                "split_seed"
            ) == STAGE1_SEED:
                ground_paths, overhead_paths, _, _ = feature_cache_source_paths(
                    config,
                    manifest,
                    seeds=(STAGE1_SEED, *STAGE2_SEEDS),
                )
                expected_fingerprint = cache_fingerprint(
                    base_checkpoint_sha256=sha256_file(
                        Path(args.base_checkpoint)
                    ),
                    config=config,
                    transform_description=TRANSFORM_DESCRIPTION,
                    ground_paths=ground_paths,
                    overhead_paths=overhead_paths,
                )
            else:
                expected_fingerprint = "invalid-split-manifest"
        else:
            expected_fingerprint = "missing-split-manifest"
    return run_preflight(
        run_dir=root,
        base_checkpoint=Path(args.base_checkpoint),
        data_roots=[Path(path) for path in args.data_root],
        cache_dir=root / "feature_cache",
        expected_cache_fingerprint=expected_fingerprint,
        device=args.device,
        python_bin=sys.executable,
        dry_run=args.dry_run,
    )


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.run_dir).resolve()
    config = resolved_campaign_config(args)
    checkpoint = Path(args.base_checkpoint).resolve()
    manifest = load_or_create_manifest(
        [Path(path) for path in config["data_root"]],
        root / "split_manifest.json",
        eligible_scenes=eligible_scene_paths(config),
    )
    cache = prepare_feature_cache(
        config,
        manifest,
        root,
        checkpoint,
        seeds=(STAGE1_SEED, *STAGE2_SEEDS),
    )
    return {
        "state": "complete",
        "status": "complete",
        "split_manifest": str(root / "split_manifest.json"),
        "feature_cache": str(root / "feature_cache"),
        "ground_shape": list(cache.ground.shape),
        "overhead_shape": list(cache.overhead.shape),
    }


def record_smoke_failure_probe(run_dir: Path) -> dict[str, Any]:
    """Exercise retry/failure reporting without changing a real experiment."""
    destination = run_dir / "smoke_probe" / "deliberate_failure" / "seed_0"
    destination.mkdir(parents=True, exist_ok=True)
    log_path = destination / "run.log"
    command = [
        sys.executable,
        "-c",
        "raise RuntimeError('deliberate smoke-test subprocess failure')",
    ]
    started = time.time()
    return_codes = []
    for attempt in range(2):
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"\n=== deliberate attempt {attempt + 1}/2 ===\n")
            completed = subprocess.run(
                command,
                cwd=REPO_ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        return_codes.append(completed.returncode)
    status = {
        "state": "failed",
        "status": "failed",
        "stage": "smoke_probe",
        "variant": "EXPECTED_FAILURE",
        "seed": 0,
        "expected_failure": True,
        "attempts": 2,
        "return_codes": return_codes,
        "runtime_seconds": time.time() - started,
        "error": "Deliberate smoke-test failure; campaign continuation is expected.",
    }
    atomic_json(destination / "status.json", status)
    return status


def campaign(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.run_dir).resolve()
    preflight_result = preflight(args)
    config = resolved_campaign_config(args)
    checkpoint = Path(args.base_checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    base_hash = sha256_file(checkpoint)
    campaign_config_path = root / "campaign_config.json"
    campaign_signature = {
        "base_checkpoint_sha256": base_hash,
        "config": resume_config_signature(config),
    }
    if campaign_config_path.is_file():
        existing_campaign = json.loads(
            campaign_config_path.read_text(encoding="utf-8")
        )
        existing_config = existing_campaign.get("config", existing_campaign)
        existing_signature = {
            "base_checkpoint_sha256": existing_campaign.get(
                "base_checkpoint_sha256", existing_campaign.get("base_sha256")
            ),
            "config": resume_config_signature(existing_config),
        }
        if existing_signature != campaign_signature:
            raise RuntimeError(
                "RUN_DIR was created for a different base checkpoint or campaign "
                "configuration. Choose a new RUN_DIR to avoid mixing experiments."
            )
    manifest = load_or_create_manifest(
        [Path(path) for path in config["data_root"]],
        root / "split_manifest.json",
        eligible_scenes=eligible_scene_paths(config),
    )
    atomic_json(
        root / "campaign_config.json",
        {
            **config,
            "base_checkpoint": str(checkpoint),
            "base_checkpoint_sha256": base_hash,
            "data_roots": list(config["data_root"]),
            "split_seed": STAGE1_SEED,
            "variants": VARIANTS,
            "seeds": STAGE2_SEEDS,
            "manifest_records": len(manifest["records"]),
            "training_scene_count": sum(
                row["split"] == "train" for row in manifest["records"]
            ),
            "validation_scene_count": sum(
                row["split"] == "validation" for row in manifest["records"]
            ),
            "excluded_scene_count": int(manifest.get("excluded_scene_count", 0)),
        },
    )
    if args.dry_run:
        plan = {
            "state": "planned",
            "status": "planned",
            "preflight": preflight_result,
            "stage1": list(VARIANTS),
            "stage2_seeds": list(STAGE2_SEEDS),
        }
        atomic_json(root / "campaign_status.json", plan)
        return plan

    prepare_feature_cache(
        config,
        manifest,
        root,
        checkpoint,
        seeds=(STAGE1_SEED, *STAGE2_SEEDS),
    )
    if args.smoke_test:
        record_smoke_failure_probe(root)

    def launch(stage: int, variant: str, seed: int) -> bool:
        command = [sys.executable, "-m", "src.training.fusion_ablation", "run-one", "--run-dir", str(root), "--base-checkpoint", str(checkpoint), "--device", args.device, "--num-workers", str(args.num_workers), "--stage", str(stage), "--variant", variant, "--seed", str(seed), "--cache-prepared"]
        for data_root in args.data_root:
            command += ["--data-root", data_root]
        if args.smoke_test:
            command.append("--smoke-test")
        destination = run_directory(root, stage, variant, seed)
        destination.mkdir(parents=True, exist_ok=True)
        log_path = destination / "run.log"
        started = time.time()
        last_return_code = None
        for attempt in range(2):
            with log_path.open("a", encoding="utf-8") as log:
                log.write(
                    f"\n=== attempt {attempt + 1}/2 at {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n"
                )
                log.flush()
                completed = subprocess.run(
                    command,
                    cwd=REPO_ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            last_return_code = completed.returncode
            from src.evaluation.fusion_ablation_report import build_report

            build_report(root)
            if completed.returncode == 0:
                return True
        atomic_json(
            destination / "status.json",
            {
                "state": "failed",
                "status": "failed",
                "stage": f"stage{stage}",
                "variant": variant,
                "seed": seed,
                "attempts": 2,
                "return_code": last_return_code,
                "error": f"subprocess exited with status {last_return_code}; see {log_path}",
                "runtime_seconds": time.time() - started,
            },
        )
        build_report(root)
        return False

    for variant in VARIANTS:
        launch(1, variant, STAGE1_SEED)
    stage1 = select_stage(root, 1)
    if len(stage1["top_variants"]) != 3:
        status = {
            "state": "failed",
            "status": "failed",
            "stage1": stage1,
            "error": "Fewer than three stage-1 variants completed successfully.",
        }
        atomic_json(root / "campaign_status.json", status)
        from src.evaluation.fusion_ablation_report import build_report

        build_report(root)
        raise RuntimeError(status["error"])
    for variant in stage1["top_variants"]:
        for seed in STAGE2_SEEDS:
            launch(2, variant, seed)
    stage2 = select_stage(root, 2)
    winner = materialize_winner(args) if stage2.get("winner_variant") else None
    if winner is None:
        status = {
            "state": "failed",
            "status": "failed",
            "stage1": stage1,
            "stage2": stage2,
            "winner": None,
            "error": "No stage-2 variant completed all three required seeds.",
        }
        atomic_json(root / "campaign_status.json", status)
        from src.evaluation.fusion_ablation_report import build_report

        build_report(root)
        raise RuntimeError(status["error"])

    from src.training.fusion_campaign_helpers import run_heldout_evaluations

    heldout = run_heldout_evaluations(
        run_dir=root,
        base_checkpoint=checkpoint,
        device=args.device,
        num_workers=args.num_workers,
        seeds=STAGE2_SEEDS,
        expected_run1_scenes=553,
        localization_subset_size=100,
        subset_seed=STAGE1_SEED,
        max_attempts=2,
        dry_run=bool(args.smoke_test),
    )
    heldout_failed = heldout.get("state") == "complete_with_failures"
    status = {
        "state": "complete_with_failures" if heldout_failed else "complete",
        "status": "complete_with_failures" if heldout_failed else "complete",
        "stage1": stage1,
        "stage2": stage2,
        "winner": winner,
        "heldout": heldout,
    }
    atomic_json(root / "campaign_status.json", status)
    from src.evaluation.fusion_ablation_report import build_report

    build_report(root)
    return status


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight_parser = subparsers.add_parser("preflight")
    _common_cli_args(preflight_parser)
    prepare_parser = subparsers.add_parser("prepare")
    _common_cli_args(prepare_parser)
    campaign_parser = subparsers.add_parser("campaign")
    _common_cli_args(campaign_parser)
    one_parser = subparsers.add_parser("run-one")
    _common_cli_args(one_parser)
    one_parser.add_argument("--stage", required=True, type=int, choices=(1, 2))
    one_parser.add_argument("--variant", required=True, choices=VARIANTS)
    one_parser.add_argument("--seed", required=True, type=int)
    one_parser.add_argument("--cache-prepared", action="store_true", help=argparse.SUPPRESS)
    select_parser = subparsers.add_parser("select")
    select_parser.add_argument("--run-dir", required=True)
    select_parser.add_argument("--stage", required=True, type=int, choices=(1, 2))
    materialize_parser = subparsers.add_parser("materialize")
    _common_cli_args(materialize_parser)
    args = normalize_cli_args(parser.parse_args(argv))
    if args.command == "preflight":
        result = preflight(args)
    elif args.command == "prepare":
        if args.dry_run:
            result = {
                "state": "planned",
                "status": "planned",
                "feature_cache": str(Path(args.run_dir).resolve() / "feature_cache"),
            }
        else:
            result = prepare(args)
    elif args.command == "campaign":
        result = campaign(args)
    elif args.command == "run-one":
        result = run_one(args)
    elif args.command == "select":
        result = select_stage(Path(args.run_dir).resolve(), args.stage)
    else:
        result = materialize_winner(args)
    print(json.dumps(_jsonable(result), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
