"""Deterministic, resumable retrieval and localization evaluation for SnapViT.

This entrypoint deliberately resolves the checkpoint configuration before creating
the dataset.  It supports either a standalone full-model checkpoint or a small
adapter checkpoint layered strictly over a base model checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.dataset import VineyardDataset
from src.evaluation.common import (
    compute_aggregate_stats,
    compute_position_stats,
    compute_retrieval_metrics,
    evaluate_pose_grid,
    get_gt_yaw,
    parse_angle_list,
    pool_bev_embeddings,
    pose_to_local_xy,
    project_scene_features,
    write_results_csv,
)
from src.models.snapvit import SnapViT


DEFAULT_CONFIG: dict[str, Any] = {
    "vit_model": "vit_small_patch16_224",
    "train_img_size": (224, 224),
    "feature_dim": 128,
    "ground_fusion_mode": "avg",
    "use_height_positional_encoding": False,
    "num_ugv_views": 8,
    "grid_size": (34, 34, 8),
    "grid_resolution": 0.3,
    "batch_size": 1,
    "device": "cuda:0" if torch.cuda.is_available() else "cpu",
    "use_depth": True,
    "depth_range": (0.0, 5.0),
    "ground_tile_size": 10.0,
    "consecutive_frames": True,
    "edge_margin_m": 1.0,
    "pretrained_backbones": False,
}

_FULL_STATE_KEYS = ("full_model_state_dict", "model_state_dict", "state_dict")
_ADAPTER_STATE_KEYS = ("adapter_state_dict", "trainable_state_dict")
_CONFIG_KEYS = ("resolved_config", "config", "model_config")


def _json_load(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
    os.replace(temporary, path)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _state_dict_from(payload: Any, keys: Iterable[str]) -> dict[str, torch.Tensor] | None:
    if not isinstance(payload, dict):
        return None
    for key in keys:
        candidate = payload.get(key)
        if isinstance(candidate, dict) and candidate and all(torch.is_tensor(v) for v in candidate.values()):
            return candidate
    if payload and all(isinstance(k, str) and torch.is_tensor(v) for k, v in payload.items()):
        return payload
    return None


def _checkpoint_sidecar(checkpoint: Path) -> dict[str, Any]:
    for candidate in (checkpoint.with_name("config.json"), checkpoint.parent.parent / "config.json"):
        if candidate.is_file():
            value = _json_load(candidate)
            if not isinstance(value, dict):
                raise TypeError(f"Checkpoint config must be an object: {candidate}")
            return value
    return {}


def resolve_eval_config(
    checkpoint: Path,
    checkpoint_payload: Any,
    *,
    device: str | None,
    num_ugv_views: int,
) -> dict[str, Any]:
    """Restore all saved fields before dataset/model construction."""
    config = dict(DEFAULT_CONFIG)
    config.update(_checkpoint_sidecar(checkpoint))
    if isinstance(checkpoint_payload, dict):
        for key in _CONFIG_KEYS:
            embedded = checkpoint_payload.get(key)
            if isinstance(embedded, dict):
                config.update(embedded)
                break

    for tuple_key in ("train_img_size", "grid_size", "depth_range"):
        if tuple_key in config:
            config[tuple_key] = tuple(config[tuple_key])
    if "model_name" not in config and "vit_model" in config:
        config["model_name"] = config["vit_model"]
    if "vit_model" not in config and "model_name" in config:
        config["vit_model"] = config["model_name"]

    config["device"] = device or str(config.get("device", DEFAULT_CONFIG["device"]))
    config["num_ugv_views"] = int(num_ugv_views)
    config["batch_size"] = 1
    # Loading a state dict does not require pretrained initialization and must not
    # unexpectedly access the network on an unattended evaluation worker.
    config["pretrained_backbones"] = False
    return config


def _resolve_base_checkpoint(
    adapter_payload: dict[str, Any], adapter_path: Path, cli_base: str | None
) -> Path:
    raw = cli_base or adapter_payload.get("base_checkpoint") or adapter_payload.get("base_checkpoint_path")
    if not raw:
        raise ValueError("Adapter checkpoint requires --base-checkpoint or embedded base_checkpoint_path")
    base = Path(raw).expanduser()
    if not base.is_absolute():
        base = (adapter_path.parent / base).resolve()
    if not base.is_file():
        raise FileNotFoundError(f"Base checkpoint not found: {base}")
    return base


def load_model_strict(
    model: SnapViT,
    checkpoint_path: Path,
    checkpoint_payload: Any,
    base_checkpoint: str | None = None,
) -> str:
    """Strictly load a full model, or reconstruct one from base plus adapter."""
    adapter_state = _state_dict_from(checkpoint_payload, _ADAPTER_STATE_KEYS)
    if adapter_state is None:
        full_state = _state_dict_from(checkpoint_payload, _FULL_STATE_KEYS)
        if full_state is None:
            raise TypeError(f"No model state dict found in {checkpoint_path}")
        model.load_state_dict(full_state, strict=True)
        return "full"

    if not isinstance(checkpoint_payload, dict):
        raise TypeError("Adapter checkpoint payload must be a dictionary")
    base_path = _resolve_base_checkpoint(checkpoint_payload, checkpoint_path, base_checkpoint)
    expected_base_hash = checkpoint_payload.get("base_checkpoint_sha256") or checkpoint_payload.get("base_checkpoint_hash")
    if not isinstance(expected_base_hash, str) or not expected_base_hash:
        raise ValueError("Adapter checkpoint is missing required base_checkpoint_sha256")
    actual_base_hash = _file_sha256(base_path)
    if actual_base_hash.lower() != expected_base_hash.lower():
        raise RuntimeError(
            f"Base checkpoint SHA-256 mismatch for {base_path}: expected {expected_base_hash}, got {actual_base_hash}"
        )
    base_payload = torch.load(base_path, map_location="cpu", weights_only=False)
    base_state = _state_dict_from(base_payload, _FULL_STATE_KEYS)
    if base_state is None:
        raise TypeError(f"No full model state dict found in base checkpoint {base_path}")

    model_state = model.state_dict()
    fusion_prefixes = (
        "ground_encoder.feature_norm.",
        "ground_encoder.token_mlp.",
        "ground_encoder.column_mlp.",
        "ground_encoder.token_attention.",
        "ground_encoder.view_attention.",
    )

    def is_fusion_key(key: str) -> bool:
        return key.startswith(fusion_prefixes) or key == "ground_encoder.column_residual_scale"

    adapter_config: dict[str, Any] = {}
    for config_key in _CONFIG_KEYS:
        candidate = checkpoint_payload.get(config_key)
        if isinstance(candidate, dict):
            adapter_config = candidate
            break
    try:
        is_stage2_adapter = int(adapter_config.get("fusion_ablation_stage", -1)) == 2
    except (TypeError, ValueError):
        is_stage2_adapter = False
    stage2_prefix = "ground_encoder.feature_extractor.convnext.stages_2."
    required_adapter_keys = {key for key in model_state if is_fusion_key(key)}
    if is_stage2_adapter:
        required_adapter_keys.update(
            key for key in model_state if key.startswith(stage2_prefix)
        )
        missing_adapter = required_adapter_keys - set(adapter_state)
        extra_adapter = set(adapter_state) - required_adapter_keys
        if missing_adapter or extra_adapter:
            raise RuntimeError(
                "Stage-2 adapter must contain exactly the fusion and stages_2 tensors: "
                f"missing={sorted(missing_adapter)[:20]}, "
                f"unexpected={sorted(extra_adapter)[:20]}"
            )

    reconstructed: dict[str, torch.Tensor] = {}
    for key, expected in model_state.items():
        # A variant must never inherit a coincidentally shape-compatible fusion
        # tensor from the N0 base.  Every fusion tensor comes from the adapter;
        # every other tensor comes from the strict base (unless explicitly saved
        # as another trainable parameter, e.g. the stage-2 fine-tune).
        if is_fusion_key(key) and key not in adapter_state:
            raise RuntimeError(f"Adapter is missing required fusion parameter {key!r}")
        source = adapter_state.get(key, base_state.get(key))
        if source is None:
            raise RuntimeError(f"Neither base nor adapter provides required parameter {key!r}")
        if tuple(source.shape) != tuple(expected.shape):
            raise RuntimeError(
                f"Shape mismatch for {key}: checkpoint {tuple(source.shape)} != model {tuple(expected.shape)}"
            )
        reconstructed[key] = source

    unexpected_adapter = set(adapter_state) - set(model_state)
    if unexpected_adapter:
        raise RuntimeError(f"Unexpected adapter parameters: {sorted(unexpected_adapter)[:20]}")
    unexpected = set(base_state) - set(model_state)
    # The base is commonly N0. Other variants intentionally omit or reshape its
    # fusion head, so base-only fusion keys are irrelevant to reconstruction.
    unexpected = {key for key in unexpected if not is_fusion_key(key)}
    # Known removed legacy projection keys are consumed by GroundEncoder's loader.
    unexpected -= {
        "ground_encoder.projection_layer.weight",
        "ground_encoder.projection_layer.bias",
    }
    if unexpected:
        raise RuntimeError(f"Unexpected non-fusion base parameters: {sorted(unexpected)[:20]}")
    model.load_state_dict(reconstructed, strict=True)
    return f"base+adapter:{base_path}"


def _manifest_entries(manifest_path: str | None, split: str) -> set[str] | None:
    if not manifest_path:
        return None
    manifest = _json_load(Path(manifest_path))
    if isinstance(manifest, dict) and "splits" in manifest:
        manifest = manifest["splits"]
    if isinstance(manifest, dict):
        if split not in manifest:
            raise KeyError(f"Split {split!r} is absent from {manifest_path}")
        manifest = manifest[split]
    if not isinstance(manifest, list):
        raise TypeError("Scene manifest must be a list or a mapping of split names to lists")

    entries: set[str] = set()
    for item in manifest:
        if isinstance(item, dict):
            value = item.get("scene_key") or item.get("scene_id") or item.get("path")
        else:
            value = item
        if not isinstance(value, str):
            raise TypeError(f"Invalid scene manifest entry: {item!r}")
        entries.add(value.rstrip("/"))
    return entries


def scene_key(scene_path: str) -> str:
    path = Path(scene_path)
    return f"{path.parent.name}/{path.name}"


def filter_dataset_by_manifest(dataset: VineyardDataset, entries: set[str] | None) -> None:
    if entries is None:
        return
    selected = []
    for folder in dataset.scene_folders:
        path = Path(folder)
        candidates = {scene_key(folder), path.name, str(path), str(path.resolve())}
        if candidates & entries:
            selected.append(folder)
    if not selected:
        raise ValueError("Scene manifest selected zero scenes from the configured data roots")
    dataset.scene_folders = selected


class DeterministicSceneDataset(Dataset):
    """Seed random view selection independently for each stable scene identity."""

    def __init__(self, dataset: VineyardDataset, seed: int):
        self.dataset = dataset
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        key = scene_key(self.dataset.scene_folders[index])
        digest = hashlib.sha256(f"{self.seed}:{key}".encode("utf-8")).digest()
        derived_seed = int.from_bytes(digest[:8], "big")
        random.seed(derived_seed)
        np.random.seed(derived_seed % (2**32))
        torch.manual_seed(derived_seed % (2**63 - 1))
        sample = self.dataset[index]
        sample["scene_key"] = key
        return sample


def deterministic_subset(scene_keys: list[str], count: int, seed: int) -> set[str]:
    if count <= 0 or count >= len(scene_keys):
        return set(scene_keys)
    ordered = sorted(
        scene_keys,
        key=lambda key: hashlib.sha256(f"{seed}:{key}".encode("utf-8")).hexdigest(),
    )
    return set(ordered[:count])


def _cache_path(cache_dir: Path, key: str) -> Path:
    safe = key.replace("/", "__").replace(os.sep, "__")
    return cache_dir / f"{safe}.json"


def _worker_init(worker_id: int) -> None:
    # DeterministicSceneDataset reseeds again per item; this covers library code
    # that might execute before its __getitem__ body.
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def _build_dataset(roots: list[str], config: dict[str, Any], manifest: str | None, split: str) -> VineyardDataset:
    image_transforms = transforms.Compose(
        [
            transforms.Resize(config["train_img_size"], antialias=True),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    depth_transforms = transforms.Compose(
        [
            transforms.Resize(config["train_img_size"], antialias=True),
            transforms.ConvertImageDtype(torch.float),
        ]
    )
    dataset = VineyardDataset(
        root_dir=roots,
        config=config,
        transforms=image_transforms,
        depth_transforms=depth_transforms,
        consecutive_frames=bool(config.get("consecutive_frames", True)),
    )
    filter_dataset_by_manifest(dataset, _manifest_entries(manifest, split))
    if not dataset.scene_folders:
        raise ValueError(f"No scenes found under {roots}")
    return dataset


def main(args: argparse.Namespace) -> dict[str, Any]:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    logger = logging.getLogger(__name__)
    output_dir = Path(args.output_dir)
    cache_dir = output_dir / "per_scene"
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = resolve_eval_config(
        checkpoint,
        checkpoint_payload,
        device=args.device,
        num_ugv_views=args.num_ugv_views,
    )

    roots = [str(Path(root).expanduser().resolve()) for root in args.data_root]
    for root in roots:
        if not Path(root).is_dir():
            raise FileNotFoundError(f"Data root not found: {root}")
    dataset = _build_dataset(roots, config, args.scene_manifest, args.manifest_split)

    adapter_state = _state_dict_from(checkpoint_payload, _ADAPTER_STATE_KEYS)
    resolved_base = (
        _resolve_base_checkpoint(checkpoint_payload, checkpoint, args.base_checkpoint)
        if adapter_state is not None and isinstance(checkpoint_payload, dict)
        else None
    )
    evaluation_context = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _file_sha256(checkpoint),
        "base_checkpoint": str(resolved_base) if resolved_base else None,
        "base_checkpoint_sha256": _file_sha256(resolved_base) if resolved_base else None,
        "data_roots": roots,
        "scene_manifest": str(Path(args.scene_manifest).resolve()) if args.scene_manifest else None,
        "scene_manifest_sha256": (
            _file_sha256(Path(args.scene_manifest).expanduser().resolve())
            if args.scene_manifest
            else None
        ),
        "manifest_split": args.manifest_split,
        "num_ugv_views": config["num_ugv_views"],
        "seed": args.seed,
        "subset_seed": args.subset_seed,
        "localization_samples": args.localization_samples,
        "top_k": args.top_k,
        "softmax_temp": args.softmax_temp,
        "grid_range_m": args.grid_range_m,
        "grid_resolution_m": args.grid_resolution_m,
        "yaw_search_angles_deg": args.yaw_search_angles_deg,
        "localization_thresholds_m": args.localization_thresholds_m,
    }
    context_path = output_dir / "evaluation_context.json"
    if args.resume and context_path.is_file() and _json_load(context_path) != evaluation_context:
        raise RuntimeError(
            f"Evaluation context changed for {output_dir}; use a new --output-dir or pass --no-resume"
        )
    _atomic_json(context_path, evaluation_context)
    wrapped = DeterministicSceneDataset(dataset, args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    dataloader = DataLoader(
        wrapped,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        worker_init_fn=_worker_init,
        generator=generator,
        pin_memory=str(config["device"]).startswith("cuda"),
    )

    model = SnapViT(config).to(config["device"])
    load_kind = load_model_strict(model, checkpoint, checkpoint_payload, args.base_checkpoint)
    model.eval()
    logger.info("Loaded %s checkpoint strictly; evaluating %d scenes with %d UGV views", load_kind, len(dataset), config["num_ugv_views"])

    all_keys = [scene_key(path) for path in dataset.scene_folders]
    localization_keys = deterministic_subset(all_keys, args.localization_samples, args.subset_seed)
    yaw_candidates = parse_angle_list(args.yaw_search_angles_deg)
    scene_records: dict[str, dict[str, Any]] = {}

    with torch.inference_mode():
        for batch in tqdm(dataloader, total=len(wrapped), desc="Held-out evaluation"):
            key = batch.pop("scene_key")[0]
            cache_path = _cache_path(cache_dir, key)
            record = _json_load(cache_path) if args.resume and cache_path.is_file() else {"scene_id": key}
            needs_retrieval = "ground_embedding" not in record or "overhead_embedding" not in record
            needs_localization = key in localization_keys and "localization" not in record

            if needs_retrieval or needs_localization:
                uav_data = {name: value.to(config["device"]) for name, value in batch["uav_data"].items()}
                ugv_data = {name: value.to(config["device"]) for name, value in batch["ugv_data"].items()}
                encoded_ground = model.ground_encoder.encode_from_dict(ugv_data)
                encoded_overhead = model.overhead_encoder.encode_from_dict(uav_data)

                if needs_retrieval:
                    ground_bev, overhead_bev, validity = project_scene_features(
                        model, ugv_data, uav_data, encoded_ground, encoded_overhead
                    )
                    record["ground_embedding"] = pool_bev_embeddings(ground_bev, validity)[0].cpu().tolist()
                    record["overhead_embedding"] = pool_bev_embeddings(overhead_bev)[0].cpu().tolist()

                if needs_localization:
                    gt_xy = pose_to_local_xy(ugv_data["camera_poses"][0, 0])
                    gt_yaw = get_gt_yaw(ugv_data["camera_poses"][0, 0])
                    _, prob_map, yaw_map, _, poses = evaluate_pose_grid(
                        model=model,
                        ugv_data=ugv_data,
                        uav_data=uav_data,
                        grid_range_m=args.grid_range_m,
                        grid_resolution_m=args.grid_resolution_m,
                        softmax_temp=args.softmax_temp,
                        yaw_candidates_deg=yaw_candidates,
                        encoded_ground=encoded_ground,
                        encoded_overhead=encoded_overhead,
                    )
                    stats = compute_position_stats(
                        prob_map,
                        yaw_map,
                        poses,
                        gt_xy,
                        gt_yaw,
                        yaw_candidates,
                        top_k=args.top_k,
                        distance_thresholds=args.localization_thresholds_m,
                    )
                    stats["scene_id"] = key
                    record["localization"] = stats
                _atomic_json(cache_path, record)
            scene_records[key] = record

    missing_retrieval = [key for key in all_keys if key not in scene_records or "ground_embedding" not in scene_records[key]]
    if missing_retrieval:
        raise RuntimeError(f"Missing retrieval records for {len(missing_retrieval)} scenes")
    ground = torch.tensor([scene_records[key]["ground_embedding"] for key in all_keys], dtype=torch.float32)
    overhead = torch.tensor([scene_records[key]["overhead_embedding"] for key in all_keys], dtype=torch.float32)
    retrieval = compute_retrieval_metrics(ground, overhead, ks=(1, 5))

    localization_rows = [
        scene_records[key]["localization"]
        for key in all_keys
        if key in localization_keys and "localization" in scene_records[key]
    ]
    if len(localization_rows) != len(localization_keys):
        raise RuntimeError(f"Expected {len(localization_keys)} localization rows, found {len(localization_rows)}")
    localization = compute_aggregate_stats(localization_rows, args.localization_thresholds_m)
    write_results_csv(str(output_dir / "localization_per_scene.csv"), localization_rows)
    _atomic_json(output_dir / "localization_summary.json", localization)

    summary = {
        "checkpoint": str(checkpoint),
        "checkpoint_load": load_kind,
        "data_roots": roots,
        "scene_manifest": args.scene_manifest,
        "scene_manifest_sha256": evaluation_context["scene_manifest_sha256"],
        "manifest_split": args.manifest_split,
        "num_retrieval_scenes": len(all_keys),
        "num_localization_scenes": len(localization_rows),
        "num_ugv_views": config["num_ugv_views"],
        "retrieval": retrieval,
        "localization": localization,
        "top_k": args.top_k,
        "grid_range_m": args.grid_range_m,
        "grid_resolution_m": args.grid_resolution_m,
        "yaw_candidates_deg": yaw_candidates,
        "localization_thresholds_m": args.localization_thresholds_m,
        "seed": args.seed,
        "localization_subset_seed": args.subset_seed,
    }
    _atomic_json(output_dir / "evaluation_summary.json", summary)
    _atomic_json(output_dir / "resolved_config.json", config)
    logger.info("Recall@1 %.4f, Recall@5 %.4f, MRR %.4f", retrieval["recall@1"], retrieval["recall@5"], retrieval["mrr"])
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run deterministic held-out SnapViT evaluation.")
    parser.add_argument("--data-root", "--data_root", dest="data_root", action="append", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--base-checkpoint", default=None, help="Base full-model checkpoint for a small adapter.")
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", default="evaluation_pipeline")
    parser.add_argument("--scene-manifest", default=None)
    parser.add_argument("--manifest-split", default="test")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-ugv-views", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--localization-samples", type=int, default=100, help="<=0 evaluates localization on every scene.")
    parser.add_argument("--subset-seed", type=int, default=42)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--softmax-temp", type=float, default=0.07)
    parser.add_argument("--grid-range-m", type=float, default=5.0)
    parser.add_argument("--grid-resolution-m", type=float, default=0.25)
    parser.add_argument("--yaw-search-angles-deg", default="-90,-45,0,45,90,135,180")
    parser.add_argument("--localization-thresholds-m", type=float, nargs="+", default=[1.0, 2.0, 3.0])
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
