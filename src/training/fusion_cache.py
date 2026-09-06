"""Read-only FP32 feature cache used by fusion-only ablation training.

The cache is keyed by canonical source image paths, so an image referenced by
many scenes is encoded once. Geometry (depth, pose, intrinsics and grid) stays
live; only the frozen ground/overhead backbone outputs are cached.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import torch
from torchvision.io import ImageReadMode, read_image


CACHE_VERSION = 1
FINGERPRINT_CONFIG_KEYS = (
    "vit_model",
    "model_name",
    "train_img_size",
    "feature_dim",
    "num_ugv_views",
    "pretrained_backbones",
)


def canonical_paths(paths: Iterable[str | os.PathLike[str]]) -> list[str]:
    return sorted({str(Path(path).resolve()) for path in paths})


def _stable_json(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()


def cache_fingerprint(
    *,
    base_checkpoint_sha256: str,
    config: Mapping[str, Any],
    transform_description: str,
    ground_paths: Sequence[str | os.PathLike[str]],
    overhead_paths: Sequence[str | os.PathLike[str]],
) -> str:
    """Hash every input that can affect a frozen encoder output.

    File size and nanosecond mtime catch replaced source images without hashing
    several thousand PNG files. The checkpoint itself is represented by its
    already-computed SHA256.
    """
    sources = []
    for kind, paths in (("ground", ground_paths), ("overhead", overhead_paths)):
        for path_string in canonical_paths(paths):
            path = Path(path_string)
            stat = path.stat()
            sources.append((kind, path_string, stat.st_size, stat.st_mtime_ns))
    payload = {
        "cache_version": CACHE_VERSION,
        "base_checkpoint_sha256": base_checkpoint_sha256,
        "config": {key: config.get(key) for key in FINGERPRINT_CONFIG_KEYS},
        "transform": transform_description,
        "sources": sources,
    }
    return hashlib.sha256(_stable_json(payload)).hexdigest()


def _owning_root(scene_path: Path, root_dirs: Sequence[str]) -> Path:
    scene = scene_path.resolve()
    matches = []
    for root_string in root_dirs:
        root = Path(root_string).resolve()
        try:
            scene.relative_to(root)
        except ValueError:
            continue
        matches.append(root)
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one owning root for scene {scene}, found {matches}")
    return matches[0]


def _resolve_source_path(scene: Path, owner: Path, value: str) -> str:
    source = Path(value)
    if source.is_absolute():
        candidate = source
    elif (scene / source).is_file():
        candidate = scene / source
    else:
        candidate = owner / source
    if not candidate.is_file():
        raise FileNotFoundError(f"Referenced image does not exist: {candidate}")
    return str(candidate.resolve())


def _consecutive_selection(views: Sequence[Mapping[str, Any]], count: int, rng: random.Random):
    # VineyardDataset resolves a selected camera id to the first matching view.
    # Preserve that detail for metadata containing duplicate camera_idx values.
    by_id: dict[int, Mapping[str, Any]] = {}
    for view in views:
        by_id.setdefault(int(view["camera_idx"]), view)
    ids = sorted(by_id)
    groups: list[list[int]] = []
    current = [ids[0]]
    for frame_id in ids[1:]:
        if frame_id == current[-1] + 1:
            current.append(frame_id)
        else:
            groups.append(current)
            current = [frame_id]
    groups.append(current)
    valid = [group for group in groups if len(group) >= count]
    if valid:
        group = rng.choice(valid)
        start = rng.randint(0, len(group) - count)
        selected = group[start : start + count]
    else:
        group = rng.choice(groups)
        selected = sorted((group * (count * count))[:count])
    return [by_id[frame_id] for frame_id in selected]


def selected_scene_sources(dataset: Any, dataset_index: int, seed: int) -> dict[str, Any]:
    """Reproduce ``VineyardDataset.__getitem__`` view selection without image I/O.

    This function intentionally uses a local RNG, preventing cache preparation
    from perturbing global training randomness.
    """
    scene = Path(dataset.scene_folders[dataset_index]).resolve()
    owner = _owning_root(scene, dataset.root_dirs)
    with (scene / "metadata.json").open(encoding="utf-8") as stream:
        metadata = json.load(stream)
    views = dataset.filter_views_by_edge_margin(metadata["ugv_images"])
    if not views:
        raise ValueError(f"Scene contains no valid ground views: {scene}")
    count = int(dataset.config["num_ugv_views"])
    rng = random.Random(int(seed) * 1_000_003 + int(dataset_index))
    if dataset.consecutive_frames:
        selected = _consecutive_selection(views, count, rng)
    else:
        selected = list(views)
        rng.shuffle(selected)
        if len(selected) < count:
            selected = (selected * (count // len(selected) + 1))[:count]
        else:
            selected = selected[:count]
    return {
        "scene": str(scene),
        "overhead_path": _resolve_source_path(scene, owner, metadata["uav_image_path"]),
        "ground_paths": [
            _resolve_source_path(scene, owner, view["image_path"]) for view in selected
        ],
    }


def scene_source_records(dataset: Any, indices: Iterable[int], seed: int) -> list[dict[str, Any]]:
    return [selected_scene_sources(dataset, int(index), seed) for index in indices]


class FP32FeatureCache:
    """Memory-mapped frozen encoder features, indexed by canonical image path."""

    MANIFEST = "manifest.json"
    GROUND_ARRAY = "ground_features.npy"
    OVERHEAD_ARRAY = "overhead_features.npy"

    def __init__(self, cache_dir: str | os.PathLike[str], expected_fingerprint: str | None = None):
        self.cache_dir = Path(cache_dir)
        with (self.cache_dir / self.MANIFEST).open(encoding="utf-8") as stream:
            self.manifest = json.load(stream)
        if self.manifest.get("version") != CACHE_VERSION:
            raise RuntimeError("Unsupported feature-cache version")
        if expected_fingerprint and self.manifest.get("fingerprint") != expected_fingerprint:
            raise RuntimeError("Feature-cache fingerprint mismatch")
        self.ground = np.load(self.cache_dir / self.GROUND_ARRAY, mmap_mode="r")
        self.overhead = np.load(self.cache_dir / self.OVERHEAD_ARRAY, mmap_mode="r")
        if self.ground.dtype != np.float32 or self.overhead.dtype != np.float32:
            raise RuntimeError("Feature cache must use FP32 arrays")
        if self.ground.flags.writeable or self.overhead.flags.writeable:
            raise RuntimeError("Feature cache unexpectedly opened writeable")
        ground_paths = self.manifest.get("ground_paths", [])
        overhead_paths = self.manifest.get("overhead_paths", [])
        expected_ground_shape = (
            len(ground_paths),
            *tuple(self.manifest.get("ground_feature_shape", ())),
        )
        expected_overhead_shape = (
            len(overhead_paths),
            *tuple(self.manifest.get("overhead_feature_shape", ())),
        )
        if self.ground.ndim != 4 or tuple(self.ground.shape) != expected_ground_shape:
            raise RuntimeError(
                "Ground feature array shape does not match its cache manifest: "
                f"{tuple(self.ground.shape)} != {expected_ground_shape}"
            )
        if self.overhead.ndim != 4 or tuple(self.overhead.shape) != expected_overhead_shape:
            raise RuntimeError(
                "Overhead feature array shape does not match its cache manifest: "
                f"{tuple(self.overhead.shape)} != {expected_overhead_shape}"
            )
        if len(set(ground_paths)) != len(ground_paths) or len(set(overhead_paths)) != len(overhead_paths):
            raise RuntimeError("Feature-cache path indices must be unique")
        self.ground_index = {path: i for i, path in enumerate(ground_paths)}
        self.overhead_index = {path: i for i, path in enumerate(overhead_paths)}

    @classmethod
    def build(
        cls,
        cache_dir: str | os.PathLike[str],
        *,
        fingerprint: str,
        ground_paths: Sequence[str | os.PathLike[str]],
        overhead_paths: Sequence[str | os.PathLike[str]],
        ground_encoder: Callable[[torch.Tensor], torch.Tensor],
        overhead_encoder: Callable[[torch.Tensor], torch.Tensor],
        image_transform: Callable[[torch.Tensor], torch.Tensor],
        device: str | torch.device,
        batch_size: int = 32,
        image_loader: Callable[[str], torch.Tensor] | None = None,
    ) -> "FP32FeatureCache":
        """Build atomically and return the new read-only cache.

        Encoder callables must return rank-4 ``NCHW`` tensors. Both are put in
        evaluation mode by the caller/model; this method additionally disables
        gradients throughout construction.
        """
        destination = Path(cache_dir)
        if (destination / cls.MANIFEST).is_file():
            try:
                return cls(destination, expected_fingerprint=fingerprint)
            except (RuntimeError, OSError, ValueError):
                pass
        temporary = destination.with_name(destination.name + ".building")
        if temporary.exists():
            shutil.rmtree(temporary)
        temporary.mkdir(parents=True)
        ground = canonical_paths(ground_paths)
        overhead = canonical_paths(overhead_paths)
        loader = image_loader or (lambda path: read_image(path, mode=ImageReadMode.RGB))

        def encode(paths: list[str], encoder, filename: str) -> tuple[int, ...]:
            output_map = None
            output_shape: tuple[int, ...] | None = None
            with torch.no_grad():
                for start in range(0, len(paths), batch_size):
                    selected = paths[start : start + batch_size]
                    images = torch.stack([image_transform(loader(path)) for path in selected])
                    features = encoder(images.to(device, non_blocking=True))
                    if isinstance(features, Mapping):
                        features = features.get("features_2d")
                    if not torch.is_tensor(features) or features.ndim != 4:
                        raise ValueError("Cached encoder must return an NCHW tensor")
                    values = features.detach().float().cpu().numpy()
                    if output_map is None:
                        output_shape = tuple(values.shape[1:])
                        output_map = np.lib.format.open_memmap(
                            temporary / filename,
                            mode="w+",
                            dtype=np.float32,
                            shape=(len(paths), *output_shape),
                        )
                    elif tuple(values.shape[1:]) != output_shape:
                        raise RuntimeError("Encoder returned inconsistent feature shapes")
                    output_map[start : start + len(selected)] = values
            if output_map is None:
                raise ValueError("Cannot build a feature cache from an empty path list")
            output_map.flush()
            del output_map
            assert output_shape is not None
            return output_shape

        ground_shape = encode(ground, ground_encoder, cls.GROUND_ARRAY)
        overhead_shape = encode(overhead, overhead_encoder, cls.OVERHEAD_ARRAY)
        manifest = {
            "version": CACHE_VERSION,
            "fingerprint": fingerprint,
            "dtype": "float32",
            "ground_paths": ground,
            "overhead_paths": overhead,
            "ground_feature_shape": ground_shape,
            "overhead_feature_shape": overhead_shape,
        }
        manifest_tmp = temporary / (cls.MANIFEST + ".tmp")
        manifest_tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(manifest_tmp, temporary / cls.MANIFEST)
        if destination.exists():
            old = destination.with_name(destination.name + ".old")
            if old.exists():
                shutil.rmtree(old)
            os.replace(destination, old)
            os.replace(temporary, destination)
            shutil.rmtree(old)
        else:
            os.replace(temporary, destination)
        return cls(destination, expected_fingerprint=fingerprint)

    def _lookup(self, paths: Sequence[str | os.PathLike[str]], index, array) -> torch.Tensor:
        canonical = [str(Path(path).resolve()) for path in paths]
        try:
            rows = [index[path] for path in canonical]
        except KeyError as error:
            raise KeyError(f"Image is absent from feature cache: {error.args[0]}") from error
        # Copy protects the read-only memory map from accidental tensor writes.
        return torch.from_numpy(np.array(array[rows], dtype=np.float32, copy=True))

    def ground_tensor(
        self, paths: Sequence[str | os.PathLike[str]], device: str | torch.device | None = None
    ) -> torch.Tensor:
        tensor = self._lookup(paths, self.ground_index, self.ground)
        return tensor.to(device, non_blocking=True) if device is not None else tensor

    def overhead_tensor(
        self, paths: Sequence[str | os.PathLike[str]], device: str | torch.device | None = None
    ) -> torch.Tensor:
        tensor = self._lookup(paths, self.overhead_index, self.overhead)
        return tensor.to(device, non_blocking=True) if device is not None else tensor

    def encoded_ground(
        self,
        batch_paths: Sequence[Sequence[str | os.PathLike[str]]],
        image_size: tuple[int, int],
        device: str | torch.device,
    ) -> dict[str, Any]:
        batch, views = len(batch_paths), len(batch_paths[0])
        flat = [path for paths in batch_paths for path in paths]
        features = self.ground_tensor(flat, device=device)
        return {
            "features_2d": features.view(batch, views, *features.shape[1:]),
            "image_size": tuple(image_size),
        }

    def encoded_overhead(
        self,
        paths: Sequence[str | os.PathLike[str]],
        device: str | torch.device,
    ) -> torch.Tensor:
        """Return the tensor accepted by ``OverheadEncoder.project``."""
        return self.overhead_tensor(paths, device=device)
