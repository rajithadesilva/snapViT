import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.training.fusion_cache import (
    FP32FeatureCache,
    cache_fingerprint,
    selected_scene_sources,
)


class _SourceDataset:
    def __init__(self, scene: Path, roots: list[Path]):
        self.scene_folders = [str(scene)]
        self.root_dirs = [str(root) for root in roots]
        self.config = {"num_ugv_views": 8}
        self.consecutive_frames = True

    @staticmethod
    def filter_views_by_edge_margin(views):
        return list(views)


class FusionFeatureCacheTests(unittest.TestCase):
    def test_fingerprint_changes_with_checkpoint_config_transform_and_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "image.png"
            source.write_bytes(b"pixels")
            arguments = dict(
                base_checkpoint_sha256="abc",
                config={"vit_model": "convnext_base", "train_img_size": [224, 224]},
                transform_description="resize-normalize-v1",
                ground_paths=[source],
                overhead_paths=[source],
            )
            first = cache_fingerprint(**arguments)
            self.assertEqual(first, cache_fingerprint(**arguments))
            arguments["transform_description"] = "resize-normalize-v2"
            self.assertNotEqual(first, cache_fingerprint(**arguments))

    def test_build_deduplicates_paths_and_reopens_read_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = [root / "a.png", root / "b.png"]
            for index, path in enumerate(paths):
                path.write_bytes(bytes([index]))

            def loader(path):
                value = float(paths.index(Path(path)))
                return torch.full((3, 2, 2), value)

            cache = FP32FeatureCache.build(
                root / "cache",
                fingerprint="fingerprint",
                ground_paths=[paths[1], paths[0], paths[0]],
                overhead_paths=paths,
                ground_encoder=lambda images: images[:, :1] + 10,
                overhead_encoder=lambda images: images[:, :2] + 20,
                image_transform=lambda image: image,
                image_loader=loader,
                device="cpu",
                batch_size=1,
            )
            self.assertEqual(cache.ground.shape, (2, 1, 2, 2))
            self.assertEqual(cache.overhead.shape, (2, 2, 2, 2))
            self.assertFalse(cache.ground.flags.writeable)
            values = cache.ground_tensor([paths[1], paths[0]])
            self.assertEqual(values.dtype, torch.float32)
            self.assertTrue(torch.equal(values[:, 0, 0, 0], torch.tensor([11.0, 10.0])))
            values[0] = -1
            self.assertNotEqual(float(cache.ground[1, 0, 0, 0]), -1.0)

    def test_expected_fingerprint_is_enforced(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "a.png"
            path.write_bytes(b"a")
            FP32FeatureCache.build(
                root / "cache",
                fingerprint="correct",
                ground_paths=[path],
                overhead_paths=[path],
                ground_encoder=lambda images: images,
                overhead_encoder=lambda images: images,
                image_transform=lambda image: image,
                image_loader=lambda _: torch.zeros(3, 1, 1),
                device="cpu",
            )
            with self.assertRaisesRegex(RuntimeError, "fingerprint mismatch"):
                FP32FeatureCache(root / "cache", expected_fingerprint="wrong")

    def test_same_fingerprint_corrupt_array_is_rebuilt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "a.png"
            path.write_bytes(b"a")
            build_args = dict(
                cache_dir=root / "cache",
                fingerprint="stable",
                ground_paths=[path],
                overhead_paths=[path],
                ground_encoder=lambda images: images[:, :1],
                overhead_encoder=lambda images: images[:, :2],
                image_transform=lambda image: image,
                image_loader=lambda _: torch.zeros(3, 2, 2),
                device="cpu",
            )
            FP32FeatureCache.build(**build_args)
            np.save(
                root / "cache" / FP32FeatureCache.GROUND_ARRAY,
                np.zeros((1, 1, 1, 1), dtype=np.float32),
            )
            with self.assertRaisesRegex(RuntimeError, "shape"):
                FP32FeatureCache(root / "cache", expected_fingerprint="stable")

            rebuilt = FP32FeatureCache.build(**build_args)
            self.assertEqual(rebuilt.ground.shape, (1, 1, 2, 2))

    def test_same_seed_source_selection_matches_first_entry_and_owning_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            roots = [base / "run2", base / "run3"]
            scene = roots[0] / "scene_0001"
            scene.mkdir(parents=True)
            (roots[1] / "ugv_rgb").mkdir(parents=True)
            (roots[0] / "ugv_rgb").mkdir(parents=True)
            (roots[0] / "uav_rgb").mkdir(parents=True)
            (roots[0] / "uav_rgb" / "tile.png").write_bytes(b"uav")

            views = []
            first = roots[0] / "ugv_rgb" / "first.png"
            duplicate = roots[0] / "ugv_rgb" / "duplicate.png"
            first.write_bytes(b"first")
            duplicate.write_bytes(b"duplicate")
            views.extend(
                [
                    {"camera_idx": 1, "image_path": "ugv_rgb/first.png"},
                    {"camera_idx": 1, "image_path": "ugv_rgb/duplicate.png"},
                ]
            )
            for frame_id in range(2, 9):
                source = roots[0] / "ugv_rgb" / f"frame_{frame_id}.png"
                source.write_bytes(bytes([frame_id]))
                views.append(
                    {
                        "camera_idx": frame_id,
                        "image_path": f"ugv_rgb/frame_{frame_id}.png",
                    }
                )
            # A colliding relative path in run3 must never be selected for a run2 scene.
            (roots[1] / "ugv_rgb" / "first.png").write_bytes(b"wrong root")
            (scene / "metadata.json").write_text(
                json.dumps(
                    {
                        "uav_image_path": "uav_rgb/tile.png",
                        "ugv_images": views,
                    }
                ),
                encoding="utf-8",
            )

            sources = selected_scene_sources(
                _SourceDataset(scene, roots), dataset_index=0, seed=42
            )
            repeated = selected_scene_sources(
                _SourceDataset(scene, roots), dataset_index=0, seed=42
            )
            self.assertEqual(repeated, sources)
            self.assertEqual(sources["ground_paths"][0], str(first.resolve()))
            self.assertNotIn(str(duplicate.resolve()), sources["ground_paths"])
            self.assertTrue(
                all(Path(path).is_relative_to(roots[0]) for path in sources["ground_paths"])
            )


if __name__ == "__main__":
    unittest.main()
