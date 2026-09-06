from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from src.data.dataset import VineyardDataset


class VineyardDatasetMultiRootTests(unittest.TestCase):
    def test_relative_sources_never_fall_through_to_another_run_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            roots = [Path(temporary) / "run2", Path(temporary) / "run3"]
            for root in roots:
                scene = root / "scene_0001"
                scene.mkdir(parents=True)
                (root / "ugv_rgb").mkdir()
                (root / "ugv_depth").mkdir()
                (scene / "uav_rgb").mkdir()
                for relative in (
                    "ugv_rgb/frame.png",
                    "ugv_depth/frame.png",
                ):
                    (root / relative).write_bytes(root.name.encode("utf-8"))
                (scene / "uav_rgb" / "tile.png").write_bytes(
                    root.name.encode("utf-8")
                )
                (scene / "metadata.json").write_text(
                    json.dumps(
                        {
                            "uav_image_path": "uav_rgb/tile.png",
                            "ugv_images": [
                                {
                                    "camera_idx": 1,
                                    "image_path": "ugv_rgb/frame.png",
                                    "depth_path": "ugv_depth/frame.png",
                                    "camera_pose_w2c": torch.eye(4).tolist(),
                                    "camera_intrinsics": torch.eye(3).tolist(),
                                }
                            ],
                        }
                    ),
                    encoding="utf-8",
                )

            dataset = VineyardDataset(
                root_dir=[str(root) for root in roots],
                config={
                    "num_ugv_views": 1,
                    "use_depth": True,
                    "depth_range": (0.0, 5.0),
                    "ground_tile_size": 10.0,
                    "grid_size": (2, 2, 2),
                    "grid_resolution": 1.0,
                    "edge_margin_m": 0.0,
                },
                consecutive_frames=True,
            )
            requested: list[Path] = []

            def fake_read_image(path, mode=None):
                requested.append(Path(path).resolve())
                return torch.zeros(3, 2, 2, dtype=torch.uint8)

            with patch("src.data.dataset.read_image", side_effect=fake_read_image):
                dataset[1]

            run3 = roots[1].resolve()
            self.assertEqual(len(requested), 3)
            self.assertTrue(
                all(path.is_relative_to(run3) for path in requested), requested
            )


if __name__ == "__main__":
    unittest.main()
