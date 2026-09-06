from __future__ import annotations

import json
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from src.training.fusion_campaign_helpers import (
    build_heldout_evaluation_specs,
    evaluation_command,
    load_or_create_heldout_manifest,
    run_heldout_evaluations,
    run_preflight,
    validate_dataset_root,
    validate_feature_cache,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


class FusionCampaignPreflightTests(unittest.TestCase):
    def test_dataset_validation_uses_the_scene_owning_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run2"
            scene = root / "scene_0001"
            (root / "ugv_rgb").mkdir(parents=True)
            (root / "ugv_depth").mkdir()
            (scene / "uav_image").mkdir(parents=True)
            (scene / "uav_image" / "tile.png").write_bytes(b"uav")
            (root / "ugv_rgb" / "frame.png").write_bytes(b"rgb")
            (root / "ugv_depth" / "frame.png").write_bytes(b"depth")
            _write_json(
                scene / "metadata.json",
                {
                    "uav_image_path": "uav_image/tile.png",
                    "ugv_images": [
                        {
                            "image_path": "ugv_rgb/frame.png",
                            "depth_path": "ugv_depth/frame.png",
                        }
                    ],
                },
            )
            details = validate_dataset_root(root, expected_scene_count=1)
            self.assertEqual(details["reference_count"], 3)

            (root / "ugv_depth" / "frame.png").unlink()
            with self.assertRaisesRegex(RuntimeError, "depth_path"):
                validate_dataset_root(root)

    def test_cache_validation_marks_absent_or_stale_cache_for_rebuild(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            absent = validate_feature_cache(root / "missing")
            self.assertTrue(absent["rebuild_required"])

            cache = root / "cache"
            cache.mkdir()
            source = root / "source.png"
            source.write_bytes(b"pixels")
            np.save(cache / "ground_features.npy", np.zeros((1, 2, 3, 4), dtype=np.float32))
            np.save(cache / "overhead_features.npy", np.zeros((1, 2, 3, 4), dtype=np.float32))
            _write_json(
                cache / "manifest.json",
                {
                    "version": 1,
                    "fingerprint": "right",
                    "dtype": "float32",
                    "ground_paths": [str(source)],
                    "overhead_paths": [str(source)],
                },
            )
            self.assertTrue(validate_feature_cache(cache, expected_fingerprint="right")["valid"])
            stale = validate_feature_cache(cache, expected_fingerprint="wrong")
            self.assertFalse(stale["valid"])
            self.assertIn("fingerprint", stale["reason"])

    def test_preflight_collects_cache_rebuild_without_failing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "base.pth"
            checkpoint.write_bytes(b"base")
            with (
                mock.patch(
                    "src.training.fusion_campaign_helpers.validate_imports",
                    return_value={"torch": "test"},
                ),
                mock.patch(
                    "src.training.fusion_campaign_helpers.validate_device",
                    return_value={"requested": "cpu"},
                ),
                mock.patch(
                    "src.training.fusion_campaign_helpers.validate_disk_space",
                    return_value={"free_gib": 100},
                ),
                mock.patch(
                    "src.training.fusion_campaign_helpers.validate_base_checkpoint",
                    return_value={"checkpoint": str(checkpoint)},
                ),
                mock.patch(
                    "src.training.fusion_campaign_helpers.validate_dataset_root",
                    return_value={"scene_count": 1},
                ),
            ):
                report = run_preflight(
                    run_dir=root / "run",
                    base_checkpoint=checkpoint,
                    data_roots=[root / "run2", root / "run3"],
                    heldout_root=root / "run1",
                    cache_dir=root / "absent_cache",
                    device="cpu",
                    run_unit_tests=False,
                    dry_run=True,
                )
            self.assertTrue(report["ok"])
            self.assertTrue(report["cache_rebuild_required"])
            self.assertTrue((root / "run" / "preflight.json").is_file())


class FusionCampaignHeldoutTests(unittest.TestCase):
    def _campaign_tree(self, root: Path) -> tuple[Path, Path]:
        base = root / "base.pth"
        base.write_bytes(b"base")
        _write_json(root / "stage1_selection.json", {"finalists": ["N2", "N4", "N7"]})
        _write_json(
            root / "stage2_selection.json",
            {
                "winner_variant": "N4",
                "canonical_run": {"variant": "N4", "seed": 7},
            },
        )
        for variant in ("N2", "N4", "N7"):
            for seed in (42, 7, 123):
                checkpoint = root / "stage2" / variant / f"seed_{seed}" / "best_adapter.pth"
                checkpoint.parent.mkdir(parents=True)
                checkpoint.write_bytes(b"adapter")
        winner = root / "winner" / "best_model.pth"
        winner.parent.mkdir(parents=True)
        winner.write_bytes(b"winner")
        heldout = root / "run1"
        for index in range(553):
            _write_json(
                heldout / f"scene_{index:04d}" / "metadata.json",
                {},
            )
        return base, winner

    def test_specs_cover_nine_seeds_and_canonical_full(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base, winner = self._campaign_tree(root)
            specs = build_heldout_evaluation_specs(
                run_dir=root,
                base_checkpoint=base,
                heldout_root=root / "run1",
            )
            self.assertEqual(len(specs), 10)
            self.assertEqual(sum(spec.canonical_full for spec in specs), 1)
            self.assertEqual(specs[-1].checkpoint, winner)
            self.assertEqual(specs[-1].expected_localization_scenes, 553)
            command = evaluation_command(specs[0], heldout_root=root / "run1", device="cpu")
            self.assertEqual(command[command.index("--num-ugv-views") + 1], "8")
            self.assertEqual(command[command.index("--localization-samples") + 1], "100")
            manifest = Path(command[command.index("--scene-manifest") + 1])
            self.assertEqual(manifest, root / "heldout_run1_manifest.json")
            self.assertTrue(manifest.is_file())
            self.assertIn("--yaw-search-angles-deg=-90,-45,0,45,90,135,180", command)

    def test_heldout_manifest_rejects_count_preserving_scene_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._campaign_tree(root)
            manifest, fingerprint = load_or_create_heldout_manifest(
                run_dir=root,
                heldout_root=root / "run1",
            )
            self.assertTrue(manifest.is_file())
            self.assertEqual(len(fingerprint), 64)
            (root / "run1" / "scene_0000").rename(
                root / "run1" / "scene_replacement"
            )
            with self.assertRaisesRegex(RuntimeError, "scene identities changed"):
                load_or_create_heldout_manifest(
                    run_dir=root,
                    heldout_root=root / "run1",
                )

    def test_evaluation_retries_once_then_resumes_completed_summaries(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base, _ = self._campaign_tree(root)
            calls: list[str] = []

            def runner(command, _cwd, _log_path):
                output = Path(command[command.index("--output-dir") + 1])
                samples = int(command[command.index("--localization-samples") + 1])
                checkpoint = Path(command[command.index("--checkpoint") + 1]).resolve()
                base_checkpoint = Path(
                    command[command.index("--base-checkpoint") + 1]
                ).resolve()
                heldout_root = Path(
                    command[command.index("--data-root") + 1]
                ).resolve()
                scene_manifest = Path(
                    command[command.index("--scene-manifest") + 1]
                ).resolve()
                name = output.parent.name + "/" + output.name
                calls.append(name)
                if len(calls) == 1:
                    return 1
                is_adapter = checkpoint.name.endswith("adapter.pth")
                _write_json(
                    output / "evaluation_context.json",
                    {
                        "checkpoint": str(checkpoint),
                        "checkpoint_sha256": hashlib.sha256(
                            checkpoint.read_bytes()
                        ).hexdigest(),
                        "base_checkpoint": str(base_checkpoint) if is_adapter else None,
                        "base_checkpoint_sha256": hashlib.sha256(
                            base_checkpoint.read_bytes()
                        ).hexdigest() if is_adapter else None,
                        "data_roots": [str(heldout_root)],
                        "scene_manifest": str(scene_manifest),
                        "scene_manifest_sha256": hashlib.sha256(
                            scene_manifest.read_bytes()
                        ).hexdigest(),
                        "manifest_split": "test",
                        "num_ugv_views": 8,
                        "seed": int(command[command.index("--seed") + 1]),
                        "subset_seed": 42,
                        "localization_samples": samples,
                        "top_k": 5,
                        "softmax_temp": 0.07,
                        "grid_range_m": 5.0,
                        "grid_resolution_m": 0.25,
                        "yaw_search_angles_deg": "-90,-45,0,45,90,135,180",
                        "localization_thresholds_m": [1.0, 2.0, 3.0],
                    },
                )
                _write_json(
                    output / "evaluation_summary.json",
                    {
                        "num_retrieval_scenes": 553,
                        "num_localization_scenes": 553 if samples == 0 else 100,
                        "retrieval": {"recall@1": 0.1, "recall@5": 0.5, "mrr": 0.2},
                        "localization": {"nearest_topk_distance_mean_m": 1.0},
                    },
                )
                return 0

            report_progress: list[int] = []

            def report_builder(path):
                status_path = path / "heldout_status.json"
                self.assertTrue(status_path.is_file())
                report_progress.append(
                    len(json.loads(status_path.read_text())["evaluations"])
                )

            status = run_heldout_evaluations(
                run_dir=root,
                base_checkpoint=base,
                heldout_root=root / "run1",
                device="cpu",
                command_runner=runner,
                report_builder=report_builder,
            )
            self.assertEqual(status["state"], "complete")
            self.assertEqual(status["completed"], 10)
            self.assertEqual(len(calls), 11)
            self.assertEqual(report_progress[:10], list(range(1, 11)))
            self.assertTrue(
                (root / "stage2" / "N2" / "seed_42" / "evaluation" / "evaluation_status.json").is_file()
            )
            self.assertFalse(
                (root / "stage2" / "N2" / "seed_42" / "evaluation" / "status.json").exists()
            )

            calls.clear()
            resumed = run_heldout_evaluations(
                run_dir=root,
                base_checkpoint=base,
                heldout_root=root / "run1",
                device="cpu",
                command_runner=runner,
                report_builder=lambda _path: None,
            )
            self.assertEqual(resumed["completed"], 10)
            self.assertEqual(calls, [])

            changed_adapter = (
                root / "stage2" / "N2" / "seed_42" / "best_adapter.pth"
            )
            preserved_record = (
                root
                / "stage2"
                / "N2"
                / "seed_42"
                / "evaluation"
                / "per_scene"
                / "preserved.json"
            )
            _write_json(preserved_record, {"preserved": True})
            changed_adapter.write_bytes(b"changed adapter")
            calls.clear()
            refreshed = run_heldout_evaluations(
                run_dir=root,
                base_checkpoint=base,
                heldout_root=root / "run1",
                device="cpu",
                command_runner=runner,
                report_builder=lambda _path: None,
            )
            self.assertEqual(refreshed["completed"], 10)
            self.assertEqual(len(calls), 2)
            archived = list(
                (
                    root
                    / "stage2"
                    / "N2"
                    / "seed_42"
                    / "evaluation_stale"
                ).glob("*/per_scene/preserved.json")
            )
            self.assertEqual(len(archived), 1)


if __name__ == "__main__":
    unittest.main()
