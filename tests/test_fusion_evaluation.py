from __future__ import annotations

import json
import hashlib
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from src.evaluation.evaluation_pipeline import (
    _manifest_entries,
    deterministic_subset,
    filter_dataset_by_manifest,
    load_model_strict,
    resolve_eval_config,
)
from src.evaluation.common import transform_camera_rig, yaw_from_w2c_tensor
from src.evaluation.fusion_ablation_report import aggregate_variants, build_report


class _Ground(nn.Module):
    def __init__(self, variant: str):
        super().__init__()
        self.backbone = nn.Linear(2, 2)
        self.feature_extractor = nn.Module()
        self.feature_extractor.convnext = nn.Module()
        self.feature_extractor.convnext.stages_2 = nn.Linear(2, 2)
        self.feature_norm = nn.LayerNorm(2)
        self.token_mlp = nn.Linear(2, 2)
        if variant != "N1":
            self.column_mlp = nn.Linear(2 if variant == "N2" else 4, 2)


class _Model(nn.Module):
    def __init__(self, variant: str):
        super().__init__()
        self.ground_encoder = _Ground(variant)
        self.overhead = nn.Linear(2, 2)


def _adapter_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: torch.full_like(value, 0.75)
        for key, value in model.state_dict().items()
        if key.startswith(
            (
                "ground_encoder.feature_norm.",
                "ground_encoder.token_mlp.",
                "ground_encoder.column_mlp.",
            )
        )
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FusionEvaluationTests(unittest.TestCase):
    def test_saved_scene_manifest_filters_evaluation_dataset(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run1"
            scenes = [root / f"scene_{index:04d}" for index in range(4)]
            for scene in scenes:
                scene.mkdir(parents=True)

            manifest_path = Path(temporary) / "evaluation_manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "splits": {
                            "test": [
                                {"scene_key": "run1/scene_0001"},
                                {"path": str(scenes[3].resolve())},
                            ]
                        }
                    }
                ),
                encoding="utf-8",
            )
            entries = _manifest_entries(str(manifest_path), "test")
            dataset = type(
                "DatasetStub",
                (),
                {"scene_folders": [str(scene) for scene in scenes]},
            )()
            filter_dataset_by_manifest(dataset, entries)

            self.assertEqual(
                dataset.scene_folders,
                [str(scenes[1]), str(scenes[3])],
            )

    def test_yaw_delta_is_reported_as_absolute_camera_yaw(self):
        radians = torch.deg2rad(torch.tensor(30.0))
        c2w = torch.eye(4)
        c2w[:2, :2] = torch.tensor(
            [
                [torch.cos(radians), -torch.sin(radians)],
                [torch.sin(radians), torch.cos(radians)],
            ]
        )
        base_yaw = yaw_from_w2c_tensor(torch.linalg.inv(c2w))
        predicted_absolute = torch.remainder(base_yaw + 45.0 + 180.0, 360.0) - 180.0
        torch.testing.assert_close(base_yaw, torch.tensor(30.0))
        torch.testing.assert_close(predicted_absolute, torch.tensor(75.0))

    def test_pose_hypothesis_moves_all_views_as_one_rigid_rig(self):
        c2w = torch.eye(4).repeat(3, 1, 1)
        c2w[1, 0, 3] = 1.0
        c2w[2, 1, 3] = 2.0
        transformed_w2c = transform_camera_rig(
            torch.linalg.inv(c2w), dx=3.0, dy=-2.0, yaw_degrees=90.0
        )
        transformed = torch.linalg.inv(transformed_w2c)

        expected_positions = torch.tensor(
            [[3.0, -2.0, 0.0], [3.0, -1.0, 0.0], [1.0, -2.0, 0.0]]
        )
        torch.testing.assert_close(transformed[:, :3, 3], expected_positions)
        relative_before = torch.cdist(c2w[:, :3, 3], c2w[:, :3, 3])
        relative_after = torch.cdist(
            transformed[:, :3, 3], transformed[:, :3, 3]
        )
        torch.testing.assert_close(relative_after, relative_before)

    def test_adapter_over_n0_base_loads_n1_and_n2_strictly(self):
        with tempfile.TemporaryDirectory() as temporary:
            base_path = Path(temporary) / "base.pth"
            torch.save(_Model("N0").state_dict(), base_path)
            for variant in ("N1", "N2"):
                model = _Model(variant)
                payload = {
                    "adapter_state_dict": _adapter_state(model),
                    "base_checkpoint_path": str(base_path),
                    "base_checkpoint_sha256": _sha256(base_path),
                }
                adapter_path = Path(temporary) / f"{variant}.pth"
                torch.save(payload, adapter_path)
                kind = load_model_strict(model, adapter_path, payload)
                self.assertTrue(kind.startswith("base+adapter:"))
                self.assertTrue(torch.allclose(model.ground_encoder.token_mlp.weight, torch.full_like(model.ground_encoder.token_mlp.weight, 0.75)))

    def test_adapter_cannot_fall_back_to_base_fusion_weights(self):
        with tempfile.TemporaryDirectory() as temporary:
            base_path = Path(temporary) / "base.pth"
            torch.save(_Model("N0").state_dict(), base_path)
            model = _Model("N1")
            payload = {
                "adapter_state_dict": {},
                "base_checkpoint_path": str(base_path),
                "base_checkpoint_sha256": _sha256(base_path),
            }
            with self.assertRaisesRegex((RuntimeError, TypeError), "fusion|state dict"):
                load_model_strict(model, Path(temporary) / "adapter.pth", payload)

    def test_adapter_rejects_wrong_base_checkpoint_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            base_path = Path(temporary) / "base.pth"
            torch.save(_Model("N0").state_dict(), base_path)
            model = _Model("N1")
            payload = {
                "adapter_state_dict": _adapter_state(model),
                "base_checkpoint_path": str(base_path),
                "base_checkpoint_sha256": "0" * 64,
            }
            with self.assertRaisesRegex(RuntimeError, "SHA-256 mismatch"):
                load_model_strict(model, Path(temporary) / "adapter.pth", payload)

    def test_stage2_adapter_requires_exact_fusion_and_ground_stage_tensors(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base_path = root / "base.pth"
            torch.save(_Model("N0").state_dict(), base_path)
            model = _Model("N1")
            adapter = _adapter_state(model)
            payload = {
                "adapter_state_dict": adapter,
                "base_checkpoint_path": str(base_path),
                "base_checkpoint_sha256": _sha256(base_path),
                "resolved_config": {"fusion_ablation_stage": 2},
            }
            adapter_path = root / "stage2" / "N1" / "seed_42" / "best_adapter.pth"
            adapter_path.parent.mkdir(parents=True)
            with self.assertRaisesRegex(RuntimeError, "stages_2"):
                load_model_strict(model, adapter_path, payload)

            adapter.update(
                {
                    key: torch.full_like(value, 0.5)
                    for key, value in model.state_dict().items()
                    if key.startswith(
                        "ground_encoder.feature_extractor.convnext.stages_2."
                    )
                }
            )
            load_model_strict(model, adapter_path, payload)
            self.assertTrue(
                torch.allclose(
                    model.ground_encoder.feature_extractor.convnext.stages_2.weight,
                    torch.full_like(
                        model.ground_encoder.feature_extractor.convnext.stages_2.weight,
                        0.5,
                    ),
                )
            )

    def test_saved_config_is_restored_before_eval_overrides(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "best_model.pth"
            checkpoint.touch()
            (checkpoint.parent / "config.json").write_text(
                json.dumps(
                    {
                        "vit_model": "convnext_base",
                        "ground_fusion_variant": "N7",
                        "num_ugv_views": 2,
                        "train_img_size": [192, 256],
                    }
                ),
                encoding="utf-8",
            )
            config = resolve_eval_config(
                checkpoint,
                {"resolved_config": {"feature_dim": 64}},
                device="cpu",
                num_ugv_views=8,
            )
            self.assertEqual(config["vit_model"], "convnext_base")
            self.assertEqual(config["ground_fusion_variant"], "N7")
            self.assertEqual(config["feature_dim"], 64)
            self.assertEqual(config["num_ugv_views"], 8)
            self.assertEqual(config["train_img_size"], (192, 256))

    def test_localization_subset_is_stable_and_order_independent(self):
        keys = [f"run1/scene_{index:04d}" for index in range(30)]
        first = deterministic_subset(keys, 10, 42)
        second = deterministic_subset(list(reversed(keys)), 10, 42)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 10)

    def test_report_aggregates_completed_runs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for seed, loss in ((7, 0.4), (42, 0.2)):
                run = root / "stage2" / "N4" / f"seed_{seed}"
                run.mkdir(parents=True)
                (run / "status.json").write_text(
                    json.dumps({"status": "complete", "variant": "N4", "seed": seed, "stage": "stage2"}),
                    encoding="utf-8",
                )
                (run / "result.json").write_text(
                    json.dumps({"best_metrics": {"val_loss": loss, "recall@1": 0.5, "recall@5": 0.8, "mrr": 0.6}}),
                    encoding="utf-8",
                )
            records, variants = build_report(root)
            self.assertEqual(len(records), 2)
            self.assertEqual(len(variants), 1)
            self.assertAlmostEqual(variants[0]["best_val_loss_mean"], 0.3)
            self.assertTrue((root / "summary.md").is_file())
            self.assertTrue((root / "per_run.csv").is_file())

    def test_report_does_not_mix_screen_and_confirmation_stages(self):
        records = [
            {"stage": "stage1", "variant": "N0", "status": "complete", "best_val_loss": 0.8},
            {"stage": "stage2", "variant": "N0", "status": "complete", "best_val_loss": 0.2},
        ]
        variants = aggregate_variants(records)
        self.assertEqual(len(variants), 2)
        self.assertEqual({row["stage"] for row in variants}, {"stage1", "stage2"})

    def test_heldout_aggregate_reports_successful_seed_count(self):
        records = [
            {
                "stage": "stage2",
                "variant": "N4",
                "seed": seed,
                "status": "complete",
                "best_val_loss": 0.2,
                "recall@1": 0.5,
                "mrr": 0.6,
                "heldout_recall@1": 0.4 if seed != 123 else None,
            }
            for seed in (42, 7, 123)
        ]
        row = aggregate_variants(records)[0]
        self.assertEqual(row["num_runs"], 3)
        self.assertEqual(row["heldout_num_runs"], 2)

    def test_report_includes_canonical_full_heldout_result(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "heldout_status.json").write_text(
                json.dumps(
                    {
                        "evaluations": [
                            {
                                "state": "complete",
                                "status": "complete",
                            "canonical_full": True,
                            "variant": "N4",
                            "seed": 7,
                            "runtime_seconds": 7200.0,
                            "summary": {
                                    "retrieval": {
                                        "recall@1": 0.4,
                                        "recall@5": 0.8,
                                        "mrr": 0.55,
                                    },
                                    "localization": {
                                        "nearest_topk_distance_mean_m": 1.2,
                                        "nearest_topk_distance_median_m": 1.0,
                                        "nearest_topk_within_1.0m_ratio": 0.4,
                                        "nearest_topk_within_2.0m_ratio": 0.7,
                                        "nearest_topk_within_3.0m_ratio": 0.9,
                                    },
                                },
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            records, variants = build_report(root)
            self.assertEqual(records[0]["stage"], "heldout_full")
            self.assertAlmostEqual(records[0]["heldout_recall@1"], 0.4)
            self.assertAlmostEqual(records[0]["localization_success_2m"], 0.7)
            self.assertAlmostEqual(records[0]["heldout_runtime_seconds"], 7200.0)
            self.assertEqual(variants[0]["stage"], "heldout_full")
            self.assertIn(
                "N4 (canonical full)",
                (root / "summary.md").read_text(encoding="utf-8"),
            )

    def test_report_preserves_campaign_selection_failures_and_heldout_metrics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "campaign_config.json").write_text(
                json.dumps({"base_checkpoint": "base.pth", "split_seed": 42}),
                encoding="utf-8",
            )
            (root / "stage1_selection.json").write_text(
                json.dumps({"finalists": ["N4", "N0", "N7"]}),
                encoding="utf-8",
            )
            (root / "winner_selection.json").write_text(
                json.dumps({"winner": "N4"}),
                encoding="utf-8",
            )

            for variant, loss in (("N0", 0.5), ("N4", 0.3)):
                run = root / "stage1" / variant / "seed_42"
                (run / "evaluation").mkdir(parents=True)
                (run / "status.json").write_text(
                    json.dumps(
                        {
                            "status": "complete",
                            "variant": variant,
                            "seed": 42,
                            "stage": "stage1",
                        }
                    ),
                    encoding="utf-8",
                )
                (run / "result.json").write_text(
                    json.dumps(
                        {
                            "best_metrics": {
                                "val_loss": loss,
                                "recall@1": 0.6 if variant == "N4" else 0.4,
                                "recall@5": 0.9,
                                "mrr": 0.7 if variant == "N4" else 0.5,
                            }
                        }
                    ),
                    encoding="utf-8",
                )
                (run / "evaluation" / "evaluation_summary.json").write_text(
                    json.dumps(
                        {
                            "retrieval": {"recall@1": 0.55, "recall@5": 0.85, "mrr": 0.65},
                            "localization": {
                                "nearest_topk_distance_mean_m": 1.25,
                                "nearest_topk_distance_median_m": 1.0,
                                "nearest_topk_within_1.0m_ratio": 0.3,
                                "nearest_topk_within_2.0m_ratio": 0.6,
                                "nearest_topk_within_3.0m_ratio": 0.8,
                            },
                        }
                    ),
                    encoding="utf-8",
                )
                (run / "evaluation" / "evaluation_status.json").write_text(
                    json.dumps({"status": "complete", "runtime_seconds": 3600.0}),
                    encoding="utf-8",
                )

            failed = root / "stage1" / "N9" / "seed_42"
            failed.mkdir(parents=True)
            (failed / "status.json").write_text(
                json.dumps(
                    {
                        "status": "failed",
                        "variant": "N9",
                        "seed": 42,
                        "stage": "stage1",
                        "error": "deliberate failure",
                    }
                ),
                encoding="utf-8",
            )

            _, variants = build_report(root)
            n4 = next(row for row in variants if row["variant"] == "N4")
            self.assertAlmostEqual(n4["delta_best_val_loss_mean_vs_n0"], -0.2)
            self.assertAlmostEqual(n4["heldout_recall@1_mean"], 0.55)
            self.assertAlmostEqual(n4["localization_success_1m_mean"], 0.3)
            self.assertAlmostEqual(n4["heldout_runtime_seconds_total"], 3600.0)
            self.assertEqual(n4["heldout_num_runs"], 1)
            summary = (root / "summary.md").read_text(encoding="utf-8")
            self.assertIn("Validation-selected winner: **N4**", summary)
            self.assertIn("Stage-1 finalists", summary)
            self.assertIn("deliberate failure", summary)
            self.assertIn("Held-out run1 confirmation", summary)
            self.assertIn("Top-5 ≤1m", summary)
            self.assertIn("Δ R@1 vs N0", summary)
            per_run_header = (root / "per_run.csv").read_text(
                encoding="utf-8"
            ).splitlines()[0]
            self.assertIn("localization_success_1m", per_run_header)
            self.assertIn("heldout_runtime_seconds", per_run_header)


if __name__ == "__main__":
    unittest.main()
