import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from src.training.fusion_ablation import (
    atomic_json,
    configure_trainable_parameters,
    create_split_manifest,
    fusion_parameter_names,
    rank_results,
    select_stage,
)


class _Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Module()
        self.feature_extractor.convnext = nn.Module()
        self.feature_extractor.convnext.stages_2 = nn.Linear(2, 2)
        self.feature_norm = nn.LayerNorm(2)
        self.token_mlp = nn.Linear(2, 2)
        self.column_mlp = nn.Linear(2, 2)


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.ground_encoder = _Encoder()
        self.overhead_encoder = nn.Linear(2, 2)
        self.temperature = nn.Parameter(torch.tensor(0.07))


class FusionAblationDriverTests(unittest.TestCase):
    def test_root_stratified_manifest_is_deterministic(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            roots = [base / "run2", base / "run3"]
            for root, count in zip(roots, (10, 5)):
                for index in range(count):
                    scene = root / f"scene_{index:03d}"
                    scene.mkdir(parents=True)
                    (scene / "metadata.json").write_text("{}")
            first = create_split_manifest(roots, base / "first.json", split_seed=42)
            second = create_split_manifest(roots, base / "second.json", split_seed=42)
            self.assertEqual(first["records"], second["records"])
            for root, expected_validation in zip(roots, (2, 1)):
                rows = [row for row in first["records"] if row["root"] == str(root.resolve())]
                self.assertEqual(sum(row["split"] == "validation" for row in rows), expected_validation)

    def test_fusion_discovery_and_stage_two_schedule(self):
        model = _Model()
        fusion = fusion_parameter_names(model)
        self.assertTrue(all(name.startswith("ground_encoder.") for name in fusion))
        stage1 = configure_trainable_parameters(model, stage=1, epoch=0)
        self.assertTrue(stage1["fusion"])
        self.assertFalse(stage1["ground_stage"])
        self.assertFalse(model.temperature.requires_grad)
        stage2 = configure_trainable_parameters(model, stage=2, epoch=20)
        self.assertTrue(stage2["ground_stage"])
        self.assertTrue(model.ground_encoder.feature_extractor.convnext.stages_2.weight.requires_grad)
        self.assertFalse(model.overhead_encoder.weight.requires_grad)

    def test_rank_sum_uses_loss_as_tie_breaker(self):
        rows = [
            {"state": "complete", "variant": "N0", "best_val_loss": 0.8, "recall@1": 0.5, "mrr": 0.6},
            {"state": "complete", "variant": "N1", "best_val_loss": 0.7, "recall@1": 0.4, "mrr": 0.7},
            {"state": "failed", "variant": "N2", "best_val_loss": 0.1, "recall@1": 1.0, "mrr": 1.0},
        ]
        ranked = rank_results(rows)
        self.assertEqual({row["variant"] for row in ranked}, {"N0", "N1"})
        self.assertTrue(all("rank_sum" in row for row in ranked))

    def test_stage_one_selection_writes_reporter_aliases(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for index in range(4):
                run = root / "stage1" / f"N{index}" / "seed_42"
                run.mkdir(parents=True)
                atomic_json(
                    run / "result.json",
                    {
                        "state": "complete",
                        "stage": "stage1",
                        "variant": f"N{index}",
                        "seed": 42,
                        "best_val_loss": 1.0 + index,
                        "recall@1": 1.0 / (index + 1),
                        "recall@5": 1.0,
                        "mrr": 1.0 / (index + 1),
                    },
                )
            selection = select_stage(root, 1)
            self.assertEqual(selection["top3"], selection["finalists"])
            self.assertEqual(len(selection["top3"]), 3)
            self.assertTrue((root / "stage1_selection.json").is_file())


if __name__ == "__main__":
    unittest.main()
