from __future__ import annotations

import copy
import json
import os
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from unittest import mock

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation.common import compute_retrieval_metrics
from src.training.fusion_ablation import (
    STAGE2_SEEDS,
    _adapter_payload,
    _load_adapter,
    build_loaders,
    configure_trainable_parameters,
    build_datasets,
    create_split_manifest,
    eligible_scene_paths,
    forward_loss,
    fusion_parameter_names,
    keep_frozen_modules_in_eval,
    rank_results,
    record_smoke_failure_probe,
    require_finite_metrics,
    select_stage,
    seed_everything,
    sha256_file,
)
from src.training.fusion_cache import FP32FeatureCache


class _FusionGround(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.feature_norm = nn.LayerNorm(2)
        self.token_mlp = nn.Linear(2, 2)
        self.column_mlp = nn.Linear(4, 2)
        self.column_residual_scale = nn.Parameter(torch.tensor(0.1))
        self.feature_extractor = nn.Module()
        self.feature_extractor.convnext = nn.Module()
        self.feature_extractor.convnext.stages_2 = nn.Sequential(nn.Linear(2, 2))
        self.feature_extractor.convnext.stages_3 = nn.Sequential(nn.Linear(2, 2))

    def fusion_named_parameters(self):
        for module_name in ("feature_norm", "token_mlp", "column_mlp"):
            for name, parameter in getattr(self, module_name).named_parameters():
                yield f"{module_name}.{name}", parameter
        yield "column_residual_scale", self.column_residual_scale


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ground_encoder = _FusionGround()
        self.overhead_encoder = nn.Sequential(nn.Linear(2, 2))
        self.temperature = nn.Parameter(torch.tensor(0.07))


class _CachedGround(nn.Module):
    def forward(self, **kwargs):
        raise AssertionError("live ground encoder must not run on the cached path")

    def project(self, *, encoded_ground, ugv_data):
        features = encoded_ground["features_2d"].mean(dim=1)
        validity = torch.ones(
            features.shape[0], 1, *features.shape[-2:], dtype=torch.bool
        )
        return features, validity


class _CachedForwardModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ground_encoder = _CachedGround()
        self.overhead_encoder = nn.Module()
        self.overhead_encoder.forward = lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("live overhead encoder must not run when a cache is present")
        )
        self.temperature = nn.Parameter(torch.tensor(0.07), requires_grad=False)


class _CacheStub:
    def __init__(self) -> None:
        self.batch_paths = None

    def encoded_ground(self, batch_paths, image_size, device):
        self.batch_paths = batch_paths
        values = torch.tensor(
            [
                [[[[1.0]], [[0.0]]], [[[0.0]], [[1.0]]]],
                [[[[0.5]], [[0.5]]], [[[1.0]], [[0.0]]]],
            ],
            device=device,
        )
        return {"features_2d": values, "image_size": image_size}

    def overhead_tensor(self, paths, device):
        return torch.tensor(
            [[[[0.5]], [[0.5]]], [[[0.75]], [[0.25]]]], device=device
        )


class _ParityGround(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.feature_extractor = nn.Conv2d(3, 2, kernel_size=1, bias=False)
        with torch.no_grad():
            self.feature_extractor.weight.copy_(
                torch.tensor([[[[1.0]], [[0.5]], [[-0.25]]], [[[0.0]], [[1.0]], [[0.5]]]])
            )

    def encode(self, images):
        batch, views, channels, height, width = images.shape
        features = self.feature_extractor(images.reshape(batch * views, channels, height, width))
        return {
            "features_2d": features.view(batch, views, *features.shape[1:]),
            "image_size": (height, width),
        }

    def project(self, *, encoded_ground=None, ugv_data=None, **kwargs):
        data = ugv_data if ugv_data is not None else kwargs
        encoded = encoded_ground or self.encode(data["ugv_images"])
        features = encoded["features_2d"].mean(dim=1)
        validity = torch.ones(
            features.shape[0], 1, *features.shape[-2:], dtype=torch.bool, device=features.device
        )
        return features, validity

    def forward(self, **kwargs):
        return self.project(**kwargs)


class _ParityOverhead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.feature_extractor = nn.Conv2d(3, 2, kernel_size=1, bias=False)
        self.projection = nn.Conv2d(2, 2, kernel_size=1, bias=False)
        with torch.no_grad():
            self.feature_extractor.weight.copy_(
                torch.tensor([[[[0.5]], [[1.0]], [[0.0]]], [[[1.0]], [[0.0]], [[0.5]]]])
            )
            self.projection.weight.copy_(
                torch.tensor([[[[1.0]], [[0.25]]], [[[0.5]], [[1.0]]]])
            )

    def forward(self, uav_image):
        return self.projection(self.feature_extractor(uav_image))


class _ParityModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ground_encoder = _ParityGround()
        self.overhead_encoder = _ParityOverhead()
        self.temperature = nn.Parameter(torch.tensor(0.07), requires_grad=False)


def _result(variant: str, seed: int, loss: float, recall: float, mrr: float) -> dict:
    return {
        "state": "complete",
        "status": "complete",
        "stage": "stage2",
        "variant": variant,
        "seed": seed,
        "best_val_loss": loss,
        "recall@1": recall,
        "recall@5": min(1.0, recall + 0.2),
        "mrr": mrr,
    }


class FusionAblationUtilityTests(unittest.TestCase):
    @staticmethod
    def _toy_epoch(model, optimizer, generator):
        indices = torch.arange(8)
        features = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
                [-1.0, 1.0],
                [0.5, -0.5],
                [-0.5, 0.25],
                [0.25, 0.75],
                [-0.75, -0.25],
            ]
        )
        targets = torch.stack((features[:, 0] - features[:, 1], features.sum(1)), dim=1)
        loader = DataLoader(
            TensorDataset(indices, features, targets),
            batch_size=2,
            shuffle=True,
            generator=generator,
            num_workers=0,
        )
        trace = []
        for sample_ids, batch_features, batch_targets in loader:
            python_draw = random.random()
            numpy_draw = float(np.random.random())
            torch_draw = torch.rand(())
            scale = 1.0 + 1e-3 * (python_draw + numpy_draw + torch_draw)
            optimizer.zero_grad(set_to_none=True)
            loss = (model(batch_features * scale) - batch_targets).square().mean()
            loss.backward()
            optimizer.step()
            trace.append(
                (
                    sample_ids.tolist(),
                    python_draw,
                    numpy_draw,
                    float(torch_draw),
                    float(loss.detach()),
                )
            )
        return trace

    def test_epoch_boundary_resume_matches_uninterrupted_execution(self):
        previous_deterministic = torch.are_deterministic_algorithms_enabled()
        try:
            seed_everything(42)
            model = nn.Linear(2, 2)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
            loader_generator = torch.Generator().manual_seed(42)
            self._toy_epoch(model, optimizer, loader_generator)

            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                base_checkpoint = root / "base.pth"
                base_checkpoint.write_bytes(b"toy base checkpoint")
                adapter_path = root / "epoch_1_adapter.pth"
                payload = _adapter_payload(
                    model,
                    optimizer,
                    {"ground_fusion_variant": "N0"},
                    base_checkpoint,
                    epoch=0,
                    history=[{"epoch": 1, "val_loss": 0.5}],
                    best={"val_loss": 0.5},
                    no_improvement=0,
                    base_checkpoint_sha256=sha256_file(base_checkpoint),
                    dataloader_generator=loader_generator,
                )
                torch.save(payload, adapter_path)

                uninterrupted_trace = self._toy_epoch(
                    model, optimizer, loader_generator
                )
                uninterrupted_model = {
                    name: tensor.detach().clone()
                    for name, tensor in model.state_dict().items()
                }
                uninterrupted_optimizer = copy.deepcopy(optimizer.state_dict())
                uninterrupted_generator_state = loader_generator.get_state().clone()

                # A fresh process would initialize unrelated state before loading.
                # The adapter must replace all of it, including every RNG stream.
                seed_everything(999)
                resumed_model = nn.Linear(2, 2)
                resumed_optimizer = torch.optim.Adam(
                    resumed_model.parameters(), lr=1e-2
                )
                resumed_generator = torch.Generator().manual_seed(999)
                restored = _load_adapter(
                    resumed_model, resumed_optimizer, adapter_path
                )
                resumed_generator.set_state(
                    restored["dataloader_generator_state"]
                )
                resumed_trace = self._toy_epoch(
                    resumed_model, resumed_optimizer, resumed_generator
                )

                self.assertEqual(resumed_trace, uninterrupted_trace)
                for name, expected in uninterrupted_model.items():
                    torch.testing.assert_close(
                        resumed_model.state_dict()[name], expected, rtol=0, atol=0
                    )
                self.assertEqual(
                    resumed_optimizer.state_dict()["param_groups"],
                    uninterrupted_optimizer["param_groups"],
                )
                for parameter_id, expected_state in uninterrupted_optimizer[
                    "state"
                ].items():
                    actual_state = resumed_optimizer.state_dict()["state"][
                        parameter_id
                    ]
                    self.assertEqual(set(actual_state), set(expected_state))
                    for key, expected in expected_state.items():
                        if torch.is_tensor(expected):
                            torch.testing.assert_close(
                                actual_state[key], expected, rtol=0, atol=0
                            )
                        else:
                            self.assertEqual(actual_state[key], expected)
                torch.testing.assert_close(
                    resumed_generator.get_state(),
                    uninterrupted_generator_state,
                    rtol=0,
                    atol=0,
                )
        finally:
            torch.use_deterministic_algorithms(previous_deterministic)

    def test_same_seed_reproduces_fusion_init_batch_order_and_metrics(self):
        dataset = TensorDataset(
            torch.arange(8),
            torch.tensor(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 1.0],
                    [-1.0, 1.0],
                    [0.5, -0.5],
                    [-0.5, 0.25],
                    [0.25, 0.75],
                    [-0.75, -0.25],
                ]
            ),
        )
        config = {
            "batch_size": 2,
            "num_workers": 0,
            "pin_memory": False,
        }

        def run(seed):
            seed_everything(seed)
            model = _TinyModel()
            names = fusion_parameter_names(model)
            initial = {
                name: dict(model.named_parameters())[name].detach().clone()
                for name in names
            }
            optimizer = torch.optim.Adam(
                [dict(model.named_parameters())[name] for name in names], lr=1e-2
            )
            with mock.patch(
                "src.training.fusion_ablation.build_datasets",
                return_value=(dataset, dataset),
            ):
                train_loader, _, _ = build_loaders(config, {}, seed)

            order = []
            for sample_ids, features in train_loader:
                order.extend(sample_ids.tolist())
                optimizer.zero_grad(set_to_none=True)
                embeddings = model.ground_encoder.token_mlp(
                    model.ground_encoder.feature_norm(features)
                )
                loss = (embeddings - features).square().mean()
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                ground = model.ground_encoder.token_mlp(
                    model.ground_encoder.feature_norm(dataset.tensors[1])
                )
            metrics = compute_retrieval_metrics(
                ground, dataset.tensors[1], ks=(1, 5)
            )
            return initial, order, metrics

        previous_deterministic = torch.are_deterministic_algorithms_enabled()
        try:
            first_initial, first_order, first_metrics = run(123)
            second_initial, second_order, second_metrics = run(123)
        finally:
            torch.use_deterministic_algorithms(previous_deterministic)

        self.assertEqual(first_order, second_order)
        self.assertEqual(first_metrics, second_metrics)
        self.assertEqual(set(first_initial), set(second_initial))
        for name in first_initial:
            torch.testing.assert_close(
                first_initial[name], second_initial[name], rtol=0, atol=0
            )

    def test_non_finite_metrics_fail_the_experiment(self):
        require_finite_metrics({"loss": 0.2, "mrr": 0.8}, "test")
        with self.assertRaisesRegex(FloatingPointError, "val_loss"):
            require_finite_metrics({"val_loss": float("nan")}, "validation")

    def test_seed_enables_strict_deterministic_algorithms(self):
        previous = torch.are_deterministic_algorithms_enabled()
        try:
            seed_everything(42)
            self.assertTrue(torch.are_deterministic_algorithms_enabled())
            self.assertEqual(os.environ["CUBLAS_WORKSPACE_CONFIG"], ":4096:8")
        finally:
            torch.use_deterministic_algorithms(previous)

    def test_smoke_failure_probe_retries_and_records_expected_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            status = record_smoke_failure_probe(Path(temporary))
            self.assertTrue(status["expected_failure"])
            self.assertEqual(status["attempts"], 2)
            self.assertTrue(all(code != 0 for code in status["return_codes"]))
            self.assertTrue(
                (
                    Path(temporary)
                    / "smoke_probe"
                    / "deliberate_failure"
                    / "seed_0"
                    / "status.json"
                ).is_file()
            )

    def test_cached_forward_bypasses_backbones_and_uncollates_view_paths(self):
        model = _CachedForwardModel()
        cache = _CacheStub()
        batch = {
            "ugv_data": {"ugv_images": torch.full((2, 2, 3, 1, 1), float("nan"))},
            "uav_data": {"uav_image": torch.full((2, 3, 1, 1), float("nan"))},
            # torch's default collator transposes a per-sample list of view paths.
            "cache_ground_paths": [("a_view0", "b_view0"), ("a_view1", "b_view1")],
            "cache_overhead_path": ["a_uav", "b_uav"],
        }
        loss, pixel, global_loss, ground, overhead, validity = forward_loss(
            model,
            batch,
            {
                "device": "cpu",
                "train_img_size": (1, 1),
                "use_pixel_loss": True,
                "use_global_loss": True,
                "pixel_loss_weight": 0.5,
                "mixed_loss_delay": 0,
            },
            epoch=0,
            feature_cache=cache,
            cache_ground=True,
        )
        self.assertEqual(cache.batch_paths, [["a_view0", "a_view1"], ["b_view0", "b_view1"]])
        self.assertEqual(ground.shape, overhead.shape)
        self.assertEqual(validity.shape, (2, 1, 1, 1))
        self.assertTrue(torch.isfinite(torch.stack((loss, pixel, global_loss))).all())

    def test_real_memmap_cached_and_live_forward_are_numerically_identical(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ground_paths = [root / f"g{sample}{view}.png" for sample in range(2) for view in range(2)]
            overhead_paths = [root / f"o{sample}.png" for sample in range(2)]
            tensors = {}
            for index, path in enumerate(ground_paths + overhead_paths, start=1):
                path.write_bytes(b"test")
                tensors[str(path.resolve())] = torch.arange(12, dtype=torch.float32).view(3, 2, 2) / index

            model = _ParityModel().eval()
            cache = FP32FeatureCache.build(
                root / "cache",
                fingerprint="parity",
                ground_paths=ground_paths,
                overhead_paths=overhead_paths,
                ground_encoder=model.ground_encoder.feature_extractor,
                overhead_encoder=model.overhead_encoder,
                image_transform=lambda image: image,
                image_loader=lambda path: tensors[str(Path(path).resolve())],
                device="cpu",
                batch_size=2,
            )
            ugv_images = torch.stack(
                [torch.stack([tensors[str(ground_paths[sample * 2 + view].resolve())] for view in range(2)]) for sample in range(2)]
            )
            uav_images = torch.stack([tensors[str(path.resolve())] for path in overhead_paths])
            batch = {
                "ugv_data": {"ugv_images": ugv_images},
                "uav_data": {"uav_image": uav_images},
                "cache_ground_paths": [
                    tuple(str(ground_paths[sample * 2 + view]) for sample in range(2))
                    for view in range(2)
                ],
                "cache_overhead_path": [str(path) for path in overhead_paths],
            }
            config = {
                "device": "cpu",
                "train_img_size": (2, 2),
                "use_pixel_loss": True,
                "use_global_loss": True,
                "pixel_loss_weight": 0.5,
                "mixed_loss_delay": 0,
            }
            live = forward_loss(model, batch, config, epoch=0)
            cached = forward_loss(
                model,
                batch,
                config,
                epoch=0,
                feature_cache=cache,
                cache_ground=True,
            )
            for live_value, cached_value in zip(live, cached):
                torch.testing.assert_close(live_value, cached_value, rtol=0, atol=0)

    def test_split_is_reproducible_and_stratified_per_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            roots = [base / "run2", base / "run3"]
            for root, count in zip(roots, (10, 15)):
                for index in range(count):
                    scene = root / f"scene_{index:04d}"
                    scene.mkdir(parents=True)
                    (scene / "metadata.json").write_text("{}", encoding="utf-8")

            first = create_split_manifest(roots, base / "first.json", split_seed=42)
            second = create_split_manifest(roots, base / "second.json", split_seed=42)
            self.assertEqual(first, second)

            for root, expected_validation in zip(roots, (2, 3)):
                rows = [row for row in first["records"] if row["root"] == str(root.resolve())]
                self.assertEqual(sum(row["split"] == "validation" for row in rows), expected_validation)
                self.assertEqual(sum(row["split"] == "train" for row in rows), len(rows) - expected_validation)
                self.assertTrue(all(row["scene_id"].startswith(f"{root.name}/") for row in rows))

    def test_manifest_uses_only_scenes_accepted_by_dataset_filtering(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            roots = [base / "run2", base / "run3"]

            def write_scene(root: Path, name: str, camera_x: float) -> Path:
                scene = root / name
                scene.mkdir(parents=True)
                pose = torch.eye(4)
                # Metadata stores world-to-camera, so this produces the desired
                # positive camera-to-world X translation after inversion.
                pose[0, 3] = -camera_x
                (scene / "metadata.json").write_text(
                    json.dumps(
                        {
                            "uav_image_path": "uav.png",
                            "ugv_images": [
                                {
                                    "camera_idx": 1,
                                    "image_path": "ugv.png",
                                    "depth_path": "depth.png",
                                    "camera_pose_w2c": pose.tolist(),
                                    "camera_intrinsics": torch.eye(3).tolist(),
                                }
                            ],
                        }
                    ),
                    encoding="utf-8",
                )
                return scene.resolve()

            valid_scenes, rejected_scenes = [], []
            for root in roots:
                root.mkdir(parents=True, exist_ok=True)
                for filename in ("uav.png", "ugv.png", "depth.png"):
                    (root / filename).write_bytes(b"placeholder")
                valid_scenes.append(write_scene(root, "scene_valid", camera_x=0.0))
                rejected_scenes.append(write_scene(root, "scene_edge", camera_x=4.5))

            config = {
                "data_root": [str(root) for root in roots],
                "train_img_size": (4, 4),
                "num_ugv_views": 1,
                "use_depth": True,
                "depth_range": (0.0, 5.0),
                "ground_tile_size": 10.0,
                "grid_size": (2, 2, 2),
                "grid_resolution": 1.0,
                "edge_margin_m": 1.0,
                "consecutive_frames": True,
            }
            eligible = eligible_scene_paths(config)
            manifest = create_split_manifest(
                roots,
                base / "split_manifest.json",
                eligible_scenes=eligible,
            )
            recorded = {Path(row["scene"]).resolve() for row in manifest["records"]}
            self.assertEqual(recorded, set(valid_scenes))
            self.assertTrue(recorded.isdisjoint(rejected_scenes))
            self.assertEqual(manifest["excluded_scene_count"], 2)

            train, validation = build_datasets(config, manifest, seed=42)
            self.assertEqual(len(train) + len(validation), 2)
            with patch(
                "src.data.dataset.read_image",
                return_value=torch.zeros(3, 4, 4, dtype=torch.uint8),
            ):
                for subset in (train, validation):
                    for index in range(len(subset)):
                        self.assertIn(Path(subset[index]["scene_id"]), recorded)

    def test_stage1_ranking_is_pure_and_does_not_force_n0(self):
        rows = [
            _result("N0", 42, 0.9, 0.1, 0.2),
            _result("N1", 42, 0.2, 0.9, 0.8),
            _result("N2", 42, 0.3, 0.8, 0.7),
            _result("N3", 42, 0.4, 0.7, 0.6),
        ]
        ranking = rank_results(rows)
        self.assertEqual([row["variant"] for row in ranking[:3]], ["N1", "N2", "N3"])

    def test_metric_ties_share_ranks_then_validation_loss_breaks_tie(self):
        rows = [
            _result("N2", 42, 0.3, 0.5, 0.6),
            _result("N1", 42, 0.2, 0.5, 0.6),
        ]
        ranking = rank_results(rows)
        self.assertEqual([row["variant"] for row in ranking], ["N1", "N2"])
        self.assertEqual(ranking[0]["rank_sum"], 3)
        self.assertEqual(ranking[1]["rank_sum"], 4)

    def test_stage2_selection_requires_all_three_seeds(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "stage1_selection.json").write_text(
                json.dumps({"finalists": ["N4", "N7", "N8"]}),
                encoding="utf-8",
            )
            complete = [
                _result("N4", seed, 0.2 + index * 0.01, 0.8, 0.75)
                for index, seed in enumerate(STAGE2_SEEDS)
            ]
            incomplete = [
                _result("N7", seed, 0.1, 0.95, 0.9)
                for seed in STAGE2_SEEDS[:-1]
            ]
            for row in complete + incomplete:
                destination = root / "stage2" / row["variant"] / f"seed_{row['seed']}"
                destination.mkdir(parents=True)
                row["run_dir"] = str(destination)
                (destination / "result.json").write_text(json.dumps(row), encoding="utf-8")

            selection = select_stage(root, 2)
            self.assertEqual(selection["winner_variant"], "N4")
            self.assertEqual([row["variant"] for row in selection["ranking"]], ["N4"])
            self.assertIn(selection["canonical_run"]["seed"], STAGE2_SEEDS)

    def test_stage2_selection_ignores_stale_dropped_finalist_runs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "stage1_selection.json").write_text(
                json.dumps({"finalists": ["N0", "N1", "N2"]}),
                encoding="utf-8",
            )
            rows = []
            for variant, loss in (("N0", 0.4), ("N9", 0.01)):
                rows.extend(
                    _result(variant, seed, loss, 0.8, 0.7)
                    for seed in STAGE2_SEEDS
                )
            for row in rows:
                destination = root / "stage2" / row["variant"] / f"seed_{row['seed']}"
                destination.mkdir(parents=True)
                row["run_dir"] = str(destination)
                (destination / "result.json").write_text(
                    json.dumps(row), encoding="utf-8"
                )

            selection = select_stage(root, 2)
            self.assertEqual(selection["winner_variant"], "N0")
            self.assertNotIn(
                "N9", [row["variant"] for row in selection["ranking"]]
            )

    def test_n9_scalar_is_discovered_as_a_fusion_parameter(self):
        model = _TinyModel()
        names = set(fusion_parameter_names(model))
        self.assertIn("ground_encoder.column_residual_scale", names)
        self.assertIn("ground_encoder.feature_norm.weight", names)
        self.assertNotIn("temperature", names)

    def test_only_fusion_then_exact_ground_stage_becomes_trainable(self):
        model = _TinyModel()
        fusion_names = set(fusion_parameter_names(model))

        before = configure_trainable_parameters(model, stage=2, epoch=19)
        self.assertTrue(before["fusion"])
        self.assertFalse(before["ground_stage"])
        self.assertEqual(
            {name for name, parameter in model.named_parameters() if parameter.requires_grad},
            fusion_names,
        )

        after = configure_trainable_parameters(model, stage=2, epoch=20)
        trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
        self.assertTrue(after["ground_stage"])
        self.assertTrue(
            any(name.startswith("ground_encoder.feature_extractor.convnext.stages_2.") for name in trainable)
        )
        self.assertFalse(any("stages_3" in name for name in trainable))
        self.assertNotIn("temperature", trainable)

        keep_frozen_modules_in_eval(model, stage=2, epoch=20)
        self.assertFalse(model.overhead_encoder.training)
        self.assertFalse(model.ground_encoder.feature_extractor.training)
        self.assertTrue(model.ground_encoder.feature_extractor.convnext.stages_2.training)
        self.assertFalse(model.ground_encoder.feature_extractor.convnext.stages_3.training)

    def test_adapter_round_trip_contains_only_changed_parameters_and_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base_checkpoint = root / "base.pth"
            base_checkpoint.write_bytes(b"base checkpoint bytes")
            model = _TinyModel()
            groups = configure_trainable_parameters(model, stage=2, epoch=20)
            optimizer = torch.optim.Adam(
                [
                    {"params": groups["fusion"], "lr": 1e-4},
                    {"params": groups["ground_stage"], "lr": 1e-5},
                ]
            )
            dataloader_generator = torch.Generator().manual_seed(123)
            payload = _adapter_payload(
                model,
                optimizer,
                {"ground_fusion_variant": "N9"},
                base_checkpoint,
                epoch=20,
                history=[{"epoch": 21, "val_loss": 0.2}],
                best={"val_loss": 0.2, "recall@1": 0.8, "mrr": 0.7},
                no_improvement=0,
                base_checkpoint_sha256=sha256_file(base_checkpoint),
                dataloader_generator=dataloader_generator,
            )
            adapter_names = set(payload["adapter_state_dict"])
            self.assertIn("ground_encoder.column_residual_scale", adapter_names)
            self.assertTrue(any("stages_2" in name for name in adapter_names))
            self.assertFalse(any("stages_3" in name for name in adapter_names))
            self.assertFalse(any(name.startswith("overhead_encoder.") for name in adapter_names))
            self.assertEqual(payload["base_checkpoint_sha256"], sha256_file(base_checkpoint))
            self.assertIn("optimizer_state_dict", payload)
            self.assertIn("rng_state", payload)
            torch.testing.assert_close(
                payload["dataloader_generator_state"],
                dataloader_generator.get_state(),
            )

            adapter_path = root / "adapter.pth"
            torch.save(payload, adapter_path)
            replacement = _TinyModel()
            replacement_groups = configure_trainable_parameters(replacement, stage=2, epoch=20)
            replacement_optimizer = torch.optim.Adam(
                [
                    {"params": replacement_groups["fusion"], "lr": 1e-4},
                    {"params": replacement_groups["ground_stage"], "lr": 1e-5},
                ]
            )
            restored = _load_adapter(replacement, replacement_optimizer, adapter_path)
            self.assertEqual(restored["epoch"], 20)
            torch.testing.assert_close(
                replacement.ground_encoder.column_residual_scale,
                model.ground_encoder.column_residual_scale,
            )


if __name__ == "__main__":
    unittest.main()
