import unittest
from unittest.mock import patch

import torch
import torch.nn as nn

from src.models import snapvit


class _DummyFeatureExtractor(nn.Module):
    """Small feature extractor used to avoid constructing a timm backbone."""

    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = 1

    def forward(self, images):
        return images[:, : self.embed_dim].contiguous()


class _TakeFirst(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.output_dim = output_dim

    def forward(self, inputs):
        return inputs[..., : self.output_dim]


class _TakeLast(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.output_dim = output_dim

    def forward(self, inputs):
        return inputs[..., -self.output_dim :]


class _ZeroColumnUpdate(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.output_dim = output_dim

    def forward(self, inputs):
        # Retain a differentiable connection to the input while returning zero.
        return inputs[..., : self.output_dim] * 0.0


class _TakeFeatureChunk(nn.Module):
    def __init__(self, output_dim, chunk):
        super().__init__()
        self.output_dim = output_dim
        self.chunk = chunk

    def forward(self, inputs):
        start = self.chunk * self.output_dim
        return inputs[..., start : start + self.output_dim]


class _SumN4Statistics(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.output_dim = output_dim

    def forward(self, inputs):
        channel_statistics = inputs[..., : 3 * self.output_dim].reshape(
            *inputs.shape[:-1], 3, self.output_dim
        ).sum(dim=-2)
        return channel_statistics + inputs[..., -1:]


class _ScoreFirst(nn.Module):
    def forward(self, inputs):
        return inputs[..., :1]


class GroundEncoderFusionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def _make_encoder(
        self,
        *,
        embed_dim=4,
        feature_dim=3,
        fusion_mode="mlp",
        use_height_positional_encoding=False,
        fusion_variant=None,
    ):
        extractor = _DummyFeatureExtractor(embed_dim)
        with patch.object(snapvit, "create_feature_extractor", return_value=extractor):
            return snapvit.GroundEncoder(
                model_name="dummy",
                feature_dim=feature_dim,
                pretrained=False,
                fusion_mode=fusion_mode,
                use_height_positional_encoding=use_height_positional_encoding,
                fusion_variant=fusion_variant,
            )

    @staticmethod
    def _height_grid(z_values, batch_size=None, dtype=torch.float32):
        z_values = torch.as_tensor(z_values, dtype=dtype)
        grid = torch.zeros(1, 1, z_values.numel(), 3, dtype=dtype)
        grid[0, 0, :, 2] = z_values
        if batch_size is not None:
            grid = grid.unsqueeze(0).expand(batch_size, -1, -1, -1, -1).clone()
        return grid

    @staticmethod
    def _projection_inputs(batch_size=1, num_views=2):
        depths = torch.full(
            (batch_size, num_views, 1, 1, 1),
            1.0 / 65.535,
            dtype=torch.float32,
        )
        poses = torch.eye(4).view(1, 1, 4, 4).expand(
            batch_size, num_views, -1, -1
        ).clone()
        intrinsics = torch.eye(3).view(1, 1, 3, 3).expand(
            batch_size, num_views, -1, -1
        ).clone()
        return {
            "ugv_depths": depths,
            "camera_poses": poses,
            "intrinsics": intrinsics,
            "depth_range": torch.tensor([0.0, 10.0]),
            "ground_tile_size": 10.0,
        }

    @staticmethod
    def _use_identity_token_path(encoder):
        if encoder.feature_dim > encoder.feature_norm.normalized_shape[0]:
            raise ValueError("feature_dim must not exceed the dummy embed dimension")
        encoder.feature_norm = nn.Identity()
        encoder.token_mlp = _TakeFirst(encoder.feature_dim)
        encoder.column_mlp = _ZeroColumnUpdate(encoder.feature_dim)

    def test_exact_mean_max_residual_formula_for_single_and_multiple_tokens(self):
        encoder = self._make_encoder(embed_dim=4, feature_dim=3)
        src_feats = torch.randn(1, 1, 4, 4)
        linear_indices = torch.tensor([[[1, 1, 1, 0]]])
        valid = torch.ones_like(linear_indices, dtype=torch.bool)
        heights = torch.tensor([[[0.2, 0.7, 1.4, 2.0]]])

        actual, validity = encoder._fuse_projected_features(
            src_feats,
            linear_indices,
            valid,
            heights,
            bev_size=(1, 2),
        )

        height_encoding = torch.zeros(4, encoder._HEIGHT_ENCODING_DIM)
        tokens = encoder.token_mlp(
            torch.cat((encoder.feature_norm(src_feats[0, 0]), height_encoding), dim=-1)
        )
        expected_single = tokens[3] + encoder.column_mlp(
            torch.cat((tokens[3], tokens[3])).unsqueeze(0)
        ).squeeze(0)
        multi_tokens = tokens[:3]
        multi_mean = multi_tokens.mean(dim=0)
        expected_multi = multi_mean + encoder.column_mlp(
            torch.cat((multi_mean, multi_tokens.amax(dim=0))).unsqueeze(0)
        ).squeeze(0)

        torch.testing.assert_close(actual[0, :, 0, 0], expected_single)
        torch.testing.assert_close(actual[0, :, 0, 1], expected_multi)
        torch.testing.assert_close(
            validity, torch.tensor([[[[True, True]]]], dtype=torch.bool)
        )

    def test_n0_n1_n2_n3_n4_n7_n9_exact_column_formulas(self):
        src_feats = torch.tensor([[[[1.0, 3.0], [3.0, 7.0]]]])
        linear_indices = torch.zeros(1, 1, 2, dtype=torch.long)
        valid = torch.ones_like(linear_indices, dtype=torch.bool)
        heights = torch.zeros(1, 1, 2)
        mean = torch.tensor([2.0, 5.0])
        maximum = torch.tensor([3.0, 7.0])

        expected_by_variant = {
            "N0": mean + maximum,
            "N1": mean,
            "N2": mean + mean,
            "N3": mean + maximum,
            "N4": 2 * mean + maximum + torch.tensor([1.0, 2.0])
                + torch.log1p(torch.tensor(2.0)),
            "N9": mean + 0.1 * maximum,
        }
        for variant, expected in expected_by_variant.items():
            with self.subTest(variant=variant):
                encoder = self._make_encoder(
                    embed_dim=2, feature_dim=2, fusion_variant=variant
                )
                encoder.feature_norm = nn.Identity()
                encoder.token_mlp = _TakeFirst(2)
                if variant in {"N0", "N3", "N9"}:
                    encoder.column_mlp = _TakeFeatureChunk(2, 1 if variant != "N3" else 0)
                elif variant == "N2":
                    encoder.column_mlp = _TakeFeatureChunk(2, 0)
                elif variant == "N4":
                    self.assertEqual(
                        encoder.column_mlp[1].in_features,
                        3 * encoder.feature_dim + 1,
                    )
                    encoder.column_mlp = _SumN4Statistics(2)

                bev, validity = encoder._fuse_projected_features(
                    src_feats, linear_indices, valid, heights, bev_size=(1, 1)
                )
                torch.testing.assert_close(bev[0, :, 0, 0], expected)
                self.assertTrue(validity.item())

        attention_encoder = self._make_encoder(
            embed_dim=2, feature_dim=2, fusion_variant="N7"
        )
        attention_encoder.feature_norm = nn.Identity()
        attention_encoder.token_mlp = _TakeFirst(2)
        attention_encoder.token_attention = _ScoreFirst()
        attention_encoder.column_mlp = _TakeFeatureChunk(2, 1)
        weights = torch.softmax(torch.tensor([1.0, 3.0]), dim=0)
        attention_pool = (src_feats[0, 0] * weights.unsqueeze(-1)).sum(dim=0)
        bev, _ = attention_encoder._fuse_projected_features(
            src_feats, linear_indices, valid, heights, bev_size=(1, 1)
        )
        torch.testing.assert_close(bev[0, :, 0, 0], mean + attention_pool)

    def test_n8_exact_masked_cross_view_attention_and_view_permutation(self):
        encoder = self._make_encoder(
            embed_dim=2, feature_dim=2, fusion_variant="N8"
        )
        encoder.feature_norm = nn.Identity()
        encoder.token_mlp = _TakeFirst(2)
        encoder.column_mlp = _ZeroColumnUpdate(2)
        encoder.view_attention = _ScoreFirst()
        src_feats = torch.tensor(
            [[[[0.0, 2.0]], [[2.0, 4.0]], [[100.0, 100.0]]]]
        )
        linear_indices = torch.zeros(1, 3, 1, dtype=torch.long)
        valid = torch.tensor([[[True], [True], [False]]])
        heights = torch.zeros(1, 3, 1)

        bev, validity = encoder._fuse_projected_features(
            src_feats, linear_indices, valid, heights, bev_size=(1, 1)
        )
        weights = torch.softmax(torch.tensor([0.0, 2.0]), dim=0)
        expected = (src_feats[0, :2, 0] * weights.unsqueeze(-1)).sum(dim=0)
        torch.testing.assert_close(bev[0, :, 0, 0], expected)
        self.assertTrue(validity.item())

        permutation = torch.tensor([2, 0, 1])
        permuted, permuted_validity = encoder._fuse_projected_features(
            src_feats[:, permutation],
            linear_indices[:, permutation],
            valid[:, permutation],
            heights[:, permutation],
            bev_size=(1, 1),
        )
        torch.testing.assert_close(permuted, bev)
        torch.testing.assert_close(permuted_validity, validity)

        changed_invalid = src_feats.clone()
        changed_invalid[0, 2, 0] = torch.tensor([-10000.0, 10000.0])
        unchanged, unchanged_validity = encoder._fuse_projected_features(
            changed_invalid, linear_indices, valid, heights, bev_size=(1, 1)
        )
        torch.testing.assert_close(unchanged, bev)
        torch.testing.assert_close(unchanged_validity, validity)

        empty, empty_validity = encoder._fuse_projected_features(
            src_feats,
            linear_indices,
            torch.zeros_like(valid),
            heights,
            bev_size=(1, 1),
        )
        torch.testing.assert_close(empty, torch.zeros_like(empty))
        self.assertFalse(empty_validity.any())

    def test_n6_uses_clamped_normalized_scalar_height_in_first_channel(self):
        encoder = self._make_encoder(
            embed_dim=2, feature_dim=2, fusion_variant="N6"
        )
        heights = torch.tensor([-1.0, 0.0, 1.0, 2.0, 3.0])
        encoding = encoder._height_positional_encoding(
            heights, self._height_grid([0.0, 1.0, 2.0]), 0, 1
        )
        torch.testing.assert_close(
            encoding[:, 0], torch.tensor([0.0, 0.0, 0.5, 1.0, 1.0])
        )
        torch.testing.assert_close(encoding[:, 1:], torch.zeros(5, 15))

        half_encoding = encoder._height_positional_encoding(
            heights.half(), self._height_grid([0.0, 1.0, 2.0]).half(), 0, 1
        )
        self.assertEqual(half_encoding.dtype, torch.float32)

    def test_exact_fused_n5_and_n6_height_paths(self):
        src_feats = torch.zeros(1, 1, 3, 2)
        linear_indices = torch.zeros(1, 1, 3, dtype=torch.long)
        valid = torch.ones_like(linear_indices, dtype=torch.bool)
        grid = self._height_grid([0.0, 1.0, 2.0])

        n5 = self._make_encoder(
            embed_dim=2, feature_dim=16, fusion_variant="N5"
        )
        n5.feature_norm = nn.Identity()
        n5.token_mlp = _TakeLast(16)
        n5.column_mlp = _ZeroColumnUpdate(16)
        n5_heights = torch.tensor([[[0.0, 1.0, 2.0]]])
        n5_bev, n5_validity = n5._fuse_projected_features(
            src_feats,
            linear_indices,
            valid,
            n5_heights,
            bev_size=(1, 1),
            grid_points_3d=grid,
        )
        frequencies = torch.exp(
            torch.arange(8, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0)) / 8)
        )
        positions = torch.tensor([0.0, 1.0, 2.0])
        angles = positions.unsqueeze(1) * frequencies.unsqueeze(0)
        expected_n5 = torch.stack(
            (angles.sin(), angles.cos()), dim=-1
        ).flatten(start_dim=1).mean(dim=0)
        torch.testing.assert_close(n5_bev[0, :, 0, 0], expected_n5)
        self.assertTrue(n5_validity.item())

        n6 = self._make_encoder(
            embed_dim=2, feature_dim=16, fusion_variant="N6"
        )
        n6.feature_norm = nn.Identity()
        n6.token_mlp = _TakeLast(16)
        n6.column_mlp = _ZeroColumnUpdate(16)
        n6_bev, n6_validity = n6._fuse_projected_features(
            src_feats,
            linear_indices,
            valid,
            torch.tensor([[[-1.0, 1.0, 3.0]]]),
            bev_size=(1, 1),
            grid_points_3d=grid,
        )
        expected_n6 = torch.zeros(16)
        expected_n6[0] = 0.5
        torch.testing.assert_close(n6_bev[0, :, 0, 0], expected_n6)
        self.assertTrue(n6_validity.item())

    def test_n7_permutation_single_token_and_empty_cell_behavior(self):
        encoder = self._make_encoder(
            embed_dim=2, feature_dim=2, fusion_variant="N7"
        )
        encoder.feature_norm = nn.Identity()
        encoder.token_mlp = _TakeFirst(2)
        encoder.token_attention = _ScoreFirst()
        encoder.column_mlp = _TakeFeatureChunk(2, 1)
        src_feats = torch.tensor(
            [[[[1.0, 2.0], [3.0, 6.0], [5.0, 10.0]]]]
        )
        linear_indices = torch.tensor([[[0, 0, 1]]])
        valid = torch.ones_like(linear_indices, dtype=torch.bool)
        heights = torch.zeros(1, 1, 3)

        bev, validity = encoder._fuse_projected_features(
            src_feats, linear_indices, valid, heights, bev_size=(1, 3)
        )
        permutation = torch.tensor([2, 0, 1])
        permuted, permuted_validity = encoder._fuse_projected_features(
            src_feats[:, :, permutation],
            linear_indices[:, :, permutation],
            valid[:, :, permutation],
            heights[:, :, permutation],
            bev_size=(1, 3),
        )
        torch.testing.assert_close(permuted, bev)
        torch.testing.assert_close(permuted_validity, validity)
        torch.testing.assert_close(
            bev[0, :, 0, 1], 2.0 * src_feats[0, 0, 2]
        )
        torch.testing.assert_close(bev[0, :, 0, 2], torch.zeros(2))
        torch.testing.assert_close(
            validity, torch.tensor([[[[True, True, False]]]])
        )

    def test_fusion_is_permutation_invariant_with_height_encoding(self):
        encoder = self._make_encoder(
            embed_dim=4,
            feature_dim=3,
            use_height_positional_encoding=True,
        )
        src_feats = torch.randn(1, 2, 5, 4)
        linear_indices = torch.tensor([[[0, 1, 1, 3, -1], [1, 2, 2, 2, 3]]])
        valid = torch.tensor(
            [[[True, True, True, True, False], [True, True, True, True, True]]]
        )
        heights = torch.tensor(
            [[[0.0, 0.5, 1.0, 1.5, 2.0], [2.0, 1.5, 1.0, 0.5, 0.0]]]
        )
        grid = self._height_grid([0.0, 1.0, 2.0])

        expected = encoder._fuse_projected_features(
            src_feats,
            linear_indices,
            valid,
            heights,
            bev_size=(2, 2),
            grid_points_3d=grid,
        )
        permutation = torch.tensor([3, 0, 4, 1, 2])
        actual = encoder._fuse_projected_features(
            src_feats[:, :, permutation],
            linear_indices[:, :, permutation],
            valid[:, :, permutation],
            heights[:, :, permutation],
            bev_size=(2, 2),
            grid_points_3d=grid,
        )

        torch.testing.assert_close(actual[0], expected[0])
        torch.testing.assert_close(actual[1], expected[1])

    def test_views_are_equally_weighted_after_within_view_pooling(self):
        encoder = self._make_encoder(embed_dim=2, feature_dim=2)
        self._use_identity_token_path(encoder)
        src_feats = torch.tensor(
            [[
                [[0.0, 0.0], [2.0, 2.0], [4.0, 4.0]],
                [[10.0, 10.0], [100.0, 100.0], [100.0, 100.0]],
                [[200.0, 200.0], [200.0, 200.0], [200.0, 200.0]],
            ]]
        )
        linear_indices = torch.zeros(1, 3, 3, dtype=torch.long)
        valid = torch.tensor(
            [[[True, True, True], [True, False, False], [False, False, False]]]
        )
        heights = torch.zeros(1, 3, 3)

        bev, validity = encoder._fuse_projected_features(
            src_feats,
            linear_indices,
            valid,
            heights,
            bev_size=(1, 1),
        )

        # View 0 pools to [2, 2], view 1 to [10, 10], and the empty view is ignored.
        torch.testing.assert_close(bev[0, :, 0, 0], torch.tensor([6.0, 6.0]))
        self.assertTrue(validity.item())

    def test_invalid_contributions_are_filtered_and_empty_cells_stay_zero(self):
        encoder = self._make_encoder(embed_dim=2, feature_dim=2)
        self._use_identity_token_path(encoder)
        src_feats = torch.tensor(
            [
                [[[1.0, 2.0], [100.0, 100.0], [200.0, 200.0], [300.0, 300.0]]],
                [[[9.0, 9.0], [9.0, 9.0], [9.0, 9.0], [9.0, 9.0]]],
            ]
        )
        linear_indices = torch.tensor([[[0, -999, 999, 0]], [[0, 0, 0, 0]]])
        valid = torch.tensor(
            [[[True, True, True, False]], [[False, False, False, False]]]
        )
        heights = torch.zeros(2, 1, 4)

        bev, validity = encoder._fuse_projected_features(
            src_feats,
            linear_indices,
            valid,
            heights,
            bev_size=(1, 2),
        )

        torch.testing.assert_close(bev[0, :, 0, 0], torch.tensor([1.0, 2.0]))
        torch.testing.assert_close(bev[0, :, 0, 1], torch.zeros(2))
        torch.testing.assert_close(bev[1], torch.zeros_like(bev[1]))
        torch.testing.assert_close(
            validity,
            torch.tensor([[[[True, False]]], [[[False, False]]]]),
        )

    def test_disabled_height_encoding_makes_fusion_height_independent(self):
        encoder = self._make_encoder(embed_dim=3, feature_dim=3)
        src_feats = torch.randn(1, 1, 3, 3)
        linear_indices = torch.zeros(1, 1, 3, dtype=torch.long)
        valid = torch.ones_like(linear_indices, dtype=torch.bool)

        low, _ = encoder._fuse_projected_features(
            src_feats,
            linear_indices,
            valid,
            torch.tensor([[[-100.0, -50.0, 0.0]]]),
            bev_size=(1, 1),
        )
        high, _ = encoder._fuse_projected_features(
            src_feats,
            linear_indices,
            valid,
            torch.tensor([[[10.0, 100.0, 1000.0]]]),
            bev_size=(1, 1),
        )

        torch.testing.assert_close(low, high)

    def test_enabled_height_encoding_distinguishes_positions_and_clamps(self):
        encoder = self._make_encoder(
            embed_dim=3,
            feature_dim=16,
            use_height_positional_encoding=True,
        ).double()
        encoder.feature_norm = nn.Identity()
        encoder.token_mlp = _TakeLast(16)
        encoder.column_mlp = _ZeroColumnUpdate(16)
        src_feats = torch.zeros(1, 1, 1, 3, dtype=torch.float64)
        linear_indices = torch.zeros(1, 1, 1, dtype=torch.long)
        valid = torch.ones_like(linear_indices, dtype=torch.bool)
        grid = self._height_grid([0.0, 1.0, 2.0], dtype=torch.float64)

        def fused_at(height):
            bev, _ = encoder._fuse_projected_features(
                src_feats,
                linear_indices,
                valid,
                torch.tensor([[[height]]], dtype=torch.float64),
                bev_size=(1, 1),
                grid_points_3d=grid,
            )
            return bev[0, :, 0, 0]

        below, minimum = fused_at(-1.0), fused_at(0.0)
        middle = fused_at(1.0)
        maximum, above = fused_at(2.0), fused_at(3.0)

        torch.testing.assert_close(below, minimum)
        torch.testing.assert_close(maximum, above)
        self.assertFalse(torch.allclose(minimum, middle))
        self.assertFalse(torch.allclose(middle, maximum))
        self.assertEqual(middle.dtype, torch.float64)
        self.assertEqual(middle.device, src_feats.device)

    def test_height_encoding_rejects_missing_or_degenerate_grids(self):
        encoder = self._make_encoder(use_height_positional_encoding=True)
        heights = torch.tensor([1.0])

        with self.assertRaisesRegex(ValueError, "grid_points_3d must be provided"):
            encoder._height_positional_encoding(heights, None, 0, 1)
        with self.assertRaisesRegex(ValueError, "at least two vertical grid levels"):
            encoder._height_positional_encoding(
                heights, self._height_grid([1.0]), 0, 1
            )
        with self.assertRaisesRegex(ValueError, "non-zero vertical grid span"):
            encoder._height_positional_encoding(
                heights, self._height_grid([1.0, 1.0]), 0, 1
            )
        with self.assertRaisesRegex(ValueError, "non-finite height values"):
            encoder._height_positional_encoding(
                heights, self._height_grid([0.0, float("nan")]), 0, 1
            )

        with self.assertRaisesRegex(ValueError, "grid_points_3d must be provided"):
            encoder._fuse_projected_features(
                torch.zeros(1, 1, 1, 4),
                torch.zeros(1, 1, 1, dtype=torch.long),
                torch.zeros(1, 1, 1, dtype=torch.bool),
                torch.zeros(1, 1, 1),
                bev_size=(1, 1),
                grid_points_3d=None,
            )

    def test_gradients_reach_features_and_both_fusion_mlps(self):
        encoder = self._make_encoder(embed_dim=4, feature_dim=3)
        src_feats = torch.randn(1, 2, 4, 4, requires_grad=True)
        linear_indices = torch.tensor([[[0, 0, 1, 1], [0, 1, 1, 0]]])
        valid = torch.ones_like(linear_indices, dtype=torch.bool)
        heights = torch.randn(1, 2, 4)

        bev, _ = encoder._fuse_projected_features(
            src_feats,
            linear_indices,
            valid,
            heights,
            bev_size=(1, 2),
        )
        bev.square().mean().backward()

        self.assertIsNotNone(src_feats.grad)
        self.assertTrue(torch.isfinite(src_feats.grad).all())
        self.assertGreater(src_feats.grad.abs().sum().item(), 0.0)
        for module in (encoder.token_mlp, encoder.column_mlp):
            gradients = [parameter.grad for parameter in module.parameters()]
            self.assertTrue(all(gradient is not None for gradient in gradients))
            self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))
            self.assertGreater(sum(gradient.abs().sum().item() for gradient in gradients), 0.0)

    def test_n4_single_token_has_finite_feature_and_parameter_gradients(self):
        encoder = self._make_encoder(
            embed_dim=4, feature_dim=3, fusion_variant="N4"
        )
        src_feats = torch.randn(1, 1, 1, 4, requires_grad=True)
        bev, validity = encoder._fuse_projected_features(
            src_feats,
            torch.zeros(1, 1, 1, dtype=torch.long),
            torch.ones(1, 1, 1, dtype=torch.bool),
            torch.zeros(1, 1, 1),
            bev_size=(1, 1),
        )
        bev.square().sum().backward()

        self.assertTrue(validity.item())
        self.assertIsNotNone(src_feats.grad)
        self.assertTrue(torch.isfinite(src_feats.grad).all())
        for name, parameter in encoder.fusion_named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_half_and_bfloat16_inputs_use_stable_aggregation_and_restore_dtype(self):
        for input_dtype in (torch.float16, torch.bfloat16):
            for variant in ("N0", "N4", "N7", "N8"):
                with self.subTest(dtype=input_dtype, variant=variant):
                    encoder = self._make_encoder(
                        embed_dim=4, feature_dim=3, fusion_variant=variant
                    )
                    src_feats = torch.tensor(
                        [[
                            [[1.0, 2.0, 3.0, 4.0], [3.0, 4.0, 5.0, 6.0]],
                            [[7.0, 8.0, 9.0, 10.0], [9.0, 10.0, 11.0, 12.0]],
                        ]],
                        dtype=input_dtype,
                        requires_grad=True,
                    )
                    bev, validity = encoder._fuse_projected_features(
                        src_feats,
                        torch.zeros(1, 2, 2, dtype=torch.long),
                        torch.ones(1, 2, 2, dtype=torch.bool),
                        torch.zeros(1, 2, 2, dtype=input_dtype),
                        bev_size=(1, 1),
                    )
                    self.assertEqual(bev.dtype, input_dtype)
                    self.assertTrue(torch.isfinite(bev).all())
                    self.assertTrue(validity.item())
                    bev.float().square().sum().backward()
                    self.assertIsNotNone(src_feats.grad)
                    self.assertTrue(torch.isfinite(src_feats.grad).all())

    def test_variant_specific_attention_and_gate_parameters_receive_gradients(self):
        for variant, parameter_prefix in (
            ("N7", "token_attention."),
            ("N8", "view_attention."),
            ("N9", "column_residual_scale"),
        ):
            with self.subTest(variant=variant):
                encoder = self._make_encoder(
                    embed_dim=4, feature_dim=3, fusion_variant=variant
                )
                src_feats = torch.randn(1, 2, 4, 4, requires_grad=True)
                linear_indices = torch.tensor([[[0, 0, 1, 1], [0, 1, 1, 0]]])
                valid = torch.ones_like(linear_indices, dtype=torch.bool)
                heights = torch.zeros(1, 2, 4)
                bev, _ = encoder._fuse_projected_features(
                    src_feats, linear_indices, valid, heights, bev_size=(1, 2)
                )
                bev.square().sum().backward()

                named = dict(encoder.fusion_named_parameters())
                matching = [
                    parameter
                    for name, parameter in named.items()
                    if name == parameter_prefix or name.startswith(parameter_prefix)
                ]
                self.assertTrue(matching)
                self.assertTrue(all(parameter.grad is not None for parameter in matching))
                self.assertTrue(all(torch.isfinite(parameter.grad).all() for parameter in matching))
                self.assertGreater(
                    sum(parameter.grad.abs().sum().item() for parameter in matching), 0.0
                )

    def test_fusion_handles_autocast_output_dtype(self):
        encoder = self._make_encoder(embed_dim=4, feature_dim=3)
        src_feats = torch.randn(1, 1, 2, 4)
        linear_indices = torch.tensor([[[0, 0]]])
        valid = torch.ones_like(linear_indices, dtype=torch.bool)
        heights = torch.zeros(1, 1, 2)

        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            bev, validity = encoder._fuse_projected_features(
                src_feats,
                linear_indices,
                valid,
                heights,
                bev_size=(1, 1),
            )

        self.assertEqual(bev.dtype, src_feats.dtype)
        self.assertTrue(torch.isfinite(bev).all())
        self.assertTrue(validity.item())

    def test_avg_mode_matches_global_average_followed_by_legacy_mlp(self):
        encoder = self._make_encoder(
            embed_dim=4,
            feature_dim=3,
            fusion_mode="avg",
        )
        features = torch.tensor(
            [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]
        ).view(1, 2, 4, 1, 1)
        projection_inputs = self._projection_inputs()

        actual, validity = encoder.project(
            encoded_ground={"features_2d": features, "image_size": (1, 1)},
            **projection_inputs,
        )
        expected = encoder.fusion_mlp(features.mean(dim=1).flatten(2).transpose(1, 2))
        expected = expected.transpose(1, 2).view(1, 3, 1, 1)

        torch.testing.assert_close(actual, expected)
        self.assertTrue(validity.item())

    def test_direct_and_cached_mlp_projection_have_the_same_public_shapes(self):
        encoder = self._make_encoder(
            embed_dim=4,
            feature_dim=3,
            use_height_positional_encoding=True,
        )
        images = torch.arange(8, dtype=torch.float32).view(1, 2, 4, 1, 1)
        projection_inputs = self._projection_inputs()
        projection_inputs["grid_points_3d"] = self._height_grid(
            [0.0, 1.0, 2.0], batch_size=1
        )

        direct_bev, direct_validity = encoder.project(
            ugv_images=images,
            **projection_inputs,
        )
        cached = encoder.encode(images)
        cached_bev, cached_validity = encoder.project(
            encoded_ground=cached,
            **projection_inputs,
        )

        self.assertEqual(direct_bev.shape, (1, 3, 1, 1))
        self.assertEqual(direct_validity.shape, (1, 1, 1, 1))
        torch.testing.assert_close(direct_bev, cached_bev)
        torch.testing.assert_close(direct_validity, cached_validity)

    def test_mode_validation_and_mode_specific_state_dicts(self):
        with self.assertRaisesRegex(ValueError, "Unsupported ground fusion mode"):
            self._make_encoder(fusion_mode="attention")
        with self.assertRaisesRegex(ValueError, "only supported"):
            self._make_encoder(
                fusion_mode="avg",
                use_height_positional_encoding=True,
            )
        with self.assertRaisesRegex(ValueError, "Unsupported ground fusion variant"):
            self._make_encoder(fusion_variant="N10")
        with self.assertRaisesRegex(ValueError, "only supported"):
            self._make_encoder(fusion_mode="avg", fusion_variant="N0")

        legacy_n0 = self._make_encoder(use_height_positional_encoding=False)
        explicit_n0 = self._make_encoder(
            use_height_positional_encoding=True, fusion_variant="N0"
        )
        self.assertEqual(legacy_n0.fusion_variant, "N0")
        self.assertEqual(explicit_n0.fusion_variant, "N0")
        self.assertFalse(explicit_n0.use_height_positional_encoding)
        self.assertEqual(
            explicit_n0.load_state_dict(legacy_n0.state_dict(), strict=True).missing_keys,
            [],
        )
        legacy_n5 = self._make_encoder(use_height_positional_encoding=True)
        explicit_n5 = self._make_encoder(
            use_height_positional_encoding=False, fusion_variant="n5"
        )
        self.assertEqual(legacy_n5.fusion_variant, "N5")
        self.assertEqual(explicit_n5.fusion_variant, "N5")
        self.assertTrue(explicit_n5.use_height_positional_encoding)
        self.assertEqual(
            explicit_n5.load_state_dict(legacy_n5.state_dict(), strict=True).missing_keys,
            [],
        )

        avg_source = self._make_encoder(fusion_mode="avg")
        avg_target = self._make_encoder(fusion_mode="avg")
        avg_result = avg_target.load_state_dict(avg_source.state_dict(), strict=True)
        self.assertEqual(avg_result.missing_keys, [])
        self.assertEqual(avg_result.unexpected_keys, [])
        legacy_state = avg_source.state_dict()
        legacy_state["projection_layer.weight"] = torch.randn(3, 4, 1, 1)
        legacy_state["projection_layer.bias"] = torch.randn(3)
        legacy_result = avg_target.load_state_dict(legacy_state, strict=True)
        self.assertEqual(legacy_result.missing_keys, [])
        self.assertEqual(legacy_result.unexpected_keys, [])
        avg_keys = tuple(avg_source.state_dict())
        self.assertTrue(any(key.startswith("fusion_mlp.") for key in avg_keys))
        self.assertFalse(any(key.startswith("token_mlp.") for key in avg_keys))

        mlp_source = self._make_encoder(fusion_mode="mlp")
        mlp_target = self._make_encoder(fusion_mode="mlp")
        mlp_result = mlp_target.load_state_dict(mlp_source.state_dict(), strict=True)
        self.assertEqual(mlp_result.missing_keys, [])
        self.assertEqual(mlp_result.unexpected_keys, [])
        mlp_keys = tuple(mlp_source.state_dict())
        self.assertTrue(any(key.startswith("feature_norm.") for key in mlp_keys))
        self.assertTrue(any(key.startswith("token_mlp.") for key in mlp_keys))
        self.assertTrue(any(key.startswith("column_mlp.") for key in mlp_keys))
        self.assertFalse(any(key.startswith("fusion_mlp.") for key in mlp_keys))

        expected_variant_members = {
            "N1": (False, False, False, False),
            "N7": (True, True, False, False),
            "N8": (True, False, True, False),
            "N9": (True, False, False, True),
        }
        for variant, expected in expected_variant_members.items():
            encoder = self._make_encoder(fusion_variant=variant)
            actual = (
                hasattr(encoder, "column_mlp"),
                hasattr(encoder, "token_attention"),
                hasattr(encoder, "view_attention"),
                hasattr(encoder, "column_residual_scale"),
            )
            self.assertEqual(actual, expected)
            names = dict(encoder.fusion_named_parameters())
            self.assertTrue(any(name.startswith("feature_norm.") for name in names))
            self.assertTrue(any(name.startswith("token_mlp.") for name in names))


if __name__ == "__main__":
    unittest.main()
