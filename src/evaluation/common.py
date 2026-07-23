from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R


def apply_checkpoint_fusion_config(config: dict, checkpoint_path: str) -> None:
    """Restore fusion and vertical-grid settings from the nearest checkpoint config."""
    checkpoint = Path(checkpoint_path).resolve()
    candidates = (
        checkpoint.with_name("config.json"),
        checkpoint.parent.parent / "config.json",
    )

    for candidate in candidates:
        if not candidate.exists():
            continue
        with open(candidate, "r", encoding="utf-8") as f:
            checkpoint_config = json.load(f)
        for key in (
            "ground_fusion_mode",
            "use_height_positional_encoding",
            "grid_size",
            "grid_resolution",
        ):
            if key in checkpoint_config:
                config[key] = checkpoint_config[key]
        return


def parse_angle_list(angle_str: str) -> list[float]:
    """Parse comma-separated angles in degrees, preserving order and uniqueness."""
    if not angle_str.strip():
        return [0.0]

    out: list[float] = []
    seen = set()
    for token in angle_str.split(","):
        angle = float(token.strip())
        key = round(angle, 8)
        if key in seen:
            continue
        seen.add(key)
        out.append(angle)

    return out if out else [0.0]


def yaw_rotation_matrix(degrees: float | torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Create a 3x3 yaw rotation matrix around +Z axis for an angle in degrees."""
    degrees_tensor = torch.as_tensor(degrees, device=device, dtype=dtype)
    radians = torch.deg2rad(degrees_tensor)
    c = torch.cos(radians)
    s = torch.sin(radians)
    zero = torch.zeros((), device=device, dtype=dtype)
    one = torch.ones((), device=device, dtype=dtype)
    row0 = torch.stack((c, -s, zero))
    row1 = torch.stack((s, c, zero))
    row2 = torch.stack((zero, zero, one))
    return torch.stack((row0, row1, row2), dim=0)


def pose_to_local_xy(camera_pose_w2c: torch.Tensor) -> Tuple[float, float]:
    """Extract camera local (x, y) position in meters from a world-to-camera matrix."""
    w2c = camera_pose_w2c.detach().cpu().numpy()
    c2w = np.linalg.inv(w2c)
    return float(c2w[0, 3]), float(c2w[1, 3])


def get_gt_yaw(camera_pose_w2c: torch.Tensor) -> float:
    """Extract ground truth yaw angle (degrees) from camera pose."""
    w2c = camera_pose_w2c.detach().cpu().numpy()
    c2w = np.linalg.inv(w2c)
    r = R.from_matrix(c2w[:3, :3])
    _, _, yaw = r.as_euler('xyz', degrees=False)
    return float(np.degrees(yaw))


def angular_distance(angle1_deg: float, angle2_deg: float) -> float:
    """Compute shortest angular distance between two angles in degrees."""
    delta = (angle2_deg - angle1_deg) % 360
    if delta > 180:
        delta = 360 - delta
    return delta


def clone_ugv_data_with_pose(ugv_data: dict[str, torch.Tensor], camera_poses: torch.Tensor) -> dict[str, torch.Tensor]:
    """Clone the batch dictionary while swapping in a new camera pose tensor."""
    cloned = dict(ugv_data)
    cloned["camera_poses"] = camera_poses
    return cloned


def masked_avg_pool(features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Masked average pooling for BEV feature maps."""
    if mask.dim() == 4 and mask.size(1) == 1:
        mask = mask.squeeze(1)
    elif mask.dim() != 3:
        raise ValueError(f"Expected mask shape (B,H,W) or (B,1,H,W), got {tuple(mask.shape)}")

    mask = mask.unsqueeze(1).float()
    masked_features = features * mask
    sum_feat = masked_features.sum(dim=(2, 3))
    valid_counts = mask.sum(dim=(2, 3)).clamp(min=1e-6)
    return sum_feat / valid_counts


def pool_bev_embeddings(features: torch.Tensor, validity_mask: torch.Tensor | None = None) -> torch.Tensor:
    """Pool a BEV feature map into a single embedding per sample."""
    if validity_mask is None:
        return features.mean(dim=(2, 3))
    return masked_avg_pool(features, validity_mask)


def compute_cosine_map(ground_bev: torch.Tensor, overhead_bev: torch.Tensor) -> torch.Tensor:
    """Compute per-pixel cosine similarity map with shape (B, H, W)."""
    if ground_bev.shape[0] != overhead_bev.shape[0]:
        batch_size = min(ground_bev.shape[0], overhead_bev.shape[0])
        ground_bev = ground_bev[:batch_size]
        overhead_bev = overhead_bev[:batch_size]

    overhead_bev_resized = F.interpolate(
        overhead_bev,
        size=ground_bev.shape[2:],
        mode="bilinear",
        align_corners=False,
    )
    return F.cosine_similarity(ground_bev, overhead_bev_resized, dim=1)


def score_bev_pair(
    ground_bev: torch.Tensor,
    overhead_bev: torch.Tensor,
    validity_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a scalar score and the cosine map for one BEV pair."""
    cosine_map = compute_cosine_map(ground_bev, overhead_bev)
    if validity_mask is None:
        return cosine_map.mean(), cosine_map

    valid = validity_mask.squeeze(1) > 0
    values = cosine_map[valid]
    if values.numel() == 0:
        return cosine_map.new_tensor(0.0), cosine_map
    return values.mean(), cosine_map


def project_scene_features(
    model,
    ugv_data: dict[str, torch.Tensor],
    uav_data: dict[str, torch.Tensor],
    encoded_ground=None,
    encoded_overhead=None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Project one scene using cached encoders when available."""
    if encoded_ground is not None and encoded_overhead is not None and hasattr(model, "ground_encoder"):
        ground_bev, ground_validity = model.ground_encoder.project(encoded_ground=encoded_ground, ugv_data=ugv_data)
        overhead_bev = model.overhead_encoder.project(encoded_overhead)
        return ground_bev, overhead_bev, ground_validity

    ground_bev, overhead_bev, ground_validity = model(ugv_data, uav_data)
    return ground_bev, overhead_bev, ground_validity


def evaluate_pose_grid(
    model,
    ugv_data: dict[str, torch.Tensor],
    uav_data: dict[str, torch.Tensor],
    grid_range_m: float,
    grid_resolution_m: float,
    softmax_temp: float,
    yaw_candidates_deg: list[float],
    encoded_ground=None,
    encoded_overhead=None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate explicit camera pose hypotheses over a local XY translation grid."""
    if ugv_data["camera_poses"].shape[0] != 1:
        raise ValueError("evaluate_pose_grid expects batch_size == 1")

    device = ugv_data["camera_poses"].device
    dtype = ugv_data["camera_poses"].dtype

    offsets = torch.arange(
        -grid_range_m,
        grid_range_m + 0.5 * grid_resolution_m,
        grid_resolution_m,
        device=device,
        dtype=dtype,
    )
    grid_h = int(offsets.numel())
    grid_w = int(offsets.numel())
    pose_scores = torch.empty((grid_h, grid_w), device=device, dtype=torch.float32)
    best_yaw_idx = torch.zeros((grid_h, grid_w), device=device, dtype=torch.long)
    yaw_candidates_tensor = torch.tensor(yaw_candidates_deg, device=device, dtype=dtype)
    if yaw_candidates_tensor.numel() == 0:
        yaw_candidates_tensor = torch.tensor([0.0], device=device, dtype=dtype)

    original_camera_poses = ugv_data["camera_poses"].clone()
    original_pose = original_camera_poses[0, 0].clone()
    camera_w2c = original_pose
    poses = torch.empty((grid_h, grid_w, 2), device=device, dtype=dtype)

    use_cached = encoded_ground is not None and encoded_overhead is not None and hasattr(model, "ground_encoder")
    if use_cached:
        overhead_bev = model.overhead_encoder.project(encoded_overhead)
    else:
        overhead_bev = None

    for iy, dy in enumerate(offsets):
        for ix, dx in enumerate(offsets):
            best_score = None
            best_angle_idx = 0

            for angle_idx, yaw_deg in enumerate(yaw_candidates_tensor):
                modified_pose = camera_w2c.clone()
                modified_pose[0, 3] -= dx
                modified_pose[1, 3] -= dy
                r_delta = yaw_rotation_matrix(yaw_deg, device=device, dtype=dtype)
                modified_pose[:3, :3] = torch.matmul(r_delta, modified_pose[:3, :3])

                candidate_camera_poses = original_camera_poses.clone()
                candidate_camera_poses[0, 0] = modified_pose
                candidate_ugv_data = clone_ugv_data_with_pose(ugv_data, candidate_camera_poses)

                if use_cached:
                    ground_bev, ground_validity = model.ground_encoder.project(
                        encoded_ground=encoded_ground,
                        ugv_data=candidate_ugv_data,
                    )
                    current_overhead_bev = overhead_bev
                else:
                    ground_bev, current_overhead_bev, ground_validity = model(candidate_ugv_data, uav_data)

                score, _ = score_bev_pair(ground_bev, current_overhead_bev, ground_validity)

                if best_score is None or score > best_score:
                    best_score = score
                    best_angle_idx = angle_idx

            pose_scores[iy, ix] = best_score.float()
            best_yaw_idx[iy, ix] = best_angle_idx

            c2w = torch.linalg.inv(camera_w2c)
            poses[iy, ix, 0] = c2w[0, 3] + dx
            poses[iy, ix, 1] = c2w[1, 3] + dy

    logits = pose_scores / max(softmax_temp, 1e-6)
    probability_map = F.softmax(logits.view(-1), dim=0).view_as(logits)
    best_yaw_map_deg = yaw_candidates_tensor[best_yaw_idx]
    return pose_scores, probability_map, best_yaw_map_deg, offsets, poses


def compute_position_stats(
    prob_map: torch.Tensor,
    best_yaw_map_deg: torch.Tensor,
    poses: torch.Tensor,
    gt_xy_m: Tuple[float, float],
    gt_yaw_deg: float,
    yaw_candidates_deg: list[float],
    top_k: int = 5,
    distance_thresholds: list[float] | None = None,
) -> dict:
    """Compute localization accuracy statistics for a single scene."""
    if distance_thresholds is None:
        distance_thresholds = [1.0, 2.0, 3.0]

    gt_x, gt_y = gt_xy_m
    flat_prob = prob_map.view(-1)
    grid_w = prob_map.shape[1]
    vals, idxs = torch.topk(flat_prob, k=min(top_k, len(flat_prob)))

    stats = {
        "gt_x_m": gt_x,
        "gt_y_m": gt_y,
        "gt_yaw_deg": gt_yaw_deg,
    }

    best_flat_idx = int(flat_prob.argmax().item())
    best_iy = best_flat_idx // grid_w
    best_ix = best_flat_idx % grid_w

    pred_x_m = float(poses[best_iy, best_ix, 0].item())
    pred_y_m = float(poses[best_iy, best_ix, 1].item())
    pred_yaw_deg = float(best_yaw_map_deg[best_iy, best_ix].item())

    distance_m = float(np.hypot(pred_x_m - gt_x, pred_y_m - gt_y))
    stats["most_probable_pred_x_m"] = pred_x_m
    stats["most_probable_pred_y_m"] = pred_y_m
    stats["most_probable_distance_m"] = distance_m
    stats["most_probable_prob"] = float(vals[0].item())

    yaw_error_deg = angular_distance(pred_yaw_deg, gt_yaw_deg)
    stats["most_probable_pred_yaw_deg"] = pred_yaw_deg
    stats["most_probable_yaw_error_deg"] = yaw_error_deg

    for thresh in distance_thresholds:
        key = f"most_probable_within_{thresh}m"
        stats[key] = 1 if distance_m <= thresh else 0

    min_distance_m = distance_m
    nearest_prob = float(vals[0].item())
    topk_hits_by_threshold = {thresh: 0 for thresh in distance_thresholds}

    for j in range(len(vals)):
        iy = int((idxs[j] // grid_w).item())
        ix = int((idxs[j] % grid_w).item())
        cand_x_m = float(poses[iy, ix, 0].item())
        cand_y_m = float(poses[iy, ix, 1].item())
        cand_distance_m = float(np.hypot(cand_x_m - gt_x, cand_y_m - gt_y))

        if cand_distance_m < min_distance_m:
            min_distance_m = cand_distance_m
            nearest_prob = float(vals[j].item())

        for thresh in distance_thresholds:
            if cand_distance_m <= thresh:
                topk_hits_by_threshold[thresh] += 1

    stats["nearest_in_topk_distance_m"] = min_distance_m
    stats["nearest_in_topk_prob"] = nearest_prob

    for thresh in distance_thresholds:
        key = f"nearest_in_topk_within_{thresh}m"
        stats[key] = 1 if min_distance_m <= thresh else 0

        topk_hit_key = f"topk_recall_within_{thresh}m"
        topk_precision_key = f"topk_precision_within_{thresh}m"
        stats[topk_hit_key] = 1 if topk_hits_by_threshold[thresh] > 0 else 0
        stats[topk_precision_key] = topk_hits_by_threshold[thresh] / max(len(vals), 1)

    return stats


def compute_aggregate_stats(rows: list[dict], distance_thresholds: list[float] | None = None) -> dict:
    """Compute aggregate statistics across all scenes."""
    if not rows:
        return {}

    if distance_thresholds is None:
        distance_thresholds = [1.0, 2.0, 3.0]

    stats: dict[str, float | int] = {}
    most_prob_distances = [r["most_probable_distance_m"] for r in rows]
    nearest_topk_distances = [r["nearest_in_topk_distance_m"] for r in rows]

    stats["num_scenes"] = len(rows)
    stats["most_probable_distance_mean_m"] = float(np.mean(most_prob_distances))
    stats["most_probable_distance_median_m"] = float(np.median(most_prob_distances))
    stats["most_probable_distance_std_m"] = float(np.std(most_prob_distances))
    stats["most_probable_distance_max_m"] = float(np.max(most_prob_distances))

    stats["nearest_topk_distance_mean_m"] = float(np.mean(nearest_topk_distances))
    stats["nearest_topk_distance_median_m"] = float(np.median(nearest_topk_distances))
    stats["nearest_topk_distance_std_m"] = float(np.std(nearest_topk_distances))
    stats["nearest_topk_distance_max_m"] = float(np.max(nearest_topk_distances))

    most_prob_yaw_errors = [r["most_probable_yaw_error_deg"] for r in rows]
    stats["most_probable_yaw_error_mean_deg"] = float(np.mean(most_prob_yaw_errors))
    stats["most_probable_yaw_error_median_deg"] = float(np.median(most_prob_yaw_errors))
    stats["most_probable_yaw_error_std_deg"] = float(np.std(most_prob_yaw_errors))
    stats["most_probable_yaw_error_max_deg"] = float(np.max(most_prob_yaw_errors))

    for thresh in distance_thresholds:
        most_prob_key = f"most_probable_within_{thresh}m"
        nearest_key = f"nearest_in_topk_within_{thresh}m"
        topk_hit_key = f"topk_recall_within_{thresh}m"
        topk_precision_key = f"topk_precision_within_{thresh}m"

        if most_prob_key in rows[0]:
            count_most_prob = sum(r[most_prob_key] for r in rows)
            count_nearest = sum(r[nearest_key] for r in rows)
            count_topk_hits = sum(r[topk_hit_key] for r in rows)

            stats[f"most_probable_within_{thresh}m_count"] = count_most_prob
            stats[f"most_probable_within_{thresh}m_ratio"] = count_most_prob / len(rows)
            stats[f"nearest_topk_within_{thresh}m_count"] = count_nearest
            stats[f"nearest_topk_within_{thresh}m_ratio"] = count_nearest / len(rows)
            stats[f"topk_recall_within_{thresh}m_count"] = count_topk_hits
            stats[f"topk_recall_within_{thresh}m_ratio"] = count_topk_hits / len(rows)
            stats[f"topk_precision_within_{thresh}m_mean"] = float(np.mean([r[topk_precision_key] for r in rows]))

    return stats


def write_results_csv(path: str, rows: list[dict]) -> None:
    """Write evaluation results to CSV."""
    if not rows:
        return

    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def compute_retrieval_metrics(
    ground_embeddings: torch.Tensor,
    overhead_embeddings: torch.Tensor,
    ks: Iterable[int] = (1, 2, 5, 10, 20),
) -> dict:
    """Compute ground-to-overhead retrieval metrics from pooled embeddings."""
    if ground_embeddings.shape != overhead_embeddings.shape:
        raise ValueError(
            f"Expected matching embedding shapes, got {tuple(ground_embeddings.shape)} and {tuple(overhead_embeddings.shape)}"
        )

    ground_embeddings = F.normalize(ground_embeddings, p=2, dim=1)
    overhead_embeddings = F.normalize(overhead_embeddings, p=2, dim=1)
    similarity = ground_embeddings @ overhead_embeddings.t()
    ordering = similarity.argsort(dim=1, descending=True)
    targets = torch.arange(similarity.size(0), device=similarity.device).view(-1, 1)

    metrics: dict[str, float] = {}
    for k in ks:
        hit = (ordering[:, :k] == targets).any(dim=1).float().mean().item()
        metrics[f"recall@{k}"] = float(hit)

    ranks = (ordering == targets).nonzero(as_tuple=False)[:, 1] + 1
    metrics["mean_rank"] = float(ranks.float().mean().item())
    metrics["median_rank"] = float(ranks.float().median().item())
    metrics["mrr"] = float((1.0 / ranks.float()).mean().item())
    return metrics
