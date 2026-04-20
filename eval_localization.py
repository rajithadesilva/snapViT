import argparse
import json
import os
from typing import Tuple
import csv

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from dataset import VineyardDataset
from model import SnapViT


CONFIG = {
    "vit_model": "vit_small_patch16_224",
    "train_img_size": (224, 224),
    "feature_dim": 128,
    "num_ugv_views": 1,
    "grid_size": (34, 34, 8),
    "grid_resolution": 0.3,
    "batch_size": 1,
    "device": "cuda:0" if torch.cuda.is_available() else "cpu",
    "use_depth": True,
    "depth_range": (0.0, 5.0),
    "ground_tile_size": 10.0,
    "consecutive_frames": False,
}


def parse_angle_list(angle_str: str) -> list[float]:
    """Parse comma-separated angles in degrees, preserving order and uniqueness."""
    if not angle_str.strip():
        return [0.0]

    out = []
    seen = set()
    for token in angle_str.split(","):
        angle = float(token.strip())
        key = round(angle, 8)
        if key in seen:
            continue
        seen.add(key)
        out.append(angle)

    return out if out else [0.0]


def yaw_rotation_matrix(degrees: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Create a 3x3 yaw rotation matrix around +Z axis for an angle in degrees."""
    radians = torch.deg2rad(degrees.to(device=device, dtype=dtype))
    c = torch.cos(radians)
    s = torch.sin(radians)
    zero = torch.zeros((), device=device, dtype=dtype)
    one = torch.ones((), device=device, dtype=dtype)
    row0 = torch.stack((c, -s, zero))
    row1 = torch.stack((s, c, zero))
    row2 = torch.stack((zero, zero, one))
    return torch.stack((row0, row1, row2), dim=0)


def evaluate_pose_grid(
    model: SnapViT,
    ugv_data: dict[str, torch.Tensor],
    uav_data: dict[str, torch.Tensor],
    grid_range_m: float,
    grid_resolution_m: float,
    softmax_temp: float,
    yaw_candidates_deg: list[float],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Evaluate explicit camera pose hypotheses over a local XY translation grid.
    
    Returns:
        similarity_map: (grid_H, grid_W) best raw score per translation hypothesis
        probability_map: (grid_H, grid_W) temperature-softmax over best scores
        best_yaw_map_deg: (grid_H, grid_W) best yaw offset (deg) per translation hypothesis
        grid_offsets_m: (grid_W,) metric offsets in meters used for x/y grid axes
        poses: (grid_H, grid_W, 2) local (x, y) positions in meters for each grid cell
    """
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

    for iy, dy in enumerate(offsets):
        for ix, dx in enumerate(offsets):
            best_score = None
            best_angle_idx = 0

            for angle_idx, yaw_deg in enumerate(yaw_candidates_tensor):
                modified_poses = original_camera_poses.clone()
                modified_pose = camera_w2c.clone()
                
                # Apply translation
                modified_pose[0, 3] -= dx  # camera frame x (subtract for standard conventions)
                modified_pose[1, 3] -= dy  # camera frame y
                
                # Apply yaw rotation
                r_delta = yaw_rotation_matrix(yaw_deg, device, dtype)
                modified_pose[:3, :3] = torch.matmul(r_delta, modified_pose[:3, :3])
                
                modified_poses[0, 0] = modified_pose
                ugv_data["camera_poses"] = modified_poses

                with torch.no_grad():
                    ground_bev, overhead_bev, _ = model(ugv_data, uav_data)
                
                # Compute mean cosine similarity
                overhead_bev_resized = F.interpolate(
                    overhead_bev,
                    size=ground_bev.shape[2:],
                    mode="bilinear",
                    align_corners=False,
                )
                cosine_sim = F.cosine_similarity(ground_bev, overhead_bev_resized, dim=1)
                score = cosine_sim.mean()

                if best_score is None or score > best_score:
                    best_score = score
                    best_angle_idx = angle_idx

            pose_scores[iy, ix] = best_score.float()
            best_yaw_idx[iy, ix] = best_angle_idx
            
            # Store the (x, y) position for this grid cell in world coordinates
            c2w = torch.linalg.inv(camera_w2c)
            poses[iy, ix, 0] = c2w[0, 3] + dx
            poses[iy, ix, 1] = c2w[1, 3] + dy

    # Restore original pose tensor
    ugv_data["camera_poses"] = original_camera_poses

    logits = pose_scores / max(softmax_temp, 1e-6)
    probability_map = F.softmax(logits.view(-1), dim=0).view_as(logits)
    best_yaw_map_deg = yaw_candidates_tensor[best_yaw_idx]

    return pose_scores, probability_map, best_yaw_map_deg, offsets, poses


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


def compute_position_stats(
    prob_map: torch.Tensor,
    best_yaw_map_deg: torch.Tensor,
    poses: torch.Tensor,
    gt_xy_m: Tuple[float, float],
    gt_yaw_deg: float,
    yaw_candidates_deg: list[float],
    top_k: int = 5,
    distance_thresholds: list[float] = None,
) -> dict:
    """
    Compute localization accuracy statistics for a single scene.
    
    Args:
        prob_map: (grid_H, grid_W) probability map
        best_yaw_map_deg: (grid_H, grid_W) best yaw per grid cell
        poses: (grid_H, grid_W, 2) local (x, y) positions
        gt_xy_m: (gt_x, gt_y) ground truth position in meters
        gt_yaw_deg: ground truth yaw in degrees
        yaw_candidates_deg: list of candidate yaw angles tested
        top_k: number of top candidates to track
        distance_thresholds: thresholds for counting nearby predictions
    
    Returns:
        dict with statistics
    """
    if distance_thresholds is None:
        distance_thresholds = [1.0, 2.0, 3.0]
    
    gt_x, gt_y = gt_xy_m
    
    # Find top-k predictions by probability
    flat_prob = prob_map.view(-1)
    grid_w = prob_map.shape[1]
    
    vals, idxs = torch.topk(flat_prob, k=min(top_k, len(flat_prob)))
    
    stats = {
        "gt_x_m": gt_x,
        "gt_y_m": gt_y,
        "gt_yaw_deg": gt_yaw_deg,
    }
    
    # Most probable position
    best_flat_idx = int(flat_prob.argmax().item())
    best_iy = best_flat_idx // grid_w
    best_ix = best_flat_idx % grid_w
    
    pred_x_m = float(poses[best_iy, best_ix, 0].item())
    pred_y_m = float(poses[best_iy, best_ix, 1].item())
    pred_yaw_deg = float(best_yaw_map_deg[best_iy, best_ix].item())
    
    # Distance metrics for position
    distance_m = float(np.hypot(pred_x_m - gt_x, pred_y_m - gt_y))
    stats["most_probable_pred_x_m"] = pred_x_m
    stats["most_probable_pred_y_m"] = pred_y_m
    stats["most_probable_distance_m"] = distance_m
    stats["most_probable_prob"] = float(vals[0].item())
    
    # Yaw metrics
    yaw_error_deg = angular_distance(pred_yaw_deg, gt_yaw_deg)
    stats["most_probable_pred_yaw_deg"] = pred_yaw_deg
    stats["most_probable_yaw_error_deg"] = yaw_error_deg
    
    # Distance thresholds for most probable position
    for thresh in distance_thresholds:
        key = f"most_probable_within_{thresh}m"
        stats[key] = 1 if distance_m <= thresh else 0
    
    # Top-k nearest position statistics
    min_distance_m = distance_m  # Start with most probable
    nearest_prob = float(vals[0].item())
    
    for j in range(len(vals)):
        iy = int((idxs[j] // grid_w).item())
        ix = int((idxs[j] % grid_w).item())
        cand_x_m = float(poses[iy, ix, 0].item())
        cand_y_m = float(poses[iy, ix, 1].item())
        cand_yaw_deg = float(best_yaw_map_deg[iy, ix].item())
        
        cand_distance_m = float(np.hypot(cand_x_m - gt_x, cand_y_m - gt_y))
        
        if cand_distance_m < min_distance_m:
            min_distance_m = cand_distance_m
            nearest_prob = float(vals[j].item())
    
    stats["nearest_in_topk_distance_m"] = min_distance_m
    stats["nearest_in_topk_prob"] = nearest_prob
    
    for thresh in distance_thresholds:
        key = f"nearest_in_topk_within_{thresh}m"
        stats[key] = 1 if min_distance_m <= thresh else 0
    
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


def compute_aggregate_stats(rows: list[dict]) -> dict:
    """Compute aggregate statistics across all scenes."""
    if not rows:
        return {}
    
    stats = {}
    
    # Position metrics
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
    
    # Yaw metrics
    most_prob_yaw_errors = [r["most_probable_yaw_error_deg"] for r in rows]
    
    stats["most_probable_yaw_error_mean_deg"] = float(np.mean(most_prob_yaw_errors))
    stats["most_probable_yaw_error_median_deg"] = float(np.median(most_prob_yaw_errors))
    stats["most_probable_yaw_error_std_deg"] = float(np.std(most_prob_yaw_errors))
    stats["most_probable_yaw_error_max_deg"] = float(np.max(most_prob_yaw_errors))
    
    # Distance thresholds
    for thresh in [1.0, 2.0, 3.0]:
        most_prob_key = f"most_probable_within_{thresh}m"
        nearest_key = f"nearest_in_topk_within_{thresh}m"
        
        if most_prob_key in rows[0]:
            count_most_prob = sum(r[most_prob_key] for r in rows)
            count_nearest = sum(r[nearest_key] for r in rows)
            
            stats[f"most_probable_within_{thresh}m_count"] = count_most_prob
            stats[f"most_probable_within_{thresh}m_ratio"] = count_most_prob / len(rows)
            stats[f"nearest_topk_within_{thresh}m_count"] = count_nearest
            stats[f"nearest_topk_within_{thresh}m_ratio"] = count_nearest / len(rows)
    
    return stats


def main(args: argparse.Namespace) -> None:
    print(f"Using device: {CONFIG['device']}")
    os.makedirs(args.output_dir, exist_ok=True)

    image_transforms = transforms.Compose(
        [
            transforms.Resize(CONFIG["train_img_size"], antialias=True),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    depth_transforms = transforms.Compose(
        [
            transforms.Resize(CONFIG["train_img_size"], antialias=True),
            transforms.ConvertImageDtype(torch.float),
        ]
    )

    dataset = VineyardDataset(
        root_dir=args.data_root,
        config=CONFIG,
        transforms=image_transforms,
        depth_transforms=depth_transforms,
        consecutive_frames=CONFIG["consecutive_frames"],
    )
    
    if len(dataset) == 0:
        raise ValueError(f"No scenes found in {args.data_root}")

    dataloader = DataLoader(
        dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=args.shuffle,
        num_workers=args.num_workers,
    )

    model = SnapViT(CONFIG).to(CONFIG["device"])
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint file not found at {args.checkpoint}")
    model.load_state_dict(torch.load(args.checkpoint, map_location=CONFIG["device"]))
    model.eval()

    if args.num_samples <= 0:
        target_samples = len(dataset)
    else:
        target_samples = min(args.num_samples, len(dataset))

    print(f"Evaluating localization accuracy on {target_samples} samples...")
    print(f"Grid range: {args.grid_range_m}m, resolution: {args.grid_resolution_m}m")
    print(f"Top-k: {args.top_k}, softmax temperature: {args.softmax_temp}")

    yaw_candidates_deg = parse_angle_list(args.yaw_search_angles_deg)
    print(f"Yaw candidates: {yaw_candidates_deg}")

    all_scene_stats = []
    
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(dataloader, total=target_samples, desc="Evaluating")):
            if idx >= target_samples:
                break

            uav_data = {k: v.to(CONFIG["device"]) for k, v in batch["uav_data"].items()}
            ugv_data = {k: v.to(CONFIG["device"]) for k, v in batch["ugv_data"].items()}

            # Get ground truth position and yaw
            gt_x_m, gt_y_m = pose_to_local_xy(ugv_data["camera_poses"][0, 0])
            gt_yaw_deg = get_gt_yaw(ugv_data["camera_poses"][0, 0])

            # Evaluate pose grid
            sim_map, prob_map, best_yaw_map_deg, grid_offsets_m, poses = evaluate_pose_grid(
                model=model,
                ugv_data=ugv_data,
                uav_data=uav_data,
                grid_range_m=args.grid_range_m,
                grid_resolution_m=args.grid_resolution_m,
                softmax_temp=args.softmax_temp,
                yaw_candidates_deg=yaw_candidates_deg,
            )

            # Compute statistics
            scene_stats = compute_position_stats(
                prob_map=prob_map,
                best_yaw_map_deg=best_yaw_map_deg,
                poses=poses,
                gt_xy_m=(gt_x_m, gt_y_m),
                gt_yaw_deg=gt_yaw_deg,
                yaw_candidates_deg=yaw_candidates_deg,
                top_k=args.top_k,
            )
            scene_stats["scene_id"] = f"scene_{idx:04d}"
            all_scene_stats.append(scene_stats)

    # Write per-scene results
    results_csv_path = os.path.join(args.output_dir, "localization_per_scene.csv")
    write_results_csv(results_csv_path, all_scene_stats)
    print(f"Saved per-scene results: {results_csv_path}")

    # Compute and write aggregate statistics
    aggregate_stats = compute_aggregate_stats(all_scene_stats)
    
    aggregate_json_path = os.path.join(args.output_dir, "localization_aggregate_stats.json")
    with open(aggregate_json_path, "w", encoding="utf-8") as f:
        json.dump(aggregate_stats, f, indent=2)
    print(f"Saved aggregate stats: {aggregate_json_path}")

    # Print summary
    print("\n" + "="*80)
    print("LOCALIZATION EVALUATION SUMMARY")
    print("="*80)
    print(f"Number of scenes evaluated: {aggregate_stats['num_scenes']}")
    print("\nPosition Accuracy (Most Probable):")
    print(f"  Mean distance: {aggregate_stats['most_probable_distance_mean_m']:.3f} m")
    print(f"  Median distance: {aggregate_stats['most_probable_distance_median_m']:.3f} m")
    print(f"  Std deviation: {aggregate_stats['most_probable_distance_std_m']:.3f} m")
    print(f"  Max distance: {aggregate_stats['most_probable_distance_max_m']:.3f} m")
    
    print("\nPosition Accuracy (Nearest in Top-k):")
    print(f"  Mean distance: {aggregate_stats['nearest_topk_distance_mean_m']:.3f} m")
    print(f"  Median distance: {aggregate_stats['nearest_topk_distance_median_m']:.3f} m")
    print(f"  Std deviation: {aggregate_stats['nearest_topk_distance_std_m']:.3f} m")
    print(f"  Max distance: {aggregate_stats['nearest_topk_distance_max_m']:.3f} m")
    
    print("\nOrientation Accuracy (Most Probable):")
    print(f"  Mean yaw error: {aggregate_stats['most_probable_yaw_error_mean_deg']:.1f}°")
    print(f"  Median yaw error: {aggregate_stats['most_probable_yaw_error_median_deg']:.1f}°")
    print(f"  Std deviation: {aggregate_stats['most_probable_yaw_error_std_deg']:.1f}°")
    print(f"  Max yaw error: {aggregate_stats['most_probable_yaw_error_max_deg']:.1f}°")
    
    print("\nWithin Distance Thresholds (Most Probable):")
    for thresh in [1.0, 2.0, 3.0]:
        key_count = f"most_probable_within_{thresh}m_count"
        key_ratio = f"most_probable_within_{thresh}m_ratio"
        if key_count in aggregate_stats:
            count = aggregate_stats[key_count]
            ratio = aggregate_stats[key_ratio]
            print(f"  Within {thresh}m: {count}/{aggregate_stats['num_scenes']} ({ratio*100:.1f}%)")
    
    print("\nWithin Distance Thresholds (Nearest in Top-k):")
    for thresh in [1.0, 2.0, 3.0]:
        key_count = f"nearest_topk_within_{thresh}m_count"
        key_ratio = f"nearest_topk_within_{thresh}m_ratio"
        if key_count in aggregate_stats:
            count = aggregate_stats[key_count]
            ratio = aggregate_stats[key_ratio]
            print(f"  Within {thresh}m: {count}/{aggregate_stats['num_scenes']} ({ratio*100:.1f}%)")
    
    print("="*80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate localization accuracy of SnapViT.")
    parser.add_argument(
        "--data_root",
        type=str,
        default="datasets/vineyard_dataset",
        help="Path to the root of the processed dataset.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the trained model checkpoint (.pth file).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation_localization",
        help="Directory to save evaluation results.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Number of samples to evaluate; use <= 0 to evaluate the full dataset.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of dataloader workers.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle dataset before selecting evaluation samples.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of top candidates to track for 'nearest in top-k' metrics.",
    )
    parser.add_argument(
        "--softmax_temp",
        type=float,
        default=0.07,
        help="Temperature for softmax conversion of similarity scores to probabilities.",
    )
    parser.add_argument(
        "--grid_range_m",
        type=float,
        default=5.0,
        help="Pose search range in meters for both x and y around the ground truth pose.",
    )
    parser.add_argument(
        "--grid_resolution_m",
        type=float,
        default=0.25,
        help="Pose search grid resolution in meters.",
    )
    parser.add_argument(
        "--yaw_search_angles_deg",
        type=str,
        default="-90,-45,0,45,90,135,180",
        help="Comma-separated yaw offsets (degrees) tested at each (x,y) pose hypothesis.",
    )

    args = parser.parse_args()
    main(args)
