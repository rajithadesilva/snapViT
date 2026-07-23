from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data.dataset import VineyardDataset
from models.snapvit import SnapViT

from evaluation.common import (
    apply_checkpoint_fusion_config,
    compute_aggregate_stats,
    compute_position_stats,
    evaluate_pose_grid,
    get_gt_yaw,
    parse_angle_list,
    pose_to_local_xy,
    write_results_csv,
)


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


def main(args: argparse.Namespace) -> None:
    print(f"Using device: {CONFIG['device']}")
    os.makedirs(args.output_dir, exist_ok=True)

    image_transforms = transforms.Compose(
        [
            transforms.Resize(CONFIG["train_img_size"], antialias=True),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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

    apply_checkpoint_fusion_config(CONFIG, args.checkpoint)
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

            gt_x_m, gt_y_m = pose_to_local_xy(ugv_data["camera_poses"][0, 0])
            gt_yaw_deg = get_gt_yaw(ugv_data["camera_poses"][0, 0])

            encoded_ground = model.ground_encoder.encode_from_dict(ugv_data)
            encoded_overhead = model.overhead_encoder.encode_from_dict(uav_data)

            _, prob_map, best_yaw_map_deg, _, poses = evaluate_pose_grid(
                model=model,
                ugv_data=ugv_data,
                uav_data=uav_data,
                grid_range_m=args.grid_range_m,
                grid_resolution_m=args.grid_resolution_m,
                softmax_temp=args.softmax_temp,
                yaw_candidates_deg=yaw_candidates_deg,
                encoded_ground=encoded_ground,
                encoded_overhead=encoded_overhead,
            )

            scene_stats = compute_position_stats(
                prob_map=prob_map,
                best_yaw_map_deg=best_yaw_map_deg,
                poses=poses,
                gt_xy_m=(gt_x_m, gt_y_m),
                gt_yaw_deg=gt_yaw_deg,
                yaw_candidates_deg=yaw_candidates_deg,
                top_k=args.top_k,
                distance_thresholds=args.distance_thresholds_m,
            )
            scene_stats["scene_id"] = f"scene_{idx:04d}"
            all_scene_stats.append(scene_stats)

    results_csv_path = os.path.join(args.output_dir, "localization_per_scene.csv")
    write_results_csv(results_csv_path, all_scene_stats)

    aggregate_stats = compute_aggregate_stats(all_scene_stats)
    aggregate_json_path = os.path.join(args.output_dir, "localization_aggregate_stats.json")
    with open(aggregate_json_path, "w", encoding="utf-8") as f:
        json.dump(aggregate_stats, f, indent=2)

    print("\n" + "=" * 80)
    print("LOCALIZATION EVALUATION SUMMARY")
    print("=" * 80)
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

    print("\nTop-k localization metrics:")
    for thresh in args.distance_thresholds_m:
        recall_key = f"topk_recall_within_{thresh}m_ratio"
        precision_key = f"topk_precision_within_{thresh}m_mean"
        if recall_key in aggregate_stats:
            print(f"  Recall within {thresh}m: {aggregate_stats[recall_key] * 100:.1f}%")
            print(f"  Precision within {thresh}m: {aggregate_stats[precision_key] * 100:.1f}%")

    print("=" * 80 + "\n")


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
        help="Number of top candidates to track for localization metrics.",
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
    parser.add_argument(
        "--distance_thresholds_m",
        type=float,
        nargs="+",
        default=[1.0, 2.0, 3.0],
        help="Distance thresholds used to report top-k hit and precision metrics.",
    )

    main(parser.parse_args())
