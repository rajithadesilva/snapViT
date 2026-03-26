import argparse
import csv
import json
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
from tqdm import tqdm

from dataset import VineyardDataset
from model import SnapViT

# Configuration should match training settings.
CONFIG = {
    "vit_model": "vit_small_patch16_224",
    "train_img_size": (224, 224),
    "feature_dim": 128,
    "num_ugv_views": 8,
    "grid_size": (34, 34, 8),
    "grid_resolution": 0.3,
    "batch_size": 1,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "use_depth": True,
    "depth_range": (0.0, 5.0),
    "ground_tile_size": 10.0,
    "consecutive_frames": True,
}


def compute_cosine_map(ground_bev: torch.Tensor, overhead_bev: torch.Tensor) -> torch.Tensor:
    """Compute per-pixel cosine similarity map with shape (B, H, W)."""
    overhead_bev_resized = F.interpolate(
        overhead_bev,
        size=ground_bev.shape[2:],
        mode="bilinear",
        align_corners=False,
    )
    return F.cosine_similarity(ground_bev, overhead_bev_resized, dim=1)


def scene_stats(cosine_map: torch.Tensor, validity_mask: torch.Tensor | None) -> dict:
    """Compute scalar statistics for one scene from a cosine similarity map."""
    if validity_mask is not None:
        valid = validity_mask.squeeze(1) > 0
        values = cosine_map[valid]
        valid_pixels = int(valid.sum().item())
        total_pixels = int(valid.numel())
    else:
        values = cosine_map.reshape(-1)
        valid_pixels = int(values.numel())
        total_pixels = int(values.numel())

    if values.numel() == 0:
        return {
            "valid_pixels": valid_pixels,
            "total_pixels": total_pixels,
            "valid_ratio": 0.0,
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "p10": 0.0,
            "p90": 0.0,
        }

    return {
        "valid_pixels": valid_pixels,
        "total_pixels": total_pixels,
        "valid_ratio": float(valid_pixels / max(total_pixels, 1)),
        "mean": float(values.mean().item()),
        "median": float(values.median().item()),
        "std": float(values.std(unbiased=False).item()),
        "min": float(values.min().item()),
        "max": float(values.max().item()),
        "p10": float(torch.quantile(values, 0.10).item()),
        "p90": float(torch.quantile(values, 0.90).item()),
    }


def write_scene_csv(path: str, rows: list[dict]) -> None:
    """Write per-scene metrics to CSV for easy spreadsheet inspection."""
    if not rows:
        return

    fieldnames = [
        "scenario",
        "scene_id",
        "valid_pixels",
        "total_pixels",
        "valid_ratio",
        "mean",
        "median",
        "std",
        "min",
        "max",
        "p10",
        "p90",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_summary_csv(path: str, rows: list[dict]) -> None:
    """Write one row per test-case scenario for quick benchmark comparison."""
    if not rows:
        return

    fieldnames = [
        "scenario",
        "global_mean",
        "delta_global_mean_vs_baseline",
        "global_median",
        "global_std",
        "global_min",
        "global_max",
        "global_p10",
        "global_p90",
        "total_valid_pixels",
        "uav_rotate_deg",
        "use_swapped_uav",
        "ground_camera_yaw_deg",
        "ground_image_rotate_deg",
        "use_swapped_ground",
        "num_scenes",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_angle_list(angle_str: str) -> list[float]:
    """Parse comma-separated angles string to unique float list preserving order."""
    if not angle_str.strip():
        return []

    out = []
    seen = set()
    for token in angle_str.split(","):
        angle = float(token.strip())
        key = round(angle, 8)
        if key in seen:
            continue
        seen.add(key)
        out.append(angle)
    return out


def clone_to_device(sample: dict) -> tuple[dict, dict]:
    """Move one sample to device and deep-clone tensors for safe scenario mutation."""
    uav_data = {k: v.to(CONFIG["device"]).clone() for k, v in sample["uav_data"].items()}
    ugv_data = {k: v.to(CONFIG["device"]).clone() for k, v in sample["ugv_data"].items()}
    return uav_data, ugv_data


def rotate_images_batch(images: torch.Tensor, degrees: float, fill: float = 0.0) -> torch.Tensor:
    """Rotate a tensor batch of images with shape (N, C, H, W)."""
    if abs(degrees) < 1e-8:
        return images
    original_shape = images.shape
    if images.dim() == 5:
        b, n, c, h, w = images.shape
        images = images.reshape(b * n, c, h, w)

    rotated = [
        TF.rotate(
            img,
            angle=degrees,
            interpolation=InterpolationMode.BILINEAR,
            expand=False,
            fill=fill,
        )
        for img in images
    ]
    out = torch.stack(rotated, dim=0)

    if len(original_shape) == 5:
        out = out.reshape(original_shape)
    return out


def yaw_rotation_matrix(degrees: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Create a 3x3 yaw rotation matrix around +Z axis."""
    radians = torch.deg2rad(torch.tensor(degrees, device=device, dtype=dtype))
    c = torch.cos(radians)
    s = torch.sin(radians)
    zero = torch.zeros((), device=device, dtype=dtype)
    one = torch.ones((), device=device, dtype=dtype)
    row0 = torch.stack((c, -s, zero))
    row1 = torch.stack((s, c, zero))
    row2 = torch.stack((zero, zero, one))
    return torch.stack((row0, row1, row2), dim=0)


def perturb_ground_camera_yaw(camera_poses: torch.Tensor, yaw_deg: float) -> torch.Tensor:
    """Apply a yaw perturbation to all world-to-camera rotations: R' = R_delta @ R."""
    if abs(yaw_deg) < 1e-8:
        return camera_poses
    out = camera_poses.clone()
    r_delta = yaw_rotation_matrix(yaw_deg, out.device, out.dtype)
    out[:, :, :3, :3] = torch.matmul(r_delta.view(1, 1, 3, 3), out[:, :, :3, :3])
    return out


def evaluate_one_scenario(
    model: SnapViT,
    samples: list[dict],
    scenario_name: str,
    uav_rotate_deg: float = 0.0,
    use_swapped_uav: bool = False,
    use_swapped_ground: bool = False,
    ground_camera_yaw_deg: float = 0.0,
    ground_image_rotate_deg: float = 0.0,
) -> tuple[list[dict], dict]:
    """Evaluate one scenario over all samples and return per-scene rows + summary."""
    scene_rows = []
    all_values = []

    for i in tqdm(range(len(samples)), desc=f"Scenario {scenario_name}"):
        sample = samples[i]
        uav_data, ugv_data = clone_to_device(sample)

        if use_swapped_uav and len(samples) > 1:
            swap_idx = (i + 1) % len(samples)
            swapped_uav = samples[swap_idx]["uav_data"]["uav_image"].to(CONFIG["device"]).clone()
            uav_data["uav_image"] = swapped_uav

        if use_swapped_ground and len(samples) > 1:
            swap_idx = (i + 1) % len(samples)
            ugv_data = {
                k: v.to(CONFIG["device"]).clone() for k, v in samples[swap_idx]["ugv_data"].items()
            }

        if abs(uav_rotate_deg) > 1e-8:
            uav_data["uav_image"] = rotate_images_batch(uav_data["uav_image"], uav_rotate_deg, fill=0.0)

        if abs(ground_image_rotate_deg) > 1e-8:
            ugv_data["ugv_images"] = rotate_images_batch(ugv_data["ugv_images"], ground_image_rotate_deg, fill=0.0)
            if "ugv_depths" in ugv_data and ugv_data["ugv_depths"] is not None:
                ugv_data["ugv_depths"] = rotate_images_batch(ugv_data["ugv_depths"], ground_image_rotate_deg, fill=0.0)

        if abs(ground_camera_yaw_deg) > 1e-8:
            ugv_data["camera_poses"] = perturb_ground_camera_yaw(ugv_data["camera_poses"], ground_camera_yaw_deg)

        ground_bev, overhead_bev, ground_validity = model(ugv_data, uav_data)
        cosine_map = compute_cosine_map(ground_bev, overhead_bev)

        stats = scene_stats(cosine_map, ground_validity)
        stats["scene_id"] = f"scene_{i:04d}"
        stats["scenario"] = scenario_name
        scene_rows.append(stats)

        if ground_validity is not None:
            valid = ground_validity.squeeze(1) > 0
            values = cosine_map[valid]
        else:
            values = cosine_map.reshape(-1)

        if values.numel() > 0:
            all_values.append(values.detach().cpu())

    summary = {
        "scenario": scenario_name,
        "num_scenes": len(scene_rows),
        "uav_rotate_deg": float(uav_rotate_deg),
        "use_swapped_uav": bool(use_swapped_uav),
        "use_swapped_ground": bool(use_swapped_ground),
        "ground_camera_yaw_deg": float(ground_camera_yaw_deg),
        "ground_image_rotate_deg": float(ground_image_rotate_deg),
    }

    if all_values:
        concat = torch.cat(all_values)
        summary.update(
            {
                "global_mean": float(concat.mean().item()),
                "global_median": float(concat.median().item()),
                "global_std": float(concat.std(unbiased=False).item()),
                "global_min": float(concat.min().item()),
                "global_max": float(concat.max().item()),
                "global_p10": float(torch.quantile(concat, 0.10).item()),
                "global_p90": float(torch.quantile(concat, 0.90).item()),
                "total_valid_pixels": int(concat.numel()),
            }
        )
    else:
        summary.update(
            {
                "global_mean": 0.0,
                "global_median": 0.0,
                "global_std": 0.0,
                "global_min": 0.0,
                "global_max": 0.0,
                "global_p10": 0.0,
                "global_p90": 0.0,
                "total_valid_pixels": 0,
            }
        )

    return scene_rows, summary


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

    print(f"Preparing {target_samples} samples for scenario evaluation...")

    samples: list[dict] = []
    iterator = iter(dataloader)
    for _ in tqdm(range(target_samples), desc="Loading samples"):
        try:
            batch = next(iterator)
        except StopIteration:
            break
        samples.append(batch)

    if not samples:
        raise RuntimeError("No samples were loaded from dataset. Check data_root and dataset content.")

    scenarios = [
        {
            "name": "baseline",
            "uav_rotate_deg": 0.0,
            "use_swapped_uav": False,
            "ground_camera_yaw_deg": 0.0,
            "ground_image_rotate_deg": 0.0,
        }
    ]

    if args.run_uav_negative:
        scenarios.append(
            {
                "name": "uav_negative",
                "uav_rotate_deg": args.uav_negative_rotate_deg,
                "use_swapped_uav": args.uav_negative_swap,
                "ground_camera_yaw_deg": 0.0,
                "ground_image_rotate_deg": 0.0,
            }
        )

    if args.run_ground_negative:
        scenarios.append(
            {
                "name": "ground_negative",
                "uav_rotate_deg": 0.0,
                "use_swapped_uav": False,
                "use_swapped_ground": False,
                "ground_camera_yaw_deg": args.ground_camera_yaw_deg,
                "ground_image_rotate_deg": args.ground_image_rotate_deg,
            }
        )

    if args.run_ground_benchmark:
        angles = parse_angle_list(args.ground_benchmark_angles)

        # Rotation-only cases (ground images)
        for angle in angles:
            scenarios.append(
                {
                    "name": f"ground_rot_{angle:g}",
                    "uav_rotate_deg": 0.0,
                    "use_swapped_uav": False,
                    "use_swapped_ground": False,
                    "ground_camera_yaw_deg": 0.0,
                    "ground_image_rotate_deg": angle,
                }
            )

        # Swap-only case (ground sample replaced by another scene)
        if args.include_ground_swap_case:
            scenarios.append(
                {
                    "name": "ground_swap",
                    "uav_rotate_deg": 0.0,
                    "use_swapped_uav": False,
                    "use_swapped_ground": True,
                    "ground_camera_yaw_deg": 0.0,
                    "ground_image_rotate_deg": 0.0,
                }
            )

            # Rotation + swap combined cases
            if args.include_ground_swap_rotated_cases:
                for angle in angles:
                    scenarios.append(
                        {
                            "name": f"ground_swap_rot_{angle:g}",
                            "uav_rotate_deg": 0.0,
                            "use_swapped_uav": False,
                            "use_swapped_ground": True,
                            "ground_camera_yaw_deg": 0.0,
                            "ground_image_rotate_deg": angle,
                        }
                    )

    all_summaries = []
    all_scene_rows = []
    baseline_global_mean = None

    with torch.no_grad():
        for scenario in scenarios:
            scene_rows, summary = evaluate_one_scenario(
                model=model,
                samples=samples,
                scenario_name=scenario["name"],
                uav_rotate_deg=scenario["uav_rotate_deg"],
                use_swapped_uav=scenario["use_swapped_uav"],
                use_swapped_ground=scenario.get("use_swapped_ground", False),
                ground_camera_yaw_deg=scenario["ground_camera_yaw_deg"],
                ground_image_rotate_deg=scenario["ground_image_rotate_deg"],
            )
            summary["requested_samples"] = int(target_samples)
            summary["checkpoint"] = args.checkpoint
            summary["data_root"] = args.data_root
            all_summaries.append(summary)
            all_scene_rows.extend(scene_rows)

            if scenario["name"] == "baseline":
                baseline_global_mean = summary["global_mean"]

    if baseline_global_mean is not None:
        for summary in all_summaries:
            summary["delta_global_mean_vs_baseline"] = float(summary["global_mean"] - baseline_global_mean)

    summary_path = os.path.join(args.output_dir, "cosine_summary.json")
    scenes_json_path = os.path.join(args.output_dir, "cosine_per_scene.json")
    scenes_csv_path = os.path.join(args.output_dir, "cosine_per_scene.csv")
    benchmark_csv_path = os.path.join(args.output_dir, "cosine_benchmark.csv")

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2)
    with open(scenes_json_path, "w", encoding="utf-8") as f:
        json.dump(all_scene_rows, f, indent=2)
    write_scene_csv(scenes_csv_path, all_scene_rows)
    write_summary_csv(benchmark_csv_path, all_summaries)

    print(f"Saved summary: {summary_path}")
    print(f"Saved per-scene JSON: {scenes_json_path}")
    print(f"Saved per-scene CSV: {scenes_csv_path}")
    print(f"Saved benchmark CSV: {benchmark_csv_path}")

    print("\nScenario benchmark (global_mean and delta vs baseline):")
    for s in all_summaries:
        print(
            f"  - {s['scenario']}: "
            f"mean={s['global_mean']:.4f}, "
            f"delta={s['delta_global_mean_vs_baseline']:.4f}, "
            f"ground_rot={s['ground_image_rotate_deg']}, "
            f"ground_swap={s.get('use_swapped_ground', False)}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SnapViT cosine similarity metrics.")
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
        default="evaluation",
        help="Directory to save metric outputs.",
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
        "--run_uav_negative",
        action="store_true",
        help="Run negative UAV test by swapping and/or rotating UAV image.",
    )
    parser.add_argument(
        "--uav_negative_swap",
        action="store_true",
        help="In UAV negative test, replace UAV image with next scene UAV image.",
    )
    parser.add_argument(
        "--uav_negative_rotate_deg",
        type=float,
        default=0.0,
        help="In UAV negative test, rotate UAV image by this angle in degrees.",
    )
    parser.add_argument(
        "--run_ground_negative",
        action="store_true",
        help="Run negative ground-camera test by perturbing camera yaw and/or rotating UGV inputs.",
    )
    parser.add_argument(
        "--ground_camera_yaw_deg",
        type=float,
        default=0.0,
        help="In ground negative test, perturb camera world-to-camera rotation by yaw degrees.",
    )
    parser.add_argument(
        "--ground_image_rotate_deg",
        type=float,
        default=0.0,
        help="In ground negative test, rotate UGV RGB/depth images by this angle in degrees.",
    )
    parser.add_argument(
        "--run_ground_benchmark",
        action="store_true",
        help="Run a benchmark sweep for ground-image negative tests across multiple angles.",
    )
    parser.add_argument(
        "--ground_benchmark_angles",
        type=str,
        default="45,90,135,180",
        help="Comma-separated angles for ground-image benchmark sweep.",
    )
    parser.add_argument(
        "--include_ground_swap_case",
        action="store_true",
        help="Include ground-swap-only scenario in benchmark.",
    )
    parser.add_argument(
        "--include_ground_swap_rotated_cases",
        action="store_true",
        help="Include combined ground swap + rotated image scenarios in benchmark.",
    )

    args = parser.parse_args()
    main(args)
