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
import logging

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data.dataset import VineyardDataset
from models.snapvit import SnapViT

from evaluation.common import (
    compute_aggregate_stats,
    compute_position_stats,
    compute_retrieval_metrics,
    evaluate_pose_grid,
    get_gt_yaw,
    parse_angle_list,
    pose_to_local_xy,
    pool_bev_embeddings,
    project_scene_features,
    write_results_csv,
)


DEFAULT_CONFIG = {
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
    "edge_margin_m": 2.5,
}


def main(args: argparse.Namespace) -> None:
    os.makedirs(args.output_dir, exist_ok=True)

    # Work on a per-run config copy to avoid mutating module globals.
    config = dict(DEFAULT_CONFIG)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )
    logger = logging.getLogger(__name__)

    logger.info(f"Evaluation output directory: {args.output_dir}")

    # quick argument validation
    if not os.path.exists(args.data_root):
        logger.error(f"Data root not found: {args.data_root}")
        raise FileNotFoundError(f"Data root not found: {args.data_root}")

    image_transforms = transforms.Compose(
        [
            transforms.Resize(config["train_img_size"], antialias=True),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    depth_transforms = transforms.Compose(
        [
            transforms.Resize(config["train_img_size"], antialias=True),
            transforms.ConvertImageDtype(torch.float),
        ]
    )

    dataset = VineyardDataset(
        root_dir=args.data_root,
        config=config,
        transforms=image_transforms,
        depth_transforms=depth_transforms,
        consecutive_frames=config["consecutive_frames"],

    )
    if len(dataset) == 0:
        raise ValueError(f"No scenes found in {args.data_root}")

    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=args.shuffle,
        num_workers=args.num_workers,
    )

    # allow overriding or auto-detecting the backbone model name
    if getattr(args, "vit_model", None):
        config["vit_model"] = args.vit_model
        logger.info(f"Overriding vit_model from CLI: {config['vit_model']}")
    else:
        # try to read a config.json next to the checkpoint
        try:
            ckpt_path = Path(args.checkpoint)
            for c in (ckpt_path.parent / "config.json", ckpt_path.parent.parent / "config.json"):
                if c.exists():
                    try:
                        with open(c, "r", encoding="utf-8") as fh:
                            cfg = json.load(fh)
                        if "vit_model" in cfg:
                            config["vit_model"] = cfg["vit_model"]
                            logger.info(f"Auto-detected vit_model='{config['vit_model']}' from {c}")
                            # also read feature_dim if available
                            if "feature_dim" in cfg:
                                config["feature_dim"] = int(cfg["feature_dim"])
                                logger.info(f"Auto-detected feature_dim={config['feature_dim']} from {c}")
                            break
                        if "model_name" in cfg:
                            config["vit_model"] = cfg["model_name"]
                            logger.info(f"Auto-detected vit_model='{config['vit_model']}' from {c}")
                            if "feature_dim" in cfg:
                                config["feature_dim"] = int(cfg["feature_dim"])
                                logger.info(f"Auto-detected feature_dim={config['feature_dim']} from {c}")
                            break
                    except Exception:
                        continue
        except Exception:
            pass

    logger.info(f"Using backbone model: {config['vit_model']}")
    model = SnapViT(config).to(config["device"])
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint file not found at {args.checkpoint}")

    # Load checkpoint with strict=True first; require explicit opt-in for non-strict load.
    checkpoint_obj = torch.load(args.checkpoint, map_location=config["device"])
    if isinstance(checkpoint_obj, dict) and "state_dict" in checkpoint_obj and isinstance(checkpoint_obj["state_dict"], dict):
        sd = checkpoint_obj["state_dict"]
    else:
        sd = checkpoint_obj

    if not isinstance(sd, dict):
        raise RuntimeError(
            f"Unsupported checkpoint format at {args.checkpoint}. Expected a state_dict or a dict containing 'state_dict'."
        )

    try:
        model.load_state_dict(sd)
        logger.info(f"Loaded checkpoint {args.checkpoint} with strict=True")
    except RuntimeError:
        if not args.allow_non_strict_checkpoint:
            raise RuntimeError(
                "Strict checkpoint load failed. Re-run with --allow_non_strict_checkpoint to proceed with partial loading."
            )

        logger.warning(
            "Strict checkpoint load failed (likely architecture mismatch). Trying non-strict load."
        )
        load_result = model.load_state_dict(sd, strict=False)
        loaded_key_count = len(sd) - len(load_result.unexpected_keys)
        if loaded_key_count <= 0:
            raise RuntimeError(
                "Non-strict checkpoint load matched zero parameters. Aborting to avoid invalid evaluation."
            )
        logger.warning(
            "Non-strict load completed. "
            f"loaded_keys={loaded_key_count}, "
            f"missing_keys={len(load_result.missing_keys)}, "
            f"unexpected_keys={len(load_result.unexpected_keys)}"
        )
    model.eval()

    if args.num_samples <= 0:
        target_samples = len(dataset)
    else:
        target_samples = min(args.num_samples, len(dataset))

    logger.info(
        f"Running evaluation on {target_samples} samples from {args.data_root} using device {config['device']}"
    )

    yaw_candidates_deg = parse_angle_list(args.yaw_search_angles_deg)
    retrieval_ks = [1, 2, 5, 10, 20]

    scene_rows = []
    ground_embeddings = []
    overhead_embeddings = []

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(dataloader, total=target_samples, desc="Evaluating")):
            if idx >= target_samples:
                break

            logger.info(f"Processing scene {idx+1}/{target_samples}")

            uav_data = {k: v.to(config["device"]) for k, v in batch["uav_data"].items()}
            ugv_data = {k: v.to(config["device"]) for k, v in batch["ugv_data"].items()}

            encoded_ground = model.ground_encoder.encode_from_dict(ugv_data)
            encoded_overhead = model.overhead_encoder.encode_from_dict(uav_data)

            logger.info("Cached encoder outputs for ground and overhead inputs")

            ground_bev, overhead_bev, ground_validity = project_scene_features(
                model=model,
                ugv_data=ugv_data,
                uav_data=uav_data,
                encoded_ground=encoded_ground,
                encoded_overhead=encoded_overhead,
            )

            ground_emb = pool_bev_embeddings(ground_bev, ground_validity)
            ground_embeddings.append(ground_emb)
            logger.debug(f"Pooled ground embedding shape: {tuple(ground_emb.shape)}")

            overhead_emb = pool_bev_embeddings(overhead_bev)
            overhead_embeddings.append(overhead_emb)
            logger.debug(f"Pooled overhead embedding shape: {tuple(overhead_emb.shape)}")

            gt_x_m, gt_y_m = pose_to_local_xy(ugv_data["camera_poses"][0, 0])
            gt_yaw_deg = get_gt_yaw(ugv_data["camera_poses"][0, 0])

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

            logger.info(f"Completed pose-grid evaluation for scene {idx+1}")

            scene_stats = compute_position_stats(
                prob_map=prob_map,
                best_yaw_map_deg=best_yaw_map_deg,
                poses=poses,
                gt_xy_m=(gt_x_m, gt_y_m),
                gt_yaw_deg=gt_yaw_deg,
                yaw_candidates_deg=yaw_candidates_deg,
                top_k=args.top_k,
                distance_thresholds=args.localization_thresholds_m,
            )
            scene_stats["scene_id"] = f"scene_{idx:04d}"
            scene_rows.append(scene_stats)

    retrieval_metrics = compute_retrieval_metrics(
        torch.cat(ground_embeddings, dim=0),
        torch.cat(overhead_embeddings, dim=0),
        ks=retrieval_ks,
    )
    localization_metrics = compute_aggregate_stats(
        scene_rows,
        distance_thresholds=args.localization_thresholds_m,
    )

    write_results_csv(os.path.join(args.output_dir, "localization_per_scene.csv"), scene_rows)
    with open(os.path.join(args.output_dir, "localization_summary.json"), "w", encoding="utf-8") as f:
        json.dump(localization_metrics, f, indent=2)

    logger.info(f"Wrote per-scene localization CSV to {os.path.join(args.output_dir, 'localization_per_scene.csv')}")
    logger.info(f"Wrote localization summary JSON to {os.path.join(args.output_dir, 'localization_summary.json')}")

    summary = {
        "retrieval": retrieval_metrics,
        "localization": localization_metrics,
        "num_samples": len(scene_rows),
        "top_k": args.top_k,
        "yaw_candidates_deg": yaw_candidates_deg,
        "localization_thresholds_m": args.localization_thresholds_m,
    }
    with open(os.path.join(args.output_dir, "evaluation_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("Retrieval metrics:")
    for k in retrieval_ks:
        logger.info(f"  Recall@{k}: {retrieval_metrics[f'recall@{k}'] * 100:.2f}%")

    logger.info("Localization metrics:")
    for thresh in args.localization_thresholds_m:
        logger.info(
            f"  Top-{args.top_k} recall within {thresh}m: "
            f"{localization_metrics.get(f'topk_recall_within_{thresh}m_ratio', 0.0) * 100:.2f}%"
        )
        logger.info(
            f"  Top-{args.top_k} precision within {thresh}m: "
            f"{localization_metrics.get(f'topk_precision_within_{thresh}m_mean', 0.0) * 100:.2f}%"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the shared SnapViT evaluation pipeline.")
    parser.add_argument("--data_root", type=str, required=True, help="Path to the processed dataset root.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the trained model checkpoint.")
    parser.add_argument("--output_dir", type=str, default="evaluation_pipeline", help="Directory for outputs.")
    parser.add_argument("--num_samples", type=int, default=0, help="Number of samples to evaluate; <=0 means full set.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of DataLoader workers.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle dataset before evaluation.")
    parser.add_argument("--vit_model", type=str, default=None, help="Override timm backbone name used for evaluation.")
    parser.add_argument(
        "--allow_non_strict_checkpoint",
        action="store_true",
        help="Allow partial checkpoint loading if strict loading fails.",
    )
    parser.add_argument("--top_k", type=int, default=5, help="Localization top-k candidates.")
    parser.add_argument("--softmax_temp", type=float, default=0.07, help="Softmax temperature for grid scoring.")
    parser.add_argument("--grid_range_m", type=float, default=5.0, help="Pose search range in meters.")
    parser.add_argument("--grid_resolution_m", type=float, default=0.25, help="Pose search resolution in meters.")
    parser.add_argument(
        "--yaw_search_angles_deg",
        type=str,
        default="-90,-45,0,45,90,135,180",
        help="Comma-separated yaw offsets tested for each pose hypothesis.",
    )
    parser.add_argument(
        "--localization_thresholds_m",
        type=float,
        nargs="+",
        default=[1.0, 2.0, 3.0],
        help="Distance thresholds used for localization recall/precision.",
    )

    main(parser.parse_args())
