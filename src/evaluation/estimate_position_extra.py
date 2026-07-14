import argparse
import os
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from scipy.spatial.transform import Rotation as R
import sys

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data.dataset import VineyardDataset
from models.snapvit import SnapViT
from visualization.visualize_dataset_samples import visualize_data


CONFIG = {
	"vit_model": "vit_small_patch16_224",
	"train_img_size": (224, 224),
	"feature_dim": 128,
	"num_ugv_views": 1,
	"grid_size": (34, 34, 8),
	"grid_resolution": 0.3,
	"batch_size": 1,
	"device": "cuda" if torch.cuda.is_available() else "cpu",
	"use_depth": True,
	"depth_range": (0.0, 5.0),
	"ground_tile_size": 10.0,
	"consecutive_frames": False,
}


def denormalize_image(img_chw: torch.Tensor) -> np.ndarray:
	"""Convert normalized CHW tensor to uint8 HWC image for plotting."""
	mean = torch.tensor([0.485, 0.456, 0.406], device=img_chw.device, dtype=img_chw.dtype).view(3, 1, 1)
	std = torch.tensor([0.229, 0.224, 0.225], device=img_chw.device, dtype=img_chw.dtype).view(3, 1, 1)
	img = img_chw * std + mean
	img = img.clamp(0.0, 1.0)
	return (img.permute(1, 2, 0).detach().cpu().numpy() * 255.0).astype(np.uint8)


def denormalize_tensor(img_chw: torch.Tensor) -> torch.Tensor:
	"""Convert normalized CHW tensor back to [0,1] float CHW tensor."""
	mean = torch.tensor([0.485, 0.456, 0.406], device=img_chw.device, dtype=img_chw.dtype).view(3, 1, 1)
	std = torch.tensor([0.229, 0.224, 0.225], device=img_chw.device, dtype=img_chw.dtype).view(3, 1, 1)
	img = img_chw * std + mean
	return img.clamp(0.0, 1.0)


def cosine_grid_match(
	ground_bev: torch.Tensor,
	overhead_bev: torch.Tensor,
	validity_mask: torch.Tensor | None,
	softmax_temp: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
	"""
	Build a UAV-grid match map by comparing a single ground descriptor to each overhead BEV cell.

	Returns:
		similarity_map: (H, W) cosine similarity in [-1, 1]
		probability_map: (H, W) softmax over valid cells
	"""
	# Use one sample because this script runs with batch_size=1.
	g = ground_bev[0]  # (C, H, W)
	o = overhead_bev[0]  # (C, H, W)

	g_norm = F.normalize(g, dim=0)
	o_norm = F.normalize(o, dim=0)

	if validity_mask is not None:
		valid = validity_mask[0] > 0
		if valid.dim() == 3:
			valid = valid.squeeze(0)
	else:
		valid = torch.ones(g.shape[1:], dtype=torch.bool, device=g.device)

	# Average only valid ground cells into a single descriptor.
	if valid.any():
		ground_desc = g_norm[:, valid].mean(dim=1)
	else:
		ground_desc = g_norm.view(g_norm.shape[0], -1).mean(dim=1)

	ground_desc = F.normalize(ground_desc, dim=0)

	similarity_map = torch.einsum("c,chw->hw", ground_desc, o_norm)

	logits = similarity_map / max(softmax_temp, 1e-6)
	masked_logits = torch.full_like(logits, -1e9)
	masked_logits[valid] = logits[valid]
	probability_map = F.softmax(masked_logits.view(-1), dim=0).view_as(masked_logits)

	return similarity_map, probability_map


def gt_pose_to_pixel(
	camera_pose_w2c: torch.Tensor,
	image_wh: Tuple[int, int],
	tile_ground_size_m: float,
) -> Tuple[float, float, float, float]:
	"""
	Convert GT pose to UAV image pixel location + forward direction (dx, dy).

	Returns:
		(x_px, y_px, dir_x, dir_y)
	"""
	w2c = camera_pose_w2c.detach().cpu().numpy()
	c2w = np.linalg.inv(w2c)

	local_x, local_y, _ = c2w[:3, 3]
	width, height = image_wh

	pixels_per_meter = width / float(tile_ground_size_m)
	center_x = width / 2.0
	center_y = height / 2.0

	x_px = center_x + local_x * pixels_per_meter
	y_px = center_y - local_y * pixels_per_meter

	# Use GT orientation from camera forward axis in world frame.
	r = R.from_matrix(c2w[:3, :3])
	_, _, yaw = r.as_euler('xyz', degrees=False)
	dx = -np.sin(yaw)
	dy = np.cos(yaw)
	norm = np.hypot(dx, dy)
	if norm > 1e-8:
		dx /= norm
		dy /= norm
	else:
		dx, dy = 0.0, 1.0

	return x_px, y_px, dx, dy


def render_scene(
	scene_name: str,
	out_dir: str,
	uav_img: np.ndarray,
	ugv_img: np.ndarray,
	sim_map: torch.Tensor,
	prob_map: torch.Tensor,
	gt_xy_dir: Tuple[float, float, float, float],
	top_k: int,
) -> None:
	"""Save a visualization with UAV overlay, heatmap, top-k predictions and GT pose."""
	os.makedirs(out_dir, exist_ok=True)

	h_uav, w_uav = uav_img.shape[:2]
	sim_up = F.interpolate(sim_map.unsqueeze(0).unsqueeze(0), size=(h_uav, w_uav), mode="bilinear", align_corners=False)
	prob_up = F.interpolate(prob_map.unsqueeze(0).unsqueeze(0), size=(h_uav, w_uav), mode="bilinear", align_corners=False)
	sim_up = sim_up.squeeze().detach().cpu().numpy()
	prob_up = prob_up.squeeze().detach().cpu().numpy()

	flat = prob_map.view(-1)
	k = min(top_k, int(flat.numel()))
	vals, idxs = torch.topk(flat, k=k)
	grid_w = prob_map.shape[1]
	pred_xy = []
	for j in range(k):
		iy = int((idxs[j] // grid_w).item())
		ix = int((idxs[j] % grid_w).item())
		x_px = (ix + 0.5) * (w_uav / prob_map.shape[1])
		y_px = (iy + 0.5) * (h_uav / prob_map.shape[0])
		pred_xy.append((x_px, y_px, float(vals[j].item())))

	gt_x, gt_y, dir_x, dir_y = gt_xy_dir

	fig = plt.figure(figsize=(14, 5))
	ax0 = fig.add_subplot(1, 3, 1)
	ax1 = fig.add_subplot(1, 3, 2)
	ax2 = fig.add_subplot(1, 3, 3)

	ax0.set_title("UGV Query (single image)")
	ax0.imshow(ugv_img)
	ax0.axis("off")

	ax1.set_title("Cosine Similarity + Predictions")
	ax1.imshow(uav_img)
	hm = ax1.imshow(sim_up, cmap="plasma", alpha=0.6, vmin=-1.0, vmax=1.0)
	for i, (x_px, y_px, p) in enumerate(pred_xy):
		ax1.scatter([x_px], [y_px], s=30, c="white", edgecolors="black", linewidths=1.0)
		ax1.text(x_px + 3, y_px - 3, f"#{i+1} ({p:.2f})", color="white", fontsize=8)

	# Ground-truth position and GT orientation arrow.
	ax1.scatter([gt_x], [gt_y], s=80, c="yellow", edgecolors="black", linewidths=1.5, label="GT position")
	arrow_len = 20.0
	ax1.arrow(
		gt_x,
		gt_y,
		dir_x * arrow_len,
		-dir_y * arrow_len,
		width=2.0,
		head_width=8.0,
		head_length=8.0,
		color="red",
		length_includes_head=True,
	)
	ax1.set_xlim(0, w_uav)
	ax1.set_ylim(h_uav, 0)
	ax1.legend(loc="lower right")
	fig.colorbar(hm, ax=ax1, fraction=0.046, pad=0.04)

	ax2.set_title("Probability Heatmap")
	im2 = ax2.imshow(prob_up, cmap="magma", vmin=0.0, vmax=max(float(prob_up.max()), 1e-6))
	ax2.set_xlim(0, w_uav)
	ax2.set_ylim(h_uav, 0)
	fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

	fig.suptitle(f"Scene: {scene_name}")
	fig.tight_layout()

	out_path = os.path.join(out_dir, f"{scene_name}_localization.png")
	fig.savefig(out_path, dpi=180)
	plt.close(fig)


def main(args: argparse.Namespace) -> None:
	print(f"Using device: {CONFIG['device']}")

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

	model = SnapViT(CONFIG).to(CONFIG["device"])
	if not os.path.exists(args.checkpoint):
		raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
	model.load_state_dict(torch.load(args.checkpoint, map_location=CONFIG["device"]))
	model.eval()

	dataloader = DataLoader(dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=args.num_workers)

	with torch.no_grad():
		for idx, batch in enumerate(dataloader):
			if idx >= args.num_samples:
				break
			scene_name = os.path.basename(dataset.scene_folders[idx])

			uav_data = {k: v.to(CONFIG["device"]) for k, v in batch["uav_data"].items()}
			ugv_data = {k: v.to(CONFIG["device"]) for k, v in batch["ugv_data"].items()}

			ground_bev, overhead_bev, validity = model(ugv_data, uav_data)
			overhead_bev = F.interpolate(
				overhead_bev,
				size=ground_bev.shape[2:],
				mode="bilinear",
				align_corners=False,
			)

			sim_map, prob_map = cosine_grid_match(
				ground_bev,
				overhead_bev,
				validity,
				softmax_temp=args.softmax_temp,
			)

			uav_img = denormalize_image(uav_data["uav_image"][0])
			ugv_img = denormalize_image(ugv_data["ugv_images"][0, 0])
			scene_number = int(''.join(filter(str.isdigit, scene_name)))
			
			if args.sample_visualization:
				sample_vis_dir = os.path.join(args.output_dir, "sample_projection")
				visualize_data(
					uav_img=denormalize_tensor(uav_data["uav_image"][0]),
					ugv_imgs=torch.stack([denormalize_tensor(x) for x in ugv_data["ugv_images"][0]], dim=0),
					ugv_depths=ugv_data["ugv_depths"][0],
					camera_intrinsics=ugv_data["intrinsics"][0],
					camera_poses_w2c=ugv_data["camera_poses"][0],
					depth_range=CONFIG["depth_range"],
					tile_ground_size=CONFIG["ground_tile_size"],
					id=scene_number,
					plot_colors=True,
					voxelize=False,
					scene_output_dir=sample_vis_dir,
				)

			# For now use GT orientation directly from first UGV camera pose.
			tile_size = float(ugv_data.get("ground_tile_size", torch.tensor([CONFIG["ground_tile_size"]], device=CONFIG["device"]))[0].item())
			gt_xy_dir = gt_pose_to_pixel(
				ugv_data["camera_poses"][0, 0],
				(uav_img.shape[1], uav_img.shape[0]),
				tile_size,
			)

			render_scene(
				scene_name=scene_name,
				out_dir=args.output_dir,
				uav_img=uav_img,
				ugv_img=ugv_img,
				sim_map=sim_map,
				prob_map=prob_map,
				gt_xy_dir=gt_xy_dir,
				top_k=args.top_k,
			)

			print(f"Saved localization plot for {scene_name}")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Estimate UGV position on UAV image using SnapViT features.")
	parser.add_argument("--data_root", type=str, required=True, help="Path to dataset root containing scene folders.")
	parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pth).")
	parser.add_argument("--output_dir", type=str, default="visualizations/estimate_position", help="Directory to save plots.")
	parser.add_argument("--num_samples", type=int, default=10, help="How many scenes to process when scene_index is not set.")
	parser.add_argument("--scene_index", type=int, default=None, help="Process one specific scene index.")
	parser.add_argument("--top_k", type=int, default=5, help="Number of top candidate locations to annotate.")
	parser.add_argument("--softmax_temp", type=float, default=0.07, help="Temperature used to convert similarity to probability.")
	parser.add_argument("--sample_visualization", action=argparse.BooleanOptionalAction, default=True, help="Save sample projection visualization by calling visualize_data.")
	parser.add_argument("--num_workers", type=int, default=2, help="Dataloader workers.")

	main(parser.parse_args())
