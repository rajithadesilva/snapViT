import argparse
import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from scipy.spatial.transform import Rotation as R

from dataset import VineyardDataset
from model import SnapViT
from visualize_dataset_samples import visualize_data


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
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
	"""
	Evaluate explicit camera pose hypotheses over a local XY translation grid.

	For each candidate translation offset (dx, dy), only the translation component of
	the first UGV camera pose is modified. The model is run and a scalar score is
	computed as the mean cosine similarity between the generated ground and overhead
	BEV features.

	For each translation hypothesis, a set of yaw offsets is also tested. The best
	yaw (highest score) is retained per grid cell.

	Returns:
		similarity_map: (grid_H, grid_W) best raw score per translation hypothesis
		probability_map: (grid_H, grid_W) temperature-softmax over best scores
		best_yaw_map_deg: (grid_H, grid_W) best yaw offset (deg) per translation hypothesis
		grid_offsets_m: (grid_W,) metric offsets in meters used for x/y grid axes
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
				candidate_w2c = camera_w2c.clone()
				candidate_c2w = torch.linalg.inv(candidate_w2c)
				candidate_c2w[0, 3] = dx
				candidate_c2w[1, 3] = dy
				
				#candidate_pose = original_pose.clone()
				#candidate_pose[0, 3] = dx
				#candidate_pose[2, 3] = dy
				r_delta = yaw_rotation_matrix(yaw_deg, device=device, dtype=dtype)
				candidate_c2w[:3, :3] = r_delta
				#candidate_c2w[:3, :3] = torch.matmul(r_delta, candidate_c2w[:3, :3])

				candidate_camera_poses = original_camera_poses.clone()
				candidate_camera_poses[0, 0] = torch.linalg.inv(candidate_c2w)
				ugv_data["camera_poses"] = candidate_camera_poses

				ground_bev, overhead_bev, validity_mask = model(ugv_data, uav_data)
				overhead_bev = F.interpolate(
					overhead_bev,
					size=ground_bev.shape[2:],
					mode="nearest",
					#align_corners=False,
				)

				if validity_mask is not None:
					ground_bev = ground_bev * validity_mask

				ground_bev_flat = ground_bev.view(ground_bev.shape[1], -1)
				overhead_bev_flat = overhead_bev.view(overhead_bev.shape[1], -1)
				g = F.normalize(ground_bev_flat, dim=1)
				o = F.normalize(overhead_bev_flat, dim=1)

				similarity = F.cosine_similarity(g, o, dim=0)
				similarity = similarity.view(ground_bev.shape[2], ground_bev.shape[3])

				if validity_mask is not None:
					valid_mask = validity_mask[0, 0] > 0
					valid_similarity = similarity[valid_mask]
					score = valid_similarity.mean() if valid_similarity.numel() > 0 else torch.tensor(0.0, device=device)
				else:
					score = similarity.mean()

				if best_score is None or score > best_score:
					best_score = score
					best_angle_idx = angle_idx
				
				poses[iy, ix] = torch.tensor([dx, dy], device=device, dtype=dtype)

			pose_scores[iy, ix] = best_score.float()
			best_yaw_idx[iy, ix] = best_angle_idx

	flat_best_idx = int(pose_scores.argmax().item())
	best_iy = flat_best_idx // grid_w
	best_ix = flat_best_idx % grid_w
	best_angle_deg = float(yaw_candidates_tensor[best_yaw_idx[best_iy, best_ix]].item())
	print(
		f"Highest pose score: {pose_scores.max().item():.4f} "
		f"at offset (dx={offsets[best_ix].item():.2f}, dy={offsets[best_iy].item():.2f}, yaw={best_angle_deg:.1f}deg)"
	)

	# Restore original pose tensor in ugv_data for downstream logic.
	ugv_data["camera_poses"] = original_camera_poses

	#logits = pose_scores/max(softmax_temp, 1e-6)
	#likelihoods = torch.exp(logits)
	#probability_map = likelihoods
	logits = pose_scores / max(softmax_temp, 1e-6)
	probability_map = F.softmax(logits.view(-1), dim=0).view_as(logits)
	best_yaw_map_deg = yaw_candidates_tensor[best_yaw_idx]

	return pose_scores, probability_map, best_yaw_map_deg, offsets, poses


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


def pose_to_local_xy(camera_pose_w2c: torch.Tensor) -> Tuple[float, float]:
	"""Extract camera local (x, y) position in meters from a world-to-camera matrix."""
	w2c = camera_pose_w2c.detach().cpu().numpy()
	c2w = np.linalg.inv(w2c)
	return float(c2w[0, 3]), float(c2w[1, 3])


def local_xy_to_pixel(
	local_x_m: float,
	local_y_m: float,
	image_wh: Tuple[int, int],
	tile_ground_size_m: float,
) -> Tuple[float, float]:
	"""Convert local metric coordinates (meters) to UAV image pixel coordinates."""
	width, height = image_wh
	pixels_per_meter = width / float(tile_ground_size_m)
	center_x = width / 2.0
	center_y = height / 2.0
	x_px = center_x + local_x_m * pixels_per_meter
	y_px = center_y - local_y_m * pixels_per_meter
	return x_px, y_px


def render_scene(
	scene_name: str,
	out_dir: str,
	uav_img: np.ndarray,
	ugv_img: np.ndarray,
	sim_map: torch.Tensor,
	prob_map: torch.Tensor,
	best_yaw_map_deg: torch.Tensor,
	grid_offsets_m: torch.Tensor,
	poses: torch.Tensor,
	base_local_xy_m: Tuple[float, float],
	tile_ground_size_m: float,
	gt_xy_dir: Tuple[float, float, float, float],
	top_k: int,
	angle_arrow_stride: int,
	angle_arrow_len_px: float,
) -> None:
	"""Save a visualization with UAV overlay, heatmap, top-k predictions and GT pose."""
	os.makedirs(out_dir, exist_ok=True)

	h_uav, w_uav = uav_img.shape[:2]
	sim_up = F.interpolate(sim_map.unsqueeze(0).unsqueeze(0), size=(h_uav, w_uav), mode='nearest')  # mode="bilinear", align_corners=False)
	prob_up = F.interpolate(prob_map.unsqueeze(0).unsqueeze(0), size=(h_uav, w_uav), mode="nearest") # mode="bilinear", align_corners=False)
	sim_up = sim_up.squeeze().detach().cpu().numpy()
	prob_up = prob_up.squeeze().detach().cpu().numpy()

	flat = prob_map.view(-1)
	k = min(top_k, int(flat.numel()))
	vals, idxs = torch.topk(flat, k=k)
	offsets_np = grid_offsets_m.detach().cpu().numpy()
	grid_w = prob_map.shape[1]
	base_x_m, base_y_m = base_local_xy_m
	pred_xy = []
	for j in range(k):
		iy = int((idxs[j] // grid_w).item())
		ix = int((idxs[j] % grid_w).item())
		local_x_m = float(poses[iy, ix, 0].item())
		local_y_m = float(poses[iy, ix, 1].item())
		x_px, y_px = local_xy_to_pixel(
			local_x_m,
			local_y_m,	
			(w_uav, h_uav),
			tile_ground_size_m,
		)
		pred_xy.append((x_px, y_px, float(vals[j].item())))

	gt_x, gt_y, dir_x, dir_y = gt_xy_dir

	fig = plt.figure(figsize=(18, 5))
	ax0 = fig.add_subplot(1, 4, 1)
	ax1 = fig.add_subplot(1, 4, 2)
	ax2 = fig.add_subplot(1, 4, 3)
	ax3 = fig.add_subplot(1, 4, 4)

	ax0.set_title("UGV Query (single image)")
	ax0.imshow(ugv_img)
	ax0.axis("off")

	ax1.set_title("Cosine Similarity + Predictions")
	ax1.imshow(uav_img)
	sim_up_inverse_y = np.flipud(sim_up)
	hm = ax1.imshow(sim_up_inverse_y, cmap="plasma", alpha=0.6, vmin=0.0, vmax=1.0)
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
	prob_up_inverse_y = np.flipud(prob_up)
	im2 = ax2.imshow(prob_up_inverse_y, cmap="magma", vmin=0.0, vmax=max(float(prob_up.max()), 1e-6))
	ax2.set_xlim(0, w_uav)
	ax2.set_ylim(h_uav, 0)
	fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

	ax3.set_title("Best Yaw per Grid Point")
	ax3.imshow(uav_img, alpha=0.6)
	ax3.imshow(prob_up_inverse_y, cmap="magma", alpha=0.30, vmin=0.0, vmax=max(float(prob_up.max()), 1e-6))

	best_yaw_np = best_yaw_map_deg.detach().cpu().numpy()
	prob_np = prob_map.detach().cpu().numpy()
	grid_h, grid_w = best_yaw_np.shape
	step = max(1, int(angle_arrow_stride))
	max_prob = float(prob_np.max()) if prob_np.size > 0 else 1.0

	for iy in range(0, grid_h, step):
		for ix in range(0, grid_w, step):
			local_x_m = float(poses[iy, ix, 0])
			local_y_m = float(poses[iy, ix, 1])
			x_px, y_px = local_xy_to_pixel(
				local_x_m,
				local_y_m,
				(w_uav, h_uav),
				tile_ground_size_m,
			)
			angle_deg = float(best_yaw_np[iy, ix])
			angle_rad = np.deg2rad(angle_deg)
			arrow_dx = -np.sin(angle_rad)
			arrow_dy = np.cos(angle_rad)
			conf = float(prob_np[iy, ix] / max(max_prob, 1e-8))
			color = plt.cm.viridis(conf)
			ax3.arrow(
				x_px,
				y_px,
				arrow_dx * angle_arrow_len_px,
				-arrow_dy * angle_arrow_len_px,
				width=0.8,
				head_width=4.0,
				head_length=4.0,
				color=color,
				alpha=0.95,
				length_includes_head=True,
			)

	ax3.scatter([gt_x], [gt_y], s=60, c="yellow", edgecolors="black", linewidths=1.0)
	ax3.arrow(
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
	ax3.set_xlim(0, w_uav)
	ax3.set_ylim(h_uav, 0)

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
	yaw_candidates_deg = parse_angle_list(args.yaw_search_angles_deg)

	with torch.no_grad():
		for idx, batch in enumerate(dataloader):
			if idx >= args.num_samples:
				break
			scene_name = os.path.basename(dataset.scene_folders[idx])

			uav_data = {k: v.to(CONFIG["device"]) for k, v in batch["uav_data"].items()}
			ugv_data = {k: v.to(CONFIG["device"]) for k, v in batch["ugv_data"].items()}

			sim_map, prob_map, best_yaw_map_deg, grid_offsets_m, poses = evaluate_pose_grid(
				model=model,
				ugv_data=ugv_data,
				uav_data=uav_data,
				grid_range_m=args.grid_range_m,
				grid_resolution_m=args.grid_resolution_m,
				softmax_temp=args.softmax_temp,
				yaw_candidates_deg=yaw_candidates_deg,
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
					show_z_degrees=True
				)

			# For now use GT orientation directly from first UGV camera pose.
			tile_size = float(ugv_data.get("ground_tile_size", torch.tensor([CONFIG["ground_tile_size"]], device=CONFIG["device"]))[0].item())
			base_local_xy_m = pose_to_local_xy(ugv_data["camera_poses"][0, 0])
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
				best_yaw_map_deg=best_yaw_map_deg,
				grid_offsets_m=grid_offsets_m,
				poses=poses,
				base_local_xy_m=base_local_xy_m,
				tile_ground_size_m=tile_size,
				gt_xy_dir=gt_xy_dir,
				top_k=args.top_k,
				angle_arrow_stride=args.angle_arrow_stride,
				angle_arrow_len_px=args.angle_arrow_len_px,
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
	parser.add_argument("--grid_range_m", type=float, default=5.0, help="Pose search range in meters for both x and y around the base pose.")
	parser.add_argument("--grid_resolution_m", type=float, default=0.25, help="Pose search grid resolution in meters.")
	parser.add_argument(
		"--yaw_search_angles_deg",
		type=str,
		default="-90,-45,0,45,90,135,180",
		help="Comma-separated yaw offsets (degrees) tested at each (x,y) pose hypothesis.",
	)
	parser.add_argument("--angle_arrow_stride", type=int, default=0, help="Stride for plotting best-angle arrows on the yaw map.")
	parser.add_argument("--angle_arrow_len_px", type=float, default=9.0, help="Arrow length in pixels for the best-angle map.")
	parser.add_argument("--sample_visualization", action=argparse.BooleanOptionalAction, default=True, help="Save sample projection visualization by calling visualize_data.")
	parser.add_argument("--num_workers", type=int, default=2, help="Dataloader workers.")

	main(parser.parse_args())
