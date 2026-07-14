import argparse
import torch
import os
import numpy as np
from pathlib import Path
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import sys

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data.dataset import VineyardDataset
from matplotlib.patches import Circle, FancyArrow
# --- Configuration ---
CONFIG = {
    'data_root': '/media/hdd/ale_navone/GAIA/tempovine/dataset_tempovine_new', # Path to the dataset root directory
    'train_img_size': (224, 224),  # Use None to avoid resizing in this script
    'num_ugv_views': 1,
    'grid_size': (34, 34, 8), # parameter to be removed later
    'grid_resolution': 0.3, # meters per grid cell to be removed later
    'batch_size': 1, # Adjust based on your GPU memory
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'use_depth': True,
    'tile_ground_size': 10.0, # meters
    'depth_range': (0.0, 5.0), # meters
    'voxelize': False,
    'plot_colors': True,
    'scene_output_dir': 'visualizations/dataset_samples_visualizations',  # Directory to save visualizations
    'num_samples': 20, # Number of samples to visualize
    'consecutive_frames': False, # Whether to select consecutive frames for UGV views
}

def main():
    parser = argparse.ArgumentParser(description="Visualize dataset samples using UAV and UGV images.")
    parser.add_argument('--data_root', type=str, default=CONFIG['data_root'], help="Path to the root of the processed dataset (e.g., 'vineyard_dataset').")
    parser.add_argument('--scene_output_dir', type=str, default=CONFIG['scene_output_dir'], help="Directory to save the output visualizations.")
    parser.add_argument('--num_samples', type=int, default=CONFIG['num_samples'], help="Number of samples to visualize.")
    parser.add_argument('--show_z_degrees', action='store_true', default=False, help="Show camera z-axis rotation degrees in the image title.")
    args = parser.parse_args()
    print(f"Using device: {CONFIG['device']}")
    
    # --- Data ---
    image_transforms = transforms.Compose([
        transforms.Resize(CONFIG['train_img_size'], antialias=True) if CONFIG['train_img_size'] is not None else transforms.Lambda(lambda x: x),
        transforms.ConvertImageDtype(torch.float),
    ])
    depth_transforms = transforms.Compose([
        transforms.Resize(CONFIG['train_img_size'], antialias=True) if CONFIG['train_img_size'] is not None else transforms.Lambda(lambda x: x),
        transforms.ConvertImageDtype(torch.float),
    ])
    # Create dataset and dataloader

    plot_colors = CONFIG['plot_colors']
    voxelize = CONFIG['voxelize']
    output_dir = CONFIG['scene_output_dir']

    full_dataset = VineyardDataset(root_dir=CONFIG['data_root'], config=CONFIG, transforms=image_transforms, depth_transforms=depth_transforms, consecutive_frames=CONFIG['consecutive_frames'])

    if len(full_dataset) == 0:
        print("No data found in the specified data root.")
        return
    tile_ground_size = CONFIG.get('tile_ground_size', 10.0)

    dataloader = DataLoader(full_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=4)

    # Visualize a few samples without resetting the dataloader iterator every loop.
    for i, data in enumerate(dataloader):
        if i >= CONFIG['num_samples']:
            break
        print(f"Visualizing sample {i}")
        data = {k: {kk: vv.to(CONFIG['device']) for kk, vv in v.items()} for k, v in data.items()}
        uav_img = data['uav_data']['uav_image'][0]
        ugv_imgs = data['ugv_data']['ugv_images'][0]
        ugv_depths = data['ugv_data']['ugv_depths'][0]
        camera_intrinsics = data['ugv_data']['intrinsics'][0]
        camera_poses_w2c = data['ugv_data']['camera_poses'][0]
        
        visualize_data(uav_img,
                        ugv_imgs,
                        ugv_depths,
                        camera_intrinsics,
                        camera_poses_w2c, 
                        depth_range=CONFIG['depth_range'], 
                        tile_ground_size=tile_ground_size, 
                        id=i, 
                        plot_colors=plot_colors,
                        voxelize=voxelize,
                        scene_output_dir=output_dir,
                        show_z_degrees=args.show_z_degrees
                        )

    if len(full_dataset) < CONFIG['num_samples']:
        print(f"Requested {CONFIG['num_samples']} samples but dataset has {len(full_dataset)} scenes.")

def visualize_data(uav_img, ugv_imgs, ugv_depths, 
                   camera_intrinsics, camera_poses_w2c, 
                   depth_range, tile_ground_size, id, plot_colors, voxelize=True, scene_output_dir=".", show_z_degrees=False
                   ):

    # Convert tensors to numpy
    ugv_imgs = ugv_imgs.cpu().numpy()
    ugv_depths = ugv_depths.cpu().numpy()
    camera_intrinsics = np.array(camera_intrinsics.cpu().numpy())
    camera_poses_w2c = np.array(camera_poses_w2c.cpu().numpy())
    uav_img_np = uav_img.cpu().numpy().transpose(1, 2, 0)  # HWC
    
    # Get image dimensions and pixels per meter
    img_height, img_width = uav_img_np.shape[:2]
    pixels_per_meter = img_width / tile_ground_size
    center_x_px = img_width / 2
    center_y_px = img_height / 2


    # Plot UAV image in background with alpha
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(uav_img_np, alpha=0.50)
    arrows = []
    z_degrees_list = []  # Store z-axis rotation degrees if requested

    # Main loop over UGV images
    for i in range(len(ugv_imgs)):
        w2c_matrix = camera_poses_w2c[i]

        # Try inverting pose
        try:
            c2w_matrix = np.linalg.inv(w2c_matrix)
        except np.linalg.LinAlgError:
            print(f"[WARN] Cannot invert pose {i}, skipping")
            continue
        
        # Load UGV images
        color_image = ugv_imgs[i].transpose(1, 2, 0)
        depth = ugv_depths[i]
        if depth.shape[0] == 3:
            depth = depth[0]
        else:
            depth = depth.squeeze()

        depth = depth*65535.0/1000  # assuming depth was normalized to [0,1] on uint16 range during loading

        intr = camera_intrinsics[i]
        fx, fy = intr[0, 0], intr[1, 1]
        cx, cy = intr[0, 2], intr[1, 2]

        height, width = depth.shape

        mask = (depth > depth_range[0]) & (depth < depth_range[1])


        # Pixel grid
        u_coords, v_coords = np.meshgrid(np.arange(width), np.arange(height))

        # Flatten and apply mask
        u_f = u_coords.flatten()[mask.flatten()]
        v_f = v_coords.flatten()[mask.flatten()]
        z_f = depth.flatten()[mask.flatten()]

        # Backproject to camera frame
        Xc = (u_f - cx) * z_f / fx
        Yc = (v_f - cy) * z_f / fy
        Zc = z_f

        ## Fix OpenCV → world-aligned camera frame
        #Yc_fixed = -Yc  # flip vertical axis

        # Build homogeneous coordinates
        pts_cam = np.vstack((Xc, Yc, Zc, np.ones_like(Zc)))

        pts_world = c2w_matrix @ pts_cam
        Xw = pts_world[0]
        Yw = pts_world[1]
        Zw = pts_world[2]

        px = (Xw * pixels_per_meter) + center_x_px
        py = -(Yw * pixels_per_meter) + center_y_px

        if voxelize == True:
            px = np.round(px)
            py = np.round(py)

        colors = color_image[v_f, u_f]

        # Filter inside bounds
        #inside = (pxs >= 0) & (pxs < img_width) & (pys >= 0) & (pys < img_height)

        # Draw point cloud as large matplotlib scatter dots
        if plot_colors:
            plt.scatter(px, py, c=colors, s=1.5, marker='.', linewidths=2.)
        else:
            sc = plt.scatter(px, py, c=Zw, s=1.5, marker='.', linewidths=0, cmap='viridis', vmin=-1., vmax=2.)

        # Extract UGV position (x,y)
        local_x, local_y, local_z = c2w_matrix[:3, 3]

        ugv_px = center_x_px + (local_x * pixels_per_meter)
        ugv_py = center_y_px - (local_y * pixels_per_meter)

        # Direction of travel = camera Z axis in world
        
        r = R.from_matrix(c2w_matrix[:3, :3])
        _, _, yaw = r.as_euler('xyz', degrees=True)
        if show_z_degrees:
            z_degrees_list.append(yaw)
        dx = -np.sin(np.deg2rad(yaw))
        dy = np.cos(np.deg2rad(yaw))
        norm = np.sqrt(dx*dx + dy*dy)
        if norm == 0:
            print(f"[WARN] Zero direction vector for pose {i}, skipping")
            continue
        dx /= norm
        dy /= norm

        # Arrow properties
        arrow_length = 0.02*img_width  # 2% of image width
        end_x = ugv_px + dx * arrow_length
        end_y = ugv_py - dy * arrow_length
        arrows.append([ugv_px, ugv_py, end_x, end_y, yaw])

    # Draw UGV positions
    for arrow in arrows:
        ugv_px, ugv_py, end_x, end_y, yaw = arrow
        ax.add_patch(FancyArrow(
            ugv_px, ugv_py,
            end_x - ugv_px, end_y - ugv_py,
            width=0.005*img_width,  # 0.2% of image width
            color='red'
        ))
            # Draw UGV position circle
        ax.add_patch(Circle((ugv_px, ugv_py), radius=0.005*img_width,
                            color='yellow', ec='black', lw=2))
        label_x = end_x + 0.01 * img_width
        label_y = end_y - 0.01 * img_height
        ax.text(
            label_x,
            label_y,
            f"{yaw:.1f} deg",
            color='white',
            fontsize=9,
            fontweight='bold',
            bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=1.0)
        )
    
    if not plot_colors: 
        plt.colorbar(sc, ax=ax, label='Height (m)')

    title = "UAV Image with UGV Positions + Points"
    if show_z_degrees and z_degrees_list:
        avg_z_deg = np.mean(z_degrees_list)
        title += f" (Z-axis: {avg_z_deg:.1f}°)"
    ax.set_title(title)
    ax.set_xlim(0, img_width)
    ax.set_ylim(img_height, 0)  # invert Y axis so it matches image coordinates
    ax.set_aspect('equal')
    plt.tight_layout()

    # Save the figure to an outputs directory with a timestamped filename
    out_dir = scene_output_dir if scene_output_dir else "outputs"
    os.makedirs(out_dir, exist_ok=True)
    fname = f"sample_{id:04d}_rgb_proj.png"
    save_path = os.path.join(out_dir, fname)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved visualization to: {save_path}")

if __name__ == "__main__":
    main()
