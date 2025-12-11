import torch
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import numpy as np
import matplotlib.pyplot as plt

from dataset import VineyardDataset
from matplotlib.patches import Circle, FancyArrow
import time
# --- Configuration ---
CONFIG = {
    'data_root': 'data/row_1_to_6', # Path to the dataset root directory
    'vit_model': 'vit_small_patch16_224', # Use a smaller model for faster training
    'feature_dim': 128,
    'train_img_size': (224, 224),  # Use None to avoid resizing in this script
    'num_ugv_views': 16,
    'grid_size': (34, 34, 8), # Smaller grid for faster training
    'grid_resolution': 0.3, # meters per grid cell
    'batch_size': 1, # Adjust based on your GPU memory
    'learning_rate': 1e-4,
    'epochs': 1000,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'val_split_ratio': 0.2, # 20% of the data will be used for validation
    'use_depth': True,
    'tile_ground_size': 10.0, # meters
    'depth_range': (0.0, 5.0), # meters
}

def main():
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

    plot_colors = True

    full_dataset = VineyardDataset(root_dir=CONFIG['data_root'], config=CONFIG, transforms=image_transforms, depth_transforms=depth_transforms)

    if len(full_dataset) == 0:
        print("No data found in the specified data root.")
        return
    tile_ground_size = CONFIG.get('tile_ground_size', 10.0)

    dataloader = DataLoader(full_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=4)

    # Visualize a few samples
    for i, data in enumerate(dataloader):
        data = {k: {kk: vv.to(CONFIG['device']) for kk, vv in v.items()} for k, v in data.items()}
        uav_img = data['uav_data']['uav_image'][0]
        ugv_imgs = data['ugv_data']['ugv_images'][0]
        ugv_depths = data['ugv_data']['ugv_depths'][0]
        camera_intrinsics = data['ugv_data']['intrinsics'][0]
        camera_poses_w2c = data['ugv_data']['camera_poses'][0]
        
        visualize_data(uav_img, ugv_imgs, ugv_depths, camera_intrinsics, camera_poses_w2c, depth_range=CONFIG['depth_range'], tile_ground_size=tile_ground_size, id=i, plot_colors=plot_colors)

def visualize_data(uav_img, ugv_imgs, ugv_depths, 
                   camera_intrinsics, camera_poses_w2c, 
                   depth_range, tile_ground_size, id, plot_colors
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

    # Function to swap X and Y axes in a matrix
    def swipe_xy_axis(matrix):
        matrix[[0, 1]] = matrix[[1, 0]]
        return matrix
    
    elapsed_time = 0.0


    # Plot UAV image in background with alpha
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(uav_img_np, alpha=0.25)

    arrows = []

    # Main loop over UGV images
    for i in range(len(ugv_imgs)):
        start_time = time.time()
        w2c_matrix = camera_poses_w2c[i]
        w2c_matrix = swipe_xy_axis(w2c_matrix)

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
            depth = depth

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

        Xc = -Xc

        ## Fix OpenCV → world-aligned camera frame
        #Yc_fixed = -Yc  # flip vertical axis

        # Build homogeneous coordinates
        pts_cam = np.vstack((Xc, Yc, Zc, np.ones_like(Zc)))

        # Transform to world coordinates
        pts_world = c2w_matrix @ pts_cam
        Xw = pts_world[0]
        Yw = pts_world[1]
        Zw = pts_world[2]

        px = (Xw * pixels_per_meter) + center_x_px
        py = -(Yw * pixels_per_meter) + center_y_px

        colors = color_image[v_f, u_f]
        elapsed_time += time.time() - start_time

        # Filter inside bounds
        #inside = (pxs >= 0) & (pxs < img_width) & (pys >= 0) & (pys < img_height)

        # Draw point cloud as large matplotlib scatter dots
        if plot_colors:
            plt.scatter(px, py, c=colors, s=1.5, marker='.', linewidths=2)
        else:
            sc = plt.scatter(px, py, c=Zw, s=1.5, marker='.', linewidths=0, cmap='viridis', vmin=-1., vmax=2.)



        start_time = time.time()
        # Extract UGV position (x,y)
        local_x, local_y, _ = c2w_matrix[:3, 3]

        ugv_px = center_x_px + (local_x * pixels_per_meter)
        ugv_py = center_y_px - (local_y * pixels_per_meter)

        # Direction of travel = camera Z axis in world
        direction_vector = c2w_matrix[:3, 2]
        dx, dy = direction_vector[0], direction_vector[1]

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

        elapsed_time += time.time() - start_time
        arrows.append([ugv_px, ugv_py, end_x, end_y])

    # Draw UGV positions
    for arrow in arrows:
        ugv_px, ugv_py, end_x, end_y = arrow
        ax.add_patch(FancyArrow(
            ugv_px, ugv_py,
            end_x - ugv_px, end_y - ugv_py,
            width=0.005*img_width,  # 0.2% of image width
            color='red'
        ))
            # Draw UGV position circle
        ax.add_patch(Circle((ugv_px, ugv_py), radius=0.005*img_width,
                            color='yellow', ec='black', lw=2))
    
    if not plot_colors: 
        plt.colorbar(sc, ax=ax, label='Height (m)')
    print(f"Time for elaboration of sample {id}: {elapsed_time:.4f} seconds")

    start_time_visual = time.time()
    ax.set_title("UAV Image with UGV Positions + Points")
    ax.set_xlim(0, img_width)
    ax.set_ylim(img_height, 0)  # invert Y axis so it matches image coordinates
    ax.set_aspect('equal')
    plt.tight_layout()

    # Save the figure to an outputs directory with a timestamped filename
    out_dir = "dataset_samples_visualizations"
    os.makedirs(out_dir, exist_ok=True)
    fname = f"uav_ugv_vis_{id}.png"
    save_path = os.path.join(out_dir, fname)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved visualization to: {save_path}")
    end_time_visual = time.time()
    print(f"Time for visualization saving of sample {id}: {end_time_visual - start_time_visual:.4f} seconds")


if __name__ == "__main__":
    main()
