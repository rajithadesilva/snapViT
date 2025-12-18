import os
import torch
import numpy as np
import argparse
from torch.utils.data import DataLoader
import plotly.graph_objects as go
import math
from scipy.spatial.transform import Rotation as R

# Import necessary classes from your project files
from dataset import VineyardDataset

# --- Configuration (should match training script for consistency) ---
CONFIG = {
    'vit_model': 'vit_small_patch16_224',
    'feature_dim': 128,
    'num_ugv_views': 8,
    'grid_size': (20, 50, 8),
    'grid_resolution': 0.2,
    'batch_size': 1,
    'device': 'cpu',
}
DRONE_HEIGHT = 11.8872

def get_frustum_corners_from_fov(c2w_matrix, hfov_deg=87, vfov_deg=58, near_plane=0.1, far_plane=2.0):
    """Calculates the 8 corners of a camera frustum in world coordinates based on FoV."""
    hfov_rad, vfov_rad = math.radians(hfov_deg), math.radians(vfov_deg)
    far_height, far_width = 2 * math.tan(vfov_rad / 2) * far_plane, 2 * math.tan(hfov_rad / 2) * far_plane
    near_height, near_width = 2 * math.tan(vfov_rad / 2) * near_plane, 2 * math.tan(hfov_rad / 2) * near_plane

    frustum_points_cam = np.array([
        [-near_width / 2, -near_height / 2, near_plane], [near_width / 2, -near_height / 2, near_plane],
        [near_width / 2,  near_height / 2, near_plane], [-near_width / 2,  near_height / 2, near_plane],
        [-far_width / 2, -far_height / 2, far_plane], [far_width / 2, -far_height / 2, far_plane],
        [far_width / 2,  far_height / 2, far_plane], [-far_width / 2,  far_height / 2, far_plane],
    ])
    frustum_points_cam = np.vstack([np.array([0, 0, 0]), frustum_points_cam])
    frustum_points_hom = np.hstack([frustum_points_cam, np.ones((9, 1))])
    frustum_points_world = (c2w_matrix @ frustum_points_hom.T).T[:, :3]
    return frustum_points_world

def check_points_in_frustum(points_3d_flat, w2c_matrix, intrinsics, img_size=(224, 224)):
    """Checks which 3D points fall within the camera's view."""
    points_hom = np.hstack([points_3d_flat, np.ones((points_3d_flat.shape[0], 1))])
    points_cam = (w2c_matrix @ points_hom.T).T
    points_img = (intrinsics @ points_cam[:, :3].T).T
    depth = points_img[:, 2]
    projected_coords_2d = points_img[:, :2] / (depth[:, np.newaxis] + 1e-8)
    mask_depth = depth > 0
    W, H = img_size
    mask_x = (projected_coords_2d[:, 0] >= 0) & (projected_coords_2d[:, 0] <= W)
    mask_y = (projected_coords_2d[:, 1] >= 0) & (projected_coords_2d[:, 1] <= H)
    return mask_depth & mask_x & mask_y

def main(args):
    dataset = VineyardDataset(root_dir=args.data_root, config=CONFIG, transforms=None)
    scene_idx_0_based = args.scene_idx - 1
    if not (0 <= scene_idx_0_based < len(dataset)):
        raise ValueError(f"Scene index {args.scene_idx} is out of bounds. Please provide a value between 1 and {len(dataset)}.")

    print(f"Visualizing scene {args.scene_idx} (index {scene_idx_0_based})...")
    scene_data = dataset[scene_idx_0_based]
    grid_points_flat = scene_data['ugv_data']['grid_points_3d'].numpy().reshape(-1, 3)
    ugv_poses_w2c = scene_data['ugv_data']['camera_poses'].numpy()
    ugv_intrinsics = scene_data['ugv_data']['intrinsics'].numpy()

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=grid_points_flat[:, 0], y=grid_points_flat[:, 1], z=grid_points_flat[:, 2],
        mode='markers', marker=dict(size=2, color='black', opacity=0.3), name='3D Grid Points'))

    colors = ['red', 'green', 'blue', 'orange', 'purple', 'yellow', 'cyan', 'magenta']
    for i in range(CONFIG['num_ugv_views']):
        w2c = ugv_poses_w2c[i]
        intrinsics = ugv_intrinsics[i]
        color = colors[i % len(colors)]

        try:
            # Use the camera-to-world matrix directly from the dataset
            c2w = np.linalg.inv(w2c)
        except np.linalg.LinAlgError:
            print(f"Warning: Could not invert pose matrix for UGV view {i}. Skipping.")
            continue
        
        # The uncorrected c2w matrix is used to calculate the frustum corners
        corners = get_frustum_corners_from_fov(c2w, far_plane=args.frustum_depth)
        
        cam_center = corners[0]
        near_corners, far_corners = corners[1:5], corners[5:9]

        for k in range(4):
            fig.add_trace(go.Scatter3d(x=[cam_center[0], far_corners[k, 0]], y=[cam_center[1], far_corners[k, 1]], z=[cam_center[2], far_corners[k, 2]], mode='lines', line=dict(color=color, width=2), name=f'UGV {i} Frustum'))
            fig.add_trace(go.Scatter3d(x=[far_corners[k, 0], far_corners[(k+1)%4, 0]], y=[far_corners[k, 1], far_corners[(k+1)%4, 1]], z=[far_corners[k, 2], far_corners[(k+1)%4, 2]], mode='lines', line=dict(color=color, width=2), showlegend=False))
            fig.add_trace(go.Scatter3d(x=[near_corners[k, 0], near_corners[(k+1)%4, 0]], y=[near_corners[k, 1], near_corners[(k+1)%4, 1]], z=[near_corners[k, 2], near_corners[(k+1)%4, 2]], mode='lines', line=dict(color=color, width=2), showlegend=False))

        visible_mask = check_points_in_frustum(grid_points_flat, w2c, intrinsics)
        visible_points = grid_points_flat[visible_mask]
        if visible_points.shape[0] > 0:
            fig.add_trace(go.Scatter3d(x=visible_points[:, 0], y=visible_points[:, 1], z=visible_points[:, 2],
                mode='markers', marker=dict(size=3, color=color), name=f'Points in UGV {i} view'))

    fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[np.float32(DRONE_HEIGHT)],
        mode='markers', marker=dict(size=10, color='gold', symbol='diamond'), name='Aerial Camera Position'))

    fig.update_layout(
        title=f'Interactive 3D Visualization for Scene {args.scene_idx}',
        scene=dict(xaxis_title='X (meters)', yaxis_title='Y (meters)', zaxis_title='Z (meters)', aspectmode='data'),
        margin=dict(r=20, b=10, l=10, t=40))

    print("Showing interactive plot...")
    fig.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate an interactive 3D visualization of camera frustums for a SnapViT scene.")
    parser.add_argument('--data_root', type=str, default='datasets/row_wise_dataset', help="Path to the root of the processed dataset.")
    parser.add_argument('--scene_idx', type=int, default=3, help="The 1-based index of the scene to visualize.")
    parser.add_argument('--frustum_depth', type=float, default=1.0, help="The depth/length of the visualized camera frustum in meters.")
    args = parser.parse_args()
    main(args)