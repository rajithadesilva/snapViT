import os
import torch
import numpy as np
import argparse
from torch.utils.data import DataLoader
import plotly.graph_objects as go

# Import necessary classes from your project files
from dataset import VineyardDataset

# --- Configuration (should match training script for consistency) ---
CONFIG = {
    'vit_model': 'vit_small_patch16_224',
    'feature_dim': 128,
    'num_ugv_views': 8,
    'grid_size': (34, 34, 8),
    'grid_resolution': 0.3,
    'batch_size': 1,
    'device': 'cpu', # No need for GPU for this script
}
DRONE_HEIGHT = 11.8872

def get_frustum_corners(c2w_matrix, intrinsics, img_size=(224, 224), frustum_depth=2.0):
    """Calculates the 8 corners of a camera frustum in world coordinates."""
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    W, H = img_size

    # Define the 4 corners of the image plane in camera coordinates
    # (0,0), (W,0), (W,H), (0,H)
    corners_2d = np.array([
        [0, 0], [W, 0], [W, H], [0, H]
    ])

    # Unproject 2D corners to 3D points at depth=1
    corners_3d_d1 = np.zeros((4, 3))
    corners_3d_d1[:, 0] = (corners_2d[:, 0] - cx) / fx
    corners_3d_d1[:, 1] = (corners_2d[:, 1] - cy) / fy
    corners_3d_d1[:, 2] = 1.0

    # Scale to desired frustum depth to get the far plane corners
    near_plane = corners_3d_d1 * 0.1 # A small near plane
    far_plane = corners_3d_d1 * frustum_depth

    # Combine with camera origin (0,0,0) to form the 8 points of the frustum volume
    # Point 0 is camera center, 1-4 are near plane, 5-8 are far plane
    frustum_points_cam = np.vstack([
        np.array([0, 0, 0]),
        near_plane,
        far_plane
    ])

    # Convert to homogeneous coordinates
    frustum_points_hom = np.hstack([frustum_points_cam, np.ones((9, 1))])

    # Transform to world coordinates using the camera-to-world matrix
    frustum_points_world = (c2w_matrix @ frustum_points_hom.T).T[:, :3]

    return frustum_points_world

def check_points_in_frustum(points_3d_flat, w2c_matrix, intrinsics, img_size=(224, 224)):
    """
    Checks which 3D points fall within the camera's view.
    Returns a boolean mask.
    """
    # Homogeneous coordinates for 3D points
    points_hom = np.hstack([points_3d_flat, np.ones((points_3d_flat.shape[0], 1))])

    # Transform points to camera coordinates
    points_cam = (w2c_matrix @ points_hom.T).T

    # Project points to image plane
    points_img = (intrinsics @ points_cam[:, :3].T).T

    # Normalize to get pixel coordinates
    depth = points_img[:, 2]
    # Add epsilon to prevent division by zero
    projected_coords_2d = points_img[:, :2] / (depth[:, np.newaxis] + 1e-8)

    # Visibility mask
    # 1. Points must be in front of the camera (positive depth)
    mask_depth = depth > 0
    # 2. Points must be within image boundaries
    W, H = img_size
    mask_x = (projected_coords_2d[:, 0] >= 0) & (projected_coords_2d[:, 0] <= W)
    mask_y = (projected_coords_2d[:, 1] >= 0) & (projected_coords_2d[:, 1] <= H)

    return mask_depth & mask_x & mask_y


def main(args):
    # --- Data ---
    # No transforms needed as we are not passing images to a model
    dataset = VineyardDataset(root_dir=args.data_root, config=CONFIG, transforms=None)

    # Adjust scene index from 1-based user input to 0-based list index
    scene_idx_0_based = args.scene_idx - 1

    if not (0 <= scene_idx_0_based < len(dataset)):
        raise ValueError(f"Scene index {args.scene_idx} is out of bounds. Please provide a value between 1 and {len(dataset)}.")

    print(f"Visualizing scene {args.scene_idx} (index {scene_idx_0_based})...")
    scene_data = dataset[scene_idx_0_based]

    # --- Prepare data for visualization ---
    grid_points = scene_data['ugv_data']['grid_points_3d'].numpy()
    grid_points_flat = grid_points.reshape(-1, 3)

    ugv_poses_w2c = scene_data['ugv_data']['camera_poses'].numpy()
    ugv_intrinsics = scene_data['ugv_data']['intrinsics'].numpy()

    # --- Setup Plotly Figure ---
    fig = go.Figure()

    # --- Plot the main point grid in black ---
    fig.add_trace(go.Scatter3d(
        x=grid_points_flat[:, 0],
        y=grid_points_flat[:, 1],
        z=grid_points_flat[:, 2],
        mode='markers',
        marker=dict(size=2, color='black', opacity=0.3),
        name='3D Grid Points'
    ))

    # Define a color cycle for the UGV views
    colors = ['red', 'green', 'blue', 'orange', 'purple', 'yellow', 'cyan', 'magenta']

    # --- Process and visualize each UGV camera ---
    for i in range(CONFIG['num_ugv_views']):
        w2c = ugv_poses_w2c[i]
        intrinsics = ugv_intrinsics[i]
        color = colors[i % len(colors)]

        # Get Camera-to-World matrix for placing the frustum in the scene
        try:
            c2w = np.linalg.inv(w2c)
        except np.linalg.LinAlgError:
            print(f"Warning: Could not invert pose matrix for UGV view {i}. Skipping.")
            continue

        # 1. Visualize the frustum
        corners = get_frustum_corners(c2w, intrinsics)
        cam_center = corners[0]
        near_corners = corners[1:5]
        far_corners = corners[5:9]

        # Lines from camera center to far plane
        for k in range(4):
            fig.add_trace(go.Scatter3d(x=[cam_center[0], far_corners[k, 0]], y=[cam_center[1], far_corners[k, 1]], z=[cam_center[2], far_corners[k, 2]], mode='lines', line=dict(color=color, width=2), name=f'UGV {i} Frustum'))
        # Lines for the far plane rectangle
        for k in range(4):
            fig.add_trace(go.Scatter3d(x=[far_corners[k, 0], far_corners[(k+1)%4, 0]], y=[far_corners[k, 1], far_corners[(k+1)%4, 1]], z=[far_corners[k, 2], far_corners[(k+1)%4, 2]], mode='lines', line=dict(color=color, width=2), showlegend=False))


        # 2. Find and color points inside this frustum
        visible_mask = check_points_in_frustum(grid_points_flat, w2c, intrinsics)
        visible_points = grid_points_flat[visible_mask]

        if visible_points.shape[0] > 0:
            fig.add_trace(go.Scatter3d(
                x=visible_points[:, 0],
                y=visible_points[:, 1],
                z=visible_points[:, 2],
                mode='markers',
                marker=dict(size=3, color=color),
                name=f'Points in UGV {i} view'
            ))

    # --- Visualize the Aerial "Camera" ---
    # For the aerial view, the origin is the center of the world. We can draw a simple marker for it.
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[np.float32(DRONE_HEIGHT)], # Position it above the grid
        mode='markers',
        marker=dict(size=10, color='gold', symbol='diamond'),
        name='Aerial Camera Position'
    ))


    # --- Final Touches on Layout ---
    fig.update_layout(
        title=f'Interactive 3D Visualization for Scene {args.scene_idx}',
        scene=dict(
            xaxis_title='X (meters)',
            yaxis_title='Y (meters)',
            zaxis_title='Z (meters)',
            aspectmode='data' # This makes the aspect ratio realistic
        ),
        margin=dict(r=20, b=10, l=10, t=40)
    )

    print("Showing interactive plot...")
    fig.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate an interactive 3D visualization of camera frustums and the point grid for a SnapViT scene.")
    parser.add_argument('--data_root', type=str, default='datasets/vineyard_dataset', help="Path to the root of the processed dataset.")
    parser.add_argument('--scene_idx', type=int, default=6, help="The 1-based index of the scene to visualize.")
    args = parser.parse_args()
    main(args)