import os
import json
import torch
import random
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode
from torchvision import transforms

class VineyardDataset(Dataset):
    def __init__(self, root_dir, config, transforms=None):
        self.root_dir = root_dir
        self.scene_folders = [os.path.join(root_dir, d) for d in sorted(os.listdir(root_dir)) if os.path.isdir(os.path.join(root_dir, d))]
        self.config = config
        self.transforms = transforms

    def __len__(self):
        return len(self.scene_folders)

    def __getitem__(self, idx):
        scene_path = self.scene_folders[idx]
        with open(os.path.join(scene_path, 'metadata.json'), 'r') as f:
            metadata = json.load(f)

        # 1. Load Overhead UAV Image
        uav_image_path = os.path.join(scene_path, metadata['uav_image_path'])
        uav_image = read_image(uav_image_path, mode=ImageReadMode.RGB)

        # 2. Select and Load UGV Images
        target_num_views = self.config['num_ugv_views']
        available_views = metadata['ugv_images']
        num_available = len(available_views)

        if num_available == 0:
            # This will prevent crashes if a scene folder has no UGV images.
            raise ValueError(f"Error: Scene {scene_path} contains no UGV images.")

        # Shuffle the available views to get a random selection
        random.shuffle(available_views)

        # If we don't have enough views, repeat the list of available views
        # until it's long enough, then take the exact number we need.
        if num_available < target_num_views:
            padded_views = (available_views * (target_num_views // num_available + 1))[:target_num_views]
            ugv_metadata_sample = padded_views
        else:
            # If we have enough (or more) views, just take the number we need.
            ugv_metadata_sample = available_views[:target_num_views]
        
        ugv_images, ugv_poses, ugv_intrinsics = [], [], []
        for view_meta in ugv_metadata_sample:
            img_path = os.path.join(scene_path, view_meta['image_path'])
            ugv_images.append(read_image(img_path, mode=ImageReadMode.RGB))
            ugv_poses.append(torch.tensor(view_meta['camera_pose_w2c'], dtype=torch.float32))
            ugv_intrinsics.append(torch.tensor(view_meta['camera_intrinsics'], dtype=torch.float32))
            
        ugv_images = torch.stack(ugv_images)
        ugv_poses = torch.stack(ugv_poses)
        ugv_intrinsics = torch.stack(ugv_intrinsics)

        # 3. Apply transformations
        if self.transforms:
            uav_image = self.transforms(uav_image)
            # Apply transform to each image in the stack
            ugv_images = torch.stack([self.transforms(img) for img in ugv_images])
        
        # 4. Define the 3D Grid
        grid_points_3d = self.create_bev_grid(self.config['grid_size'], self.config['grid_resolution'])

        return {
            'uav_data': {'uav_image': uav_image},
            'ugv_data': {
                'ugv_images': ugv_images,
                'camera_poses': ugv_poses,
                'intrinsics': ugv_intrinsics,
                'grid_points_3d': grid_points_3d
            }
        }

    def create_bev_grid(self, grid_size, resolution):
        X, Y, Z = grid_size
        x_coords = torch.linspace(-X * resolution / 2, X * resolution / 2, X)
        y_coords = torch.linspace(Y * resolution / 2, -Y * resolution / 2, Y)
        z_coords = torch.linspace(0, Z * resolution, Z)
        grid_y, grid_x, grid_z = torch.meshgrid(y_coords, x_coords, z_coords, indexing='ij')
        grid_points = torch.stack([grid_x, grid_y, grid_z], dim=-1)
        return grid_points
