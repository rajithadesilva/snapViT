import os
import json
import numpy as np
import torch
import random
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode
from torchvision import transforms

class VineyardDataset(Dataset):
    def __init__(self, root_dir, config, transforms=None, depth_transforms=None, consecutive_frames=False):
        self.root_dir = root_dir
        self.config = config
        self.transforms = transforms
        self.depth_transforms = depth_transforms
        self.depth_range = self.config.get('depth_range', (0.0, 5.0))
        self.ground_tile_size = self.config.get('ground_tile_size', 10.0)
        self.edge_margin_m = float(self.config.get('edge_margin_m', 1.0))
        self.consecutive_frames = consecutive_frames
        all_scene_folders = [os.path.join(root_dir, d) for d in sorted(os.listdir(root_dir)) if os.path.isdir(os.path.join(root_dir, d)) and (not 'ugv_rgb' in d and not 'ugv_depth' in d and not 'temp' in d)]
        self.scene_folders = self._filter_scene_folders(all_scene_folders)

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
        available_views = self.filter_views_by_edge_margin(metadata['ugv_images'])
        num_available = len(available_views)

        if num_available == 0:
            # This will prevent crashes if a scene folder has no UGV images.
            raise ValueError(f"Error: Scene {scene_path} contains no UGV images.")
        
        if self.consecutive_frames:
            views_ids = [view['camera_idx'] for view in available_views]
            views_ids.sort()
            selected_ids = self.identify_consecutive_frames(views_ids, target_num_views)
            ugv_metadata_sample = []
            for view_id in selected_ids:
                for view_meta in available_views:
                    if view_meta['camera_idx'] == view_id:
                        ugv_metadata_sample.append(view_meta)
                        break
            

            
        else: 
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
        
        ugv_images, ugv_depths, ugv_poses, ugv_intrinsics= [], [], [], []
        for view_meta in ugv_metadata_sample:
            img_path = os.path.join(self.root_dir, view_meta['image_path'])
            if self.config.get('use_depth', False) and 'depth_path' in view_meta:
                depth_path = os.path.join(self.root_dir, view_meta['depth_path'])
                ugv_depths.append(read_image(depth_path, mode=ImageReadMode.RGB))
            ugv_images.append(read_image(img_path, mode=ImageReadMode.RGB))
            ugv_poses.append(torch.tensor(view_meta['camera_pose_w2c'], dtype=torch.float32))
            ugv_intrinsics.append(torch.tensor(view_meta['camera_intrinsics'], dtype=torch.float32))
            
        ugv_images = torch.stack(ugv_images)
        ugv_depths = torch.stack(ugv_depths) if ugv_depths else None
        ugv_poses = torch.stack(ugv_poses)
        ugv_intrinsics = torch.stack(ugv_intrinsics)
        depth_range = torch.tensor(self.depth_range, dtype=torch.float32)
        ground_tile_size = torch.tensor(self.ground_tile_size, dtype=torch.float32)

        if self.config.get('train_img_size', None) is not None:
            ugv_h, ugv_v = ugv_images.shape[2:]
            ugv_intrinsics = self.intrinsics_transform(ugv_intrinsics, original_size=(ugv_h, ugv_v), new_size=self.config['train_img_size'])

        # 3. Apply transformations
        if self.transforms:
            uav_image = self.transforms(uav_image)
            # Apply transform to each image in the stack
            ugv_images = torch.stack([self.transforms(img) for img in ugv_images])
            if ugv_depths is not None and self.depth_transforms:
                ugv_depths = torch.stack([self.depth_transforms(depth) for depth in ugv_depths])
                # keep only the first channel
                ugv_depths = ugv_depths[:, 0:1, ...]
        
        # 4. Define the 3D Grid
        grid_points_3d = self.create_bev_grid(self.config['grid_size'], self.config['grid_resolution'])
    
        return {
            'uav_data': {'uav_image': uav_image},
            'ugv_data': {
                'ugv_images': ugv_images,
                'ugv_depths': ugv_depths,
                'camera_poses': ugv_poses,
                'intrinsics': ugv_intrinsics,
                'depth_range': depth_range,
                'ground_tile_size': ground_tile_size,
                'grid_points_3d': grid_points_3d
            }
        }

    def _filter_scene_folders(self, scene_folders):
        if self.edge_margin_m <= 0:
            return scene_folders

        valid_scene_folders = []
        filtered_scene_count = 0

        for scene_path in scene_folders:
            metadata_path = os.path.join(scene_path, 'metadata.json')
            if not os.path.exists(metadata_path):
                filtered_scene_count += 1
                continue

            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            valid_views = self.filter_views_by_edge_margin(metadata.get('ugv_images', []))
            if valid_views:
                valid_scene_folders.append(scene_path)
            else:
                filtered_scene_count += 1

        if filtered_scene_count > 0:
            print(
                f"Filtered {filtered_scene_count} scenes with no UGV views farther than {self.edge_margin_m:.2f} m from the aerial image edge."
            )

        return valid_scene_folders

    def filter_views_by_edge_margin(self, views_metadata):
        if self.edge_margin_m <= 0:
            return list(views_metadata)

        return [
            view_meta for view_meta in views_metadata
            if self.is_pose_far_from_edge(view_meta.get('camera_pose_w2c'))
        ]

    def is_pose_far_from_edge(self, camera_pose_w2c):
        if camera_pose_w2c is None:
            return False

        half_tile_size = self.ground_tile_size / 2.0
        if self.edge_margin_m >= half_tile_size:
            return False

        try:
            c2w_matrix = np.linalg.inv(np.asarray(camera_pose_w2c, dtype=np.float32))
        except np.linalg.LinAlgError:
            return False

        local_x, local_y = c2w_matrix[:2, 3]
        distance_to_edge_x = half_tile_size - abs(float(local_x))
        distance_to_edge_y = half_tile_size - abs(float(local_y))

        return min(distance_to_edge_x, distance_to_edge_y) > self.edge_margin_m
    

    def intrinsics_transform(self, intrinsics, original_size, new_size):
        """
        Adjusts camera intrinsics based on image resizing.
        """
        if new_size is None:
            return intrinsics
        scale_x = new_size[1] / original_size[1]
        scale_y = new_size[0] / original_size[0]
        intrinsics[:, 0, 0] *= scale_x  # fx
        intrinsics[:, 1, 1] *= scale_y  # fy
        intrinsics[:, 0, 2] *= scale_x  # cx
        intrinsics[:, 1, 2] *= scale_y  # cy
        return intrinsics
    
    def create_bev_grid(self, grid_size, resolution):
        X, Y, Z = grid_size
        x_coords = torch.linspace(-X * resolution / 2, X * resolution / 2, X)
        y_coords = torch.linspace(Y * resolution / 2, -Y * resolution / 2, Y)
        z_coords = torch.linspace(0, Z * resolution, Z)
        grid_y, grid_x, grid_z = torch.meshgrid(y_coords, x_coords, z_coords, indexing='ij')
        grid_points = torch.stack([grid_x, grid_y, grid_z], dim=-1)
        return grid_points
    
    def identify_consecutive_frames(self, view_ids, target_num_views):
        """
        Identifies sequences of consecutive frame IDs and selects a random sequence of the desired length.
        """
        view_ids.sort()
        consecutive_groups = []
        current_group = [view_ids[0]]

        for i in range(1, len(view_ids)):
            if view_ids[i] == view_ids[i-1] + 1:
                current_group.append(view_ids[i])
            else:
                consecutive_groups.append(current_group)
                current_group = [view_ids[i]]
        
        consecutive_groups.append(current_group)

        # Filter groups that are long enough
        valid_groups = [group for group in consecutive_groups if len(group) >= target_num_views]

        if valid_groups:
            selected_group = random.choice(valid_groups)
            start_idx = random.randint(0, len(selected_group) - target_num_views)
            ids = selected_group[start_idx:start_idx + target_num_views]
        else: 
            selected_group = random.choice(consecutive_groups)
            # Repeat ids if not enough
            padded_group = (selected_group * (target_num_views*target_num_views))[:target_num_views]
            padded_group.sort()
            ids = padded_group[:target_num_views]

        return ids