import os
import json
import torch
import random
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode


class VineyardDataset(Dataset):
    """
    Updated to support the new dataset structure:

    root_dir/
      ugv_rgb/                     (shared across scenes)
        frame_*.png
      ugv_depth/                   (shared across scenes)
        frame_*.png
      scene_0001/
        uav_image/
          tile_*.png
        metadata.json              (ugv image/depth paths point to root-level shared folders)
      scene_0002/
        ...
    """

    def __init__(self, root_dir, config, transforms=None):
        self.root_dir = root_dir
        self.config = config
        self.transforms = transforms

        # Only treat directories that look like scenes as scenes.
        # This avoids mistakenly including ugv_rgb/ and ugv_depth/ as "scenes".
        scene_dirs = []
        for d in sorted(os.listdir(root_dir)):
            full = os.path.join(root_dir, d)
            if not os.path.isdir(full):
                continue
            if d.startswith("scene_"):
                scene_dirs.append(full)

        # Fallback: if you ever use different naming, you can still keep the old behaviour
        # by uncommenting this block.
        # if len(scene_dirs) == 0:
        #     scene_dirs = [
        #         os.path.join(root_dir, d)
        #         for d in sorted(os.listdir(root_dir))
        #         if os.path.isdir(os.path.join(root_dir, d))
        #         and d not in ("ugv_rgb", "ugv_depth")
        #     ]

        self.scene_folders = scene_dirs

    def __len__(self):
        return len(self.scene_folders)

    def __getitem__(self, idx):
        scene_path = self.scene_folders[idx]

        metadata_path = os.path.join(scene_path, "metadata.json")
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        # 1) Load Overhead UAV Image (still stored inside the scene folder)
        uav_image_path = os.path.join(scene_path, metadata["uav_image_path"])
        uav_image = read_image(uav_image_path, mode=ImageReadMode.RGB)

        # 2) Select UGV views
        target_num_views = int(self.config["num_ugv_views"])
        available_views = metadata.get("ugv_images", [])
        num_available = len(available_views)

        if num_available == 0:
            raise ValueError(f"Error: Scene {scene_path} contains no UGV images in metadata.json")

        # IMPORTANT: do not shuffle metadata list in-place (it comes from JSON).
        available_views = list(available_views)
        random.shuffle(available_views)

        if num_available < target_num_views:
            ugv_metadata_sample = (available_views * (target_num_views // num_available + 1))[:target_num_views]
        else:
            ugv_metadata_sample = available_views[:target_num_views]

        # 3) Load UGV RGB (+ optional depth), intrinsics, poses
        ugv_images = []
        ugv_depths = []
        ugv_poses = []
        ugv_intrinsics = []

        load_depth = bool(self.config.get("use_depth", False))

        for view_meta in ugv_metadata_sample:
            # NEW STRUCTURE: these paths are relative to root_dir (e.g., "ugv_rgb/frame_....png")
            rgb_path = os.path.join(self.root_dir, view_meta["image_path"])
            ugv_images.append(read_image(rgb_path, mode=ImageReadMode.RGB))

            if load_depth:
                # Depth saved as 16-bit PNG. torchvision returns uint8/uint16 depending on build;
                # we read as UNCHANGED to preserve bit-depth.
                if "depth_path" not in view_meta:
                    raise KeyError(f"Missing 'depth_path' in metadata for view {view_meta}")
                depth_path = os.path.join(self.root_dir, view_meta["depth_path"])
                depth_img = read_image(depth_path, mode=ImageReadMode.UNCHANGED)  # typically [1,H,W]
                ugv_depths.append(depth_img)

            ugv_poses.append(torch.tensor(view_meta["camera_pose_w2c"], dtype=torch.float32))
            ugv_intrinsics.append(torch.tensor(view_meta["camera_intrinsics"], dtype=torch.float32))

        ugv_images = torch.stack(ugv_images)                # [V,3,H,W]
        ugv_poses = torch.stack(ugv_poses)                  # [V,4,4]
        ugv_intrinsics = torch.stack(ugv_intrinsics)        # [V,3,3]

        if load_depth:
            ugv_depths = torch.stack(ugv_depths)            # [V,1,H,W] (or [V,H,W] depending on reader)

        # 4) Apply transformations
        if self.transforms:
            uav_image = self.transforms(uav_image)
            ugv_images = torch.stack([self.transforms(img) for img in ugv_images])

            # Depth transforms are tricky (don’t normalise like RGB).
            # If you want depth transforms, do it explicitly in your training code,
            # or add a dedicated depth_transform in config.
            # For now we leave depth untouched.

        # 5) Define the 3D Grid
        grid_points_3d = self.create_bev_grid(self.config["grid_size"], self.config["grid_resolution"])

        sample = {
            "uav_data": {"uav_image": uav_image},
            "ugv_data": {
                "ugv_images": ugv_images,
                "camera_poses": ugv_poses,
                "intrinsics": ugv_intrinsics,
                "grid_points_3d": grid_points_3d,
            },
        }

        if load_depth:
            sample["ugv_data"]["ugv_depths"] = ugv_depths

        return sample

    def create_bev_grid(self, grid_size, resolution):
        X, Y, Z = grid_size
        x_coords = torch.linspace(-X * resolution / 2, X * resolution / 2, X)
        y_coords = torch.linspace(Y * resolution / 2, -Y * resolution / 2, Y)
        z_coords = torch.linspace(0, Z * resolution, Z)
        grid_y, grid_x, grid_z = torch.meshgrid(y_coords, x_coords, z_coords, indexing="ij")
        grid_points = torch.stack([grid_x, grid_y, grid_z], dim=-1)
        return grid_points
