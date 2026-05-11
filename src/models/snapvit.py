import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


def infer_model_type_from_name(model_name):
    """Infers the backbone family from a timm model name."""
    name = model_name.lower()

    if 'dinov3' in name or 'dinov2' in name:
        return 'dinov3'
    if 'swin' in name:
        return 'swin'
    if 'convnext' in name:
        return 'convnext'
    if 'resnet' in name:
        return 'resnet'
    if 'vit' in name or 'deit' in name or 'beit' in name:
        return 'vit'

    raise ValueError(
        f"Could not infer model type from model name '{model_name}'. "
        "Supported families by name are: vit/deit/beit, swin, dinov2/dinov3, convnext, resnet."
    )


def create_feature_extractor(model_name='vit_base_patch16_224', pretrained=True, model_type=None):
    """Builds the requested backbone feature extractor.

    If model_type is omitted, it is inferred from model_name.
    """
    model_type = infer_model_type_from_name(model_name) if model_type is None else model_type.lower()
    if model_type == 'vit':
        return ViTFeatureExtractor(model_name=model_name, pretrained=pretrained)
    if model_type == 'resnet':
        return ResNetFeatureExtractor(model_name=model_name, pretrained=pretrained)
    if model_type == 'convnext':
        return ConvNeXTFeatureExtractor(model_name=model_name, pretrained=pretrained)
    if model_type == 'swin':
        return SwintTFeatureExtractor(model_name=model_name, pretrained=pretrained)
    if model_type == 'dinov3':
        return Dinov3FeatureExtractor(model_name=model_name, pretrained=pretrained)
    raise ValueError(
        f"Unsupported model_type '{model_type}'. "
        "Expected one of: vit, resnet, convnext, swin, dinov3."
    )

class ViTFeatureExtractor(nn.Module):
    """
    A Vision Transformer that outputs a 2D feature map.
    This is the shared backbone for both the ground and overhead encoders.
    """
    def __init__(self, model_name='vit_base_patch16_224', pretrained=True):
        super().__init__()
        self.vit = timm.create_model(model_name, pretrained=pretrained)
        self.vit.head = nn.Identity()  # Remove the classification head
        self.patch_size = self.vit.patch_embed.patch_size[0]
        self.embed_dim = self.vit.embed_dim
        print(f"Initialized ViTFeatureExtractor with model {model_name}, patch size {self.patch_size}, embed dim {self.embed_dim}")

    def forward(self, x):
        B, C, H, W = x.shape
        patch_embeddings = self.vit.forward_features(x)
        patch_embeddings = patch_embeddings[:, 1:, :]  # Remove [CLS] token
        H_feat, W_feat = H // self.patch_size, W // self.patch_size
        features = patch_embeddings.permute(0, 2, 1).reshape(B, self.embed_dim, H_feat, W_feat)
        
        # Enforce channels-first output: (B, C, H, W)
        if features.dim() == 4 and features.shape[-1] > features.shape[1]:
            print("Permuting features to channels-first format")
            features = features.permute(0, 3, 1, 2).contiguous()
        return features.contiguous()
    
class ResNetFeatureExtractor(nn.Module):
    """
    A ResNet that outputs a 2D feature map.
    This is the shared backbone for both the ground and overhead encoders.
    """
    def __init__(self, model_name='resnet50', pretrained=True):
        super().__init__()
        self.resnet = timm.create_model(model_name, pretrained=pretrained)
        self.resnet.head = nn.Identity()  # Remove the classification head
        self.embed_dim = self.resnet.num_features
        print(f"Initialized ResNetFeatureExtractor with embed dim {self.embed_dim}")

    def forward(self, x):
        features = self.resnet.forward_features(x)

        # Enforce channels-first output: (B, C, H, W)
        if features.dim() == 4 and features.shape[-1] > features.shape[1]:
            features = features.permute(0, 3, 1, 2).contiguous()
        return features.contiguous()
    

class ConvNeXTFeatureExtractor(nn.Module):
    """
    A ConvNeXT that outputs a 2D feature map.
    This is the shared backbone for both the ground and overhead encoders.
    """
    def __init__(self, model_name='convnext_tiny', pretrained=True):
        super().__init__()
        self.convnext = timm.create_model(model_name, pretrained=pretrained)
        self.convnext.head = nn.Identity()  # Remove the classification head
        self.embed_dim = self.convnext.num_features
        print(f"Initialized ConvNeXTFeatureExtractor with embed dim {self.embed_dim}")

    def forward(self, x):
        features = self.convnext.forward_features(x)

        # Enforce channels-first output: (B, C, H, W)
        if features.dim() == 4 and features.shape[-1] > features.shape[1]:
            features = features.permute(0, 3, 1, 2).contiguous()
        return features.contiguous()
    
class SwintTFeatureExtractor(nn.Module):
    """
    A Swin Transformer that outputs a 2D feature map.
    This is the shared backbone for both the ground and overhead encoders.
    """
    def __init__(self, model_name='swin_base_patch4_window7_224', pretrained=True):
        super().__init__()
        self.swin = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(3,),
        )
        self.embed_dim = self.swin.feature_info.channels()[-1]
        print(f"Initialized SwintTFeatureExtractor with embed dim {self.embed_dim}")

    def forward(self, x):
        features = self.swin(x)[0]

        # Enforce channels-first output: (B, C, H, W)
        if features.dim() == 4 and features.shape[-1] > features.shape[1]:
            features = features.permute(0, 3, 1, 2).contiguous()
        return features.contiguous()
    
class Dinov3FeatureExtractor(nn.Module):
    """
    A DINOv3 Vision Transformer that outputs a 2D feature map.
    This is the shared backbone for both the ground and overhead encoders.
    """
    def __init__(self, model_name='vit_7b_patch16_dinov3', pretrained=True):
        super().__init__()
        self.dinov3 = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(0,),
        )
        # Use the channel size corresponding to the selected out_indices (index 0 here)
        self.embed_dim = self.dinov3.feature_info.channels()[0]
        print(f"Initialized Dinov3FeatureExtractor with embed dim {self.embed_dim}")

    def forward(self, x):
        features = self.dinov3(x)[0]

        # Enforce channels-first output: (B, C, H, W)
        if features.dim() == 3:
            b, n, c = features.shape
            h = int(n ** 0.5)
            if h * h != n:
                raise ValueError(f"Token count {n} is not a perfect square.")
            features = features.permute(0, 2, 1).reshape(b, c, h, h).contiguous()
        elif features.dim() == 4 and features.shape[-1] > features.shape[1]:
            features = features.permute(0, 3, 1, 2).contiguous()

        return features.contiguous()

class _GroundEncoder(nn.Module):
    """
    Encodes multiple ground-level images into a single BEV feature map.
    """
    def __init__(self, vit_model_name='vit_base_patch16_224', feature_dim=256):
        super().__init__()
        self.feature_extractor = ViTFeatureExtractor(vit_model_name)
        vit_embed_dim = self.feature_extractor.embed_dim

        # MLP to fuse features from multiple views for a single 3D point
        self.fusion_mlp = nn.Sequential(
            nn.Linear(vit_embed_dim, 512),
            nn.ReLU(),
            nn.Linear(512, feature_dim)
        )
        
        self.vertical_pool = nn.AdaptiveMaxPool1d(1)

    def project_points(self, points_3d, poses_w2c, intrinsics):
        """
        Projects 3D points from world coordinates into 2D image coordinates.
        Args:
            points_3d (Tensor): (B, X, Y, Z, 3)
            poses_w2c (Tensor): World-to-camera poses (B, N_views, 4, 4)
            intrinsics (Tensor): Camera intrinsics (B, N_views, 3, 3)
        Returns:
            projected_coords (Tensor): Normalized 2D coordinates for sampling (B, N_views, X*Y*Z, 2)
            visibility_mask (Tensor): Mask of points visible in each view (B, N_views, X*Y*Z)
        """
        B, N_views, _, _ = poses_w2c.shape
        _, X, Y, Z, _ = points_3d.shape
        
        points_flat = points_3d.reshape(B, -1, 3) # (B, P, 3) where P = X*Y*Z
        points_hom = F.pad(points_flat, (0, 1), value=1.0) # (B, P, 4)
        
        # Transform points to camera coordinates
        points_cam = torch.einsum('bvij,bpj->bvpi', poses_w2c, points_hom) # (B, N_views, P, 4)
        
        # Project points to image plane
        points_img = torch.einsum('bvij,bvpj->bvpi', intrinsics, points_cam[..., :3]) # (B, N_views, P, 3)
        
        depth = points_img[..., 2]
        
        # Normalize to get pixel coordinates
        # Add epsilon to prevent division by zero
        projected_coords_2d = points_img[..., :2] / (depth.unsqueeze(-1) + 1e-8)
        
        # Create visibility mask (points must be in front of the camera)
        visibility_mask = depth > 0

        # Normalize coordinates to [-1, 1] for grid_sample
        # Assuming image dimensions are known (e.g., 224x224)
        img_w, img_h = 224, 224
        projected_coords_normalized = projected_coords_2d.clone()
        projected_coords_normalized[..., 0] = (projected_coords_normalized[..., 0] / (img_w - 1)) * 2 - 1
        projected_coords_normalized[..., 1] = (projected_coords_normalized[..., 1] / (img_h - 1)) * 2 - 1
        
        return projected_coords_normalized.view(B, N_views, -1, 2), visibility_mask.view(B, N_views, -1)

    def forward(self, ugv_images, ugv_depths, camera_poses, intrinsics, grid_points_3d):
        B, N_views, C_img, H_img, W_img = ugv_images.shape
        _, X, Y, Z, _ = grid_points_3d.shape

        # 1. Extract 2D features for all views
        img_features_2d = self.feature_extractor(ugv_images.reshape(B * N_views, C_img, H_img, W_img))
        _, C_feat, H_feat, W_feat = img_features_2d.shape
        img_features_2d = img_features_2d.view(B, N_views, C_feat, H_feat, W_feat)

        # 2. Project 3D grid points into each camera view
        coords, mask = self.project_points(grid_points_3d, camera_poses, intrinsics) # coords: (B, V, P, 2), mask: (B, V, P)

        # 3. Sample features
        sampled_features = F.grid_sample(
            img_features_2d.reshape(B * N_views, C_feat, H_feat, W_feat),
            coords.reshape(B * N_views, 1, -1, 2), # Reshape for grid_sample
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True
        )
        sampled_features = sampled_features.view(B, N_views, C_feat, -1)
        
        # Apply visibility mask
        sampled_features = sampled_features * mask.unsqueeze(2)

        # 4. Fuse features across views (simple mean for now)
        # Summing and dividing by the mask sum avoids division by zero for non-visible points
        fused_features = sampled_features.sum(dim=1) / (mask.sum(dim=1).unsqueeze(1) + 1e-8) # (B, C_feat, P)
        
        # 5. Pass through MLP
        fused_features = fused_features.permute(0, 2, 1) # (B, P, C_feat)
        feature_volume_flat = self.fusion_mlp(fused_features) # (B, P, C_out)
        
        # 6. Reshape and collapse to BEV
        C_out = feature_volume_flat.shape[-1]
        feature_volume_3d = feature_volume_flat.permute(0, 2, 1).reshape(B, C_out, X, Y, Z)
        
        # Reshape for vertical pooling
        B, C_out, Y_dim, X_dim, Z_dim = feature_volume_3d.shape
        # Permute and reshape to (B*Y*X, C, Z) to apply 1D pooling on the Z dimension
        feature_volume_reshaped = feature_volume_3d.permute(0, 2, 3, 1, 4).reshape(-1, C_out, Z_dim)

        # Apply vertical pooling
        bev_features_flat = self.vertical_pool(feature_volume_reshaped).squeeze(-1)

        # Reshape back to a 4D BEV map (B, C, Y, X)
        bev_features = bev_features_flat.reshape(B, Y_dim, X_dim, C_out).permute(0, 3, 1, 2).contiguous()

        return bev_features


class GroundEncoder(nn.Module):
    """
    Encodes multiple ground-level images into a single BEV feature map.
    """
    def __init__(self, model_name='vit_base_patch16_224', feature_dim=256, pretrained=True):
        super().__init__()
        self.feature_extractor = create_feature_extractor(
            model_name=model_name,
            pretrained=pretrained,
        )
        vit_embed_dim = self.feature_extractor.embed_dim

        # MLP to fuse features from multiple views for a single 3D point
        self.fusion_mlp = nn.Sequential(
            nn.Linear(vit_embed_dim, 512),
            nn.ReLU(),
            nn.Linear(512, feature_dim)
        )
        
        self.vertical_pool = nn.AdaptiveMaxPool1d(1)

        self.projection_layer = nn.Conv2d(vit_embed_dim, feature_dim, kernel_size=1)

    def forward(self, ugv_images, ugv_depths, camera_poses, intrinsics, depth_range, ground_tile_size, grid_points_3d):
        #ground_tile_size = ground_tile_size[0].item()
        ground_tile_size = 10.
        w2c_matrices=camera_poses
        B, N_views, C_img, H_img, W_img = ugv_images.shape
        # Extract 2D features for all views

        img_features_2d = self.feature_extractor(ugv_images.reshape(B * N_views, C_img, H_img, W_img))
        _, C_feat, H_feat, W_feat = img_features_2d.shape
        img_features_2d = img_features_2d.view(B, N_views, C_feat, H_feat, W_feat)


        # Swap the first and second rows for all world-to-camera matrices
        #TODO remove after correction of dataset generation
        w2c_matrices = w2c_matrices.clone()
        #w2c_matrices[:, :, [0, 1], :] = w2c_matrices[:, :, [1, 0], :]

        # Interpolate depth maps to feature map size
        ugv_depths = F.interpolate(
                    ugv_depths.reshape(B * N_views, 1, H_img, W_img).float(),
                    size=(H_feat, W_feat),
                    mode='bilinear', # TODO: try nearest maybe better for depth
                    align_corners=False
                ).view(B, N_views, H_feat, W_feat)
        ugv_depths = ugv_depths*65535.0 / 1000.0  # Scale back to meters due to normalization during loading

        #convert it to monodimensional array
        depth_range = depth_range.to(ugv_depths.device).view(-1)

        # Compute camera-to-world (c2w) matrices by inverting w2c
        c2w_matrices = torch.linalg.inv(w2c_matrices)

        fx = intrinsics[:, :, 0, 0]
        fy = intrinsics[:, :, 1, 1]
        cx = intrinsics[:, :, 0, 2]
        cy = intrinsics[:, :, 1, 2]

        # Adjust intrinsics so they match the feature map resolution (H_feat, W_feat)
        # This rescales focal lengths and principal points from image pixels to feature-map pixels.
        patch_size = getattr(self.feature_extractor, "patch_size", 1)
        # If patch_size is a tuple/list, take first element
        if isinstance(patch_size, (tuple, list)):
            patch_size = patch_size[0]

        # Compute scale from original image to feature map
        scale_x = W_feat / float(W_img)
        scale_y = H_feat / float(H_img)

        # Create a scaled copy of intrinsics (preserve dtype/device)
        intrinsics_scaled = intrinsics.clone().to(intrinsics.dtype)

        intrinsics_scaled[:, :, 0, 0] = fx * scale_x    # fx'
        intrinsics_scaled[:, :, 1, 1] = fy * scale_y    # fy'
        intrinsics_scaled[:, :, 0, 2] = cx * scale_x    # cx'
        intrinsics_scaled[:, :, 1, 2] = cy * scale_y    # cy'

        fx = intrinsics_scaled[:, :, 0, 0]
        fy = intrinsics_scaled[:, :, 1, 1]
        cx = intrinsics_scaled[:, :, 0, 2]
        cy = intrinsics_scaled[:, :, 1, 2]

        # If the ViT operates on patch centers, optionally divide by patch_size
        # (uncomment if your projection should be in patch-space rather than resized image pixels)
        #intrinsics_scaled[:, :, :2, :] = intrinsics_scaled[:, :, :2, :] / float(patch_size)

        # Use the corrected intrinsics for projection

        # Ensure variable name expected later
        camera_poses = w2c_matrices

        mask = (ugv_depths > depth_range[0]) & (ugv_depths < depth_range[1])

        u_coords, v_coords = torch.meshgrid(
            torch.arange(W_feat, device=ugv_depths.device, dtype=ugv_depths.dtype),
            torch.arange(H_feat, device=ugv_depths.device, dtype=ugv_depths.dtype),   
            indexing='xy'
        )

        u_f = u_coords.view(1, 1, 1, H_feat, W_feat).expand(B, N_views, 1, H_feat, W_feat)
        v_f = v_coords.view(1, 1, 1, H_feat, W_feat).expand(B, N_views, 1, H_feat, W_feat)
        z_f = ugv_depths.view(B, N_views, 1, H_feat, W_feat)

        # Backproject to camera frame
        Xc = (u_f - cx.view(B, N_views, 1, 1, 1)) * z_f / (fx.view(B, N_views, 1, 1, 1) + 1e-8)
        Yc = (v_f - cy.view(B, N_views, 1, 1, 1)) * z_f / (fy.view(B, N_views, 1, 1, 1) + 1e-8)
        Zc = z_f

        Xc = -Xc  # account for coordinate convention for no reason

        # Build homogeneous coordinates
        pts_cam = torch.stack((Xc, Yc, Zc, torch.ones_like(Zc)), dim=-1)  # (B, N_views, 1, H_feat, W_feat, 4)
        pts_cam = pts_cam.unsqueeze(-1)  # (B, N_views, 1, H_feat, W_feat, 4, 1)
        world_pts = c2w_matrices.unsqueeze(2).unsqueeze(2).unsqueeze(2) @ pts_cam  # (B, N_views, 1, H_feat, W_feat, 4, 1)

        world_pts = world_pts.squeeze(-1)[..., :3]  # (B, N_views, 1, H_feat, W_feat, 3)
        Xw, Yw, Zw = world_pts[..., 0], world_pts[..., 1], world_pts[..., 2]

        px = (Xw * (W_feat / ground_tile_size)) + (W_feat / 2)
        py = -(Yw * (H_feat / ground_tile_size)) + (H_feat / 2)

        

        # Voxelize continuous BEV coordinates (px, py) into integer pixel indices according to pixel size.
        # px, py shape: (B, N_views, 1, H_feat, W_feat) -> remove singleton dim for indexing
        px = px.squeeze(2)  # (B, N_views, H_feat, W_feat)
        py = py.squeeze(2)

        # Convert to nearest pixel indices (you can use floor instead if preferred)
        px_idx = torch.round(px).long()  # (B, N_views, H_feat, W_feat)
        py_idx = torch.round(py).long()

        # In-bounds mask for the BEV grid
        in_bounds = (px_idx >= 0) & (px_idx < W_feat) & (py_idx >= 0) & (py_idx < H_feat)

        # Combine with depth-valid mask computed earlier
        valid_bev_mask = mask & in_bounds  # (B, N_views, H_feat, W_feat)

        # Flatten spatial dims for later accumulation/splatting
        px_flat = px_idx.view(B, N_views, -1)         # (B, N_views, P)
        py_flat = py_idx.view(B, N_views, -1)         # (B, N_views, P)
        valid_flat = valid_bev_mask.view(B, N_views, -1)  # (B, N_views, P)

        # Linearized BEV indices (optional, convenient for scatter operations)
        lin_idx = (py_flat * W_feat) + px_flat  # (B, N_views, P)

        # Expose these variables for downstream accumulation/splatting
        voxel_px_idx = px_flat
        voxel_py_idx = py_flat
        voxel_lin_idx = lin_idx
        voxel_valid_mask = valid_flat

        # Splat features into BEV grid and average duplicates
        device = ugv_images.device
        bev_accum = torch.zeros(B, C_feat, H_feat, W_feat, device=device, dtype=img_features_2d.dtype)
        count = torch.zeros(B, 1, H_feat, W_feat, device=device, dtype=img_features_2d.dtype)

        # prepare source features as (B, V, P, C_feat)
        src_feats = img_features_2d.view(B, N_views, C_feat, -1).permute(0, 1, 3, 2)

        aggregation_method = 'max'  # ' avg', 'max' or 'avgmax'
        if aggregation_method == 'avg':
            bev_accum_list = []
            count_list = []
            
            # Splat views jointly per batch (vectorized over views)
            for b in range(B):
                bev_accum_b = torch.zeros(C_feat, H_feat, W_feat, device=device, dtype=img_features_2d.dtype)
                count_b = torch.zeros(1, H_feat, W_feat, device=device, dtype=img_features_2d.dtype)
                
                valid = voxel_valid_mask[b]                       # (V, P)
                valid_flat = valid.reshape(-1)                    # (V*P,)
                if valid_flat.sum() == 0:
                    bev_accum_list.append(bev_accum_b)
                    count_list.append(count_b)
                    continue

                # gather flattened per-view data for valid locations
                feats_flat = src_feats[b].reshape(-1, C_feat)[valid_flat]   # (N_valid, C_feat)
                px_flat = voxel_px_idx[b].reshape(-1)[valid_flat]           # (N_valid,)
                py_flat = voxel_py_idx[b].reshape(-1)[valid_flat]           # (N_valid,)

                # linear indices into H*W
                lin_idx_flat = (py_flat * W_feat + px_flat).to(torch.long)  # (N_valid,)

                # accumulate features into flattened BEV grid
                bev_accum_flat = bev_accum_b.view(C_feat, -1)               # (C_feat, H*W)
                bev_accum_flat = bev_accum_flat.scatter_add(1, lin_idx_flat.unsqueeze(0).expand(C_feat, -1), feats_flat.t())
                bev_accum_b = bev_accum_flat.view(C_feat, H_feat, W_feat)

                # accumulate counts
                count_flat = count_b[0].view(-1)
                ones = torch.ones_like(lin_idx_flat, dtype=count_flat.dtype, device=count_flat.device)
                count_flat = count_flat.scatter_add(0, lin_idx_flat, ones)
                count_b[0] = count_flat.view(H_feat, W_feat)
                
                bev_accum_list.append(bev_accum_b)
                count_list.append(count_b)

            # Stack all results
            bev_accum = torch.stack(bev_accum_list, dim=0)  # (B, C_feat, H_feat, W_feat)
            count = torch.stack(count_list, dim=0)          # (B, 1, H_feat, W_feat)

            # Average features where multiple contributions exist
            
            bev_features = bev_accum / (count + 1e-8)
        elif aggregation_method == 'max':
            bev_max_list = []
            bev_count_list = []
            
            for b in range(B):
                bev_max_b = torch.full(
                    (C_feat, H_feat * W_feat),
                    0.0,
                    device=device,
                    dtype=img_features_2d.dtype,
                )
                bev_count_b = torch.zeros(H_feat * W_feat, device=device, dtype=img_features_2d.dtype)
                
                valid = voxel_valid_mask[b].reshape(-1)
                if valid.sum() > 0:
                    feats = src_feats[b].reshape(-1, C_feat)[valid]  # (N_valid, C)
                    lin_idx = voxel_lin_idx[b].reshape(-1)[valid]    # (N_valid,)
                    ones = torch.ones_like(lin_idx, dtype=bev_count_b.dtype, device=bev_count_b.device)
                    bev_count_b = bev_count_b.scatter_add(0, lin_idx, ones)

                    bev_max_b = bev_max_b.scatter_reduce(
                        dim=1,
                        index=lin_idx.unsqueeze(0).expand(C_feat, -1),
                        src=feats.t(),
                        reduce="amax",
                        include_self=True,
                    )
                
                bev_max_list.append(bev_max_b)
                bev_count_list.append(bev_count_b)

            bev_max = torch.stack(bev_max_list, dim=0)      # (B, C_feat, H_feat*W_feat)
            bev_count = torch.stack(bev_count_list, dim=0).unsqueeze(1)  # (B, 1, H_feat*W_feat)
            bev_features = bev_max.view(B, C_feat, H_feat, W_feat)
            validity_mask = bev_count.view(B, 1, H_feat, W_feat) > 0

        elif aggregation_method == "avgmax":
            # Use a list to collect results to avoid in-place assignment issues
            bev_sum_list = []
            bev_count_list = []
            bev_max_list = []
            
            for b in range(B):
                bev_sum_b = torch.zeros(C_feat, H_feat * W_feat, device=device, dtype=img_features_2d.dtype)
                bev_count_b = torch.zeros(H_feat * W_feat, device=device, dtype=img_features_2d.dtype)
                bev_max_b = torch.full(
                    (C_feat, H_feat * W_feat),
                    0.,
                    device=device,
                    dtype=img_features_2d.dtype,
                )
                
                valid = voxel_valid_mask[b].reshape(-1)
                if valid.sum() > 0:
                    feats = src_feats[b].reshape(-1, C_feat)[valid]     # (N_valid, C)
                    lin_idx = voxel_lin_idx[b].reshape(-1)[valid]       # (N_valid,)

                    # Sum and count for average
                    bev_sum_b = bev_sum_b.scatter_add(
                        dim=1,
                        index=lin_idx.unsqueeze(0).expand(C_feat, -1),
                        src=feats.t()
                    )
                    ones = torch.ones_like(lin_idx, dtype=bev_count_b.dtype, device=bev_count_b.device)
                    bev_count_b = bev_count_b.scatter_add(0, lin_idx, ones)

                    # Max
                    bev_max_b = bev_max_b.scatter_reduce(
                        dim=1,
                        index=lin_idx.unsqueeze(0).expand(C_feat, -1),
                        src=feats.t(),
                        reduce="amax",
                        include_self=True,
                    )
                
                bev_sum_list.append(bev_sum_b)
                bev_count_list.append(bev_count_b)
                bev_max_list.append(bev_max_b)

            bev_sum = torch.stack(bev_sum_list, dim=0)      # (B, C_feat, H_feat*W_feat)
            bev_count = torch.stack(bev_count_list, dim=0)  # (B, H_feat*W_feat)
            bev_max = torch.stack(bev_max_list, dim=0)      # (B, C_feat, H_feat*W_feat)

            bev_count = bev_count.unsqueeze(1)  # (B, 1, H_feat*W_feat) for broadcasting
            bev_avg = bev_sum / (bev_count + 1e-8)
            bev_features = (bev_avg + bev_max) / 2.0
            validity_mask = bev_count.view(B, 1, H_feat, W_feat) > 0
        else: 
            raise ValueError(f"Unknown aggregation method: {aggregation_method}")
        
        # Optionally, further process BEV features with an MLP or conv layers
        bev_features_flat = bev_features.view(B, C_feat, -1).permute(0, 2, 1)  # (B, H_feat*W_feat, C_feat)
        bev_features_flat = self.fusion_mlp(bev_features_flat)                # (B, H_feat*W_feat, feature_dim)
        bev_features = bev_features_flat.permute(0, 2, 1).view(B, -1,  H_feat, W_feat)  # (B, feature_dim, H_feat, W_feat)

        # use projection layer instead of MLP
        #bev_features = self.projection_layer(bev_features_flat.permute(0, 2, 1).view(B, C_feat, H_feat, W_feat))

        if aggregation_method == 'avg':
            validity_mask = count > 0

        return bev_features, validity_mask


class OverheadEncoder(nn.Module):
    """
    Encodes a single top-down image into a BEV feature map. 
    """
    def __init__(self, model_name='vit_base_patch16_224', feature_dim=256, pretrained=True):
        super().__init__()
        self.feature_extractor = create_feature_extractor(
            model_name=model_name,
            pretrained=pretrained,
        )
        vit_embed_dim = self.feature_extractor.embed_dim
        self.projection = nn.Conv2d(vit_embed_dim, feature_dim, kernel_size=1)

    def forward(self, uav_image):
        features_2d = self.feature_extractor(uav_image)
        bev_features = self.projection(features_2d)
        return bev_features

class SnapViT(nn.Module):
    """
    The main SnapViT model that combines the two encoders.
    """
    def __init__(self, config):
        super().__init__()
        # Ground and overhead backbones are intentionally shared.
        # Prefer model_name, then vit_model for backward compatibility.
        shared_model_name = config.get('model_name', config.get('vit_model', 'vit_base_patch16_224'))

        # If legacy per-encoder keys are present, enforce they match the shared model.
        if 'ground_model_name' in config and config['ground_model_name'] != shared_model_name:
            raise ValueError(
                "ground_model_name differs from shared model_name/vit_model. "
                "Ground and overhead must use the same model."
            )
        if 'overhead_model_name' in config and config['overhead_model_name'] != shared_model_name:
            raise ValueError(
                "overhead_model_name differs from shared model_name/vit_model. "
                "Ground and overhead must use the same model."
            )

        pretrained_backbones = config.get('pretrained_backbones', True)

        self.ground_encoder = GroundEncoder(
            model_name=shared_model_name,
            feature_dim=config['feature_dim'],
            pretrained=pretrained_backbones,
        )
        self.overhead_encoder = OverheadEncoder(
            model_name=shared_model_name,
            feature_dim=config['feature_dim'],
            pretrained=pretrained_backbones,
        )
        self.temperature = nn.Parameter(torch.ones([]) * 0.07)

    def forward(self, ugv_data, uav_data):
        ground_bev, ground_validity = self.ground_encoder(**ugv_data)
        overhead_bev = self.overhead_encoder(**uav_data)
        return ground_bev, overhead_bev, ground_validity
