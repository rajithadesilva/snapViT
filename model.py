import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

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

    def forward(self, x):
        B, C, H, W = x.shape
        patch_embeddings = self.vit.forward_features(x)
        patch_embeddings = patch_embeddings[:, 1:, :]  # Remove [CLS] token
        H_feat, W_feat = H // self.patch_size, W // self.patch_size
        features = patch_embeddings.permute(0, 2, 1).reshape(B, self.embed_dim, H_feat, W_feat)
        return features

class GroundEncoder(nn.Module):
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


    def forward(self, ugv_images, camera_poses, intrinsics, grid_points_3d):
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

class OverheadEncoder(nn.Module):
    """
    Encodes a single top-down image into a BEV feature map.
    """
    def __init__(self, vit_model_name='vit_base_patch16_224', feature_dim=256):
        super().__init__()
        self.feature_extractor = ViTFeatureExtractor(vit_model_name)
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
        self.ground_encoder = GroundEncoder(config['vit_model'], config['feature_dim'])
        self.overhead_encoder = OverheadEncoder(config['vit_model'], config['feature_dim'])
        self.temperature = nn.Parameter(torch.ones([]) * 0.07)

    def forward(self, ugv_data, uav_data):
        ground_bev = self.ground_encoder(**ugv_data)
        overhead_bev = self.overhead_encoder(**uav_data)
        return ground_bev, overhead_bev

