import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm
import os
import csv
import json
import numpy as np

from torch.utils.tensorboard import SummaryWriter

from model import SnapViT
from dataset import VineyardDataset

# --- Configuration ---
CONFIG = {
    'data_root': '/media/hdd/ale_navone/GAIA/tempovine/dataset_tempovine_new',
    'vit_model': 'vit_small_patch16_224', # Use a smaller model for faster training
    'train_img_size': (224, 224),
    'feature_dim': 128,
    'num_ugv_views': 8,
    'grid_size': (34, 34, 8), # Smaller grid for faster training
    'grid_resolution': 0.3, # meters per grid cell
    'batch_size': 8, # Adjust based on your GPU memory
    'learning_rate': 1e-4,
    'epochs': 1000,
    'device': 'cuda:0',
    'val_split_ratio': 0.2, 
    'use_depth': True,
    'depth_range': (0.0, 5.0), # meters
    'ground_tile_size': 10.0, # meters,
    'output_model_path': 'models/tempovine_2026_03_26_consecutive_frames_mixed_loss_lambda_0_5',
    'consecutive_frames': True, # Whether to select consecutive frames for UGV views
    'pixel_loss_weight': 0.5, # Weight for the pixel-level loss component (if implemented)
    'mixed_loss_delay': 5, # Number of epochs to wait before starting to include the pixel-level loss in the total loss calculation
    
}

#TODO: write a proper infoNCE loss function
def info_nce_loss(features1, features2, temperature):
    """
    Calculates the InfoNCE loss for two sets of feature maps.
    """
    B, C, H, W = features1.shape

    # Normalize features for stable cosine similarity
    features1 = F.normalize(features1, p=2, dim=1)
    features2 = F.normalize(features2, p=2, dim=1)

    # Reshape for matrix multiplication
    features1_flat = features1.permute(0, 2, 3, 1).reshape(B * H * W, C)
    features2_flat = features2.permute(0, 2, 3, 1).reshape(B * H * W, C).T

    # Compute similarity matrix
    logits = torch.matmul(features1_flat, features2_flat) / temperature

    # Labels are the diagonal indices
    labels = torch.arange(B * H * W, device=features1.device)

    # Cross-entropy loss
    return F.cross_entropy(logits, labels)

def masked_info_nce_loss(features1, features2, validity_mask, temperature):
    """
    Calculates the InfoNCE loss for two sets of feature maps, considering only valid positions.
    validity_mask: (B, H, W) boolean tensor indicating valid positions
    """
    B, C, H, W = features1.shape

    safe_temperature = torch.clamp(temperature, min=1e-6)

    # Normalize features for stable cosine similarity
    features1 = F.normalize(features1, p=2, dim=1)
    features2 = F.normalize(features2, p=2, dim=1)

    # Reshape for matrix multiplication
    features1_flat = features1.permute(0, 2, 3, 1).reshape(B * H * W, C)
    features2_flat = features2.permute(0, 2, 3, 1).reshape(B * H * W, C).T

    # Compute similarity matrix
    logits = torch.matmul(features1_flat, features2_flat) / safe_temperature

    # Create mask for valid positions
    validity_mask_flat = validity_mask.reshape(-1).bool()  # (B*H*W,)
    valid_indices = torch.where(validity_mask_flat)[0]

    if valid_indices.numel() == 0:
        return logits.sum() * 0.0

    filtered_logits = logits[valid_indices][:, valid_indices]

    if not torch.isfinite(filtered_logits).all():
        filtered_logits = torch.nan_to_num(filtered_logits, nan=0.0, posinf=1e4, neginf=-1e4)

    labels = torch.arange(len(valid_indices), device=features1.device)

    # Cross-entropy loss
    return F.cross_entropy(filtered_logits, labels)

def masked_avg_pool(features, mask):
    """
    features: (B, C, H, W)
    mask: (B, H, W) or (B, 1, H, W) bool
    """
    # Accept both (B,H,W) and (B,1,H,W) validity masks.
    if mask.dim() == 4 and mask.size(1) == 1:
        mask = mask.squeeze(1)
    elif mask.dim() != 3:
        raise ValueError(f"Expected mask shape (B,H,W) or (B,1,H,W), got {tuple(mask.shape)}")

    mask = mask.unsqueeze(1).float()  # (B,1,H,W)

    masked_features = features * mask

    sum_feat = masked_features.sum(dim=(2, 3))           # (B,C)
    valid_counts = mask.sum(dim=(2, 3)).clamp(min=1e-6)  # (B,1)

    return sum_feat / valid_counts



def symmetric_info_nce_loss_masked(ground_bev, overhead_bev, validity_mask, temperature):
    """
    ground_bev:   (B, C, H, W)
    overhead_bev: (B, C, H, W)
    validity_mask:(B, H, W) bool
    """

    safe_temperature = torch.clamp(temperature, min=1e-6)

    # masked global pooling
    ground_feat = masked_avg_pool(ground_bev, validity_mask)
    overhead_feat = masked_avg_pool(overhead_bev, validity_mask)

    # normalize embeddings
    ground_feat = F.normalize(ground_feat, p=2, dim=1)
    overhead_feat = F.normalize(overhead_feat, p=2, dim=1)

    # similarity matrices (B,C) x (C,B) -> (B,B)
    logits_g2o = torch.matmul(ground_feat, overhead_feat.transpose(0, 1)) / safe_temperature
    logits_o2g = torch.matmul(overhead_feat, ground_feat.transpose(0, 1)) / safe_temperature

    # positives on the diagonal
    labels = torch.arange(ground_feat.size(0), device=ground_feat.device)

    loss_g2o = F.cross_entropy(logits_g2o, labels)
    loss_o2g = F.cross_entropy(logits_o2g, labels)

    return 0.5 * (loss_g2o + loss_o2g)


def main():
    print(f"Using device: {CONFIG['device']}")
    best_val_loss = float('inf')

    print(f"Saving models to: {CONFIG['output_model_path']}")
    output_model_path = f"{CONFIG['output_model_path']}/best_model.pth"
    final_model_path = f"{CONFIG['output_model_path']}/final_model.pth"
    history_dir = os.path.join(CONFIG['output_model_path'], 'history')
    history_csv_path = os.path.join(history_dir, 'loss_history.csv')
    history_config_path = os.path.join(history_dir, 'training_config.json')
    if not os.path.exists(CONFIG['output_model_path']):
        os.makedirs(CONFIG['output_model_path'], exist_ok=True)
    if not os.path.exists(history_dir):
        os.makedirs(history_dir, exist_ok=True)

    with open(history_config_path, mode='w') as f:
        json.dump(CONFIG, f, indent=2)

    with open(history_csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'epoch',
            'train_loss',
            'train_pixel_loss',
            'train_global_loss',
            'val_loss',
            'val_pixel_loss',
            'val_global_loss'
        ])

    # --- Data ---

    image_transforms = transforms.Compose([
        transforms.Resize(CONFIG['train_img_size'], antialias=True),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    depth_transforms = transforms.Compose([
        transforms.Resize(CONFIG['train_img_size'], antialias=True), #TODO: try interpolation=transforms.InterpolationMode.NEAREST
        transforms.ConvertImageDtype(torch.float),
    ])

    # --- Dataset Splitting ---
    full_dataset = VineyardDataset(root_dir=CONFIG['data_root'], config=CONFIG, transforms=image_transforms, depth_transforms=depth_transforms, consecutive_frames=CONFIG['consecutive_frames'])
    
    # Ensure dataset is not empty
    if len(full_dataset) == 0:
        raise ValueError("Dataset is empty. Please check the data_root path.")

    val_size = int(CONFIG['val_split_ratio'] * len(full_dataset))
    train_size = len(full_dataset) - val_size

    print(f"Dataset size: {len(full_dataset)}. Splitting into {train_size} training and {val_size} validation samples.")
    
    # Use a generator for reproducible splits
    generator = torch.Generator()#.manual_seed()
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)


    train_dataloader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=8)
    val_dataloader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=8)


    # --- Model ---
    model = SnapViT(CONFIG).to(CONFIG['device'])
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])

    writer = SummaryWriter(log_dir='runs')

    print("Starting training...")
    for epoch in range(CONFIG['epochs']):
        # --- Training Loop ---
        model.train()
        total_train_loss = 0
        total_global_train_loss = 0
        total_pixel_train_loss = 0
        train_progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [Training]", leave=False)

        for batch in train_progress_bar:
            optimizer.zero_grad()

            uav_data = {k: v.to(CONFIG['device']) for k, v in batch['uav_data'].items()}
            ugv_data = {k: v.to(CONFIG['device']) for k, v in batch['ugv_data'].items()}

            ground_bev, overhead_bev, ground_validity = model(ugv_data, uav_data)
            overhead_bev_resized = F.interpolate(overhead_bev, size=ground_bev.shape[2:], mode='bilinear', align_corners=False)

            #loss = info_nce_loss(ground_bev, overhead_bev_resized, model.temperature)
            pixel_loss = masked_info_nce_loss(ground_bev, overhead_bev_resized, ground_validity, model.temperature)  
            global_loss = symmetric_info_nce_loss_masked(ground_bev, overhead_bev_resized, ground_validity, model.temperature) 

            if epoch < CONFIG['mixed_loss_delay']:
                loss = global_loss
            else:
                loss = CONFIG['pixel_loss_weight'] * pixel_loss + (1 - CONFIG['pixel_loss_weight']) * global_loss
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            total_pixel_train_loss += pixel_loss.item()
            total_global_train_loss += global_loss.item()
            train_progress_bar.set_postfix({'loss': loss.item()})

        avg_train_loss = total_train_loss / len(train_dataloader)
        avg_pixel_loss = total_pixel_train_loss / len(train_dataloader)
        avg_global_loss = total_global_train_loss / len(train_dataloader)

        writer.add_scalar('Loss/Train', avg_train_loss, epoch+1)
        writer.add_scalar('PixelLoss/Train', avg_pixel_loss, epoch+1)
        writer.add_scalar('GlobalLoss/Train', avg_global_loss, epoch+1)

        # --- Validation Loop ---
        model.eval()
        total_val_loss = 0
        total_pixel_val_loss = 0
        total_global_val_loss = 0   
        val_progress_bar = tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [Validation]", leave=False)
        
        with torch.no_grad():
            for batch in val_progress_bar:
                uav_data = {k: v.to(CONFIG['device']) for k, v in batch['uav_data'].items()}
                ugv_data = {k: v.to(CONFIG['device']) for k, v in batch['ugv_data'].items()}

                ground_bev, overhead_bev, ground_validity = model(ugv_data, uav_data)
                overhead_bev_resized = F.interpolate(overhead_bev, size=ground_bev.shape[2:], mode='bilinear', align_corners=False)

                #loss = info_nce_loss(ground_bev, overhead_bev_resized, model.temperature)
                pixel_loss = masked_info_nce_loss(ground_bev, overhead_bev_resized, ground_validity, model.temperature)  
                global_loss = symmetric_info_nce_loss_masked(ground_bev, overhead_bev_resized, ground_validity, model.temperature) 
                loss = CONFIG['pixel_loss_weight'] * pixel_loss + (1 - CONFIG['pixel_loss_weight']) * global_loss
                total_val_loss += loss.item()
                total_pixel_val_loss += pixel_loss.item()
                total_global_val_loss += global_loss.item()
                val_progress_bar.set_postfix({'loss': loss.item()})
        
        avg_val_loss = total_val_loss / len(val_dataloader)
        avg_val_pixel_loss = total_pixel_val_loss / len(val_dataloader)
        avg_val_global_loss = total_global_val_loss / len(val_dataloader)
        writer.add_scalar('Loss/Validation', avg_val_loss, epoch+1)
        writer.add_scalar('PixelLoss/Validation', avg_val_pixel_loss, epoch+1)
        writer.add_scalar('GlobalLoss/Validation', avg_val_global_loss, epoch+1)

        with open(history_csv_path, mode='a', newline='') as f:
            csv.writer(f).writerow([
                epoch + 1,
                avg_train_loss,
                avg_pixel_loss,
                avg_global_loss,
                avg_val_loss,
                avg_val_pixel_loss,
                avg_val_global_loss,
            ])

        print(f"Epoch {epoch+1}/{CONFIG['epochs']} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # --- Save the best model ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), output_model_path)
            print(f"  -> New best model found! Saved to {output_model_path} (Val Loss: {best_val_loss:.4f})")
    torch.save(model.state_dict(), final_model_path)
    writer.close()

if __name__ == '__main__':
    main()
