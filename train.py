import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm
import os
import numpy as np

from model import SnapViT
from dataset import VineyardDataset

# --- Configuration ---
CONFIG = {
    'data_root': 'datasets/vineyard_dataset',
    'vit_model': 'vit_small_patch16_224', # Use a smaller model for faster training
    'feature_dim': 128,
    'num_ugv_views': 8,
    'grid_size': (34, 34, 8), # Smaller grid for faster training
    'grid_resolution': 0.3, # meters per grid cell
    'batch_size': 4, # Adjust based on your GPU memory
    'learning_rate': 1e-4,
    'epochs': 1000,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'val_split_ratio': 0.2, # 20% of the data will be used for validation
}

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


def main():
    print(f"Using device: {CONFIG['device']}")
    best_val_loss = float('inf')
    output_model_path = 'models/best_model.pth'
    final_model_path = 'models/final_model.pth'

    # --- Data ---
    image_transforms = transforms.Compose([
        transforms.Resize((224, 224), antialias=True),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # --- Dataset Splitting ---
    full_dataset = VineyardDataset(root_dir=CONFIG['data_root'], config=CONFIG, transforms=image_transforms)
    
    # Ensure dataset is not empty
    if len(full_dataset) == 0:
        raise ValueError("Dataset is empty. Please check the data_root path.")

    val_size = int(CONFIG['val_split_ratio'] * len(full_dataset))
    train_size = len(full_dataset) - val_size

    print(f"Dataset size: {len(full_dataset)}. Splitting into {train_size} training and {val_size} validation samples.")
    
    # Use a generator for reproducible splits
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

    train_dataloader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=4)


    # --- Model ---
    model = SnapViT(CONFIG).to(CONFIG['device'])
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])

    print("Starting training...")
    for epoch in range(CONFIG['epochs']):
        # --- Training Loop ---
        model.train()
        total_train_loss = 0
        train_progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [Training]", leave=False)

        for batch in train_progress_bar:
            optimizer.zero_grad()

            uav_data = {k: v.to(CONFIG['device']) for k, v in batch['uav_data'].items()}
            ugv_data = {k: v.to(CONFIG['device']) for k, v in batch['ugv_data'].items()}

            ground_bev, overhead_bev = model(ugv_data, uav_data)
            overhead_bev_resized = F.interpolate(overhead_bev, size=ground_bev.shape[2:], mode='bilinear', align_corners=False)

            loss = info_nce_loss(ground_bev, overhead_bev_resized, model.temperature)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            train_progress_bar.set_postfix({'loss': loss.item()})

        avg_train_loss = total_train_loss / len(train_dataloader)

        # --- Validation Loop ---
        model.eval()
        total_val_loss = 0
        val_progress_bar = tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [Validation]", leave=False)
        
        with torch.no_grad():
            for batch in val_progress_bar:
                uav_data = {k: v.to(CONFIG['device']) for k, v in batch['uav_data'].items()}
                ugv_data = {k: v.to(CONFIG['device']) for k, v in batch['ugv_data'].items()}

                ground_bev, overhead_bev = model(ugv_data, uav_data)
                overhead_bev_resized = F.interpolate(overhead_bev, size=ground_bev.shape[2:], mode='bilinear', align_corners=False)

                loss = info_nce_loss(ground_bev, overhead_bev_resized, model.temperature)
                total_val_loss += loss.item()
                val_progress_bar.set_postfix({'loss': loss.item()})
        
        avg_val_loss = total_val_loss / len(val_dataloader)

        print(f"Epoch {epoch+1}/{CONFIG['epochs']} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # --- Save the best model ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), output_model_path)
            print(f"  -> New best model found! Saved to {output_model_path} (Val Loss: {best_val_loss:.4f})")
    torch.save(model.state_dict(), final_model_path)

if __name__ == '__main__':
    main()
