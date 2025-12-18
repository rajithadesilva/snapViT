import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm
import os
import argparse 

from model import SnapViT
from dataset import VineyardDataset

# --- Configuration ---
CONFIG = {
    'data_root': 'datasets/vineyard_dataset',
    'vit_model': 'vit_small_patch16_224',
    'feature_dim': 128,
    'num_ugv_views': 8,
    'grid_size': (34, 34, 8),#  (25, 50, 8),
    'grid_resolution': 0.3,
    'batch_size': 4,
    'learning_rate': 1e-3,
    'epochs': 200,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'val_split_ratio': 0.2,
}

def info_nce_loss(features1, features2, temperature):
    """
    Calculates the InfoNCE loss for two sets of feature maps.
    """
    B, C, H, W = features1.shape
    features1 = F.normalize(features1, p=2, dim=1)
    features2 = F.normalize(features2, p=2, dim=1)
    features1_flat = features1.permute(0, 2, 3, 1).reshape(B * H * W, C)
    features2_flat = features2.permute(0, 2, 3, 1).reshape(B * H * W, C).T
    logits = torch.matmul(features1_flat, features2_flat) / temperature
    labels = torch.arange(B * H * W, device=features1.device)
    return F.cross_entropy(logits, labels)


def main():
    parser = argparse.ArgumentParser(description="Train the SnapViT model.")
    parser.add_argument(
        '--use_validation',
        action='store_true',
        help="Enable validation splitting and evaluation. Default is to train on the full dataset."
    )
    args = parser.parse_args()

    print(f"Using device: {CONFIG['device']}")
    best_train_loss = float('inf')
    best_val_loss = float('inf')
    output_model_path = 'models/best_model.pth'
    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)

    # --- Data ---
    image_transforms = transforms.Compose([
        transforms.Resize((224, 224), antialias=True),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    full_dataset = VineyardDataset(root_dir=CONFIG['data_root'], config=CONFIG, transforms=image_transforms)

    if len(full_dataset) == 0:
        raise ValueError("Dataset is empty. Please check the data_root path.")

    # --- Conditional dataset splitting ---
    if args.use_validation:
        val_size = int(CONFIG['val_split_ratio'] * len(full_dataset))
        train_size = len(full_dataset) - val_size
        print(f"Splitting dataset: {train_size} training samples and {val_size} validation samples.")
        generator = torch.Generator().manual_seed(42)
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)
        train_dataloader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=4)
        val_dataloader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=4)
    else:
        print(f"Training on the full dataset with {len(full_dataset)} samples. No validation.")
        train_dataloader = DataLoader(full_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=4)
        val_dataloader = None # Ensure val_dataloader is None if not used

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

        # --- Conditional validation loop ---
        if args.use_validation:
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

            # Save based on validation loss
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), output_model_path)
                print(f"  -> New best model found! Saved to {output_model_path} (Val Loss: {best_val_loss:.4f})")
        else:
            # --- Logic for when validation is disabled ---
            print(f"Epoch {epoch+1}/{CONFIG['epochs']} | Train Loss: {avg_train_loss:.4f}")
            # Save based on training loss
            if avg_train_loss < best_train_loss:
                best_train_loss = avg_train_loss
                torch.save(model.state_dict(), output_model_path)
                print(f"  -> New best model found! Saved to {output_model_path} (Train Loss: {best_train_loss:.4f})")

if __name__ == '__main__':
    main()