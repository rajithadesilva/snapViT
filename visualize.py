import os
import torch
import numpy as np
import argparse
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sklearn.decomposition import PCA
from tqdm import tqdm
import shutil
import json

# Import necessary classes from your project files
from model import SnapViT
from dataset import VineyardDataset

# --- Configuration (should match training script) ---
CONFIG = {
    'vit_model': 'vit_small_patch16_224',
    'feature_dim': 128,
    'num_ugv_views': 8,
    'grid_size': (34, 34, 8),
    'grid_resolution': 0.3,
    'batch_size': 1, # Process one scene at a time
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}

def feature_map_to_rgb(feature_map: torch.Tensor) -> Image.Image:
    """
    Converts a high-dimensional feature map to an RGB image using PCA.
    
    Args:
        feature_map (torch.Tensor): A tensor of shape (1, C, H, W).
        
    Returns:
        PIL.Image.Image: The resulting RGB image.
    """
    if feature_map.dim() != 4 or feature_map.shape[0] != 1:
        raise ValueError("Input tensor must have shape (1, C, H, W)")

    # Remove batch dimension and move to CPU
    fm = feature_map.squeeze(0).cpu().detach().numpy()
    C, H, W = fm.shape
    
    # Reshape for PCA: (H*W, C)
    fm_reshaped = fm.reshape(C, H * W).T
    
    # Apply PCA to reduce from C to 3 dimensions
    pca = PCA(n_components=3)
    principal_components = pca.fit_transform(fm_reshaped)
    
    # Reshape back to image format (H, W, 3)
    img_array = principal_components.reshape(H, W, 3)
    
    # Normalize each channel to the [0, 255] range
    normalized_array = np.zeros_like(img_array, dtype=np.uint8)
    for i in range(3):
        channel = img_array[:, :, i]
        min_val, max_val = channel.min(), channel.max()
        if max_val > min_val:
            normalized_array[:, :, i] = ((channel - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        else:
            normalized_array[:, :, i] = np.zeros_like(channel, dtype=np.uint8)
            
    return Image.fromarray(normalized_array)

def create_ugv_collage(scene_path: str, metadata: dict, num_images: int = 4) -> Image.Image:
    """Creates a collage from a sample of UGV images."""
    image_paths = [os.path.join(scene_path, view['image_path']) for view in metadata['ugv_images']]
    sample_paths = image_paths[:min(len(image_paths), num_images)]
    
    images = [Image.open(p).resize((224, 224)) for p in sample_paths]
    
    if not images:
        return Image.new('RGB', (448, 224), 'black')
        
    width = images[0].width
    height = images[0].height
    
    # Create a 2x2 grid
    collage = Image.new('RGB', (width * 2, height * 2))
    
    positions = [(0, 0), (width, 0), (0, height), (width, height)]
    for img, pos in zip(images, positions):
        collage.paste(img, pos)
        
    return collage


def main(args):
    print(f"Using device: {CONFIG['device']}")
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Data ---
    image_transforms = transforms.Compose([
        transforms.Resize((224, 224), antialias=True),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = VineyardDataset(root_dir=args.data_root, config=CONFIG, transforms=image_transforms)
    dataloader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=4)
    
    # --- Model ---
    model = SnapViT(CONFIG).to(CONFIG['device'])
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint file not found at {args.checkpoint}")
    model.load_state_dict(torch.load(args.checkpoint, map_location=CONFIG['device']))
    model.eval()
    
    print("Starting visualization...")
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Generating visualizations")
    
    with torch.no_grad():
        for i, batch in progress_bar:
            scene_folder_path = dataset.scene_folders[i]
            scene_id = os.path.basename(scene_folder_path)
            
            # Create a directory for the current scene's output
            scene_output_dir = os.path.join(args.output_dir, scene_id)
            os.makedirs(scene_output_dir, exist_ok=True)

            # Move data to device
            uav_data = {k: v.to(CONFIG['device']) for k, v in batch['uav_data'].items()}
            ugv_data = {k: v.to(CONFIG['device']) for k, v in batch['ugv_data'].items()}

            # Forward pass
            ground_bev, overhead_bev = model(ugv_data, uav_data)
            
            # Resize overhead BEV to match ground BEV for comparison if needed
            overhead_bev_resized = F.interpolate(overhead_bev, size=ground_bev.shape[2:], mode='bilinear', align_corners=False)

            # Convert feature maps to images
            ground_bev_img = feature_map_to_rgb(ground_bev)
            overhead_bev_img = feature_map_to_rgb(overhead_bev_resized)

            # Get the size of the original UAV image for resizing
            with open(os.path.join(scene_folder_path, 'metadata.json'), 'r') as f:
                metadata = json.load(f)
            original_uav_path = os.path.join(scene_folder_path, metadata['uav_image_path'])
            
            with Image.open(original_uav_path) as uav_img_for_size:
                target_size = uav_img_for_size.size # Get (width, height)

            # Resize the BEV images to match the original UAV image dimensions
            # Using LANCZOS for high-quality resizing
            ground_bev_img = ground_bev_img.resize(target_size, Image.Resampling.LANCZOS)
            overhead_bev_img = overhead_bev_img.resize(target_size, Image.Resampling.LANCZOS)

            # Save the generated BEV maps
            ground_bev_img.save(os.path.join(scene_output_dir, f"{scene_id}_ground_bev.png"))
            overhead_bev_img.save(os.path.join(scene_output_dir, f"{scene_id}_overhead_bev.png"))

            # Save the original UAV image and a collage of UGV images for context
            # (Metadata was already loaded above)
            shutil.copy(original_uav_path, os.path.join(scene_output_dir, f"{scene_id}_uav_original.png"))
            
            # Create and save UGV collage
            ugv_collage = create_ugv_collage(scene_folder_path, metadata)
            ugv_collage.save(os.path.join(scene_output_dir, f"{scene_id}_ugv_sample_collage.png"))

    print(f"\nVisualizations saved to '{args.output_dir}'.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize SnapViT feature maps.")
    parser.add_argument('--data_root', type=str, default='datasets/vineyard_dataset', help="Path to the root of the processed dataset.")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to the trained model checkpoint (.pth file).")
    parser.add_argument('--output_dir', type=str, default='visualisations', help="Directory to save the output images.")
    
    args = parser.parse_args()
    main(args)