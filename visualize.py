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
import matplotlib.pyplot as plt

# Import necessary classes from your project files
from model import SnapViT
from dataset import VineyardDataset
from visualize_dataset_samples import visualize_data

# --- Configuration (should match training script) ---
CONFIG = {
    'vit_model': 'vit_small_patch16_224',
    'train_img_size': (224, 224),
    'feature_dim': 128,
    'num_ugv_views': 8,
    'grid_size': (34, 34, 8),
    'grid_resolution': 0.3,
    'batch_size': 1, # Process one scene at a time
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'use_depth': True,
    'depth_range': (0.0, 5.0), # meters
    'ground_tile_size': 10.0, # meters
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

def create_ugv_collage(ugv_imgs_tensor: torch.Tensor, num_images: int = 4) -> Image.Image:
    """Creates a collage from a sample of UGV images."""
    images = [transforms.ToPILImage()(ugv_imgs_tensor[i]).resize((224, 224)) for i in range(min(ugv_imgs_tensor.shape[0], num_images))]
    
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

def plot_cosine_similarity(cosine_sim, validity_mask, id, output_dir):
    """Plots the cosine similarity heatmap with valid areas highlighted."""
    
    
    plt.figure(figsize=(10, 8))
    plt.imshow(cosine_sim.cpu(), cmap='viridis', vmin=0, vmax=1)
    plt.gca().set_facecolor('black')
    
    plt.colorbar(label='Cosine Similarity')
    
    # Overlay validity mask (assuming it's binary)
    if validity_mask is not None:
        #validity_mask = torch.rot90(validity_mask, k=2, dims=(1, 2))
        valid_mask = validity_mask.squeeze().cpu().numpy() if validity_mask.dim() > 2 else validity_mask.cpu().numpy()
        #plt.contour(valid_mask, colors='red', linewidths=0.5)
    
    plt.title(f'Cosine Similarity Heatmap for Sample {id}')
    plt.xlabel('Overhead BEV Pixels')
    plt.ylabel('Ground BEV Pixels')
    plt.savefig(os.path.join(output_dir, f'{id}_cosine_similarity.png'))
    plt.close()

def main(args):
    print(f"Using device: {CONFIG['device']}")
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Data ---
    image_transforms = transforms.Compose([
        transforms.Resize(CONFIG['train_img_size'], antialias=True),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    depth_transforms = transforms.Compose([
        transforms.Resize(CONFIG['train_img_size'], antialias=True),
        transforms.ConvertImageDtype(torch.float),
    ])
    
    dataset = VineyardDataset(root_dir=args.data_root, config=CONFIG, transforms=image_transforms, depth_transforms=depth_transforms)
    dataloader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=4)
    
    # --- Model ---
    model = SnapViT(CONFIG).to(CONFIG['device'])
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint file not found at {args.checkpoint}")
    model.load_state_dict(torch.load(args.checkpoint, map_location=CONFIG['device']))
    model.eval()
    
    print("Starting visualization...")
    
    with torch.no_grad():
        for i in range(args.num_samples):
            print(f"Visualizing sample {i+1}/{args.num_samples}")
            try:
                batch = next(iter(dataloader))
            except StopIteration:
                print("No more data available in the dataloader.")

            scene_id = f"scene_{i:04d}"
            
            # Create a directory for the current scene's output
            scene_output_dir = os.path.join(args.output_dir, scene_id)
            os.makedirs(scene_output_dir, exist_ok=True)

            # Move data to device
            uav_data = {k: v.to(CONFIG['device']) for k, v in batch['uav_data'].items()}
            ugv_data = {k: v.to(CONFIG['device']) for k, v in batch['ugv_data'].items()}

            # Get normalized tensors
            uav_img_tensor = uav_data['uav_image'][0]
            ugv_imgs_tensor = ugv_data['ugv_images'][0]

            # Denormalize (inverse of Normalize(mean, std)) and convert to PIL
            def denormalize(tensor):
                # tensor: (3, H, W)
                mean = torch.tensor([0.485, 0.456, 0.406], dtype=tensor.dtype, device=tensor.device).view(3,1,1)
                std  = torch.tensor([0.229, 0.224, 0.225], dtype=tensor.dtype, device=tensor.device).view(3,1,1)
                tensor = tensor * std + mean
                tensor = torch.clamp(tensor, 0.0, 1.0)
                #tensor = (tensor * 255).byte().cpu()
                return tensor

            
            uav_img = denormalize(uav_img_tensor)
            ugv_imgs = torch.stack([denormalize(ugv_imgs_tensor[j]) for j in range(ugv_imgs_tensor.shape[0])])
            ugv_depths = ugv_data['ugv_depths'][0]
            camera_intrinsics = ugv_data['intrinsics'][0]
            camera_poses_w2c = ugv_data['camera_poses'][0]
            depth_range = (CONFIG['depth_range'][0], CONFIG['depth_range'][1])
            tile_ground_size = CONFIG['ground_tile_size']

            visualize_data(uav_img=uav_img,
                ugv_imgs=ugv_imgs, 
                ugv_depths=ugv_depths, 
                camera_poses_w2c=camera_poses_w2c, 
                camera_intrinsics=camera_intrinsics, 
                depth_range=depth_range, 
                tile_ground_size=tile_ground_size, 
                id=i, 
                plot_colors=True, 
                voxelize=True, 
                scene_output_dir=scene_output_dir  
                )

            # Forward pass
            ground_bev, overhead_bev, ground_validity = model(ugv_data, uav_data)
            
            # Resize overhead BEV to match ground BEV for comparison if needed
            overhead_bev_resized = F.interpolate(overhead_bev, size=ground_bev.shape[2:], mode='bilinear', align_corners=False)

            if ground_validity is not None:
                ground_bev = ground_bev * ground_validity

            # Convert feature maps to images
            ground_bev_img = feature_map_to_rgb(ground_bev)
            overhead_bev_img = feature_map_to_rgb(overhead_bev_resized)

            #evaluate cosine similarity between each point of the two feature maps
            ground_bev_flat = ground_bev.view(ground_bev.shape[1], -1)
            overhead_bev_flat = overhead_bev_resized.view(overhead_bev_resized.shape[1], -1)
            cosine_sim = F.cosine_similarity(ground_bev_flat, overhead_bev_flat, dim=0)
            cosine_sim = cosine_sim.view(ground_bev.shape[2], ground_bev.shape[3])

            plot_cosine_similarity(cosine_sim, ground_validity, scene_id, output_dir=scene_output_dir)
            # Get the size of the original UAV image for resizing
            #with open(os.path.join(scene_folder_path, 'metadata.json'), 'r') as f:
            #    metadata = json.load(f)
            #original_uav_path = os.path.join(scene_folder_path, metadata['uav_image_path'])
            
            #with Image.open(original_uav_path) as uav_img_for_size:
            #    target_size = uav_img_for_size.size # Get (width, height)
            target_size = (uav_img.shape[2], uav_img.shape[1])  # (width, height)

            # Resize the BEV images to match the original UAV image dimensions
            # Using LANCZOS for high-quality resizing
            ground_bev_img = ground_bev_img.resize(target_size, Image.Resampling.LANCZOS)
            overhead_bev_img = overhead_bev_img.resize(target_size, Image.Resampling.LANCZOS)

            # Save the generated BEV maps
            ground_bev_img.save(os.path.join(scene_output_dir, f"{scene_id}_ground_bev.png"))
            overhead_bev_img.save(os.path.join(scene_output_dir, f"{scene_id}_overhead_bev.png"))

            # Save the original UAV image and a collage of UGV images for context
            # (Metadata was already loaded above)
            # Save the original UAV image
            uav_img_pil = transforms.ToPILImage()(uav_img)
            uav_img_pil.save(os.path.join(scene_output_dir, f"{scene_id}_uav_original.png"))
            
            # Create and save UGV collage
            ugv_collage = create_ugv_collage(ugv_imgs, num_images=4)
            ugv_collage.save(os.path.join(scene_output_dir, f"{scene_id}_ugv_sample_collage.png"))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize SnapViT feature maps.")
    parser.add_argument('--data_root', type=str, default='datasets/vineyard_dataset', help="Path to the root of the processed dataset.")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to the trained model checkpoint (.pth file).")
    parser.add_argument('--output_dir', type=str, default='visualisations', help="Directory to save the output images.")
    parser.add_argument('--num_samples', type=int, default=10, help="Number of samples to visualize. If None, visualizes all samples.")
    
    args = parser.parse_args()
    main(args)