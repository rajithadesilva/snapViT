import os
import torch
import numpy as np
import argparse
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd
from torchvision import transforms

# Import necessary classes from your project files
from model import SnapViT
from dataset import VineyardDataset

# --- Configuration (should match training script for consistency) ---
CONFIG = {
    'vit_model': 'vit_small_patch16_224',
    'feature_dim': 128,
    'num_ugv_views': 8,
    'grid_size': (25, 50, 8),
    'grid_resolution': 0.2,
    'batch_size': 1, # Process one scene at a time
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}

def calculate_masked_cosine_similarity(ground_bev, overhead_bev):
    """
    Calculates the mean cosine similarity between two feature maps,
    but only for the pixels where the ground_bev is not empty (background).

    Args:
        ground_bev (torch.Tensor): The ground BEV feature map (B, C, H, W).
        overhead_bev (torch.Tensor): The overhead BEV feature map (B, C, H, W).

    Returns:
        float: The mean cosine similarity over the valid pixels.
    """
    # Ensure overhead BEV is resized to match ground BEV
    if ground_bev.shape[2:] != overhead_bev.shape[2:]:
        overhead_bev = F.interpolate(overhead_bev, size=ground_bev.shape[2:], mode='bilinear', align_corners=False)

    # Create a mask to identify valid (non-background) pixels in the ground BEV.
    # A pixel is considered valid if its feature vector is not effectively zero.
    with torch.no_grad():
        valid_pixel_mask = torch.sum(torch.abs(ground_bev), dim=1) > 1e-6 # Shape: (B, H, W)

        # If there are no valid pixels, similarity is undefined, return 0.
        if not torch.any(valid_pixel_mask):
            return 0.0

    # Normalize the feature vectors along the channel dimension (L2 norm)
    ground_bev_norm = F.normalize(ground_bev, p=2, dim=1)
    overhead_bev_norm = F.normalize(overhead_bev, p=2, dim=1)

    # Calculate element-wise dot product, which is the cosine similarity for normalized vectors
    # We multiply the feature maps and sum along the channel dimension
    cosine_similarity_map = torch.sum(ground_bev_norm * overhead_bev_norm, dim=1) # Shape: (B, H, W)

    # Apply the mask to the similarity map
    masked_similarity = cosine_similarity_map[valid_pixel_mask]

    # Return the mean of the similarity values for the valid pixels
    return masked_similarity.mean().item()


def main(args):
    print(f"Using device: {CONFIG['device']}")

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
    
    print("Starting similarity quantification...")
    
    results = []
    
    with torch.no_grad():
        for i, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc="Quantifying Scene Similarity"):
            scene_folder_path = dataset.scene_folders[i]
            scene_id = os.path.basename(scene_folder_path)
            
            # Move data to device
            uav_data = {k: v.to(CONFIG['device']) for k, v in batch['uav_data'].items()}
            ugv_data = {k: v.to(CONFIG['device']) for k, v in batch['ugv_data'].items()}

            # Forward pass
            ground_bev, overhead_bev = model(ugv_data, uav_data)
            
            # Calculate the masked cosine similarity
            similarity_score = calculate_masked_cosine_similarity(ground_bev, overhead_bev)
            
            results.append({'scene_id': scene_id, 'cosine_similarity': similarity_score})

    # --- Display Results ---
    if not results:
        print("No scenes were processed.")
        return

    results_df = pd.DataFrame(results)
    average_similarity = results_df['cosine_similarity'].mean()
    
    print("\n--- Similarity Quantification Complete ---")
    print(f"\nResults for each scene (saved to {args.output_file}):")
    print(results_df.to_string(index=False))
    
    print(f"\nAverage Cosine Similarity across all scenes: {average_similarity:.4f}")
    
    # Save results to a CSV file
    results_df.to_csv(args.output_file, index=False)
    print(f"\nDetailed results have been saved to '{args.output_file}'")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Quantify the similarity between UGV and UAV BEV feature maps using a trained SnapViT model.")
    parser.add_argument('--data_root', type=str, default='datasets/row_wise_dataset2', help="Path to the root of the processed dataset.")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to the trained model checkpoint (.pth file).")
    parser.add_argument('--output_file', type=str, default='similarity_scores.csv', help="Path to save the output CSV file with similarity scores.")
    
    args = parser.parse_args()
    main(args)