import os
import json
import argparse
import numpy as np
from PIL import Image, ImageDraw

def visualize_scene(scene_path, output_dir, tile_ground_size):
    """
    Visualizes the UGV frame locations and orientations on the scene's overhead UAV image.

    Args:
        scene_path (str): Path to the individual scene directory (e.g., 'vineyard_dataset/scene_0001').
        output_dir (str): Directory where the verification images will be saved.
        tile_ground_size (float): The width/height of the UAV tile in meters, used during dataset creation.
    """
    metadata_path = os.path.join(scene_path, 'metadata.json')
    if not os.path.exists(metadata_path):
        print(f"Warning: metadata.json not found in {scene_path}. Skipping.")
        return

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    uav_image_path_relative = metadata.get("uav_image_path")
    if not uav_image_path_relative:
        print(f"Warning: uav_image_path not found in metadata for {scene_path}. Skipping.")
        return
        
    uav_image_path = os.path.join(scene_path, uav_image_path_relative)
    if not os.path.exists(uav_image_path):
        print(f"Warning: UAV image not found at {uav_image_path}. Skipping.")
        return

    # Load the overhead UAV image
    try:
        uav_image = Image.open(uav_image_path).convert("RGB")
        draw = ImageDraw.Draw(uav_image)
        img_width, img_height = uav_image.size
    except Exception as e:
        print(f"Error loading image {uav_image_path}: {e}")
        return

    # Calculate pixels per meter. Assumes square tiles.
    pixels_per_meter = img_width / tile_ground_size

    # Center of the image in pixels
    center_x_px = img_width / 2
    center_y_px = img_height / 2

    ugv_images_metadata = metadata.get("ugv_images", [])
    if not ugv_images_metadata:
        print(f"Note: No UGV images found in metadata for {scene_path}.")

    for ugv_meta in ugv_images_metadata:
        w2c_matrix = np.array(ugv_meta['camera_pose_w2c'])
        
        # Invert the world-to-camera matrix to get camera-to-world
        try:
            c2w_matrix = np.linalg.inv(w2c_matrix)
        except np.linalg.LinAlgError:
            print(f"Warning: Could not invert pose matrix for {ugv_meta['image_path']}. Skipping.")
            continue

        # Extract the local position (translation) in meters
        # This is the UGV's position relative to the UAV image center
        local_x, local_y, _ = c2w_matrix[:3, 3]

        # Convert local meter coordinates to pixel coordinates
        # Y-axis is inverted because image origin (0,0) is top-left
        ugv_px = center_x_px + (local_x * pixels_per_meter)
        ugv_py = center_y_px - (local_y * pixels_per_meter)

        # Extract the rotation to determine the direction of travel
        # Assume camera's X-axis is the forward direction
        # The first column of the rotation matrix represents the X-axis direction vector
        direction_vector = c2w_matrix[:3, 0]
        
        # Project direction onto the 2D plane (x, y)
        dir_x, dir_y = direction_vector[0], direction_vector[1]
        
        # Normalize the direction vector
        norm = np.sqrt(dir_x**2 + dir_y**2)
        if norm == 0:
            continue # Skip if direction is not defined
            
        dir_x /= norm
        dir_y /= norm

        # Define arrow properties
        arrow_length_px = 25  # Length of the arrow in pixels
        arrow_color = "red"
        
        # Calculate arrow endpoint
        end_x = ugv_px + dir_x * arrow_length_px
        end_y = ugv_py - dir_y * arrow_length_px # Invert y for image coordinates

        # Draw the arrow
        draw.line([(ugv_px, ugv_py), (end_x, end_y)], fill=arrow_color, width=15)
        
        # Draw a point at the base of the arrow
        draw.ellipse([(ugv_px-4, ugv_py-4), (ugv_px+4, ugv_py+4)], fill="yellow", outline="black")


    # Save the visualized image
    scene_name = os.path.basename(scene_path)
    output_path = os.path.join(output_dir, f"{scene_name}_verification.png")
    uav_image.save(output_path)
    print(f"Saved verification for {scene_name} to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Visualize UGV poses on UAV images for dataset verification.")
    parser.add_argument('--dataset_dir', type=str, default='datasets/old/vineyard_dataset_10m', help="Path to the root of the processed dataset (e.g., 'vineyard_dataset').")
    parser.add_argument('--output_dir', type=str, default='verification', help="Directory to save the output verification images.")
    parser.add_argument('--tile_ground_size', type=float, default=10.0, help="The width and height of the GeoTIFF tiles in meters, as used in create_dataset.py.")
    
    args = parser.parse_args()

    if not os.path.isdir(args.dataset_dir):
        print(f"Error: Dataset directory not found at '{args.dataset_dir}'")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Starting verification. Output will be saved to '{args.output_dir}'")

    # Find all scene directories
    for scene_name in sorted(os.listdir(args.dataset_dir)):
        scene_path = os.path.join(args.dataset_dir, scene_name)
        if os.path.isdir(scene_path) and scene_name.startswith('scene_'):
            visualize_scene(scene_path, args.output_dir, args.tile_ground_size)

    print("\nVerification process complete.")

if __name__ == '__main__':
    main()