import os
import json
import shutil
import argparse
import pandas as pd
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from pyproj import Proj, transform
from tqdm import tqdm

def get_gps_from_exif(image_path):
    """Extracts GPS info from an image's EXIF data."""
    try:
        image = Image.open(image_path)
        exif_data = image._getexif()
        if not exif_data:
            return None

        gps_info = {}
        for tag, value in exif_data.items():
            decoded_tag = TAGS.get(tag, tag)
            if decoded_tag == "GPSInfo":
                for t in value:
                    sub_decoded = GPSTAGS.get(t, t)
                    gps_info[sub_decoded] = value[t]
        
        if 'GPSLatitude' in gps_info and 'GPSLongitude' in gps_info:
            lat_data = gps_info['GPSLatitude']
            lon_data = gps_info['GPSLongitude']
            
            lat = float(lat_data[0] + lat_data[1]/60 + lat_data[2]/3600)
            lon = float(lon_data[0] + lon_data[1]/60 + lon_data[2]/3600)
            
            if gps_info['GPSLatitudeRef'] == 'S': lat = -lat
            if gps_info['GPSLongitudeRef'] == 'W': lon = -lon
            
            alt = float(gps_info.get('GPSAltitude', 0))

            return lat, lon, alt
            
    except Exception as e:
        print(f"Could not read EXIF from {os.path.basename(image_path)}: {e}")
    return None

def convert_gps_to_local_cartesian(lat, lon, alt, origin_lat, origin_lon, origin_alt):
    """
    Converts GPS coordinates to a local East, North, Up (ENU) Cartesian coordinate system.
    """
    # Use a local Azimuthal Equidistant projection centered on the origin
    proj_string = f"+proj=aeqd +lat_0={origin_lat} +lon_0={origin_lon} +x_0=0 +y_0=0 +ellps=WGS84"
    projector = Proj(proj_string)
    
    # pyproj's transform expects (longitude, latitude)
    x, y = projector(lon, lat)
    z = alt - origin_alt
    
    return x, y, z

def process_real_data(args):
    """
    Processes real UGV and UAV/drone data into the required scene-based structure.
    """
    print("Starting data preprocessing...")
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load drone images and their GPS coordinates
    print("Reading drone images and their GPS locations...")
    drone_data = []
    for fname in tqdm(os.listdir(args.drone_folder)):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            fpath = os.path.join(args.drone_folder, fname)
            gps = get_gps_from_exif(fpath)
            if gps:
                drone_data.append({'path': fpath, 'gps': gps})

    if not drone_data:
        raise ValueError("No drone images with GPS EXIF data found.")

    # 2. Load ground vehicle (UGV) GPS data from CSV
    print("Reading ground vehicle CSV...")
    ugv_df = pd.read_csv(args.ground_csv)
    # Assuming CSV columns are 'image_file_name', 'latitude', 'longitude', 'altitude'
    ugv_locations = {row['image_file_name']: (row['latitude'], row['longitude'], row.get('altitude', 0)) for _, row in ugv_df.iterrows()}

    # 3. Match ground images to drone images to create scenes
    print("Matching ground images to drone images to create scenes...")
    scenes = {i: [] for i in range(len(drone_data))}
    pbar = tqdm(total=len(ugv_locations), desc="Assigning UGV images to scenes")

    # Define a projection for distance calculation (UTM is good for this)
    # We'll just use the first drone image to define the UTM zone
    first_drone_lat, first_drone_lon, _ = drone_data[0]['gps']
    utm_zone = int((first_drone_lon + 180) / 6) + 1
    proj_utm = Proj(proj='utm', zone=utm_zone, ellps='WGS84')

    drone_xy = np.array([proj_utm(d['gps'][1], d['gps'][0]) for d in drone_data])

    for img_name, (lat, lon, alt) in ugv_locations.items():
        ugv_xy = proj_utm(lon, lat)
        distances = np.linalg.norm(drone_xy - ugv_xy, axis=1)
        closest_drone_idx = np.argmin(distances)
        
        if distances[closest_drone_idx] < args.scene_radius:
            scenes[closest_drone_idx].append({'name': img_name, 'gps': (lat, lon, alt)})
        pbar.update(1)
    pbar.close()

    # 4. Create the dataset structure
    print("Building final dataset structure...")
    scene_count = 0
    for drone_idx, ugv_list in scenes.items():
        if not ugv_list:
            continue
        
        scene_count += 1
        scene_id = f'scene_{scene_count:04d}'
        scene_path = os.path.join(args.output_dir, scene_id)
        ugv_out_path = os.path.join(scene_path, 'ugv_images')
        uav_out_path = os.path.join(scene_path, 'uav_image')
        os.makedirs(ugv_out_path, exist_ok=True)
        os.makedirs(uav_out_path, exist_ok=True)
        
        # --- Copy UAV image ---
        drone_info = drone_data[drone_idx]
        uav_fname = os.path.basename(drone_info['path'])
        shutil.copy(drone_info['path'], os.path.join(uav_out_path, uav_fname))
        
        # --- Process UGV images and create metadata ---
        origin_lat, origin_lon, origin_alt = drone_info['gps']
        ugv_metadata = []

        for ugv_info in ugv_list:
            # Copy UGV image
            ugv_src_path = os.path.join(args.ground_folder, ugv_info['name'])
            if not os.path.exists(ugv_src_path):
                print(f"Warning: UGV image not found, skipping: {ugv_src_path}")
                continue
            shutil.copy(ugv_src_path, os.path.join(ugv_out_path, ugv_info['name']))
            
            # Convert UGV GPS to local XYZ
            lat, lon, alt = ugv_info['gps']
            x, y, z = convert_gps_to_local_cartesian(lat, lon, alt, origin_lat, origin_lon, origin_alt)

            # NOTE: This is a simplified pose. It assumes the camera is level and
            # its orientation is fixed. For a real system, you would get orientation
            # from an IMU and construct a proper rotation matrix.
            rotation_matrix = np.eye(3)
            translation_vector = np.array([x, y, z])
            
            w2c_matrix = np.eye(4)
            w2c_matrix[:3, :3] = rotation_matrix
            w2c_matrix[:3, 3] = translation_vector
            
            # This should be world-to-camera, so we need to invert the camera-to-world pose
            w2c_matrix = np.linalg.inv(w2c_matrix).tolist()

            ugv_metadata.append({
                "image_path": os.path.join('ugv_images', ugv_info['name']),
                "camera_intrinsics": [[925.348, 0.0, 639.5],[0.0, 925.348, 359.5],[0.0, 0.0, 1.0]],
                "camera_pose_w2c": w2c_matrix
            })
            
        # --- Create Metadata JSON ---
        metadata = {
            "scene_origin_gps": {"lat": origin_lat, "lon": origin_lon, "alt": origin_alt},
            "uav_image_path": os.path.join('uav_image', uav_fname),
            "ugv_images": ugv_metadata,
        }
        with open(os.path.join(scene_path, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)

    print(f"\nPreprocessing complete. Created {scene_count} scenes in '{args.output_dir}'.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocess UGV/UAV data for SnapViT.")
    parser.add_argument('--drone_folder', type=str, required=True, help="Path to the folder with geotagged drone images.")
    parser.add_argument('--ground_folder', type=str, required=True, help="Path to the folder with ground vehicle images.")
    parser.add_argument('--ground_csv', type=str, required=True, help="Path to the CSV file with ground image names and GPS coordinates.")
    parser.add_argument('--output_dir', type=str, default='vineyard_dataset', help="Path to save the processed dataset.")
    parser.add_argument('--scene_radius', type=float, default=10.0, help="Radius in meters to group ground images under a drone image.")
    
    args = parser.parse_args()
    process_real_data(args)

