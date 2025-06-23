import os
import json
import shutil
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from pyproj import Proj, Transformer
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from datetime import datetime, timezone, timedelta
import pyrealsense2 as rs
import rasterio
from rasterio.windows import Window

# --- CONFIGURATION ---
SYNC_THRESHOLD = 1.0
UGV_HEIGHT = 1.0

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
                for t, v in value.items():
                    sub_decoded = GPSTAGS.get(t, t)
                    gps_info[sub_decoded] = v

        if 'GPSLatitude' in gps_info and 'GPSLongitude' in gps_info:
            lat_data = gps_info['GPSLatitude']
            lon_data = gps_info['GPSLongitude']

            lat = float(lat_data[0] + lat_data[1]/60 + lat_data[2]/3600)
            lon = float(lon_data[0] + lon_data[1]/60 + lon_data[2]/3600)

            if gps_info.get('GPSLatitudeRef') == 'S': lat = -lat
            if gps_info.get('GPSLongitudeRef') == 'W': lon = -lon

            alt = float(gps_info.get('GPSAltitude', 0))

            return lat, lon, alt

    except Exception as e:
        print(f"Could not read EXIF from {os.path.basename(image_path)}: {e}")
    return None

def parse_llh_file(llh_path):
    """
    Parses your specific .llh file format into a pandas DataFrame.
    """
    print(f"Parsing LLH file for raw timestamps: {llh_path}")
    gps_data = []
    datetime_format = "%Y/%m/%d %H:%M:%S.%f"

    with open(llh_path, 'r') as f:
        for line in f:
            if line.startswith('%') or not line.strip():
                continue

            parts = line.strip().split()
            if len(parts) < 4:
                continue

            try:
                datetime_str = f"{parts[0]} {parts[1]}"
                dt_object = datetime.strptime(datetime_str, datetime_format)
                timestamp = dt_object.timestamp()

                lat = float(parts[2])
                lon = float(parts[3])
                alt = float(parts[4])
                gps_data.append({'timestamp': timestamp, 'lat': lat, 'lon': lon, 'alt': alt})
            except (ValueError, IndexError):
                continue

    if not gps_data:
        raise ValueError("No data was parsed from the LLH file.")

    gps_df = pd.DataFrame(gps_data).set_index('timestamp').sort_index()
    return gps_df

def process_geotiff(geotiff_path, ground_size_m=3.0, output_dir="temp_drone_tiles"):
    """
    Splits a large GeoTIFF into smaller tiles and gets the projected coordinate for each.
    Returns the drone data and the EPSG code of the GeoTIFF's coordinate system.
    """
    print(f"Processing GeoTIFF: {geotiff_path}")
    os.makedirs(output_dir, exist_ok=True)
    drone_data = []

    with rasterio.open(geotiff_path) as src:
        drone_crs = src.crs.to_epsg()
        print(f"GeoTIFF Coordinate Reference System (CRS) is EPSG:{drone_crs}")

        gsd_x = src.transform.a
        gsd_y = -src.transform.e

        if gsd_x <= 0 or gsd_y <= 0:
            raise ValueError("Could not determine a valid resolution from the GeoTIFF.")

        tile_w_px = int(round(ground_size_m / gsd_x))
        tile_h_px = int(round(ground_size_m / gsd_y))
        
        print(f"GeoTIFF GSD (meters/pixel): x={gsd_x:.4f}, y={gsd_y:.4f}")
        print(f"Calculated tile size: {tile_w_px}x{tile_h_px} pixels for {ground_size_m}m coverage")

        width, height = src.width, src.height

        for j in tqdm(range(0, height, tile_h_px), desc="Tiling GeoTIFF"):
            for i in range(0, width, tile_w_px):
                window = Window(i, j, tile_w_px, tile_h_px)
                tile = src.read(window=window)

                if tile.shape[0] >= 3:
                     tile_for_pil = np.moveaxis(tile[:3], 0, -1)
                else:
                    tile_for_pil = tile[0]

                if np.count_nonzero(tile_for_pil) < tile_for_pil.size * 0.1:
                    continue
                
                center_x, center_y = src.xy(j + tile_h_px / 2, i + tile_w_px / 2)
                
                # We need the original lat/lon for metadata, so we transform back
                transformer = Transformer.from_crs(f"epsg:{drone_crs}", "epsg:4326", always_xy=True)
                lon, lat = transformer.transform(center_x, center_y)
                alt = 0.0

                tile_img = Image.fromarray(tile_for_pil)
                tile_name = f"tile_{j}_{i}.png"
                tile_path = os.path.join(output_dir, tile_name)
                tile_img.save(tile_path)

                drone_data.append({
                    'path': tile_path,
                    'xy': (center_x, center_y), 
                    'gps': (lat, lon, alt)
                })

    return drone_data, drone_crs

def extract_frames_from_bag(bag_file, output_folder, frame_skip):
    """
    Extracts and saves color image frames from a Realsense .bag file,
    and returns the camera intrinsics.
    """
    print(f"Extracting frames from {bag_file} every {frame_skip} frames...")
    os.makedirs(output_folder, exist_ok=True)
    extracted_frames = []
    intrinsics_matrix = None

    pipeline = rs.pipeline()
    config = rs.config()
    rs.config.enable_device_from_file(config, bag_file, repeat_playback=False)
    config.enable_stream(rs.stream.color)

    try:
        profile = pipeline.start(config)

        # Get intrinsics from the color stream profile
        color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intrinsics = color_stream.get_intrinsics()
        intrinsics_matrix = [
            [intrinsics.fx, 0, intrinsics.ppx],
            [0, intrinsics.fy, intrinsics.ppy],
            [0, 0, 1]
        ]
        print(f"Successfully extracted camera intrinsics: {intrinsics_matrix}")

        playback = profile.get_device().as_playback()
        playback.set_real_time(False)

        frame_number = 0
        while True:
            # Use a timeout to prevent waiting indefinitely at the end of the file
            frames = pipeline.wait_for_frames(timeout_ms=1000)

            if frame_number % frame_skip == 0:
                color_frame = frames.get_color_frame()
                if color_frame:
                    timestamp = color_frame.get_timestamp() / 1000.0

                    timestamp_str = f"{timestamp:.4f}".replace('.', '_')
                    image_name = f"frame_{timestamp_str}.png"
                    image_path = os.path.join(output_folder, image_name)

                    image_data = np.asanyarray(color_frame.get_data())
                    img = Image.fromarray(image_data)
                    img.save(image_path)

                    extracted_frames.append({'timestamp': timestamp, 'name': image_name})

            frame_number += 1
    except RuntimeError:
        print("End of bag file reached or no frames were received in the timeout window.")
    finally:
        pipeline.stop()

    if not extracted_frames:
        print("Warning: No frames were extracted from the bag file. Check if the color stream topic exists.")
    
    return sorted(extracted_frames, key=lambda x: x['timestamp']), intrinsics_matrix

def calculate_rotation_from_gps_window(current_gps_idx, gps_data_list, window_size=5):
    """
    Calculates the camera's rotation matrix based on its direction of travel.
    """
    start_idx = max(0, current_gps_idx - window_size // 2)
    end_idx = min(len(gps_data_list) - 1, current_gps_idx + window_size // 2)

    if start_idx >= end_idx:
        return np.eye(3)

    transformer = Transformer.from_crs("epsg:4326", "epsg:32630", always_xy=True)
    
    points_xy = [transformer.transform(p['lon'], p['lat']) for p in gps_data_list[start_idx:end_idx+1]]

    if len(points_xy) < 2:
        return np.eye(3)

    direction_vector = np.array(points_xy[-1]) - np.array(points_xy[0])

    if np.linalg.norm(direction_vector) < 1e-6:
        return np.eye(3)

    yaw = np.arctan2(direction_vector[1], direction_vector[0])
    rotation = R.from_euler('z', yaw, degrees=False)

    return rotation.as_matrix()

def process_real_data(args):
    """
    Main processing function to create the dataset.
    """
    print("Starting data preprocessing...")
    os.makedirs(args.output_dir, exist_ok=True)
    ugv_temp_folder = os.path.join(args.output_dir, "temp_ugv_extracted_images")
    drone_temp_folder = os.path.join(args.output_dir, "temp_drone_tiles")

    # 1. DRONE DATA LOADING
    drone_crs = None
    if args.drone_geotiff:
        drone_data, drone_crs = process_geotiff(args.drone_geotiff, ground_size_m=args.tile_ground_size, output_dir=drone_temp_folder)
        drone_xy = np.array([d['xy'] for d in drone_data])
    elif args.drone_folder:
        print("Reading drone images from folder and their GPS locations...")
        drone_data = []
        for fname in tqdm(os.listdir(args.drone_folder), desc="Reading drone images"):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                fpath = os.path.join(args.drone_folder, fname)
                gps = get_gps_from_exif(fpath)
                if gps:
                    drone_data.append({'path': fpath, 'gps': gps})
        # Use the CRS found in your debug data if a GeoTIFF isn't provided.
        drone_crs = 32630 
        print(f"Assuming drone imagery uses EPSG:{drone_crs}")
        transformer = Transformer.from_crs("epsg:4326", f"epsg:{drone_crs}", always_xy=True)
        for d in drone_data:
            d['xy'] = transformer.transform(d['gps'][1], d['gps'][0])
        drone_xy = np.array([d['xy'] for d in drone_data])
    else:
        raise ValueError("You must provide either --drone_folder or --drone_geotiff")

    if not drone_data:
        raise ValueError("No drone images could be loaded.")

    # 2. Parse the UGV's .llh file
    gps_df = parse_llh_file(args.ground_llh)
    gps_data_list = gps_df.reset_index().to_dict('records')

    # 3. Extract UGV frames and intrinsics
    ugv_frames, ugv_intrinsics = extract_frames_from_bag(args.ground_rosbag, ugv_temp_folder, args.frame_skip)
    if not ugv_frames or gps_df.empty:
        raise ValueError("Cannot process, a data source (UGV frames or GPS) is empty.")
    if ugv_intrinsics is None:
        raise ValueError("Could not extract camera intrinsics from the rosbag file. Please check the bag file integrity and topics.")

    # 4. DYNAMIC OFFSET CALCULATION
    first_bag_ts = ugv_frames[0]['timestamp']
    first_gps_ts = gps_df.index.min()
    
    # The offset is the direct difference between the first camera timestamp and the first GPS timestamp.
    # This aligns the two timelines without relying on an inaccurate, hardcoded lag value.
    total_offset = first_bag_ts - first_gps_ts
    print(f"First camera timestamp (s): {first_bag_ts}")
    print(f"First GPS timestamp (s):    {first_gps_ts}")
    print(f"Calculated timeline offset (s): {total_offset:.4f}")

    # 5. SYNCHRONIZE
    print("Synchronizing UGV frames with GPS data...")
    ugv_synced_data = []
    gps_timestamps = gps_df.index.to_numpy()
    for frame in tqdm(ugv_frames, desc="Syncing frames to GPS"):
        # Apply the calculated offset to align the frame's timestamp with the GPS timeline
        adjusted_frame_ts = frame['timestamp'] - total_offset
        
        # Find the closest GPS point in time
        idx = np.searchsorted(gps_timestamps, adjusted_frame_ts, side='right')
        if idx > 0:
            best_match_idx = idx - 1
            # Check if the time difference is within the acceptable threshold
            if abs(adjusted_frame_ts - gps_timestamps[best_match_idx]) < SYNC_THRESHOLD:
                best_match_data = gps_data_list[best_match_idx]
                rotation_matrix = calculate_rotation_from_gps_window(best_match_idx, gps_data_list)
                ugv_synced_data.append({
                    'name': frame['name'],
                    'gps': (best_match_data['lat'], best_match_data['lon'], best_match_data['alt']),
                    'rotation_matrix': rotation_matrix
                })

    # 6. MATCH SCENES
    print(f"Found {len(ugv_synced_data)} synchronized UGV frames. Now matching to drone scenes...")
    scenes = {i: [] for i in range(len(drone_data))}

    if ugv_synced_data:
        ugv_transformer = Transformer.from_crs("epsg:4326", f"epsg:{drone_crs}", always_xy=True)
        for ugv_info in tqdm(ugv_synced_data, desc="Assigning UGV images to scenes"):
            lon, lat = ugv_info['gps'][1], ugv_info['gps'][0]
            ugv_x, ugv_y = ugv_transformer.transform(lon, lat)
            ugv_info['xy'] = (ugv_x, ugv_y)
            distances = np.linalg.norm(drone_xy - np.array([ugv_x, ugv_y]), axis=1)
            closest_drone_idx = np.argmin(distances)
            if distances[closest_drone_idx] < args.scene_radius:
                scenes[closest_drone_idx].append(ugv_info)

    # 7. CREATE DATASET
    print("Building final dataset structure...")
    scene_count = 0
    for drone_idx, ugv_list in scenes.items():
        if not ugv_list: continue
        scene_count += 1
        scene_id = f'scene_{scene_count:04d}'
        scene_path = os.path.join(args.output_dir, scene_id)
        ugv_out_path = os.path.join(scene_path, 'ugv_images')
        uav_out_path = os.path.join(scene_path, 'uav_image')
        os.makedirs(ugv_out_path, exist_ok=True)
        os.makedirs(uav_out_path, exist_ok=True)

        drone_info = drone_data[drone_idx]
        uav_fname = os.path.basename(drone_info['path'])
        shutil.copy(drone_info['path'], os.path.join(uav_out_path, uav_fname))

        origin_x, origin_y = drone_info['xy']
        origin_lat, origin_lon, origin_alt = drone_info['gps']
        ugv_metadata = []

        for ugv_info in ugv_list:
            ugv_src_path = os.path.join(ugv_temp_folder, ugv_info['name'])
            if not os.path.exists(ugv_src_path): continue
            shutil.copy(ugv_src_path, os.path.join(ugv_out_path, ugv_info['name']))

            ugv_x, ugv_y = ugv_info['xy']
            local_x = ugv_x - origin_x
            local_y = ugv_y - origin_y
            local_z = UGV_HEIGHT #ugv_info['gps'][2] - origin_alt 
            
            rotation_matrix = ugv_info['rotation_matrix']
            translation_vector = np.array([local_x, local_y, local_z])
            axis_correction_matrix = np.array([
                [0, 0, 1],
                [0, 1, 0],
                [-1, 0, 0]
            ])
            final_rotation = rotation_matrix @ axis_correction_matrix

            c2w_matrix = np.eye(4)
            c2w_matrix[:3, :3] = final_rotation
            c2w_matrix[:3, 3] = translation_vector
            w2c_matrix = np.linalg.inv(c2w_matrix).tolist()

            ugv_metadata.append({
                "image_path": os.path.join('ugv_images', ugv_info['name']),
                "camera_intrinsics": ugv_intrinsics,
                "camera_pose_w2c": w2c_matrix
            })

        metadata = {
            "scene_origin_gps": {"lat": origin_lat, "lon": origin_lon, "alt": origin_alt},
            "uav_image_path": os.path.join('uav_image', uav_fname),
            "ugv_images": ugv_metadata,
        }
        with open(os.path.join(scene_path, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)

    print(f"\nPreprocessing complete. Created {scene_count} scenes in '{args.output_dir}'.")
    if os.path.exists(ugv_temp_folder):
        print(f"Cleaning up temporary directory: {ugv_temp_folder}")
        shutil.rmtree(ugv_temp_folder)
    if args.drone_geotiff and os.path.exists(drone_temp_folder):
        print(f"Cleaning up temporary drone tile directory: {drone_temp_folder}")
        shutil.rmtree(drone_temp_folder)
    print("Done.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocess UGV/UAV data for dataset creation.")
    
    drone_source_group = parser.add_mutually_exclusive_group(required=True)
    drone_source_group.add_argument('--drone_folder', type=str, help="Path to the folder with geotagged drone images.")
    drone_source_group.add_argument('--drone_geotiff', type=str, help="Path to a single GeoTIFF orthophoto to be tiled.")
    
    parser.add_argument('--ground_rosbag', type=str, required=True, help="Path to the Realsense .bag file.")
    parser.add_argument('--ground_llh', type=str, required=True, help="Path to the .llh file with ground vehicle GPS coordinates.")
    parser.add_argument('--output_dir', type=str, default='processed_dataset', help="Path to save the processed dataset.")
    parser.add_argument('--scene_radius', type=float, default=15.0, help="Radius in meters to group ground images under a drone image.")
    parser.add_argument('--frame_skip', type=int, default=1, help="Process only every N-th frame from the rosbag.")
    parser.add_argument('--tile_ground_size', type=float, default=10.0, help="The desired width and height of the GeoTIFF tiles in meters.")

    args = parser.parse_args()
    process_real_data(args)