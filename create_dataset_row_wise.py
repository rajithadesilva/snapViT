import os
import json
import shutil
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from pyproj import Proj, Transformer
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from datetime import datetime
import pyrealsense2 as rs
import rasterio
from rasterio.mask import mask
from shapely.geometry import Polygon

# --- CONFIGURATION ---
SYNC_THRESHOLD = 1.0  # Max time difference (seconds) for UGV frame-GPS sync
UGV_HEIGHT = 0.5      # Assumed height of the UGV camera in meters
CROP_HALF_WIDTH = 2.5 # Fixed half-width of the crop area in meters (total width will be 3.0m)

def parse_llh_file(llh_path):
    """Parses your specific .llh file format into a pandas DataFrame."""
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
                dt_object = datetime.strptime(f"{parts[0]} {parts[1]}", datetime_format)
                timestamp = dt_object.timestamp()
                gps_data.append({
                    'timestamp': timestamp,
                    'lat': float(parts[2]),
                    'lon': float(parts[3]),
                    'alt': float(parts[4])
                })
            except (ValueError, IndexError):
                continue
    if not gps_data:
        raise ValueError("No data was parsed from the LLH file.")
    return pd.DataFrame(gps_data).set_index('timestamp').sort_index()

def process_ortho_with_geojson(geotiff_path, geojson_path, output_dir="temp_drone_rows"):
    """
    Crops a GeoTIFF based on row definitions from a GeoJSON file with a fixed width.
    """
    print(f"Processing GeoTIFF '{geotiff_path}' using '{geojson_path}'")
    os.makedirs(output_dir, exist_ok=True)
    
    with open(geojson_path, 'r') as f:
        geojson_data = json.load(f)
    
    rows = {}
    for feature in geojson_data['features']:
        props = feature['properties']
        row_post_id = props.get('row_post_id') or props.get('feature_name')
        if not row_post_id or not row_post_id.startswith('row_'):
            continue
        
        parts = row_post_id.split('_')
        row_num = int(parts[1])
        post_num = int(parts[3])
        
        if row_num not in rows:
            rows[row_num] = {}
        rows[row_num][post_num] = feature['geometry']['coordinates']

    print(f"Found definitions for {len(rows)} rows in the GeoJSON file.")

    drone_data = []
    with rasterio.open(geotiff_path) as src:
        ortho_crs = src.crs
        transformer_to_ortho = Transformer.from_crs("epsg:4326", ortho_crs, always_xy=True)
        transformer_to_wgs84 = Transformer.from_crs(ortho_crs, "epsg:4326", always_xy=True)

        for row_num, posts in tqdm(sorted(rows.items()), desc="Cropping rows from orthophoto"):
            if 0 not in posts or 3 not in posts:
                print(f"Warning: Skipping row {row_num}, missing post 0 or post 3.")
                continue

            p0_proj = np.array(transformer_to_ortho.transform(posts[0][0], posts[0][1]))
            p3_proj = np.array(transformer_to_ortho.transform(posts[3][0], posts[3][1]))

            direction_vector = p3_proj - p0_proj
            
            normal_vector = np.array([-direction_vector[1], direction_vector[0]])
            
            norm = np.linalg.norm(normal_vector)
            if norm > 0:
                 normal_vector = (normal_vector / norm) * CROP_HALF_WIDTH
            
            c1 = p0_proj - normal_vector
            c2 = p3_proj - normal_vector
            c3 = p3_proj + normal_vector
            c4 = p0_proj + normal_vector

            poly = Polygon([c1, c2, c3, c4])

            try:
                crop_img, crop_transform = mask(src, [poly], crop=True)
                meta = src.meta.copy()
                meta.update({
                    "driver": "PNG", "height": crop_img.shape[1], "width": crop_img.shape[2],
                    "transform": crop_transform, "count": 3
                })

                row_fname = f"row_{row_num:04d}.png"
                row_fpath = os.path.join(output_dir, row_fname)
                with rasterio.open(row_fpath, "w", **meta) as dest:
                    dest.write(crop_img[:3, :, :])

                center_proj = poly.centroid.coords[0]
                center_lon, center_lat = transformer_to_wgs84.transform(center_proj[0], center_proj[1])
                
                drone_data.append({
                    'path': row_fpath,
                    'xy': center_proj,
                    'gps': (center_lat, center_lon, 0),
                    'p0_proj': p0_proj,
                    'p3_proj': p3_proj
                })
            except ValueError as e:
                print(f"Warning: Could not crop row {row_num}. Error: {e}")
                continue

    return drone_data, ortho_crs.to_epsg()

def extract_frames_from_bag(bag_file, output_folder, frame_skip):
    """Extracts frames and intrinsics from a ROS bag file."""
    print(f"Extracting frames from {bag_file} every {frame_skip} frames...")
    os.makedirs(output_folder, exist_ok=True)
    extracted_frames, intrinsics_matrix = [], None
    pipeline = rs.pipeline()
    config = rs.config()
    rs.config.enable_device_from_file(config, bag_file, repeat_playback=False)
    config.enable_stream(rs.stream.color)
    try:
        profile = pipeline.start(config)
        color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intrinsics = color_stream.get_intrinsics()
        intrinsics_matrix = [[intrinsics.fx, 0, intrinsics.ppx], [0, intrinsics.fy, intrinsics.ppy], [0, 0, 1]]
        playback = profile.get_device().as_playback()
        playback.set_real_time(False)
        frame_number = 0
        while True:
            frames = pipeline.wait_for_frames(timeout_ms=1000)
            if frame_number % frame_skip == 0:
                color_frame = frames.get_color_frame()
                if color_frame:
                    timestamp = color_frame.get_timestamp() / 1000.0
                    timestamp_str = f"{timestamp:.4f}".replace('.', '_')
                    image_name = f"frame_{timestamp_str}.png"
                    image_path = os.path.join(output_folder, image_name)
                    Image.fromarray(np.asanyarray(color_frame.get_data())).save(image_path)
                    extracted_frames.append({'timestamp': timestamp, 'name': image_name})
            frame_number += 1
    except RuntimeError:
        print("End of bag file reached.")
    finally:
        pipeline.stop()
    return sorted(extracted_frames, key=lambda x: x['timestamp']), intrinsics_matrix

def calculate_rotation_from_gps_window(current_gps_idx, gps_data_list, window_size=5):
    """Calculates camera rotation from GPS travel direction."""
    start_idx, end_idx = max(0, current_gps_idx - window_size // 2), min(len(gps_data_list) - 1, current_gps_idx + window_size // 2)
    if start_idx >= end_idx: return np.eye(3)
    transformer = Transformer.from_crs("epsg:4326", "epsg:32630", always_xy=True)
    points_xy = [transformer.transform(p['lon'], p['lat']) for p in gps_data_list[start_idx:end_idx+1]]
    if len(points_xy) < 2: return np.eye(3)
    direction_vector = np.array(points_xy[-1]) - np.array(points_xy[0])
    if np.linalg.norm(direction_vector) < 1e-6: return np.eye(3)
    yaw = np.arctan2(direction_vector[1], direction_vector[0])
    return R.from_euler('z', yaw).as_matrix()

def main(args):
    """Main processing function to create the dataset."""
    print("Starting data preprocessing...")
    os.makedirs(args.output_dir, exist_ok=True)
    ugv_temp_folder = os.path.join(args.output_dir, "temp_ugv_extracted_images")
    drone_temp_folder = os.path.join(args.output_dir, "temp_drone_rows")

    drone_data, drone_crs = process_ortho_with_geojson(
        args.drone_geotiff, args.geojson_path, output_dir=drone_temp_folder
    )
    if not drone_data: raise ValueError("No drone row images could be created.")

    gps_df = parse_llh_file(args.ground_llh)
    gps_data_list = gps_df.reset_index().to_dict('records')
    ugv_frames, ugv_intrinsics = extract_frames_from_bag(args.ground_rosbag, ugv_temp_folder, args.frame_skip)
    if not ugv_frames or ugv_intrinsics is None: raise ValueError("Failed to extract UGV data.")

    print("Synchronizing UGV frames with GPS data...")
    ugv_synced_data = []
    time_offset = ugv_frames[0]['timestamp'] - gps_df.index.min()
    gps_timestamps = gps_df.index.to_numpy()
    for frame in tqdm(ugv_frames, desc="Syncing frames"):
        adjusted_frame_ts = frame['timestamp'] - time_offset
        idx = np.searchsorted(gps_timestamps, adjusted_frame_ts, side='right')
        if idx > 0 and abs(adjusted_frame_ts - gps_timestamps[idx - 1]) < SYNC_THRESHOLD:
            match_idx = idx - 1
            ugv_synced_data.append({
                'name': frame['name'],
                'gps': (gps_data_list[match_idx]['lat'], gps_data_list[match_idx]['lon'], gps_data_list[match_idx]['alt']),
                'rotation_matrix': calculate_rotation_from_gps_window(match_idx, gps_data_list)
            })

    print(f"Found {len(ugv_synced_data)} UGV frames. Matching to drone scenes via geometric check...")
    scenes = {i: [] for i in range(len(drone_data))}
    ugv_transformer = Transformer.from_crs("epsg:4326", f"epsg:{drone_crs}", always_xy=True)

    for ugv_info in tqdm(ugv_synced_data, desc="Assigning UGV images to scenes"):
        ugv_pos_proj = np.array(ugv_transformer.transform(ugv_info['gps'][1], ugv_info['gps'][0]))
        ugv_info['xy'] = ugv_pos_proj

        for i, row_data in enumerate(drone_data):
            p0, p3 = row_data['p0_proj'], row_data['p3_proj']
            line_vec, ugv_vec = p3 - p0, ugv_pos_proj - p0
            line_len_sq = np.dot(line_vec, line_vec)
            if line_len_sq == 0: continue

            t = np.dot(ugv_vec, line_vec) / line_len_sq
            if 0 <= t <= 1:
                projection_point = p0 + t * line_vec
                lateral_dist = np.linalg.norm(ugv_pos_proj - projection_point)
                if lateral_dist <= CROP_HALF_WIDTH:
                    scenes[i].append(ugv_info)
                    break 

    print("Building final dataset structure...")
    scene_count = 0
    for drone_idx, ugv_list in scenes.items():
        if not ugv_list: continue
        scene_count += 1
        scene_id = f'scene_{scene_count:04d}'
        scene_path = os.path.join(args.output_dir, scene_id)
        ugv_out_path, uav_out_path = os.path.join(scene_path, 'ugv_images'), os.path.join(scene_path, 'uav_image')
        os.makedirs(ugv_out_path, exist_ok=True)
        os.makedirs(uav_out_path, exist_ok=True)

        drone_info = drone_data[drone_idx]
        uav_fname = os.path.basename(drone_info['path'])
        shutil.copy(drone_info['path'], os.path.join(uav_out_path, uav_fname))
        origin_x, origin_y = drone_info['xy']
        ugv_metadata = []

        for ugv_info in ugv_list:
            shutil.copy(os.path.join(ugv_temp_folder, ugv_info['name']), ugv_out_path)
            local_x, local_y = ugv_info['xy'][0] - origin_x, ugv_info['xy'][1] - origin_y
            translation_vector = np.array([local_x, local_y, UGV_HEIGHT])

            # --- ROTATION CORRECTION IS APPLIED HERE ---
            rotation_matrix = ugv_info['rotation_matrix']
            axis_correction_matrix = np.array([[0, 0, 1],
                                               [0, 1, 0],
                                               [-1, 0, 0]])
            final_rotation = rotation_matrix @ axis_correction_matrix
            # --- END ROTATION CORRECTION ---
            
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
            "scene_origin_gps": {"lat": drone_info['gps'][0], "lon": drone_info['gps'][1], "alt": drone_info['gps'][2]},
            "uav_image_path": os.path.join('uav_image', uav_fname), 
            "ugv_images": ugv_metadata,
        }
        with open(os.path.join(scene_path, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)

    print(f"\nPreprocessing complete. Created {scene_count} scenes in '{args.output_dir}'.")
    shutil.rmtree(ugv_temp_folder)
    shutil.rmtree(drone_temp_folder)
    print("Done.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create a SnapViT dataset by cropping an orthophoto based on GeoJSON row definitions.")
    parser.add_argument('--drone_geotiff', type=str, default='data/odm_orthophoto.tif', help="Path to a single GeoTIFF orthophoto to be cropped.")
    parser.add_argument('--geojson_path', type=str, default='data/riseholme_poles_trunk.geojson', help="Path to the GeoJSON file defining the rows and posts.")
    parser.add_argument('--ground_rosbag', type=str, default='data/row_1_to_6.bag', help="Path to the Realsense .bag file with UGV imagery.")
    parser.add_argument('--ground_llh', type=str, default='data/1_6.LLH', help="Path to the .llh file with UGV GPS coordinates.")
    parser.add_argument('--output_dir', type=str, default='datasets/row_wise_dataset2', help="Path to save the processed dataset.")
    parser.add_argument('--frame_skip', type=int, default=1, help="Process only every N-th frame from the rosbag to reduce data density.")
    args = parser.parse_args()
    main(args)