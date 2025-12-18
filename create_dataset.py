import os
import math
import random
import json
import shutil
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from pyproj import Transformer
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from datetime import datetime
import pyrealsense2 as rs
import rasterio
from rasterio.windows import Window

# --- CONFIGURATION ---
SYNC_THRESHOLD = 1.0
UGV_HEIGHT = 1.0


# =========================
# EXIF + LLH
# =========================
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

            lat = float(lat_data[0] + lat_data[1] / 60 + lat_data[2] / 3600)
            lon = float(lon_data[0] + lon_data[1] / 60 + lon_data[2] / 3600)

            if gps_info.get('GPSLatitudeRef') == 'S':
                lat = -lat
            if gps_info.get('GPSLongitudeRef') == 'W':
                lon = -lon

            alt = float(gps_info.get('GPSAltitude', 0))
            return lat, lon, alt

    except Exception as e:
        print(f"Could not read EXIF from {os.path.basename(image_path)}: {e}")
    return None


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
            if len(parts) < 5:
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


# =========================
# UAV TILE HELPERS (uniform centroids + rotation)
# =========================
def _centre_crop(arr, out_w, out_h):
    """Centre-crop numpy array HxWxC (or HxW) to (out_h, out_w)."""
    h, w = arr.shape[:2]
    cx, cy = w // 2, h // 2
    x0 = max(0, cx - out_w // 2)
    y0 = max(0, cy - out_h // 2)
    x1 = x0 + out_w
    y1 = y0 + out_h
    return arr[y0:y1, x0:x1]


def _rotate_and_crop(tile_np, angle_deg, out_w, out_h):
    """Rotate tile around centre (expand), then centre-crop."""
    pil = Image.fromarray(tile_np)
    rot = pil.rotate(angle_deg, resample=Image.BILINEAR, expand=True)
    rot_np = np.array(rot)
    cropped = _centre_crop(rot_np, out_w, out_h)
    return cropped


def _uniform_grid_centroids(width, height, tile_w_px, tile_h_px, tile_count):
    """
    Create exactly tile_count centroids uniformly distributed in pixel space,
    ensuring the *unrotated* tile fits inside the image.
    """
    margin_x = tile_w_px // 2
    margin_y = tile_h_px // 2

    usable_w = max(1, width - 2 * margin_x)
    usable_h = max(1, height - 2 * margin_y)

    aspect = usable_w / usable_h
    nx = max(1, int(math.ceil(math.sqrt(tile_count * aspect))))
    ny = max(1, int(math.ceil(tile_count / nx)))

    xs = np.linspace(margin_x, width - margin_x - 1, nx)
    ys = np.linspace(margin_y, height - margin_y - 1, ny)

    centroids = []
    for y in ys:
        for x in xs:
            centroids.append((float(x), float(y)))

    return centroids[:tile_count]


def process_geotiff(
    geotiff_path,
    ground_size_m=3.0,
    tile_count=500,
    tile_rot_deg=0.0,
    output_dir="temp_drone_tiles",
    padding_factor=1.6,
    seed=0
):
    """
    Creates (up to) `tile_count` tiles with uniformly distributed centroids over the orthophoto.
    Each tile covers `ground_size_m x ground_size_m` on the ground.
    Each tile is rotated by a random angle in [-tile_rot_deg, +tile_rot_deg] degrees.
    """
    print(f"Processing GeoTIFF: {geotiff_path}")
    os.makedirs(output_dir, exist_ok=True)
    drone_data = []
    random.seed(seed)

    with rasterio.open(geotiff_path) as src:
        drone_crs = src.crs.to_epsg()
        print(f"GeoTIFF CRS is EPSG:{drone_crs}")

        gsd_x = src.transform.a
        gsd_y = -src.transform.e
        if gsd_x <= 0 or gsd_y <= 0:
            raise ValueError("Could not determine a valid resolution from the GeoTIFF.")

        tile_w_px = int(round(ground_size_m / gsd_x))
        tile_h_px = int(round(ground_size_m / gsd_y))

        if tile_w_px < 8 or tile_h_px < 8:
            raise ValueError(f"Tile size too small in pixels: {tile_w_px}x{tile_h_px}. Check ground_size_m / GSD.")

        print(f"GSD (m/px): x={gsd_x:.4f}, y={gsd_y:.4f}")
        print(f"Tile size: {tile_w_px}x{tile_h_px} px for {ground_size_m}m coverage")
        print(f"Requested tile count: {tile_count}, rotation range: ±{tile_rot_deg} deg")

        width, height = src.width, src.height
        centroids_px = _uniform_grid_centroids(width, height, tile_w_px, tile_h_px, tile_count)

        pad_w = int(math.ceil(tile_w_px * padding_factor))
        pad_h = int(math.ceil(tile_h_px * padding_factor))
        pad_w = max(pad_w, tile_w_px)
        pad_h = max(pad_h, tile_h_px)

        transformer_to_wgs84 = Transformer.from_crs(f"epsg:{drone_crs}", "epsg:4326", always_xy=True)

        for idx, (cx, cy) in enumerate(tqdm(centroids_px, desc="Creating uniform tiles")):
            x0 = int(round(cx - pad_w / 2))
            y0 = int(round(cy - pad_h / 2))

            x0 = max(0, min(x0, width - pad_w))
            y0 = max(0, min(y0, height - pad_h))
            window = Window(x0, y0, pad_w, pad_h)

            tile = src.read(window=window)

            if tile.shape[0] >= 3:
                tile_np = np.moveaxis(tile[:3], 0, -1)  # HxWx3
            else:
                tile_np = tile[0]  # HxW

            if np.count_nonzero(tile_np) < tile_np.size * 0.1:
                continue

            angle = 0.0
            if tile_rot_deg and tile_rot_deg > 0:
                angle = random.uniform(-tile_rot_deg, tile_rot_deg)

            final_np = _rotate_and_crop(tile_np, angle, tile_w_px, tile_h_px)

            centre_x, centre_y = src.xy(cy, cx)  # (row, col) => (y, x) in px
            lon, lat = transformer_to_wgs84.transform(centre_x, centre_y)
            alt = 0.0

            tile_img = Image.fromarray(final_np)
            tile_name = f"tile_{idx:06d}_ang_{angle:+.2f}.png"
            tile_path = os.path.join(output_dir, tile_name)
            tile_img.save(tile_path)

            drone_data.append({
                "path": tile_path,
                "xy": (centre_x, centre_y),
                "gps": (lat, lon, alt),
                "tile_angle_deg": angle
            })

    print(f"Created {len(drone_data)} tiles (requested {tile_count}).")
    return drone_data, drone_crs


# =========================
# UGV EXTRACTION (colour + depth aligned)
# =========================
def extract_frames_from_bag(bag_file, rgb_output_folder, depth_output_folder, frame_skip):
    """
    Extracts and saves color + depth image frames from a Realsense .bag file,
    aligns depth to color, and returns (frames, intrinsics_matrix).
    Depth saved as 16-bit PNG (raw units from RealSense depth frame).
    """
    print(f"Extracting RGB+Depth frames from {bag_file} every {frame_skip} frames...")
    os.makedirs(rgb_output_folder, exist_ok=True)
    os.makedirs(depth_output_folder, exist_ok=True)

    extracted_frames = []
    intrinsics_matrix = None

    pipeline = rs.pipeline()
    config = rs.config()
    rs.config.enable_device_from_file(config, bag_file, repeat_playback=False)
    config.enable_stream(rs.stream.color)
    config.enable_stream(rs.stream.depth)

    align = rs.align(rs.stream.color)

    try:
        profile = pipeline.start(config)

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
            frames = pipeline.wait_for_frames(timeout_ms=1000)
            frames = align.process(frames)

            if frame_number % frame_skip == 0:
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()
                if color_frame and depth_frame:
                    timestamp = color_frame.get_timestamp() / 1000.0  # seconds

                    timestamp_str = f"{timestamp:.4f}".replace('.', '_')
                    fname = f"frame_{timestamp_str}.png"

                    # RGB
                    rgb_path = os.path.join(rgb_output_folder, fname)
                    rgb = np.asanyarray(color_frame.get_data())
                    Image.fromarray(rgb).save(rgb_path)

                    # Depth (16-bit)
                    depth_path = os.path.join(depth_output_folder, fname)
                    depth = np.asanyarray(depth_frame.get_data()).astype(np.uint16)
                    Image.fromarray(depth).save(depth_path)

                    extracted_frames.append({"timestamp": timestamp, "name": fname})

            frame_number += 1

    except RuntimeError:
        print("End of bag file reached or no frames were received in the timeout window.")
    finally:
        pipeline.stop()

    if not extracted_frames:
        print("Warning: No frames were extracted from the bag file. Check if color+depth streams exist.")

    return sorted(extracted_frames, key=lambda x: x["timestamp"]), intrinsics_matrix


# =========================
# ROTATION FROM GPS
# =========================
def calculate_rotation_from_gps_window(current_gps_idx, gps_data_list, window_size=5):
    """Calculates camera rotation matrix based on direction of travel (yaw)."""
    start_idx = max(0, current_gps_idx - window_size // 2)
    end_idx = min(len(gps_data_list) - 1, current_gps_idx + window_size // 2)

    if start_idx >= end_idx:
        return np.eye(3)

    transformer = Transformer.from_crs("epsg:4326", "epsg:32630", always_xy=True)
    points_xy = [transformer.transform(p['lon'], p['lat']) for p in gps_data_list[start_idx:end_idx + 1]]

    if len(points_xy) < 2:
        return np.eye(3)

    direction_vector = np.array(points_xy[-1]) - np.array(points_xy[0])
    if np.linalg.norm(direction_vector) < 1e-6:
        return np.eye(3)

    yaw = np.arctan2(direction_vector[1], direction_vector[0])
    rotation = R.from_euler('z', yaw, degrees=False)
    return rotation.as_matrix()


# =========================
# MAIN PIPELINE
# =========================
def process_real_data(args):
    print("Starting data preprocessing...")
    os.makedirs(args.output_dir, exist_ok=True)

    ugv_rgb_temp = os.path.join(args.output_dir, "temp_ugv_extracted_images")
    ugv_depth_temp = os.path.join(args.output_dir, "temp_ugv_extracted_depths")
    drone_temp = os.path.join(args.output_dir, "temp_drone_tiles")

    # 1) UAV loading (GeoTIFF uniform tiles OR folder)
    drone_crs = None
    if args.drone_geotiff:
        drone_data, drone_crs = process_geotiff(
            args.drone_geotiff,
            ground_size_m=args.tile_ground_size,
            tile_count=args.tile_count,
            tile_rot_deg=args.tile_rot_deg,
            output_dir=drone_temp,
            padding_factor=args.tile_padding_factor,
            seed=args.tile_seed,
        )
        drone_xy = np.array([d["xy"] for d in drone_data])
    elif args.drone_folder:
        print("Reading drone images from folder and their GPS locations...")
        drone_data = []
        for fname in tqdm(os.listdir(args.drone_folder), desc="Reading drone images"):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                fpath = os.path.join(args.drone_folder, fname)
                gps = get_gps_from_exif(fpath)
                if gps:
                    drone_data.append({"path": fpath, "gps": gps})

        drone_crs = 32630  # your assumption
        print(f"Assuming drone imagery uses EPSG:{drone_crs}")
        transformer = Transformer.from_crs("epsg:4326", f"epsg:{drone_crs}", always_xy=True)
        for d in drone_data:
            d["xy"] = transformer.transform(d["gps"][1], d["gps"][0])
        drone_xy = np.array([d["xy"] for d in drone_data])
    else:
        raise ValueError("You must provide either --drone_folder or --drone_geotiff")

    if not drone_data:
        raise ValueError("No drone images could be loaded.")

    # 2) UGV GPS
    gps_df = parse_llh_file(args.ground_llh)
    gps_data_list = gps_df.reset_index().to_dict("records")

    # 3) Extract UGV RGB+Depth + intrinsics
    ugv_frames, ugv_intrinsics = extract_frames_from_bag(
        args.ground_rosbag,
        ugv_rgb_temp,
        ugv_depth_temp,
        args.frame_skip
    )
    if not ugv_frames or gps_df.empty:
        raise ValueError("Cannot process: UGV frames or GPS is empty.")
    if ugv_intrinsics is None:
        raise ValueError("Could not extract camera intrinsics from rosbag.")

    # 4) Dynamic offset
    first_bag_ts = ugv_frames[0]["timestamp"]
    first_gps_ts = gps_df.index.min()
    total_offset = first_bag_ts - first_gps_ts
    print(f"First camera timestamp (s): {first_bag_ts}")
    print(f"First GPS timestamp (s):    {first_gps_ts}")
    print(f"Calculated timeline offset (s): {total_offset:.4f}")

    # 5) Sync frames to GPS
    print("Synchronizing UGV frames with GPS data...")
    ugv_synced = []
    gps_timestamps = gps_df.index.to_numpy()

    for frame in tqdm(ugv_frames, desc="Syncing frames to GPS"):
        adjusted_ts = frame["timestamp"] - total_offset
        idx = np.searchsorted(gps_timestamps, adjusted_ts, side="right")
        if idx > 0:
            best = idx - 1
            if abs(adjusted_ts - gps_timestamps[best]) < SYNC_THRESHOLD:
                best_gps = gps_data_list[best]
                rot = calculate_rotation_from_gps_window(best, gps_data_list)
                ugv_synced.append({
                    "name": frame["name"],
                    "gps": (best_gps["lat"], best_gps["lon"], best_gps["alt"]),
                    "rotation_matrix": rot
                })

    # 6) Assign UGV to nearest UAV scene
    print(f"Found {len(ugv_synced)} synchronized UGV frames. Matching to drone scenes...")
    scenes = {i: [] for i in range(len(drone_data))}

    ugv_transformer = Transformer.from_crs("epsg:4326", f"epsg:{drone_crs}", always_xy=True)
    for ugv_info in tqdm(ugv_synced, desc="Assigning UGV images to scenes"):
        lon, lat = ugv_info["gps"][1], ugv_info["gps"][0]
        ugv_x, ugv_y = ugv_transformer.transform(lon, lat)
        ugv_info["xy"] = (ugv_x, ugv_y)

        distances = np.linalg.norm(drone_xy - np.array([ugv_x, ugv_y]), axis=1)
        closest = int(np.argmin(distances))
        if distances[closest] < args.scene_radius:
            scenes[closest].append(ugv_info)

    # 7) Build dataset
    print("Building final dataset structure...")
    scene_count = 0

    for drone_idx, ugv_list in scenes.items():
        if not ugv_list:
            continue

        scene_count += 1
        scene_id = f"scene_{scene_count:04d}"
        scene_path = os.path.join(args.output_dir, scene_id)

        ugv_out = os.path.join(scene_path, "ugv_images")
        depth_out = os.path.join(scene_path, "ugv_depths")
        uav_out = os.path.join(scene_path, "uav_image")

        os.makedirs(ugv_out, exist_ok=True)
        os.makedirs(depth_out, exist_ok=True)
        os.makedirs(uav_out, exist_ok=True)

        drone_info = drone_data[drone_idx]
        uav_fname = os.path.basename(drone_info["path"])
        shutil.copy(drone_info["path"], os.path.join(uav_out, uav_fname))

        origin_x, origin_y = drone_info["xy"]
        origin_lat, origin_lon, origin_alt = drone_info["gps"]

        ugv_metadata = []

        for ugv_info in ugv_list:
            rgb_src = os.path.join(ugv_rgb_temp, ugv_info["name"])
            dep_src = os.path.join(ugv_depth_temp, ugv_info["name"])
            if not os.path.exists(rgb_src):
                continue
            if not os.path.exists(dep_src):
                continue

            shutil.copy(rgb_src, os.path.join(ugv_out, ugv_info["name"]))
            shutil.copy(dep_src, os.path.join(depth_out, ugv_info["name"]))

            ugv_x, ugv_y = ugv_info["xy"]
            local_x = ugv_x - origin_x
            local_y = ugv_y - origin_y
            local_z = UGV_HEIGHT

            rotation_matrix = ugv_info["rotation_matrix"]
            translation_vector = np.array([local_x, local_y, local_z])

            axis_correction_matrix = np.array([
                [0, 0, 1],
                [0, 1, 0],
                [-1, 0, 0]
            ])
            final_rotation = rotation_matrix @ axis_correction_matrix

            c2w = np.eye(4)
            c2w[:3, :3] = final_rotation
            c2w[:3, 3] = translation_vector
            w2c = np.linalg.inv(c2w).tolist()

            ugv_metadata.append({
                "image_path": os.path.join("ugv_images", ugv_info["name"]),
                "depth_path": os.path.join("ugv_depths", ugv_info["name"]),
                "camera_intrinsics": ugv_intrinsics,
                "camera_pose_w2c": w2c
            })

        metadata = {
            "scene_origin_gps": {"lat": origin_lat, "lon": origin_lon, "alt": origin_alt},
            "uav_image_path": os.path.join("uav_image", uav_fname),
            "ugv_images": ugv_metadata,
        }

        with open(os.path.join(scene_path, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

    print(f"\nPreprocessing complete. Created {scene_count} scenes in '{args.output_dir}'.")

    # Cleanup
    if os.path.exists(ugv_rgb_temp):
        print(f"Cleaning up temporary directory: {ugv_rgb_temp}")
        shutil.rmtree(ugv_rgb_temp)
    if os.path.exists(ugv_depth_temp):
        print(f"Cleaning up temporary depth directory: {ugv_depth_temp}")
        shutil.rmtree(ugv_depth_temp)
    if args.drone_geotiff and os.path.exists(drone_temp):
        print(f"Cleaning up temporary drone tile directory: {drone_temp}")
        shutil.rmtree(drone_temp)

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess UGV/UAV data for dataset creation.")

    drone_source_group = parser.add_mutually_exclusive_group()
    drone_source_group.add_argument("--drone_folder", type=str, default="data/aerial39",
                                    help="Path to the folder with geotagged drone images.")
    drone_source_group.add_argument("--drone_geotiff", type=str, default="data/odm_orthophoto.tif",
                                    help="Path to a single GeoTIFF orthophoto to be tiled.")

    parser.add_argument("--ground_rosbag", type=str, default="data/row_1_to_6.bag",
                        help="Path to the Realsense .bag file.")
    parser.add_argument("--ground_llh", type=str, default="data/1_6.LLH",
                        help="Path to the .llh file with ground vehicle GPS coordinates.")
    parser.add_argument("--output_dir", type=str, default="datasets/dataset2",
                        help="Path to save the processed dataset.")
    parser.add_argument("--scene_radius", type=float, default=15.0,
                        help="Radius in meters to group ground images under a drone image.")
    parser.add_argument("--frame_skip", type=int, default=10,
                        help="Process only every N-th frame from the rosbag.")
    parser.add_argument("--tile_ground_size", type=float, default=10.0,
                        help="Tile width/height in metres.")

    # New tiling controls
    parser.add_argument("--tile_count", type=int, default=20,
                        help="Exact number of aerial tiles to create (attempted; may be fewer if empty tiles are skipped).")
    parser.add_argument("--tile_rot_deg", type=float, default=90.0,
                        help="Max absolute rotation (degrees) for each tile (±).")
    parser.add_argument("--tile_seed", type=int, default=0,
                        help="Random seed for tile rotations.")
    parser.add_argument("--tile_padding_factor", type=float, default=1.6,
                        help="Read padding factor to prevent rotation clipping.")

    args = parser.parse_args()
    process_real_data(args)
