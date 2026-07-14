import os
import math
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from pyproj import Transformer
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
import rasterio
import random
import cv2
from rasterio.windows import Window
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rosbags.highlevel import AnyReader
from rosbags.image import message_to_cvimage
from rosbags.typesys import Stores, get_typestore
from collections import deque
import matplotlib.pyplot as plt
import json
import shutil
from matplotlib.path import Path as MplPath
from matplotlib.lines import Line2D


GPS_TOPIC = "/husky/sensors/gps_0/navsatfix"
IMU_TOPIC = "/husky/sensors/imu_0/imu/data"
RGB_TOPIC = "/husky/sensors/camera_0/color/compressed"
DEPTH_TOPIC = "/husky/sensors/camera_0/aligned_depth_to_color/image"
CAMERA_INFO_TOPIC = "/husky/sensors/camera_0/color/camera_info"

CAMERA_TRANSLATION = [-0.4, -0.34, 0.0]
UGV_HEIGHT = 1.0
ROS_TYPESTORE = get_typestore(Stores.LATEST)


#-----------------------------
# Metods for tiling and cropping drone images, and generating centroids for tiles.
#------------------------------
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

# -----------------------------
# Methods for reprojection of drone orthophoto to metric CRS (UTM Zone 32N, EPSG:32632) suitable for Torino.
#-----------------------------
def reproject_to_metric_torino(src_path, dst_path=None):
    """
    Reprojects a GeoTIFF to a metric CRS suitable for Torino (UTM Zone 32N, EPSG:32632).
    If the input is already projected in meters, it is copied unchanged.

    Returns the output path.
    """

    TORINO_UTM_CRS = "EPSG:32632"

    if dst_path is None:
        base, ext = os.path.splitext(src_path)
        dst_path = f"{base}_utm32n{ext}"

    with rasterio.open(src_path) as src:
        # If already projected in meters, do nothing
        if not src.crs.is_geographic:
            print("GeoTIFF already in a projected CRS. No reprojection needed.")
            return src_path

        print(f"Reprojecting from {src.crs} to {TORINO_UTM_CRS}")

        transform, width, height = calculate_default_transform(
            src.crs,
            TORINO_UTM_CRS,
            src.width,
            src.height,
            *src.bounds
        )

        profile = src.profile.copy()
        profile.update({
            "crs": TORINO_UTM_CRS,
            "transform": transform,
            "width": width,
            "height": height
        })

        with rasterio.open(dst_path, "w", **profile) as dst:
            for band_idx in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, band_idx),
                    destination=rasterio.band(dst, band_idx),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=TORINO_UTM_CRS,
                    resampling=Resampling.bilinear
                )

    print(f"Reprojected GeoTIFF written to: {dst_path}")
    return dst_path

#-----------------------------
# Methods for tiling and cropping drone images, and generating centroids for tiles.
#-----------------------------

def process_geotiff(
    geotiff_path,
        ground_size_m=10.0,
        tile_count=500,
        tile_rot_deg=0.0,
        output_dir="temp_drone_tiles",
        padding_factor=1.0,
        seed=0,
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
            if src.crs.is_geographic:
                raise ValueError(
                    "GeoTIFF CRS is geographic (degrees). "
                    "Reproject to a metric CRS before calling process_geotiff."
        )
            drone_crs = src.crs.to_epsg()
            print(f"GeoTIFF CRS is EPSG:{drone_crs}")

            gsd_x = src.transform.a
            gsd_y = -src.transform.e
            if gsd_x <= 0 or gsd_y <= 0:
                raise ValueError("Could not determine a valid resolution from the GeoTIFF.")

            tile_w_px = int(round(ground_size_m / gsd_x))
            tile_h_px = int(round(ground_size_m / gsd_y))

            if tile_w_px < 8 or tile_h_px < 8:
                raise ValueError(
                    f"Tile size too small in pixels: {tile_w_px}x{tile_h_px}. Check ground_size_m / GSD."
                )

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

                centre_x, centre_y = src.xy(cy, cx)
                lon, lat = transformer_to_wgs84.transform(centre_x, centre_y)
                alt = 0.0

                # Remove near-white pixels (likely background) and discard tiles with too much background
                mask = np.all(final_np >= 250, axis=-1) if final_np.ndim == 3 else final_np >= 250
                final_np[mask] = 0

                black_ratio = np.sum(final_np == 0) / final_np.size
                if black_ratio > 0.25:
                    continue

                tile_img = Image.fromarray(final_np)
                tile_name = f"original_tile_{idx:06d}_ang_{angle:+.2f}.png"
                tile_path = os.path.join(output_dir, tile_name)
                tile_img.save(tile_path)

                drone_data.append(
                    {"path": tile_path, "xy": (centre_x, centre_y), "gps": (lat, lon, alt), "tile_angle_deg": angle}
                )

        print(f"Created {len(drone_data)} tiles (requested {tile_count}).")
        return drone_data, drone_crs

#-----------------------------
# Methods for synchronizing UGV data from the rosbag, matching RGB and depth frames,
#------------------------------

def detect_bag_format(bag_path):
    """
    Detect whether the bag_path is a single bag or a directory with multiple sub-bags.
    
    Returns:
        - 'single': bag_path is a single bag file or a directory containing metadata.yaml and one bag file
        - 'multi': bag_path is a directory containing camera/ and platform/ subfolders
        - 'unknown': unable to determine format
    """
    bag_path = Path(bag_path)
    
    if bag_path.is_file() and bag_path.suffix in {'.db3', '.mcap'}:
        return 'single'
    
    if bag_path.is_dir():
        camera_dir = bag_path / 'camera'
        platform_dir = bag_path / 'platform'
        
        camera_bags = list(camera_dir.glob('*.db3')) if camera_dir.exists() else []
        platform_bags = list(platform_dir.glob('*.db3')) if platform_dir.exists() else []
        
        if camera_bags or platform_bags:
            return 'multi'
        
        metadata_file = bag_path / 'metadata.yaml'
        if metadata_file.exists():
            has_mcap = any(bag_path.glob('*.mcap'))
            has_db3 = any(bag_path.glob('*.db3'))
            if has_mcap or has_db3:
                return 'single'
    
    return 'unknown'

def get_all_bag_files(bag_path):
    """
    Get all bag files from a multi-bag directory structure.
    
    Returns:
        - camera_bags: list of paths to camera/*.db3 files
        - platform_bags: list of paths to platform/*.db3 files
    """
    bag_path = Path(bag_path)
    
    camera_dir = bag_path / 'camera'
    platform_dir = bag_path / 'platform'
    
    camera_bags = sorted(camera_dir.glob('*.db3')) if camera_dir.exists() else []
    platform_bags = sorted(platform_dir.glob('*.db3')) if platform_dir.exists() else []
    
    return camera_bags, platform_bags


def resolve_single_bag_source(bag_path):
    """Resolve the entry point to open for a single-bag recording."""
    bag_path = Path(bag_path)

    if bag_path.is_file():
        return bag_path

    # Check for actual data files first (metadata.yaml is not a valid rosbag source)
    mcap_files = sorted(bag_path.glob('*.mcap'))
    if mcap_files:
        return mcap_files[0]

    db3_files = sorted(bag_path.glob('*.db3'))
    if db3_files:
        return db3_files[0]

    # If no data files found, return the directory (rosbags may handle it)
    return bag_path

def _find_nearest_msg(msgs, target_timestamp):
    if not msgs:
        return None
    idx = np.argmin([abs(ts - target_timestamp) for ts, _ in msgs])

    return msgs[idx]

def _find_nearest_synced_frame(synced_frames, target_timestamp):
    if not synced_frames:
        return None
    idx = np.argmin([abs(frame['timestamp'] - target_timestamp) for frame in synced_frames])
    return synced_frames[idx]

def _sync_msgs(gps_msgs, imu_msgs, camera_info_msgs, depth_msgs, rgb_msgs, msg_count, ugv_ros_synced_data):
    print(f"Processing synchronization at message count: {msg_count}")
    synced_frames = []
    
    while rgb_msgs and depth_msgs:
        rgb_timestamp, rgb_img = rgb_msgs[0]
        depth_timestamp, depth_img = depth_msgs[0]

        time_diff = abs(rgb_timestamp - depth_timestamp)

        if time_diff < 2*1e7:  # 20 millisecond tolerance
            synced_frames.append({
                'timestamp': rgb_timestamp,
                'rgb_image': rgb_img,
                'depth_image': depth_img
            })
            rgb_msgs.popleft()
            depth_msgs.popleft()
        elif rgb_timestamp < depth_timestamp:
            rgb_msgs.popleft()
        else:
            depth_msgs.popleft()

    for ts, data in gps_msgs:
        nearest_imu = _find_nearest_msg(imu_msgs, ts)
        nearest_camera_info = _find_nearest_msg(camera_info_msgs, ts)
        nearest_synced_frame = _find_nearest_synced_frame(synced_frames, ts)

        if nearest_imu and nearest_camera_info and nearest_synced_frame:
            ugv_ros_synced_data.append({
                'timestamp': ts,
                'gps': data,
                'imu': nearest_imu[1],
                'camera_info': nearest_camera_info[1],
                'rgb_image': nearest_synced_frame['rgb_image'],
                'depth_image': nearest_synced_frame['depth_image']
            })

    print(f"Synchronized {len(ugv_ros_synced_data)} data entries.")

    gps_msgs.clear()
    imu_msgs.clear()
    rgb_msgs.clear()
    depth_msgs.clear()
    camera_info_msgs.clear()
    synced_frames.clear()

    return ugv_ros_synced_data

def synchronize_ros_data_multi_bag(bag_path=None, gps_topic=None, imu_topic=None, rgb_topic=None, depth_topic=None, camera_info_topic=None):
    """
    Synchronize ROS data from a multi-bag setup (camera/ and platform/ subfolders).
    Camera topics (RGB, depth, camera_info) are in camera/*.db3 files.
    Platform topics (GPS, IMU) are in platform/*.db3 files.
    """
    ugv_ros_synced_data = []
    
    print(f"Opening multi-bag structure from: {bag_path}")
    
    camera_bags, platform_bags = get_all_bag_files(bag_path)
    
    if not camera_bags and not platform_bags:
        raise ValueError(f"No .db3 files found in camera/ or platform/ subfolders of {bag_path}")
    
    print(f"Found {len(camera_bags)} camera bag(s) and {len(platform_bags)} platform bag(s)")
    
    # Combine all bags for reading (AnyReader expects Path objects)
    all_bags = camera_bags + platform_bags
    
    print("Opening combined bag files...")
    with AnyReader(all_bags, default_typestore=ROS_TYPESTORE) as reader:
        connections = [
            conn for conn in reader.connections
            if conn.topic in {gps_topic, imu_topic, rgb_topic, depth_topic, camera_info_topic}
        ]

        if not connections:
            raise ValueError(
                "No matching topics found in the provided bags: "
                f"{gps_topic}, {imu_topic}, {rgb_topic}, {depth_topic}, {camera_info_topic}"
            )
        
        gps_msgs = deque()
        imu_msgs = deque()
        rgb_msgs = deque()
        depth_msgs = deque()
        camera_info_msgs = deque()

        msg_count = 0

        for connection, timestamp, rawdata in reader.messages(connections=connections):
            if connection.topic == gps_topic:
                msg = reader.deserialize(rawdata, connection.msgtype)
                gps_msgs.append((timestamp, msg))
            elif connection.topic == imu_topic:
                msg = reader.deserialize(rawdata, connection.msgtype)
                imu_msgs.append((timestamp, msg))
            elif connection.topic == rgb_topic:
                msg = reader.deserialize(rawdata, connection.msgtype)
                rgb_img = message_to_cvimage(msg)
                rgb_msgs.append((timestamp, rgb_img))
            elif connection.topic == depth_topic:
                msg = reader.deserialize(rawdata, connection.msgtype)
                depth_img = message_to_cvimage(msg)
                depth_msgs.append((timestamp, depth_img))
            elif connection.topic == camera_info_topic:
                msg = reader.deserialize(rawdata, connection.msgtype)
                camera_info_msgs.append((timestamp, msg))

            if msg_count % 50000 == 0 and msg_count > 0:
                _sync_msgs(gps_msgs, imu_msgs, camera_info_msgs, depth_msgs, rgb_msgs, msg_count, ugv_ros_synced_data)
            msg_count += 1
        
        # Process last batch of messages after loop ends
        _sync_msgs(gps_msgs, imu_msgs, camera_info_msgs, depth_msgs, rgb_msgs, msg_count, ugv_ros_synced_data)
        
    return ugv_ros_synced_data

def synchronize_ros_data(bag_path=None, gps_topic=None, imu_topic=None, rgb_topic=None, depth_topic=None, camera_info_topic=None):
    """
    Auto-detect bag format (single or multi-bag) and synchronize ROS data accordingly.
    """
    bag_format = detect_bag_format(bag_path)
    
    if bag_format == 'unknown':
        raise ValueError(
            f"Unable to determine bag format for: {bag_path}\n"
            f"Expected either a .db3 file or a directory with camera/ and platform/ subfolders."
        )
    
    print(f"Detected bag format: {bag_format}")
    
    if bag_format == 'multi':
        return synchronize_ros_data_multi_bag(
            bag_path=bag_path,
            gps_topic=gps_topic,
            imu_topic=imu_topic,
            rgb_topic=rgb_topic,
            depth_topic=depth_topic,
            camera_info_topic=camera_info_topic
        )
    
    # Single bag format
    ugv_ros_synced_data = []

    single_bag_source = resolve_single_bag_source(bag_path)
    print(f"Opening rosbag: {single_bag_source}")

    with AnyReader([single_bag_source], default_typestore=ROS_TYPESTORE) as reader:
        connections = {conn.topic: conn for conn in reader.connections}
        
        gps_msgs = deque()
        imu_msgs = deque()
        rgb_msgs = deque()
        depth_msgs = deque()
        camera_info_msgs = deque()

        msg_count = 0

        for connection, timestamp, rawdata in reader.messages():
            if connection.topic == gps_topic:
                msg = reader.deserialize(rawdata, connection.msgtype)
                gps_msgs.append((timestamp, msg))
            elif connection.topic == imu_topic:
                msg = reader.deserialize(rawdata, connection.msgtype)
                imu_msgs.append((timestamp, msg))
            elif connection.topic == rgb_topic:
                msg = reader.deserialize(rawdata, connection.msgtype)
                rgb_img = message_to_cvimage(msg)
                rgb_msgs.append((timestamp, rgb_img))
            elif connection.topic == depth_topic:
                msg = reader.deserialize(rawdata, connection.msgtype)
                depth_img = message_to_cvimage(msg)
                depth_msgs.append((timestamp, depth_img))
            elif connection.topic == camera_info_topic:
                msg = reader.deserialize(rawdata, connection.msgtype)
                camera_info_msgs.append((timestamp, msg))

            if msg_count % 50000 == 0 and msg_count > 0:
                _sync_msgs(gps_msgs, imu_msgs, camera_info_msgs, depth_msgs, rgb_msgs, msg_count, ugv_ros_synced_data)
            msg_count += 1
            #processing last batch of messages after loop ends
        _sync_msgs(gps_msgs, imu_msgs, camera_info_msgs, depth_msgs, rgb_msgs, msg_count, ugv_ros_synced_data)
        
    return ugv_ros_synced_data

def unpack_ros_data(ugv_ros_synced_data, ugv_temp_rgb, ugv_temp_depth):
    ugv_synced_data = []


    for idx, entry in tqdm(enumerate(ugv_ros_synced_data)):
        ts = entry['timestamp']
        lat = entry['gps'].latitude
        lon = entry['gps'].longitude

        camera_info = entry['camera_info'].p
        fx = camera_info[0]
        fy = camera_info[5]
        cx = camera_info[2]
        cy = camera_info[6]

        # Create intrinsic matrix (3x3)
        ugv_intrinsics = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])
        entry['intrinsics'] = ugv_intrinsics

        fname  = f"frame_{idx:06d}.png"
        rgb_path = os.path.join(ugv_temp_rgb, fname)
        depth_path = os.path.join(ugv_temp_depth, fname)

        cv2.imwrite(rgb_path, entry['rgb_image'])
        cv2.imwrite(depth_path, entry['depth_image'])
        entry["name"] = fname

        orientation = entry['imu'].orientation  
        qx, qy, qz, qw = orientation.x, orientation.y, orientation.z, orientation.w

        ugv_synced_data.append({
            'timestamp': ts,
            'idx': idx,
            'name': fname,
            'gps': {'lat': lat, 'lon': lon},
            'intrinsics': ugv_intrinsics,
            'orientation_quat_imu': {'qx': qx, 'qy': qy, 'qz': qz, 'qw': qw},
        })

    ugv_ros_synced_data.clear()
    return ugv_synced_data

#-----------------------------
# Method to calculate rotation matrices from GPS coordinates of synchronized UGV data.
#-----------------------------
def calculate_rotation_from_gps(synced_data):
    for id in range(len(synced_data) - 1):
        lat1 = synced_data[id]['gps']['lat']
        lon1 = synced_data[id]['gps']['lon']
        lat2 = synced_data[id + 1]['gps']['lat']
        lon2 = synced_data[id + 1]['gps']['lon']

        # Calculate bearing
        #dLon = math.radians(lon2 - lon1)
        #y = math.sin(dLon) * math.cos(math.radians(lat2))
        #x = math.cos(math.radians(lat1)) * math.sin(math.radians(lat2)) - math.sin(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.cos(dLon)  
        #bearing = math.atan2(y, x)
        transformer = Transformer.from_crs("epsg:4326", "epsg:3857", always_xy=True)
        x1, y1 = transformer.transform(lon1, lat1)
        x2, y2 = transformer.transform(lon2, lat2)
        bearing = math.atan2(y2 - y1, x2 - x1)
        # Create rotation matrix from bearing (rotation around z-axis)
        rotation_matrix = R.from_euler('z', bearing, degrees=False).as_matrix()

        synced_data[id]['rotation_matrix'] = rotation_matrix
    
    #eliminate last entry without rotation
    synced_data.pop()

#-----------------------------
# Method to correct camera position using GPS coordinates and rotation matrices of synchronized UGV data.
#-----------------------------

def correct_camera_pos(synced_data, translation=np.array([-0.4, -0.34, 0.0])):
    for entry in synced_data:
        rot_matrix = entry['rotation_matrix']
        lat, lon = entry['gps']['lat'], entry['gps']['lon']
        # Transform translation vector from robot frame to world frame using rotation matrix
        translation_world = rot_matrix @ translation

        # Apply translation to the current (x, y) position
        # You need to convert lat/lon to metric coordinates first, apply translation, then convert back
        transformer = Transformer.from_crs("epsg:4326", "epsg:32632", always_xy=True)
        inv_transformer = Transformer.from_crs("epsg:32632", "epsg:4326", always_xy=True)

        # Convert lat/lon to metric coordinates
        orig_x, orig_y = transformer.transform(lon, lat)

        # Apply translation
        new_x = orig_x - translation_world[0]
        new_y = orig_y - translation_world[1]

        entry['pos_xy'] = (new_x, new_y)

        # Convert back to lat/lon
        new_lon, new_lat = inv_transformer.transform(new_x, new_y)

        # Update entry with corrected GPS
        entry['gps_corrected'] = {'lat': new_lat, 'lon': new_lon}

#-----------------------------
# Method to assign UGV data entries to the nearest UAV tiles based on corrected GPS coordinates.
#-----------------------------

def assign_ugv_to_uav_tiles(drone_data, ugv_synced_data):
    print("Assigning UGV to nearest UAV tiles...")

    print(f"Found {len(drone_data)} drone tiles and {len(ugv_synced_data)} UGV synced data entries.")
    scenes = {i: [] for i in range(len(drone_data))}
    drone_xy = [tile['xy'] for tile in drone_data]

    for ugv_entry in tqdm(ugv_synced_data, desc="Assigning UGV entries to UAV tiles"):
        x, y = ugv_entry['pos_xy']

        for scene_idx, d_xy in enumerate(drone_xy):
            if abs(d_xy[0] - x) < 5.0 and abs(d_xy[1] - y) < 5.0:
                scenes[scene_idx].append(ugv_entry)
    return scenes

#-----------------------------
# Methods for loading poles and assigning row regions.
#-----------------------------

def load_poles_from_file(poles_file_path):
    """Load pole positions and row numbers from a CSV/TXT file."""
    if not os.path.exists(poles_file_path):
        print(f"Warning: Poles file not found: {poles_file_path}")
        return None

    try:
        df_poles = pd.read_csv(poles_file_path, delimiter=",", comment="#")
    except Exception as exc:
        print(f"Error reading poles file: {exc}")
        return None

    col_names = df_poles.columns.tolist()
    x_col = None
    y_col = None
    row_col = None

    for col in col_names:
        col_lower = col.lower()
        if row_col is None and "row_number" in col_lower:
            row_col = col
        if x_col is None and ("x" in col_lower or "easting" in col_lower or "longitude" in col_lower):
            x_col = col
        if y_col is None and ("y" in col_lower or "northing" in col_lower or "latitude" in col_lower):
            y_col = col

    if x_col is None:
        for col in col_names:
            if col.lower().startswith("x"):
                x_col = col
                break
    if y_col is None:
        for col in col_names:
            if col.lower().startswith("y"):
                y_col = col
                break
    if row_col is None:
        for col in col_names:
            if "row" in col.lower():
                row_col = col
                break

    if x_col is None or y_col is None or row_col is None:
        print(f"Error: Could not identify x/y/row columns in poles file. Available columns: {col_names}")
        return None

    df_poles = df_poles[[x_col, y_col, row_col]].copy()
    df_poles[x_col] = pd.to_numeric(df_poles[x_col], errors="coerce")
    df_poles[y_col] = pd.to_numeric(df_poles[y_col], errors="coerce")
    df_poles[row_col] = pd.to_numeric(df_poles[row_col], errors="coerce")
    df_poles = df_poles.dropna(subset=[x_col, y_col, row_col])

    if df_poles.empty:
        print(f"Warning: no valid pole rows found in {poles_file_path}")
        return None

    poles_by_row = {}
    for row_id, group in df_poles.groupby(row_col):
        poles_by_row[int(row_id)] = group[[x_col, y_col]].to_numpy(dtype=float)

    print(f"Loaded {len(df_poles)} poles from {poles_file_path}, organized into {len(poles_by_row)} rows.")
    return poles_by_row


def build_row_corridor_model(poles_by_row):
    """Build corridor polygons between adjacent pole rows in a shared PCA frame."""
    if poles_by_row is None or len(poles_by_row) < 2:
        return None

    valid_points = [points for points in poles_by_row.values() if len(points) > 0]
    if len(valid_points) < 2:
        return None

    all_points = np.vstack(valid_points)
    centroid = all_points.mean(axis=0)
    centered = all_points - centroid
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    basis = vt[:2].T

    row_curves = {}
    for row_id, points in poles_by_row.items():
        if len(points) < 2:
            continue

        projected = (points - centroid) @ basis
        order = np.argsort(projected[:, 0])
        projected = projected[order]
        pc1 = projected[:, 0]
        pc2 = projected[:, 1]

        unique_pc1, unique_idx = np.unique(pc1, return_index=True)
        if len(unique_pc1) < 2:
            continue

        row_curves[int(row_id)] = {
            "pc1": unique_pc1,
            "pc2": pc2[unique_idx],
        }

    ordered_rows = sorted(row_curves)
    corridors = []
    for lower_row, upper_row in zip(ordered_rows[:-1], ordered_rows[1:]):
        lower = row_curves[lower_row]
        upper = row_curves[upper_row]

        overlap_start = max(float(lower["pc1"].min()), float(upper["pc1"].min()))
        overlap_end = min(float(lower["pc1"].max()), float(upper["pc1"].max()))
        if overlap_end <= overlap_start:
            continue

        sample_count = max(len(lower["pc1"]), len(upper["pc1"]), 64)
        grid = np.linspace(overlap_start, overlap_end, sample_count)
        lower_y = np.interp(grid, lower["pc1"], lower["pc2"])
        upper_y = np.interp(grid, upper["pc1"], upper["pc2"])

        bottom_y = np.minimum(lower_y, upper_y)
        top_y = np.maximum(lower_y, upper_y)
        polygon_pc = np.column_stack([
            np.r_[grid, grid[::-1]],
            np.r_[bottom_y, top_y[::-1]],
        ])

        corridors.append(
            {
                "row_id": int(lower_row),
                "lower_row": int(lower_row),
                "upper_row": int(upper_row),
                "polygon_pc": polygon_pc,
            }
        )

    if not corridors:
        return None

    return {
        "centroid": centroid,
        "basis": basis,
        "corridors": corridors,
        "ordered_rows": ordered_rows,
        "row_curves": row_curves,
    }


def assign_points_to_row_regions(positions, corridor_model):
    """Assign each position to a corridor if it falls inside one; otherwise return 0."""
    if corridor_model is None or len(positions) == 0:
        return np.zeros(len(positions), dtype=int)

    positions = np.asarray(positions, dtype=float)
    projected = (positions - corridor_model["centroid"]) @ corridor_model["basis"]
    row_ids = np.zeros(len(positions), dtype=int)

    for corridor in corridor_model["corridors"]:
        if np.all(row_ids != 0):
            break

        path = MplPath(corridor["polygon_pc"], closed=True)
        inside = path.contains_points(projected, radius=1e-9)
        row_ids[(row_ids == 0) & inside] = corridor["row_id"]

    return row_ids


def create_verification_visualization(orthophoto_path, ugv_synced_data, output_path, row_assignments=None, poles_by_row=None):
    """
    Create a PNG visualization showing orthophoto, trajectory (optionally colored by row), and poles (if available).
    
    Args:
        orthophoto_path: Path to orthophoto GeoTIFF
        ugv_synced_data: List of UGV entries with 'pos_xy' field
        output_path: Output PNG path
        row_assignments: Optional array of row IDs for each position (if None, trajectory is single color)
        poles_by_row: Optional dict of poles grouped by row ID
    """

    with rasterio.open(orthophoto_path) as src:
        ortho_array = src.read([1, 2, 3] if src.count >= 3 else [1])
        if src.count >= 3:
            ortho_array = np.moveaxis(ortho_array, 0, -1)
        else:
            ortho_array = np.moveaxis(ortho_array, 0, -1)
            ortho_array = np.stack([ortho_array] * 3, axis=-1)
        if ortho_array.max() > 1:
            ortho_array = ortho_array.astype(float) / 255.0
        bounds = src.bounds

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.imshow(ortho_array, extent=(bounds.left, bounds.right, bounds.bottom, bounds.top))

    trajectory = np.array([entry["pos_xy"] for entry in ugv_synced_data], dtype=float)
    if len(trajectory) > 0:
        if row_assignments is not None:
            colors = plt.cm.tab20(np.linspace(0, 1, max(int(row_assignments.max()) + 1, 2)))
            ax.plot(
                trajectory[:, 0],
                trajectory[:, 1],
                color="white",
                linewidth=1.2,
                alpha=0.55,
                zorder=2,
            )
            for row_id in sorted(set(int(value) for value in row_assignments)):
                mask = row_assignments == row_id
                if not np.any(mask):
                    continue
                ax.scatter(
                    trajectory[mask, 0],
                    trajectory[mask, 1],
                    s=18,
                    color=colors[row_id % len(colors)],
                    edgecolors="black",
                    linewidths=0.3,
                    alpha=0.95,
                    label=f"Region {row_id}" if row_id != 0 else "Outside regions",
                    zorder=3,
                )
        else:
            ax.plot(
                trajectory[:, 0],
                trajectory[:, 1],
                color="red",
                linewidth=2,
                alpha=0.85,
                label="Trajectory",
                zorder=2,
            )
            ax.scatter(
                trajectory[:, 0],
                trajectory[:, 1],
                s=16,
                color="red",
                edgecolors="black",
                linewidths=0.2,
                alpha=0.9,
                zorder=3,
            )

    if poles_by_row is not None:
        colors = plt.cm.tab20(np.linspace(0, 1, max(len(poles_by_row) + 1, 2)))
        for row_id, points in sorted(poles_by_row.items()):
            ax.scatter(
                points[:, 0],
                points[:, 1],
                s=24,
                c=[colors[int(row_id) % len(colors)]],
                edgecolors="black",
                linewidths=0.4,
                alpha=0.9,
            )

    legend_handles = [Line2D([0], [0], color="red" if row_assignments is None else "white", linewidth=2, label="Trajectory")]
    if poles_by_row is not None:
        legend_handles.append(
            Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", markeredgecolor="black", markersize=6, label="Poles")
        )
    if row_assignments is not None:
        legend_handles.append(Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", markeredgecolor="black", markersize=6, label="Region-assigned points"))
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Trajectory and poles on orthophoto" if poles_by_row else "Trajectory on orthophoto")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Verification plot saved to: {output_path}")
    plt.close()

#-----------------------------
# Method to build the final dataset structure, copying UAV tiles and UGV RGB/depth images and writing metadata for each scene.
#-----------------------------
def build_final_dataset(scenes, drone_data, ugv_temp_rgb, ugv_temp_depth, temp_drone_dir, final_rgb_dir, final_depth_dir, output_dir, poles_file=None, verification=False, orthophoto_path=None):
    print("Building final dataset structure...")
    poles_by_row = None
    corridor_model = None

    if poles_file is not None:
        poles_by_row = load_poles_from_file(poles_file)
        if poles_by_row is not None:
            corridor_model = build_row_corridor_model(poles_by_row)
            all_ugv_entries = []
            all_ugv_indices = []
            for drone_idx, ugv_entries in scenes.items():
                for entry_idx, ugv_entry in enumerate(ugv_entries):
                    all_ugv_entries.append(ugv_entry)
                    all_ugv_indices.append((drone_idx, entry_idx))

            if all_ugv_entries and corridor_model is not None:
                all_positions = np.array([entry["pos_xy"] for entry in all_ugv_entries], dtype=float)
                row_assignments = assign_points_to_row_regions(all_positions, corridor_model)
                for (drone_idx, entry_idx), row_id in zip(all_ugv_indices, row_assignments):
                    scenes[drone_idx][entry_idx]["row_id"] = int(row_id)

    scene_count = 0
    for drone_idx, ugv_entries in tqdm(scenes.items()):
        if not ugv_entries:
            continue

        scene_count += 1
        scene_id = f"scene_{scene_count:04d}"
        scene_path = os.path.join(output_dir, scene_id)
        uav_output_path = os.path.join(scene_path, "uav_image")

        os.makedirs(uav_output_path, exist_ok=True)

        # Copy UAV tile to scene directory
        drone_info = drone_data[drone_idx]
        uav_fname = os.path.join(uav_output_path, os.path.basename(drone_info['path']))
        shutil.copy(drone_info['path'], uav_fname)

        origin_x, origin_y = drone_info['xy']
        origin_lat, origin_lon, _ = drone_info['gps']


        ugv_metadata = []

        for ugv_info in ugv_entries:
            rgb_src = os.path.join(ugv_temp_rgb, ugv_info["name"])
            depth_src = os.path.join(ugv_temp_depth, ugv_info["name"])

            if not os.path.exists(rgb_src) or not os.path.exists(depth_src):
                print(f"Warning: Missing RGB or depth image for {ugv_info['name']}. Skipping this entry.")
                continue

            # Copy once into global shared folders (skip if already exists  
            rgb_dst = os.path.join(final_rgb_dir, ugv_info["name"])
            depth_dst = os.path.join(final_depth_dir, ugv_info["name"])
            if not os.path.exists(rgb_dst):
                try:
                    shutil.copy(rgb_src, rgb_dst)   
                except Exception as e:
                    print(f"Warning: Failed to copy RGB image {ugv_info['name']}: {e}")
            if not os.path.exists(depth_dst):
                try:
                    shutil.copy(depth_src, depth_dst)
                except Exception as e:
                    print(f"Warning: Failed to copy depth image {ugv_info['name']}: {e}")

            ugv_x, ugv_y = ugv_info['pos_xy']
            local_x = ugv_x - origin_x
            local_y = ugv_y - origin_y
            local_z = UGV_HEIGHT

            rotation_matrix = ugv_info['rotation_matrix']
            if drone_info.get("tile_angle_deg") is not None:
                tile_rot = drone_info['tile_angle_deg']
                R_tile = R.from_euler('z', tile_rot, degrees=True).as_matrix()
                rototranslation_tile = np.eye(4)
                rototranslation_tile[:3, :3] = R_tile

            translation_vector = np.array([local_x, local_y, local_z])

            camera_to_baselink_correction = np.array(
                [[0, 0, 1, 0],
                [-1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, 0, 1]]
            )

            rototranslation_matrix = np.eye(4)
            rototranslation_matrix[:3, :3] = rotation_matrix
            rototranslation_matrix[:3, 3] = translation_vector

            rototranslation_matrix = rototranslation_tile @ rototranslation_matrix
            final_rotation = rototranslation_matrix @ camera_to_baselink_correction

            w2c_matrix = np.linalg.inv(final_rotation).tolist()

            ugv_intrinsics = ugv_info['intrinsics']
            # Store relative paths in metadata to avoid saving full absolute paths
            try:
                rel_rgb_dir = os.path.relpath(final_rgb_dir, start=str(output_dir))
            except Exception:
                rel_rgb_dir = os.path.basename(final_rgb_dir)
            try:
                rel_depth_dir = os.path.relpath(final_depth_dir, start=str(output_dir))
            except Exception:
                rel_depth_dir = os.path.basename(final_depth_dir)

            image_rel_path = os.path.join(rel_rgb_dir, ugv_info["name"]) if rel_rgb_dir else ugv_info["name"]
            depth_rel_path = os.path.join(rel_depth_dir, ugv_info["name"]) if rel_depth_dir else ugv_info["name"]

            camera_metadata = {
                "camera_idx": ugv_info['idx'],
                "image_path": image_rel_path,
                "depth_path": depth_rel_path,
                "camera_intrinsics": ugv_intrinsics.tolist(),
                "camera_pose_w2c": w2c_matrix,
            }
            if 'row_id' in ugv_info:
                camera_metadata['row_id'] = int(ugv_info['row_id'])
            ugv_metadata.append(camera_metadata)

        metadata = {
            "scene_origin_gps": {
                "latitude": origin_lat,
                "longitude": origin_lon,
                "altitude": 0.0
                },
            "uav_image_path": os.path.join("uav_image", os.path.basename(uav_fname)),
            "ugv_images": ugv_metadata
        }

        with open(os.path.join(scene_path, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

    print(f"Created {scene_count} scenes in {output_dir}.")

    if os.path.exists(ugv_temp_rgb):
        shutil.rmtree(ugv_temp_rgb)
    if os.path.exists(ugv_temp_depth):
        shutil.rmtree(ugv_temp_depth)
    if os.path.exists(temp_drone_dir):
        shutil.rmtree(temp_drone_dir)  


            


def main(args):
    # Parse arguments
    orthophoto_path = Path(args.ortho_path)
    orthophoto_reprojected_path = Path(args.ortho_reprojected_path)
    bag_path = Path(args.bag_path)
    output_dir = Path(args.output_dir)

    # tiling parameters
    tile_count = args.tile_count
    tile_rot_deg = args.tile_rot_deg
    tile_seed = args.tile_seed
    tile_padding_factor = args.tile_padding_factor
    tile_ground_size = args.tile_ground_size

    # reproject orthophoto if needed
    if not os.path.exists(orthophoto_reprojected_path):
        orthophoto_reprojected_path = reproject_to_metric_torino(orthophoto_path, orthophoto_reprojected_path)

    # process drone data
    temp_drone_dir = os.path.join(output_dir, "temp_drone_tiles")
    drone_data, drone_crs = process_geotiff(
        ground_size_m=tile_ground_size,
        geotiff_path = orthophoto_reprojected_path,
        output_dir = temp_drone_dir,
        tile_count=tile_count,
        padding_factor=tile_padding_factor,
        tile_rot_deg=tile_rot_deg,
        seed=tile_seed
    )

    print("Opening rosbag and processing UGV data...")
    bag_format = detect_bag_format(bag_path)
    if bag_format == 'multi':
        camera_bags, platform_bags = get_all_bag_files(bag_path)
        debug_sources = camera_bags + platform_bags
    else:
        debug_sources = [resolve_single_bag_source(bag_path)]

    with AnyReader(debug_sources, default_typestore=ROS_TYPESTORE) as reader:
        print("Available topics:")
        for conn in reader.connections:
            print(f"- {conn.topic} ({conn.msgtype})")

    ugv_ros_synced_data = synchronize_ros_data(
        bag_path=bag_path,
        gps_topic=GPS_TOPIC,
        imu_topic=IMU_TOPIC,
        rgb_topic=RGB_TOPIC,
        depth_topic=DEPTH_TOPIC,
        camera_info_topic=CAMERA_INFO_TOPIC
    )

    ugv_temp_rgb = os.path.join(output_dir, "temp_ugv_rgb")
    ugv_temp_depth = os.path.join(output_dir, "temp_ugv_depth")
    os.makedirs(ugv_temp_rgb, exist_ok=True)
    os.makedirs(ugv_temp_depth, exist_ok=True)

    print("Unpacking synchronized UGV data and saving RGB/depth images...")
    print(f"Saving temporary UGV RGB images to: {ugv_temp_rgb}")
    print(f"Saving temporary UGV depth images to: {ugv_temp_depth}")
    ugv_synced_data = unpack_ros_data(ugv_ros_synced_data, ugv_temp_rgb, ugv_temp_depth)

    calculate_rotation_from_gps(ugv_synced_data)
    correct_camera_pos(ugv_synced_data, translation=np.array(CAMERA_TRANSLATION))

    poles_file = getattr(args, 'poles_file', None)
    verification = getattr(args, 'verification', False)

    if verification and os.path.exists(orthophoto_reprojected_path):
        if poles_file is not None:
            poles_by_row = load_poles_from_file(poles_file)
            corridor_model = build_row_corridor_model(poles_by_row) if poles_by_row is not None else None
            if poles_by_row is not None and corridor_model is not None:
                all_positions = np.array([entry["pos_xy"] for entry in ugv_synced_data], dtype=float)
                row_assignments = assign_points_to_row_regions(all_positions, corridor_model)
                for entry, row_id in zip(ugv_synced_data, row_assignments):
                    entry["row_id"] = int(row_id)
                viz_output = os.path.join(output_dir, "verification_trajectory.png")
                create_verification_visualization(
                    orthophoto_reprojected_path,
                    ugv_synced_data,
                    viz_output,
                    row_assignments=row_assignments,
                    poles_by_row=poles_by_row,
                )
        else:
            viz_output = os.path.join(output_dir, "verification_trajectory_only.png")
            create_verification_visualization(
                orthophoto_reprojected_path,
                ugv_synced_data,
                viz_output,
                row_assignments=None,
                poles_by_row=None,
            )

    scenes = assign_ugv_to_uav_tiles(drone_data, ugv_synced_data)

    final_rgb_dir = os.path.join(output_dir, "ugv_rgb")
    final_depth_dir = os.path.join(output_dir, "ugv_depth")
    os.makedirs(final_rgb_dir, exist_ok=True)
    os.makedirs(final_depth_dir, exist_ok=True)
    build_final_dataset(scenes, drone_data, ugv_temp_rgb, ugv_temp_depth, temp_drone_dir, final_rgb_dir, final_depth_dir, output_dir, poles_file=poles_file, verification=verification, orthophoto_path=str(orthophoto_reprojected_path))
    print("DONE")

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess UGV/UAV data for dataset creation.")

    parser.add_argument("--ortho-path", type=str, default="tempovine/2025_11_19_vineyard_run1/orthophoto.tif", help="Path to drone orthophoto GeoTIFF.")
    parser.add_argument("--ortho-reprojected-path", type=str, default="tempovine/2025_11_19_vineyard_run1/orthophoto_utm32n.tif", help="Path to reprojected orthophoto GeoTIFF (output of reprojection step).")
    parser.add_argument("--bag-path", type=str, default="tempovine/2025_11_19_vineyard_run1/metadata.yaml", help="Path to a bag file or a bag directory.")
    parser.add_argument("--output-dir", type=str, default="tempovine/dataset_tempovine", help="Output directory.")

    # Uniform tiling controls
    parser.add_argument("--tile-count", type=int, default=20, help="Aerial tile count to attempt.")
    parser.add_argument("--tile-rot-deg", type=float, default=00.0, help="Max abs rotation for tiles (± degrees).")
    parser.add_argument("--tile-seed", type=int, default=0, help="Random seed for tile rotations.")
    parser.add_argument("--tile-padding-factor", type=float, default=1.6, help="Padding factor when reading tiles.")
    parser.add_argument("--tile-ground-size", type=float, default=10.0, help="Tile width and height in metres.")

    # Pole row identification and verification
    parser.add_argument("--poles-file", type=str, default=None, help="Path to CSV/TXT file with pole positions and row numbers (optional).")
    parser.add_argument("--verification", action="store_true", help="If set, save a PNG visualization of the trajectory colored by row assignments and poles.")

    args = parser.parse_args()
    main(args)