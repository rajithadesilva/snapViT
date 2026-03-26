#!/bin/bash

python3 snapViT/create_dataset_new.py \
    --drone_geotiff /media/hdd/ale_navone/GAIA/data/odm_orthophoto.tif \
    --ground_rosbag /media/hdd/ale_navone/GAIA/data/row_1_to_6.bag \
    --ground_llh /media/hdd/ale_navone/GAIA/data/1_6.LLH \
    --output_dir /media/hdd/ale_navone/GAIA//datasets/dataset_5k \
    --scene_radius 15.0 \
    --frame_skip 1 \
    --tile_ground_size 10.0 \
    --tile_count 5000 \
    --tile_seed 42 \
    --tile_padding_factor 0.0


