#!/bin/bash

python3 snapViT/create_dataset.py \
    --drone_geotiff /media/hdd/ale_navone/GAIA/data/odm_orthophoto.tif \
    --ground_rosbag /media/hdd/ale_navone/GAIA/data/row_1_to_6.bag \
    --ground_llh /media/hdd/ale_navone/GAIA/data/1_6.LLH \
    --output_dir /home/ale_navone/ws_pytorch/GAIA/snapViT/data/new1 \
    --scene_radius 15 \
    --tile_ground_size 10.0
