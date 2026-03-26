#!/bin/bash

python3 snapViT/create_dataset_tempovine.py \
    --ortho-path //media/hdd/ale_navone/GAIA/tempovine/ortofoto_completa_25m.tif \
    --ortho-reprojected-path /media/hdd/ale_navone/GAIA/tempovine/orthophoto_utm32n.tif \
    --bag-path /media/hdd/ale_navone/GAIA/tempovine/2025_11_19_vineyard_run3 \
    --output-dir /media/hdd/ale_navone/GAIA/tempovine/dataset_tempovine_test \
    --tile-count 10000 \
    --tile-rot-deg 180 \
    --tile-seed 42 \
    --tile-padding-factor 3 \
    --tile-ground-size 10