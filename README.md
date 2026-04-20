
# SnapViT

SnapViT is a contrastive learning pipeline that uses Vision Transformers (ViT) to create unified Bird's-Eye View (BEV) representations from both aerial (UAV) and ground-level (UGV) images of the same scene. By training the model to recognize corresponding ground and aerial views, it learns to generate powerful, viewpoint-invariant feature maps.

## Overview

The core of the project is the SnapViT model, which consists of two main components:

- Ground encoder: processes one or more UGV images plus camera poses and projects them into a BEV feature map.
- Overhead encoder: processes a UAV image and produces a matching BEV feature map.

Training uses contrastive losses so matching ground and overhead views become closer in feature space than non-matching pairs.

## Repository Layout

| File | Purpose |
|------|---------|
| `model.py` | SnapViT model definition and feature extraction backbone. |
| `dataset.py` | `VineyardDataset` loader for scene-based UAV/UGV samples. |
| `train.py` | Main training loop and model checkpointing. |
| `create_dataset.py` | Dataset builder for orthophoto + ROS bag + LLH workflows. |
| `create_dataset_new.py` | Updated dataset builder with more configurable tiling and frame handling. |
| `create_dataset_tempovine.py` | Tempovine-specific dataset pipeline for orthophoto + bag data. |
| `create_dataset_from_frames.py` | Builds scenes from image folders and a GPS CSV. |
| `verify_dataset.py` | Visual checks for dataset alignment and metadata consistency. |
| `estimate_position.py` | Localization and pose search over candidate XY/yaw hypotheses. |
| `estimate_position_prob_heatmap.py` | Probability heatmap variant for localization analysis. |
| `eval_metrics.py` | Scene-level similarity metrics and negative/benchmark scenarios. |
| `eval_localization.py` | Localization evaluation and summary statistics. |
| `visualize.py` | Generates BEV feature visualizations from a trained checkpoint. |
| `visualize3D.py` | 3D visualization utilities for dataset inspection. |
| `visualize_dataset_samples.py` | Helper routines for plotting and sample inspection. |
| `plot_outputs.py` | Plotting helper for saved outputs. |
| `make_plots.ipynb` | Notebook for analysis and plotting. |
| `visualize_one_sample_dev.ipynb` | Notebook for one-off visualization/debugging. |
| `paper.md` | Project notes and paper draft. |
| `requirements.txt` | Python package list for this project. |
| `bash_files/` | Shell wrappers for common dataset, evaluation, and visualization commands. |

## Installation

From the repository root:

```bash
cd snapViT
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The requirements file includes the Python dependencies used by the training, dataset creation, evaluation, and visualization scripts. Some packages, especially `rasterio`, `pyrealsense2`, and `rosbags`, can also depend on platform-specific system libraries.

## Data Format

Most scripts expect a scene-based dataset with this general structure:

```text
dataset_root/
    scene_0001/
        metadata.json
        uav_image/
            *.png or *.jpg
        ugv_images/
            *.png or *.jpg
```

The exact contents of `metadata.json` depend on the dataset builder, but it typically stores:

- the UAV image path
- UGV image paths
- camera intrinsics
- world-to-camera pose matrices
- optional scene origin or GPS metadata

## Dataset Creation

### 1. Orthophoto + ROS Bag + LLH

Use `create_dataset.py` when you have a drone orthophoto, a UGV ROS bag, and a UGV LLH file.

```bash
python create_dataset.py \
    --drone_geotiff /path/to/orthophoto.tif \
    --ground_rosbag /path/to/ugv_data.bag \
    --ground_llh /path/to/ugv_gps.llh \
    --output_dir datasets/vineyard_dataset \
    --scene_radius 15 \
    --tile_ground_size 10.0
```

Key inputs:

- `--drone_geotiff`: orthophoto used for UAV tiles
- `--ground_rosbag`: ROS bag containing the ground vehicle frames
- `--ground_llh`: timestamped UGV GPS/LLH trace
- `--scene_radius`: scene matching radius in meters
- `--tile_ground_size`: ground coverage of each UAV tile

### 2. Updated Orthophoto + ROS Bag Pipeline

Use `create_dataset_new.py` for the newer tiled dataset workflow. The provided shell wrapper is a good reference for the expected invocation.

```bash
python create_dataset_new.py \
    --drone_geotiff /path/to/orthophoto.tif \
    --ground_rosbag /path/to/ugv_data.bag \
    --ground_llh /path/to/ugv_gps.llh \
    --output_dir /path/to/output \
    --scene_radius 15.0 \
    --frame_skip 1 \
    --tile_ground_size 10.0 \
    --tile_count 5000 \
    --tile_seed 42 \
    --tile_padding_factor 0.0
```

### 3. Tempovine Workflow

Use `create_dataset_tempovine.py` for the Tempovine dataset pipeline.

```bash
python create_dataset_tempovine.py \
    --ortho-path /path/to/orthophoto.tif \
    --ortho-reprojected-path /path/to/orthophoto_utm32n.tif \
    --bag-path /path/to/metadata.yaml \
    --output-dir /path/to/output \
    --tile-count 10000 \
    --tile-rot-deg 180 \
    --tile-seed 42 \
    --tile-padding-factor 3 \
    --tile-ground-size 10
```

This workflow is tuned for the Tempovine data layout and the orthophoto reprojection step used by the project.

### 4. Frame Folder + CSV Workflow

Use `create_dataset_from_frames.py` when you already have UAV and UGV images and a CSV with image-to-GPS mapping.

```bash
python create_dataset_from_frames.py \
    --drone_folder /path/to/drone_images \
    --ground_folder /path/to/ground_images \
    --ground_csv /path/to/ground_gps.csv \
    --output_dir datasets/vineyard_dataset \
    --scene_radius 10.0
```

## Dataset Verification

Use `verify_dataset.py` to visually inspect scene alignment and metadata before training.

```bash
python verify_dataset.py \
    --dataset_dir datasets/vineyard_dataset \
    --output_dir verification \
    --tile_ground_size 10.0
```

Make sure `tile_ground_size` matches the value used during dataset creation.

## Training

Train the model with:

```bash
python train.py
```

Training settings are controlled through the `CONFIG` dictionary in `train.py`. The script writes:

- `best_model.pth`
- `final_model.pth`
- `history/loss_history.csv`
- `history/training_config.json`

## Feature Visualization

Generate BEV feature visualizations from a trained checkpoint:

```bash
python visualize.py \
    --data_root datasets/vineyard_dataset \
    --checkpoint models/best_model.pth \
    --output_dir visualisations \
    --num_samples 30
```

This produces, per scene:

- the original UAV image
- a UGV collage
- the ground BEV map
- the overhead BEV map
- a cosine-similarity heatmap

## Localization / Pose Estimation

Use `estimate_position.py` to search over local XY offsets and yaw hypotheses.

```bash
python estimate_position.py \
    --data_root /path/to/dataset \
    --checkpoint /path/to/best_model.pth \
    --output_dir visualizations/estimate_position \
    --num_samples 20 \
    --top_k 10 \
    --grid_resolution_m 0.5 \
    --yaw_search_angles_deg "0,15,30,45,60,75,90,105,120,135,150,165,180,195,210,225,240,255,270,285,300,315,330,345"
```

Important options:

- `--scene_index`: run one scene instead of sampling multiple scenes
- `--grid_range_m`: search radius in meters
- `--grid_resolution_m`: grid step in meters
- `--top_k`: how many candidate locations to annotate
- `--sample_visualization`: save the sample projection visualization

The script saves the localization grid, best candidate annotations, and per-scene plots.

`estimate_position_prob_heatmap.py` provides a related heatmap-driven localization workflow and can be used when you want probability maps rather than only the best-scoring pose.

## Evaluation

Use `eval_metrics.py` to compute similarity-based metrics and benchmark scenarios.

```bash
python eval_metrics.py \
    --data_root /path/to/dataset \
    --checkpoint /path/to/best_model.pth \
    --output_dir evaluation/run_01 \
    --num_samples 1500 \
    --run_ground_benchmark \
    --ground_benchmark_angles "45,90,135,180,225,270,315" \
    --include_ground_swap_case
```

Useful evaluation flags:

- `--run_uav_negative`: test mismatched UAV inputs
- `--run_ground_negative`: test perturbed UGV inputs
- `--run_ground_benchmark`: sweep multiple ground-image angles
- `--include_ground_swap_case`: include a swapped-scene baseline
- `--include_ground_swap_rotated_cases`: include swap-plus-rotation cases

`eval_localization.py` complements this with localization-oriented evaluation metrics for checkpoint comparison and ablation analysis.

## Convenience Scripts

The `bash_files/` directory contains ready-to-run wrappers for common workflows:

- `create_dataset.sh`: orthophoto + ROS bag dataset creation
- `create_dataset_new.sh`: larger-scale dataset creation with extra tiling controls
- `create_dataset_tempovine.sh`: Tempovine dataset creation pipeline
- `estimate_position.sh`: localization run with a fixed checkpoint and yaw sweep
- `eval_metrics.bash`: similarity benchmark run with negative-test scenarios
- `visualize_outputs.bash`: visualization pass over saved outputs

## Notebooks and Analysis

- `make_plots.ipynb`: plotting and result analysis
- `visualize_one_sample_dev.ipynb`: interactive inspection of a single scene

## Notes

- The codebase assumes scene folders are consistent between dataset creation, training, and evaluation.
- Most scripts share the same image normalization and BEV feature dimensions.
- If you change the training `CONFIG`, keep the evaluation and visualization configs aligned.

## License

[MIT License](LICENSE)

## Contributions

Contributions are welcome. Please open an issue or pull request for changes.
