# SnapViT Methodology

## Abstract
This document describes the methodology implemented in the SnapViT codebase for learning aligned Bird's-Eye View (BEV) representations from aerial (UAV) and ground (UGV) imagery. The approach combines a dual-encoder architecture based on Vision Transformers, geometry-aware projection of UGV observations into BEV space, and contrastive training objectives that enforce cross-view representation consistency. The full pipeline includes (i) dataset construction from GeoTIFF/ROS/GPS sources, (ii) scene-based multi-view sampling, (iii) BEV feature generation for UAV and UGV branches, (iv) masked contrastive optimization, and (v) downstream localization and robustness evaluation.

## 1. Problem Formulation
Given one UAV overhead image and a set of synchronized UGV observations from nearby positions, the goal is to learn two functions:

- $f_g$: maps multi-view UGV images, depth, intrinsics, and poses to a ground BEV feature map.
- $f_o$: maps a UAV image to an overhead BEV feature map.

Training enforces alignment between $f_g(\cdot)$ and $f_o(\cdot)$ at both global and local levels so that corresponding UAV/UGV scenes produce similar BEV embeddings.

## 2. Data Methodology

### 2.1 Raw Data Sources
The repository supports multiple data-generation modes:

- GeoTIFF-based UAV orthophoto tiling with geospatial metadata.
- Drone image folders with EXIF GPS.
- UGV ROS bag extraction with aligned RGB/depth streams and camera intrinsics.
- UGV GPS trajectories from `.llh` logs or ROS topics.

Main scripts:

- `create_dataset.py`
- `create_dataset_new.py`
- `create_dataset_tempovine.py`
- `create_dataset_from_frames.py`

### 2.2 UAV Tile Construction
For GeoTIFF-based workflows, orthophotos are partitioned into local tiles representing fixed metric ground areas (typically around 10 m, script-dependent). In advanced variants:

- Tile centroids are distributed uniformly across the orthophoto.
- Optional random in-plane tile rotation is applied.
- Center-cropping preserves final tile size after rotation.
- Low-information tiles (mostly empty/black) are discarded.

Each tile stores:

- image path,
- projected map coordinates $(x,y)$,
- geographic coordinates (lat/lon/alt),
- optional tile rotation angle metadata.

### 2.3 UGV Extraction and Synchronization
From ROS bags, the pipeline extracts:

- RGB frames,
- depth frames aligned to RGB,
- camera intrinsics,
- GPS/IMU/camera-info messages.

Temporal alignment is performed via nearest-timestamp matching under a threshold. In `.llh` workflows, a dynamic timeline offset is estimated from first-frame and first-GPS timestamps to avoid hardcoded lag assumptions.

### 2.4 Scene Assignment
Each UGV sample is projected to the same metric CRS as UAV tiles. Scenes are constructed by nearest-tile assignment with a radius constraint (`scene_radius`), resulting in per-scene folders:

- `uav_image/...`
- `ugv_images/...`
- optional depth storage
- `metadata.json`

### 2.5 Pose and Camera Metadata
For each UGV frame, metadata includes:

- `camera_intrinsics` ($3 \times 3$),
- `camera_pose_w2c` ($4 \times 4$ world-to-camera transform),
- optional frame index and depth path.

Depending on script variant, orientation is estimated from local trajectory direction (yaw from GPS window) or from synchronized IMU in ROS-based preprocessing.

## 3. Training Dataset Construction

Implemented in `dataset.py` via `VineyardDataset`.

### 3.1 Scene Filtering
Scenes and views can be filtered by edge margin (`edge_margin_m`) so that selected UGV poses remain sufficiently inside the UAV tile footprint.

### 3.2 Multi-View Sampling Strategy
For each scene, the loader selects `num_ugv_views` images using:

- random selection (default), or
- consecutive-frame selection (`consecutive_frames=True`), where contiguous camera indices are preferred.

If available views are fewer than requested, view IDs are repeated/padded deterministically.

### 3.3 Transformations
Both UAV and UGV RGB inputs are resized and normalized using ImageNet statistics. Depth maps are resized and kept as float tensors. Intrinsics are scaled to match resized image dimensions.

### 3.4 BEV Grid Definition
The loader provides a structured 3D grid generated from `grid_size=(X,Y,Z)` and `grid_resolution`:

$$
\mathcal{G} = \{(x_i, y_j, z_k)\}, \quad
x_i \in \left[-\frac{Xr}{2}, \frac{Xr}{2}\right],\
y_j \in \left[\frac{Yr}{2}, -\frac{Yr}{2}\right],\
z_k \in [0, Zr]
$$

where $r$ is the metric resolution in meters per cell.

## 4. Model Architecture

Implemented in `model.py`.

### 4.1 Shared ViT Backbone
Both branches use `ViTFeatureExtractor` (from `timm`) with classification head removed. Patch tokens (excluding CLS) are reshaped into 2D feature maps.

### 4.2 Overhead Encoder
`OverheadEncoder` applies:

- ViT feature extraction on UAV input,
- $1\times1$ convolutional projection to target feature dimension (`feature_dim`).

Output: overhead BEV tensor $F_o \in \mathbb{R}^{B \times C \times H \times W}$.

### 4.3 Ground Encoder
`GroundEncoder` processes multi-view UGV RGB/depth/pose data:

1. Extract per-view ViT feature maps.
2. Resize depth maps to feature resolution.
3. Back-project valid depth pixels to 3D camera coordinates.
4. Transform to world coordinates using $\mathbf{T}_{c2w} = \mathbf{T}_{w2c}^{-1}$.
5. Map world points to BEV pixel coordinates using tile metric scale.
6. Splat per-view features into BEV bins and aggregate (default: average).
7. Apply MLP fusion to obtain final BEV channels.

A validity mask is generated from occupancy counts:

$$
M(u,v) = \mathbb{1}[\text{count}(u,v) > 0]
$$

Output: ground BEV tensor $F_g$ and validity mask $M$.

### 4.4 Full Model
`SnapViT` combines both encoders and learns a trainable contrastive temperature parameter $\tau$.

## 5. Optimization Objectives

Implemented in `train.py`.

### 5.1 Pixel-Level Masked InfoNCE
After resizing the overhead BEV to ground resolution, a masked pixel-wise contrastive loss is computed only on valid BEV cells.

For valid index set $\Omega$:

$$
\mathcal{L}_{pix} = -\frac{1}{|\Omega|} \sum_{i \in \Omega}
\log \frac{\exp(\langle g_i, o_i \rangle / \tau)}{\sum_{j \in \Omega} \exp(\langle g_i, o_j \rangle / \tau)}
$$

where $g_i, o_i$ are normalized features at location $i$.

### 5.2 Global Symmetric InfoNCE
Masked average pooling produces one embedding per scene for each branch. Bidirectional contrastive loss is applied across the batch:

$$
\mathcal{L}_{glob} = \frac{1}{2}
\left(\mathcal{L}_{g\rightarrow o} + \mathcal{L}_{o\rightarrow g}\right)
$$

### 5.3 Mixed Loss Schedule
Training uses a delayed mixed objective:

- Early epochs (`mixed_loss_delay`): optimize only global loss.
- Later epochs: combine losses

$$
\mathcal{L} = \lambda \mathcal{L}_{pix} + (1-\lambda)\mathcal{L}_{glob}
$$

with `pixel_loss_weight = \lambda`.

## 6. Training Protocol

### 6.1 Data Split and Loading
The full dataset is split into train/validation subsets via `random_split`, controlled by `val_split_ratio`. Data is loaded with multi-worker PyTorch `DataLoader`.

### 6.2 Hyperparameters (Default Configuration)
Typical defaults in current training script:

- ViT backbone: `vit_small_patch16_224`
- image size: $224 \times 224$
- feature dimension: 128
- UGV views: 8
- BEV grid: $(34, 34, 8)$ at 0.3 m resolution
- batch size: 8
- optimizer: Adam, learning rate $1 \times 10^{-4}$
- epochs: up to 1000

### 6.3 Checkpointing and Logging
The pipeline stores:

- best validation checkpoint (`best_model.pth`),
- final checkpoint (`final_model.pth`),
- TensorBoard logs,
- CSV/JSON training history.

## 7. Inference and Evaluation Methodology

### 7.1 Feature Visualization
`visualize.py` projects high-dimensional BEV features to RGB with PCA and exports:

- UAV input,
- UGV collage,
- ground BEV visualization,
- overhead BEV visualization,
- cosine similarity heatmaps.

### 7.2 Pose Estimation via Hypothesis Grid
`estimate_position.py` evaluates candidate UGV poses over a 2D translation grid (and optional yaw candidates). For each hypothesis:

1. modify camera pose,
2. run forward model,
3. compute BEV cosine score under validity mask,
4. select best-scoring pose.

A softmax over scores yields a probability map for localization confidence.

### 7.3 Robustness and Benchmark Metrics
`eval_metrics.py` evaluates scene-level and global cosine statistics under controlled perturbations:

- UAV image rotation,
- UGV image/depth rotation,
- camera yaw perturbation,
- cross-scene swapping (negative controls).

Reported metrics include mean, median, std, quantiles, and valid-pixel ratios per scene and per scenario.

## 8. Verification and Quality Control

`verify_dataset.py` overlays UGV poses/orientations on UAV tiles to visually validate:

- metric alignment,
- coordinate transforms,
- orientation consistency.

This step is used to identify synchronization and georeferencing issues prior to training.

## 9. Reproducibility Notes
Methodology is configuration-driven and implemented with explicit scripts for each stage. To reproduce a given experiment, preserve:

- dataset generation script and arguments,
- training `CONFIG` snapshot,
- model checkpoint and corresponding history CSV/JSON.

## 10. Scope and Current Limitations
Current implementation reflects active research/development tradeoffs:

- some coordinate-convention handling and aggregation options are marked as experimental in code comments,
- multiple dataset-generation variants coexist (baseline and improved workflows),
- loss design emphasizes contrastive alignment and may require scenario-specific tuning.

Despite these limitations, the repository provides a complete end-to-end methodology for cross-view BEV representation learning and localization-oriented evaluation.
# SnapViT Methodology

## 1. Problem Formulation

This work addresses cross-view representation learning and localization between:

- Ground-level RGB-D observations acquired from an Unmanned Ground Vehicle (UGV)
- Overhead imagery acquired from an Unmanned Aerial Vehicle (UAV) or orthomosaic tiles

The objective is to learn a shared Bird's-Eye View (BEV) feature space where corresponding UGV and UAV observations of the same scene are close, while non-corresponding observations are far apart.

---

## 2. Dataset Construction Methodology

The repository includes multiple dataset creation pipelines (`create_dataset.py`, `create_dataset_new.py`, `create_dataset_tempovine.py`, `create_dataset_from_frames.py`) that implement the same core methodology with different input modalities.

### 2.1 Inputs

Depending on script variant, inputs can include:

- Orthophoto GeoTIFFs (possibly tiled)
- Geotagged UAV image folders
- ROS bag streams from UGV sensors
- UGV GPS logs (`.llh` files or CSV)
- Camera intrinsics from Realsense or ROS camera info

### 2.2 UAV Scene Generation

Two strategies are implemented:

1. Direct UAV images from folder with EXIF GPS extraction
2. Orthophoto tiling into fixed metric-size tiles (e.g., 10 m x 10 m)

For orthophoto tiling, newer scripts generate uniformly distributed tile centroids over the map extent and optionally apply random in-plane tile rotation. Tile metadata stores both projected-map coordinates and WGS84 GPS.

### 2.3 UGV RGB-D Extraction and Synchronization

UGV frames are extracted from rosbag data. Depth is aligned to color, and timestamps are synchronized with GPS tracks by:

- Estimating an offset between initial camera and GPS timestamps
- Matching each frame to nearest GPS sample under a threshold

The processing retains RGB frame path, depth frame path, camera intrinsics, and pose-related metadata.

### 2.4 Pose Estimation for UGV Metadata

Yaw orientation is estimated from local trajectory direction computed over a GPS temporal window. Camera pose is represented as a rigid transform matrix and stored as world-to-camera (`camera_pose_w2c`) in each scene metadata record.

### 2.5 Scene Assignment

Each synchronized UGV sample is assigned to one or more UAV scenes using geometric proximity criteria:

- Radius-based nearest assignment in some scripts
- Tile-bound checks in newer scripts (inside tile footprint)

### 2.6 Scene-Centric Output Format

Each scene includes:

- One UAV image path (`uav_image_path`)
- Multiple UGV observations (`ugv_images` list), each with:
	- RGB path
	- Depth path (if used)
	- Intrinsics matrix
	- `camera_pose_w2c`

This is serialized in `metadata.json` per scene.

---

## 3. Training Dataset Sampling and Preprocessing

`VineyardDataset` in `dataset.py` is responsible for loading scenes and producing model-ready tensors.

### 3.1 Scene Filtering

Scenes are excluded if no valid UGV views remain after applying optional edge-margin filtering (`edge_margin_m`) in local tile coordinates.

### 3.2 UGV View Selection

Two selection modes are implemented:

- Random view sampling
- Consecutive frame sampling (`consecutive_frames=True`) via contiguous camera index groups

If available views are fewer than requested (`num_ugv_views`), views are repeated to meet fixed cardinality.

### 3.3 Image/Depth Transformations

Typical training transforms:

- Resize to fixed input size (default 224 x 224)
- Float conversion
- ImageNet-style RGB normalization for RGB
- Depth resized and converted to float

Intrinsics are scaled consistently with resize operations.

### 3.4 BEV Grid Definition

A 3D grid is generated from:

- `grid_size = (X, Y, Z)`
- `grid_resolution` in meters

Grid points are centered over local scene coordinates and used for geometric projection/splatting inside the model.

---

## 4. Model Architecture

The core model (`SnapViT`) contains two encoders with a shared design principle: transform each modality into BEV-aligned feature maps.

### 4.1 Shared Backbone: ViT Feature Extractor

`ViTFeatureExtractor` uses a timm Vision Transformer backbone. The classifier head is removed, and patch tokens are reshaped into a dense 2D feature map of shape `(C, H_feat, W_feat)`.

### 4.2 Ground Encoder (UGV -> BEV)

`GroundEncoder` performs geometry-aware lifting of multi-view UGV RGB-D observations:

1. Extract per-view ViT feature maps
2. Resize depth to feature resolution
3. Backproject depth pixels to 3D camera coordinates via intrinsics
4. Transform points from camera to world using inverse `w2c`
5. Map world `(x, y)` to BEV pixel indices
6. Splat per-view features into BEV grid
7. Aggregate overlapping contributions (implemented default: average)
8. Project channel dimension through an MLP (`vit_embed_dim -> 512 -> feature_dim`)

The encoder also outputs a BEV validity mask (`count > 0`) used by masked losses.

### 4.3 Overhead Encoder (UAV -> BEV)

`OverheadEncoder` applies the same ViT feature extraction to UAV input and uses a `1x1` convolution to project to `feature_dim`, producing an overhead BEV feature map.

### 4.4 Temperature Parameter

`SnapViT` includes a learnable scalar temperature parameter used by contrastive objectives.

---

## 5. Learning Objectives

Training in `train.py` combines global and local contrastive signals.

### 5.1 Pixel-Level Masked InfoNCE

`masked_info_nce_loss` computes dense contrastive loss at valid BEV positions only. Similarity logits are built from normalized feature vectors and divided by temperature.

### 5.2 Global Symmetric Contrastive Loss

`symmetric_info_nce_loss_masked` performs masked average pooling over BEV maps to obtain one embedding per scene for each modality, then applies bidirectional InfoNCE:

- Ground-to-overhead classification
- Overhead-to-ground classification

Final global loss is the mean of both directions.

### 5.3 Mixed Loss Schedule

The training schedule is staged:

- Early epochs (`epoch < mixed_loss_delay`): optimize global loss only
- Later epochs: optimize weighted mixture

$$
\mathcal{L} = \lambda \mathcal{L}_{pixel} + (1-\lambda) \mathcal{L}_{global}
$$

with `lambda = pixel_loss_weight`.

---

## 6. Optimization and Training Protocol

Default protocol in `train.py`:

- Optimizer: Adam
- Learning rate: `1e-4`
- Batch size: typically `8`
- Epochs: up to `1000`
- Train/validation split via random split (`val_split_ratio`)
- TensorBoard logging for train/validation total, pixel, and global losses
- Best checkpoint by validation loss + final checkpoint at end

Training history is persisted as CSV and JSON config under model output directory.

---

## 7. Evaluation Methodology

`eval_metrics.py` implements robustness-oriented benchmark scenarios using cosine similarity statistics over valid BEV regions.

### 7.1 Core Metric

Given predicted ground and overhead BEV maps, overhead features are resized to ground resolution and per-pixel cosine similarity is computed.

For each scene/scenario, summary statistics include:

- mean, median, std, min, max, p10, p90
- valid pixel count and valid ratio

### 7.2 Perturbation Scenarios

The evaluation pipeline supports controlled perturbations, including:

- UAV image rotations
- Ground camera yaw perturbations
- Ground image rotations (and depth rotation)
- Swapped UAV or swapped ground data (mismatch stress tests)

Results are exported as per-scene and per-scenario CSV reports.

---

## 8. Localization / Pose Search Methodology

`estimate_position.py` performs explicit hypothesis-based localization by scanning translation and yaw candidates:

1. Define local XY grid around candidate area
2. For each `(dx, dy)` and yaw candidate:
	 - Modify UGV camera pose hypothesis
	 - Run model forward
	 - Compute scalar similarity score from BEV cosine agreement
3. Keep best yaw per translation cell
4. Convert score grid to probability map with temperature-softmax
5. Return top-K hypotheses and visualize against UAV image

This realizes model-based pose inference in BEV feature space rather than direct regression.

---

## 9. Visualization Methodology

`visualize.py` generates qualitative diagnostics:

- Feature-map RGB projections via PCA
- Ground vs overhead BEV map comparisons
- Valid-region cosine similarity heatmaps
- Saved scene panels (UAV image, UGV collage, BEV outputs)

These visual outputs complement scalar metrics and facilitate failure analysis.

---

## 10. Implementation Notes and Reproducibility Considerations

- The codebase contains multiple dataset scripts reflecting iterative experimentation; methodology is consistent but implementation details vary by dataset version.
- In `GroundEncoder`, `ground_tile_size` is currently forced to `10.0` in forward pass, which should match dataset generation assumptions.
- Aggregation alternatives (`max`, `avgmax`) are scaffolded but average aggregation is the active stable path.
- Validity masking is central to both training and evaluation, reducing bias from unobserved BEV cells.

---

## 11. End-to-End Pipeline Summary

1. Build synchronized scene dataset from UAV and UGV sources.
2. Load scenes with fixed multi-view UGV sampling and BEV grid generation.
3. Encode UGV RGB-D and UAV image into BEV feature maps.
4. Train with staged global + pixel masked contrastive objectives.
5. Validate and checkpoint by best validation loss.
6. Evaluate robustness via perturbation scenarios and cosine-statistics benchmarks.
7. Perform pose hypothesis search for localization and inspect outputs with qualitative visualization tools.

This methodology defines the complete scientific workflow currently implemented in the repository.
