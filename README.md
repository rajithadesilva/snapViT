
# SnapViT

SnapViT is a contrastive learning pipeline that uses Vision Transformers (ViT) to create unified Bird's-Eye View (BEV) representations from both aerial (UAV) and ground-level (UGV) images of the same scene. By training the model to recognize corresponding ground and aerial views, it learns to generate powerful, viewpoint-invariant feature maps.

## Overview

The core of the project is the SnapViT model, which consists of two main components:

- **Ground Encoder**: Processes multiple UGV images and their camera poses, projecting their features into a unified BEV feature map.
- **Overhead Encoder**: Processes a single UAV image to create a corresponding BEV feature map.

Both encoders share a Vision Transformer (ViT) backbone for feature extraction. The model is trained using an **InfoNCE contrastive loss** function, which encourages BEV maps from corresponding ground and aerial views to be similar while being dissimilar from non-corresponding views.

---

## 📁 Project Structure

| File | Description |
|------|-------------|
| `model.py` | Defines the core SnapViT architecture: GroundEncoder, OverheadEncoder, and ViTFeatureExtractor. |
| `train.py` | Main script for training the SnapViT model. |
| `dataset.py` | Contains `VineyardDataset` for loading UAV and UGV scene data. |
| `create_dataset.py` | Converts raw data (GeoTIFFs, ROS bags, GPS `.llh` files) into structured datasets. |
| `create_dataset_from_frames.py` | Simpler dataset creation script from image folders and CSV. |
| `verify_dataset.py` | Visualizes and verifies alignment between UAV and UGV data. |
| `visualize.py` | Runs a trained model and saves generated BEV maps for inspection. |
| `requirements.txt` | Python dependencies for the project. |

---

## 🚀 Setup and Installation

```bash
git clone https://github.com/rajithadesilva/snapViT.git
cd snapViT
```

Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

For dataset creation from raw robotics data:

```bash
pip install pyrealsense2 rasterio
```

---

## 📦 Usage Pipeline

### 1. Data Preparation

#### Option A: Using ROS Bags and GeoTIFFs

```bash
python create_dataset.py \
    --drone_geotiff /path/to/orthophoto.tif \
    --ground_rosbag /path/to/ugv_data.bag \
    --ground_llh /path/to/ugv_gps.llh \
    --output_dir datasets/vineyard_dataset \
    --scene_radius 15 \
    --tile_ground_size 10.0
```

#### Option B: Using Image Folders and CSV

```bash
python create_dataset_from_frames.py \
    --drone_folder /path/to/drone_images \
    --ground_folder /path/to/ground_images \
    --ground_csv /path/to/ground_gps.csv \
    --output_dir datasets/vineyard_dataset \
    --scene_radius 10.0
```

---

### 2. Dataset Verification (Optional)

Visually inspect alignment between UGV and UAV data:

```bash
python verify_dataset.py \
    --dataset_dir datasets/vineyard_dataset \
    --output_dir verification \
    --tile_ground_size 10.0
```

> Note: Ensure `tile_ground_size` matches the value used during dataset creation.

---

### 3. Model Training

Train the SnapViT model:

```bash
python train.py
```

- Adjust hyperparameters via the `CONFIG` dictionary in `train.py`.
- Best model is saved to `models/best_model.pth`.

---

### 4. Visualization

Generate BEV feature maps using a trained model:

```bash
python visualize.py \
    --data_root datasets/vineyard_dataset \
    --checkpoint models/best_model.pth \
    --output_dir visualisations
```

Outputs include:
- UAV image
- UGV image collage
- Ground BEV map
- Aerial BEV map

---

## 📄 License

[MIT License](LICENSE)

## 🤝 Contributions

Contributions are welcome! Please open issues or pull requests.
