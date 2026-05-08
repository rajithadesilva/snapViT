import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm
import os
import csv
import json
import numpy as np

from torch.utils.tensorboard import SummaryWriter

from src.models.snapvit import SnapViT
from src.data.dataset import VineyardDataset

# --- Configuration ---
CONFIG = {
    'data_root': '/media/data/alessandro/GAIA/tempovine/dataset_tempovine_train',
    'vit_model': 'vit_small_patch16_224', # Use a smaller model for faster training
    'train_img_size': (224, 224),
    'feature_dim': 128,
    'num_ugv_views': 8,
    'grid_size': (34, 34, 8), # Smaller grid for faster training
    'grid_resolution': 0.3, # meters per grid cell
    'batch_size': 8, # Adjust based on your GPU memory
    'num_workers': 4, # Reduced from 8 (8 workers for batch_size=8 is inefficient)
    'pin_memory': True, # Pin data to CPU memory for faster GPU transfer
    'learning_rate': 1e-4,
    'epochs': 1000,
    'device': 'cuda:0',
    'val_split_ratio': 0.2, 
    'use_depth': True,
    'depth_range': (0.0, 5.0), # meters
    'ground_tile_size': 10.0, # meters,
    'output_model_path': 'models/da_buttare',
    'consecutive_frames': True, # Whether to select consecutive frames for UGV views
    'pixel_loss_weight': 1.0, # Weight for the pixel-level loss component (if implemented)
    'mixed_loss_delay': 5, # Number of epochs to wait before starting to include the pixel-level loss in the total loss calculation
        'gradient_checkpointing': False,  # Set to True if running out of GPU memory
        'monitor_gpu_memory': True,  # Print GPU memory usage per epoch
    
}

#TODO: write a proper infoNCE loss function
def info_nce_loss(features1, features2, temperature):
    """
    Calculates the InfoNCE loss for two sets of feature maps.
    """
    B, C, H, W = features1.shape

    # Normalize features for stable cosine similarity
    features1 = F.normalize(features1, p=2, dim=1)
    features2 = F.normalize(features2, p=2, dim=1)

    # Reshape for matrix multiplication
    features1_flat = features1.permute(0, 2, 3, 1).reshape(B * H * W, C)
    features2_flat = features2.permute(0, 2, 3, 1).reshape(B * H * W, C).T

    # Compute similarity matrix
    logits = torch.matmul(features1_flat, features2_flat) / temperature

    # Labels are the diagonal indices
    labels = torch.arange(B * H * W, device=features1.device)

    # Cross-entropy loss
    return F.cross_entropy(logits, labels)

def masked_info_nce_loss(features1, features2, validity_mask, temperature):
    """
    Calculates the InfoNCE loss for two sets of feature maps, considering only valid positions.
    validity_mask: (B, H, W) boolean tensor indicating valid positions
    """
    B, C, H, W = features1.shape

    safe_temperature = torch.clamp(temperature, min=1e-6)

    # Normalize features for stable cosine similarity
    features1 = F.normalize(features1, p=2, dim=1)
    features2 = F.normalize(features2, p=2, dim=1)

    # Reshape for matrix multiplication
    features1_flat = features1.permute(0, 2, 3, 1).reshape(B * H * W, C)
    features2_flat = features2.permute(0, 2, 3, 1).reshape(B * H * W, C).T

    # Compute similarity matrix
    logits = torch.matmul(features1_flat, features2_flat) / safe_temperature

    # Create mask for valid positions
    validity_mask_flat = validity_mask.reshape(-1).bool()  # (B*H*W,)
    valid_indices = torch.where(validity_mask_flat)[0]

    if valid_indices.numel() == 0:
        return logits.sum() * 0.0

    filtered_logits = logits[valid_indices][:, valid_indices]

    if not torch.isfinite(filtered_logits).all():
        filtered_logits = torch.nan_to_num(filtered_logits, nan=0.0, posinf=1e4, neginf=-1e4)

    labels = torch.arange(len(valid_indices), device=features1.device)

    # Cross-entropy loss
    return F.cross_entropy(filtered_logits, labels)

def masked_avg_pool(features, mask):
    """
    features: (B, C, H, W)
    mask: (B, H, W) or (B, 1, H, W) bool
    """
    # Accept both (B,H,W) and (B,1,H,W) validity masks.
    if mask.dim() == 4 and mask.size(1) == 1:
        mask = mask.squeeze(1)
    elif mask.dim() != 3:
        raise ValueError(f"Expected mask shape (B,H,W) or (B,1,H,W), got {tuple(mask.shape)}")

    mask = mask.unsqueeze(1).float()  # (B,1,H,W)

    masked_features = features * mask

    sum_feat = masked_features.sum(dim=(2, 3))           # (B,C)
    valid_counts = mask.sum(dim=(2, 3)).clamp(min=1e-6)  # (B,1)

    return sum_feat / valid_counts



def symmetric_info_nce_loss_masked(ground_bev, overhead_bev, validity_mask, temperature):
    """
    ground_bev:   (B, C, H, W)
    overhead_bev: (B, C, H, W)
    validity_mask:(B, H, W) bool
    """

    safe_temperature = torch.clamp(temperature, min=1e-6)

    # masked global pooling
    ground_feat = masked_avg_pool(ground_bev, validity_mask)
    overhead_feat = masked_avg_pool(overhead_bev, validity_mask)

    # normalize embeddings
    ground_feat = F.normalize(ground_feat, p=2, dim=1)
    overhead_feat = F.normalize(overhead_feat, p=2, dim=1)

    # similarity matrices (B,C) x (C,B) -> (B,B)
    logits_g2o = torch.matmul(ground_feat, overhead_feat.transpose(0, 1)) / safe_temperature
    logits_o2g = torch.matmul(overhead_feat, ground_feat.transpose(0, 1)) / safe_temperature

    # positives on the diagonal
    labels = torch.arange(ground_feat.size(0), device=ground_feat.device)

    loss_g2o = F.cross_entropy(logits_g2o, labels)
    loss_o2g = F.cross_entropy(logits_o2g, labels)

    return 0.5 * (loss_g2o + loss_o2g)


def pool_bev_embeddings(features, validity_mask=None):
    """Pool a BEV feature map into a single embedding per sample.

    If `validity_mask` is provided, use masked average pooling; otherwise use
    standard global average pooling.
    """
    if validity_mask is None:
        return features.mean(dim=(2, 3))
    return masked_avg_pool(features, validity_mask)


def cosine_distance(x, y, eps=1e-8):
    """Returns 1 - cosine similarity for matching rows in x and y."""
    x = F.normalize(x, p=2, dim=1)
    y = F.normalize(y, p=2, dim=1)
    return 1.0 - (x * y).sum(dim=1).clamp(-1.0 + eps, 1.0 - eps)


def masked_mse_loss(features1, features2, validity_mask=None):
    """Masked mean squared error on BEV feature maps."""
    diff = (features1 - features2) ** 2
    if validity_mask is None:
        return diff.mean()

    if validity_mask.dim() == 3:
        mask = validity_mask.unsqueeze(1).float()
    elif validity_mask.dim() == 4 and validity_mask.size(1) == 1:
        mask = validity_mask.float()
    else:
        raise ValueError(f"Expected validity_mask shape (B,H,W) or (B,1,H,W), got {tuple(validity_mask.shape)}")

    masked_diff = diff * mask
    return masked_diff.sum() / mask.sum().clamp(min=1e-8)


def masked_smooth_l1_loss(features1, features2, validity_mask=None, beta=1.0):
    """Masked Smooth L1 loss on BEV feature maps."""
    diff = F.smooth_l1_loss(features1, features2, reduction='none', beta=beta)
    if validity_mask is None:
        return diff.mean()

    if validity_mask.dim() == 3:
        mask = validity_mask.unsqueeze(1).float()
    elif validity_mask.dim() == 4 and validity_mask.size(1) == 1:
        mask = validity_mask.float()
    else:
        raise ValueError(f"Expected validity_mask shape (B,H,W) or (B,1,H,W), got {tuple(validity_mask.shape)}")

    masked_diff = diff * mask
    return masked_diff.sum() / mask.sum().clamp(min=1e-8)


def masked_cosine_loss(features1, features2, validity_mask=None):
    """Masked cosine alignment loss on BEV feature maps."""
    f1 = F.normalize(features1, p=2, dim=1)
    f2 = F.normalize(features2, p=2, dim=1)
    loss_map = 1.0 - (f1 * f2).sum(dim=1)

    if validity_mask is None:
        return loss_map.mean()

    if validity_mask.dim() == 3:
        mask = validity_mask.float()
    elif validity_mask.dim() == 4 and validity_mask.size(1) == 1:
        mask = validity_mask.squeeze(1).float()
    else:
        raise ValueError(f"Expected validity_mask shape (B,H,W) or (B,1,H,W), got {tuple(validity_mask.shape)}")

    return (loss_map * mask).sum() / mask.sum().clamp(min=1e-8)


def nt_xent_loss(anchor, positive, temperature=0.1, validity_mask=None):
    """Standard NT-Xent / InfoNCE loss for paired embeddings or BEV maps.

    If `validity_mask` is provided, `anchor` and `positive` are treated as BEV
    feature maps and are pooled before computing the loss.
    """
    if validity_mask is not None:
        anchor = pool_bev_embeddings(anchor, validity_mask)
        positive = pool_bev_embeddings(positive, validity_mask)

    anchor = F.normalize(anchor, p=2, dim=1)
    positive = F.normalize(positive, p=2, dim=1)

    logits = torch.matmul(anchor, positive.t()) / torch.clamp(temperature, min=1e-6)
    labels = torch.arange(anchor.size(0), device=anchor.device)
    return F.cross_entropy(logits, labels)


def supervised_contrastive_loss(embeddings, labels, temperature=0.1, validity_mask=None):
    """Supervised contrastive loss for embeddings or BEV maps.

    If `validity_mask` is provided, `embeddings` is pooled before computing the loss.
    """
    if validity_mask is not None:
        embeddings = pool_bev_embeddings(embeddings, validity_mask)

    embeddings = F.normalize(embeddings, p=2, dim=1)
    logits = torch.matmul(embeddings, embeddings.t()) / torch.clamp(temperature, min=1e-6)
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()

    labels = labels.view(-1, 1)
    positive_mask = torch.eq(labels, labels.t()).to(embeddings.device)
    eye_mask = torch.eye(labels.size(0), dtype=torch.bool, device=embeddings.device)
    positive_mask = positive_mask & ~eye_mask

    log_prob = logits - torch.logsumexp(logits.masked_fill(eye_mask, float('-inf')), dim=1, keepdim=True)
    positive_count = positive_mask.sum(dim=1).clamp(min=1)
    mean_log_prob_pos = (positive_mask.float() * log_prob).sum(dim=1) / positive_count

    return -mean_log_prob_pos.mean()


def triplet_loss(anchor, positive, negative, margin=0.2, distance='cosine', validity_mask=None):
    """Triplet loss on embeddings or BEV maps.

    If `validity_mask` is provided, all three inputs are pooled before computing the loss.
    """
    if validity_mask is not None:
        anchor = pool_bev_embeddings(anchor, validity_mask)
        positive = pool_bev_embeddings(positive, validity_mask)
        negative = pool_bev_embeddings(negative, validity_mask)

    if distance == 'cosine':
        pos_dist = cosine_distance(anchor, positive)
        neg_dist = cosine_distance(anchor, negative)
    elif distance == 'euclidean':
        pos_dist = torch.norm(anchor - positive, dim=1)
        neg_dist = torch.norm(anchor - negative, dim=1)
    else:
        raise ValueError("distance must be 'cosine' or 'euclidean'")

    return F.relu(pos_dist - neg_dist + margin).mean()


def batch_hard_triplet_loss(embeddings, labels, margin=0.2, distance='cosine', validity_mask=None):
    """Batch-hard triplet loss using hardest positive and negative within the batch.

    If `validity_mask` is provided, `embeddings` is treated as a BEV feature map and
    pooled before the loss is computed.
    """
    if validity_mask is not None:
        embeddings = pool_bev_embeddings(embeddings, validity_mask)

    if embeddings.size(0) != labels.size(0):
        raise ValueError("embeddings and labels must have the same batch size")

    if distance == 'cosine':
        sim = F.normalize(embeddings, p=2, dim=1) @ F.normalize(embeddings, p=2, dim=1).t()
        dist = 1.0 - sim
    elif distance == 'euclidean':
        dist = torch.cdist(embeddings, embeddings, p=2)
    else:
        raise ValueError("distance must be 'cosine' or 'euclidean'")

    labels = labels.view(-1, 1)
    same = labels.eq(labels.t())
    eye = torch.eye(labels.size(0), dtype=torch.bool, device=embeddings.device)
    pos_mask = same & ~eye
    neg_mask = ~same

    hardest_pos = dist.masked_fill(~pos_mask, float('-inf')).max(dim=1).values
    hardest_neg = dist.masked_fill(~neg_mask, float('inf')).min(dim=1).values

    valid = torch.isfinite(hardest_pos) & torch.isfinite(hardest_neg)
    if not valid.any():
        return dist.sum() * 0.0

    return F.relu(hardest_pos[valid] - hardest_neg[valid] + margin).mean()


def margin_ranking_triplet_loss(anchor, positive, negative, margin=0.2, distance='cosine', validity_mask=None):
    """Margin ranking loss view of triplet learning on embeddings or BEV maps."""
    if validity_mask is not None:
        anchor = pool_bev_embeddings(anchor, validity_mask)
        positive = pool_bev_embeddings(positive, validity_mask)
        negative = pool_bev_embeddings(negative, validity_mask)

    if distance == 'cosine':
        pos_score = 1.0 - cosine_distance(anchor, positive)
        neg_score = 1.0 - cosine_distance(anchor, negative)
    elif distance == 'euclidean':
        pos_score = -torch.norm(anchor - positive, dim=1)
        neg_score = -torch.norm(anchor - negative, dim=1)
    else:
        raise ValueError("distance must be 'cosine' or 'euclidean'")

    target = torch.ones_like(pos_score)
    return F.margin_ranking_loss(pos_score, neg_score, target, margin=margin)


def cosine_embedding_pair_loss(anchor, positive, negative=None, validity_mask=None):
    """Cosine embedding loss for positive pairs, optionally with negatives.

    If `validity_mask` is provided, inputs are pooled before computing the loss.
    """
    if validity_mask is not None:
        anchor = pool_bev_embeddings(anchor, validity_mask)
        positive = pool_bev_embeddings(positive, validity_mask)
        if negative is not None:
            negative = pool_bev_embeddings(negative, validity_mask)

    anchor = F.normalize(anchor, p=2, dim=1)
    positive = F.normalize(positive, p=2, dim=1)
    pos_loss = F.cosine_embedding_loss(anchor, positive, torch.ones(anchor.size(0), device=anchor.device))

    if negative is None:
        return pos_loss

    negative = F.normalize(negative, p=2, dim=1)
    neg_loss = F.cosine_embedding_loss(anchor, negative, -torch.ones(anchor.size(0), device=anchor.device))
    return 0.5 * (pos_loss + neg_loss)


def barlow_twins_loss(z1, z2, lambda_coeff=5e-3, validity_mask=None):
    """Barlow Twins loss for embeddings or BEV maps.

    If `validity_mask` is provided, inputs are pooled before computing the loss.
    """
    if validity_mask is not None:
        z1 = pool_bev_embeddings(z1, validity_mask)
        z2 = pool_bev_embeddings(z2, validity_mask)

    z1 = (z1 - z1.mean(dim=0)) / (z1.std(dim=0) + 1e-9)
    z2 = (z2 - z2.mean(dim=0)) / (z2.std(dim=0) + 1e-9)

    n = z1.size(0)
    c = (z1.T @ z2) / n

    on_diag = torch.diagonal(c).add_(-1).pow(2).sum()
    off_diag = (c - torch.diag(torch.diagonal(c))).pow(2).sum()
    return on_diag + lambda_coeff * off_diag


def vicreg_loss(z1, z2, sim_coeff=25.0, var_coeff=25.0, cov_coeff=1.0, validity_mask=None):
    """VICReg loss for embeddings or BEV maps.

    If `validity_mask` is provided, inputs are pooled before computing the loss.
    """
    if validity_mask is not None:
        z1 = pool_bev_embeddings(z1, validity_mask)
        z2 = pool_bev_embeddings(z2, validity_mask)

    repr_loss = F.mse_loss(z1, z2)

    std_z1 = torch.sqrt(z1.var(dim=0) + 1e-4)
    std_z2 = torch.sqrt(z2.var(dim=0) + 1e-4)
    var_loss = torch.mean(F.relu(1.0 - std_z1)) + torch.mean(F.relu(1.0 - std_z2))

    z1 = z1 - z1.mean(dim=0)
    z2 = z2 - z2.mean(dim=0)
    n, d = z1.shape
    cov_z1 = (z1.T @ z1) / (n - 1)
    cov_z2 = (z2.T @ z2) / (n - 1)
    off_diag_mask = ~torch.eye(d, dtype=torch.bool, device=z1.device)
    cov_loss = cov_z1.masked_select(off_diag_mask).pow(2).sum() / d + cov_z2.masked_select(off_diag_mask).pow(2).sum() / d

    return sim_coeff * repr_loss + var_coeff * var_loss + cov_coeff * cov_loss


def main():
    print(f"Using device: {CONFIG['device']}")
    best_val_loss = float('inf')

    print(f"Saving models to: {CONFIG['output_model_path']}")
    output_model_path = f"{CONFIG['output_model_path']}/best_model.pth"
    final_model_path = f"{CONFIG['output_model_path']}/final_model.pth"
    history_dir = os.path.join(CONFIG['output_model_path'], 'history')
    history_csv_path = os.path.join(history_dir, 'loss_history.csv')
    history_config_path = os.path.join(history_dir, 'training_config.json')
    if not os.path.exists(CONFIG['output_model_path']):
        os.makedirs(CONFIG['output_model_path'], exist_ok=True)
    if not os.path.exists(history_dir):
        os.makedirs(history_dir, exist_ok=True)

    with open(history_config_path, mode='w') as f:
        json.dump(CONFIG, f, indent=2)

    with open(history_csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'epoch',
            'train_loss',
            'train_pixel_loss',
            'train_global_loss',
            'val_loss',
            'val_pixel_loss',
            'val_global_loss'
        ])

    # --- Data ---

    image_transforms = transforms.Compose([
        transforms.Resize(CONFIG['train_img_size'], antialias=True),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    depth_transforms = transforms.Compose([
        transforms.Resize(CONFIG['train_img_size'], antialias=True), #TODO: try interpolation=transforms.InterpolationMode.NEAREST
        transforms.ConvertImageDtype(torch.float),
    ])

    # --- Dataset Splitting ---
    full_dataset = VineyardDataset(root_dir=CONFIG['data_root'], config=CONFIG, transforms=image_transforms, depth_transforms=depth_transforms, consecutive_frames=CONFIG['consecutive_frames'])
    
    # Ensure dataset is not empty
    if len(full_dataset) == 0:
        raise ValueError("Dataset is empty. Please check the data_root path.")

    val_size = int(CONFIG['val_split_ratio'] * len(full_dataset))
    train_size = len(full_dataset) - val_size

    print(f"Dataset size: {len(full_dataset)}. Splitting into {train_size} training and {val_size} validation samples.")
    
    # Use a generator for reproducible splits
    generator = torch.Generator()#.manual_seed()
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)


    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=CONFIG['batch_size'], 
        shuffle=True, 
        num_workers=CONFIG.get('num_workers', 4),
        pin_memory=CONFIG.get('pin_memory', True),
        prefetch_factor=2
    )
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=CONFIG['batch_size'], 
        shuffle=False, 
        num_workers=CONFIG.get('num_workers', 4),
        pin_memory=CONFIG.get('pin_memory', True),
        prefetch_factor=2
    )


    # --- Model ---
    model = SnapViT(CONFIG).to(CONFIG['device'])
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])

    writer = SummaryWriter(log_dir='runs')

    # Helper function to monitor GPU memory
    def log_gpu_memory(label=""):
        if torch.cuda.is_available() and CONFIG.get('monitor_gpu_memory', False):
            reserved = torch.cuda.memory_reserved(CONFIG['device']) / 1e9
            allocated = torch.cuda.memory_allocated(CONFIG['device']) / 1e9
            print(f"  {label} | GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

    print("Starting training...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU Memory available: {torch.cuda.get_device_properties(CONFIG['device']).total_memory / 1e9:.2f} GB")

    for epoch in range(CONFIG['epochs']):
        # --- Training Loop ---
        model.train()
        total_train_loss = 0
        total_global_train_loss = 0
        total_pixel_train_loss = 0
        train_progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [Training]", leave=False)

        if epoch == 0:
            print("Loading first training batch...")

        for batch in train_progress_bar:
            optimizer.zero_grad()

            # Data to GPU (pre-loaded with pin_memory)
            uav_data = {k: v.to(CONFIG['device'], non_blocking=True) for k, v in batch['uav_data'].items()}
            ugv_data = {k: v.to(CONFIG['device'], non_blocking=True) for k, v in batch['ugv_data'].items()}

            ground_bev, overhead_bev, ground_validity = model(ugv_data, uav_data)
            overhead_bev_resized = F.interpolate(overhead_bev, size=ground_bev.shape[2:], mode='bilinear', align_corners=False)

            #loss = info_nce_loss(ground_bev, overhead_bev_resized, model.temperature)
            pixel_loss = masked_info_nce_loss(ground_bev, overhead_bev_resized, ground_validity, model.temperature)  
            global_loss = symmetric_info_nce_loss_masked(ground_bev, overhead_bev_resized, ground_validity, model.temperature) 

            if epoch < CONFIG['mixed_loss_delay']:
                loss = global_loss
            else:
                loss = CONFIG['pixel_loss_weight'] * pixel_loss + (1 - CONFIG['pixel_loss_weight']) * global_loss
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            total_pixel_train_loss += pixel_loss.item()
            total_global_train_loss += global_loss.item()
            train_progress_bar.set_postfix({'loss': loss.item()})

        avg_train_loss = total_train_loss / len(train_dataloader)
        avg_pixel_loss = total_pixel_train_loss / len(train_dataloader)
        avg_global_loss = total_global_train_loss / len(train_dataloader)

        writer.add_scalar('Loss/Train', avg_train_loss, epoch+1)
        writer.add_scalar('PixelLoss/Train', avg_pixel_loss, epoch+1)
        writer.add_scalar('GlobalLoss/Train', avg_global_loss, epoch+1)

        # Clear cache between epochs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # --- Validation Loop ---
        model.eval()
        total_val_loss = 0
        total_pixel_val_loss = 0
        total_global_val_loss = 0   
        val_progress_bar = tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [Validation]", leave=False)
        
        with torch.no_grad():
            for batch in val_progress_bar:
                uav_data = {k: v.to(CONFIG['device'], non_blocking=True) for k, v in batch['uav_data'].items()}
                ugv_data = {k: v.to(CONFIG['device'], non_blocking=True) for k, v in batch['ugv_data'].items()}

                ground_bev, overhead_bev, ground_validity = model(ugv_data, uav_data)
                overhead_bev_resized = F.interpolate(overhead_bev, size=ground_bev.shape[2:], mode='bilinear', align_corners=False)

                #loss = info_nce_loss(ground_bev, overhead_bev_resized, model.temperature)
                pixel_loss = masked_info_nce_loss(ground_bev, overhead_bev_resized, ground_validity, model.temperature)  
                global_loss = symmetric_info_nce_loss_masked(ground_bev, overhead_bev_resized, ground_validity, model.temperature) 
                loss = CONFIG['pixel_loss_weight'] * pixel_loss + (1 - CONFIG['pixel_loss_weight']) * global_loss
                total_val_loss += loss.item()
                total_pixel_val_loss += pixel_loss.item()
                total_global_val_loss += global_loss.item()
                val_progress_bar.set_postfix({'loss': loss.item()})
        
        avg_val_loss = total_val_loss / len(val_dataloader)
        avg_val_pixel_loss = total_pixel_val_loss / len(val_dataloader)
        avg_val_global_loss = total_global_val_loss / len(val_dataloader)
        writer.add_scalar('Loss/Validation', avg_val_loss, epoch+1)
        writer.add_scalar('PixelLoss/Validation', avg_val_pixel_loss, epoch+1)
        writer.add_scalar('GlobalLoss/Validation', avg_val_global_loss, epoch+1)

        with open(history_csv_path, mode='a', newline='') as f:
            csv.writer(f).writerow([
                epoch + 1,
                avg_train_loss,
                avg_pixel_loss,
                avg_global_loss,
                avg_val_loss,
                avg_val_pixel_loss,
                avg_val_global_loss,
            ])

        print(f"Epoch {epoch+1}/{CONFIG['epochs']} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # --- Save the best model ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), output_model_path)
            print(f"  -> New best model found! Saved to {output_model_path} (Val Loss: {best_val_loss:.4f})")
    torch.save(model.state_dict(), final_model_path)
    writer.close()

if __name__ == '__main__':
    main()
