import argparse
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.dataset import VineyardDataset
from src.models.snapvit import SnapViT
from src.training.losses import masked_info_nce_loss, symmetric_info_nce_loss_masked


CONFIG = {
    'data_root': '/media/data/alessandro/GAIA/tempovine/dataset_tempovine_train',
    'vit_model': 'vit_small_patch16_224',
    'train_img_size': (224, 224),
    'feature_dim': 128,
    'ground_fusion_mode': 'mlp',
    'use_height_positional_encoding': False,
    'num_ugv_views': 1,
    'grid_size': (34, 34, 8),
    'grid_resolution': 0.3,
    'batch_size': 8,
    'num_workers': 4,
    'pin_memory': True,
    'learning_rate': 1e-4,
    'epochs': 1000,
    'device': 'cuda:0',
    'val_split_ratio': 0.2,
    'use_depth': True,
    'depth_range': (0.0, 5.0),
    'ground_tile_size': 10.0,
    'output_model_path': 'models/da_buttare',
    'consecutive_frames': True,
    'pixel_loss_weight': 1.0,
    'mixed_loss_delay': 5,
    'gradient_checkpointing': False,
    'monitor_gpu_memory': True,
    'patience': None,
}


def normalize_config(config):
    resolved = dict(config)
    for key in ('train_img_size', 'grid_size', 'depth_range'):
        if key in resolved and isinstance(resolved[key], list):
            resolved[key] = tuple(resolved[key])

    resolved.setdefault('num_workers', 4)
    resolved.setdefault('pin_memory', True)
    resolved.setdefault('ground_fusion_mode', 'avg')
    resolved.setdefault('use_height_positional_encoding', False)
    resolved.setdefault('use_pixel_loss', True)
    resolved.setdefault('use_global_loss', True)
    resolved.setdefault('pixel_loss_weight', 1.0)
    resolved.setdefault('mixed_loss_delay', 0)
    resolved.setdefault('save_history_csv', True)
    resolved.setdefault('tensorboard_log_dir', 'runs')
    resolved.setdefault('patience', None)
    return resolved


def build_dataloaders(config, seed=None):
    resolved = normalize_config(config)

    image_transforms = transforms.Compose([
        transforms.Resize(resolved['train_img_size'], antialias=True),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    depth_transforms = transforms.Compose([
        transforms.Resize(resolved['train_img_size'], antialias=True),
        transforms.ConvertImageDtype(torch.float),
    ])

    full_dataset = VineyardDataset(
        root_dir=resolved['data_root'],
        config=resolved,
        transforms=image_transforms,
        depth_transforms=depth_transforms,
        consecutive_frames=resolved['consecutive_frames'],
    )

    if len(full_dataset) == 0:
        raise ValueError('Dataset is empty. Please check the data_root path.')

    val_size = int(resolved['val_split_ratio'] * len(full_dataset))
    train_size = len(full_dataset) - val_size

    print(f"Dataset size: {len(full_dataset)}. Splitting into {train_size} training and {val_size} validation samples.")

    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

    dataloader_kwargs = {
        'batch_size': resolved['batch_size'],
        'num_workers': resolved.get('num_workers', 4),
        'pin_memory': resolved.get('pin_memory', True),
    }
    if dataloader_kwargs['num_workers'] > 0:
        dataloader_kwargs['prefetch_factor'] = 2

    train_dataloader = DataLoader(train_dataset, shuffle=True, **dataloader_kwargs)
    val_dataloader = DataLoader(val_dataset, shuffle=False, **dataloader_kwargs)
    return train_dataloader, val_dataloader


def _select_loss(config, pixel_loss, global_loss, epoch):
    use_pixel_loss = config.get('use_pixel_loss', True)
    use_global_loss = config.get('use_global_loss', True)
    pixel_loss_weight = config.get('pixel_loss_weight', 1.0)
    mixed_loss_delay = config.get('mixed_loss_delay', 0)

    if epoch < mixed_loss_delay:
        if use_global_loss:
            return global_loss
        if use_pixel_loss:
            return pixel_loss
        return global_loss + pixel_loss

    if use_pixel_loss and use_global_loss:
        return pixel_loss_weight * pixel_loss + (1 - pixel_loss_weight) * global_loss
    if use_pixel_loss:
        return pixel_loss
    return global_loss


def train_loop(config, train_dataloader=None, val_dataloader=None, save_history=True, writer_log_dir='runs'):
    training_config = CONFIG.copy()
    if config is not None:
        if 'ground_fusion_mode' not in config:
            training_config['ground_fusion_mode'] = 'avg'
        if 'use_height_positional_encoding' not in config:
            training_config['use_height_positional_encoding'] = False
        training_config.update(config)

    resolved = normalize_config(training_config)

    if train_dataloader is None or val_dataloader is None:
        train_dataloader, val_dataloader = build_dataloaders(resolved)

    print(f"Using device: {resolved['device']}")

    output_model_path = resolved['output_model_path']
    best_model_path = os.path.join(output_model_path, 'best_model.pth')
    final_model_path = os.path.join(output_model_path, 'final_model.pth')
    history_dir = os.path.join(output_model_path, 'history')
    history_csv_path = os.path.join(history_dir, 'loss_history.csv')
    config_path = os.path.join(output_model_path, 'config.json')

    print(f"Saving models to: {output_model_path}")
    os.makedirs(output_model_path, exist_ok=True)
    if save_history:
        os.makedirs(history_dir, exist_ok=True)

    with open(config_path, 'w') as f:
        json.dump(resolved, f, indent=2)

    if save_history:
        with open(history_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'epoch',
                'train_loss',
                'train_pixel_loss',
                'train_global_loss',
                'val_loss',
                'val_pixel_loss',
                'val_global_loss',
            ])

    model = SnapViT(resolved).to(resolved['device'])
    optimizer = torch.optim.Adam(model.parameters(), lr=resolved['learning_rate'])
    writer = SummaryWriter(log_dir=writer_log_dir)
    history = []
    best_val_loss = float('inf')
    patience_start_epoch = resolved.get('mixed_loss_delay', 0)
    patience = resolved.get('patience')
    epochs_without_improvement = 0
    start_time = datetime.now()

    def log_gpu_memory(label=''):
        if torch.cuda.is_available() and resolved.get('monitor_gpu_memory', False):
            reserved = torch.cuda.memory_reserved(resolved['device']) / 1e9
            allocated = torch.cuda.memory_allocated(resolved['device']) / 1e9
            print(f"  {label} | GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

    print('Starting training...')
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU Memory available: {torch.cuda.get_device_properties(resolved['device']).total_memory / 1e9:.2f} GB")

    for epoch in range(resolved['epochs']):
        model.train()
        total_train_loss = 0.0
        total_global_train_loss = 0.0
        total_pixel_train_loss = 0.0
        train_progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{resolved['epochs']} [Training]", leave=False)

        if epoch == 0:
            print('Loading first training batch...')

        for batch in train_progress_bar:
            optimizer.zero_grad()

            uav_data = {k: v.to(resolved['device'], non_blocking=True) for k, v in batch['uav_data'].items()}
            ugv_data = {k: v.to(resolved['device'], non_blocking=True) for k, v in batch['ugv_data'].items()}

            ground_bev, overhead_bev, ground_validity = model(ugv_data, uav_data)
            overhead_bev_resized = F.interpolate(overhead_bev, size=ground_bev.shape[2:], mode='bilinear', align_corners=False)

            pixel_loss = masked_info_nce_loss(ground_bev, overhead_bev_resized, ground_validity, model.temperature)
            global_loss = symmetric_info_nce_loss_masked(ground_bev, overhead_bev_resized, ground_validity, model.temperature)
            loss = _select_loss(resolved, pixel_loss, global_loss, epoch)

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            total_pixel_train_loss += pixel_loss.item()
            total_global_train_loss += global_loss.item()
            train_progress_bar.set_postfix({'loss': loss.item()})

        avg_train_loss = total_train_loss / len(train_dataloader)
        avg_pixel_loss = total_pixel_train_loss / len(train_dataloader)
        avg_global_loss = total_global_train_loss / len(train_dataloader)

        writer.add_scalar('Loss/Train', avg_train_loss, epoch + 1)
        writer.add_scalar('PixelLoss/Train', avg_pixel_loss, epoch + 1)
        writer.add_scalar('GlobalLoss/Train', avg_global_loss, epoch + 1)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        model.eval()
        total_val_loss = 0.0
        total_pixel_val_loss = 0.0
        total_global_val_loss = 0.0
        val_progress_bar = tqdm(val_dataloader, desc=f"Epoch {epoch + 1}/{resolved['epochs']} [Validation]", leave=False)

        with torch.no_grad():
            for batch in val_progress_bar:
                uav_data = {k: v.to(resolved['device'], non_blocking=True) for k, v in batch['uav_data'].items()}
                ugv_data = {k: v.to(resolved['device'], non_blocking=True) for k, v in batch['ugv_data'].items()}

                ground_bev, overhead_bev, ground_validity = model(ugv_data, uav_data)
                overhead_bev_resized = F.interpolate(overhead_bev, size=ground_bev.shape[2:], mode='bilinear', align_corners=False)

                pixel_loss = masked_info_nce_loss(ground_bev, overhead_bev_resized, ground_validity, model.temperature)
                global_loss = symmetric_info_nce_loss_masked(ground_bev, overhead_bev_resized, ground_validity, model.temperature)
                loss = _select_loss(resolved, pixel_loss, global_loss, epoch)

                total_val_loss += loss.item()
                total_pixel_val_loss += pixel_loss.item()
                total_global_val_loss += global_loss.item()
                val_progress_bar.set_postfix({'loss': loss.item()})

        avg_val_loss = total_val_loss / len(val_dataloader)
        avg_val_pixel_loss = total_pixel_val_loss / len(val_dataloader)
        avg_val_global_loss = total_global_val_loss / len(val_dataloader)

        writer.add_scalar('Loss/Validation', avg_val_loss, epoch + 1)
        writer.add_scalar('PixelLoss/Validation', avg_val_pixel_loss, epoch + 1)
        writer.add_scalar('GlobalLoss/Validation', avg_val_global_loss, epoch + 1)

        record = {
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'train_pixel_loss': avg_pixel_loss,
            'train_global_loss': avg_global_loss,
            'val_loss': avg_val_loss,
            'val_pixel_loss': avg_val_pixel_loss,
            'val_global_loss': avg_val_global_loss,
            'best_val_loss': best_val_loss,
            'training_time_sec': (datetime.now() - start_time).total_seconds(),
            'timestamp': datetime.now().isoformat(),
        }

        if save_history:
            with open(history_csv_path, 'a', newline='') as f:
                csv.writer(f).writerow([
                    record['epoch'],
                    record['train_loss'],
                    record['train_pixel_loss'],
                    record['train_global_loss'],
                    record['val_loss'],
                    record['val_pixel_loss'],
                    record['val_global_loss'],
                ])

        log_gpu_memory(f'Epoch {epoch + 1}')
        print(f"Epoch {epoch + 1}/{resolved['epochs']} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if epoch >= patience_start_epoch and avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> New best model found! Saved to {best_model_path} (Val Loss: {best_val_loss:.4f})")
        elif epoch >= patience_start_epoch and patience is not None:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(
                    f"  -> Early stopping triggered after {patience} epochs without validation improvement."
                )
                record['best_val_loss'] = best_val_loss
                history.append(record)
                break

        record['best_val_loss'] = best_val_loss
        history.append(record)

    torch.save(model.state_dict(), final_model_path)
    writer.close()

    return {
        'best_val_loss': best_val_loss,
        'history': history,
        'best_model_path': best_model_path,
        'final_model_path': final_model_path,
        'config_path': config_path,
    }


def main():
    parser = argparse.ArgumentParser(description='Train the model')
    parser.add_argument('--config', type=str, default=None, help='Path to the configuration file')
    args = parser.parse_args()

    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
        config.setdefault('ground_fusion_mode', 'avg')
        config.setdefault('use_height_positional_encoding', False)
        CONFIG.update(config)

    train_loop(CONFIG)


if __name__ == '__main__':
    main()
