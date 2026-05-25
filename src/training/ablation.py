"""
Ablation Study Script for SnapViT
Systematically trains multiple model configurations to evaluate the impact of:
- Model architectures (ViT, ResNet, ConvNeXT, Swin, DinoV3)
- Feature dimensions
- Loss weights (pixel vs global)
- Mixed loss delay
- Results saved to CSV for analysis
"""

import os
import sys
import csv
import json
import copy
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm
from datetime import datetime
import itertools

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.snapvit import SnapViT
from data.dataset import VineyardDataset
from train_loop import (
    info_nce_loss,
    masked_info_nce_loss,
    symmetric_info_nce_loss_masked
)


class AblationConfig:
    """Configuration for ablation study"""
    
    # Base configuration (shared across all experiments)
    BASE_CONFIG = {
        'data_root': '/media/hdd/ale_navone/GAIA/tempovine/dataset_tempovine_new',
        'train_img_size': (224, 224),
        'num_ugv_views': 8,
        'grid_size': (34, 34, 8),
        'grid_resolution': 0.3,
        'batch_size': 8,
        'learning_rate': 1e-4,
        'epochs': 50,  # Reduced for ablation study
        'device': 'cuda:0',
        'val_split_ratio': 0.2,
        'use_depth': True,
        'depth_range': (0.0, 5.0),
        'ground_tile_size': 10.0,
        'consecutive_frames': True,
        'pretrained_backbones': True,
    }
    
    # Parameters to ablate
    MODELS = [
        'vit_base_patch16_224',
        'resnet50',
        'convnext_base',
        'swin_small_patch4_window7_224'
    ]
    
    FEATURE_DIMS = [128, 256, 512]
    
    PIXEL_LOSS_WEIGHTS = [0.0, 0.3, 0.5, 0.7, 1.0]
    
    MIXED_LOSS_DELAYS = [0, 5, 10]
    
    # Loss combinations: (use_pixel_loss, use_global_loss)
    LOSS_COMBINATIONS = [
        {'pixel': True, 'global': True, 'name': 'both'},
        {'pixel': True, 'global': False, 'name': 'pixel_only'},
        {'pixel': False, 'global': True, 'name': 'global_only'},
    ]

    @classmethod
    def get_default(cls):
        """Return the default complete ablation configuration."""
        return {
            'base_config': copy.deepcopy(cls.BASE_CONFIG),
            'models': list(cls.MODELS),
            'feature_dims': list(cls.FEATURE_DIMS),
            'pixel_loss_weights': list(cls.PIXEL_LOSS_WEIGHTS),
            'mixed_loss_delays': list(cls.MIXED_LOSS_DELAYS),
            'loss_combinations': [dict(c) for c in cls.LOSS_COMBINATIONS],
        }

    @staticmethod
    def _deep_update(base, updates):
        """Recursively merge dictionaries while preserving default keys."""
        for key, value in updates.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                AblationConfig._deep_update(base[key], value)
            else:
                base[key] = value
        return base

    @classmethod
    def load_from_json(cls, config_path):
        """Load, merge, and validate ablation configuration from JSON."""
        resolved = cls.get_default()
        if config_path is None:
            cls.validate(resolved)
            return resolved

        with open(config_path, 'r') as f:
            loaded = json.load(f)

        if not isinstance(loaded, dict):
            raise ValueError('Ablation config JSON must contain a top-level object.')

        cls._deep_update(resolved, loaded)
        cls.validate(resolved)
        return resolved

    @staticmethod
    def validate(config):
        """Validate configuration structure and convert list-like tuple fields."""
        required_top_keys = [
            'base_config',
            'models',
            'feature_dims',
            'pixel_loss_weights',
            'mixed_loss_delays',
            'loss_combinations',
        ]

        for key in required_top_keys:
            if key not in config:
                raise ValueError(f"Missing required key in ablation config: {key}")

        base_cfg = config['base_config']
        required_base_keys = [
            'data_root',
            'train_img_size',
            'num_ugv_views',
            'grid_size',
            'grid_resolution',
            'batch_size',
            'learning_rate',
            'epochs',
            'device',
            'val_split_ratio',
            'use_depth',
            'depth_range',
            'ground_tile_size',
            'consecutive_frames',
            'pretrained_backbones',
        ]

        for key in required_base_keys:
            if key not in base_cfg:
                raise ValueError(f"Missing required key in base_config: {key}")

        # JSON stores arrays as lists; convert tuple-like fields used in the pipeline.
        base_cfg['train_img_size'] = tuple(base_cfg['train_img_size'])
        base_cfg['grid_size'] = tuple(base_cfg['grid_size'])
        base_cfg['depth_range'] = tuple(base_cfg['depth_range'])

        if not config['models']:
            raise ValueError('models cannot be empty in ablation config.')
        if not config['feature_dims']:
            raise ValueError('feature_dims cannot be empty in ablation config.')
        if not config['loss_combinations']:
            raise ValueError('loss_combinations cannot be empty in ablation config.')

        for comb in config['loss_combinations']:
            if not all(k in comb for k in ['pixel', 'global', 'name']):
                raise ValueError('Each item in loss_combinations must contain pixel, global, and name.')


class AblationStudy:
    """Main ablation study class"""
    
    def __init__(self, ablation_config, output_dir='ablation_results', num_experiments=None):
        self.ablation_config = ablation_config
        self.base_config = ablation_config['base_config']
        self.models = ablation_config['models']
        self.feature_dims = ablation_config['feature_dims']
        self.pixel_loss_weights = ablation_config['pixel_loss_weights']
        self.mixed_loss_delays = ablation_config['mixed_loss_delays']
        self.loss_combinations = ablation_config['loss_combinations']
        self.output_dir = output_dir
        self.num_experiments = num_experiments
        self.results_csv = os.path.join(output_dir, f'ablation_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
        self.config_dir = os.path.join(output_dir, 'configs')
        self.model_dir = os.path.join(output_dir, 'models')
        
        # Create directories
        os.makedirs(self.config_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Initialize CSV
        self._init_results_csv()

        # Persist resolved run configuration for full reproducibility.
        self._save_ablation_config_snapshot()
        
        # Setup data loaders (shared across all experiments)
        self.train_dataloader, self.val_dataloader = self._setup_dataloaders()

    def _save_ablation_config_snapshot(self):
        """Save merged and validated ablation config used for this run."""
        snapshot_path = os.path.join(self.config_dir, 'ablation_config_resolved.json')
        with open(snapshot_path, 'w') as f:
            json.dump(self.ablation_config, f, indent=2)
        
    def _init_results_csv(self):
        """Initialize results CSV file with headers"""
        headers = [
            'experiment_id',
            'model',
            'feature_dim',
            'pixel_loss_weight',
            'mixed_loss_delay',
            'loss_type',
            'epoch',
            'train_loss',
            'train_pixel_loss',
            'train_global_loss',
            'val_loss',
            'val_pixel_loss',
            'val_global_loss',
            'best_val_loss',
            'training_time_sec',
            'timestamp'
        ]
        
        with open(self.results_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
    
    def _setup_dataloaders(self):
        """Setup data loaders once and reuse"""
        print("Loading dataset...")
        
        image_transforms = transforms.Compose([
            transforms.Resize(self.base_config['train_img_size'], antialias=True),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        depth_transforms = transforms.Compose([
            transforms.Resize(self.base_config['train_img_size'], antialias=True),
            transforms.ConvertImageDtype(torch.float),
        ])
        
        full_dataset = VineyardDataset(
            root_dir=self.base_config['data_root'],
            config=self.base_config,
            transforms=image_transforms,
            depth_transforms=depth_transforms,
            consecutive_frames=self.base_config['consecutive_frames']
        )
        
        if len(full_dataset) == 0:
            raise ValueError("Dataset is empty. Please check the data_root path.")
        
        val_size = int(self.base_config['val_split_ratio'] * len(full_dataset))
        train_size = len(full_dataset) - val_size
        
        print(f"Dataset size: {len(full_dataset)}. Splitting into {train_size} training and {val_size} validation samples.")
        
        generator = torch.Generator().manual_seed(42)  # Fixed seed for reproducibility
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)
        
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.base_config['batch_size'],
            shuffle=True,
            num_workers=8
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=self.base_config['batch_size'],
            shuffle=False,
            num_workers=8
        )
        
        return train_dataloader, val_dataloader
    
    def generate_experiments(self):
        """Generate all experiment configurations"""
        experiments = []
        
        for exp_id, (model, feat_dim, loss_comb, pixel_weight, loss_delay) in enumerate(
            itertools.product(
                self.models,
                self.feature_dims,
                self.loss_combinations,
                self.pixel_loss_weights,
                self.mixed_loss_delays,
            )
        ):
            # Skip invalid combinations
            if not loss_comb['pixel'] and pixel_weight != 0.0:
                continue
            if loss_comb['name'] == 'pixel_only' and loss_delay > 0:
                continue
            if loss_comb['name'] == 'global_only' and loss_delay > 0:
                continue
            
            # Remove '+Neck' suffix to get base model name
            base_model_name = model.replace('+Neck', '')
            
            config = self.base_config.copy()
            config.update({
                'experiment_id': exp_id,
                'model_name': base_model_name,
                'feature_dim': feat_dim,
                'pixel_loss_weight': pixel_weight if loss_comb['pixel'] else 0.0,
                'mixed_loss_delay': loss_delay if loss_comb['pixel'] and loss_comb['global'] else 0,
                'loss_type': loss_comb['name'],
                'use_pixel_loss': loss_comb['pixel'],
                'use_global_loss': loss_comb['global'],
                'output_model_path': os.path.join(
                    self.model_dir,
                    f"exp_{exp_id:04d}_{model}_{feat_dim}_{loss_comb['name']}"
                ),
            })
            
            experiments.append(config)
        
        if self.num_experiments is not None:
            experiments = experiments[:self.num_experiments]
        
        return experiments
    
    def train_experiment(self, exp_id, config):
        """Train a single experiment"""
        print(f"\n{'='*80}")
        print(f"Experiment {exp_id}: {config['model_name']}, "
              f"feat_dim={config['feature_dim']}, "
              f"loss={config['loss_type']}, "
              f"pixel_weight={config['pixel_loss_weight']}, "
              f"delay={config['mixed_loss_delay']}")
        print(f"{'='*80}")
        
        start_time = datetime.now()
        
        # Create and load model
        try:
            model = SnapViT(config).to(config['device'])
        except Exception as e:
            print(f"Error creating model: {e}")
            return None
        
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
        best_val_loss = float('inf')
        
        # Create output directory
        os.makedirs(config['output_model_path'], exist_ok=True)
        
        # Save config
        config_path = os.path.join(config['output_model_path'], 'config.json')
        config_copy = config.copy()
        config_copy['device'] = str(config_copy['device'])  # Convert device to string for JSON
        with open(config_path, 'w') as f:
            json.dump(config_copy, f, indent=2)
        
        # Training loop
        for epoch in range(config['epochs']):
            # Training phase
            model.train()
            total_train_loss = 0.0
            total_pixel_loss = 0.0
            total_global_loss = 0.0
            
            train_pbar = tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}/{config['epochs']} [Train]", leave=False)
            for batch in train_pbar:
                optimizer.zero_grad()
                
                try:
                    uav_data = {k: v.to(config['device']) for k, v in batch['uav_data'].items()}
                    ugv_data = {k: v.to(config['device']) for k, v in batch['ugv_data'].items()}
                    
                    ground_bev, overhead_bev, ground_validity = model(ugv_data, uav_data)
                    overhead_bev_resized = F.interpolate(
                        overhead_bev,
                        size=ground_bev.shape[2:],
                        mode='bilinear',
                        align_corners=False
                    )
                    
                    # Compute losses based on configuration
                    pixel_loss = 0.0
                    global_loss = 0.0
                    
                    if config['use_pixel_loss']:
                        pixel_loss = masked_info_nce_loss(
                            ground_bev, overhead_bev_resized, ground_validity, model.temperature
                        )
                    
                    if config['use_global_loss']:
                        global_loss = symmetric_info_nce_loss_masked(
                            ground_bev, overhead_bev_resized, ground_validity, model.temperature
                        )
                    
                    # Combine losses
                    if epoch < config['mixed_loss_delay']:
                        loss = global_loss if config['use_global_loss'] else pixel_loss
                    else:
                        if config['use_pixel_loss'] and config['use_global_loss']:
                            loss = (config['pixel_loss_weight'] * pixel_loss +
                                   (1 - config['pixel_loss_weight']) * global_loss)
                        elif config['use_pixel_loss']:
                            loss = pixel_loss
                        else:
                            loss = global_loss
                    
                    loss.backward()
                    optimizer.step()
                    
                    total_train_loss += loss.item()
                    total_pixel_loss += float(pixel_loss) if config['use_pixel_loss'] else 0.0
                    total_global_loss += float(global_loss) if config['use_global_loss'] else 0.0
                    
                    train_pbar.set_postfix({'loss': loss.item()})
                    
                except Exception as e:
                    print(f"Error during training batch: {e}")
                    return None
            
            avg_train_loss = total_train_loss / len(self.train_dataloader)
            avg_pixel_loss = total_pixel_loss / len(self.train_dataloader)
            avg_global_loss = total_global_loss / len(self.train_dataloader)
            
            # Validation phase
            model.eval()
            total_val_loss = 0.0
            total_val_pixel_loss = 0.0
            total_val_global_loss = 0.0
            
            val_pbar = tqdm(self.val_dataloader, desc=f"Epoch {epoch+1}/{config['epochs']} [Val]", leave=False)
            with torch.no_grad():
                for batch in val_pbar:
                    try:
                        uav_data = {k: v.to(config['device']) for k, v in batch['uav_data'].items()}
                        ugv_data = {k: v.to(config['device']) for k, v in batch['ugv_data'].items()}
                        
                        ground_bev, overhead_bev, ground_validity = model(ugv_data, uav_data)
                        overhead_bev_resized = F.interpolate(
                            overhead_bev,
                            size=ground_bev.shape[2:],
                            mode='bilinear',
                            align_corners=False
                        )
                        
                        pixel_loss = 0.0
                        global_loss = 0.0
                        
                        if config['use_pixel_loss']:
                            pixel_loss = masked_info_nce_loss(
                                ground_bev, overhead_bev_resized, ground_validity, model.temperature
                            )
                        
                        if config['use_global_loss']:
                            global_loss = symmetric_info_nce_loss_masked(
                                ground_bev, overhead_bev_resized, ground_validity, model.temperature
                            )
                        
                        if epoch < config['mixed_loss_delay']:
                            loss = global_loss if config['use_global_loss'] else pixel_loss
                        else:
                            if config['use_pixel_loss'] and config['use_global_loss']:
                                loss = (config['pixel_loss_weight'] * pixel_loss +
                                       (1 - config['pixel_loss_weight']) * global_loss)
                            elif config['use_pixel_loss']:
                                loss = pixel_loss
                            else:
                                loss = global_loss
                        
                        total_val_loss += loss.item()
                        total_val_pixel_loss += float(pixel_loss) if config['use_pixel_loss'] else 0.0
                        total_val_global_loss += float(global_loss) if config['use_global_loss'] else 0.0
                        
                        val_pbar.set_postfix({'loss': loss.item()})
                        
                    except Exception as e:
                        print(f"Error during validation batch: {e}")
                        return None
            
            avg_val_loss = total_val_loss / len(self.val_dataloader)
            avg_val_pixel_loss = total_val_pixel_loss / len(self.val_dataloader)
            avg_val_global_loss = total_val_global_loss / len(self.val_dataloader)
            
            # Track best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                # Save best model
                best_model_path = os.path.join(config['output_model_path'], 'best_model.pth')
                torch.save(model.state_dict(), best_model_path)
            
            # Save results to CSV
            self._save_epoch_result(
                exp_id, config, epoch,
                avg_train_loss, avg_pixel_loss, avg_global_loss,
                avg_val_loss, avg_val_pixel_loss, avg_val_global_loss,
                best_val_loss,
                (datetime.now() - start_time).total_seconds()
            )
        
        # Final model save
        final_model_path = os.path.join(config['output_model_path'], 'final_model.pth')
        torch.save(model.state_dict(), final_model_path)
        
        return {
            'exp_id': exp_id,
            'best_val_loss': best_val_loss,
            'training_time': (datetime.now() - start_time).total_seconds()
        }
    
    def _save_epoch_result(self, exp_id, config, epoch, train_loss, pixel_loss, global_loss,
                          val_loss, val_pixel_loss, val_global_loss, best_val_loss, training_time):
        """Save epoch results to CSV"""
        with open(self.results_csv, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'experiment_id', 'model', 'feature_dim', 'pixel_loss_weight', 'mixed_loss_delay',
                'loss_type', 'epoch', 'train_loss', 'train_pixel_loss', 'train_global_loss',
                'val_loss', 'val_pixel_loss', 'val_global_loss', 'best_val_loss', 'training_time_sec',
                'timestamp'
            ])
            writer.writerow({
                'experiment_id': exp_id,
                'model': config['model_name'],
                'feature_dim': config['feature_dim'],
                'pixel_loss_weight': config['pixel_loss_weight'],
                'mixed_loss_delay': config['mixed_loss_delay'],
                'loss_type': config['loss_type'],
                'epoch': epoch + 1,
                'train_loss': f"{train_loss:.6f}",
                'train_pixel_loss': f"{pixel_loss:.6f}",
                'train_global_loss': f"{global_loss:.6f}",
                'val_loss': f"{val_loss:.6f}",
                'val_pixel_loss': f"{val_pixel_loss:.6f}",
                'val_global_loss': f"{val_global_loss:.6f}",
                'best_val_loss': f"{best_val_loss:.6f}",
                'training_time_sec': f"{training_time:.2f}",
                'timestamp': datetime.now().isoformat()
            })
    
    def run(self):
        """Run the complete ablation study"""
        print("Generating experiment configurations...")
        experiments = self.generate_experiments()
        
        print(f"Total experiments to run: {len(experiments)}")
        print(f"Results will be saved to: {self.results_csv}")
        
        results_summary = []
        
        for i, exp_config in enumerate(experiments):
            print(f"\n[{i+1}/{len(experiments)}] Running experiment {exp_config['experiment_id']}...")
            
            result = self.train_experiment(exp_config['experiment_id'], exp_config)
            
            if result is not None:
                results_summary.append(result)
            
            # Optional: early stopping if memory issues occur
            torch.cuda.empty_cache()
        
        # Save summary
        self._save_summary(results_summary)
        
        print(f"\nAblation study complete!")
        print(f"Results saved to: {self.results_csv}")
        return results_summary
    
    def _save_summary(self, results):
        """Save summary of all experiments"""
        summary_path = os.path.join(self.output_dir, 'summary.json')
        summary = {
            'total_experiments': len(results),
            'successful_experiments': len([r for r in results if r is not None]),
            'results': results,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Summary saved to: {summary_path}")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run SnapViT ablation study')
    parser.add_argument('--config', default='cfg/ablation_config.json', help='Path to ablation JSON configuration file')
    parser.add_argument('--output-dir', default='ablation_results', help='Output directory for results')
    parser.add_argument('--num-experiments', type=int, default=None, help='Limit number of experiments (for testing)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs per experiment')
    
    args = parser.parse_args()

    ablation_config = AblationConfig.load_from_json(args.config)

    # CLI epoch override has precedence over JSON.
    if args.epochs != 50:
        ablation_config['base_config']['epochs'] = args.epochs
    
    ablation = AblationStudy(
        ablation_config=ablation_config,
        output_dir=args.output_dir,
        num_experiments=args.num_experiments
    )
    
    ablation.run()


if __name__ == '__main__':
    main()
