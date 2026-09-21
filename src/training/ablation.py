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
from datetime import datetime
import itertools
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.training.train_loop import build_dataloaders, train_loop


class AblationConfig:
    """Configuration for ablation study"""
    
    # Base configuration (shared across all experiments)
    BASE_CONFIG = {
        'data_root': '/media/hdd/ale_navone/GAIA/tempovine/dataset_tempovine_new',
        'train_img_size': (224, 224),
        'ground_fusion_mode': 'mlp',
        'ground_fusion_variant': 'height_aware_mean_max',
        'use_height_positional_encoding': True,
        'num_ugv_views': 8,
        'grid_size': (34, 34, 8),
        'grid_resolution': 0.3,
        'batch_size': 8,
        'num_workers': 8,
        'pin_memory': True,
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

        base_cfg.setdefault('ground_fusion_mode', 'mlp')
        base_cfg.setdefault('ground_fusion_variant', 'height_aware_mean_max')
        base_cfg.setdefault('use_height_positional_encoding', True)

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
        self.train_dataloader, self.val_dataloader = build_dataloaders(self.base_config, seed=42)

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
        try:
            train_result = train_loop(
                config,
                self.train_dataloader,
                self.val_dataloader,
                save_history=False,
                writer_log_dir=os.path.join(self.output_dir, 'tensorboard', f'exp_{exp_id:04d}')
            )
        except Exception as e:
            print(f"Error while training experiment {exp_id}: {e}")
            return None

        training_time = (datetime.now() - start_time).total_seconds()

        for record in train_result['history']:
            self._save_epoch_result(exp_id, config, record)

        return {
            'exp_id': exp_id,
            'best_val_loss': train_result['best_val_loss'],
            'training_time': training_time
        }
    
    def _save_epoch_result(self, exp_id, config, record):
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
                'epoch': record['epoch'],
                'train_loss': f"{record['train_loss']:.6f}",
                'train_pixel_loss': f"{record['train_pixel_loss']:.6f}",
                'train_global_loss': f"{record['train_global_loss']:.6f}",
                'val_loss': f"{record['val_loss']:.6f}",
                'val_pixel_loss': f"{record['val_pixel_loss']:.6f}",
                'val_global_loss': f"{record['val_global_loss']:.6f}",
                'best_val_loss': f"{record['best_val_loss']:.6f}",
                'training_time_sec': f"{record['training_time_sec']:.2f}",
                'timestamp': record['timestamp']
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
