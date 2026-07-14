"""
Analysis script for ablation study results
Generates summaries and visualizations of experiment outcomes
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
import json
from datetime import datetime


class AblationAnalyzer:
    """Analyze ablation study results"""
    
    def __init__(self, results_csv):
        """
        Initialize analyzer with results CSV
        
        Args:
            results_csv: Path to ablation results CSV file
        """
        if not os.path.exists(results_csv):
            raise FileNotFoundError(f"Results file not found: {results_csv}")
        
        self.df = pd.read_csv(results_csv)
        self.output_dir = os.path.dirname(results_csv)
        
        # Convert string columns to numeric
        numeric_cols = [
            'train_loss', 'train_pixel_loss', 'train_global_loss',
            'val_loss', 'val_pixel_loss', 'val_global_loss',
            'best_val_loss', 'training_time_sec'
        ]
        for col in numeric_cols:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
    
    def summary_by_model(self):
        """Summarize results by model architecture"""
        print("\n" + "="*80)
        print("SUMMARY BY MODEL ARCHITECTURE")
        print("="*80)
        
        summary = self.df.groupby('model').agg({
            'val_loss': ['min', 'mean', 'std'],
            'training_time_sec': 'mean',
            'experiment_id': 'count'
        }).round(6)
        
        summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
        print(summary)
        
        return summary
    
    def summary_by_feature_dim(self):
        """Summarize results by feature dimension"""
        print("\n" + "="*80)
        print("SUMMARY BY FEATURE DIMENSION")
        print("="*80)
        
        summary = self.df.groupby('feature_dim').agg({
            'val_loss': ['min', 'mean', 'std'],
            'training_time_sec': 'mean',
            'experiment_id': 'count'
        }).round(6)
        
        summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
        print(summary)
        
        return summary
    
    def summary_by_loss_type(self):
        """Summarize results by loss type"""
        print("\n" + "="*80)
        print("SUMMARY BY LOSS TYPE")
        print("="*80)
        
        summary = self.df.groupby('loss_type').agg({
            'val_loss': ['min', 'mean', 'std'],
            'pixel_loss_weight': lambda x: f"{x.iloc[0]:.1f}",
            'training_time_sec': 'mean',
            'experiment_id': 'count'
        }).round(6)
        
        print(summary)
        
        return summary
    
    def summary_by_pixel_loss_weight(self):
        """Summarize results by pixel loss weight"""
        print("\n" + "="*80)
        print("SUMMARY BY PIXEL LOSS WEIGHT")
        print("="*80)
        
        # Only for mixed loss experiments
        mixed = self.df[self.df['loss_type'] == 'both'].copy()
        
        if len(mixed) == 0:
            print("No mixed loss experiments found")
            return None
        
        summary = mixed.groupby('pixel_loss_weight').agg({
            'val_loss': ['min', 'mean', 'std'],
            'training_time_sec': 'mean',
            'experiment_id': 'count'
        }).round(6)
        
        summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
        print(summary)
        
        return summary
    
    def summary_by_mixed_loss_delay(self):
        """Summarize results by mixed loss delay"""
        print("\n" + "="*80)
        print("SUMMARY BY MIXED LOSS DELAY")
        print("="*80)
        
        # Only for experiments with mixed loss
        mixed = self.df[self.df['loss_type'] == 'both'].copy()
        
        if len(mixed) == 0:
            print("No mixed loss experiments found")
            return None
        
        summary = mixed.groupby('mixed_loss_delay').agg({
            'val_loss': ['min', 'mean', 'std'],
            'training_time_sec': 'mean',
            'experiment_id': 'count'
        }).round(6)
        
        summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
        print(summary)
        
        return summary
    
    def best_configurations(self, top_n=20):
        """Find best performing configurations"""
        print("\n" + "="*80)
        print(f"TOP {top_n} BEST CONFIGURATIONS")
        print("="*80)
        
        # Get the last epoch for each experiment to avoid duplicates
        best_per_exp = self.df.sort_values('epoch').drop_duplicates(
            'experiment_id', keep='last'
        )
        
        # Sort by validation loss
        top = best_per_exp.nsmallest(top_n, 'val_loss')[
            ['experiment_id', 'model', 'feature_dim', 'pixel_loss_weight',
             'mixed_loss_delay', 'loss_type', 'epoch', 'val_loss',
             'best_val_loss', 'training_time_sec']
        ]
        
        print(top.to_string(index=False))
        
        return top
    
    def worst_configurations(self, top_n=10):
        """Find worst performing configurations"""
        print("\n" + "="*80)
        print(f"TOP {top_n} WORST CONFIGURATIONS")
        print("="*80)
        
        # Get the last epoch for each experiment
        worst_per_exp = self.df.sort_values('epoch').drop_duplicates(
            'experiment_id', keep='last'
        )
        
        # Sort by validation loss (highest = worst)
        worst = worst_per_exp.nlargest(top_n, 'val_loss')[
            ['experiment_id', 'model', 'feature_dim', 'pixel_loss_weight',
             'mixed_loss_delay', 'loss_type', 'epoch', 'val_loss']
        ]
        
        print(worst.to_string(index=False))
        
        return worst
    
    def model_feature_dim_matrix(self):
        """Create best validation loss matrix for model vs feature dimension"""
        print("\n" + "="*80)
        print("BEST VALIDATION LOSS: MODEL vs FEATURE DIMENSION")
        print("="*80)
        
        # Get last epoch per experiment
        last_epoch = self.df.sort_values('epoch').drop_duplicates(
            'experiment_id', keep='last'
        )
        
        # Pivot table
        pivot = last_epoch.pivot_table(
            values='val_loss',
            index='model',
            columns='feature_dim',
            aggfunc='min'
        )
        
        print(pivot.round(6))
        
        return pivot
    
    def loss_comparison(self):
        """Compare different loss functions"""
        print("\n" + "="*80)
        print("LOSS FUNCTION COMPARISON")
        print("="*80)
        
        # Get last epoch per experiment
        last_epoch = self.df.sort_values('epoch').drop_duplicates(
            'experiment_id', keep='last'
        )
        
        print("\nBest per loss type:")
        for loss_type in ['global_only', 'pixel_only', 'both']:
            subset = last_epoch[last_epoch['loss_type'] == loss_type]
            if len(subset) > 0:
                best = subset['val_loss'].min()
                mean = subset['val_loss'].mean()
                std = subset['val_loss'].std()
                count = len(subset)
                print(f"  {loss_type:15s}: best={best:.6f}, mean={mean:.6f}, std={std:.6f}, count={count}")
    
    def convergence_analysis(self, top_n=5):
        """Analyze convergence of top configurations"""
        print("\n" + "="*80)
        print(f"CONVERGENCE ANALYSIS: TOP {top_n} CONFIGURATIONS")
        print("="*80)
        
        # Get last epoch per experiment
        last_epoch = self.df.sort_values('epoch').drop_duplicates(
            'experiment_id', keep='last'
        )
        
        # Get top experiments
        top_exps = last_epoch.nsmallest(top_n, 'val_loss')['experiment_id'].values
        
        for i, exp_id in enumerate(top_exps):
            exp_data = self.df[self.df['experiment_id'] == exp_id].sort_values('epoch')
            config = exp_data.iloc[0]
            
            print(f"\n{i+1}. Exp {exp_id}: {config['model']} "
                  f"(dim={config['feature_dim']}, loss={config['loss_type']}, "
                  f"weight={config['pixel_loss_weight']:.1f})")
            
            if len(exp_data) > 0:
                first_val_loss = exp_data.iloc[0]['val_loss']
                last_val_loss = exp_data.iloc[-1]['val_loss']
                improvement = ((first_val_loss - last_val_loss) / first_val_loss * 100)
                
                print(f"   Epochs: 1-{len(exp_data)}")
                print(f"   Initial val_loss: {first_val_loss:.6f}")
                print(f"   Final val_loss:   {last_val_loss:.6f}")
                print(f"   Improvement:      {improvement:.2f}%")
    
    def export_summary(self, filename='ablation_summary.txt'):
        """Export comprehensive summary to text file"""
        output_path = os.path.join(self.output_dir, filename)
        
        with open(output_path, 'w') as f:
            f.write(f"Ablation Study Summary\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write(f"Results file: {self.df.shape[0]} records from experiments\n")
            f.write(f"Unique experiments: {self.df['experiment_id'].nunique()}\n")
            f.write(f"Epochs per experiment: {self.df['epoch'].max()}\n")
            f.write("\n" + "="*80 + "\n\n")
            
            # Model summary
            f.write("BEST BY MODEL:\n")
            f.write("-" * 80 + "\n")
            last_epoch = self.df.sort_values('epoch').drop_duplicates(
                'experiment_id', keep='last'
            )
            by_model = last_epoch.groupby('model')['val_loss'].min().sort_values()
            for model, loss in by_model.items():
                f.write(f"  {model:40s}: {loss:.6f}\n")
            
            f.write("\n" + "="*80 + "\n\n")
            f.write("TOP 10 CONFIGURATIONS:\n")
            f.write("-" * 80 + "\n")
            top_10 = last_epoch.nsmallest(10, 'val_loss')
            for idx, (_, row) in enumerate(top_10.iterrows(), 1):
                f.write(f"{idx:2d}. Exp {row['experiment_id']:04d}: "
                       f"{row['model']:30s} "
                       f"dim={row['feature_dim']:3.0f} "
                       f"w={row['pixel_loss_weight']:.1f} "
                       f"loss={row['val_loss']:.6f}\n")
        
        print(f"\nSummary exported to: {output_path}")
        return output_path
    
    def run_all_analysis(self):
        """Run all analysis"""
        print(f"\nAnalyzing {len(self.df)} records from {self.df['experiment_id'].nunique()} experiments")
        
        self.summary_by_model()
        self.summary_by_feature_dim()
        self.summary_by_loss_type()
        self.summary_by_pixel_loss_weight()
        self.summary_by_mixed_loss_delay()
        self.best_configurations(top_n=20)
        self.worst_configurations(top_n=10)
        self.model_feature_dim_matrix()
        self.loss_comparison()
        self.convergence_analysis(top_n=5)
        self.export_summary()


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze ablation study results')
    parser.add_argument('results_csv', help='Path to ablation results CSV file')
    parser.add_argument('--top', type=int, default=20, help='Number of top configurations to show')
    parser.add_argument('--model', help='Filter by model (optional)')
    parser.add_argument('--export', action='store_true', help='Export summary to text file')
    
    args = parser.parse_args()
    
    try:
        analyzer = AblationAnalyzer(args.results_csv)
        
        if args.model:
            print(f"\nFiltering for model: {args.model}")
            analyzer.df = analyzer.df[analyzer.df['model'].str.contains(args.model)]
        
        analyzer.run_all_analysis()
        
        if args.export:
            analyzer.export_summary()
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
