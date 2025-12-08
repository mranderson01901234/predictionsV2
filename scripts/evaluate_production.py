#!/usr/bin/env python3
"""
Comprehensive Evaluation Script for NFL Prediction Models

Generates detailed evaluation reports including:
- Accuracy by season
- Calibration curves
- Feature importance
- ROI simulation
- Confidence tier analysis

Usage:
    python scripts/evaluate_production.py --model-path artifacts/models/nfl_gbm/model.pkl --test-data data/nfl/features/baseline.parquet
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.base import BaseModel
from models.calibration import compute_calibration_metrics
from eval.metrics import accuracy, brier_score, log_loss, calibration_buckets

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Comprehensive model evaluation."""
    
    def __init__(self, model: BaseModel, output_dir: Path):
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        df: pd.DataFrame,
        set_name: str = "test",
    ) -> Dict:
        """
        Comprehensive evaluation.
        
        Returns:
            Dictionary with all metrics
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating on {set_name} set")
        logger.info(f"{'='*60}")
        
        # Predictions
        y_pred_proba = self.model.predict_proba(X)
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        # Core metrics
        metrics = {
            'accuracy': accuracy(y.values, y_pred),
            'brier_score': brier_score(y.values, y_pred_proba),
            'log_loss': log_loss(y.values, y_pred_proba),
        }
        
        # Calibration metrics
        cal_metrics = compute_calibration_metrics(y.values, y_pred_proba)
        metrics.update({
            'ece': cal_metrics['ece'],
            'mce': cal_metrics['mce'],
        })
        
        # Per-season breakdown
        if 'season' in df.columns:
            metrics['by_season'] = self._evaluate_by_season(df, y.values, y_pred, y_pred_proba)
        
        # Confidence tiers
        metrics['confidence_tiers'] = self._evaluate_confidence_tiers(y.values, y_pred_proba)
        
        # ROI simulation
        metrics['roi'] = self._simulate_roi(y.values, y_pred_proba)
        
        # Feature importance (if available)
        if hasattr(self.model, 'get_feature_importance'):
            metrics['feature_importance'] = self._get_feature_importance(X)
        
        # Log summary
        logger.info(f"\nMetrics Summary:")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  Brier Score: {metrics['brier_score']:.4f}")
        logger.info(f"  Log Loss: {metrics['log_loss']:.4f}")
        logger.info(f"  ECE: {metrics['ece']:.4f}")
        logger.info(f"  MCE: {metrics['mce']:.4f}")
        
        return metrics
    
    def _evaluate_by_season(
        self,
        df: pd.DataFrame,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray,
    ) -> pd.DataFrame:
        """Evaluate performance by season."""
        results = []
        
        for season in sorted(df['season'].unique()):
            mask = df['season'] == season
            season_y_true = y_true[mask]
            season_y_pred = y_pred[mask]
            season_y_pred_proba = y_pred_proba[mask]
            
            if len(season_y_true) == 0:
                continue
            
            results.append({
                'season': season,
                'n_games': len(season_y_true),
                'accuracy': accuracy(season_y_true, season_y_pred),
                'brier_score': brier_score(season_y_true, season_y_pred_proba),
            })
        
        return pd.DataFrame(results)
    
    def _evaluate_confidence_tiers(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
    ) -> pd.DataFrame:
        """Evaluate accuracy by confidence tier."""
        tiers = [
            (0.50, 0.55, "50-55%"),
            (0.55, 0.60, "55-60%"),
            (0.60, 0.65, "60-65%"),
            (0.65, 0.70, "65-70%"),
            (0.70, 0.75, "70-75%"),
            (0.75, 1.00, "75%+"),
        ]
        
        results = []
        for low, high, name in tiers:
            # Both sides of 0.5
            mask_high = (y_pred_proba >= low) & (y_pred_proba < high)
            mask_low = (y_pred_proba > (1 - high)) & (y_pred_proba <= (1 - low))
            mask = mask_high | mask_low
            
            if mask.sum() == 0:
                continue
            
            tier_y_true = y_true[mask]
            tier_y_pred = (y_pred_proba[mask] >= 0.5).astype(int)
            
            results.append({
                'tier': name,
                'n_games': mask.sum(),
                'accuracy': accuracy(tier_y_true, tier_y_pred),
                'mean_confidence': np.abs(y_pred_proba[mask] - 0.5).mean() + 0.5,
            })
        
        return pd.DataFrame(results)
    
    def _simulate_roi(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        edge_thresholds: List[float] = [0.03, 0.05, 0.07],
    ) -> Dict:
        """Simulate ROI with different edge thresholds."""
        results = {}
        
        for threshold in edge_thresholds:
            edge = np.abs(y_pred_proba - 0.5)
            mask = edge >= threshold
            
            if mask.sum() == 0:
                results[f'roi_{threshold}'] = 0.0
                continue
            
            tier_y_true = y_true[mask]
            tier_y_pred = (y_pred_proba[mask] >= 0.5).astype(int)
            
            wins = (tier_y_pred == tier_y_true).sum()
            losses = len(tier_y_true) - wins
            
            # Assuming -110 odds
            profit = wins * 100 - losses * 110
            wagered = len(tier_y_true) * 110
            roi = profit / wagered if wagered > 0 else 0.0
            
            results[f'roi_{threshold}'] = roi
            results[f'n_bets_{threshold}'] = mask.sum()
        
        return results
    
    def _get_feature_importance(self, X: pd.DataFrame) -> pd.DataFrame:
        """Get feature importance."""
        importances = self.model.get_feature_importance()
        
        return pd.DataFrame({
            'feature': X.columns,
            'importance': importances,
        }).sort_values('importance', ascending=False)
    
    def plot_calibration_curve(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        set_name: str = "test",
    ):
        """Plot calibration curve."""
        from sklearn.calibration import calibration_curve
        
        prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=10)
        
        plt.figure(figsize=(8, 6))
        plt.plot(prob_pred, prob_true, marker='o', label='Model')
        plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title(f'Calibration Curve ({set_name})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        output_path = self.output_dir / f"calibration_curve_{set_name}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved calibration curve to {output_path}")
    
    def generate_report(
        self,
        metrics: Dict,
        output_path: Optional[Path] = None,
    ):
        """Generate markdown evaluation report."""
        if output_path is None:
            output_path = self.output_dir / "evaluation_report.md"
        
        with open(output_path, 'w') as f:
            f.write("# Model Evaluation Report\n\n")
            f.write("## Core Metrics\n\n")
            f.write(f"- **Accuracy**: {metrics['accuracy']:.4f}\n")
            f.write(f"- **Brier Score**: {metrics['brier_score']:.4f}\n")
            f.write(f"- **Log Loss**: {metrics['log_loss']:.4f}\n")
            f.write(f"- **ECE**: {metrics['ece']:.4f}\n")
            f.write(f"- **MCE**: {metrics['mce']:.4f}\n\n")
            
            if 'by_season' in metrics:
                f.write("## Performance by Season\n\n")
                f.write(metrics['by_season'].to_markdown(index=False))
                f.write("\n\n")
            
            if 'confidence_tiers' in metrics:
                f.write("## Accuracy by Confidence Tier\n\n")
                f.write(metrics['confidence_tiers'].to_markdown(index=False))
                f.write("\n\n")
            
            if 'roi' in metrics:
                f.write("## ROI Simulation\n\n")
                for key, value in metrics['roi'].items():
                    f.write(f"- **{key}**: {value:.2%}\n")
                f.write("\n")
            
            if 'feature_importance' in metrics:
                f.write("## Top 20 Features by Importance\n\n")
                f.write(metrics['feature_importance'].head(20).to_markdown(index=False))
                f.write("\n")
        
        logger.info(f"Saved evaluation report to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive model evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        '--model-path',
        type=str,
        required=True,
        help='Path to saved model file'
    )
    parser.add_argument(
        '--test-data',
        type=str,
        required=True,
        help='Path to test data parquet file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/evaluation/',
        help='Output directory for reports'
    )
    parser.add_argument(
        '--set-name',
        type=str,
        default='test',
        help='Name of the evaluation set'
    )
    
    args = parser.parse_args()
    
    # Load model
    logger.info(f"Loading model from {args.model_path}")
    model = BaseModel.load(Path(args.model_path))
    
    # Load test data
    logger.info(f"Loading test data from {args.test_data}")
    df = pd.read_parquet(args.test_data)
    
    # Prepare features
    exclude_cols = [
        'game_id', 'season', 'week', 'date', 'home_team', 'away_team',
        'home_score', 'away_score', 'home_win', 'close_spread', 'close_total',
        'open_spread', 'open_total',
    ]
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    X = df[feature_cols].fillna(0)
    y = df['home_win'] if 'home_win' in df.columns else (df['home_score'] > df['away_score']).astype(int)
    
    # Evaluate
    evaluator = ModelEvaluator(model, Path(args.output_dir))
    metrics = evaluator.evaluate(X, y, df, set_name=args.set_name)
    
    # Generate plots
    evaluator.plot_calibration_curve(y.values, model.predict_proba(X), set_name=args.set_name)
    
    # Generate report
    evaluator.generate_report(metrics)
    
    logger.info(f"\nEvaluation complete. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()

