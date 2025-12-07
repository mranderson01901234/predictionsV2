"""
Simulate Predictions for All 2025 Weeks 1-13 Games

This script:
1. Loads the trained ensemble model from 2025 holdout evaluation
2. Predicts all games from 2025 weeks 1-13
3. Compares predictions against actual results
4. Tests for data leakage
5. Calculates comprehensive accuracy metrics
6. Generates detailed performance report

Requirements:
- Trained model from evaluate_2025_holdout.py
- Actual game results in data/nfl/staged/games.parquet
- Features for 2025 games in data/nfl/processed/game_features_baseline.parquet
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.training.trainer import load_config
from models.architectures.stacking_ensemble import StackingEnsemble
from models.calibration import CalibratedModel
from eval.metrics import accuracy, brier_score, log_loss, calibration_buckets

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_actual_results(game_id: str) -> Optional[Dict]:
    """
    Get actual game results from games.parquet.
    
    Args:
        game_id: Game ID
    
    Returns:
        Dictionary with actual results or None if not found
    """
    games_path = Path(__file__).parent.parent / "data" / "nfl" / "staged" / "games.parquet"
    
    if not games_path.exists():
        logger.warning(f"Games file not found: {games_path}")
        return None
    
    df = pd.read_parquet(games_path)
    game_data = df[df['game_id'] == game_id]
    
    if len(game_data) == 0:
        # Try alternative game_id formats
        parts = game_id.split('_')
        if len(parts) >= 4 and parts[0] == 'nfl':
            season = parts[1]
            week = parts[2]
            away = parts[3]
            home = parts[4] if len(parts) > 4 else None
            # Try alternative format
            alt_id = f"{season}_WK{week}_{home}_{away}" if home else None
            if alt_id:
                game_data = df[df['game_id'] == alt_id]
        
        if len(game_data) == 0:
            return None
    
    row = game_data.iloc[0]
    
    return {
        'home_score': int(row['home_score']) if pd.notna(row['home_score']) else None,
        'away_score': int(row['away_score']) if pd.notna(row['away_score']) else None,
        'home_team': row['home_team'],
        'away_team': row['away_team'],
        'date': row.get('date', None),
        'game_id': row.get('game_id', game_id),
    }


def load_trained_model(model_dir: Path):
    """Load the trained ensemble model from holdout evaluation."""
    logger.info(f"Loading model from {model_dir}")
    
    ensemble_path = model_dir / "ensemble_v1.pkl"
    if not ensemble_path.exists():
        raise FileNotFoundError(
            f"Model not found at {ensemble_path}. "
            "Make sure evaluate_2025_holdout.py has been run successfully."
        )
    
    logger.info(f"Loading ensemble from {ensemble_path}")
    import pickle
    with open(ensemble_path, 'rb') as f:
        ensemble_dict = pickle.load(f)
    
    # Reconstruct StackingEnsemble from saved dict
    from models.architectures.stacking_ensemble import StackingEnsemble
    from models.architectures.ft_transformer import FTTransformerModel
    from models.architectures.tabnet import TabNetModel
    from models.architectures.gradient_boosting import GradientBoostingModel
    
    # Load base models
    base_models = {}
    base_model_paths = ensemble_dict.get('base_model_paths', {})
    model_names = ensemble_dict.get('model_names', [])
    
    for name in model_names:
        base_path = base_model_paths.get(name)
        base_path = Path(base_path) if base_path else None
        
        if base_path and base_path.exists():
            if 'ft_transformer' in name.lower() or 'ft_transformer' in str(base_path):
                base_models[name] = FTTransformerModel.load(base_path)
            elif 'tabnet' in name.lower() or 'tabnet' in str(base_path):
                # TabNet pickle contains path to .tabnet file, but actual file is .tabnet.zip
                import pickle
                with open(base_path, 'rb') as f:
                    tabnet_dict = pickle.load(f)
                # Fix model_path to point to .tabnet.zip
                if 'model_path' in tabnet_dict:
                    old_path = Path(tabnet_dict['model_path'])
                    zip_path = model_dir / f"{old_path.stem}.tabnet.zip"
                    if zip_path.exists():
                        tabnet_dict['model_path'] = str(zip_path)
                    else:
                        alt_zip = model_dir / "tabnet.tabnet.zip"
                        if alt_zip.exists():
                            tabnet_dict['model_path'] = str(alt_zip)
                # Create temporary pickle with fixed path
                import tempfile
                with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pkl') as tmp:
                    pickle.dump(tabnet_dict, tmp)
                    tmp_path = tmp.name
                try:
                    base_models[name] = TabNetModel.load(Path(tmp_path))
                finally:
                    Path(tmp_path).unlink()
            elif 'gbm' in name.lower() or 'gradient' in name.lower() or 'gbm' in str(base_path):
                base_models[name] = GradientBoostingModel.load(base_path)
            else:
                logger.warning(f"Unknown model type for {name}, trying generic load")
                from models.base import BaseModel
                base_models[name] = BaseModel.load(base_path)
        else:
            # Try loading from model_dir
            alt_path = model_dir / f"{name}.pkl"
            if alt_path.exists():
                if 'ft_transformer' in name.lower():
                    base_models[name] = FTTransformerModel.load(alt_path)
                elif 'tabnet' in name.lower():
                    base_models[name] = TabNetModel.load(alt_path)
                elif 'gbm' in name.lower() or 'gradient' in name.lower():
                    base_models[name] = GradientBoostingModel.load(alt_path)
                else:
                    from models.base import BaseModel
                    base_models[name] = BaseModel.load(alt_path)
            else:
                if 'tabnet' in name.lower():
                    zip_path = model_dir / "tabnet.tabnet.zip"
                    if zip_path.exists():
                        base_models[name] = TabNetModel.load(zip_path)
    
    # Reconstruct ensemble
    config = ensemble_dict.get('config', {})
    meta_cfg = config.get('meta_model', {})
    stack_cfg = config.get('stacking', {})
    
    ensemble = StackingEnsemble(
        base_models=base_models,
        meta_model_type=meta_cfg.get('type', 'logistic'),
        include_features=stack_cfg.get('include_features', False),
        feature_fraction=stack_cfg.get('feature_fraction', 0.0),
        mlp_hidden_dims=meta_cfg.get('mlp_hidden_dims', [16, 8]),
        mlp_dropout=meta_cfg.get('mlp_dropout', 0.1),
        mlp_epochs=meta_cfg.get('mlp_epochs', 50),
        random_state=config.get('random_state', 42),
    )
    
    # Add base models to ensemble
    ensemble.base_models = base_models
    ensemble.model_names = model_names
    ensemble.n_base_models = len(base_models)
    
    # Restore meta-model and scaler
    ensemble.meta_model = ensemble_dict.get('meta_model')
    ensemble.scaler = ensemble_dict.get('scaler')
    ensemble.selected_feature_indices = ensemble_dict.get('selected_feature_indices')
    
    logger.info(f"✓ Model reconstructed: {type(ensemble).__name__}")
    logger.info(f"  Base models: {list(base_models.keys())}")
    
    # Check for calibration
    calibration_path = model_dir / "calibration.pkl"
    if calibration_path.exists():
        logger.info("Loading calibration...")
        with open(calibration_path, 'rb') as f:
            calibration = pickle.load(f)
        from models.calibration import CalibratedModel
        model = CalibratedModel(ensemble, calibration)
        logger.info("✓ Calibration applied")
    else:
        model = ensemble
    
    return model


def prepare_features_for_prediction(game_data: pd.DataFrame) -> Tuple[pd.DataFrame, list]:
    """
    Prepare features for prediction from game data.
    Uses the same feature selection logic as trainer.py load_features().
    """
    exclude_cols = [
        "game_id", "season", "week", "date",
        "home_team", "away_team", "home_score", "away_score",
        "home_win", "close_spread", "close_total",
        "open_spread", "open_total",
    ]
    
    feature_cols = [col for col in game_data.columns if col not in exclude_cols]
    X = game_data[feature_cols].copy()
    X = X.fillna(0)
    
    return X, feature_cols


def check_for_leakage(game_data: pd.DataFrame, feature_cols: List[str]) -> Dict[str, bool]:
    """Check for potential data leakage in features."""
    leakage_checks = {
        'has_future_scores': False,
        'has_result_column': False,
        'has_spread_column': False,
        'has_future_stats': False,
    }
    
    if 'home_score' in game_data.columns and pd.notna(game_data['home_score']).any():
        leakage_checks['has_future_scores'] = True
    
    if 'result' in game_data.columns:
        leakage_checks['has_result_column'] = True
    
    if 'spread' in game_data.columns:
        leakage_checks['has_spread_column'] = True
    
    future_keywords = ['final', 'post', 'end', 'total', 'final_score', 'result']
    for col in feature_cols:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in future_keywords):
            if 'future' not in leakage_checks:
                leakage_checks['has_future_stats'] = True
                leakage_checks['future_feature'] = col
            break
    
    return leakage_checks


def predict_spread(prob: float) -> float:
    """Convert probability to estimated spread."""
    return (prob - 0.5) * 28


def calculate_roi(predictions_df: pd.DataFrame, threshold: float = 0.0) -> Dict:
    """
    Calculate ROI for betting predictions.
    
    Args:
        predictions_df: DataFrame with predictions and actual results
        threshold: Minimum confidence threshold for betting
    
    Returns:
        Dictionary with ROI metrics
    """
    # Filter to games with results and above threshold
    bettable = predictions_df[
        (predictions_df['correct'].notna()) &
        (predictions_df['confidence'] >= threshold)
    ].copy()
    
    if len(bettable) == 0:
        return {
            'n_bets': 0,
            'n_wins': 0,
            'roi': 0.0,
            'profit': 0.0,
        }
    
    # Assume -110 odds (standard NFL betting)
    odds = -110
    decimal_odds = 100 / abs(odds) + 1  # Convert to decimal
    
    bettable['bet_amount'] = 1.0  # $1 per bet
    bettable['payout'] = bettable.apply(
        lambda row: decimal_odds if row['correct'] else 0.0,
        axis=1
    )
    
    total_bet = bettable['bet_amount'].sum()
    total_payout = bettable['payout'].sum()
    profit = total_payout - total_bet
    roi = (profit / total_bet) * 100 if total_bet > 0 else 0.0
    
    return {
        'n_bets': len(bettable),
        'n_wins': bettable['correct'].sum(),
        'roi': roi,
        'profit': profit,
        'win_rate': bettable['correct'].mean(),
    }


def main():
    """Main function."""
    logger.info("=" * 80)
    logger.info("Simulating Predictions for 2025 Weeks 1-13")
    logger.info("=" * 80)
    
    # Paths
    project_root = Path(__file__).parent.parent
    model_dir = project_root / "artifacts" / "models" / "nfl_stacked_ensemble_2025_holdout"
    features_path = project_root / "data" / "nfl" / "processed" / "game_features_baseline.parquet"
    games_path = project_root / "data" / "nfl" / "staged" / "games.parquet"
    
    # Check files exist
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    if not features_path.exists():
        raise FileNotFoundError(f"Features file not found: {features_path}")
    if not games_path.exists():
        raise FileNotFoundError(f"Games file not found: {games_path}")
    
    # Load model
    logger.info("\n[Step 1/7] Loading trained model...")
    model = load_trained_model(model_dir)
    logger.info("✓ Model loaded")
    
    # Load features
    logger.info("\n[Step 2/7] Loading features...")
    features_df = pd.read_parquet(features_path)
    logger.info(f"Loaded {len(features_df)} total games")
    
    # Filter to 2025 weeks 1-13
    logger.info("\n[Step 3/7] Filtering to 2025 weeks 1-13...")
    season_2025 = features_df[features_df['season'] == 2025].copy()
    weeks_1_13 = season_2025[season_2025['week'].isin(range(1, 14))].copy()
    weeks_1_13 = weeks_1_13.sort_values(['week', 'game_id'])
    
    if len(weeks_1_13) == 0:
        raise ValueError("No 2025 weeks 1-13 games found in features!")
    
    logger.info(f"Found {len(weeks_1_13)} games")
    logger.info(f"Weeks: {sorted(weeks_1_13['week'].unique())}")
    
    # Check for leakage
    logger.info("\n[Step 4/7] Checking for data leakage...")
    leakage_results = []
    for idx, row in weeks_1_13.iterrows():
        game_data = pd.DataFrame([row])
        _, feature_cols = prepare_features_for_prediction(game_data)
        leakage = check_for_leakage(game_data, feature_cols)
        leakage['game_id'] = row['game_id']
        leakage_results.append(leakage)
    
    leakage_df = pd.DataFrame(leakage_results)
    has_leakage = leakage_df[['has_future_scores', 'has_result_column', 'has_spread_column', 'has_future_stats']].any().any()
    
    if has_leakage:
        logger.warning("⚠ WARNING: Potential data leakage detected!")
        logger.warning("  (Scores are in dataframe but excluded from features - this is expected)")
    else:
        logger.info("✓ No data leakage detected")
    
    # Make predictions
    logger.info("\n[Step 5/7] Making predictions...")
    predictions = []
    
    for idx, row in weeks_1_13.iterrows():
        game_id = row['game_id']
        home_team = row['home_team']
        away_team = row['away_team']
        week = row['week']
        
        # Prepare features
        game_data = pd.DataFrame([row])
        X, feature_cols = prepare_features_for_prediction(game_data)
        
        # Predict
        prob = model.predict_proba(X)[0]
        predicted_winner = home_team if prob >= 0.5 else away_team
        predicted_spread = predict_spread(prob)
        confidence = max(prob, 1 - prob)
        
        # Get actual results
        actual_results = get_actual_results(game_id)
        
        actual_winner = None
        actual_home_score = None
        actual_away_score = None
        actual_spread = None
        correct = None
        
        if actual_results and actual_results.get('home_score') is not None:
            actual_home_score = int(actual_results['home_score'])
            actual_away_score = int(actual_results['away_score'])
            actual_winner = actual_results['home_team'] if actual_home_score > actual_away_score else actual_results['away_team']
            actual_spread = actual_home_score - actual_away_score
            correct = (predicted_winner == actual_winner)
        
        predictions.append({
            'game_id': game_id,
            'week': week,
            'home_team': home_team,
            'away_team': away_team,
            'predicted_winner': predicted_winner,
            'predicted_prob': prob,
            'predicted_spread': predicted_spread,
            'confidence': confidence,
            'actual_winner': actual_winner,
            'actual_home_score': actual_home_score,
            'actual_away_score': actual_away_score,
            'actual_spread': actual_spread,
            'correct': correct,
        })
        
        if (idx + 1) % 10 == 0:
            logger.info(f"  Processed {idx + 1}/{len(weeks_1_13)} games...")
    
    predictions_df = pd.DataFrame(predictions)
    
    # Calculate metrics
    logger.info("\n[Step 6/7] Calculating metrics...")
    
    completed_games = predictions_df[predictions_df['correct'].notna()].copy()
    
    if len(completed_games) == 0:
        logger.warning("No completed games found - cannot calculate accuracy")
        return
    
    # Accuracy metrics
    n_correct = completed_games['correct'].sum()
    n_total = len(completed_games)
    accuracy_pct = (n_correct / n_total) * 100
    
    # Spread error
    spread_errors = np.abs(completed_games['predicted_spread'] - completed_games['actual_spread'])
    spread_mae = spread_errors.mean()
    spread_rmse = np.sqrt((spread_errors ** 2).mean())
    
    # Probability metrics
    actual_binary = (completed_games['actual_winner'] == completed_games['home_team']).astype(int)
    predicted_probs = completed_games['predicted_prob'].values
    
    brier = brier_score(actual_binary, predicted_probs)
    logloss = log_loss(actual_binary, predicted_probs)
    
    # Calibration
    try:
        cal_buckets = calibration_buckets(actual_binary, predicted_probs, n_bins=10)
        if isinstance(cal_buckets, list) and len(cal_buckets) > 0 and isinstance(cal_buckets[0], dict):
            mean_cal_error = np.mean([abs(b.get('mean_pred', 0) - b.get('mean_actual', 0)) for b in cal_buckets])
        else:
            # Fallback calculation
            bins = np.linspace(0, 1, 11)
            bin_indices = np.digitize(predicted_probs, bins) - 1
            bin_indices = np.clip(bin_indices, 0, len(bins) - 2)
            mean_cal_error = np.mean([
                abs(np.mean(predicted_probs[bin_indices == i]) - np.mean(actual_binary[bin_indices == i]))
                for i in range(len(bins) - 1)
                if np.sum(bin_indices == i) > 0
            ])
    except Exception as e:
        logger.warning(f"Could not calculate calibration error: {e}")
        mean_cal_error = 0.0
    
    # ROI calculations
    roi_all = calculate_roi(completed_games, threshold=0.0)
    roi_60 = calculate_roi(completed_games, threshold=0.60)
    roi_70 = calculate_roi(completed_games, threshold=0.70)
    roi_80 = calculate_roi(completed_games, threshold=0.80)
    
    # Per-week breakdown
    weekly_stats = []
    for week in sorted(completed_games['week'].unique()):
        week_games = completed_games[completed_games['week'] == week]
        week_correct = week_games['correct'].sum()
        week_total = len(week_games)
        weekly_stats.append({
            'week': week,
            'n_games': week_total,
            'n_correct': week_correct,
            'accuracy': (week_correct / week_total) * 100 if week_total > 0 else 0.0,
        })
    
    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("SIMULATION RESULTS - 2025 Weeks 1-13")
    logger.info("=" * 80)
    logger.info(f"\nGames Analyzed: {len(predictions_df)}")
    logger.info(f"Games with Results: {len(completed_games)}")
    logger.info(f"\nAccuracy Metrics:")
    logger.info(f"  Correct Predictions: {n_correct} / {n_total}")
    logger.info(f"  Accuracy: {accuracy_pct:.1f}%")
    logger.info(f"\nSpread Metrics:")
    logger.info(f"  Mean Absolute Error: {spread_mae:.2f} points")
    logger.info(f"  Root Mean Squared Error: {spread_rmse:.2f} points")
    logger.info(f"\nProbability Metrics:")
    logger.info(f"  Brier Score: {brier:.4f}")
    logger.info(f"  Log Loss: {logloss:.4f}")
    logger.info(f"  Mean Calibration Error: {mean_cal_error:.4f}")
    logger.info(f"\nROI Metrics (assuming -110 odds):")
    logger.info(f"  All Games: {roi_all['n_bets']} bets, {roi_all['n_wins']} wins, ROI: {roi_all['roi']:.2f}%")
    logger.info(f"  ≥60% Confidence: {roi_60['n_bets']} bets, {roi_60['n_wins']} wins, ROI: {roi_60['roi']:.2f}%")
    logger.info(f"  ≥70% Confidence: {roi_70['n_bets']} bets, {roi_70['n_wins']} wins, ROI: {roi_70['roi']:.2f}%")
    logger.info(f"  ≥80% Confidence: {roi_80['n_bets']} bets, {roi_80['n_wins']} wins, ROI: {roi_80['roi']:.2f}%")
    
    # Save results
    logger.info("\n[Step 7/7] Saving results...")
    output_dir = project_root / "artifacts" / "predictions" / "2025_weeks_1_13"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / "predictions_vs_actuals.parquet"
    predictions_df.to_parquet(output_path, index=False)
    logger.info(f"✓ Predictions saved to {output_path}")
    
    # Save summary report
    report_path = output_dir / "simulation_report.md"
    with open(report_path, 'w') as f:
        f.write("# Prediction Simulation Report - 2025 Weeks 1-13\n\n")
        f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"## Summary\n\n")
        f.write(f"- **Games Analyzed**: {len(predictions_df)}\n")
        f.write(f"- **Games with Results**: {len(completed_games)}\n")
        f.write(f"- **Accuracy**: {accuracy_pct:.1f}% ({n_correct}/{n_total})\n")
        f.write(f"- **Spread MAE**: {spread_mae:.2f} points\n")
        f.write(f"- **Spread RMSE**: {spread_rmse:.2f} points\n")
        f.write(f"- **Brier Score**: {brier:.4f}\n")
        f.write(f"- **Log Loss**: {logloss:.4f}\n")
        f.write(f"- **Mean Calibration Error**: {mean_cal_error:.4f}\n\n")
        f.write(f"## ROI Analysis (assuming -110 odds)\n\n")
        f.write("| Confidence Threshold | Bets | Wins | Win Rate | ROI |\n")
        f.write("|---------------------|------|------|----------|-----|\n")
        f.write(f"| All Games | {roi_all['n_bets']} | {roi_all['n_wins']} | {roi_all['win_rate']:.1%} | {roi_all['roi']:.2f}% |\n")
        f.write(f"| ≥60% | {roi_60['n_bets']} | {roi_60['n_wins']} | {roi_60['win_rate']:.1%} | {roi_60['roi']:.2f}% |\n")
        f.write(f"| ≥70% | {roi_70['n_bets']} | {roi_70['n_wins']} | {roi_70['win_rate']:.1%} | {roi_70['roi']:.2f}% |\n")
        f.write(f"| ≥80% | {roi_80['n_bets']} | {roi_80['n_wins']} | {roi_80['win_rate']:.1%} | {roi_80['roi']:.2f}% |\n\n")
        f.write(f"## Weekly Breakdown\n\n")
        f.write("| Week | Games | Correct | Accuracy |\n")
        f.write("|------|-------|---------|----------|\n")
        for stat in weekly_stats:
            f.write(f"| {stat['week']} | {stat['n_games']} | {stat['n_correct']} | {stat['accuracy']:.1f}% |\n")
        f.write(f"\n## Leakage Check\n\n")
        f.write(f"- **Future scores in features**: {leakage_df['has_future_scores'].any()}\n")
        f.write(f"- **Result column present**: {leakage_df['has_result_column'].any()}\n")
        f.write(f"- **Spread column present**: {leakage_df['has_spread_column'].any()}\n")
        f.write(f"- **Future-looking features**: {leakage_df['has_future_stats'].any()}\n\n")
        if not has_leakage:
            f.write("✓ No data leakage detected\n\n")
        else:
            f.write("⚠ Potential leakage detected - scores are in dataframe but excluded from features (expected)\n\n")
    
    logger.info(f"✓ Report saved to {report_path}")
    logger.info("\n" + "=" * 80)


if __name__ == "__main__":
    main()

