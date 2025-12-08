"""
NFL Game Prediction Script using NGS Model

Generates predictions for upcoming NFL games using the trained NGS model.

Usage:
    # Predict specific week
    python scripts/predict_ngs.py --season 2024 --week 14

    # Predict with custom threshold
    python scripts/predict_ngs.py --season 2024 --week 14 --threshold 0.55

    # Output to file
    python scripts/predict_ngs.py --season 2024 --week 14 --output predictions.csv
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.ngs_production_model import NGSProductionModel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NFLPredictor:
    """
    NFL game predictor using NGS production model.
    """

    def __init__(
        self,
        model_path: str = "models/artifacts/nfl_ngs_production",
        features_path: str = "data/nfl/processed/game_features_with_ngs.parquet"
    ):
        """
        Initialize the predictor.

        Args:
            model_path: Path to saved model
            features_path: Path to features file
        """
        self.model_path = Path(model_path)
        self.features_path = Path(features_path)

        self.model: Optional[NGSProductionModel] = None
        self.features_df: Optional[pd.DataFrame] = None

    def load(self) -> None:
        """Load model and features."""
        logger.info(f"Loading model from {self.model_path}")
        self.model = NGSProductionModel.load(str(self.model_path))

        logger.info(f"Loading features from {self.features_path}")
        self.features_df = pd.read_parquet(self.features_path)

    def get_games(self, season: int, week: int) -> pd.DataFrame:
        """
        Get games for a specific week.

        Args:
            season: Season year
            week: Week number

        Returns:
            DataFrame with games for the week
        """
        mask = (self.features_df['season'] == season) & (self.features_df['week'] == week)
        games = self.features_df[mask].copy()

        if len(games) == 0:
            logger.warning(f"No games found for {season} Week {week}")

        return games

    def predict_week(
        self,
        season: int,
        week: int,
        threshold: float = 0.52
    ) -> pd.DataFrame:
        """
        Generate predictions for all games in a week.

        Args:
            season: Season year
            week: Week number
            threshold: Confidence threshold for betting recommendation

        Returns:
            DataFrame with predictions
        """
        if self.model is None:
            self.load()

        games = self.get_games(season, week)

        if len(games) == 0:
            return pd.DataFrame()

        # Get predictions
        proba = self.model.predict_proba(games)

        # Build results
        results = []
        for idx, (_, game) in enumerate(games.iterrows()):
            home_prob = proba[idx]
            away_prob = 1 - home_prob

            # Determine prediction
            if home_prob >= 0.5:
                predicted_winner = game['home_team']
                confidence = home_prob
                pick = game['home_team']
            else:
                predicted_winner = game['away_team']
                confidence = away_prob
                pick = game['away_team']

            # Edge calculation
            edge = abs(home_prob - 0.5)

            # Check actual result if available
            actual_winner = None
            correct = None
            if pd.notna(game.get('home_score')) and pd.notna(game.get('away_score')):
                if game['home_score'] > game['away_score']:
                    actual_winner = game['home_team']
                else:
                    actual_winner = game['away_team']
                correct = (predicted_winner == actual_winner)

            results.append({
                'game_id': game['game_id'],
                'season': season,
                'week': week,
                'away_team': game['away_team'],
                'home_team': game['home_team'],
                'away_prob': away_prob,
                'home_prob': home_prob,
                'pick': pick,
                'confidence': confidence,
                'edge': edge,
                'bet': 'YES' if edge >= (threshold - 0.5) else 'NO',
                'away_score': game.get('away_score'),
                'home_score': game.get('home_score'),
                'actual_winner': actual_winner,
                'correct': correct
            })

        return pd.DataFrame(results)

    def print_predictions(
        self,
        predictions: pd.DataFrame,
        show_details: bool = True
    ) -> None:
        """
        Print predictions in a formatted table.

        Args:
            predictions: DataFrame with predictions
            show_details: Whether to show detailed stats
        """
        if len(predictions) == 0:
            print("No predictions available")
            return

        season = predictions['season'].iloc[0]
        week = predictions['week'].iloc[0]

        print("\n" + "=" * 80)
        print(f"NFL Predictions: {season} Week {week}")
        print("=" * 80)

        # Header
        print(f"\n{'Matchup':<25} {'Pick':<5} {'Conf':>6} {'Edge':>6} {'Bet':>4}", end="")
        if predictions['actual_winner'].notna().any():
            print(f" {'Result':>8}")
        else:
            print()

        print("-" * 80)

        for _, row in predictions.iterrows():
            matchup = f"{row['away_team']} @ {row['home_team']}"

            result_str = ""
            if pd.notna(row['actual_winner']):
                if row['correct']:
                    result_str = f" {'✓ WIN':>8}"
                else:
                    result_str = f" {'✗ LOSS':>8}"

            print(f"{matchup:<25} {row['pick']:<5} {row['confidence']:>5.1%} {row['edge']:>5.1%} {row['bet']:>4}{result_str}")

        print("-" * 80)

        # Summary
        bet_games = predictions[predictions['bet'] == 'YES']
        print(f"\nTotal games: {len(predictions)}")
        print(f"Recommended bets: {len(bet_games)} (edge >= threshold)")

        if predictions['correct'].notna().any():
            correct = predictions['correct'].sum()
            total = predictions['correct'].notna().sum()
            print(f"\nResults: {correct}/{total} correct ({correct/total:.1%})")

            if len(bet_games) > 0 and bet_games['correct'].notna().any():
                bet_correct = bet_games['correct'].sum()
                bet_total = bet_games['correct'].notna().sum()
                print(f"Bet results: {bet_correct}/{bet_total} correct ({bet_correct/bet_total:.1%})")

        print()

    def backtest_season(
        self,
        season: int,
        start_week: int = 1,
        end_week: int = 18,
        threshold: float = 0.52
    ) -> pd.DataFrame:
        """
        Backtest predictions for a season.

        Args:
            season: Season year
            start_week: First week
            end_week: Last week
            threshold: Betting threshold

        Returns:
            DataFrame with all predictions
        """
        if self.model is None:
            self.load()

        all_predictions = []

        for week in range(start_week, end_week + 1):
            preds = self.predict_week(season, week, threshold)
            if len(preds) > 0:
                all_predictions.append(preds)

        if not all_predictions:
            return pd.DataFrame()

        results = pd.concat(all_predictions, ignore_index=True)

        # Summary
        if results['correct'].notna().any():
            correct = results['correct'].sum()
            total = results['correct'].notna().sum()
            print(f"\n=== {season} Season Summary ===")
            print(f"Overall: {correct}/{total} ({correct/total:.1%})")

            bet_games = results[results['bet'] == 'YES']
            if len(bet_games) > 0 and bet_games['correct'].notna().any():
                bet_correct = bet_games['correct'].sum()
                bet_total = bet_games['correct'].notna().sum()
                print(f"Bets (edge >= {threshold-0.5:.0%}): {bet_correct}/{bet_total} ({bet_correct/bet_total:.1%})")

        return results


def main():
    parser = argparse.ArgumentParser(description="NFL Game Predictions with NGS Model")
    parser.add_argument('--season', type=int, default=2024, help='Season year')
    parser.add_argument('--week', type=int, default=14, help='Week number')
    parser.add_argument('--threshold', type=float, default=0.52,
                        help='Confidence threshold for bet recommendation')
    parser.add_argument('--model', default='models/artifacts/nfl_ngs_production',
                        help='Path to model')
    parser.add_argument('--features', default='data/nfl/processed/game_features_with_ngs.parquet',
                        help='Path to features')
    parser.add_argument('--output', type=str, default=None,
                        help='Output CSV file (optional)')
    parser.add_argument('--backtest', action='store_true',
                        help='Run full season backtest')
    parser.add_argument('--start-week', type=int, default=1,
                        help='Start week for backtest')
    parser.add_argument('--end-week', type=int, default=18,
                        help='End week for backtest')

    args = parser.parse_args()

    predictor = NFLPredictor(
        model_path=args.model,
        features_path=args.features
    )

    if args.backtest:
        # Run season backtest
        results = predictor.backtest_season(
            season=args.season,
            start_week=args.start_week,
            end_week=args.end_week,
            threshold=args.threshold
        )

        if args.output:
            results.to_csv(args.output, index=False)
            print(f"Results saved to {args.output}")
    else:
        # Single week prediction
        predictions = predictor.predict_week(
            season=args.season,
            week=args.week,
            threshold=args.threshold
        )

        predictor.print_predictions(predictions)

        if args.output:
            predictions.to_csv(args.output, index=False)
            print(f"Predictions saved to {args.output}")


if __name__ == "__main__":
    main()
