"""
Backtesting Module

Implements ROI calculation vs closing line using simple betting strategy.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

from eval.metrics import accuracy, brier_score, log_loss, calibration_buckets

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def spread_to_implied_probability(spread: float) -> float:
    """
    Convert point spread to implied win probability.
    
    Simple approximation: uses logistic function.
    More sophisticated methods could use historical spread-to-probability mapping.
    
    Args:
        spread: Point spread (from home team perspective, negative = home favored)
    
    Returns:
        Implied home team win probability
    """
    # Simple logistic mapping: p = 1 / (1 + exp(spread / 3))
    # This approximates that a 3-point spread ≈ 60% win probability
    # Adjust the divisor (3) based on historical data if needed
    p = 1 / (1 + np.exp(spread / 3))
    return p


def compute_market_implied_probabilities(df: pd.DataFrame) -> pd.Series:
    """
    Calculate market-implied home win probabilities.
    
    Uses spread if available, otherwise uses moneyline if available.
    
    CRITICAL: Only uses PRE-GAME market data (closing line).
    No post-game information is used.
    
    Args:
        df: DataFrame with close_spread (and optionally moneyline columns)
    
    Returns:
        Series of market-implied probabilities
    """
    # Ensure we're using pre-game data only
    # close_spread and moneyline are pre-game market data
    # Note: If neither is available, we'll fall back to 0.5 (fair market assumption)
    
    if "moneyline_home" in df.columns and "moneyline_away" in df.columns:
        # Priority 1: Use moneyline (most accurate)
        from models.architectures.market_baseline import moneyline_to_probability
        
        p_market = df.apply(
            lambda row: moneyline_to_probability(
                row.get("moneyline_home", np.nan),
                row.get("moneyline_away", np.nan)
            ) if pd.notna(row.get("moneyline_home")) and pd.notna(row.get("moneyline_away")) else np.nan,
            axis=1
        )
        
        # Fill NaN with spread-based probabilities
        if "close_spread" in df.columns:
            spread_probs = df["close_spread"].apply(spread_to_implied_probability)
            p_market = p_market.fillna(spread_probs)
    elif "close_spread" in df.columns:
        # Priority 2: Use spread
        p_market = df["close_spread"].apply(spread_to_implied_probability)
    else:
        # Fallback: assume 50/50 if no market data
        logger.warning("No market data available, using 0.5 as default probability")
        p_market = pd.Series(0.5, index=df.index)
    
    return p_market


def compute_model_edge(p_model: np.ndarray, p_market: np.ndarray) -> np.ndarray:
    """
    Compute model edge vs market.
    
    Edge = p_model - p_market
    
    Positive edge means model thinks home team is more likely to win than market.
    Negative edge means model thinks home team is less likely to win than market.
    
    Args:
        p_model: Model predicted probabilities (home win)
        p_market: Market-implied probabilities (home win)
    
    Returns:
        Array of edges (same length as inputs)
    """
    assert len(p_model) == len(p_market), "Model and market probabilities must have same length"
    edge = p_model - p_market
    return edge


def simulate_betting(
    y_true: np.ndarray,
    p_model: np.ndarray,
    p_market: np.ndarray,
    edge_threshold: float = 0.05,
    unit_bet_size: float = 1.0,
    use_actual_odds: bool = False,
    df: Optional[pd.DataFrame] = None,
) -> Dict:
    """
    Simulate betting strategy and calculate ROI.
    
    Strategy:
    - Bet on home team if model edge >= threshold
    - Edge = p_model - p_market
    - Track wins/losses and calculate ROI
    
    CRITICAL: Only uses PRE-GAME data. No post-game information leaks into betting decisions.
    
    Args:
        y_true: True outcomes (1 = home win, 0 = away win) - used only for evaluation
        p_model: Model predicted probabilities (PRE-GAME)
        p_market: Market-implied probabilities (PRE-GAME)
        edge_threshold: Minimum edge to place bet (e.g., 0.05 = 5%)
        unit_bet_size: Size of unit bet
        use_actual_odds: If True, use actual moneyline odds for payouts (requires df)
        df: DataFrame with moneyline data (if use_actual_odds=True)
    
    Returns:
        Dictionary with ROI statistics
    """
    # Calculate edge
    edge = compute_model_edge(p_model, p_market)
    
    # Find games where we bet (edge >= threshold)
    bet_mask = edge >= edge_threshold
    n_bets = bet_mask.sum()
    
    if n_bets == 0:
        return {
            "n_bets": 0,
            "win_rate": 0.0,
            "total_staked": 0.0,
            "total_profit": 0.0,
            "roi": 0.0,
            "edge_threshold": edge_threshold,
            "avg_edge": 0.0,
        }
    
    # Get outcomes for games we bet on
    bet_outcomes = y_true[bet_mask]
    
    if use_actual_odds and df is not None:
        # Use actual moneyline odds for payouts
        bet_df = df[bet_mask].copy()
        if "moneyline_home" in bet_df.columns:
            profits = []
            for idx, outcome in enumerate(bet_df.index):
                ml_home = bet_df.loc[outcome, "moneyline_home"]
                if bet_outcomes.iloc[idx] == 1:  # Home win
                    # Calculate profit from moneyline
                    if ml_home > 0:
                        profit = unit_bet_size * (ml_home / 100)
                    else:
                        profit = unit_bet_size * (100 / abs(ml_home))
                else:  # Home loss
                    profit = -unit_bet_size
                profits.append(profit)
            profits = np.array(profits)
        else:
            # Fallback to unit bet
            profits = np.where(bet_outcomes == 1, unit_bet_size, -unit_bet_size)
    else:
        # Simplified: win = +1 unit profit, loss = -1 unit loss
        profits = np.where(bet_outcomes == 1, unit_bet_size, -unit_bet_size)
    
    total_staked = n_bets * unit_bet_size
    total_profit = profits.sum()
    roi = total_profit / total_staked if total_staked > 0 else 0.0
    
    win_rate = bet_outcomes.mean()
    
    return {
        "n_bets": n_bets,
        "win_rate": win_rate,
        "total_staked": total_staked,
        "total_profit": total_profit,
        "roi": roi,
        "edge_threshold": edge_threshold,
        "avg_edge": edge[bet_mask].mean(),
    }


def calculate_roi(
    y_true: np.ndarray,
    p_model: np.ndarray,
    p_market: np.ndarray,
    edge_threshold: float = 0.05,
    unit_bet_size: float = 1.0,
    df: Optional[pd.DataFrame] = None,
) -> Dict:
    """
    Calculate ROI for a simple betting strategy.
    
    Wrapper around simulate_betting for backward compatibility.
    
    Args:
        y_true: True outcomes (1 = home win, 0 = away win)
        p_model: Model predicted probabilities
        p_market: Market-implied probabilities
        edge_threshold: Minimum edge to place bet (e.g., 0.05 = 5%)
        unit_bet_size: Size of unit bet
        df: Optional DataFrame with moneyline data for actual odds
    
    Returns:
        Dictionary with ROI statistics
    """
    return simulate_betting(
        y_true, p_model, p_market, edge_threshold, unit_bet_size, use_actual_odds=False, df=df
    )


def evaluate_model(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    df: pd.DataFrame,
    set_name: str = "test",
    edge_thresholds: List[float] = [0.03, 0.05],
) -> Dict:
    """
    Evaluate a model on a dataset.
    
    Args:
        model: Trained model (implements predict_proba)
        X: Feature matrix
        y: True labels
        df: Full dataframe (for market data)
        set_name: Name of dataset (e.g., "validation", "test")
        edge_thresholds: List of edge thresholds for ROI calculation
    
    Returns:
        Dictionary with evaluation metrics
    """
    logger.info(f"\nEvaluating model on {set_name} set ({len(X)} games)")
    
    # Predictions
    # Market model needs full dataframe, other models need feature matrix
    if hasattr(model, "__class__") and "MarketBaseline" in model.__class__.__name__:
        p_pred = model.predict_proba(df)  # Market model uses df directly
    else:
        p_pred = model.predict_proba(X)  # Other models use feature matrix
    
    y_pred_cls = (p_pred >= 0.5).astype(int)
    
    # Basic metrics
    acc = accuracy(y.values, y_pred_cls)
    brier = brier_score(y.values, p_pred)
    logloss = log_loss(y.values, p_pred)
    
    logger.info(f"Accuracy: {acc:.4f}")
    logger.info(f"Brier Score: {brier:.4f}")
    logger.info(f"Log Loss: {logloss:.4f}")
    
    # Calibration
    calib_df = calibration_buckets(y.values, p_pred, n_bins=10)
    mean_calib_error = calib_df["calibration_error"].mean()
    
    logger.info(f"Mean Calibration Error: {mean_calib_error:.4f}")
    
    # ROI calculation
    # CRITICAL: Only use pre-game market data (closing line)
    # No post-game information (scores, margins) should be used here
    p_market = compute_market_implied_probabilities(df)
    
    roi_results = {}
    for threshold in edge_thresholds:
        roi = calculate_roi(y.values, p_pred, p_market.values, threshold)
        roi_results[f"roi_threshold_{threshold:.2f}"] = roi
        logger.info(f"ROI (edge >= {threshold:.0%}): {roi['roi']:.2%} ({roi['n_bets']} bets)")
    
    return {
        "set_name": set_name,
        "n_games": len(X),
        "accuracy": acc,
        "brier_score": brier,
        "log_loss": logloss,
        "mean_calibration_error": mean_calib_error,
        "calibration_buckets": calib_df,
        "roi_results": roi_results,
    }


def run_season_by_season_analysis(
    model,
    market_model,
    X: pd.DataFrame,
    y: pd.Series,
    df: pd.DataFrame,
    train_seasons: List[int],
    test_seasons: List[int],
    edge_threshold: float = 0.03,
) -> pd.DataFrame:
    """
    Run season-by-season analysis to check stability.
    
    For each test season, evaluate model and market baseline.
    
    Args:
        model: Trained model (e.g., logistic regression)
        market_model: Market-only baseline model
        X: Feature matrix (all seasons)
        y: Target vector (all seasons)
        df: Full dataframe with season column
        train_seasons: List of seasons used for training
        test_seasons: List of seasons to evaluate
        edge_threshold: Edge threshold for ROI calculation
    
    Returns:
        DataFrame with season-by-season metrics
    """
    logger.info("=" * 60)
    logger.info("Season-by-Season Analysis")
    logger.info("=" * 60)
    
    results = []
    
    for season in test_seasons:
        logger.info(f"\nAnalyzing season {season}")
        
        # Filter to this season
        season_mask = df["season"] == season
        X_season = X[season_mask].copy()
        y_season = y[season_mask].copy()
        df_season = df[season_mask].copy()
        
        if len(X_season) == 0:
            logger.warning(f"No data for season {season}, skipping")
            continue
        
        # Model predictions
        p_model = model.predict_proba(X_season)
        y_pred_model = (p_model >= 0.5).astype(int)
        
        # Market predictions
        p_market = market_model.predict_proba(df_season)
        y_pred_market = (p_market >= 0.5).astype(int)
        
        # Model metrics
        from eval.metrics import accuracy, brier_score, log_loss
        
        model_acc = accuracy(y_season.values, y_pred_model)
        model_brier = brier_score(y_season.values, p_model)
        model_logloss = log_loss(y_season.values, p_model)
        
        # Market metrics
        market_acc = accuracy(y_season.values, y_pred_market)
        market_brier = brier_score(y_season.values, p_market)
        market_logloss = log_loss(y_season.values, p_market)
        
        # ROI
        model_roi = simulate_betting(
            y_season.values, p_model, p_market, edge_threshold=edge_threshold
        )
        market_roi = simulate_betting(
            y_season.values, p_market, p_market, edge_threshold=edge_threshold
        )  # Market vs itself should be ~0
        
        results.append({
            "season": season,
            "n_games": len(X_season),
            "model_accuracy": model_acc,
            "model_brier": model_brier,
            "model_logloss": model_logloss,
            "model_roi": model_roi["roi"],
            "model_n_bets": model_roi["n_bets"],
            "market_accuracy": market_acc,
            "market_brier": market_brier,
            "market_logloss": market_logloss,
            "market_roi": market_roi["roi"],
            "market_n_bets": market_roi["n_bets"],
        })
        
        logger.info(f"  Model: Acc={model_acc:.3f}, Brier={model_brier:.3f}, ROI={model_roi['roi']:.2%}")
        logger.info(f"  Market: Acc={market_acc:.3f}, Brier={market_brier:.3f}, ROI={market_roi['roi']:.2%}")
    
    return pd.DataFrame(results)


def run_backtest(
    logit_model,
    gbm_model,
    ensemble_model,
    market_model,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    df_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    df_test: pd.DataFrame,
    edge_thresholds: List[float] = [0.03, 0.05],
) -> Dict:
    """
    Run backtest evaluation on all models.
    
    Args:
        logit_model: Trained logistic regression model
        gbm_model: Trained gradient boosting model
        ensemble_model: Trained ensemble model
        X_val, y_val, df_val: Validation set
        X_test, y_test, df_test: Test set
        edge_thresholds: List of edge thresholds for ROI
    
    Returns:
        Dictionary with evaluation results for all models
    """
    logger.info("=" * 60)
    logger.info("Running Backtest Evaluation")
    logger.info("=" * 60)
    
    results = {}
    
    # Evaluate each model on validation and test
    for model_name, model in [
        ("logistic_regression", logit_model),
        ("gradient_boosting", gbm_model),
        ("ensemble", ensemble_model),
        ("market_baseline", market_model),
    ]:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Model: {model_name.upper()}")
        logger.info("=" * 60)
        
        val_results = evaluate_model(
            model, X_val, y_val, df_val, "validation", edge_thresholds
        )
        test_results = evaluate_model(
            model, X_test, y_test, df_test, "test", edge_thresholds
        )
        
        results[model_name] = {
            "validation": val_results,
            "test": test_results,
        }
    
    return results

