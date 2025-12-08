"""
Predictr Prediction API
Serverless-ready FastAPI application exposing ML predictions
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import pickle
import pandas as pd
import numpy as np
import math
from pathlib import Path
import logging
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.architectures.stacking_ensemble import StackingEnsemble
from models.architectures.gradient_boosting import GradientBoostingModel
from models.architectures.ft_transformer import FTTransformerModel
from models.architectures.tabnet import TabNetModel
from models.base import BaseModel as BaseModelInterface

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Predictr API",
    description="NFL game predictions powered by ML ensemble",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://predictr.app",  # Update with your domain
        "https://*.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Paths - adjust based on deployment
BASE_PATH = Path(__file__).parent.parent
MODEL_PATH = BASE_PATH / "artifacts/models/nfl_stacked_ensemble_v2/ensemble_v1.pkl"
FEATURES_PATH = BASE_PATH / "data/nfl/processed/game_features_baseline.parquet"

# Lazy-loaded artifacts
_model = None
_features_df = None

# Columns to exclude from features (from audit)
EXCLUDE_COLS = [
    "game_id", "season", "week", "date", "home_team", "away_team",
    "home_score", "away_score", "home_win", "close_spread", "close_total",
    "open_spread", "open_total"
]


def custom_base_model_loader(path):
    """Load base model with proper type detection."""
    path = Path(path)
    if 'ft_transformer' in str(path):
        return FTTransformerModel.load(path)
    elif 'tabnet' in str(path):
        model = TabNetModel.load(path)
        # Force CPU if CUDA not compatible
        if hasattr(model, 'device') and model.device == 'cuda':
            import torch
            try:
                test_tensor = torch.zeros(1).cuda()
                _ = test_tensor + 1
                del test_tensor
                torch.cuda.empty_cache()
            except Exception:
                model.device = 'cpu'
                if hasattr(model, 'model') and model.model is not None:
                    if hasattr(model.model, 'device'):
                        model.model.device = 'cpu'
                    if hasattr(model.model, 'to'):
                        model.model = model.model.to('cpu')
        return model
    elif 'gbm' in str(path):
        return GradientBoostingModel.load(path)
    else:
        return BaseModelInterface.load(path)


def get_model():
    """Lazy load model"""
    global _model
    if _model is None:
        logger.info(f"Loading model from {MODEL_PATH}")
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
        _model = StackingEnsemble.load(MODEL_PATH, base_model_loader=custom_base_model_loader)
        logger.info("Model loaded successfully")
    return _model


def get_features_df():
    """Lazy load features dataframe"""
    global _features_df
    if _features_df is None:
        logger.info(f"Loading features from {FEATURES_PATH}")
        if not FEATURES_PATH.exists():
            raise FileNotFoundError(f"Features not found at {FEATURES_PATH}")
        _features_df = pd.read_parquet(FEATURES_PATH)
        logger.info(f"Loaded {len(_features_df)} games with features")
    return _features_df


def prob_to_spread(prob: float) -> float:
    """Convert win probability to spread estimate"""
    if prob <= 0 or prob >= 1:
        return 0.0
    logit = math.log(prob / (1 - prob))
    return round(-3 * logit, 2)


def spread_to_prob(spread: float) -> float:
    """Convert spread to implied probability"""
    return 1 / (1 + math.exp(spread / 3))


# Response Models
class PredictionResponse(BaseModel):
    game_id: str
    season: int
    week: int
    home_team: str
    away_team: str
    win_prob_home: float
    win_prob_away: float
    predicted_spread: float
    confidence: float
    edge_vs_market: Optional[float] = None
    market_spread: Optional[float] = None
    top_factors: list[dict]
    model_version: str = "ensemble_v1"


class WeekPredictionsResponse(BaseModel):
    season: int
    week: int
    predictions: list[PredictionResponse]
    generated_at: str


class EdgePerformance(BaseModel):
    win_rate: float
    sample: int
    roi: float


class ModelPerformanceResponse(BaseModel):
    accuracy: float
    roi: float
    brier_score: float
    sample_size: int
    test_seasons: list[str]
    edge_performance: dict[str, EdgePerformance]


class FeatureCategory(BaseModel):
    description: str
    features: list[str]


class ModelMethodologyResponse(BaseModel):
    architecture: str
    base_models: list[str]
    meta_model: str
    feature_categories: dict[str, FeatureCategory]
    training_period: str
    validation_season: str
    test_season: str
    key_formulas: dict[str, str]


# Endpoints
@app.get("/health")
def health_check():
    """Health check endpoint"""
    model_loaded = _model is not None
    features_loaded = _features_df is not None
    return {
        "status": "healthy",
        "model_loaded": model_loaded,
        "features_loaded": features_loaded
    }


@app.get("/predictions/{game_id}", response_model=PredictionResponse)
def get_prediction(game_id: str):
    """
    Get prediction for a specific game.
    
    Returns win probability, predicted spread, confidence, and top contributing factors.
    """
    try:
        model = get_model()
        df = get_features_df()
        
        game_row = df[df["game_id"] == game_id]
        if game_row.empty:
            raise HTTPException(status_code=404, detail=f"Game {game_id} not found")
        
        row = game_row.iloc[0]
        
        # Extract features
        feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
        X = game_row[feature_cols].values
        
        # Handle missing values
        X = np.nan_to_num(X, nan=0.0)
        
        # Predict
        prob_home = float(model.predict_proba(X)[0])
        prob_away = 1 - prob_home
        predicted_spread = prob_to_spread(prob_home)
        confidence = round(max(prob_home, prob_away) * 100, 1)
        
        # Get market spread if available
        market_spread = None
        edge = None
        if "close_spread" in row and pd.notna(row["close_spread"]):
            market_spread = float(row["close_spread"])
            edge = round(predicted_spread - market_spread, 2)
        
        # Extract feature importances
        top_factors = []
        
        # Try to get feature importances from ensemble
        if hasattr(model, 'get_feature_importances'):
            try:
                importances = model.get_feature_importances()
                if importances is not None and len(importances) == len(feature_cols):
                    sorted_factors = sorted(
                        zip(feature_cols, importances),
                        key=lambda x: abs(x[1]),
                        reverse=True
                    )[:7]
                    
                    for feature, importance in sorted_factors:
                        value = float(game_row[feature].iloc[0]) if feature in game_row.columns else None
                        top_factors.append({
                            "feature": feature,
                            "importance": round(float(importance), 4),
                            "value": round(value, 3) if value is not None else None,
                            "description": get_feature_description(feature)
                        })
            except Exception as e:
                logger.warning(f"Could not extract feature importances: {e}")
        
        # Fallback: use base model importances if available
        if not top_factors and hasattr(model, 'base_models'):
            for base_name, base_model in model.base_models.items():
                if hasattr(base_model, 'feature_importances_'):
                    importances = base_model.feature_importances_
                    if len(importances) == len(feature_cols):
                        sorted_factors = sorted(
                            zip(feature_cols, importances),
                            key=lambda x: abs(x[1]),
                            reverse=True
                        )[:7]
                        
                        for feature, importance in sorted_factors:
                            value = float(game_row[feature].iloc[0]) if feature in game_row.columns else None
                            top_factors.append({
                                "feature": feature,
                                "importance": round(float(importance), 4),
                                "value": round(value, 3) if value is not None else None,
                                "description": get_feature_description(feature)
                            })
                        break
        
        # If still no importances, create placeholder factors
        if not top_factors:
            # Use some common features as placeholders
            common_features = [
                "home_win_rate_last4", "away_win_rate_last4",
                "home_pdiff_last4", "away_pdiff_last4",
                "home_epa_offensive_epa_per_play", "away_epa_offensive_epa_per_play"
            ]
            for feature in common_features[:5]:
                if feature in game_row.columns:
                    value = float(game_row[feature].iloc[0]) if pd.notna(game_row[feature].iloc[0]) else None
                    top_factors.append({
                        "feature": feature,
                        "importance": 0.1,  # Placeholder
                        "value": round(value, 3) if value is not None else None,
                        "description": get_feature_description(feature)
                    })
        
        return PredictionResponse(
            game_id=game_id,
            season=int(row["season"]),
            week=int(row["week"]),
            home_team=row["home_team"],
            away_team=row["away_team"],
            win_prob_home=round(prob_home, 4),
            win_prob_away=round(prob_away, 4),
            predicted_spread=predicted_spread,
            confidence=confidence,
            edge_vs_market=edge,
            market_spread=market_spread,
            top_factors=top_factors[:7]  # Limit to top 7
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting prediction for {game_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/predictions/week/{season}/{week}", response_model=WeekPredictionsResponse)
def get_week_predictions(season: int, week: int):
    """Get all predictions for a specific week"""
    from datetime import datetime
    
    try:
        df = get_features_df()
        week_games = df[(df["season"] == season) & (df["week"] == week)]
        
        if week_games.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No games found for {season} week {week}"
            )
        
        predictions = []
        for game_id in week_games["game_id"].unique():
            try:
                pred = get_prediction(game_id)
                predictions.append(pred)
            except Exception as e:
                logger.warning(f"Failed to get prediction for {game_id}: {e}")
        
        return WeekPredictionsResponse(
            season=season,
            week=week,
            predictions=predictions,
            generated_at=datetime.utcnow().isoformat()
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting week predictions: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/model/performance", response_model=ModelPerformanceResponse)
def get_model_performance():
    """
    Return model backtest performance metrics.
    Used to give AI chat context about model reliability.
    """
    # TODO: Load these from actual evaluation outputs in docs/reports/
    return ModelPerformanceResponse(
        accuracy=0.593,
        roi=0.1329,
        brier_score=0.238,
        sample_size=1247,
        test_seasons=["2023", "2024"],
        edge_performance={
            "0.5+": EdgePerformance(win_rate=0.55, sample=487, roi=0.06),
            "1.0+": EdgePerformance(win_rate=0.57, sample=342, roi=0.09),
            "1.5+": EdgePerformance(win_rate=0.59, sample=218, roi=0.12),
            "2.0+": EdgePerformance(win_rate=0.62, sample=124, roi=0.18),
            "2.5+": EdgePerformance(win_rate=0.64, sample=76, roi=0.22),
        }
    )


@app.get("/model/methodology", response_model=ModelMethodologyResponse)
def get_model_methodology():
    """
    Return detailed model methodology.
    Injected into AI system prompt for accurate explanations.
    """
    return ModelMethodologyResponse(
        architecture="Stacking Ensemble with Meta-Learner",
        base_models=[
            "Gradient Boosting (XGBoost-style, n_estimators=100, max_depth=3, lr=0.1)",
            "FT-Transformer (neural attention-based)",
        ],
        meta_model="Logistic Regression on stacked base predictions",
        feature_categories={
            "team_form": FeatureCategory(
                description="Rolling team performance metrics",
                features=[
                    "win_rate_last4/8/16 - Recent win percentage",
                    "pdiff_last4/8/16 - Point differential",
                    "points_for_last4/8/16 - Scoring average",
                    "points_against_last4/8/16 - Defensive average",
                    "turnover_diff_last4/8/16 - Turnover margin"
                ]
            ),
            "epa": FeatureCategory(
                description="Expected Points Added metrics from nflverse",
                features=[
                    "offensive_epa_per_play - Points added per offensive play",
                    "defensive_epa_per_play_allowed - Points allowed per defensive play",
                    "offensive_pass_epa - Passing game efficiency",
                    "offensive_run_epa - Rushing game efficiency",
                    "success_rate - Percentage of positive EPA plays"
                ]
            ),
            "qb": FeatureCategory(
                description="Quarterback-specific performance",
                features=[
                    "qb_epa_per_dropback - QB efficiency per dropback",
                    "qb_cpoe - Completion % over expected",
                    "qb_sack_rate - Sack frequency"
                ]
            ),
            "rolling_epa": FeatureCategory(
                description="EPA smoothed over recent games",
                features=[
                    "roll_epa_off_epa_last3/5/8 - Rolling offensive EPA",
                    "roll_epa_def_epa_allowed_last3/5/8 - Rolling defensive EPA"
                ]
            )
        },
        training_period="2015-2021 NFL seasons",
        validation_season="2022",
        test_season="2023",
        key_formulas={
            "spread_to_prob": "p = 1 / (1 + exp(spread / 3))",
            "prob_to_spread": "spread = -3 * ln(p / (1-p))",
            "edge": "edge = model_spread - market_spread",
            "confidence": "confidence = max(p, 1-p) * 100"
        }
    )


@app.get("/features/{game_id}")
def get_game_features(game_id: str):
    """
    Get raw feature values for a game.
    Useful for debugging and detailed analysis.
    """
    try:
        df = get_features_df()
        game_row = df[df["game_id"] == game_id]
        
        if game_row.empty:
            raise HTTPException(status_code=404, detail=f"Game {game_id} not found")
        
        row = game_row.iloc[0]
        feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
        
        features = {}
        for col in feature_cols:
            val = row[col]
            if pd.notna(val):
                features[col] = round(float(val), 4) if isinstance(val, (int, float, np.number)) else val
        
        return {
            "game_id": game_id,
            "feature_count": len(features),
            "features": features
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting features for {game_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


def get_feature_description(feature: str) -> str:
    """Human-readable feature descriptions"""
    descriptions = {
        # Team form
        "home_win_rate_last4": "Home team win rate over last 4 games",
        "home_win_rate_last8": "Home team win rate over last 8 games",
        "away_win_rate_last4": "Away team win rate over last 4 games",
        "away_win_rate_last8": "Away team win rate over last 8 games",
        "home_pdiff_last4": "Home team point differential (last 4)",
        "away_pdiff_last4": "Away team point differential (last 4)",
        
        # EPA
        "home_epa_offensive_epa_per_play": "Home offensive EPA per play",
        "away_epa_offensive_epa_per_play": "Away offensive EPA per play",
        "home_epa_defensive_epa_per_play_allowed": "Home defensive EPA allowed",
        "away_epa_defensive_epa_per_play_allowed": "Away defensive EPA allowed",
        
        # QB
        "home_qb_qb_epa_per_dropback": "Home QB EPA per dropback",
        "away_qb_qb_epa_per_dropback": "Away QB EPA per dropback",
        "home_qb_qb_cpoe": "Home QB completion % over expected",
        "away_qb_qb_cpoe": "Away QB completion % over expected",
    }
    return descriptions.get(feature, feature.replace("_", " ").title())


# Modal deployment wrapper (if using Modal)
# Uncomment and configure if deploying to Modal
"""
import modal

app_modal = modal.App("predictr-api")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "fastapi", "uvicorn", "pandas", "pyarrow", "scikit-learn", "xgboost", "pydantic"
)

volume = modal.Volume.from_name("predictr-artifacts", create_if_missing=True)

@app_modal.function(
    image=image,
    volumes={"/artifacts": volume},
    cpu=1.0,
    memory=1024,
    keep_warm=1,  # Keep one instance warm for fast responses
)
@modal.asgi_app()
def fastapi_app():
    return app
"""

