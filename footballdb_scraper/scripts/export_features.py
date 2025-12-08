"""
Export FootballDB data as model features.

Creates features:
- Surface performance differential (grass vs turf)
- Clutch performance (trailing by 1-8 performance)
- Home/Away differential
- Coach experience features
"""
import logging
from pathlib import Path
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FootballDBFeatureExtractor:
    """
    Extract model features from FootballDB splits.
    
    Key features:
    1. Surface differential (grass_yards - turf_yards)
    2. Clutch performance (trailing_rating - overall_rating)
    3. 4th quarter performance
    4. Thursday night adjustment
    5. Coach win rate
    """
    
    def __init__(self, data_dir: str = "data/raw/footballdb"):
        self.data_dir = Path(data_dir)
    
    def load_player_splits(self) -> pd.DataFrame:
        """Load player splits data."""
        path = self.data_dir / "player_splits" / "player_splits_all.parquet"
        if path.exists():
            return pd.read_parquet(path)
        return pd.DataFrame()
    
    def load_coaches(self) -> pd.DataFrame:
        """Load coach data."""
        path = self.data_dir / "coaches" / "current_coaches.parquet"
        if path.exists():
            return pd.read_parquet(path)
        return pd.DataFrame()
    
    def compute_surface_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute grass vs turf performance differential.
        
        Creates features:
        - {stat}_grass_vs_turf_diff
        - grass_preference_score
        """
        # Filter to surface splits only
        grass = df[df['split_type'] == 'surface-grass'].copy()
        turf = df[df['split_type'] == 'surface-turf'].copy()
        
        if grass.empty or turf.empty:
            logger.warning("Missing surface split data")
            return pd.DataFrame()
        
        # Merge on player/season
        merged = grass.merge(
            turf,
            on=['player_name', 'team', 'season', 'stat_type'],
            suffixes=('_grass', '_turf')
        )
        
        # Compute differentials for key stats
        stat_cols = ['yards', 'touchdowns', 'rating', 'average']
        
        for col in stat_cols:
            grass_col = f'{col}_grass'
            turf_col = f'{col}_turf'
            
            if grass_col in merged.columns and turf_col in merged.columns:
                merged[f'{col}_surface_diff'] = (
                    merged[grass_col].fillna(0) - merged[turf_col].fillna(0)
                )
        
        return merged
    
    def compute_clutch_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute clutch performance features.
        
        Compares performance when:
        - Trailing by 1-8 (clutch)
        - Leading by 1-8 (protect lead)
        - vs overall
        """
        trailing = df[df['split_type'] == 'trailing-by-1-to-8'].copy()
        leading = df[df['split_type'] == 'leading-by-1-to-8'].copy()
        overall = df[df['split_type'] == 'year-to-date'].copy()
        
        if trailing.empty or overall.empty:
            return pd.DataFrame()
        
        # Merge trailing with overall
        merged = trailing.merge(
            overall,
            on=['player_name', 'team', 'season', 'stat_type'],
            suffixes=('_trailing', '_overall')
        )
        
        # Compute clutch differential
        if 'rating_trailing' in merged.columns and 'rating_overall' in merged.columns:
            merged['clutch_rating_diff'] = (
                merged['rating_trailing'].fillna(0) - merged['rating_overall'].fillna(0)
            )
        
        return merged
    
    def compute_4th_quarter_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute 4th quarter performance features.
        
        Key for predicting close game outcomes.
        """
        q4 = df[df['split_type'] == 'fourth-quarter'].copy()
        overall = df[df['split_type'] == 'year-to-date'].copy()
        
        if q4.empty:
            return pd.DataFrame()
        
        merged = q4.merge(
            overall,
            on=['player_name', 'team', 'season', 'stat_type'],
            suffixes=('_q4', '_overall')
        )
        
        if 'rating_q4' in merged.columns:
            merged['q4_rating_diff'] = (
                merged['rating_q4'].fillna(0) - merged.get('rating_overall', 0)
            )
        
        return merged
    
    def compute_thursday_adjustment(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute Thursday Night Football adjustment.
        
        TNF games are notoriously different due to short rest.
        """
        thursday = df[df['split_type'] == 'thursday-games'].copy()
        sunday = df[df['split_type'] == 'sunday-games'].copy()
        
        if thursday.empty or sunday.empty:
            return pd.DataFrame()
        
        merged = thursday.merge(
            sunday,
            on=['player_name', 'team', 'season', 'stat_type'],
            suffixes=('_thursday', '_sunday')
        )
        
        # Thursday performance relative to Sunday
        for col in ['yards', 'rating']:
            thu_col = f'{col}_thursday'
            sun_col = f'{col}_sunday'
            
            if thu_col in merged.columns and sun_col in merged.columns:
                merged[f'{col}_thursday_diff'] = (
                    merged[thu_col].fillna(0) - merged[sun_col].fillna(0)
                )
        
        return merged
    
    def compute_coach_features(self, coaches_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute coach experience features.
        
        Features:
        - Coach win rate
        - Coach experience (years)
        - Playoff experience
        """
        if coaches_df.empty:
            return pd.DataFrame()
        
        features = coaches_df[['name', 'team', 'seasons', 'wins', 'losses', 'win_pct']].copy()
        
        # Experience tier
        features['coach_experience_tier'] = pd.cut(
            features['seasons'].fillna(0),
            bins=[0, 2, 5, 10, float('inf')],
            labels=['rookie', 'developing', 'experienced', 'veteran']
        )
        
        # Win rate tier
        features['coach_win_tier'] = pd.cut(
            features['win_pct'].fillna(0),
            bins=[0, 0.4, 0.5, 0.55, 0.6, 1.0],
            labels=['poor', 'below_avg', 'average', 'good', 'elite']
        )
        
        return features
    
    def export_all_features(self, output_path: str = "data/processed/footballdb_features.parquet"):
        """
        Export all computed features.
        """
        splits = self.load_player_splits()
        coaches = self.load_coaches()
        
        all_features = {}
        
        # Surface features
        surface = self.compute_surface_features(splits)
        if not surface.empty:
            all_features['surface'] = surface
            logger.info(f"Surface features: {len(surface)} records")
        
        # Clutch features
        clutch = self.compute_clutch_features(splits)
        if not clutch.empty:
            all_features['clutch'] = clutch
            logger.info(f"Clutch features: {len(clutch)} records")
        
        # Q4 features
        q4 = self.compute_4th_quarter_features(splits)
        if not q4.empty:
            all_features['q4'] = q4
            logger.info(f"Q4 features: {len(q4)} records")
        
        # Thursday features
        thursday = self.compute_thursday_adjustment(splits)
        if not thursday.empty:
            all_features['thursday'] = thursday
            logger.info(f"Thursday features: {len(thursday)} records")
        
        # Coach features
        coach_features = self.compute_coach_features(coaches)
        if not coach_features.empty:
            all_features['coaches'] = coach_features
            logger.info(f"Coach features: {len(coach_features)} records")
        
        # Save each feature set
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for name, df in all_features.items():
            path = output_dir / f"footballdb_{name}.parquet"
            df.to_parquet(path, index=False)
            logger.info(f"Saved: {path}")
        
        return all_features


if __name__ == "__main__":
    extractor = FootballDBFeatureExtractor()
    features = extractor.export_all_features()

