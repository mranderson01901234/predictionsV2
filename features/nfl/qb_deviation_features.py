"""
NFL QB Deviation-from-Baseline Features

Computes granular QB performance features including:
- Career baselines (entering each game)
- Season-to-date metrics
- Deviation from baseline (z-scores)
- Opponent-adjusted metrics
- Trend features
- Regression/luck indicators

Key principle: All features are computed using ONLY data available
BEFORE the game being predicted (no data leakage).

Expected Impact: +1-2% accuracy improvement by capturing:
- QBs performing above/below career norms
- Regression candidates (unsustainable INT rates, etc.)
- Hot/cold streaks
- Opponent-adjusted performance
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass, field
import logging
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Minimum sample sizes for reliable metrics
MIN_CAREER_DROPBACKS = 100
MIN_SEASON_DROPBACKS = 20
MIN_GAMES_FOR_STD = 8


@dataclass
class QBCareerBaseline:
    """Career baseline metrics for a QB entering a specific game."""
    player_id: str
    player_name: str
    games_started: int
    pass_attempts: int
    dropbacks: int

    # Efficiency metrics
    epa_per_dropback: float
    epa_std: float  # Game-level std for z-score calculation
    cpoe: float
    cpoe_std: float
    completion_pct: float
    yards_per_attempt: float
    td_rate: float
    int_rate: float
    int_rate_std: float
    sack_rate: float
    passer_rating: float
    success_rate: float

    # Game-level metrics for percentile calculations
    game_epas: List[float] = field(default_factory=list)
    game_int_rates: List[float] = field(default_factory=list)


def load_raw_pbp_data(seasons: Optional[List[int]] = None) -> pd.DataFrame:
    """
    Load raw play-by-play data from nflverse.

    This function loads raw PBP data which includes QB-specific columns
    that are not in the normalized schema (passer_id, cpoe, air_yards, etc.)

    Args:
        seasons: List of seasons to load. If None, loads all available.

    Returns:
        DataFrame with raw play-by-play data
    """
    raw_path = (
        Path(__file__).parent.parent.parent
        / "data"
        / "nfl"
        / "raw"
        / "play_by_play.parquet"
    )

    if raw_path.exists():
        logger.info(f"Loading raw PBP data from {raw_path}")
        df = pd.read_parquet(raw_path)
        if seasons is not None:
            df = df[df["season"].isin(seasons)]
        return df

    # If raw data doesn't exist, try to fetch it
    logger.info("Raw PBP data not found, attempting to fetch from nflverse...")
    try:
        import nfl_data_py as nfl

        if seasons is None:
            seasons = list(range(2015, 2025))

        df = nfl.import_pbp_data(seasons, downcast=True, cache=False)

        # Cache for future use
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(raw_path, index=False)
        logger.info(f"Cached raw PBP data to {raw_path}")

        return df
    except ImportError:
        raise ImportError(
            "nfl_data_py is required for QB deviation features. "
            "Install with: pip install nfl-data-py"
        )
    except Exception as e:
        raise RuntimeError(f"Failed to fetch PBP data: {e}")


def compute_passer_rating(
    completions: int,
    attempts: int,
    yards: float,
    tds: int,
    ints: int
) -> float:
    """Compute NFL passer rating."""
    if attempts == 0:
        return 0.0

    comp_pct = completions / attempts
    ypa = yards / attempts
    td_pct = tds / attempts
    int_pct = ints / attempts

    # NFL passer rating formula components
    a = max(0, min(2.375, ((comp_pct * 100) - 30) / 20))
    b = max(0, min(2.375, (ypa - 3) / 4))
    c = max(0, min(2.375, td_pct * 20))
    d = max(0, min(2.375, 2.375 - (int_pct * 25)))

    return ((a + b + c + d) / 6) * 100


class QBDeviationFeatureGenerator:
    """
    Generates QB deviation-from-baseline features.

    Key Features:
    - Career baseline metrics (EPA, CPOE, INT rate, etc.)
    - Season-to-date metrics
    - Deviation from career (raw and z-scores)
    - Performance trends (last 4 games)
    - Regression indicators (luck proxies)

    Usage:
        generator = QBDeviationFeatureGenerator()
        features = generator.generate_game_features(game_id, season, week, team, qb_id, game_date)
    """

    def __init__(
        self,
        pbp_df: Optional[pd.DataFrame] = None,
        games_df: Optional[pd.DataFrame] = None
    ):
        """
        Initialize with play-by-play and games data.

        Args:
            pbp_df: Raw play-by-play DataFrame with QB columns.
                    If None, will load from default location.
            games_df: Games DataFrame for date information.
                      If None, will load from default location.
        """
        if pbp_df is None:
            pbp_df = load_raw_pbp_data()

        if games_df is None:
            games_path = (
                Path(__file__).parent.parent.parent
                / "data"
                / "nfl"
                / "staged"
                / "games.parquet"
            )
            if games_path.exists():
                games_df = pd.read_parquet(games_path)
            else:
                logger.warning("games.parquet not found, some features may be limited")
                games_df = None

        self.pbp_df = pbp_df
        self.games_df = games_df

        # Identify passer column
        self.passer_col = None
        for col in ["passer_id", "passer_player_id", "passer"]:
            if col in self.pbp_df.columns:
                self.passer_col = col
                break

        if self.passer_col is None:
            raise ValueError("No passer ID column found in PBP data")

        # Identify passer name column
        self.passer_name_col = None
        for col in ["passer_player_name", "passer_name", "passer"]:
            if col in self.pbp_df.columns and col != self.passer_col:
                self.passer_name_col = col
                break

        # Precompute game dates for joining
        self._precompute_game_dates()

        # Cache for career baselines
        self._career_cache: Dict[Tuple[str, str], QBCareerBaseline] = {}

        logger.info(f"QBDeviationFeatureGenerator initialized with {len(pbp_df)} plays")

    def _precompute_game_dates(self):
        """Precompute game dates for efficient lookups."""
        if self.games_df is not None:
            self.game_dates = dict(zip(
                self.games_df["game_id"],
                pd.to_datetime(self.games_df["date"])
            ))
        else:
            self.game_dates = {}

    def _get_qb_plays(
        self,
        qb_id: str,
        before_date: pd.Timestamp,
        season_filter: Optional[int] = None,
        before_week_filter: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Get all plays for a QB before a specific date.

        Args:
            qb_id: QB player ID
            before_date: Only include plays before this date
            season_filter: If provided, only include plays from this season
            before_week_filter: If provided, only include plays before this week

        Returns:
            DataFrame of QB's plays
        """
        # Filter to QB's pass plays
        mask = (
            (self.pbp_df[self.passer_col] == qb_id) &
            (self.pbp_df["play_type"].isin(["pass", "qb_spike"]) |
             (self.pbp_df.get("sack", 0) == 1))
        )

        qb_plays = self.pbp_df[mask].copy()

        if len(qb_plays) == 0:
            return qb_plays

        # Add game date
        if "game_date" not in qb_plays.columns:
            if "game_id" in qb_plays.columns and self.game_dates:
                qb_plays["game_date"] = qb_plays["game_id"].map(self.game_dates)
            elif "gameday" in qb_plays.columns:
                qb_plays["game_date"] = pd.to_datetime(qb_plays["gameday"])

        # Filter by date
        if "game_date" in qb_plays.columns:
            qb_plays = qb_plays[qb_plays["game_date"] < before_date]

        # Filter by season/week if specified
        if season_filter is not None:
            qb_plays = qb_plays[qb_plays["season"] == season_filter]
            if before_week_filter is not None:
                qb_plays = qb_plays[qb_plays["week"] < before_week_filter]

        return qb_plays

    def _compute_play_metrics(self, plays: pd.DataFrame) -> Dict[str, float]:
        """Compute aggregate metrics from a set of plays."""
        if len(plays) == 0:
            return self._empty_play_metrics()

        # Identify dropbacks and pass attempts
        is_pass = plays["play_type"] == "pass" if "play_type" in plays.columns else pd.Series([False] * len(plays), index=plays.index)
        is_sack = plays["sack"] == 1 if "sack" in plays.columns else pd.Series([False] * len(plays), index=plays.index)

        pass_attempts = plays[is_pass].copy()
        dropbacks = plays[is_pass | is_sack].copy()

        n_dropbacks = len(dropbacks)
        n_attempts = len(pass_attempts)

        if n_dropbacks == 0:
            return self._empty_play_metrics()

        # EPA per dropback
        epa_per_dropback = dropbacks["epa"].mean() if "epa" in dropbacks.columns else 0.0

        # Success rate
        success_rate = dropbacks["success"].mean() if "success" in dropbacks.columns else 0.0

        # CPOE
        cpoe = pass_attempts["cpoe"].mean() if "cpoe" in pass_attempts.columns else None

        # Completion percentage
        completions = pass_attempts["complete_pass"].sum() if "complete_pass" in pass_attempts.columns else 0
        comp_pct = completions / n_attempts if n_attempts > 0 else 0.0

        # Yards per attempt
        yards = pass_attempts["passing_yards"].sum() if "passing_yards" in pass_attempts.columns else 0
        if yards == 0 and "yards_gained" in pass_attempts.columns:
            yards = pass_attempts["yards_gained"].sum()
        ypa = yards / n_attempts if n_attempts > 0 else 0.0

        # Air yards per attempt
        air_yards = pass_attempts["air_yards"].sum() if "air_yards" in pass_attempts.columns else 0
        aypa = air_yards / n_attempts if n_attempts > 0 else 0.0

        # TD rate
        tds = pass_attempts["pass_touchdown"].sum() if "pass_touchdown" in pass_attempts.columns else 0
        if tds == 0 and "touchdown" in pass_attempts.columns:
            tds = pass_attempts["touchdown"].sum()
        td_rate = tds / n_attempts if n_attempts > 0 else 0.0

        # INT rate
        ints = pass_attempts["interception"].sum() if "interception" in pass_attempts.columns else 0
        int_rate = ints / n_attempts if n_attempts > 0 else 0.0

        # Sack rate
        sacks = is_sack.sum()
        sack_rate = sacks / n_dropbacks if n_dropbacks > 0 else 0.0

        # Passer rating
        passer_rating = compute_passer_rating(
            completions=int(completions),
            attempts=n_attempts,
            yards=float(yards),
            tds=int(tds),
            ints=int(ints)
        )

        return {
            "dropbacks": n_dropbacks,
            "pass_attempts": n_attempts,
            "epa_per_dropback": epa_per_dropback,
            "success_rate": success_rate,
            "cpoe": cpoe,
            "completion_pct": comp_pct,
            "yards_per_attempt": ypa,
            "air_yards_per_attempt": aypa,
            "td_rate": td_rate,
            "int_rate": int_rate,
            "sack_rate": sack_rate,
            "passer_rating": passer_rating,
        }

    def _empty_play_metrics(self) -> Dict[str, float]:
        """Return empty metrics dict."""
        return {
            "dropbacks": 0,
            "pass_attempts": 0,
            "epa_per_dropback": 0.0,
            "success_rate": 0.0,
            "cpoe": None,
            "completion_pct": 0.0,
            "yards_per_attempt": 0.0,
            "air_yards_per_attempt": 0.0,
            "td_rate": 0.0,
            "int_rate": 0.0,
            "sack_rate": 0.0,
            "passer_rating": 0.0,
        }

    def get_career_baseline(
        self,
        qb_id: str,
        before_date: pd.Timestamp
    ) -> Optional[QBCareerBaseline]:
        """
        Compute career baseline for a QB using only games BEFORE the specified date.

        This is the critical function that ensures no data leakage.

        Args:
            qb_id: QB player ID
            before_date: Only include games before this date

        Returns:
            QBCareerBaseline or None if insufficient data
        """
        cache_key = (qb_id, str(before_date.date()))
        if cache_key in self._career_cache:
            return self._career_cache[cache_key]

        # Get QB's plays before this game
        qb_plays = self._get_qb_plays(qb_id, before_date)

        if len(qb_plays) < MIN_CAREER_DROPBACKS:
            return None

        # Compute aggregate metrics
        metrics = self._compute_play_metrics(qb_plays)

        if metrics["dropbacks"] < MIN_CAREER_DROPBACKS:
            return None

        # Get QB name
        qb_name = ""
        if self.passer_name_col and self.passer_name_col in qb_plays.columns:
            names = qb_plays[self.passer_name_col].dropna()
            if len(names) > 0:
                qb_name = names.iloc[0]

        # Compute game-level metrics for std and percentile calculations
        game_metrics = []

        # Use game_id or old_game_id for grouping
        game_col = "game_id" if "game_id" in qb_plays.columns else "old_game_id"
        if game_col not in qb_plays.columns:
            game_col = None

        if game_col:
            for game_id, game_plays in qb_plays.groupby(game_col):
                gm = self._compute_play_metrics(game_plays)
                if gm["dropbacks"] >= 5:  # Minimum for a game
                    game_metrics.append({
                        "epa": gm["epa_per_dropback"],
                        "int_rate": gm["int_rate"],
                        "cpoe": gm["cpoe"],
                    })

        # Compute std from game-level metrics
        if len(game_metrics) >= MIN_GAMES_FOR_STD:
            epa_std = pd.Series([g["epa"] for g in game_metrics]).std()
            int_rate_std = pd.Series([g["int_rate"] for g in game_metrics]).std()
            cpoe_values = [g["cpoe"] for g in game_metrics if g["cpoe"] is not None]
            cpoe_std = pd.Series(cpoe_values).std() if len(cpoe_values) >= MIN_GAMES_FOR_STD else 0.0
        else:
            epa_std = 0.0
            int_rate_std = 0.0
            cpoe_std = 0.0

        baseline = QBCareerBaseline(
            player_id=qb_id,
            player_name=qb_name,
            games_started=len(game_metrics),
            pass_attempts=metrics["pass_attempts"],
            dropbacks=metrics["dropbacks"],
            epa_per_dropback=metrics["epa_per_dropback"],
            epa_std=epa_std if not pd.isna(epa_std) else 0.0,
            cpoe=metrics["cpoe"] if metrics["cpoe"] is not None else 0.0,
            cpoe_std=cpoe_std if not pd.isna(cpoe_std) else 0.0,
            completion_pct=metrics["completion_pct"],
            yards_per_attempt=metrics["yards_per_attempt"],
            td_rate=metrics["td_rate"],
            int_rate=metrics["int_rate"],
            int_rate_std=int_rate_std if not pd.isna(int_rate_std) else 0.0,
            sack_rate=metrics["sack_rate"],
            passer_rating=metrics["passer_rating"],
            success_rate=metrics["success_rate"],
            game_epas=[g["epa"] for g in game_metrics],
            game_int_rates=[g["int_rate"] for g in game_metrics],
        )

        self._career_cache[cache_key] = baseline
        return baseline

    def get_season_metrics(
        self,
        qb_id: str,
        season: int,
        before_week: int,
        before_date: pd.Timestamp
    ) -> Dict[str, float]:
        """
        Compute season-to-date metrics for QB entering a specific week.

        Args:
            qb_id: QB player ID
            season: Current season
            before_week: Week number (exclusive)
            before_date: Date of game (for additional filtering)

        Returns:
            Dictionary of season metrics
        """
        qb_plays = self._get_qb_plays(
            qb_id, before_date,
            season_filter=season,
            before_week_filter=before_week
        )

        if len(qb_plays) < MIN_SEASON_DROPBACKS:
            return self._empty_season_metrics()

        metrics = self._compute_play_metrics(qb_plays)

        # Count games
        game_col = "game_id" if "game_id" in qb_plays.columns else "old_game_id"
        n_games = qb_plays[game_col].nunique() if game_col in qb_plays.columns else 0

        return {
            "season_games": n_games,
            "season_dropbacks": metrics["dropbacks"],
            "season_epa_per_dropback": metrics["epa_per_dropback"],
            "season_cpoe": metrics["cpoe"] if metrics["cpoe"] is not None else 0.0,
            "season_completion_pct": metrics["completion_pct"],
            "season_yards_per_attempt": metrics["yards_per_attempt"],
            "season_air_yards_per_attempt": metrics["air_yards_per_attempt"],
            "season_td_rate": metrics["td_rate"],
            "season_int_rate": metrics["int_rate"],
            "season_sack_rate": metrics["sack_rate"],
            "season_passer_rating": metrics["passer_rating"],
            "season_success_rate": metrics["success_rate"],
        }

    def _empty_season_metrics(self) -> Dict[str, float]:
        """Return empty season metrics dict."""
        return {
            "season_games": 0,
            "season_dropbacks": 0,
            "season_epa_per_dropback": 0.0,
            "season_cpoe": 0.0,
            "season_completion_pct": 0.0,
            "season_yards_per_attempt": 0.0,
            "season_air_yards_per_attempt": 0.0,
            "season_td_rate": 0.0,
            "season_int_rate": 0.0,
            "season_sack_rate": 0.0,
            "season_passer_rating": 0.0,
            "season_success_rate": 0.0,
        }

    def compute_deviation_features(
        self,
        career: QBCareerBaseline,
        season: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Compute deviation features comparing season to career baseline.

        The Mahomes insight: 2 INTs through Week 8 vs career avg of 8 is meaningful.
        These features capture such deviations.
        """
        features = {}

        # Raw deviations
        features["epa_vs_career"] = season["season_epa_per_dropback"] - career.epa_per_dropback
        features["cpoe_vs_career"] = season["season_cpoe"] - career.cpoe
        features["int_rate_vs_career"] = season["season_int_rate"] - career.int_rate
        features["td_rate_vs_career"] = season["season_td_rate"] - career.td_rate
        features["completion_pct_vs_career"] = season["season_completion_pct"] - career.completion_pct
        features["sack_rate_vs_career"] = season["season_sack_rate"] - career.sack_rate
        features["passer_rating_vs_career"] = season["season_passer_rating"] - career.passer_rating
        features["success_rate_vs_career"] = season["season_success_rate"] - career.success_rate

        # Z-scores (standardized deviations)
        # Positive z-score = performing better than career average
        if career.epa_std > 0:
            features["epa_zscore"] = features["epa_vs_career"] / career.epa_std
        else:
            features["epa_zscore"] = 0.0

        if career.cpoe_std > 0:
            features["cpoe_zscore"] = features["cpoe_vs_career"] / career.cpoe_std
        else:
            features["cpoe_zscore"] = 0.0

        if career.int_rate_std > 0:
            # Negative INT rate deviation = better (fewer INTs)
            # So negate for consistent interpretation (positive = better)
            features["int_rate_zscore"] = -features["int_rate_vs_career"] / career.int_rate_std
        else:
            features["int_rate_zscore"] = 0.0

        # Percentile rank (where does current season rank vs career games?)
        if len(career.game_epas) >= MIN_GAMES_FOR_STD:
            season_epa = season["season_epa_per_dropback"]
            epa_percentile = sum(1 for e in career.game_epas if e < season_epa) / len(career.game_epas) * 100
            features["epa_season_percentile"] = epa_percentile
        else:
            features["epa_season_percentile"] = 50.0

        if len(career.game_int_rates) >= MIN_GAMES_FOR_STD:
            season_int_rate = season["season_int_rate"]
            # Lower INT rate = better, so we flip
            int_percentile = sum(1 for r in career.game_int_rates if r > season_int_rate) / len(career.game_int_rates) * 100
            features["int_rate_season_percentile"] = int_percentile
        else:
            features["int_rate_season_percentile"] = 50.0

        # Performance indicators (binary flags)
        features["performing_above_career"] = 1 if features["epa_zscore"] > 0.5 else 0
        features["performing_below_career"] = 1 if features["epa_zscore"] < -0.5 else 0
        features["significantly_above_career"] = 1 if features["epa_zscore"] > 1.0 else 0
        features["significantly_below_career"] = 1 if features["epa_zscore"] < -1.0 else 0

        # INT rate flags (negative deviation = fewer INTs = better)
        features["int_rate_elevated"] = 1 if features["int_rate_vs_career"] > 0.01 else 0  # +1% INT rate
        features["int_rate_suppressed"] = 1 if features["int_rate_vs_career"] < -0.01 else 0

        return features

    def compute_trend_features(
        self,
        qb_id: str,
        season: int,
        before_week: int,
        before_date: pd.Timestamp
    ) -> Dict[str, float]:
        """
        Compute performance trend features (last 4 games).

        Captures hot/cold streaks and momentum.
        """
        qb_plays = self._get_qb_plays(
            qb_id, before_date,
            season_filter=season,
            before_week_filter=before_week
        )

        if len(qb_plays) == 0:
            return self._empty_trend_features()

        # Aggregate by game
        game_col = "game_id" if "game_id" in qb_plays.columns else "old_game_id"
        if game_col not in qb_plays.columns:
            return self._empty_trend_features()

        game_metrics = []
        for game_id, game_plays in qb_plays.groupby(game_col):
            week = game_plays["week"].iloc[0] if "week" in game_plays.columns else 0
            gm = self._compute_play_metrics(game_plays)
            if gm["dropbacks"] >= 5:
                game_metrics.append({
                    "week": week,
                    "epa": gm["epa_per_dropback"],
                    "int_rate": gm["int_rate"],
                    "success_rate": gm["success_rate"],
                })

        if len(game_metrics) < 2:
            return self._empty_trend_features()

        # Sort by week
        game_metrics = sorted(game_metrics, key=lambda x: x["week"])

        # Last 4 games
        last_4 = game_metrics[-4:] if len(game_metrics) >= 4 else game_metrics
        season_avg_epa = np.mean([g["epa"] for g in game_metrics])
        last_4_avg_epa = np.mean([g["epa"] for g in last_4])

        # Compute trend (slope) using linear regression
        if len(last_4) >= 2:
            x = np.arange(len(last_4))
            y = np.array([g["epa"] for g in last_4])
            slope = np.polyfit(x, y, 1)[0]
        else:
            slope = 0.0

        # Recent INT rate
        last_4_int_rate = np.mean([g["int_rate"] for g in last_4])

        return {
            "epa_trend_last_4": slope,
            "performance_improving": 1 if slope > 0.02 else 0,
            "performance_declining": 1 if slope < -0.02 else 0,
            "last_4_avg_epa": last_4_avg_epa,
            "last_4_vs_season": last_4_avg_epa - season_avg_epa,
            "last_4_int_rate": last_4_int_rate,
            "recent_hot_streak": 1 if last_4_avg_epa > season_avg_epa + 0.05 else 0,
            "recent_cold_streak": 1 if last_4_avg_epa < season_avg_epa - 0.05 else 0,
        }

    def _empty_trend_features(self) -> Dict[str, float]:
        """Return empty trend features dict."""
        return {
            "epa_trend_last_4": 0.0,
            "performance_improving": 0,
            "performance_declining": 0,
            "last_4_avg_epa": 0.0,
            "last_4_vs_season": 0.0,
            "last_4_int_rate": 0.0,
            "recent_hot_streak": 0,
            "recent_cold_streak": 0,
        }

    def compute_luck_features(
        self,
        career: QBCareerBaseline,
        season: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Compute luck/regression indicator features.

        These identify QBs likely to regress toward career mean.
        Without turnover-worthy play data, we use statistical proxies.
        """
        features = {}

        # INT luck: compare season INT rate to career
        # Low season INT rate vs career = possibly lucky (will regress up)
        # High season INT rate vs career = possibly unlucky (will regress down)

        int_rate_diff = season["season_int_rate"] - career.int_rate

        # League average INT rate is ~2.5%
        league_avg_int_rate = 0.025

        # Deviation from league average
        features["int_vs_league_avg"] = season["season_int_rate"] - league_avg_int_rate

        # Regression indicators based on z-scores
        # If INT rate is significantly below career average, likely to regress up
        if career.int_rate_std > 0:
            int_zscore = int_rate_diff / career.int_rate_std
            features["likely_int_regression_up"] = 1 if int_zscore < -1.0 else 0
            features["likely_int_regression_down"] = 1 if int_zscore > 1.0 else 0
        else:
            features["likely_int_regression_up"] = 0
            features["likely_int_regression_down"] = 0

        # EPA regression
        epa_diff = season["season_epa_per_dropback"] - career.epa_per_dropback
        if career.epa_std > 0:
            epa_zscore = epa_diff / career.epa_std
            features["likely_epa_regression_down"] = 1 if epa_zscore > 1.5 else 0
            features["likely_epa_regression_up"] = 1 if epa_zscore < -1.5 else 0
        else:
            features["likely_epa_regression_down"] = 0
            features["likely_epa_regression_up"] = 0

        # Overall regression likelihood
        # Based on how far from career norm
        features["regression_magnitude"] = abs(epa_diff) if career.epa_std > 0 else 0.0

        return features

    def generate_game_features(
        self,
        game_id: str,
        season: int,
        week: int,
        team: str,
        qb_id: str,
        game_date: Optional[pd.Timestamp] = None
    ) -> Dict[str, float]:
        """
        Generate all QB deviation features for a single game.

        This is the main entry point for feature generation.

        Args:
            game_id: Game identifier
            season: Season year
            week: Week number
            team: Team abbreviation
            qb_id: QB player ID
            game_date: Game date (for filtering). If None, will look up from games_df.

        Returns:
            Dictionary of features (all prefixed with 'qb_')
        """
        features = {}

        # Get game date if not provided
        if game_date is None:
            game_date = self.game_dates.get(game_id)
            if game_date is None:
                logger.warning(f"No date found for game {game_id}")
                return self._empty_all_features()

        # Career baseline
        career = self.get_career_baseline(qb_id, game_date)
        if career is None:
            logger.debug(f"Insufficient career data for QB {qb_id}")
            return self._empty_all_features()

        # Add career baseline features
        features["qb_career_games"] = career.games_started
        features["qb_career_dropbacks"] = career.dropbacks
        features["qb_career_epa"] = career.epa_per_dropback
        features["qb_career_cpoe"] = career.cpoe
        features["qb_career_int_rate"] = career.int_rate
        features["qb_career_td_rate"] = career.td_rate
        features["qb_career_completion_pct"] = career.completion_pct
        features["qb_career_sack_rate"] = career.sack_rate
        features["qb_career_passer_rating"] = career.passer_rating
        features["qb_career_success_rate"] = career.success_rate

        # Season metrics
        season_metrics = self.get_season_metrics(qb_id, season, week, game_date)
        for k, v in season_metrics.items():
            features[f"qb_{k}"] = v

        # Deviation features
        deviation = self.compute_deviation_features(career, season_metrics)
        for k, v in deviation.items():
            features[f"qb_{k}"] = v

        # Trend features
        trend = self.compute_trend_features(qb_id, season, week, game_date)
        for k, v in trend.items():
            features[f"qb_{k}"] = v

        # Luck/regression features
        luck = self.compute_luck_features(career, season_metrics)
        for k, v in luck.items():
            features[f"qb_{k}"] = v

        return features

    def _empty_all_features(self) -> Dict[str, float]:
        """Return empty feature dict with all expected keys."""
        features = {}

        # Career features
        for key in ["career_games", "career_dropbacks", "career_epa", "career_cpoe",
                    "career_int_rate", "career_td_rate", "career_completion_pct",
                    "career_sack_rate", "career_passer_rating", "career_success_rate"]:
            features[f"qb_{key}"] = 0.0

        # Season features
        for key, v in self._empty_season_metrics().items():
            features[f"qb_{key}"] = v

        # Deviation features
        for key in ["epa_vs_career", "cpoe_vs_career", "int_rate_vs_career",
                    "td_rate_vs_career", "completion_pct_vs_career", "sack_rate_vs_career",
                    "passer_rating_vs_career", "success_rate_vs_career",
                    "epa_zscore", "cpoe_zscore", "int_rate_zscore",
                    "epa_season_percentile", "int_rate_season_percentile",
                    "performing_above_career", "performing_below_career",
                    "significantly_above_career", "significantly_below_career",
                    "int_rate_elevated", "int_rate_suppressed"]:
            features[f"qb_{key}"] = 0.0

        # Trend features
        for key, v in self._empty_trend_features().items():
            features[f"qb_{key}"] = v

        # Luck features
        for key in ["int_vs_league_avg", "likely_int_regression_up",
                    "likely_int_regression_down", "likely_epa_regression_down",
                    "likely_epa_regression_up", "regression_magnitude"]:
            features[f"qb_{key}"] = 0.0

        return features


def generate_qb_deviation_features(
    games_df: Optional[pd.DataFrame] = None,
    pbp_df: Optional[pd.DataFrame] = None,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Main function to generate QB deviation features for all games.

    Args:
        games_df: Games DataFrame. If None, loads from default location.
        pbp_df: Play-by-play DataFrame. If None, loads from default location.
        output_path: Path to save features. If None, uses default.

    Returns:
        DataFrame with QB deviation features per team per game
    """
    project_root = Path(__file__).parent.parent.parent

    if games_df is None:
        games_path = project_root / "data" / "nfl" / "staged" / "games.parquet"
        if not games_path.exists():
            raise FileNotFoundError(f"games.parquet not found at {games_path}")
        games_df = pd.read_parquet(games_path)

    if output_path is None:
        output_path = project_root / "data" / "nfl" / "processed" / "qb_deviation_features.parquet"

    logger.info("Initializing QBDeviationFeatureGenerator")
    generator = QBDeviationFeatureGenerator(pbp_df=pbp_df, games_df=games_df)

    # We need to identify the starting QB for each team in each game
    # This requires play-by-play data
    from features.nfl.qb_features import identify_starting_qb

    # Load plays for QB identification
    plays_path = project_root / "data" / "nfl" / "staged" / "plays.parquet"
    if plays_path.exists():
        plays_df = pd.read_parquet(plays_path)
    else:
        logger.warning("plays.parquet not found, using raw PBP for QB identification")
        plays_df = generator.pbp_df

    result_rows = []
    games_df["date"] = pd.to_datetime(games_df["date"])

    logger.info(f"Generating QB deviation features for {len(games_df)} games")

    for idx, game in games_df.iterrows():
        game_id = game["game_id"]
        season = game["season"]
        week = game["week"]
        game_date = game["date"]
        home_team = game["home_team"]
        away_team = game["away_team"]

        # Identify starting QBs
        home_qb = identify_starting_qb(plays_df, game_id, home_team)
        away_qb = identify_starting_qb(plays_df, game_id, away_team)

        # Generate features for home team
        home_features = {"game_id": game_id, "team": home_team, "is_home": True}
        if home_qb:
            qb_features = generator.generate_game_features(
                game_id, season, week, home_team, home_qb, game_date
            )
            home_features.update(qb_features)
        else:
            home_features.update(generator._empty_all_features())
        home_features["qb_id"] = home_qb
        result_rows.append(home_features)

        # Generate features for away team
        away_features = {"game_id": game_id, "team": away_team, "is_home": False}
        if away_qb:
            qb_features = generator.generate_game_features(
                game_id, season, week, away_team, away_qb, game_date
            )
            away_features.update(qb_features)
        else:
            away_features.update(generator._empty_all_features())
        away_features["qb_id"] = away_qb
        result_rows.append(away_features)

        if (idx + 1) % 100 == 0:
            logger.info(f"Processed {idx + 1}/{len(games_df)} games")

    result_df = pd.DataFrame(result_rows)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_parquet(output_path, index=False)
    logger.info(f"Saved QB deviation features to {output_path}")
    logger.info(f"Generated features for {len(result_df)} team-games")

    return result_df


if __name__ == "__main__":
    df = generate_qb_deviation_features()
    print(f"\nGenerated QB deviation features for {len(df)} team-games")

    # Show feature columns
    feature_cols = [c for c in df.columns if c.startswith("qb_")]
    print(f"\nFeature columns ({len(feature_cols)}):")
    for col in sorted(feature_cols):
        print(f"  {col}")

    # Show sample
    print(f"\nSample features:")
    sample_cols = ["game_id", "team", "qb_id", "qb_career_epa", "qb_season_epa_per_dropback",
                   "qb_epa_vs_career", "qb_epa_zscore", "qb_int_rate_vs_career"]
    available_cols = [c for c in sample_cols if c in df.columns]
    print(df[available_cols].head(10))
