"""
Feature Registry

Central registry defining all features with their metadata.
This is the single source of truth for feature definitions.

Usage:
    from features.registry import FeatureRegistry

    # Get all features
    features = FeatureRegistry.get_all_features()

    # Get features by group
    form_features = FeatureRegistry.get_features_by_group("form")

    # Get feature columns for a specific feature table
    columns = FeatureRegistry.get_feature_columns("baseline")
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class FeatureType(Enum):
    """Feature data types."""
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    BINARY = "binary"


class FeatureGroup(Enum):
    """Feature groupings for selection and analysis."""
    FORM = "form"           # Team form/performance metrics
    SCHEDULE = "schedule"   # Schedule-related features
    MARKET = "market"       # Market/betting features
    EPA = "epa"             # Expected Points Added metrics
    QB = "qb"               # Quarterback-specific features
    ROLLING = "rolling"     # Rolling window aggregations


@dataclass
class FeatureDefinition:
    """Definition of a single feature."""
    name: str
    feature_type: FeatureType
    group: FeatureGroup
    description: str
    is_rolling: bool = False
    window_sizes: List[int] = field(default_factory=list)
    home_away_split: bool = True  # True if prefixed with home_/away_
    source_table: str = "baseline"  # Which feature table this appears in

    def get_column_names(self) -> List[str]:
        """Get actual column names for this feature in the dataset."""
        if self.home_away_split:
            if self.is_rolling and self.window_sizes:
                # Rolling features have window suffix
                names = []
                for w in self.window_sizes:
                    names.extend([f"home_{self.name}{w}", f"away_{self.name}{w}"])
                return names
            else:
                return [f"home_{self.name}", f"away_{self.name}"]
        else:
            if self.is_rolling and self.window_sizes:
                return [f"{self.name}{w}" for w in self.window_sizes]
            return [self.name]


class FeatureRegistry:
    """
    Central registry for all feature definitions.

    This provides:
    - Feature metadata (type, group, description)
    - Feature selection by group
    - Validation of feature presence
    - Column name generation for feature tables
    """

    # Core baseline features (team form)
    _BASELINE_FEATURES: Dict[str, FeatureDefinition] = {
        "win_rate_last": FeatureDefinition(
            name="win_rate_last",
            feature_type=FeatureType.NUMERIC,
            group=FeatureGroup.FORM,
            description="Rolling win rate over last N games",
            is_rolling=True,
            window_sizes=[4, 8, 16],
            source_table="baseline",
        ),
        "pdiff_last": FeatureDefinition(
            name="pdiff_last",
            feature_type=FeatureType.NUMERIC,
            group=FeatureGroup.FORM,
            description="Rolling point differential over last N games",
            is_rolling=True,
            window_sizes=[4, 8, 16],
            source_table="baseline",
        ),
        "points_for_last": FeatureDefinition(
            name="points_for_last",
            feature_type=FeatureType.NUMERIC,
            group=FeatureGroup.FORM,
            description="Rolling points scored over last N games",
            is_rolling=True,
            window_sizes=[4, 8, 16],
            source_table="baseline",
        ),
        "points_against_last": FeatureDefinition(
            name="points_against_last",
            feature_type=FeatureType.NUMERIC,
            group=FeatureGroup.FORM,
            description="Rolling points allowed over last N games",
            is_rolling=True,
            window_sizes=[4, 8, 16],
            source_table="baseline",
        ),
        "turnover_diff_last": FeatureDefinition(
            name="turnover_diff_last",
            feature_type=FeatureType.NUMERIC,
            group=FeatureGroup.FORM,
            description="Rolling turnover differential over last N games",
            is_rolling=True,
            window_sizes=[4, 8, 16],
            source_table="baseline",
        ),
    }

    # EPA features (Phase 2)
    _EPA_FEATURES: Dict[str, FeatureDefinition] = {
        "epa_offensive_epa_per_play": FeatureDefinition(
            name="epa_offensive_epa_per_play",
            feature_type=FeatureType.NUMERIC,
            group=FeatureGroup.EPA,
            description="Offensive EPA per play",
            source_table="phase2",
        ),
        "epa_offensive_pass_epa": FeatureDefinition(
            name="epa_offensive_pass_epa",
            feature_type=FeatureType.NUMERIC,
            group=FeatureGroup.EPA,
            description="Offensive EPA on pass plays",
            source_table="phase2",
        ),
        "epa_offensive_run_epa": FeatureDefinition(
            name="epa_offensive_run_epa",
            feature_type=FeatureType.NUMERIC,
            group=FeatureGroup.EPA,
            description="Offensive EPA on run plays",
            source_table="phase2",
        ),
        "epa_defensive_epa_per_play_allowed": FeatureDefinition(
            name="epa_defensive_epa_per_play_allowed",
            feature_type=FeatureType.NUMERIC,
            group=FeatureGroup.EPA,
            description="Defensive EPA allowed per play",
            source_table="phase2",
        ),
        "epa_offensive_success_rate": FeatureDefinition(
            name="epa_offensive_success_rate",
            feature_type=FeatureType.NUMERIC,
            group=FeatureGroup.EPA,
            description="Offensive success rate (EPA > 0)",
            source_table="phase2",
        ),
    }

    # Rolling EPA features (Phase 2B)
    _ROLLING_EPA_FEATURES: Dict[str, FeatureDefinition] = {
        "roll_epa_off_epa_last": FeatureDefinition(
            name="roll_epa_off_epa_last",
            feature_type=FeatureType.NUMERIC,
            group=FeatureGroup.ROLLING,
            description="Rolling offensive EPA over last N games",
            is_rolling=True,
            window_sizes=[3, 5, 8],
            source_table="phase2b",
        ),
        "roll_epa_def_epa_allowed_last": FeatureDefinition(
            name="roll_epa_def_epa_allowed_last",
            feature_type=FeatureType.NUMERIC,
            group=FeatureGroup.ROLLING,
            description="Rolling defensive EPA allowed over last N games",
            is_rolling=True,
            window_sizes=[3, 5, 8],
            source_table="phase2b",
        ),
    }

    # QB features (Phase 2B)
    _QB_FEATURES: Dict[str, FeatureDefinition] = {
        "qb_qb_epa_per_dropback": FeatureDefinition(
            name="qb_qb_epa_per_dropback",
            feature_type=FeatureType.NUMERIC,
            group=FeatureGroup.QB,
            description="QB EPA per dropback",
            source_table="phase2b",
        ),
        "qb_qb_cpoe": FeatureDefinition(
            name="qb_qb_cpoe",
            feature_type=FeatureType.NUMERIC,
            group=FeatureGroup.QB,
            description="QB completion percentage over expected",
            source_table="phase2b",
        ),
        "qb_qb_sack_rate": FeatureDefinition(
            name="qb_qb_sack_rate",
            feature_type=FeatureType.NUMERIC,
            group=FeatureGroup.QB,
            description="QB sack rate",
            source_table="phase2b",
        ),
    }

    # Metadata columns to exclude from features
    _EXCLUDE_COLUMNS: Set[str] = {
        "game_id",
        "season",
        "week",
        "date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "home_win",
        "close_spread",
        "close_total",
        "open_spread",
        "open_total",
    }

    @classmethod
    def get_all_feature_definitions(cls) -> Dict[str, FeatureDefinition]:
        """Get all feature definitions."""
        all_features = {}
        all_features.update(cls._BASELINE_FEATURES)
        all_features.update(cls._EPA_FEATURES)
        all_features.update(cls._ROLLING_EPA_FEATURES)
        all_features.update(cls._QB_FEATURES)
        return all_features

    @classmethod
    def get_features_by_group(cls, group: FeatureGroup) -> Dict[str, FeatureDefinition]:
        """Get features belonging to a specific group."""
        all_features = cls.get_all_feature_definitions()
        return {
            name: feat for name, feat in all_features.items()
            if feat.group == group
        }

    @classmethod
    def get_features_by_table(cls, table: str) -> Dict[str, FeatureDefinition]:
        """Get features available in a specific feature table."""
        all_features = cls.get_all_feature_definitions()
        # Baseline features are available in all tables
        result = {}
        for name, feat in all_features.items():
            if feat.source_table == table or feat.source_table == "baseline":
                result[name] = feat
        return result

    @classmethod
    def get_feature_columns(cls, table: str = "baseline") -> List[str]:
        """
        Get list of feature column names for a specific feature table.

        Args:
            table: Feature table name ("baseline", "phase2", "phase2b")

        Returns:
            List of column names
        """
        features = cls.get_features_by_table(table)
        columns = []
        for feat in features.values():
            columns.extend(feat.get_column_names())
        return sorted(columns)

    @classmethod
    def get_feature_groups(cls, groups: List[FeatureGroup]) -> List[str]:
        """
        Get column names for features in specified groups.

        Args:
            groups: List of feature groups to include

        Returns:
            List of column names
        """
        columns = []
        for group in groups:
            features = cls.get_features_by_group(group)
            for feat in features.values():
                columns.extend(feat.get_column_names())
        return sorted(columns)

    @classmethod
    def get_exclude_columns(cls) -> Set[str]:
        """Get columns that should be excluded from features."""
        return cls._EXCLUDE_COLUMNS.copy()

    @classmethod
    def filter_feature_columns(cls, columns: List[str],
                                groups: Optional[List[FeatureGroup]] = None,
                                table: str = "baseline") -> List[str]:
        """
        Filter a list of columns to only include valid features.

        Args:
            columns: List of column names from a dataframe
            groups: Optional list of groups to filter by
            table: Feature table name

        Returns:
            Filtered list of feature columns
        """
        exclude = cls.get_exclude_columns()

        # Start with all columns that aren't metadata
        valid = [c for c in columns if c not in exclude]

        # If groups specified, further filter
        if groups:
            allowed = set(cls.get_feature_groups(groups))
            valid = [c for c in valid if c in allowed]

        return valid

    @classmethod
    def validate_features(cls, columns: List[str],
                          required_groups: Optional[List[FeatureGroup]] = None) -> bool:
        """
        Validate that required features are present.

        Args:
            columns: List of column names to validate
            required_groups: Groups that must have at least one feature present

        Returns:
            True if validation passes

        Raises:
            ValueError: If validation fails
        """
        if required_groups:
            for group in required_groups:
                group_cols = cls.get_feature_groups([group])
                if not any(c in columns for c in group_cols):
                    raise ValueError(f"No features from group '{group.value}' found in columns")
        return True


def get_feature_columns_for_model(
    df_columns: List[str],
    groups: Optional[List[str]] = None,
    exclude_groups: Optional[List[str]] = None,
) -> List[str]:
    """
    Convenience function to get feature columns for model training.

    Args:
        df_columns: Column names from dataframe
        groups: Optional list of group names to include (e.g., ["form", "epa"])
        exclude_groups: Optional list of group names to exclude

    Returns:
        List of feature column names
    """
    exclude = FeatureRegistry.get_exclude_columns()
    features = [c for c in df_columns if c not in exclude]

    if groups:
        group_enums = [FeatureGroup(g) for g in groups]
        allowed = set(FeatureRegistry.get_feature_groups(group_enums))
        features = [c for c in features if c in allowed]

    if exclude_groups:
        exclude_enums = [FeatureGroup(g) for g in exclude_groups]
        excluded = set(FeatureRegistry.get_feature_groups(exclude_enums))
        features = [c for c in features if c not in excluded]

    return features
