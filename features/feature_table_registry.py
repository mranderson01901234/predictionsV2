"""
Feature Table Registry

Centralized mapping and validation for feature table names to file paths.
This is the single source of truth for feature table locations.
"""

from pathlib import Path
from typing import Dict
import logging

logger = logging.getLogger(__name__)


# Base directory for processed features
_BASE_DIR = Path(__file__).parent.parent / "data" / "nfl" / "processed"


# Authoritative mapping: feature table name -> filename
_FEATURE_TABLE_MAP: Dict[str, str] = {
    "baseline": "game_features_baseline.parquet",
    "phase2": "game_features_phase2.parquet",
    "phase2b": "game_features_phase2b.parquet",
    "phase3": "game_features_phase3.parquet",  # Phase 1-3 integrated features
    "full": "game_features_phase3.parquet",  # Alias for phase3
}


def get_feature_table_path(name: str, base_dir: Path = None) -> Path:
    """
    Get the full path to a feature table file.
    
    Args:
        name: Feature table name ("baseline", "phase2", "phase2b")
        base_dir: Optional base directory (defaults to data/nfl/processed)
    
    Returns:
        Path to the feature table file
    
    Raises:
        ValueError: If feature table name is unknown
    """
    if name not in _FEATURE_TABLE_MAP:
        valid_names = list(_FEATURE_TABLE_MAP.keys())
        raise ValueError(
            f"Unknown feature table name: '{name}'. "
            f"Valid options: {valid_names}"
        )
    
    filename = _FEATURE_TABLE_MAP[name]
    
    if base_dir is None:
        base_dir = _BASE_DIR
    
    return base_dir / filename


def validate_feature_table_exists(name: str, base_dir: Path = None) -> None:
    """
    Validate that a feature table file exists.
    
    Args:
        name: Feature table name ("baseline", "phase2", "phase2b")
        base_dir: Optional base directory (defaults to data/nfl/processed)
    
    Raises:
        ValueError: If feature table name is unknown
        FileNotFoundError: If feature table file does not exist
    """
    path = get_feature_table_path(name, base_dir)
    
    if not path.exists():
        raise FileNotFoundError(
            f"Feature table '{name}' not found: expected file {path}. "
            f"Please generate the feature table first or check the path."
        )


def list_feature_tables() -> list:
    """
    List all available feature table names.
    
    Returns:
        List of feature table names
    """
    return list(_FEATURE_TABLE_MAP.keys())


def get_feature_table_filename(name: str) -> str:
    """
    Get the filename for a feature table (without path).
    
    Args:
        name: Feature table name
    
    Returns:
        Filename string
    
    Raises:
        ValueError: If feature table name is unknown
    """
    if name not in _FEATURE_TABLE_MAP:
        valid_names = list(_FEATURE_TABLE_MAP.keys())
        raise ValueError(
            f"Unknown feature table name: '{name}'. "
            f"Valid options: {valid_names}"
        )
    
    return _FEATURE_TABLE_MAP[name]

