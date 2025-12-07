"""
Tests for Feature Table Registry
"""

import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from features.feature_table_registry import (
    get_feature_table_path,
    validate_feature_table_exists,
    list_feature_tables,
    get_feature_table_filename,
)


class TestFeatureTableRegistry:
    """Test feature table registry functions."""
    
    def test_list_feature_tables(self):
        """Test that we can list all feature tables."""
        tables = list_feature_tables()
        assert isinstance(tables, list)
        assert len(tables) > 0
        assert "baseline" in tables
        assert "phase2" in tables
        assert "phase2b" in tables
    
    def test_get_feature_table_filename(self):
        """Test getting filename for known feature tables."""
        assert get_feature_table_filename("baseline") == "game_features_baseline.parquet"
        assert get_feature_table_filename("phase2") == "game_features_phase2.parquet"
        assert get_feature_table_filename("phase2b") == "game_features_phase2b.parquet"
    
    def test_get_feature_table_filename_unknown(self):
        """Test that unknown feature table raises ValueError."""
        with pytest.raises(ValueError, match="Unknown feature table name"):
            get_feature_table_filename("unknown_table")
    
    def test_get_feature_table_path(self):
        """Test getting path for known feature tables."""
        baseline_path = get_feature_table_path("baseline")
        assert isinstance(baseline_path, Path)
        assert baseline_path.name == "game_features_baseline.parquet"
        assert "data/nfl/processed" in str(baseline_path)
        
        phase2_path = get_feature_table_path("phase2")
        assert phase2_path.name == "game_features_phase2.parquet"
        
        phase2b_path = get_feature_table_path("phase2b")
        assert phase2b_path.name == "game_features_phase2b.parquet"
    
    def test_get_feature_table_path_custom_base(self):
        """Test getting path with custom base directory."""
        custom_base = Path("/custom/path")
        path = get_feature_table_path("baseline", base_dir=custom_base)
        assert path.parent == custom_base
        assert path.name == "game_features_baseline.parquet"
    
    def test_get_feature_table_path_unknown(self):
        """Test that unknown feature table raises ValueError."""
        with pytest.raises(ValueError, match="Unknown feature table name"):
            get_feature_table_path("unknown_table")
    
    def test_validate_feature_table_exists_baseline(self):
        """Test that baseline table validation passes if file exists."""
        # Baseline table should exist
        try:
            validate_feature_table_exists("baseline")
        except FileNotFoundError:
            pytest.fail("Baseline feature table should exist")
    
    def test_validate_feature_table_exists_missing(self):
        """Test that missing feature table raises FileNotFoundError."""
        # phase2b should not exist (not generated yet)
        with pytest.raises(FileNotFoundError, match="Feature table 'phase2b' not found"):
            validate_feature_table_exists("phase2b")
    
    def test_validate_feature_table_exists_unknown(self):
        """Test that unknown feature table raises ValueError."""
        with pytest.raises(ValueError, match="Unknown feature table name"):
            validate_feature_table_exists("unknown_table")

