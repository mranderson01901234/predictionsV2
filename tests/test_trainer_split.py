"""
Tests for Trainer Data Splitting
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.training.trainer import split_by_season


class TestDataSplitting:
    """Test data splitting functionality."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data with multiple seasons."""
        n_per_season = 10
        seasons = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]
        
        data = []
        for season in seasons:
            for i in range(n_per_season):
                data.append({
                    "season": season,
                    "week": (i % 18) + 1,
                    "feature1": np.random.randn(),
                    "feature2": np.random.randn(),
                })
        
        df = pd.DataFrame(data)
        X = df[["feature1", "feature2"]]
        y = pd.Series(np.random.randint(0, 2, len(df)))
        
        return X, y, df
    
    def test_correct_seasons_in_splits(self, sample_data):
        """Test that splits contain correct seasons."""
        X, y, df = sample_data
        
        train_seasons = [2015, 2016, 2017, 2018, 2019, 2020, 2021]
        validation_season = 2022
        test_season = 2023
        
        X_train, y_train, X_val, y_val, X_test, y_test = split_by_season(
            X, y, df, train_seasons, validation_season, test_season
        )
        
        # Check train set
        train_seasons_actual = df[df.index.isin(X_train.index)]["season"].unique()
        assert set(train_seasons_actual) == set(train_seasons), \
            f"Train seasons mismatch: {train_seasons_actual} vs {train_seasons}"
        
        # Check validation set
        val_seasons_actual = df[df.index.isin(X_val.index)]["season"].unique()
        assert set(val_seasons_actual) == {validation_season}, \
            f"Validation season mismatch: {val_seasons_actual} vs {validation_season}"
        
        # Check test set
        test_seasons_actual = df[df.index.isin(X_test.index)]["season"].unique()
        assert set(test_seasons_actual) == {test_season}, \
            f"Test season mismatch: {test_seasons_actual} vs {test_season}"
    
    def test_no_leakage(self, sample_data):
        """Test that there's no data leakage between splits."""
        X, y, df = sample_data
        
        train_seasons = [2015, 2016, 2017, 2018, 2019, 2020, 2021]
        validation_season = 2022
        test_season = 2023
        
        X_train, y_train, X_val, y_val, X_test, y_test = split_by_season(
            X, y, df, train_seasons, validation_season, test_season
        )
        
        # Check no overlap in indices
        train_indices = set(X_train.index)
        val_indices = set(X_val.index)
        test_indices = set(X_test.index)
        
        assert len(train_indices & val_indices) == 0, "Train and validation sets overlap!"
        assert len(train_indices & test_indices) == 0, "Train and test sets overlap!"
        assert len(val_indices & test_indices) == 0, "Validation and test sets overlap!"
    
    def test_all_data_accounted_for(self, sample_data):
        """Test that all data is accounted for in splits."""
        X, y, df = sample_data
        
        train_seasons = [2015, 2016, 2017, 2018, 2019, 2020, 2021]
        validation_season = 2022
        test_season = 2023
        
        X_train, y_train, X_val, y_val, X_test, y_test = split_by_season(
            X, y, df, train_seasons, validation_season, test_season
        )
        
        # Total should equal original (assuming all seasons are in one of the splits)
        total = len(X_train) + len(X_val) + len(X_test)
        assert total == len(X), f"Total split size {total} != original size {len(X)}"
    
    def test_validation_not_in_train(self, sample_data):
        """Test that validation season is not in train."""
        X, y, df = sample_data
        
        train_seasons = [2015, 2016, 2017, 2018, 2019, 2020, 2021]
        validation_season = 2022
        
        X_train, y_train, X_val, y_val, X_test, y_test = split_by_season(
            X, y, df, train_seasons, validation_season, 2023
        )
        
        train_seasons_actual = df[df.index.isin(X_train.index)]["season"].unique()
        assert validation_season not in train_seasons_actual, \
            "Validation season found in train set!"
    
    def test_test_not_in_train(self, sample_data):
        """Test that test season is not in train."""
        X, y, df = sample_data
        
        train_seasons = [2015, 2016, 2017, 2018, 2019, 2020, 2021]
        test_season = 2023
        
        X_train, y_train, X_val, y_val, X_test, y_test = split_by_season(
            X, y, df, train_seasons, 2022, test_season
        )
        
        train_seasons_actual = df[df.index.isin(X_train.index)]["season"].unique()
        assert test_season not in train_seasons_actual, \
            "Test season found in train set!"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

