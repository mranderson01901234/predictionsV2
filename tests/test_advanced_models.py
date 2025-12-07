"""
Tests for Advanced Models (FT-Transformer, TabNet, Ensemble)

Tests cover:
1. Model import and basic training
2. Prediction interface (fit, predict_proba)
3. Ensemble behavior
4. Feature registry
5. Calibration wrapper
"""

import sys
from pathlib import Path
import pytest
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================================
# Fixtures for synthetic data
# ============================================================================

@pytest.fixture
def synthetic_data():
    """Create small synthetic dataset for fast testing."""
    np.random.seed(42)

    n_samples = 200
    n_features = 10

    # Create features with some signal
    X = np.random.randn(n_samples, n_features).astype(np.float32)

    # Target based on linear combination + noise
    weights = np.random.randn(n_features)
    logits = X @ weights
    probs = 1 / (1 + np.exp(-logits))
    y = (np.random.rand(n_samples) < probs).astype(int)

    # Split
    train_idx = slice(0, 120)
    val_idx = slice(120, 160)
    test_idx = slice(160, 200)

    return {
        'X_train': X[train_idx],
        'y_train': y[train_idx],
        'X_val': X[val_idx],
        'y_val': y[val_idx],
        'X_test': X[test_idx],
        'y_test': y[test_idx],
        'feature_names': [f"feature_{i}" for i in range(n_features)],
    }


@pytest.fixture
def synthetic_probs():
    """Create synthetic probabilities for calibration testing."""
    np.random.seed(42)
    n = 100

    # Raw probabilities (slightly miscalibrated)
    probs = np.random.beta(2, 2, n)

    # Labels correlated with probs
    y = (np.random.rand(n) < probs * 0.8 + 0.1).astype(int)

    return probs, y


# ============================================================================
# FT-Transformer Tests
# ============================================================================

class TestFTTransformer:
    """Tests for FT-Transformer model."""

    def test_import(self):
        """Test that FTTransformerModel can be imported."""
        from models.architectures.ft_transformer import FTTransformerModel
        assert FTTransformerModel is not None

    def test_initialization(self):
        """Test model initialization with default parameters."""
        from models.architectures.ft_transformer import FTTransformerModel

        model = FTTransformerModel()
        assert model.d_model == 64
        assert model.n_heads == 4
        assert model.n_layers == 3

    def test_fit_runs(self, synthetic_data):
        """Test that fit runs without error on small data."""
        from models.architectures.ft_transformer import FTTransformerModel

        model = FTTransformerModel(
            d_model=16,
            n_heads=2,
            n_layers=1,
            d_ff=32,
            epochs=2,
            patience=2,
        )

        # Should not raise
        model.fit(
            synthetic_data['X_train'],
            synthetic_data['y_train'],
            synthetic_data['X_val'],
            synthetic_data['y_val'],
        )

        assert model.model is not None
        assert model.n_features == 10

    def test_predict_proba_shape(self, synthetic_data):
        """Test predict_proba returns correct shape."""
        from models.architectures.ft_transformer import FTTransformerModel

        model = FTTransformerModel(d_model=16, n_heads=2, n_layers=1, epochs=2)
        model.fit(synthetic_data['X_train'], synthetic_data['y_train'])

        probs = model.predict_proba(synthetic_data['X_test'])

        assert probs.shape == (40,)
        assert probs.dtype == np.float64 or probs.dtype == np.float32

    def test_predict_proba_range(self, synthetic_data):
        """Test predict_proba returns values in [0, 1]."""
        from models.architectures.ft_transformer import FTTransformerModel

        model = FTTransformerModel(d_model=16, n_heads=2, n_layers=1, epochs=2)
        model.fit(synthetic_data['X_train'], synthetic_data['y_train'])

        probs = model.predict_proba(synthetic_data['X_test'])

        assert np.all(probs >= 0), "Probabilities should be >= 0"
        assert np.all(probs <= 1), "Probabilities should be <= 1"


# ============================================================================
# TabNet Tests
# ============================================================================

class TestTabNet:
    """Tests for TabNet model."""

    def test_import(self):
        """Test that TabNetModel can be imported."""
        from models.architectures.tabnet import TabNetModel
        assert TabNetModel is not None

    def test_initialization(self):
        """Test model initialization with default parameters."""
        from models.architectures.tabnet import TabNetModel

        model = TabNetModel()
        assert model.n_d == 8
        assert model.n_a == 8
        assert model.n_steps == 3

    def test_fit_runs(self, synthetic_data):
        """Test that fit runs without error on small data."""
        from models.architectures.tabnet import TabNetModel

        model = TabNetModel(
            n_d=4,
            n_a=4,
            n_steps=2,
            epochs=2,
            patience=2,
        )

        model.fit(
            synthetic_data['X_train'],
            synthetic_data['y_train'],
            synthetic_data['X_val'],
            synthetic_data['y_val'],
        )

        assert model.model is not None
        assert model.n_features == 10

    def test_predict_proba_shape(self, synthetic_data):
        """Test predict_proba returns correct shape."""
        from models.architectures.tabnet import TabNetModel

        model = TabNetModel(n_d=4, n_a=4, n_steps=2, epochs=2)
        model.fit(synthetic_data['X_train'], synthetic_data['y_train'])

        probs = model.predict_proba(synthetic_data['X_test'])

        assert probs.shape == (40,)

    def test_predict_proba_range(self, synthetic_data):
        """Test predict_proba returns values in [0, 1]."""
        from models.architectures.tabnet import TabNetModel

        model = TabNetModel(n_d=4, n_a=4, n_steps=2, epochs=2)
        model.fit(synthetic_data['X_train'], synthetic_data['y_train'])

        probs = model.predict_proba(synthetic_data['X_test'])

        assert np.all(probs >= 0), "Probabilities should be >= 0"
        assert np.all(probs <= 1), "Probabilities should be <= 1"


# ============================================================================
# Stacking Ensemble Tests
# ============================================================================

class TestStackingEnsemble:
    """Tests for Stacking Ensemble."""

    def test_import(self):
        """Test that StackingEnsemble can be imported."""
        from models.architectures.stacking_ensemble import StackingEnsemble
        assert StackingEnsemble is not None

    def test_initialization(self):
        """Test ensemble initialization."""
        from models.architectures.stacking_ensemble import StackingEnsemble

        ensemble = StackingEnsemble()
        assert ensemble.meta_model_type == "logistic"
        assert len(ensemble.base_models) == 0

    def test_add_model(self, synthetic_data):
        """Test adding base models to ensemble."""
        from models.architectures.stacking_ensemble import StackingEnsemble
        from models.architectures.logistic_regression import LogisticRegressionModel

        ensemble = StackingEnsemble()

        model1 = LogisticRegressionModel()
        model1.fit(synthetic_data['X_train'], synthetic_data['y_train'])

        ensemble.add_model("logit", model1)

        assert "logit" in ensemble.base_models
        assert len(ensemble.base_models) == 1

    def test_fit_runs(self, synthetic_data):
        """Test ensemble fit with multiple base models."""
        from models.architectures.stacking_ensemble import StackingEnsemble
        from models.architectures.logistic_regression import LogisticRegressionModel
        from models.architectures.gradient_boosting import GradientBoostingModel

        # Train base models
        logit = LogisticRegressionModel()
        logit.fit(synthetic_data['X_train'], synthetic_data['y_train'])

        gbm = GradientBoostingModel(n_estimators=10)
        gbm.fit(synthetic_data['X_train'], synthetic_data['y_train'])

        # Create and fit ensemble
        ensemble = StackingEnsemble(
            base_models={"logit": logit, "gbm": gbm},
            meta_model_type="logistic",
        )

        ensemble.fit(synthetic_data['X_val'], synthetic_data['y_val'])

        assert ensemble.meta_model is not None
        assert ensemble.n_base_models == 2

    def test_predict_proba_shape(self, synthetic_data):
        """Test ensemble predict_proba returns correct shape."""
        from models.architectures.stacking_ensemble import StackingEnsemble
        from models.architectures.logistic_regression import LogisticRegressionModel

        logit = LogisticRegressionModel()
        logit.fit(synthetic_data['X_train'], synthetic_data['y_train'])

        ensemble = StackingEnsemble(base_models={"logit": logit})
        ensemble.fit(synthetic_data['X_val'], synthetic_data['y_val'])

        probs = ensemble.predict_proba(synthetic_data['X_test'])

        assert probs.shape == (40,)

    def test_predict_proba_range(self, synthetic_data):
        """Test ensemble probabilities are in [0, 1]."""
        from models.architectures.stacking_ensemble import StackingEnsemble
        from models.architectures.logistic_regression import LogisticRegressionModel

        logit = LogisticRegressionModel()
        logit.fit(synthetic_data['X_train'], synthetic_data['y_train'])

        ensemble = StackingEnsemble(base_models={"logit": logit})
        ensemble.fit(synthetic_data['X_val'], synthetic_data['y_val'])

        probs = ensemble.predict_proba(synthetic_data['X_test'])

        assert np.all(probs >= 0)
        assert np.all(probs <= 1)

    def test_predict_all(self, synthetic_data):
        """Test predict_all returns predictions from all models."""
        from models.architectures.stacking_ensemble import StackingEnsemble
        from models.architectures.logistic_regression import LogisticRegressionModel
        from models.architectures.gradient_boosting import GradientBoostingModel

        logit = LogisticRegressionModel()
        logit.fit(synthetic_data['X_train'], synthetic_data['y_train'])

        gbm = GradientBoostingModel(n_estimators=10)
        gbm.fit(synthetic_data['X_train'], synthetic_data['y_train'])

        ensemble = StackingEnsemble(base_models={"logit": logit, "gbm": gbm})
        ensemble.fit(synthetic_data['X_val'], synthetic_data['y_val'])

        all_preds = ensemble.predict_all(synthetic_data['X_test'])

        assert "logit" in all_preds
        assert "gbm" in all_preds
        assert "ensemble" in all_preds
        assert all_preds["ensemble"].shape == (40,)


# ============================================================================
# Feature Registry Tests
# ============================================================================

class TestFeatureRegistry:
    """Tests for Feature Registry."""

    def test_import(self):
        """Test that FeatureRegistry can be imported."""
        from features.registry import FeatureRegistry
        assert FeatureRegistry is not None

    def test_get_all_features(self):
        """Test getting all feature definitions."""
        from features.registry import FeatureRegistry

        features = FeatureRegistry.get_all_feature_definitions()

        assert len(features) > 0
        assert "win_rate_last" in features

    def test_get_features_by_group(self):
        """Test filtering features by group."""
        from features.registry import FeatureRegistry, FeatureGroup

        form_features = FeatureRegistry.get_features_by_group(FeatureGroup.FORM)

        assert len(form_features) > 0
        for feat in form_features.values():
            assert feat.group == FeatureGroup.FORM

    def test_get_feature_columns(self):
        """Test getting feature column names."""
        from features.registry import FeatureRegistry

        columns = FeatureRegistry.get_feature_columns("baseline")

        assert len(columns) > 0
        # Should include home/away prefixed columns
        assert any("home_" in c for c in columns)
        assert any("away_" in c for c in columns)

    def test_get_exclude_columns(self):
        """Test getting excluded column names."""
        from features.registry import FeatureRegistry

        exclude = FeatureRegistry.get_exclude_columns()

        assert "game_id" in exclude
        assert "season" in exclude
        assert "home_score" in exclude
        assert "away_score" in exclude

    def test_filter_feature_columns(self):
        """Test filtering columns."""
        from features.registry import FeatureRegistry, FeatureGroup

        all_cols = [
            "game_id", "season", "home_team", "away_team",
            "home_win_rate_last4", "away_win_rate_last4",
            "home_pdiff_last4", "away_pdiff_last4",
            "home_score", "away_score",
        ]

        filtered = FeatureRegistry.filter_feature_columns(all_cols)

        assert "game_id" not in filtered
        assert "season" not in filtered
        assert "home_score" not in filtered
        assert "home_win_rate_last4" in filtered
        assert "away_pdiff_last4" in filtered


# ============================================================================
# Calibration Tests
# ============================================================================

class TestCalibration:
    """Tests for probability calibration."""

    def test_platt_import(self):
        """Test PlattScaler import."""
        from models.calibration import PlattScaler
        assert PlattScaler is not None

    def test_isotonic_import(self):
        """Test IsotonicCalibrator import."""
        from models.calibration import IsotonicCalibrator
        assert IsotonicCalibrator is not None

    def test_platt_fit_transform(self, synthetic_probs):
        """Test Platt scaling fit and transform."""
        from models.calibration import PlattScaler

        probs, y = synthetic_probs

        scaler = PlattScaler()
        scaler.fit(probs, y)

        calibrated = scaler.transform(probs)

        assert calibrated.shape == probs.shape
        assert np.all(calibrated >= 0)
        assert np.all(calibrated <= 1)

    def test_isotonic_fit_transform(self, synthetic_probs):
        """Test isotonic calibration fit and transform."""
        from models.calibration import IsotonicCalibrator

        probs, y = synthetic_probs

        calibrator = IsotonicCalibrator()
        calibrator.fit(probs, y)

        calibrated = calibrator.transform(probs)

        assert calibrated.shape == probs.shape
        assert np.all(calibrated >= 0)
        assert np.all(calibrated <= 1)

    def test_calibrated_model_wrapper(self, synthetic_data):
        """Test CalibratedModel wrapper."""
        from models.calibration import CalibratedModel
        from models.architectures.logistic_regression import LogisticRegressionModel

        base = LogisticRegressionModel()

        calibrated = CalibratedModel(base_model=base, method="platt")
        calibrated.fit(
            synthetic_data['X_train'],
            synthetic_data['y_train'],
            X_cal=synthetic_data['X_val'],
            y_cal=synthetic_data['y_val'],
        )

        probs = calibrated.predict_proba(synthetic_data['X_test'])

        assert probs.shape == (40,)
        assert np.all(probs >= 0)
        assert np.all(probs <= 1)

    def test_calibrated_model_both_outputs(self, synthetic_data):
        """Test getting both raw and calibrated probabilities."""
        from models.calibration import CalibratedModel
        from models.architectures.logistic_regression import LogisticRegressionModel

        base = LogisticRegressionModel()

        calibrated = CalibratedModel(base_model=base, method="platt")
        calibrated.fit(
            synthetic_data['X_train'],
            synthetic_data['y_train'],
            X_cal=synthetic_data['X_val'],
            y_cal=synthetic_data['y_val'],
        )

        raw, cal = calibrated.predict_proba_both(synthetic_data['X_test'])

        assert raw.shape == (40,)
        assert cal.shape == (40,)
        # Should be different (calibration changed them)
        assert not np.allclose(raw, cal)

    def test_compute_calibration_metrics(self, synthetic_probs):
        """Test calibration metrics computation."""
        from models.calibration import compute_calibration_metrics

        probs, y = synthetic_probs

        metrics = compute_calibration_metrics(y, probs, n_bins=5)

        assert 'ece' in metrics
        assert 'mce' in metrics
        assert 'brier' in metrics
        assert metrics['ece'] >= 0
        assert metrics['mce'] >= 0
        assert metrics['brier'] >= 0


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests combining multiple components."""

    def test_full_pipeline_ft_transformer(self, synthetic_data):
        """Test full training pipeline with FT-Transformer."""
        from models.architectures.ft_transformer import FTTransformerModel
        from models.calibration import CalibratedModel

        # Train base model
        model = FTTransformerModel(d_model=16, n_heads=2, n_layers=1, epochs=3)
        model.fit(
            synthetic_data['X_train'],
            synthetic_data['y_train'],
            synthetic_data['X_val'],
            synthetic_data['y_val'],
        )

        # Apply calibration
        calibrated = CalibratedModel(base_model=model, method="platt")
        calibrated.fit_calibration(synthetic_data['X_val'], synthetic_data['y_val'])

        # Predict
        probs = calibrated.predict_proba(synthetic_data['X_test'])

        assert probs.shape == (40,)
        assert np.all(probs >= 0)
        assert np.all(probs <= 1)

    def test_full_ensemble_pipeline(self, synthetic_data):
        """Test full ensemble training pipeline."""
        from models.architectures.logistic_regression import LogisticRegressionModel
        from models.architectures.gradient_boosting import GradientBoostingModel
        from models.architectures.stacking_ensemble import StackingEnsemble
        from models.calibration import CalibratedModel

        # Train base models
        logit = LogisticRegressionModel()
        logit.fit(synthetic_data['X_train'], synthetic_data['y_train'])

        gbm = GradientBoostingModel(n_estimators=10)
        gbm.fit(synthetic_data['X_train'], synthetic_data['y_train'])

        # Create ensemble
        ensemble = StackingEnsemble(
            base_models={"logit": logit, "gbm": gbm},
            meta_model_type="logistic",
        )
        ensemble.fit(synthetic_data['X_val'], synthetic_data['y_val'])

        # Calibrate
        calibrated = CalibratedModel(base_model=ensemble, method="platt")
        calibrated.fit_calibration(synthetic_data['X_val'], synthetic_data['y_val'])

        # Predict
        probs = calibrated.predict_proba(synthetic_data['X_test'])

        assert probs.shape == (40,)
        assert np.all(probs >= 0)
        assert np.all(probs <= 1)
