# Advanced Models Documentation

This document describes the advanced tabular models available in the NFL prediction pipeline.

## Overview

The pipeline supports several model types:

| Model | Description | Best For |
|-------|-------------|----------|
| Logistic Regression | Simple baseline | Interpretability, fast training |
| Gradient Boosting (XGBoost) | Tree-based ensemble | Strong baselines, feature importance |
| FT-Transformer | Transformer for tabular data | Complex feature interactions |
| TabNet | Attention-based neural network | Interpretable deep learning |
| Stacking Ensemble | Combines multiple models | Maximum predictive power |

## Model Types

### FT-Transformer

FT-Transformer (Feature Tokenizer Transformer) applies transformer architecture to tabular data.
Each feature is tokenized and processed through attention layers.

**Key features:**
- Handles numeric features through learned embeddings
- Captures complex feature interactions via self-attention
- Configurable depth, width, and attention heads

**Configuration (`config/models/nfl_ft_transformer.yaml`):**
```yaml
architecture:
  d_model: 64      # Embedding dimension
  n_heads: 4       # Attention heads
  n_layers: 3      # Transformer layers
  d_ff: 256        # Feedforward dimension
  dropout: 0.1

training:
  learning_rate: 1e-4
  batch_size: 64
  epochs: 100
  patience: 15
```

**Usage:**
```bash
python -m models.training.train_advanced --model ft_transformer
```

### TabNet

TabNet uses sequential attention to select features at each decision step,
providing interpretability through feature importance scores.

**Key features:**
- Sequential attention mechanism
- Built-in feature selection
- Sparse, interpretable predictions

**Configuration (`config/models/nfl_tabnet.yaml`):**
```yaml
architecture:
  n_d: 8           # Decision layer width
  n_a: 8           # Attention embedding width
  n_steps: 3       # Decision steps
  gamma: 1.3       # Feature reusage coefficient
  lambda_sparse: 1e-3

training:
  learning_rate: 2e-2
  batch_size: 256
  epochs: 100
```

**Usage:**
```bash
python -m models.training.train_advanced --model tabnet
```

### Stacking Ensemble

The stacking ensemble combines predictions from multiple base models
using a meta-learner (logistic regression or MLP).

**Key features:**
- Combines diverse model types
- Meta-model learns optimal weighting
- Can include original features in stacking

**Configuration (`config/models/nfl_ensemble_v1.yaml`):**
```yaml
base_models:
  logistic:
    type: logistic_regression
  gbm:
    type: gradient_boosting
  ft_transformer:
    type: ft_transformer
  tabnet:
    type: tabnet

meta_model:
  type: logistic   # or 'mlp'
```

**Usage:**
```bash
python -m models.training.train_advanced --model ensemble
```

## Calibration

All models support probability calibration to improve reliability:

- **Platt Scaling**: Logistic regression on raw probabilities
- **Isotonic Regression**: Non-parametric monotonic calibration

Calibration is applied automatically when `calibration.enabled: true` in the config.

```python
from models.calibration import CalibratedModel

calibrated = CalibratedModel(base_model=model, method="platt")
calibrated.fit_calibration(X_val, y_val)
probs = calibrated.predict_proba(X_test)
```

## Feature Registry

The feature registry (`features/registry.py`) provides a centralized definition
of all features with metadata:

```python
from features.registry import FeatureRegistry, FeatureGroup

# Get all feature definitions
features = FeatureRegistry.get_all_feature_definitions()

# Get features by group
form_features = FeatureRegistry.get_features_by_group(FeatureGroup.FORM)

# Get column names for a feature table
columns = FeatureRegistry.get_feature_columns("baseline")
```

**Feature Groups:**
- `form`: Team form/performance metrics (win rate, point differential)
- `epa`: Expected Points Added metrics
- `qb`: Quarterback-specific features
- `rolling`: Rolling window aggregations
- `schedule`: Schedule-related features
- `market`: Betting market features

## Training Pipeline

### Quick Start

```bash
# Train all advanced models
python -m models.training.train_advanced --model ft_transformer
python -m models.training.train_advanced --model tabnet
python -m models.training.train_advanced --model ensemble
```

### With Custom Config

```bash
python -m models.training.train_advanced \
    --model ft_transformer \
    --config config/models/my_custom_config.yaml \
    --output-dir models/artifacts/my_experiment
```

### Programmatic Usage

```python
from models.architectures import FTTransformerModel, TabNetModel, StackingEnsemble

# Train FT-Transformer
ft_model = FTTransformerModel(d_model=64, n_layers=3)
ft_model.fit(X_train, y_train, X_val, y_val)

# Train TabNet
tabnet_model = TabNetModel(n_steps=3)
tabnet_model.fit(X_train, y_train, X_val, y_val)

# Create ensemble
ensemble = StackingEnsemble(
    base_models={"ft": ft_model, "tabnet": tabnet_model},
    meta_model_type="logistic"
)
ensemble.fit(X_val, y_val)

# Predict
probs = ensemble.predict_proba(X_test)
```

## Model Artifacts

Trained models are saved to `models/artifacts/`:

```
models/artifacts/
├── nfl_baseline/           # Baseline models
│   ├── logit.pkl
│   ├── gbm.pkl
│   └── ensemble.json
└── nfl_advanced/           # Advanced models
    ├── ft_transformer.pkl
    ├── tabnet.pkl
    └── ensemble_v1.pkl
```

## Testing

Run the advanced model tests:

```bash
pytest tests/test_advanced_models.py -v
```

## Dependencies

The advanced models require:

- `torch>=2.0.0`: PyTorch for deep learning
- `pytorch-tabnet>=4.0`: TabNet implementation (optional, has fallback)

Install with:

```bash
pip install -r requirements.txt
```
