# Experiment Config Schema

## 1. Overview

### Purpose

Experiment configs are YAML files that define reproducible, end-to-end model training and evaluation runs. They serve as the single source of truth for:

- Model architecture and hyperparameters
- Feature selection and engineering
- Data splits (train/val/test seasons)
- Training configuration
- Output artifacts and evaluation settings

### Integration

- Configs are stored in `config/experiments/` (to be created) or `config/models/`
- Orchestration pipelines (`orchestration/pipelines/`) load configs and execute training/evaluation
- Feature registry (`features/feature_table_registry.py`) resolves feature table names to file paths
- Model registry (future) will track model versions and artifacts

### Naming Convention

- Config files: `{experiment_name}.yaml` (e.g., `nfl_baseline_gbm.yaml`, `nfl_ft_transformer_v1.yaml`)
- Experiment names: `{vertical}_{model_type}_{version}` (e.g., `nfl_gbm_v1`, `nfl_ensemble_v2`)

---

## 2. Top-Level Fields

### Required Fields

- `experiment_name`: Unique identifier for this experiment (string)
- `description`: Human-readable description of the experiment (string)
- `vertical`: Sport/domain identifier (string, e.g., `"nfl"`, `"nba"`)
- `model`: Model configuration block (dict, see Section 3)
- `features`: Feature configuration block (dict, see Section 4)
- `data`: Data configuration block (dict, see Section 5)
- `training`: Training configuration block (dict, see Section 6)
- `output`: Output configuration block (dict, see Section 7)

### Optional Fields

- `eval`: Evaluation configuration block (dict, see Section 8, optional for Phase A)
- `metadata`: Additional metadata (dict, optional)
  - `author`: Experiment creator (string)
  - `created_at`: Timestamp (string, ISO format)
  - `tags`: List of tags for organization (list of strings)

---

## 3. Model Block

### Structure

```yaml
model:
  type: <model_type>  # Required: "lr", "gbm", "ft_transformer", "tabnet", "ensemble"
  params:              # Required: nested dict of model-specific hyperparameters
    <param_name>: <value>
```

### Model Types

#### `lr` (Logistic Regression)
**Description**: Baseline linear model with L2 regularization.

**Required Params**:
- `C`: Inverse regularization strength (float, default: 1.0)
- `max_iter`: Maximum iterations (int, default: 1000)
- `random_state`: Random seed (int, default: 42)

**Optional Params**:
- `solver`: Solver algorithm (string, default: "lbfgs")
- `tol`: Convergence tolerance (float, default: 1e-4)

**Example**:
```yaml
model:
  type: lr
  params:
    C: 1.0
    max_iter: 1000
    random_state: 42
```

#### `gbm` (Gradient Boosting Machine)
**Description**: Gradient boosting (XGBoost or LightGBM).

**Required Params**:
- `n_estimators`: Number of boosting rounds (int, default: 100)
- `max_depth`: Maximum tree depth (int, default: 3)
- `learning_rate`: Learning rate (float, default: 0.1)
- `random_state`: Random seed (int, default: 42)

**Optional Params**:
- `subsample`: Row subsampling ratio (float, default: 1.0)
- `colsample_bytree`: Column subsampling ratio (float, default: 1.0)
- `min_child_weight`: Minimum sum of instance weight in child (float, default: 1.0)
- `reg_alpha`: L1 regularization (float, default: 0.0)
- `reg_lambda`: L2 regularization (float, default: 1.0)

**Example**:
```yaml
model:
  type: gbm
  params:
    n_estimators: 200
    max_depth: 5
    learning_rate: 0.05
    subsample: 0.8
    random_state: 42
```

#### `ft_transformer` (FT-Transformer)
**Description**: Feature Tokenizer + Transformer architecture for tabular data.

**Required Params**:
- `d_model`: Embedding dimension (int, default: 128)
- `n_layers`: Number of transformer layers (int, default: 3)
- `n_heads`: Number of attention heads (int, default: 4)
- `d_ff`: Feed-forward dimension (int, default: 512)
- `dropout`: Dropout rate (float, default: 0.1)
- `random_state`: Random seed (int, default: 42)

**Optional Params**:
- `activation`: Activation function (string, default: "gelu")
- `layer_norm_eps`: Layer normalization epsilon (float, default: 1e-5)
- `use_batch_norm`: Use batch normalization (bool, default: false)

**Example**:
```yaml
model:
  type: ft_transformer
  params:
    d_model: 256
    n_layers: 4
    n_heads: 8
    d_ff: 1024
    dropout: 0.2
    random_state: 42
```

#### `tabnet` (TabNet)
**Description**: TabNet architecture for tabular data.

**Required Params**:
- `n_d`: Dimension of decision layer (int, default: 64)
- `n_a`: Dimension of attention layer (int, default: 64)
- `n_steps`: Number of steps (int, default: 3)
- `gamma`: Relaxation parameter (float, default: 1.5)
- `lambda_sparse`: Sparsity regularization (float, default: 1e-3)
- `random_state`: Random seed (int, default: 42)

**Optional Params**:
- `optimizer_fn`: Optimizer class (string, default: "torch.optim.Adam")
- `optimizer_params`: Optimizer parameters (dict, default: `{lr: 0.02}`)
- `scheduler_fn`: Scheduler class (string, optional)
- `scheduler_params`: Scheduler parameters (dict, optional)
- `mask_type`: Mask type (string, default: "sparsemax")

**Example**:
```yaml
model:
  type: tabnet
  params:
    n_d: 128
    n_a: 128
    n_steps: 5
    gamma: 1.3
    lambda_sparse: 1e-4
    random_state: 42
```

#### `ensemble` (Ensemble)
**Description**: Weighted combination of multiple models.

**Required Params**:
- `models`: List of model configs (list of dicts)
  - Each element is a full model block (type + params)
- `weights`: List of weights for each model (list of floats, must sum to 1.0)

**Alternative Params** (if using pre-trained models):
- `model_paths`: List of paths to saved model artifacts (list of strings)
- `weights`: List of weights (list of floats)

**Example**:
```yaml
model:
  type: ensemble
  params:
    models:
      - type: lr
        params:
          C: 1.0
          max_iter: 1000
          random_state: 42
      - type: gbm
        params:
          n_estimators: 100
          max_depth: 3
          learning_rate: 0.1
          random_state: 42
    weights: [0.3, 0.7]
```

**Note**: Model-specific parameters beyond the examples above should be documented in model implementation code or model-specific docs. This schema provides common parameters only.

---

## 4. Features Block

### Structure

```yaml
features:
  groups: <list_of_feature_groups>  # Required: list of strings
  include: <explicit_feature_list>  # Optional: list of strings
  exclude: <exclusion_patterns>     # Optional: list of strings or regex patterns
```

### Feature Groups

**Definition**: Feature groups are logical collections of features defined in the feature registry.

**Available Groups** (NFL):
- `"baseline"`: Team form features (win rate, point differential, etc.)
- `"market"`: Market-derived features (spread, total, moneyline probabilities)
- `"epa"`: EPA-based features (offensive/defensive EPA per play)
- `"qb"`: QB-specific features (EPA/play, CPOE, etc.)
- `"form"`: Rolling form features (recent performance windows)
- `"schedule"`: Schedule context (rest days, travel, etc.)

**Future Groups**:
- `"injury"`: Injury burden features
- `"matchup"`: Offense vs defense matchup features
- `"weather"`: Weather context features

**Example**:
```yaml
features:
  groups:
    - baseline
    - market
    - epa
    - qb
```

### Include/Exclude

**Include**: Explicitly include specific features by name (overrides groups).

**Exclude**: Remove features matching patterns (applied after groups/include).

**Pattern Matching**:
- Exact match: `"feature_name"`
- Prefix match: `"prefix_*"` (if supported)
- Regex: `"pattern.*"` (if supported)

**Example**:
```yaml
features:
  groups:
    - baseline
    - market
  exclude:
    - "win_rate_16"  # Exclude 16-game win rate
    - "market_movement_*"  # Exclude all market movement features
```

### Feature Registry Integration

**Resolution**:
1. Load feature table(s) specified by `groups` from feature registry
2. Apply `include` filters (if specified)
3. Apply `exclude` filters (if specified)
4. Return final feature matrix

**Feature Table Names**: Use names registered in `features/feature_table_registry.py`:
- `"baseline"` → `game_features_baseline.parquet`
- `"phase2"` → `game_features_phase2.parquet`
- `"phase2b"` → `game_features_phase2b.parquet`

**Note**: If multiple groups reference different feature tables, implementation should merge them appropriately.

---

## 5. Data Block

### Structure

```yaml
data:
  train_seasons: <list_of_seasons>      # Required: list of integers
  val_seasons: <list_of_seasons>        # Required: list of integers (can be single season)
  test_seasons: <list_of_seasons>       # Required: list of integers (can be single season)
  filters:                              # Optional: data filtering rules
    include_regular_season: true        # bool, default: true
    include_playoffs: false            # bool, default: false
    min_games_per_team: null           # int, optional: filter teams with < N games
```

### Season Lists

**Format**: List of integers (e.g., `[2015, 2016, 2017]`)

**Semantics**:
- **Inclusive**: All games from specified seasons are included
- **Ordering**: Seasons should be in chronological order (for time-series splits)
- **Validation**: Implementation should validate that seasons exist in data

**Example**:
```yaml
data:
  train_seasons: [2015, 2016, 2017, 2018, 2019, 2020, 2021]
  val_seasons: [2022]
  test_seasons: [2023]
```

### Filters

**Include Regular Season**: Include regular season games (default: true).

**Include Playoffs**: Include playoff games (default: false).

**Min Games Per Team**: Filter out teams with fewer than N games in the dataset (optional, for data quality).

**Future Filters**:
- `exclude_weeks`: List of weeks to exclude (e.g., `[18]` for Week 18 if data quality issues)
- `team_whitelist`: Only include specific teams (list of team abbreviations)
- `team_blacklist`: Exclude specific teams (list of team abbreviations)

**Example**:
```yaml
data:
  train_seasons: [2015, 2016, 2017, 2018, 2019, 2020, 2021]
  val_seasons: [2022]
  test_seasons: [2023]
  filters:
    include_regular_season: true
    include_playoffs: false
```

---

## 6. Training Block

### Structure

```yaml
training:
  seed: <random_seed>                    # Required: int
  batch_size: <batch_size>               # Required: int (for neural models) or null (for tree models)
  max_epochs: <max_epochs>              # Required: int (for neural models) or null (for tree models)
  early_stopping_patience: <patience>    # Optional: int (for neural models)
  learning_rate: <learning_rate>         # Optional: float (for neural models, may be in model params)
  num_workers: <num_workers>             # Optional: int (for data loading, default: 0)
  device: <device>                      # Optional: string ("cpu" or "cuda", default: "cpu")
```

### Required Fields

**Seed**: Random seed for reproducibility (int, e.g., 42).

**Batch Size**: Batch size for training (int, e.g., 64). Set to `null` for tree-based models (GBM, LR) that don't use batches.

**Max Epochs**: Maximum training epochs (int, e.g., 100). Set to `null` for tree-based models.

### Optional Fields

**Early Stopping Patience**: Number of epochs to wait before stopping if validation loss doesn't improve (int, e.g., 10). Only applicable to neural models.

**Learning Rate**: Learning rate for optimizer (float, e.g., 0.001). May be specified in model params instead (e.g., for GBM).

**Num Workers**: Number of data loading workers (int, default: 0). Higher values speed up data loading but use more memory.

**Device**: Compute device (string, `"cpu"` or `"cuda"`). Default: `"cpu"`. Actual device handling is implementation-specific.

**Example (Neural Model)**:
```yaml
training:
  seed: 42
  batch_size: 64
  max_epochs: 100
  early_stopping_patience: 10
  learning_rate: 0.001
  num_workers: 4
  device: "cuda"
```

**Example (Tree Model)**:
```yaml
training:
  seed: 42
  batch_size: null
  max_epochs: null
  num_workers: 0
  device: "cpu"
```

---

## 7. Output Block

### Structure

```yaml
output:
  artifact_dir: <directory_path>         # Required: string (relative to models/artifacts/)
  save_predictions: <bool>               # Optional: bool (default: true)
  save_feature_importance: <bool>        # Optional: bool (default: false)
  save_calibration_model: <bool>         # Optional: bool (default: false)
```

### Artifact Directory

**Format**: String path relative to `models/artifacts/` (e.g., `"nfl_baseline"`, `"nfl_ft_transformer_v1"`).

**Full Path**: `models/artifacts/{artifact_dir}/`

**Contents** (saved automatically):
- Model weights/checkpoints: `{model_type}.pkl` or `{model_type}.pt`
- Ensemble config (if ensemble): `ensemble.json`
- Metadata: `metadata.json` (experiment name, config hash, timestamp)

### Save Flags

**Save Predictions**: Save predictions on train/val/test sets to Parquet files (default: true).
- Files: `predictions_train.parquet`, `predictions_val.parquet`, `predictions_test.parquet`
- Columns: `game_id`, `y_true`, `y_pred`, `p_pred`

**Save Feature Importance**: Save feature importance scores (if available) to JSON/CSV (default: false).
- File: `feature_importance.json` or `feature_importance.csv`
- Format: `{feature_name: importance_score}`

**Save Calibration Model**: Save calibration model (Platt scaling or isotonic regression) if applied (default: false).
- File: `calibration_model.pkl`

**Example**:
```yaml
output:
  artifact_dir: "nfl_gbm_v1"
  save_predictions: true
  save_feature_importance: true
  save_calibration_model: true
```

---

## 8. Eval Block (Phase B Placeholder)

### Structure

```yaml
eval:
  run_eval: <bool>                       # Optional: bool (default: false for Phase A)
  eval_config_name: <config_name>        # Optional: string (references eval blueprint)
  edge_thresholds: <thresholds>          # Optional: list of floats (default: [0.03, 0.05, 0.07])
  calibration_bins: <n_bins>             # Optional: int (default: 10)
  staking_modes:                         # Optional: list of strings (default: ["flat"])
    - flat
    - kelly_quarter
```

### Fields

**Run Eval**: Whether to run evaluation after training (bool, default: false in Phase A, true in Phase B).

**Eval Config Name**: Name of evaluation config (string, optional). References settings in `eval_blueprint_v1.md`.

**Edge Thresholds**: List of edge thresholds for market-relative evaluation (list of floats, default: `[0.03, 0.05, 0.07]`).

**Calibration Bins**: Number of bins for calibration analysis (int, default: 10).

**Staking Modes**: List of staking modes to simulate (list of strings, default: `["flat"]`).
- `"flat"`: Flat staking (1 unit per bet)
- `"kelly_full"`: Full Kelly criterion
- `"kelly_quarter"`: Quarter Kelly (0.25x full Kelly)
- `"kelly_half"`: Half Kelly (0.5x full Kelly)

**Example**:
```yaml
eval:
  run_eval: true
  edge_thresholds: [0.03, 0.05, 0.07]
  calibration_bins: 10
  staking_modes:
    - flat
    - kelly_quarter
```

**Note**: Full evaluation implementation will follow `eval_blueprint_v1.md`. This block is a placeholder for Phase B.

---

## 9. Example Configs

### Example 1: Baseline GBM Experiment

```yaml
experiment_name: nfl_baseline_gbm_v1
description: Baseline gradient boosting model with team form and market features
vertical: nfl

model:
  type: gbm
  params:
    n_estimators: 100
    max_depth: 3
    learning_rate: 0.1
    random_state: 42

features:
  groups:
    - baseline
    - market

data:
  train_seasons: [2015, 2016, 2017, 2018, 2019, 2020, 2021]
  val_seasons: [2022]
  test_seasons: [2023]
  filters:
    include_regular_season: true
    include_playoffs: false

training:
  seed: 42
  batch_size: null
  max_epochs: null
  num_workers: 0
  device: "cpu"

output:
  artifact_dir: "nfl_baseline_gbm_v1"
  save_predictions: true
  save_feature_importance: true
  save_calibration_model: false

eval:
  run_eval: false

metadata:
  author: "engineering_team"
  created_at: "2024-01-15T10:00:00Z"
  tags:
    - baseline
    - gbm
    - phase1
```

### Example 2: FT-Transformer Experiment

```yaml
experiment_name: nfl_ft_transformer_v1
description: FT-Transformer model with advanced features (EPA, QB metrics)
vertical: nfl

model:
  type: ft_transformer
  params:
    d_model: 128
    n_layers: 3
    n_heads: 4
    d_ff: 512
    dropout: 0.1
    random_state: 42

features:
  groups:
    - baseline
    - market
    - epa
    - qb
    - form
  exclude:
    - "win_rate_16"  # Exclude 16-game win rate (too much lookahead)

data:
  train_seasons: [2015, 2016, 2017, 2018, 2019, 2020, 2021]
  val_seasons: [2022]
  test_seasons: [2023]
  filters:
    include_regular_season: true
    include_playoffs: false

training:
  seed: 42
  batch_size: 64
  max_epochs: 100
  early_stopping_patience: 10
  learning_rate: 0.001
  num_workers: 4
  device: "cuda"

output:
  artifact_dir: "nfl_ft_transformer_v1"
  save_predictions: true
  save_feature_importance: false
  save_calibration_model: true

eval:
  run_eval: false

metadata:
  author: "engineering_team"
  created_at: "2024-01-20T14:30:00Z"
  tags:
    - advanced
    - transformer
    - phaseA
```

### Example 3: Ensemble Experiment

```yaml
experiment_name: nfl_ensemble_v2
description: Ensemble of logistic regression and gradient boosting
vertical: nfl

model:
  type: ensemble
  params:
    models:
      - type: lr
        params:
          C: 1.0
          max_iter: 1000
          random_state: 42
      - type: gbm
        params:
          n_estimators: 200
          max_depth: 5
          learning_rate: 0.05
          random_state: 42
    weights: [0.3, 0.7]

features:
  groups:
    - baseline
    - market

data:
  train_seasons: [2015, 2016, 2017, 2018, 2019, 2020, 2021]
  val_seasons: [2022]
  test_seasons: [2023]
  filters:
    include_regular_season: true
    include_playoffs: false

training:
  seed: 42
  batch_size: null
  max_epochs: null
  num_workers: 0
  device: "cpu"

output:
  artifact_dir: "nfl_ensemble_v2"
  save_predictions: true
  save_feature_importance: true
  save_calibration_model: true

eval:
  run_eval: false

metadata:
  author: "engineering_team"
  created_at: "2024-01-25T09:15:00Z"
  tags:
    - ensemble
    - baseline
    - phase1
```

---

## 10. Validation & Error Handling

### Required Validations

**Config Loading**:
- Validate YAML syntax
- Validate required fields are present
- Validate field types match schema
- Validate season lists are non-empty and contain valid years
- Validate feature groups exist in registry
- Validate model type is supported
- Validate ensemble weights sum to 1.0 (if ensemble)

**Runtime Validations**:
- Validate feature tables exist
- Validate data exists for specified seasons
- Validate model can be instantiated with given params
- Validate output directory can be created

### Error Messages

**Clear Error Messages**:
- "Missing required field: {field_name}"
- "Invalid model type: {type}. Supported types: lr, gbm, ft_transformer, tabnet, ensemble"
- "Feature group '{group}' not found in registry. Available groups: {list}"
- "No data found for seasons: {seasons}"

---

## 11. Versioning & Backward Compatibility

### Schema Versioning

**Current Version**: v1.0

**Version Field** (optional, future):
```yaml
schema_version: "1.0"
```

**Breaking Changes**: If schema changes break backward compatibility, increment major version.

**Non-Breaking Changes**: Adding optional fields is non-breaking.

### Migration

**Future**: Provide migration scripts or documentation for upgrading configs between schema versions.

---

*This schema is version 1.0. Updates will be versioned and documented.*

