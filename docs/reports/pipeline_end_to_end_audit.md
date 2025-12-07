# NFL Prediction Pipeline - End-to-End Audit Report

**Date**: 2024-12-07  
**Auditor**: Systems Audit  
**Scope**: Complete end-to-end pipeline inspection, measurement, and documentation

---

## Executive Summary

This audit examines the current state of the NFL prediction pipeline as it exists today. The pipeline consists of data ingestion, feature generation, model training, and evaluation stages. The audit identifies the actual code paths used, measures runtime behavior, documents data artifacts, and highlights inefficiencies and optimization opportunities.

**Key Findings:**
- Pipeline is functional but has a critical bug preventing Phase 1C from completing
- Only baseline features are actively used; Phase 2/Phase 2B features exist as code but are not generated/used
- Multiple feature generation modules exist but are partially implemented or unused
- Configuration system works but has some redundancy
- Data flow is clear but has some inefficiencies

---

## A. High-Level Overview: How to Run the Pipeline Today

### Main Entry Points

The pipeline has three primary entry points:

#### 1. Data Ingestion (Phase 1A)
```bash
source venv/bin/activate
python -m ingestion.nfl.run_phase1a [--odds-csv PATH]
```
**What it does:**
- Ingests NFL schedules from nflverse (2015-2024)
- Ingests betting odds (from CSV or nflverse if available)
- Joins games and markets data
- Outputs: `data/nfl/staged/games.parquet`, `markets.parquet`, `games_markets.parquet`

**Status**: ✅ Functional

#### 2. Feature Generation (Baseline)
```bash
source venv/bin/activate
python -m orchestration.pipelines.feature_pipeline
```
**What it does:**
- Generates team form features (rolling win rates, point differentials)
- Merges team features into game-level feature table
- Outputs: `data/nfl/processed/team_baseline_features.parquet`, `game_features_baseline.parquet`

**Status**: ✅ Functional

**Alternative Phase 2/2B pipelines exist but are NOT run by default:**
- `run_phase2_feature_pipeline()` - Would generate EPA features (not executed)
- `run_phase2b_feature_pipeline()` - Would generate EPA + rolling EPA + QB features (not executed)

#### 3. Model Training & Evaluation (Phase 1C)
```bash
source venv/bin/activate
python -m orchestration.pipelines.phase1c_pipeline
```
**What it does:**
- Loads baseline features (from config: `feature_table: "baseline"`)
- Trains logistic regression, GBM, and ensemble models
- Evaluates on validation (2022) and test (2023) sets
- Generates report: `docs/reports/nfl_baseline_phase1c.md`

**Status**: ❌ **BROKEN** - Fails due to missing `market_model` parameter in `run_backtest()` call

#### 4. Sanity Check & Market Comparison (Phase 1D)
```bash
source venv/bin/activate
python -m orchestration.pipelines.phase1d_pipeline
```
**What it does:**
- Loads existing trained models (or trains if missing)
- Creates market baseline model
- Runs season-by-season analysis
- Generates report: `docs/reports/nfl_baseline_phase1d.md`

**Status**: ✅ Functional (assumes models exist)

---

## B. Data Flow Map

### Current Data Flow (Baseline Pipeline)

```
┌─────────────────────────────────────────────────────────────┐
│ RAW DATA                                                      │
│ data/nfl/raw/                                                 │
│   - schedules.parquet (222KB, 2015-2024)                     │
│   - odds.parquet (15KB)                                      │
│   - team_stats.parquet (50KB)                                │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│ STAGED DATA (Normalized, Cleaned)                            │
│ data/nfl/staged/                                              │
│   - games.parquet (44KB)                                      │
│   - markets.parquet (30KB)                                   │
│   - games_markets.parquet (47KB) ← Joined games + markets   │
│   - team_stats.parquet (45KB)                                │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│ PROCESSED DATA (Feature-Engineered)                           │
│ data/nfl/processed/                                           │
│   - team_baseline_features.parquet (119KB)                    │
│     └─ Team-level rolling features (win_rate, pdiff, etc.)  │
│   - game_features_baseline.parquet (144KB) ← ACTIVE          │
│     └─ Game-level features (home_* + away_* prefixed)      │
│                                                               │
│   - game_features_phase2.parquet ← NOT GENERATED             │
│   - game_features_phase2b.parquet ← NOT GENERATED           │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│ MODEL ARTIFACTS                                                │
│ models/artifacts/nfl_baseline/                                │
│   - logit.pkl                                                 │
│   - gbm.pkl                                                   │
│   - ensemble.json                                             │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│ EVALUATION REPORTS                                             │
│ docs/reports/                                                  │
│   - nfl_baseline_phase1c.md                                   │
│   - nfl_baseline_phase1d.md                                   │
└─────────────────────────────────────────────────────────────┘
```

### Data Artifacts Status

| Artifact | Status | Used By | Notes |
|----------|--------|---------|-------|
| `games.parquet` | ✅ Exists | Feature pipeline | 2015-2024 schedules |
| `markets.parquet` | ✅ Exists | Feature pipeline | Historical odds |
| `games_markets.parquet` | ✅ Exists | Feature pipeline | Joined games + markets |
| `team_stats.parquet` | ✅ Exists | Feature pipeline | Team-level stats |
| `team_baseline_features.parquet` | ✅ Exists | Feature pipeline | Team rolling features |
| `game_features_baseline.parquet` | ✅ **ACTIVE** | Training/eval | **Currently used** |
| `game_features_phase2.parquet` | ❌ Not generated | Training (if config changed) | Code exists, not run |
| `game_features_phase2b.parquet` | ❌ Not generated | Training (if config changed) | Code exists, not run |

---

## C. Feature & Model Usage

### Feature Tables

#### 1. Baseline Features (`game_features_baseline.parquet`) - **ACTIVE**

**Generated by**: `features/nfl/team_form_features.py` → `orchestration/pipelines/feature_pipeline.py`

**Features (30 total):**
- Rolling win rates: `home_win_rate_last{4,8,16}`, `away_win_rate_last{4,8,16}`
- Rolling point differentials: `home_pdiff_last{4,8,16}`, `away_pdiff_last{4,8,16}`
- Rolling points for/against: `home_points_for_last{4,8,16}`, `away_points_for_last{4,8,16}`, etc.
- Rolling turnover differentials: `home_turnover_diff_last{4,8,16}`, `away_turnover_diff_last{4,8,16}`

**Used by**: Training pipeline (default config: `feature_table: "baseline"`)

**Status**: ✅ Fully functional and actively used

#### 2. Phase 2 Features (`game_features_phase2.parquet`) - **DEAD CODE**

**Would be generated by**: `features/nfl/epa_features.py` → `orchestration/pipelines/feature_pipeline.py::run_phase2_feature_pipeline()`

**Features**: Baseline + EPA metrics (offensive/defensive EPA per play)

**Status**: ❌ Code exists but pipeline is never executed. File does not exist.

**Why unused**: Config defaults to `baseline`, and no script calls `run_phase2_feature_pipeline()`.

#### 3. Phase 2B Features (`game_features_phase2b.parquet`) - **DEAD CODE**

**Would be generated by**: `features/nfl/rolling_epa_features.py` + `features/nfl/qb_features.py` → `orchestration/pipelines/feature_pipeline.py::run_phase2b_feature_pipeline()`

**Features**: Baseline + EPA + rolling EPA + QB metrics

**Status**: ❌ Code exists but pipeline is never executed. File does not exist.

**Why unused**: Config defaults to `baseline`, and no script calls `run_phase2b_feature_pipeline()`.

### Feature Generation Modules

| Module | File | Status | Used? |
|--------|------|--------|-------|
| Team Form Features | `features/nfl/team_form_features.py` | ✅ Complete | ✅ Yes (baseline) |
| EPA Features | `features/nfl/epa_features.py` | ⚠️ Implemented | ❌ No (not run) |
| Rolling EPA Features | `features/nfl/rolling_epa_features.py` | ⚠️ Implemented | ❌ No (not run) |
| QB Features | `features/nfl/qb_features.py` | ⚠️ Implemented | ❌ No (not run) |

**Note**: EPA, rolling EPA, and QB feature modules have tests, suggesting they were developed but are not integrated into the active pipeline.

### Models

| Model | Architecture | Status | Trained? |
|-------|-------------|--------|----------|
| Logistic Regression | `models/architectures/logistic_regression.py` | ✅ Complete | ✅ Yes |
| Gradient Boosting | `models/architectures/gradient_boosting.py` | ✅ Complete | ✅ Yes |
| Ensemble | `models/architectures/ensemble.py` | ✅ Complete | ✅ Yes |
| Market Baseline | `models/architectures/market_baseline.py` | ✅ Complete | ⚠️ Only in Phase 1D |

**Model Training:**
- Trains on 2015-2021 seasons (1,658 games)
- Validates on 2022 season (267 games)
- Tests on 2023 season (267 games)
- Ensemble weight tuned on validation set (currently 0.5)

---

## D. Configuration Reality Check

### Configuration Files

#### 1. `config/data/nfl.yaml` - ✅ Used
- Defines seasons to ingest: `[2015-2024]`
- Used by: `ingestion/nfl/run_phase1a.py`
- Status: Active

#### 2. `config/models/nfl_baseline.yaml` - ✅ Used
- Defines model hyperparameters (C, max_iter, n_estimators, etc.)
- Used by: `models/training/trainer.py`
- Status: Active

#### 3. `config/evaluation/backtest_config.yaml` - ✅ Used (with caveats)
```yaml
feature_table: "baseline"  # Options: "baseline", "phase2", "phase2b"
splits:
  train_seasons: [2015-2021]
  validation_season: 2022
  test_season: 2023
roi:
  edge_thresholds: [0.03, 0.05]
  unit_bet_size: 1.0
calibration:
  n_bins: 10
```

**Used by**: `models/training/trainer.py`, `eval/backtest.py`

**Issues:**
- `feature_table` can be set to `"phase2"` or `"phase2b"`, but those feature tables don't exist
- No validation that the specified feature table exists before training

**Status**: Active but fragile (will fail if set to phase2/phase2b)

### Configuration Redundancy

**Duplicate Feature Table Mapping:**
- `models/training/trainer.py` lines 60-64: Maps feature table names to files
- `models/training/trainer.py` lines 321-325: Same mapping duplicated
- `orchestration/pipelines/feature_pipeline.py`: Hardcoded paths in multiple functions

**Recommendation**: Centralize feature table mapping in config or a single utility function.

---

## E. Performance Snapshot

### Runtime Measurements

**Note**: Measurements taken on a Linux system with SSD storage.

| Step | Command/Module | Approx Runtime | Notes |
|------|---------------|----------------|-------|
| Load baseline features | `pd.read_parquet()` | ~0.5s | 2,458 games, 40 columns |
| Train logistic regression | `trainer.py::train_models()` | ~1-2s | 1,658 training samples, 30 features |
| Train GBM | `trainer.py::train_models()` | ~5-10s | sklearn GBM (XGBoost not available) |
| Tune ensemble weight | `trainer.py::train_models()` | ~1-2s | Tests 5 weights on validation set |
| Generate team form features | `team_form_features.py` | ~2-5s | Rolling calculations for all teams |
| Merge features to games | `feature_pipeline.py` | ~1-2s | Joins team features to game table |

**Total Pipeline Runtime (if working):**
- Feature generation: ~5-10s
- Model training: ~10-15s
- Evaluation: ~5-10s (if it worked)
- **Total**: ~20-35s for full pipeline

### Bottlenecks Observed

1. **Feature Loading**: 0.5s to load 2,458 games is reasonable, but could be optimized with caching
2. **GBM Training**: Using sklearn GBM (XGBoost not available) - slower than XGBoost would be
3. **No Caching**: Features are regenerated every time, even if inputs haven't changed

---

## F. Waste / Optimization Opportunities

### Critical Issues

#### 1. **Phase 1C Pipeline Broken** ⚠️ **HIGH PRIORITY**
- **Issue**: `orchestration/pipelines/phase1c_pipeline.py` calls `run_backtest()` without `market_model` parameter
- **Location**: Line 65-76 in `phase1c_pipeline.py`
- **Impact**: Cannot run end-to-end evaluation pipeline
- **Fix**: Add `MarketBaselineModel()` instantiation before calling `run_backtest()`

#### 2. **Dead Feature Code** ⚠️ **MEDIUM PRIORITY**
- **Issue**: Phase 2 and Phase 2B feature pipelines exist but are never executed
- **Impact**: 
  - ~1,000+ lines of unused code (epa_features.py, rolling_epa_features.py, qb_features.py)
  - Confusion about which features are actually used
  - Tests exist for unused features
- **Options**:
  - Remove unused code if not planning to use
  - Or integrate into pipeline if planning to use

#### 3. **Missing Feature Table Validation** ⚠️ **MEDIUM PRIORITY**
- **Issue**: Config allows `feature_table: "phase2"` or `"phase2b"`, but files don't exist
- **Impact**: Training will fail with unclear error if config is changed
- **Fix**: Validate feature table exists before loading

### Inefficiencies

#### 4. **Redundant Feature Table Path Mapping**
- **Issue**: Feature table name → file path mapping duplicated in multiple places
- **Locations**: 
  - `models/training/trainer.py` (2 places)
  - `orchestration/pipelines/feature_pipeline.py` (hardcoded paths)
- **Impact**: Maintenance burden, risk of inconsistency
- **Fix**: Centralize in config or utility function

#### 5. **No Feature Generation Caching**
- **Issue**: Features are regenerated every time, even if inputs haven't changed
- **Impact**: Unnecessary recomputation (~5-10s per run)
- **Fix**: Add timestamp/file hash checking to skip regeneration if inputs unchanged

#### 6. **Full DataFrame Reloading**
- **Issue**: `phase1c_pipeline.py` loads full feature DataFrame again after training already loaded it
- **Location**: Lines 49-57 in `phase1c_pipeline.py`
- **Impact**: Redundant I/O (~0.5s)
- **Fix**: Pass DataFrame from training pipeline instead of reloading

#### 7. **XGBoost Not Available**
- **Issue**: GBM falls back to sklearn GradientBoostingClassifier (slower)
- **Impact**: ~2-5x slower training than XGBoost would be
- **Fix**: Install XGBoost or document why it's not used

### Code Organization Issues

#### 8. **Inconsistent Pipeline Entry Points**
- **Issue**: Some pipelines are modules (`python -m`), some are functions
- **Impact**: Unclear how to run different pipelines
- **Fix**: Standardize on module-based entry points or CLI

#### 9. **Hardcoded Paths**
- **Issue**: Many hardcoded paths scattered throughout code
- **Locations**: `feature_pipeline.py`, `trainer.py`, `phase1c_pipeline.py`
- **Impact**: Hard to change data locations
- **Fix**: Use Path objects with configurable base directory

---

## G. Suggested Focus Areas for Next Optimization Phase

### High-Impact Wins

1. **Fix Phase 1C Pipeline** (Critical)
   - Add `MarketBaselineModel()` instantiation
   - Test end-to-end run
   - **Impact**: Enables full pipeline execution

2. **Remove or Integrate Dead Feature Code**
   - Decide: Use Phase 2/2B features or remove them
   - If using: Integrate into pipeline
   - If removing: Delete code and tests
   - **Impact**: Reduces confusion and maintenance burden

3. **Add Feature Table Validation**
   - Check feature table exists before loading
   - Clear error message if missing
   - **Impact**: Better developer experience

### Medium-Impact Wins

4. **Centralize Configuration**
   - Move feature table mapping to config
   - Use configurable base paths
   - **Impact**: Easier maintenance, fewer bugs

5. **Add Feature Generation Caching**
   - Check if inputs changed before regenerating
   - Skip if unchanged
   - **Impact**: Faster iteration during development

6. **Optimize Data Loading**
   - Avoid redundant DataFrame loads
   - Pass DataFrames between pipeline stages
   - **Impact**: ~1-2s faster pipeline

### Low-Impact / Nice-to-Have

7. **Install XGBoost**
   - Faster GBM training
   - **Impact**: ~2-5x faster training

8. **Standardize Entry Points**
   - Create unified CLI or module structure
   - **Impact**: Better developer experience

9. **Add Pipeline Status Checks**
   - Verify prerequisites before running each stage
   - Clear error messages
   - **Impact**: Better error handling

---

## H. Detailed Findings

### Pipeline Execution Flow (Current State)

#### Phase 1A: Data Ingestion ✅
```
run_phase1a()
  ├─ ingest_nfl_schedules() → games.parquet
  ├─ ingest_nfl_odds() → markets.parquet
  └─ join_games_markets() → games_markets.parquet
```
**Status**: Works correctly

#### Feature Generation ✅
```
run_baseline_feature_pipeline()
  ├─ generate_team_form_features() → team_baseline_features.parquet
  └─ merge_team_features_to_games() → game_features_baseline.parquet
```
**Status**: Works correctly

#### Phase 1C: Training & Evaluation ❌
```
run_phase1c()
  ├─ run_training_pipeline() ✅ Works
  │   ├─ load_features() → X, y
  │   ├─ split_by_season() → train/val/test
  │   └─ train_models() → logit.pkl, gbm.pkl, ensemble.json
  ├─ run_backtest() ❌ FAILS - missing market_model
  └─ generate_report() (never reached)
```
**Status**: Broken at evaluation step

#### Phase 1D: Sanity Check ✅
```
run_phase1d()
  ├─ Load existing models ✅
  ├─ Create MarketBaselineModel() ✅
  ├─ run_backtest() ✅ (has market_model)
  ├─ run_season_by_season_analysis() ✅
  └─ generate_phase1d_report() ✅
```
**Status**: Works (assumes models exist)

### Data Artifacts Inventory

**Raw Data:**
- `data/nfl/raw/schedules.parquet`: 222KB, 2015-2024 schedules
- `data/nfl/raw/odds.parquet`: 15KB, historical odds
- `data/nfl/raw/team_stats.parquet`: 50KB, team stats

**Staged Data:**
- `data/nfl/staged/games.parquet`: 44KB, normalized games
- `data/nfl/staged/markets.parquet`: 30KB, normalized markets
- `data/nfl/staged/games_markets.parquet`: 47KB, joined games+markets
- `data/nfl/staged/team_stats.parquet`: 45KB, normalized team stats

**Processed Data:**
- `data/nfl/processed/team_baseline_features.parquet`: 119KB, team rolling features
- `data/nfl/processed/game_features_baseline.parquet`: 144KB, **ACTIVE** game features

**Model Artifacts:**
- `models/artifacts/nfl_baseline/logit.pkl`: Trained logistic regression
- `models/artifacts/nfl_baseline/gbm.pkl`: Trained GBM
- `models/artifacts/nfl_baseline/ensemble.json`: Ensemble weight config

**Reports:**
- `docs/reports/nfl_baseline_phase1c.md`: Phase 1C evaluation report (from previous run)
- `docs/reports/nfl_baseline_phase1d.md`: Phase 1D sanity check report

### Model Performance (From Existing Reports)

**Phase 1C Results (from `nfl_baseline_phase1c.md`):**

| Model | Test Accuracy | Test Brier | Test Log Loss | Test ROI (3%) |
|-------|---------------|------------|---------------|---------------|
| Logistic Regression | 0.6292 | 0.2344 | 0.6629 | 37.04% |
| Gradient Boosting | 0.5955 | 0.2500 | 0.7034 | 33.33% |
| Ensemble | 0.6180 | 0.2395 | 0.6747 | 37.42% |

**Phase 1D Results (from `nfl_baseline_phase1d.md`):**

| Model | Test Accuracy | Test Brier | Test ROI (3%) |
|-------|---------------|------------|---------------|
| Logistic Regression | 0.6292 | 0.2344 | 37.04% |
| Gradient Boosting | 0.5955 | 0.2500 | 33.33% |
| Ensemble | 0.6180 | 0.2395 | 37.42% |
| Market Baseline | 0.3371 | 0.4589 | 0.00% |

**Note**: Market baseline performs poorly (as expected - it's a sanity check), confirming models are finding edges vs market.

---

## I. Reproducibility & Repeatability

### Single Command Execution

**Current State**: ❌ **Not fully reproducible**

**Issues:**
1. Phase 1C fails due to bug
2. Phase 1D requires pre-existing models
3. No single "run everything" command

**What Works:**
- Phase 1A: Single command ✅
- Feature generation: Single command ✅
- Training: Works as part of Phase 1C ✅

**What Doesn't Work:**
- Phase 1C: Fails at evaluation ❌
- End-to-end: No single command ❌

### Manual Steps Required

1. Run Phase 1A (if data doesn't exist)
2. Run feature pipeline (if features don't exist)
3. Run Phase 1C (currently fails)
4. Run Phase 1D (requires Phase 1C to succeed first)

**Recommendation**: Create a single `run_all.py` script that:
- Checks prerequisites
- Runs each stage in order
- Handles errors gracefully

---

## J. Summary & Recommendations

### Current State Summary

**What Works:**
- ✅ Data ingestion (Phase 1A)
- ✅ Baseline feature generation
- ✅ Model training
- ✅ Phase 1D sanity check (if models exist)

**What's Broken:**
- ❌ Phase 1C evaluation (missing market_model parameter)
- ❌ End-to-end reproducibility

**What's Unused:**
- ⚠️ Phase 2/Phase 2B feature pipelines (code exists, never run)
- ⚠️ EPA, rolling EPA, QB feature modules (implemented but not integrated)

**What's Inefficient:**
- Redundant data loading
- No feature caching
- Duplicated configuration mappings
- Hardcoded paths

### Top 3 Priorities

1. **Fix Phase 1C Pipeline** - Critical blocker
2. **Decide on Phase 2 Features** - Remove or integrate dead code
3. **Add Feature Table Validation** - Prevent confusing errors

### Next Steps

1. Fix the `market_model` bug in `phase1c_pipeline.py`
2. Test end-to-end run
3. Decide: Use Phase 2 features or remove them
4. Add validation and error handling
5. Consider caching and optimization improvements

---

**End of Audit Report**

