# Roadmap: Phase A & Phase B

## 1. Overview

This roadmap tracks implementation progress for **Phase A** (Advanced Models + Ensemble + Feature Registry + Calibration) and **Phase B** (Evaluation Engine). Later phases (C, D, etc.) are mentioned briefly but not detailed here.

### Phase A Objectives

- Implement advanced model architectures (FT-Transformer, TabNet)
- Build ensemble framework combining baseline and advanced models
- Introduce feature registry for centralized feature management
- Add calibration hooks (Platt scaling, isotonic regression)
- Ensure backward compatibility with existing Phase 1 baseline

### Phase B Objectives

- Implement comprehensive evaluation engine per `eval_blueprint_v1.md`
- Integrate evaluation into experiment configs (`experiment_schema.md`)
- Generate standardized evaluation reports
- Enable market-relative ROI analysis and risk metrics

### Process Rules

- **Checklist-Driven**: Every Cursor task must reference specific checklist items from this roadmap
- **Phase Gating**: Do not move to Phase B until all critical Phase A items are complete
- **Documentation Updates**: Update this roadmap as tasks are completed
- **Status Tracking**: Mark items as `[x]` when complete, `[ ]` when pending, `[~]` when in progress

---

## 2. Phase A – Tasks & Status

### 2.1 Advanced Model Implementations

- [ ] **Implement FT-Transformer Model Class**
  - [ ] Create `models/architectures/ft_transformer.py`
  - [ ] Implement `BaseModel` interface (`fit`, `predict_proba`, `save`, `load`)
  - [ ] Handle tabular data input (feature matrix)
  - [ ] Support configurable hyperparameters (d_model, n_layers, n_heads, etc.)
  - [ ] Add unit tests for FT-Transformer model
  - [ ] Document hyperparameter tuning guidelines

- [ ] **Implement TabNet Model Class**
  - [ ] Create `models/architectures/tabnet.py`
  - [ ] Implement `BaseModel` interface (`fit`, `predict_proba`, `save`, `load`)
  - [ ] Handle tabular data input (feature matrix)
  - [ ] Support configurable hyperparameters (n_d, n_a, n_steps, gamma, etc.)
  - [ ] Add unit tests for TabNet model
  - [ ] Document hyperparameter tuning guidelines

### 2.2 Model Integration

- [ ] **Integrate New Models into Training Pipeline**
  - [ ] Update `models/training/trainer.py` to support FT-Transformer and TabNet
  - [ ] Add model type detection and instantiation logic
  - [ ] Handle neural model training (epochs, batches, early stopping)
  - [ ] Ensure compatibility with existing data loading (`load_features`)
  - [ ] Add integration tests for new models in training pipeline

- [ ] **Update Model Registry/Versioning**
  - [ ] Extend model artifact saving to include neural model checkpoints
  - [ ] Ensure model versioning works for all model types
  - [ ] Update `models/artifacts/` directory structure if needed

### 2.3 Ensemble Module

- [x] **Basic Ensemble Implementation** (Phase 1C baseline exists)
  - [x] `models/architectures/ensemble.py` exists
  - [x] Supports weighted combination of models
  - [ ] **Enhance Ensemble for Advanced Models**
    - [ ] Support ensemble of FT-Transformer + TabNet + baseline models
    - [ ] Add ensemble weight optimization (grid search or validation-based)
    - [ ] Support loading pre-trained models for ensemble
    - [ ] Add ensemble-specific unit tests

### 2.4 Calibration

- [ ] **Implement Calibration Wrapper**
  - [ ] Create `models/training/calibration.py` (or extend existing)
  - [ ] Implement Platt scaling (logistic regression calibration)
  - [ ] Implement isotonic regression calibration
  - [ ] Add calibration model saving/loading
  - [ ] Integrate calibration into training pipeline (optional post-processing step)
  - [ ] Add unit tests for calibration methods

- [ ] **Calibration Integration**
  - [ ] Add calibration step to `models/training/trainer.py`
  - [ ] Make calibration optional (configurable via experiment config)
  - [ ] Ensure calibrated probabilities are used in evaluation
  - [ ] Add integration tests for calibration in training pipeline

### 2.5 Feature Registry

- [x] **Basic Feature Registry** (Phase 2B exists)
  - [x] `features/feature_table_registry.py` exists
  - [x] Maps feature table names to file paths
  - [ ] **Enhance Feature Registry**
    - [ ] Add feature group definitions (baseline, market, epa, qb, etc.)
    - [ ] Support feature group composition (merge multiple groups)
  - [ ] **Integrate Feature Registry with Feature Selection**
    - [ ] Update feature loading to use registry
    - [ ] Support `include`/`exclude` filters from experiment configs
    - [ ] Add validation for feature group names
    - [ ] Update `models/training/trainer.py` to use registry

### 2.6 Experiment Config Integration

- [ ] **Experiment Config Support**
  - [ ] Create config loader for experiment YAML files (see `experiment_schema.md`)
  - [ ] Update orchestration pipelines to load experiment configs
  - [ ] Validate experiment configs against schema
  - [ ] Add example experiment configs for FT-Transformer and TabNet

### 2.7 Testing & Quality Assurance

- [ ] **Unit Tests for Advanced Models**
  - [ ] Test FT-Transformer training and prediction
  - [ ] Test TabNet training and prediction
  - [ ] Test ensemble with advanced models
  - [ ] Test calibration methods

- [ ] **Integration Tests**
  - [ ] End-to-end test: FT-Transformer training → prediction → evaluation
  - [ ] End-to-end test: TabNet training → prediction → evaluation
  - [ ] End-to-end test: Ensemble (baseline + advanced) training → evaluation
  - [ ] Verify backward compatibility with Phase 1C baseline

- [ ] **CI/CD Updates**
  - [ ] Ensure CI passes with new models (if computationally feasible)
  - [ ] Add smoke tests for advanced models (small dataset, few epochs)
  - [ ] Document CI test timeouts and resource requirements

### 2.8 Documentation

- [ ] **Model Documentation**
  - [ ] Document FT-Transformer architecture and hyperparameters
  - [ ] Document TabNet architecture and hyperparameters
  - [ ] Document ensemble configuration options
  - [ ] Document calibration methods and when to use them

- [x] **Spec Documentation** (This task)
  - [x] `docs/eval_blueprint_v1.md`
  - [x] `docs/experiment_schema.md`
  - [x] `docs/live_output_contract.md`
  - [x] `docs/roadmap_phaseA_phaseB.md`

### 2.9 Phase A Exit Criteria

**Critical Items** (must be complete before Phase B):
- [ ] FT-Transformer model implemented and tested
- [ ] TabNet model implemented and tested
- [ ] Ensemble supports advanced models
- [ ] Calibration implemented and integrated
- [ ] Feature registry enhanced and integrated
- [ ] Experiment configs working for new models
- [ ] All tests passing
- [ ] Baseline flows remain intact (Phase 1C still works)

**Nice-to-Have** (can be deferred):
- [ ] Hyperparameter tuning automation
- [ ] Model comparison dashboard
- [ ] Advanced ensemble optimization

---

## 3. Phase B – Tasks & Status

### 3.1 Global Metrics Module

- [ ] **Log-Loss Implementation**
  - [ ] Compute log-loss on full test set
  - [ ] Compute log-loss per season
  - [ ] Compute log-loss per week
  - [ ] Add unit tests for log-loss computation

- [ ] **Brier Score Implementation**
  - [ ] Compute Brier score on full test set
  - [ ] Compute Brier score per season
  - [ ] Compute Brier score per week
  - [ ] Add unit tests for Brier score computation

- [ ] **Accuracy Implementation**
  - [ ] Compute accuracy on full test set
  - [ ] Compute accuracy per season
  - [ ] Compute accuracy per week
  - [ ] Add unit tests for accuracy computation

- [ ] **ROC AUC Implementation**
  - [ ] Compute ROC AUC on full test set
  - [ ] Compute ROC AUC per season (optional)
  - [ ] Add unit tests for ROC AUC computation

**Note**: Some metrics already exist in `eval/metrics.py`. Extend to support per-season/per-week aggregation.

### 3.2 Calibration Module

- [ ] **Calibration Binning**
  - [ ] Implement configurable binning (10 bins default, support 20 bins)
  - [ ] Compute per-bin statistics (mean predicted, actual frequency, count)
  - [ ] Handle edge cases (empty bins, single-game bins)
  - [ ] Add unit tests for binning logic

- [ ] **Calibration Tables**
  - [ ] Generate calibration table DataFrame
  - [ ] Format table for markdown reports
  - [ ] Add unit tests for table generation

- [ ] **Reliability Curves**
  - [ ] Generate reliability curve plot (matplotlib)
  - [ ] Save plots to `docs/reports/figures/`
  - [ ] Add unit tests for plot generation (optional, may be integration test)

### 3.3 Segmented Performance Module

- [ ] **Segmentation by Spread Buckets**
  - [ ] Implement favorites vs underdogs split
  - [ ] Implement spread magnitude buckets (small/medium/large)
  - [ ] Compute metrics per segment
  - [ ] Generate segmented performance tables

- [ ] **Segmentation by Home/Away**
  - [ ] Split predictions by home team vs away team
  - [ ] Compute metrics per segment
  - [ ] Generate segmented performance tables

- [ ] **Segmentation by Season**
  - [ ] Group games by season
  - [ ] Compute metrics per season
  - [ ] Generate temporal performance table

- [ ] **Segmentation by Confidence Deciles**
  - [ ] Sort games by model confidence (`|p - 0.5|`)
  - [ ] Create decile buckets
  - [ ] Compute metrics per decile
  - [ ] Generate confidence decile table

- [ ] **Unified Segmented Performance Module**
  - [ ] Create `eval/segmentation.py` module
  - [ ] Support all segmentation dimensions
  - [ ] Generate unified segmented performance report
  - [ ] Add unit tests for segmentation logic

### 3.4 Market-Relative Evaluation Module

- [ ] **Edge Computation**
  - [ ] Compute `edge = p_model - p_market` for each game
  - [ ] Handle missing market data gracefully
  - [ ] Add unit tests for edge computation

- [ ] **Market Probability Calculation**
  - [ ] Implement moneyline-to-probability conversion (accounting for vig)
  - [ ] Implement spread-to-probability conversion (logistic mapping)
  - [ ] Priority: moneyline > spread > 0.5 fallback
  - [ ] Add unit tests for market probability calculation

- [ ] **Edge Threshold Filtering**
  - [ ] Filter bets by edge thresholds (3%, 5%, 7%)
  - [ ] Create edge buckets for analysis
  - [ ] Add unit tests for threshold filtering

- [ ] **Flat Staking Simulation**
  - [ ] Implement flat staking (1 unit per bet)
  - [ ] Compute payouts using closing line odds
  - [ ] Track cumulative bankroll
  - [ ] Compute ROI and hit rate
  - [ ] Add unit tests for flat staking simulation

- [ ] **Kelly-Fraction Staking Simulation**
  - [ ] Implement Kelly criterion calculation
  - [ ] Support configurable Kelly fractions (0.25, 0.5, 1.0)
  - [ ] Track bankroll with Kelly sizing
  - [ ] Handle bankroll constraints (never bet more than bankroll)
  - [ ] Compute ROI and hit rate
  - [ ] Add unit tests for Kelly staking simulation

- [ ] **Market-Relative Outputs**
  - [ ] Generate ROI by edge bucket table
  - [ ] Generate hit rate by edge bucket table
  - [ ] Generate overall ROI summary
  - [ ] Integrate into evaluation reports

**Note**: Some ROI logic exists in `eval/backtest.py`. Extend and refactor as needed.

### 3.5 Risk Metrics Module

- [ ] **Bankroll Tracking**
  - [ ] Track bankroll over time (ordered by game date)
  - [ ] Compute final ROI
  - [ ] Add unit tests for bankroll tracking

- [ ] **Max Drawdown**
  - [ ] Compute peak-to-trough drawdown at each point
  - [ ] Find maximum drawdown
  - [ ] Add unit tests for max drawdown computation

- [ ] **Volatility**
  - [ ] Compute standard deviation of per-bet returns
  - [ ] Add unit tests for volatility computation

- [ ] **Longest Losing Streak**
  - [ ] Track consecutive losing bets
  - [ ] Compute longest streak (in bets and units)
  - [ ] Add unit tests for streak computation

- [ ] **Risk Metrics Integration**
  - [ ] Create `eval/risk.py` module
  - [ ] Integrate risk metrics into evaluation pipeline
  - [ ] Generate risk metrics summary table

### 3.6 Temporal Testing Module

- [ ] **Multiple Train/Test Split Support**
  - [ ] Support fixed window splits (configurable train/val/test seasons)
  - [ ] Support expanding window splits (optional, future)
  - [ ] Support rolling window splits (optional, future)
  - [ ] Add unit tests for split configurations

- [ ] **Per-Period Metric Computation**
  - [ ] Compute metrics for each time period (season, year group)
  - [ ] Generate temporal stability table
  - [ ] Compute coefficient of variation for stability metrics

- [ ] **Regime Shift Detection** (Optional, Future)
  - [ ] Identify potential regime shifts (rule changes, COVID season, etc.)
  - [ ] Compare metrics pre-regime vs post-regime
  - [ ] Generate regime shift analysis report

### 3.7 Report Generation

- [ ] **Report Template**
  - [ ] Create markdown report template
  - [ ] Include all metric sections (global, calibration, segmented, market-relative, risk)
  - [ ] Format tables consistently
  - [ ] Include model metadata (version, config, git SHA)

- [ ] **Report Generation Module**
  - [ ] Create `eval/reports.py` (extend existing or refactor)
  - [ ] Generate comprehensive evaluation report
  - [ ] Save reports to `docs/reports/`
  - [ ] Support multiple report formats (markdown, HTML, optional)

- [ ] **Report Integration**
  - [ ] Integrate report generation into evaluation pipeline
  - [ ] Generate reports automatically after training (if `eval.run_eval: true`)
  - [ ] Add integration tests for report generation

### 3.8 Evaluation Integration

- [ ] **Experiment Config Integration**
  - [ ] Load evaluation settings from experiment config (`eval` block)
  - [ ] Support `edge_thresholds`, `calibration_bins`, `staking_modes` from config
  - [ ] Validate evaluation config against schema

- [ ] **Training Pipeline Integration**
  - [ ] Add evaluation step to training pipeline (if `eval.run_eval: true`)
  - [ ] Ensure evaluation runs on val and test sets
  - [ ] Save evaluation results alongside model artifacts

- [ ] **Command-Line Interface**
  - [ ] Add CLI command for running evaluations standalone
  - [ ] Support evaluation of pre-trained models
  - [ ] Document CLI usage

### 3.9 Testing & Quality Assurance

- [ ] **Unit Tests**
  - [ ] Test all metric computation functions
  - [ ] Test calibration binning with various configurations
  - [ ] Test segmentation logic
  - [ ] Test market-relative evaluation (edge computation, staking simulations)
  - [ ] Test risk metrics computation

- [ ] **Integration Tests**
  - [ ] End-to-end evaluation on sample dataset
  - [ ] Verify report generation works correctly
  - [ ] Verify evaluation integrates with training pipeline

- [ ] **Regression Tests**
  - [ ] Compare metrics against Phase 1C baseline (see `docs/reports/nfl_baseline_phase1c.md`)
  - [ ] Ensure no performance regressions
  - [ ] Document expected metric ranges

### 3.10 Phase B Exit Criteria

**Critical Items** (must be complete):
- [ ] All global metrics implemented (log-loss, Brier, accuracy, ROC AUC)
- [ ] Calibration evaluation implemented (binning, tables, curves)
- [ ] Segmented performance implemented (spread, season, confidence deciles)
- [ ] Market-relative evaluation implemented (edge computation, staking simulations, ROI)
- [ ] Risk metrics implemented (drawdown, volatility, streaks)
- [ ] Report generation working
- [ ] Evaluation integrated with experiment configs
- [ ] All tests passing

**Nice-to-Have** (can be deferred):
- [ ] Advanced visualizations (bankroll curves, drawdown plots)
- [ ] Temporal stability analysis (regime shifts)
- [ ] Interactive dashboards

---

## 4. Process Rules

### 4.1 Task Management

- **Reference Checklist Items**: Every Cursor task must reference specific checklist items from this roadmap
  - Example: "Implement FT-Transformer model (Phase A, Section 2.1)"
  - Example: "Add calibration binning to evaluation (Phase B, Section 3.2)"

- **Update Status**: As tasks are completed, update this roadmap:
  - Mark items as `[x]` when complete
  - Add notes if implementation differs from spec
  - Document any blockers or dependencies

### 4.2 Phase Gating

- **Do Not Start Phase B Until**:
  - All critical Phase A items are complete (see Phase A Exit Criteria)
  - Phase A tests are passing
  - Baseline flows remain intact

- **Phase B Can Start When**:
  - Phase A critical items are checked
  - Documentation is updated
  - Team agrees Phase A is stable

### 4.3 Documentation Updates

- **Update This Roadmap**:
  - Mark items complete as they are finished
  - Add notes about implementation details or deviations
  - Document any new tasks discovered during implementation

- **Update Other Docs**:
  - If evaluation blueprint needs changes, update `eval_blueprint_v1.md`
  - If experiment schema needs changes, update `experiment_schema.md`
  - If output contract changes, update `live_output_contract.md`

### 4.4 Quality Gates

- **Before Marking Complete**:
  - Code is reviewed (if applicable)
  - Unit tests are written and passing
  - Integration tests are passing (if applicable)
  - Documentation is updated
  - No regressions introduced

---

## 5. Dependencies & Blockers

### Phase A Dependencies

- **FT-Transformer**: Requires PyTorch and transformer libraries
- **TabNet**: Requires PyTorch and TabNet library (pytorch-tabnet)
- **Calibration**: Requires scikit-learn (Platt scaling, isotonic regression)
- **Feature Registry**: Depends on existing feature tables being available

### Phase B Dependencies

- **Phase A Completion**: Phase B cannot start until Phase A critical items are done
- **Evaluation Blueprint**: Phase B implementation follows `eval_blueprint_v1.md`
- **Experiment Configs**: Phase B requires experiment config schema to be finalized

### Known Blockers

- None currently identified. Update this section as blockers arise.

---

## 6. Future Phases (Brief)

### Phase C (Post-Phase B)

- Real-time prediction pipeline
- API layer for serving predictions
- Database schema for storing predictions
- Frontend/dashboard for viewing predictions

### Phase D (Post-Phase C)

- Multi-sport generalization (NBA, MLB, etc.)
- Advanced feature engineering (injury impact, matchup features)
- Model retraining automation
- Production monitoring and alerting

**Note**: Phases C and D are not detailed here. This roadmap focuses on Phases A and B only.

---

*This roadmap is a living document. Update as tasks progress.*

