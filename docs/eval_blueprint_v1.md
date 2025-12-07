# Evaluation Blueprint v1

## 1. Overview

### Purpose

The evaluation layer provides comprehensive assessment of model performance across multiple dimensions: predictive quality, calibration, market-relative value, and risk management. This blueprint defines the evaluation framework that will guide Phase B implementation.

### Scope

The evaluation engine covers:

- **Historical Backtests**: Time-ordered evaluation on held-out test sets
- **Model Selection**: Comparative evaluation across model architectures and hyperparameters
- **Risk Assessment**: Bankroll simulation, drawdown analysis, and volatility metrics
- **Market-Relative Evaluation**: ROI analysis versus closing line value (CLV) using simulated betting strategies

### Architecture Context

- Evaluation modules live under `eval/`
- Consumes predictions from trained models (saved in `models/artifacts/`)
- Uses feature-engineered data from `data/nfl/processed/`
- Generates reports in `docs/reports/`
- Integrates with experiment configs (see `experiment_schema.md`)

---

## 2. Global Metrics

### 2.1 Log-Loss (Binary Cross-Entropy)

**Definition**: Log-loss measures the quality of probability predictions. Lower is better.

**Computation**:
- For each game: `-y_true * log(p_pred) - (1 - y_true) * log(1 - p_pred)`
- Average across all games
- Clip probabilities to [epsilon, 1-epsilon] to avoid log(0)

**Computed On**:
- Full test set (single aggregate value)
- Per-season (one value per season)
- Per-week (one value per week, for temporal analysis)

**Interpretation**:
- Perfect predictions: log-loss = 0
- Random predictions (50/50): log-loss ≈ 0.693
- Good model: log-loss < 0.7
- Excellent model: log-loss < 0.6

### 2.2 Brier Score

**Definition**: Mean squared error between predicted probabilities and actual binary outcomes. Lower is better.

**Computation**:
- For each game: `(p_pred - y_true)^2`
- Average across all games

**Computed On**:
- Full test set (single aggregate value)
- Per-season (one value per season)
- Per-week (one value per week, for temporal analysis)

**Interpretation**:
- Perfect predictions: Brier = 0
- Random predictions (50/50): Brier = 0.25
- Good model: Brier < 0.25
- Excellent model: Brier < 0.20

### 2.3 Accuracy

**Definition**: Fraction of correct binary predictions (home win vs away win).

**Computation**:
- Convert probabilities to binary predictions using threshold = 0.5
- Count correct predictions: `sum(y_pred == y_true) / n_games`

**Computed On**:
- Full test set (single aggregate value)
- Per-season (one value per season)
- Per-week (one value per week, for temporal analysis)

**Interpretation**:
- Random baseline: accuracy ≈ 0.50
- Good model: accuracy > 0.60
- Excellent model: accuracy > 0.65

**Note**: Accuracy alone is insufficient for probability models. Always report alongside log-loss and Brier score.

### 2.4 ROC AUC

**Definition**: Area under the receiver operating characteristic curve. Measures ability to rank games by win probability.

**Computation**:
- Sort games by predicted probability (descending)
- Calculate true positive rate (TPR) and false positive rate (FPR) at each threshold
- Compute area under TPR vs FPR curve using trapezoidal rule

**Computed On**:
- Full test set (single aggregate value)
- Per-season (one value per season, optional)

**Interpretation**:
- Random predictions: AUC = 0.50
- Good model: AUC > 0.65
- Excellent model: AUC > 0.70

**Note**: ROC AUC is less informative for imbalanced datasets or when probability calibration matters more than ranking.

---

## 3. Calibration Evaluation

### 3.1 Binning Strategy

**Default**: 10 equal-width bins by predicted probability.

**Alternative**: 20 bins for finer-grained analysis (optional, configurable).

**Binning Method**:
- Create bins: `[0.0, 0.1), [0.1, 0.2), ..., [0.9, 1.0]` for 10 bins
- Assign each prediction to a bin based on its predicted probability
- Handle edge cases: predictions exactly at bin boundaries

**Minimum Bin Size**: Skip bins with fewer than 5 games (configurable threshold).

### 3.2 Per-Bin Metrics

For each bin, compute:

- **Mean Predicted Probability**: Average of all `p_pred` values in the bin
- **Actual Event Frequency**: Fraction of games in bin where `y_true = 1` (home win)
- **Count of Games**: Number of games assigned to this bin
- **Calibration Error**: `|mean_predicted - actual_frequency|`

### 3.3 Calibration Tables

**Output Format**: DataFrame with columns:
- `bin` (1-10)
- `bin_min`, `bin_max` (probability range)
- `predicted_freq` (mean predicted probability)
- `actual_freq` (actual home win rate)
- `count` (number of games)
- `calibration_error` (absolute difference)

**Example Table**:
```
bin | bin_min | bin_max | predicted_freq | actual_freq | count | calibration_error
----|---------|---------|----------------|-------------|-------|------------------
1   | 0.0     | 0.1     | 0.05           | 0.08        | 12    | 0.03
2   | 0.1     | 0.2     | 0.15           | 0.18        | 28    | 0.03
...
```

### 3.4 Reliability Curves

**Definition**: Plot of predicted probability (x-axis) vs actual frequency (y-axis).

**Ideal Calibration**: Points lie on the diagonal line `y = x`.

**Output**: Matplotlib plot or saved image file.

**Visualization Elements**:
- Scatter points: (mean_predicted, actual_frequency) per bin
- Diagonal reference line: perfect calibration
- Error bars: optional, showing uncertainty in actual frequency
- Title: Model name, dataset, date

### 3.5 Calibration Quality Thresholds

**"Good Enough" Calibration**:
- Mean calibration error (average of `|predicted - actual|` per bin) < 0.05
- Maximum calibration error (max `|predicted - actual|` across bins) < 0.10
- No systematic bias (over-prediction or under-prediction) across probability ranges

**Degraded Calibration**:
- Mean calibration error > 0.10: model needs recalibration
- Maximum calibration error > 0.20: severe miscalibration in some probability ranges

**Note**: These thresholds are guidelines. Context matters (e.g., high-stakes predictions may require stricter thresholds).

---

## 4. Segmented Performance

### 4.1 Segmentation Dimensions

Evaluate performance across the following segments:

#### Spread Buckets
- **Favorites vs Underdogs**: Split by `close_spread < 0` (home favored) vs `close_spread >= 0` (away favored or pick'em)
- **Spread Magnitude**: 
  - Small spread: `|spread| <= 3`
  - Medium spread: `3 < |spread| <= 7`
  - Large spread: `|spread| > 7`
- **Extreme Favorites**: `spread <= -7` (home heavy favorite) vs `spread >= 7` (away heavy favorite)

#### Home vs Away
- **Home Team**: Predictions for home team wins
- **Away Team**: Predictions for away team wins (invert probabilities: `1 - p_home`)

#### Season / Year
- **Per-Season**: Aggregate metrics for each season (e.g., 2018, 2019, ..., 2023)
- **Season Groups**: Early seasons (2015-2017) vs recent seasons (2020-2023)

#### Confidence Deciles
- **Model Confidence**: Sort games by `|p_pred - 0.5|` (distance from 50/50)
  - Decile 1: Most confident predictions (`|p - 0.5|` highest)
  - Decile 10: Least confident predictions (`|p - 0.5|` lowest)
- **Edge Magnitude**: Sort games by `|p_model - p_market|` (model edge vs market)
  - Decile 1: Largest edges
  - Decile 10: Smallest edges (model agrees with market)

#### Game Context (Future)
- **Regular Season vs Playoffs**: Separate evaluation
- **Primetime vs Non-Primetime**: If metadata available
- **Division Games vs Non-Division**: If metadata available

### 4.2 Metrics Per Segment

For each segment, compute:

- **Log-Loss**: Average log-loss for games in segment
- **Brier Score**: Average Brier score for games in segment
- **Accuracy**: Binary accuracy for games in segment
- **Sample Size**: Number of games in segment

**Optional Metrics**:
- **ROC AUC**: If segment has sufficient sample size (> 50 games)
- **Calibration Error**: Mean calibration error for segment (if segment has enough games for binning)

### 4.3 Segmented Performance Tables

**Output Format**: DataFrame with rows = segments, columns = metrics.

**Example Table (Spread Buckets)**:
```
segment              | n_games | log_loss | brier_score | accuracy
---------------------|---------|----------|-------------|----------
Home Favorite        | 145     | 0.62     | 0.22        | 0.66
Away Favorite        | 122     | 0.68     | 0.24        | 0.61
Small Spread (<=3)   | 89      | 0.71     | 0.25        | 0.58
Large Spread (>7)    | 78      | 0.55     | 0.19        | 0.72
```

**Example Table (Confidence Deciles)**:
```
decile | n_games | mean_confidence | log_loss | brier_score | accuracy
-------|---------|-----------------|----------|-------------|----------
1      | 27      | 0.85            | 0.45     | 0.15        | 0.89
2      | 27      | 0.75            | 0.52     | 0.18        | 0.81
...
10     | 26      | 0.52            | 0.69     | 0.25        | 0.54
```

---

## 5. Market-Relative Evaluation

### 5.1 Edge Definition

**Model Edge**: `edge = p_model - p_market`

Where:
- `p_model`: Model predicted home win probability
- `p_market`: Market-implied home win probability (from closing line)

**Interpretation**:
- Positive edge: Model favors home team more than market
- Negative edge: Model favors away team more than market (or home less than market)
- Zero edge: Model agrees with market

**Edge Magnitude**: `|edge|` measures disagreement strength.

### 5.2 Market-Implied Probability Calculation

**Priority Order**:
1. **Moneyline**: Convert `moneyline_home` and `moneyline_away` to probabilities (accounting for vig)
2. **Spread**: Convert `close_spread` to probability using logistic mapping (see existing `spread_to_implied_probability`)
3. **Fallback**: If neither available, use 0.5 (fair market assumption)

**Note**: Only use pre-game closing line data. Never use post-game information.

### 5.3 Edge Thresholds

**Default Thresholds**: [3%, 5%, 7%]

**Betting Rule**: Only bet when `|edge| >= threshold`.

**Rationale**:
- Smaller thresholds (3%): More bets, higher volume, potentially lower ROI per bet
- Larger thresholds (7%): Fewer bets, higher confidence, potentially higher ROI per bet

**Edge Buckets**: For analysis, group bets by edge magnitude:
- `|edge| < 3%`: No bet
- `3% <= |edge| < 5%`: Small edge
- `5% <= |edge| < 7%`: Medium edge
- `|edge| >= 7%`: Large edge

### 5.4 Simulation Modes

#### Flat Staking
**Definition**: Bet fixed amount per game (e.g., 1 unit).

**Parameters**:
- `unit_bet_size`: Fixed stake per bet (default: 1.0)

**Payout Calculation**:
- If betting home team: `payout = unit_bet_size * (moneyline_home / 100)` if win, else `-unit_bet_size`
- If betting away team: `payout = unit_bet_size * (moneyline_away / 100)` if win, else `-unit_bet_size`

**Note**: Use closing line odds for payout calculation (simulating CLV).

#### Kelly-Fraction Staking
**Definition**: Bet size proportional to edge and bankroll, scaled by Kelly fraction.

**Kelly Criterion Formula**:
- `kelly_fraction = (p_model * odds - 1) / (odds - 1)`
- Where `odds` is decimal odds (e.g., +150 → 2.5)
- `bet_size = bankroll * kelly_fraction * fraction_multiplier`

**Parameters**:
- `kelly_fraction`: Multiplier on full Kelly (default: 0.25, i.e., quarter-Kelly)
- `starting_bankroll`: Initial bankroll in units (default: 100)

**Full Kelly**: `fraction_multiplier = 1.0` (aggressive, high volatility)
**Quarter Kelly**: `fraction_multiplier = 0.25` (conservative, lower volatility)

**Bankroll Management**:
- Update bankroll after each bet: `bankroll = bankroll + payout`
- Never bet more than current bankroll (cap bet size)
- If bankroll drops below threshold (e.g., 10 units), stop betting or reset

### 5.5 Market-Relative Outputs

#### Overall ROI
**Definition**: `ROI = (total_profit / total_wagered) * 100%`

**Computed For**:
- Each edge threshold (3%, 5%, 7%)
- Each staking mode (flat, Kelly fractions)

**Output**: Single value per configuration.

#### ROI by Edge Bucket
**Definition**: ROI computed separately for each edge magnitude bucket.

**Output Format**: Table:
```
edge_bucket      | n_bets | win_rate | total_wagered | total_profit | roi
-----------------|--------|----------|---------------|--------------|-----
3% <= edge < 5%  | 45     | 0.62     | 45.0          | 2.7          | 6.0%
5% <= edge < 7%  | 32     | 0.69     | 32.0          | 4.2          | 13.1%
edge >= 7%       | 18     | 0.78     | 18.0          | 5.0          | 27.8%
```

#### Hit Rate by Edge Bucket
**Definition**: Win rate (fraction of winning bets) for each edge bucket.

**Output**: Same table as above, with `win_rate` column.

**Interpretation**: Higher hit rate in larger edge buckets suggests model is finding genuine edges.

---

## 6. Bankroll & Risk Metrics

### 6.1 Starting Assumptions

**Default Bankroll**: 100 units

**Unit Definition**: 1 unit = base bet size (for flat staking) or 1% of starting bankroll (for Kelly staking).

**Note**: Units are abstract; actual dollar amounts depend on user's bankroll.

### 6.2 Core Risk Metrics

#### Final ROI
**Definition**: `(final_bankroll - starting_bankroll) / starting_bankroll * 100%`

**Computed For**:
- Each staking mode
- Each edge threshold

#### Max Drawdown
**Definition**: Largest peak-to-trough decline in bankroll during simulation.

**Computation**:
- Track cumulative bankroll over time (ordered by game date)
- Find peak: maximum bankroll value seen so far
- Find drawdown at each point: `(peak - current_bankroll) / peak`
- Max drawdown: maximum drawdown value

**Output**: Percentage (e.g., -25% means bankroll dropped 25% from peak).

**Interpretation**: Lower (more negative) max drawdown = higher risk.

#### Volatility (Standard Deviation of Returns)
**Definition**: Standard deviation of per-bet returns.

**Computation**:
- Calculate return per bet: `(payout / bet_size)`
- Compute standard deviation across all bets

**Output**: Standard deviation value (e.g., 0.15 = 15% volatility per bet).

**Interpretation**: Higher volatility = more unpredictable returns.

#### Longest Losing Streak
**Definition**: Maximum consecutive losing bets.

**Computed In**:
- **Bets**: Number of consecutive losing bets
- **Units**: Total units lost during longest streak

**Output**: Tuple `(n_bets, units_lost)`.

**Example**: `(8, -12.5)` means 8 consecutive losses totaling -12.5 units.

### 6.3 Risk-Adjusted Metrics (Future)

**Sharpe Ratio**: `(mean_return / std_return) * sqrt(n_bets)` (if returns are approximately normal)

**Sortino Ratio**: Similar to Sharpe, but uses downside deviation (only negative returns).

**Calmar Ratio**: `annualized_return / max_drawdown` (if we have enough data for annualization).

### 6.4 Visualizations (Future Implementation)

#### Cumulative Bankroll Curve
**Plot**: Bankroll over time (x-axis = game date or bet number, y-axis = bankroll in units).

**Elements**:
- Line: cumulative bankroll
- Horizontal line: starting bankroll (reference)
- Annotations: mark max drawdown point

#### Drawdown Curve
**Plot**: Drawdown percentage over time.

**Elements**:
- Line: drawdown percentage (negative values)
- Shaded area: fill below zero
- Annotation: mark max drawdown point

**Note**: These visualizations are specified here but may be implemented in Phase B or later.

---

## 7. Temporal & Regime Testing

### 7.1 Train/Test Split Experiments

**Experiment Types**:

#### Fixed Window Splits
- **Train**: 2015-2020, **Val**: 2021, **Test**: 2022-2023
- **Train**: 2015-2021, **Val**: 2022, **Test**: 2023
- **Train**: 2016-2022, **Val**: 2023, **Test**: 2024 (future)

#### Expanding Window
- Start with small training set (e.g., 2015-2017)
- Gradually expand training set, test on next season
- Evaluate how performance changes as training data grows

#### Rolling Window
- Fixed training window size (e.g., 5 seasons)
- Slide window forward, test on next season
- Evaluate stability over time

### 7.2 Out-of-Sample Years

**Definition**: Years that were not seen during training.

**Testing Strategy**:
- Train on 2015-2020, test on 2021-2023 (all out-of-sample)
- Compare performance on in-sample vs out-of-sample years
- Identify performance degradation over time

### 7.3 Regime Shifts

**Potential Regime Shifts**:

#### Rule Changes
- **2018**: Helmet rule changes, roughing-the-passer emphasis
- **2020**: COVID-19 season (no fans, schedule disruptions)
- **2021+**: 17-game season (vs 16-game previously)

#### Data Quality Changes
- Play-by-play data availability (nflfastR coverage improves over time)
- Market data quality (odds archives may have gaps in early years)

**Testing Approach**:
- Segment evaluation by pre-regime vs post-regime
- Compare metrics: Are predictions stable across regimes?
- Identify if model needs regime-specific calibration

### 7.4 Temporal Stability Reporting

**Output Format**: Table showing metrics per time period.

**Example**:
```
period        | n_games | log_loss | brier_score | accuracy | roi_3pct
--------------|---------|----------|-------------|----------|----------
2018-2019     | 512     | 0.65     | 0.23        | 0.64     | 12.5%
2020-2021     | 544     | 0.68     | 0.24        | 0.62     | 8.3%
2022-2023     | 534     | 0.71     | 0.25        | 0.60     | 5.1%
```

**Stability Metrics**:
- **Coefficient of Variation**: `std(metric) / mean(metric)` across periods
- Lower CV = more stable performance

**Alerts**: Flag if performance degrades > 10% from baseline period.

---

## 8. Implementation Notes

### 8.1 Document Status

**This document is a specification, not code.**

- Defines what the evaluation engine should compute
- Does not prescribe exact implementation details (data structures, function signatures)
- Implementation should live under `eval/` directory
- Code should follow existing patterns in `eval/metrics.py` and `eval/backtest.py`

### 8.2 Integration Points

**Inputs**:
- Predictions: Model outputs (probabilities) saved from training pipeline
- Ground truth: Actual game outcomes from `data/nfl/staged/games.parquet`
- Market data: Closing lines from `data/nfl/staged/markets.parquet`
- Metadata: Game dates, teams, spreads from feature tables

**Outputs**:
- Metrics: Aggregated statistics (dictionaries or DataFrames)
- Reports: Markdown files in `docs/reports/`
- Visualizations: Plots saved to `docs/reports/figures/` (future)

**Configuration**:
- Evaluation settings come from experiment configs (see `experiment_schema.md`)
- Edge thresholds, binning strategy, staking modes are configurable

### 8.3 Phase B Implementation Checklist

Derived from this blueprint, Phase B should implement:

- [ ] **Global Metrics Module**
  - [ ] Log-loss computation (full set, per-season, per-week)
  - [ ] Brier score computation (full set, per-season, per-week)
  - [ ] Accuracy computation (full set, per-season, per-week)
  - [ ] ROC AUC computation (full set, per-season)

- [ ] **Calibration Module**
  - [ ] Binning strategy (10 bins default, configurable)
  - [ ] Per-bin statistics computation
  - [ ] Calibration table generation
  - [ ] Reliability curve plotting (matplotlib)

- [ ] **Segmented Performance Module**
  - [ ] Segmentation by spread buckets
  - [ ] Segmentation by home/away
  - [ ] Segmentation by season
  - [ ] Segmentation by confidence deciles
  - [ ] Metrics computation per segment
  - [ ] Segmented performance table generation

- [ ] **Market-Relative Evaluation Module**
  - [ ] Edge computation (`p_model - p_market`)
  - [ ] Market probability calculation (moneyline priority, spread fallback)
  - [ ] Edge threshold filtering
  - [ ] Flat staking simulation
  - [ ] Kelly-fraction staking simulation
  - [ ] ROI computation (overall, by edge bucket)
  - [ ] Hit rate computation (by edge bucket)

- [ ] **Risk Metrics Module**
  - [ ] Bankroll tracking over time
  - [ ] Max drawdown computation
  - [ ] Volatility (std dev of returns) computation
  - [ ] Longest losing streak computation (bets and units)
  - [ ] Final ROI computation

- [ ] **Temporal Testing Module**
  - [ ] Multiple train/test split configurations
  - [ ] Per-period metric computation
  - [ ] Stability metrics (coefficient of variation)
  - [ ] Regime shift detection (optional)

- [ ] **Report Generation**
  - [ ] Markdown report template
  - [ ] Integration of all metrics into unified report
  - [ ] Report saving to `docs/reports/`

- [ ] **Integration**
  - [ ] Integration with experiment configs
  - [ ] Integration with training pipeline (automatic eval after training)
  - [ ] Command-line interface for running evaluations

### 8.4 Testing Requirements

**Unit Tests**:
- Test each metric computation function with known inputs/outputs
- Test edge cases (empty segments, single-game segments)
- Test calibration binning with various bin counts

**Integration Tests**:
- End-to-end evaluation on sample dataset
- Verify outputs match expected formats
- Verify reports are generated correctly

**Regression Tests**:
- Compare metrics against Phase 1C baseline (see `docs/reports/nfl_baseline_phase1c.md`)
- Ensure no performance regressions

---

## 9. Future Enhancements (Post-Phase B)

**Not in scope for Phase B, but documented for future reference**:

- **Multi-Model Comparison**: Side-by-side evaluation of multiple models
- **Feature Importance Analysis**: Which features drive predictions?
- **Ablation Studies**: Remove features and measure impact
- **Confidence Intervals**: Bootstrap or analytical CIs for metrics
- **Advanced Visualizations**: Interactive dashboards, comparison plots
- **Real-Time Monitoring**: Track live prediction performance during season
- **Automated Alerts**: Notify when metrics degrade beyond thresholds

---

*This blueprint is version 1.0. Updates will be versioned and documented.*

