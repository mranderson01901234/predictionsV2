# Live Prediction Output Contract

## 1. Overview

### Purpose

This document defines the **exact shape** of outputs produced by the live prediction pipeline. It serves as the contract between:

- **Model Pipeline**: Generates predictions from trained models
- **API Layer**: Serves predictions via REST API (future)
- **Database**: Stores predictions for historical tracking (future)
- **Frontend/Dashboard**: Displays predictions to users (future)

### Scope

- **Single Game Prediction**: JSON structure for one NFL game
- **Weekly Batch**: Structure for an entire week's predictions
- **Metadata**: Model versioning, timestamps, data provenance
- **Tabular Representation**: Parquet/CSV equivalent for bulk analysis

### Versioning

**Current Version**: v1.0

All output structures are versioned. Breaking changes require version increments and migration plans.

---

## 2. Per-Game JSON Contract

### 2.1 Structure

Each game prediction is a JSON object with the following top-level fields:

```json
{
  "season": <int>,
  "week": <int>,
  "game_id": <string>,
  "kickoff_datetime_utc": <string (ISO 8601)>,
  "home_team": <string>,
  "away_team": <string>,
  "market": { ... },
  "model": { ... },
  "edge": { ... }
}
```

### 2.2 Field Definitions

#### Core Fields

**`season`** (required, int):
- NFL season year (e.g., 2023, 2024)
- Range: 2015-present

**`week`** (required, int):
- Week number within season (1-18 for regular season, 19+ for playoffs)
- Regular season: 1-18
- Playoffs: 19 (Wild Card), 20 (Divisional), 21 (Conference), 22 (Super Bowl)

**`game_id`** (required, string):
- Unique identifier for the game
- Format: `nfl_{season}_{week:02d}_{away_team}_{home_team}`
- Example: `"nfl_2023_01_KC_DET"`

**`kickoff_datetime_utc`** (required, string):
- Game kickoff time in UTC
- Format: ISO 8601 (`YYYY-MM-DDTHH:MM:SSZ`)
- Example: `"2023-09-07T20:20:00Z"`

**`home_team`** (required, string):
- Home team abbreviation (3 letters, uppercase)
- Examples: `"KC"`, `"DET"`, `"SF"`, `"BUF"`

**`away_team`** (required, string):
- Away team abbreviation (3 letters, uppercase)
- Examples: `"KC"`, `"DET"`, `"SF"`, `"BUF"`

#### Market Sub-Object

**`market`** (required, object):
- Contains betting market data (closing line)

**Fields**:
- `spread_home` (optional, float): Point spread from home team perspective (negative = home favored)
- `spread_away` (optional, float): Point spread from away team perspective (positive = away favored)
- `moneyline_home` (optional, int): Home team moneyline odds (e.g., -150, +200)
- `moneyline_away` (optional, int): Away team moneyline odds (e.g., -150, +200)
- `implied_prob_home` (optional, float): Market-implied home win probability (0.0-1.0)
- `implied_prob_away` (optional, float): Market-implied away win probability (0.0-1.0)
- `total` (optional, float): Over/under total points
- `market_source` (optional, string): Source of market data (e.g., `"closing_line"`, `"opening_line"`)

**Note**: All market fields are optional. If market data is unavailable, the object may be empty `{}` or contain `null` values.

**Example**:
```json
"market": {
  "spread_home": -3.5,
  "spread_away": 3.5,
  "moneyline_home": -180,
  "moneyline_away": 160,
  "implied_prob_home": 0.643,
  "implied_prob_away": 0.357,
  "total": 48.5,
  "market_source": "closing_line"
}
```

#### Model Sub-Object

**`model`** (required, object):
- Contains model prediction data

**Fields**:
- `version` (required, string): Model/ensemble version identifier
  - Format: `"{experiment_name}_{git_sha_short}"` or `"{model_type}_v{version}"`
  - Example: `"nfl_ensemble_v2_abc1234"` or `"gbm_v1"`
- `prob_home_win` (required, float): Model predicted home win probability (0.0-1.0)
- `prob_away_win` (required, float): Model predicted away win probability (0.0-1.0)
  - Must satisfy: `prob_home_win + prob_away_win = 1.0`
- `calibrated` (required, bool): Whether probabilities have been post-processed with calibration (Platt scaling, isotonic regression)
- `confidence` (optional, float): Model confidence score (0.0-1.0), typically `|prob_home_win - 0.5| * 2`
- `predicted_margin` (optional, float): Predicted point differential (home - away), if model outputs margin
- `predicted_total` (optional, float): Predicted total points, if model outputs total

**Example**:
```json
"model": {
  "version": "nfl_ensemble_v2_abc1234",
  "prob_home_win": 0.68,
  "prob_away_win": 0.32,
  "calibrated": true,
  "confidence": 0.36,
  "predicted_margin": 4.2,
  "predicted_total": 47.8
}
```

#### Edge Sub-Object

**`edge`** (required, object):
- Contains market-relative edge analysis and betting recommendations

**Fields**:
- `vs_market_home` (optional, float): Model edge for home team (`prob_home_win - implied_prob_home`)
- `vs_market_away` (optional, float): Model edge for away team (`prob_away_win - implied_prob_away`)
- `recommended_side` (optional, string): Recommended bet side (`"home"`, `"away"`, or `"none"`)
  - `"none"`: No edge detected (edge below threshold)
- `recommended_stake_flat_units` (optional, float): Recommended stake in units for flat staking (e.g., 1.0)
- `recommended_stake_kelly_fraction` (optional, float): Recommended stake in units using Kelly fraction (e.g., 0.25)
- `edge_magnitude` (optional, float): Absolute value of largest edge (`max(|vs_market_home|, |vs_market_away|)`)
- `edge_threshold_met` (optional, bool): Whether edge exceeds minimum threshold (e.g., 3%)

**Note**: Edge fields are optional and may be `null` if market data is unavailable.

**Example**:
```json
"edge": {
  "vs_market_home": 0.037,
  "vs_market_away": -0.037,
  "recommended_side": "home",
  "recommended_stake_flat_units": 1.0,
  "recommended_stake_kelly_fraction": 0.15,
  "edge_magnitude": 0.037,
  "edge_threshold_met": true
}
```

### 2.3 Complete Example

```json
{
  "season": 2023,
  "week": 1,
  "game_id": "nfl_2023_01_KC_DET",
  "kickoff_datetime_utc": "2023-09-07T20:20:00Z",
  "home_team": "DET",
  "away_team": "KC",
  "market": {
    "spread_home": 3.5,
    "spread_away": -3.5,
    "moneyline_home": 160,
    "moneyline_away": -180,
    "implied_prob_home": 0.357,
    "implied_prob_away": 0.643,
    "total": 54.5,
    "market_source": "closing_line"
  },
  "model": {
    "version": "nfl_ensemble_v2_abc1234",
    "prob_home_win": 0.42,
    "prob_away_win": 0.58,
    "calibrated": true,
    "confidence": 0.16,
    "predicted_margin": -2.1,
    "predicted_total": 52.3
  },
  "edge": {
    "vs_market_home": 0.063,
    "vs_market_away": -0.063,
    "recommended_side": "home",
    "recommended_stake_flat_units": 1.0,
    "recommended_stake_kelly_fraction": 0.22,
    "edge_magnitude": 0.063,
    "edge_threshold_met": true
  }
}
```

---

## 3. Weekly Batch Structure

### 3.1 Structure

A weekly batch contains all predictions for a single week, plus metadata.

```json
{
  "season": <int>,
  "week": <int>,
  "model_version": <string>,
  "git_sha": <string>,
  "generated_at_utc": <string (ISO 8601)>,
  "n_games": <int>,
  "games": [ ... ]
}
```

### 3.2 Field Definitions

**`season`** (required, int): NFL season year

**`week`** (required, int): Week number

**`model_version`** (required, string): Model version identifier (same as `model.version` in per-game objects)

**`git_sha`** (required, string): Full Git commit SHA of the codebase used to generate predictions
- Format: 40-character hex string
- Example: `"a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8s9t0"`

**`generated_at_utc`** (required, string): Timestamp when batch was generated (ISO 8601)

**`n_games`** (required, int): Number of games in the batch

**`games`** (required, array): Array of per-game prediction objects (see Section 2)

### 3.3 Complete Example

```json
{
  "season": 2023,
  "week": 1,
  "model_version": "nfl_ensemble_v2_abc1234",
  "git_sha": "a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8s9t0",
  "generated_at_utc": "2023-09-05T12:00:00Z",
  "n_games": 16,
  "games": [
    {
      "season": 2023,
      "week": 1,
      "game_id": "nfl_2023_01_KC_DET",
      "kickoff_datetime_utc": "2023-09-07T20:20:00Z",
      "home_team": "DET",
      "away_team": "KC",
      "market": { ... },
      "model": { ... },
      "edge": { ... }
    },
    {
      "season": 2023,
      "week": 1,
      "game_id": "nfl_2023_01_BUF_NYJ",
      "kickoff_datetime_utc": "2023-09-11T17:00:00Z",
      "home_team": "NYJ",
      "away_team": "BUF",
      "market": { ... },
      "model": { ... },
      "edge": { ... }
    }
    // ... 14 more games
  ]
}
```

---

## 4. Required vs Optional Fields

### 4.1 Required Fields

**Per-Game Object**:
- `season`, `week`, `game_id`, `kickoff_datetime_utc`, `home_team`, `away_team`
- `model.version`, `model.prob_home_win`, `model.prob_away_win`, `model.calibrated`

**Weekly Batch Object**:
- `season`, `week`, `model_version`, `git_sha`, `generated_at_utc`, `n_games`, `games`

### 4.2 Optional Fields

**Market Fields**: All fields in `market` object are optional (market data may be unavailable).

**Edge Fields**: All fields in `edge` object are optional (depend on market data availability).

**Model Fields**: `confidence`, `predicted_margin`, `predicted_total` are optional (model-dependent).

### 4.3 Handling Missing Data

**Market Data Missing**:
- Set `market` object to `{}` (empty object) or include fields with `null` values
- Set `edge` object to `{}` or `null`
- Set `recommended_side` to `"none"` if edge cannot be computed

**Model Output Missing**:
- If `prob_home_win` or `prob_away_win` is missing, the entire prediction is invalid
- Do not include invalid predictions in weekly batch

**Partial Market Data**:
- Include available fields (e.g., `spread_home` but no `moneyline_home`)
- Compute `implied_prob_home` from available data (spread takes priority over moneyline)
- If no market data available, omit `edge` object entirely

---

## 5. Parquet/Tabular Equivalent

### 5.1 Flattened Representation

For bulk analysis, predictions can be stored in Parquet/CSV with flattened columns.

**Core Columns**:
- `season` (int)
- `week` (int)
- `game_id` (string)
- `kickoff_datetime_utc` (datetime)
- `home_team` (string)
- `away_team` (string)

**Market Columns** (prefix: `market_`):
- `market_spread_home` (float, nullable)
- `market_spread_away` (float, nullable)
- `market_moneyline_home` (int, nullable)
- `market_moneyline_away` (int, nullable)
- `market_implied_prob_home` (float, nullable)
- `market_implied_prob_away` (float, nullable)
- `market_total` (float, nullable)

**Model Columns** (prefix: `model_`):
- `model_version` (string)
- `model_prob_home_win` (float)
- `model_prob_away_win` (float)
- `model_calibrated` (bool)
- `model_confidence` (float, nullable)
- `model_predicted_margin` (float, nullable)
- `model_predicted_total` (float, nullable)

**Edge Columns** (prefix: `edge_`):
- `edge_vs_market_home` (float, nullable)
- `edge_vs_market_away` (float, nullable)
- `edge_recommended_side` (string, nullable)
- `edge_recommended_stake_flat_units` (float, nullable)
- `edge_recommended_stake_kelly_fraction` (float, nullable)
- `edge_magnitude` (float, nullable)

**Metadata Columns**:
- `git_sha` (string)
- `generated_at_utc` (datetime)

### 5.2 Normalized Representation (Future)

For database storage, consider normalized representation:

**Table: `predictions`**
- Primary key: `game_id`, `model_version`
- Columns: Core fields + model fields

**Table: `markets`**
- Primary key: `game_id`, `market_source`
- Columns: Market fields

**Table: `edges`**
- Primary key: `game_id`, `model_version`
- Foreign key: `predictions.game_id`, `predictions.model_version`
- Columns: Edge fields

**Join**: `predictions` LEFT JOIN `markets` LEFT JOIN `edges`

---

## 6. Versioning & Backward Compatibility

### 6.1 Versioning Strategy

**Model Version**: Tracked in `model.version` field
- Format: `"{experiment_name}_{git_sha_short}"`
- Example: `"nfl_ensemble_v2_abc1234"`

**Contract Version**: Tracked in `model_version` metadata (future)
- Format: `"v{major}.{minor}"`
- Example: `"v1.0"`

### 6.2 Backward Compatibility

**Non-Breaking Changes** (backward compatible):
- Adding new optional fields
- Adding new optional sub-objects
- Expanding allowed values for enum fields (e.g., adding new `recommended_side` values)

**Breaking Changes** (require version increment):
- Removing required fields
- Changing field types (e.g., `int` → `float`)
- Changing field names
- Changing structure of required sub-objects

### 6.3 Migration Plan

**When Making Breaking Changes**:
1. Increment contract version (e.g., v1.0 → v2.0)
2. Document changes in this contract document
3. Provide migration script or mapping for converting v1.0 → v2.0
4. Support both versions during transition period
5. Deprecate old version after transition period

**Example Migration**:
- v1.0: `model.prob_home_win` (required)
- v2.0: `model.probabilities.home_win` (nested structure)
- Migration: Extract `prob_home_win` and nest under `probabilities` object

---

## 7. Validation Rules

### 7.1 Data Validation

**Probability Constraints**:
- `prob_home_win` and `prob_away_win` must be in [0.0, 1.0]
- `prob_home_win + prob_away_win` must equal 1.0 (within floating-point tolerance, e.g., 1e-6)

**Team Abbreviations**:
- Must be 3 uppercase letters
- Must be valid NFL team abbreviations (validate against known list)

**Game ID Format**:
- Must match pattern: `nfl_{season}_{week:02d}_{away_team}_{home_team}`
- Season must be >= 2015
- Week must be in [1, 22] (regular season + playoffs)

**Datetime Format**:
- Must be valid ISO 8601 format
- Must be in UTC timezone

**Edge Constraints**:
- `vs_market_home = prob_home_win - implied_prob_home` (if both available)
- `vs_market_away = prob_away_win - implied_prob_away` (if both available)
- `edge_magnitude = max(|vs_market_home|, |vs_market_away|)` (if both available)

### 7.2 Business Logic Validation

**Recommended Side**:
- If `edge_magnitude < threshold`, `recommended_side` must be `"none"`
- If `vs_market_home > threshold`, `recommended_side` should be `"home"`
- If `vs_market_away > threshold`, `recommended_side` should be `"away"`

**Stake Recommendations**:
- `recommended_stake_flat_units` must be >= 0
- `recommended_stake_kelly_fraction` must be >= 0
- If `recommended_side = "none"`, stakes should be 0.0

---

## 8. API Considerations (Future)

### 8.1 REST API Endpoints

**Single Game Prediction**:
- `GET /api/v1/predictions/{game_id}`
- Returns: Per-game JSON object (Section 2)

**Weekly Predictions**:
- `GET /api/v1/predictions/season/{season}/week/{week}`
- Returns: Weekly batch JSON object (Section 3)

**Latest Predictions**:
- `GET /api/v1/predictions/latest`
- Returns: Weekly batch for current week

### 8.2 Response Format

**Success Response**:
- Status: `200 OK`
- Content-Type: `application/json`
- Body: JSON object (per-game or weekly batch)

**Error Response**:
- Status: `404 Not Found` (game not found), `400 Bad Request` (invalid parameters)
- Content-Type: `application/json`
- Body: `{"error": "error_message", "code": "ERROR_CODE"}`

---

*This contract is version 1.0. Updates will be versioned and documented.*

