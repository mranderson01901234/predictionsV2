# Real-World Prediction Simulation Script

## Overview

The `simulate_real_world_prediction.py` script simulates a real-world prediction for a specific NFL game using only pre-game data. It loads the trained ensemble model, runs inference, and compares predictions against actual results.

## Usage

### Basic Usage

```bash
# Use default game (2025 Week 14: Detroit @ Dallas)
python3 scripts/simulate_real_world_prediction.py

# Specify a game
python3 scripts/simulate_real_world_prediction.py --game-id 2025_WK14_DAL_DET

# Use our format
python3 scripts/simulate_real_world_prediction.py --game-id nfl_2025_14_DET_DAL
```

### Command Line Arguments

- `--game-id`: Game ID in format `2025_WK14_DAL_DET` (HOME_AWAY) or `nfl_2025_14_DET_DAL` (AWAY_HOME)
  - Default: `2025_WK14_DAL_DET`
  - User format: `SEASON_WK##_HOME_AWAY` (e.g., `2025_WK14_DAL_DET` = Dallas home vs Detroit away)
  - Our format: `nfl_SEASON_WEEK_AWAY_HOME` (e.g., `nfl_2025_14_DET_DAL` = Detroit away @ Dallas home)

- `--model-path`: Path to ensemble model (default: uses v2 ensemble from `artifacts/models/nfl_stacked_ensemble_v2/`)

- `--feature-table`: Feature table name (default: `baseline`)

- `--log-path`: Path to log file (default: `logs/simulations/predictions_vs_actuals.log`)

## Output

The script outputs:

1. **Predicted Winner**: Team predicted to win
2. **Confidence**: Prediction confidence (max of prob or 1-prob)
3. **Probability**: Home team win probability
4. **Estimated Spread**: Estimated point spread (home perspective)
5. **Actual Results**: Actual winner, score, and spread (if available)
6. **Prediction Accuracy**: Whether prediction was correct
7. **Spread Error**: Difference between predicted and actual spread

### Example Output

```
================================================================================
PREDICTION SIMULATION SUMMARY
================================================================================
Game ID: nfl_2023_14_PHI_DAL
Matchup: PHI @ DAL
Season: 2023, Week: 14

Prediction:
  Winner: DAL
  Confidence: 57.65%
  Probability (Home Win): 0.5765
  Estimated Spread: -0.9

Actual Result:
  Score: PHI 13 - 33 DAL
  Winner: DAL
  Spread: +20.0

Prediction Accuracy: ✓ CORRECT
Spread Error: 20.9 points

Logged to: /home/dp/Documents/predictionV2/logs/simulations/predictions_vs_actuals.log
================================================================================
```

## Logging

Results are logged to `logs/simulations/predictions_vs_actuals.log` in CSV format with columns:

- `timestamp`: When prediction was made
- `game_id`: Game identifier
- `home_team`: Home team abbreviation
- `away_team`: Away team abbreviation
- `predicted_winner`: Predicted winner
- `predicted_prob`: Predicted home win probability
- `predicted_spread`: Estimated spread
- `actual_winner`: Actual winner (N/A if game not played)
- `actual_home_score`: Actual home score (N/A if game not played)
- `actual_away_score`: Actual away score (N/A if game not played)
- `actual_spread`: Actual spread (N/A if game not played)
- `correct`: Whether prediction was correct (N/A if game not played)

## Requirements

1. **Game must exist in feature table**: The game must have features generated
2. **Ensemble model must be trained**: Model file must exist at default path or specified path
3. **Feature table must exist**: Baseline feature table must be available

## Limitations

- **Future games**: Games that haven't been played yet (e.g., 2025 Week 14) won't have features unless generated
- **Spread estimation**: Spread prediction is a rough approximation based on probability
- **Pre-game data only**: Uses only data available before kickoff (no post-game information)

## Examples

### Predict a specific game

```bash
python3 scripts/simulate_real_world_prediction.py --game-id 2023_WK14_DAL_PHI
```

### Use different model

```bash
python3 scripts/simulate_real_world_prediction.py \
  --game-id 2023_WK14_DAL_PHI \
  --model-path artifacts/models/nfl_stacked_ensemble/ensemble_v1.pkl
```

### Custom log location

```bash
python3 scripts/simulate_real_world_prediction.py \
  --game-id 2023_WK14_DAL_PHI \
  --log-path logs/custom_predictions.log
```

## Notes

- The script uses the **v2 ensemble** (FT-Transformer + GBM) by default
- Falls back to **v1 ensemble** if v2 doesn't exist
- Handles CUDA incompatibility automatically (falls back to CPU)
- Spread estimation uses a simple logit-based approximation

