# Feature Generation Script Usage

## Overview

The `generate_features.py` script generates features for specific seasons and weeks for prediction and simulation. It loads schedule and team-level data (odds, rosters, injuries, stats) and generates full feature tables using the production feature pipeline.

## Usage

### Basic Usage

```bash
# Generate features for 2025 season, weeks 1-18
python3 scripts/generate_features.py --seasons 2025 --weeks 1-18

# Generate features for multiple seasons, specific weeks
python3 scripts/generate_features.py --seasons 2023,2024 --weeks 14-18

# Generate features for single week
python3 scripts/generate_features.py --seasons 2025 --weeks 14

# Generate features for playoffs
python3 scripts/generate_features.py --seasons 2024 --weeks 19-22
```

### Command Line Arguments

- `--seasons` (required): Comma-separated list of seasons
  - Examples: `"2025"`, `"2023,2024,2025"`
  
- `--weeks` (required): Weeks to process
  - Single week: `"14"`
  - Range: `"1-18"`
  - List: `"14,15,16"`
  - Combination: `"1-18,20-22"` (for regular season + playoffs)

- `--output-dir` (optional): Output directory
  - Default: `data/nfl/features/`
  - Custom: `--output-dir /path/to/output`

- `--feature-table` (optional): Feature table type
  - Options: `baseline`, `phase2`, `phase2b`
  - Default: `baseline`

## Output

### File Format

Features are saved as parquet files with naming convention:
```
features_{season}_wk{week:02d}.parquet
```

Examples:
- `features_2025_wk14.parquet` - 2025 season, week 14
- `features_2023_wk01.parquet` - 2023 season, week 1

### Output Location

Default: `data/nfl/features/`

### File Contents

Each file contains:
- Game metadata: `game_id`, `season`, `week`, `date`, `home_team`, `away_team`
- Scores: `home_score`, `away_score` (if game has been played)
- Market data: `close_spread`, `close_total`, `open_spread`, `open_total`
- Team features: `home_*` and `away_*` prefixed features
  - Win rates (last 4, 8, 16 games)
  - Point differentials (last 4, 8, 16 games)
  - Points for/against (last 4, 8, 16 games)
  - Turnover differentials (last 4, 8, 16 games)

## Data Requirements

The script requires:

1. **Schedule data**: `data/nfl/staged/games.parquet`
   - Must contain games for specified seasons/weeks

2. **Market data** (optional): `data/nfl/staged/markets.parquet`
   - Odds data (spreads, totals)
   - Missing odds are logged but don't block generation

3. **Team stats**: `data/nfl/staged/team_stats.parquet`
   - Required for generating team form features
   - Missing stats are logged but features use defaults (0)

## Logging

The script provides detailed logging:

- **Data availability**: Reports how many games have schedule, odds, and stats
- **Missing data**: Lists games missing odds or team stats
- **Generation status**: Reports success/failure for each season/week
- **Summary**: Final summary of files created and games processed

### Example Output

```
================================================================================
Feature Generation Summary
================================================================================

Successfully generated features:
  Files created: 2
  Total games: 29

  Per file breakdown:
    2023 Week 14: 14 games -> features_2023_wk14.parquet
    2023 Week 15: 15 games -> features_2023_wk15.parquet

Output directory: /home/dp/Documents/predictionV2/data/nfl/features
================================================================================
```

## Use Cases

### 1. Prepare Games for Simulation

Generate features for future games (e.g., 2025 Week 14):

```bash
python3 scripts/generate_features.py --seasons 2025 --weeks 14
```

Then use in simulation:
```bash
python3 scripts/simulate_real_world_prediction.py --game-id 2025_WK14_DAL_DET
```

### 2. Batch Prediction

Generate features for multiple weeks:

```bash
python3 scripts/generate_features.py --seasons 2025 --weeks 1-18
```

Then load features for batch prediction:
```python
import pandas as pd
features = pd.read_parquet('data/nfl/features/features_2025_wk14.parquet')
# Run predictions on all games
```

### 3. Historical Analysis

Generate features for historical seasons:

```bash
python3 scripts/generate_features.py --seasons 2020,2021,2022,2023 --weeks 1-18
```

## Integration with Simulation Script

The generated feature files can be used with the simulation script by ensuring the feature table includes the games:

1. Generate features for the game:
   ```bash
   python3 scripts/generate_features.py --seasons 2025 --weeks 14
   ```

2. Merge into main feature table (or update feature table registry):
   - The simulation script uses `game_features_baseline.parquet`
   - You may need to merge weekly files into the main table

3. Run simulation:
   ```bash
   python3 scripts/simulate_real_world_prediction.py --game-id 2025_WK14_DAL_DET
   ```

## Notes

- **Missing data handling**: Games with missing odds or stats are still processed (missing values filled with 0)
- **Caching**: Team form features are cached if already generated
- **Performance**: Processing is efficient for large ranges (e.g., full season)
- **File naming**: Week numbers are zero-padded (e.g., `wk01`, `wk14`)

## Troubleshooting

### No games found
- Check that games exist in `data/nfl/staged/games.parquet` for specified seasons/weeks
- Verify season/week format is correct

### Missing team stats
- Ensure `data/nfl/staged/team_stats.parquet` exists
- Run team stats ingestion pipeline if needed

### Missing odds
- Odds are optional - features will still be generated
- Check `data/nfl/staged/markets.parquet` exists
- Run odds ingestion if needed

