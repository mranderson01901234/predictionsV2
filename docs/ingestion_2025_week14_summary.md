# 2025 Week 14 (DET @ DAL) Ingestion Summary

**Date**: 2025-12-07  
**Game**: Detroit Lions @ Dallas Cowboys  
**Game ID**: `nfl_2025_14_DET_DAL`

## Status

### ✅ Schedule Ingestion - COMPLETE

- **Status**: Game successfully added to schedule
- **Location**: `data/nfl/staged/games.parquet`
- **Details**:
  - Season: 2025
  - Week: 14
  - Home Team: DAL (Dallas Cowboys)
  - Away Team: DET (Detroit Lions)
  - Date: 2025-12-07

**Verification**:
```python
import pandas as pd
games = pd.read_parquet('data/nfl/staged/games.parquet')
game = games[games['game_id'] == 'nfl_2025_14_DET_DAL']
# Game found: nfl_2025_14_DET_DAL
```

### ⚠️ Odds Ingestion - PARTIAL

- **Status**: Game in joined data, but odds not available from The Odds API
- **Location**: `data/nfl/staged/games_markets.parquet`
- **Details**:
  - Close Spread: `NaN` (not available)
  - Close Total: `NaN` (not available)
  - Open Spread: `NaN` (not available)
  - Open Total: `NaN` (not available)

**Reason**: The Odds API does not have historical odds data for 2025 games yet (too far in the future). The API returned 0 games for season 2025.

**Next Steps**:
- Odds will become available closer to game date (typically 1-2 weeks before)
- Can manually add odds when they become available
- Or re-run ingestion script closer to game date

**Verification**:
```python
import pandas as pd
joined = pd.read_parquet('data/nfl/staged/games_markets.parquet')
game = joined[joined['game_id'] == 'nfl_2025_14_DET_DAL']
# Game found with NaN odds (expected for future games)
```

### ⚠️ Feature Generation - NOT AVAILABLE

- **Status**: Features cannot be generated yet
- **Reason**: Team form features require historical team stats (rolling windows of 4, 8, 16 games)
- **Details**:
  - 2025 games have not been played yet
  - No team_stats.parquet entries for 2025
  - Feature pipeline drops games without team stats

**Current Behavior**:
- Feature pipeline runs but drops all 2025 games (273 games missing team features)
- Final feature table: 0 games (only historical games remain)

**Options for Future**:
1. **Wait for games to be played**: As 2025 games are played, team stats will accumulate and features can be generated
2. **Use 2024 season-end stats**: Could modify feature pipeline to use most recent available stats as placeholders
3. **Manual feature creation**: Create features manually using 2024 season-end data

## Files Updated

1. **`config/data/nfl.yaml`**
   - Added 2025 to seasons list

2. **`data/nfl/staged/games.parquet`**
   - Added `nfl_2025_14_DET_DAL` game
   - Total games: 273 (all 2025 games from nflverse)

3. **`data/nfl/staged/games_markets.parquet`**
   - Added `nfl_2025_14_DET_DAL` with NaN odds
   - Total games: 273

4. **`data/nfl/processed/game_features_baseline.parquet`**
   - Refreshed but contains 0 games (all 2025 games dropped due to missing team stats)

## Usage

### Check Schedule

```python
import pandas as pd
games = pd.read_parquet('data/nfl/staged/games.parquet')
game = games[games['game_id'] == 'nfl_2025_14_DET_DAL']
print(game[['game_id', 'season', 'week', 'home_team', 'away_team', 'date']])
```

### Check Joined Data

```python
import pandas as pd
joined = pd.read_parquet('data/nfl/staged/games_markets.parquet')
game = joined[joined['game_id'] == 'nfl_2025_14_DET_DAL']
print(game[['game_id', 'home_team', 'away_team', 'close_spread', 'close_total']])
```

### Generate Features (when team stats available)

```bash
# Once 2025 games start being played and team_stats.parquet is updated
python3 scripts/generate_features.py --seasons 2025 --weeks 14
```

## Next Steps

1. **Monitor Odds Availability**:
   - Check The Odds API closer to game date (1-2 weeks before)
   - Re-run ingestion script: `python3 scripts/ingest_2025_week14.py`

2. **Generate Features**:
   - Once 2025 games are played and team_stats.parquet includes 2025 data
   - Run: `python3 scripts/generate_features.py --seasons 2025 --weeks 14`

3. **Run Simulation**:
   - Once features are available:
   ```bash
   python3 scripts/simulate_real_world_prediction.py --game-id 2025_WK14_DAL_DET
   ```

## Notes

- The Odds API free tier has a 500 requests/month limit
- Historical odds for future games are typically not available until closer to game date
- Feature generation requires historical game data (team stats) which won't exist until games are played
- The schedule ingestion successfully pulled all 2025 games from nflverse (272 games + our manually added game = 273 total)

