# NFL Data Sources Documentation

This document describes the data sources used for NFL data ingestion in Phase 1A, including URLs, data caveats, and assumptions.

## Schedule and Results Data

### Source: nflverse (nfl_data_py)

- **Package**: `nfl-data-py` (Python wrapper for nflverse)
- **Installation**: `pip install nfl-data-py`
- **Documentation**: https://github.com/nflverse/nfl-data-py
- **Data Coverage**: NFL schedules and results from 1999-present

#### Usage
```python
import nfl_data_py as nfl
schedule = nfl.import_schedules([2023])
```

#### Data Fields Used
- `season`: NFL season year
- `week`: Week number (1-18 for regular season, 19+ for playoffs)
- `gameday`: Game date
- `home_team`: Home team abbreviation (3-letter code)
- `away_team`: Away team abbreviation (3-letter code)
- `home_score`: Final home team score
- `away_score`: Final away team score

#### Team Abbreviation Normalization

nflverse uses standard 3-letter NFL team abbreviations. We normalize team names to match this standard:

- **Relocations handled**:
  - `OAK` → `LV` (Raiders: Oakland → Las Vegas, 2020)
  - `SD` → `LAC` (Chargers: San Diego → Los Angeles, 2017)
  - `STL` → `LAR` (Rams: St. Louis → Los Angeles, 2016)

- **Standard abbreviations**: ARI, ATL, BAL, BUF, CAR, CHI, CIN, CLE, DAL, DEN, DET, GB, HOU, IND, JAX, KC, LV, LAR, LAC, MIA, MIN, NE, NO, NYG, NYJ, PHI, PIT, SF, SEA, TB, TEN, WAS

#### Game ID Format

Games are identified using the format:
```
nfl_{season}_{week:02d}_{away_team}_{home_team}
```

Example: `nfl_2023_01_KC_DET` (Kansas City @ Detroit, Week 1, 2023)

#### Data Caveats

1. **Playoff Weeks**: Playoff weeks are numbered 19+ (Wild Card = 19, Divisional = 20, Conference = 21, Super Bowl = 22)
2. **Bye Weeks**: Teams on bye weeks are not included in schedule data
3. **Postponed Games**: Some games may be postponed and rescheduled; check `gameday` field for actual date
4. **Score Availability**: Final scores may not be available immediately after games; data is typically updated within 24 hours

---

## Betting Odds Data

### Source Options

#### Option 1: nflverse Schedule Data (if available)

nflverse schedule data may include spread and total columns if the data source provides them. The `odds.py` module attempts to extract these fields from schedule data.

**Limitations**: 
- Not all nflverse schedule data includes odds
- May only include closing lines, not opening lines
- May not include moneyline odds

#### Option 2: CSV File Input

If nflverse does not provide odds data, you can provide a CSV file with historical odds.

**Required CSV Format**:
```csv
season,week,away_team,home_team,close_spread,close_total,open_spread,open_total
2023,1,KC,DET,-3.0,45.5,-2.5,46.0
2023,1,GB,CHI,7.0,48.0,6.5,47.5
```

**Required Columns**:
- `season`: NFL season year
- `week`: Week number
- `away_team`: Away team abbreviation (3-letter code)
- `home_team`: Home team abbreviation (3-letter code)
- `close_spread`: Closing point spread (from home team perspective, negative = home favored)
- `close_total`: Closing over/under total

**Optional Columns**:
- `open_spread`: Opening point spread
- `open_total`: Opening over/under total

#### Option 3: Free Historical Odds Sources

**SportsOddsHistory.com**
- URL: https://www.sportsoddshistory.com/nfl-game-odds/
- Provides historical NFL game odds
- May require web scraping to extract data
- Data format varies by season

**Other Free Sources**:
- Odds Portal (may require scraping)
- Historical odds archives from various sportsbooks (publicly available)

### Spread Direction Convention

**Spread is from home team perspective**:
- **Negative spread** (e.g., -3.0): Home team is favored by 3 points
- **Positive spread** (e.g., +3.0): Away team is favored by 3 points (home team is underdog)

**Example**:
- `close_spread = -3.0` means home team is favored by 3 points
- If home team wins by 4+, they cover the spread
- If home team wins by 2 or less (or loses), they do not cover

### Data Assumptions

1. **Bookmaker Selection**: If multiple bookmakers are available, we use:
   - Average of available books (if `preferred_book: "average"` in config)
   - Or a specific book if only one source is available

2. **Closing Line Definition**: 
   - Closing line is defined as the last available line before game start
   - Typically from major sportsbooks (Vegas consensus or similar)

3. **Missing Data Handling**:
   - Games without odds data are excluded from `games_markets.parquet`
   - Validation will flag games missing market data

4. **Spread Sanity Checks**:
   - Spreads typically range from -20 to +20 points
   - Extreme spreads (>20 points) are flagged but not excluded
   - Spread direction should generally align with actual game outcomes

---

## Data Quality Validation

### Validation Checks Performed

1. **Game ID Format**: Validates format `nfl_{season}_{week}_{away}_{home}`
2. **No Duplicates**: Ensures each game_id appears only once
3. **Season Coverage**: Verifies all required seasons (2015-2024) are present
4. **Market Completeness**: Ensures every game has a matching market entry
5. **Spread Sanity**: Validates spread direction makes sense relative to game outcomes

### Validation Failures

If validation fails:
- **Missing Markets**: Games without odds will be logged but join will proceed with available data
- **Invalid Game IDs**: Invalid formats will be logged and may cause join failures
- **Missing Seasons**: Missing seasons will be logged; pipeline may fail if critical seasons are missing

---

## Data Storage

### Raw Data
- Location: `data/nfl/raw/`
- Files:
  - `schedules.parquet`: Raw schedule data from nflverse
  - `odds.parquet`: Raw odds data (if from CSV or other source)

### Staged Data
- Location: `data/nfl/staged/`
- Files:
  - `games.parquet`: Normalized Game schema (schedule + results)
  - `markets.parquet`: Normalized MarketSnapshot schema (odds)
  - `games_markets.parquet`: Joined games and markets data

### Data Schema

#### games.parquet Schema
```
game_id: string (format: nfl_{season}_{week}_{away}_{home})
season: int
week: int
date: datetime
home_team: string (3-letter abbreviation)
away_team: string (3-letter abbreviation)
home_score: int
away_score: int
```

#### markets.parquet Schema
```
game_id: string (matches games.parquet)
season: int
week: int
close_spread: float (from home team perspective)
close_total: float
open_spread: float (optional)
open_total: float (optional)
```

---

## Usage Instructions

### Running Schedule Ingestion

```bash
cd /home/dp/Documents/predictionV2
python -m ingestion.nfl.schedule
```

### Running Odds Ingestion

**Option 1: Using CSV file**
```bash
python -m ingestion.nfl.odds /path/to/odds.csv
```

**Option 2: Using nflverse (if available)**
```python
from ingestion.nfl.odds import ingest_nfl_odds
df = ingest_nfl_odds()
```

### Running Join Step

```bash
python -m ingestion.nfl.join_games_markets
```

### Running Tests

```bash
pytest tests/test_phase1a_ingestion.py -v
```

---

## Future Data Sources (Phase 2+)

### Play-by-Play Data
- **Source**: nflverse/nflfastR
- **Package**: `nfl-data-py` or `nflfastR` (R package)
- **Coverage**: 1999-present with EPA, success rate, CPOE metrics

### Player Stats
- **Source**: nflverse
- **Coverage**: Player-level statistics, rosters, snap counts

### Injury Reports
- **Source**: NFL.com official injury reports (requires scraping)
- **Coverage**: 2020-present (earlier data may be limited)

---

## Notes

- All data sources are free/open and do not require API keys or paid subscriptions
- Data may have slight delays (scores updated within 24 hours of games)
- Historical odds data may require manual collection or scraping
- Team abbreviations are normalized to nflverse standard for consistency

