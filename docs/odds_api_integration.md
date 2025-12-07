# The Odds API Integration

**Date**: 2024-12-XX  
**Change**: Added The Odds API support to NFL odds ingestion module

---

## Overview

The `ingestion/nfl/odds.py` module now supports fetching NFL odds from **The Odds API** (free tier: 500 requests/month) as the primary data source, with CSV and nflverse as fallbacks.

---

## Changes Made

### 1. Added The Odds API Integration

**File**: `ingestion/nfl/odds.py`

**New Functions**:
- `map_odds_api_team_to_abbreviation()`: Maps The Odds API team names to our normalized abbreviations
- `fetch_odds_api_historical()`: Fetches historical NFL odds from The Odds API
- `_parse_odds_api_game()`: Parses individual game data from API response

**Features**:
- ✅ Rate limit handling (429 status codes)
- ✅ Request counting to stay under 500/month limit
- ✅ Smart date-based fetching (focuses on game days: Thu, Sun, Mon)
- ✅ Automatic fallback to CSV/nflverse if API fails
- ✅ Comprehensive logging of fallback triggers

### 2. Refactored `ingest_nfl_odds()`

**Priority Order**:
1. **The Odds API** (primary source)
2. **CSV file** (manual fallback)
3. **nflverse** (last resort)

**Logging**:
- ✅ Logs when API is attempted
- ✅ Logs when fallback is triggered
- ✅ Logs which source was ultimately used
- ✅ Warns about rate limits and low remaining requests

### 3. Updated Configuration

**File**: `config/data/nfl.yaml`

**New Configuration Section**:
```yaml
odds:
  the_odds_api:
    enabled: true  # Set to false to skip API
    api_key: ""  # Your API key (leave empty to disable)
    regions: "us"  # Comma-separated regions
    markets: "spreads,totals"  # Comma-separated markets
```

### 4. Updated Dependencies

**File**: `requirements.txt`

**Added**:
- `requests>=2.31.0` (for API calls)

---

## Usage

### Setup

1. **Get API Key**:
   - Register at https://the-odds-api.com/
   - Free tier: 500 requests/month

2. **Configure API Key**:
   ```yaml
   # config/data/nfl.yaml
   nfl:
     odds:
       the_odds_api:
         api_key: "your_api_key_here"
   ```

3. **Install Dependencies**:
   ```bash
   pip install requests
   ```

### Running Ingestion

```python
from ingestion.nfl.odds import ingest_nfl_odds

# Will try API first, then CSV, then nflverse
df = ingest_nfl_odds(seasons=[2023, 2024])
```

Or via CLI:
```bash
python -m ingestion.nfl.odds
```

### Disabling API (Use CSV/nflverse Only)

```yaml
# config/data/nfl.yaml
nfl:
  odds:
    the_odds_api:
      enabled: false
```

Or leave `api_key` empty:
```yaml
nfl:
  odds:
    the_odds_api:
      api_key: ""  # Empty = disabled
```

---

## API Rate Limits

**Free Tier**: 500 requests/month

**Rate Limit Handling**:
- ✅ Tracks request count (stops at 450 to leave buffer)
- ✅ Handles 429 status codes gracefully
- ✅ Logs warnings when approaching limits
- ✅ Automatically falls back to CSV/nflverse if limit reached

**Request Optimization**:
- Fetches odds only on game days (Thu, Sun, Mon, Sat)
- Uses 300ms delay between requests
- Stops early if approaching rate limit

---

## Fallback Behavior

The ingestion function tries sources in this order:

1. **The Odds API**
   - ✅ If API key configured and enabled
   - ✅ If API returns data successfully
   - ⚠️ Falls back if: API key missing, rate limit hit, API error, no data returned

2. **CSV File**
   - ✅ If `csv_path` provided and file exists
   - ⚠️ Falls back if: File not found, parsing error

3. **nflverse**
   - ✅ Last resort fallback
   - ✅ Extracts odds from schedule data if available
   - ⚠️ May not have odds for all seasons

**Logging Example**:
```
INFO: Attempting to fetch odds from The Odds API...
INFO: ✅ Successfully fetched 256 games from The Odds API
INFO: 📊 Using odds data from: the_odds_api (256 games)
```

Or if fallback triggered:
```
INFO: Attempting to fetch odds from The Odds API...
WARNING: ⚠️ The Odds API returned no data. Falling back to CSV/nflverse...
INFO: Falling back to CSV file: data/nfl/raw/odds.csv
INFO: ✅ Successfully loaded 256 games from CSV
INFO: 📊 Using odds data from: csv (256 games)
```

---

## Team Name Mapping

The Odds API uses full team names (e.g., "Kansas City Chiefs"), while we use 3-letter abbreviations (e.g., "KC"). The integration includes comprehensive mapping:

- ✅ All 32 current NFL teams
- ✅ Historical team names (Oakland Raiders → LV, San Diego Chargers → LAC, etc.)
- ✅ Case-insensitive matching
- ✅ Warning logs for unmapped teams

---

## Data Format

The API returns odds in this format:
- **Spread**: From home team perspective (positive = home favored)
- **Total**: Over/under point total
- **Opening/Closing**: Extracted from first/last bookmaker

Data is normalized to our `MarketSnapshot` schema:
- `game_id`: `nfl_{season}_{week}_{away}_{home}`
- `close_spread`: Closing spread
- `close_total`: Closing total
- `open_spread`: Opening spread (if available)
- `open_total`: Opening total (if available)

---

## Error Handling

**Graceful Degradation**:
- ✅ API errors don't crash the pipeline
- ✅ Falls back to CSV/nflverse automatically
- ✅ Comprehensive error logging
- ✅ Clear error messages if all sources fail

**Common Scenarios**:
1. **No API Key**: Falls back to CSV/nflverse, logs warning
2. **Rate Limit Hit**: Stops API fetch, uses available data, falls back if needed
3. **API Error**: Logs error, falls back to CSV/nflverse
4. **No Data Returned**: Falls back to CSV/nflverse
5. **All Sources Fail**: Raises clear error with instructions

---

## Testing

To test the integration:

```python
from ingestion.nfl.odds import ingest_nfl_odds

# Test with API (if configured)
try:
    df = ingest_nfl_odds(seasons=[2023])
    print(f"✅ Successfully ingested {len(df)} games")
except Exception as e:
    print(f"❌ Error: {e}")
```

---

## Future Enhancements

Potential improvements:
- [ ] Cache API responses to reduce requests
- [ ] Support for more markets (moneylines, props)
- [ ] Support for multiple bookmakers
- [ ] Historical odds archive integration
- [ ] Real-time odds updates (for in-season)

---

*End of Documentation*

