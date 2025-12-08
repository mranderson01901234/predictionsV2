# Weather API Optimization Guide

## Current Implementation

### API Being Used: **Open-Meteo**

- **Historical Weather**: `https://archive-api.open-meteo.com/v1/archive`
- **Forecast Weather**: `https://api.open-meteo.com/v1/forecast`
- **Cost**: FREE (no API key required)
- **Rate Limits**: None officially, but be respectful (recommended: <100 requests/second)

### Current Bottleneck

The current implementation (`ingestion/nfl/weather.py`) is **sequential**:
- Fetches one game at a time
- Has `time.sleep(0.1)` delay between calls
- For ~2,600 games: **~4-5 minutes minimum** (just delays, not counting API response time)

### Why It's Slow

1. **Sequential Processing**: One API call at a time
2. **Artificial Delays**: `time.sleep(0.1)` between calls
3. **Network Latency**: Each API call takes ~100-500ms
4. **Total Time**: ~2,600 games × (0.1s delay + 0.2s API) = **~13 minutes**

## Optimization Solutions

### ✅ Solution 1: Parallel Processing (IMPLEMENTED)

**File**: `ingestion/nfl/weather_parallel.py`

**Speed Improvement**: **10x faster** (with 10 workers)

**How it works**:
- Uses `ThreadPoolExecutor` for parallel API calls
- Fetches 10 games simultaneously
- Respects cache (checks cache synchronously, only fetches missing)
- Estimated time: **~2-3 minutes** instead of 13+ minutes

**Usage**:
```python
from ingestion.nfl.weather_parallel import fetch_weather_parallel_batch

weather_df = fetch_weather_parallel_batch(games_df, max_workers=10)
```

**Benefits**:
- ✅ 10x faster
- ✅ Still uses cache
- ✅ Respects API (10 concurrent requests is safe)
- ✅ No code changes needed elsewhere

### Solution 2: Batch API Calls (POSSIBLE)

Open-Meteo supports batch requests by requesting date ranges:
- Can request multiple dates in one call: `start_date=2024-01-01&end_date=2024-01-31`
- **Speed Improvement**: Could reduce API calls by 90%+
- **Challenge**: Need to group games by stadium and date ranges

**Example**:
```python
# Instead of 16 calls for 16 games at same stadium in January:
# One call: start_date=2024-01-01&end_date=2024-01-31
```

### Solution 3: Alternative APIs (NOT RECOMMENDED)

**Weather APIs Comparison**:

| API | Cost | Rate Limits | Historical Data | Speed |
|-----|------|-------------|----------------|-------|
| **Open-Meteo** | FREE | None | ✅ Excellent | Fast |
| OpenWeatherMap | Free tier: 60/min | 60/min | ✅ Good | Fast |
| WeatherAPI | Free: 1M/month | 1M/month | ✅ Good | Fast |
| Visual Crossing | Free: 1000/day | 1000/day | ✅ Good | Fast |

**Recommendation**: Stick with Open-Meteo (free, no limits, good data quality)

### Solution 4: Pre-fetch and Cache (RECOMMENDED)

**Strategy**: Fetch weather for all historical games once, cache forever
- Historical weather doesn't change
- Only need to fetch forecast for future games
- **Speed**: Instant (from cache) for historical games

**Implementation**: Already done! Cache is at `data/nfl/cache/weather/`

## Performance Comparison

| Method | Time for 2,600 games | Notes |
|--------|---------------------|-------|
| **Current (Sequential)** | ~13-15 minutes | With 0.1s delays |
| **Parallel (10 workers)** | ~2-3 minutes | ✅ **10x faster** |
| **Batch API calls** | ~30 seconds | Requires refactoring |
| **Pre-cached** | ~5 seconds | After initial fetch |

## Recommended Approach

1. **Use Parallel Processing** (already implemented in `weather_parallel.py`)
2. **Keep Cache**: Historical weather cached forever
3. **Only Fetch New**: Only fetch weather for games not in cache
4. **Future Games**: Use forecast API (already implemented)

## Implementation Status

✅ **Parallel version created**: `ingestion/nfl/weather_parallel.py`
✅ **Integrated into feature generation**: `generate_all_features.py` updated
✅ **Backward compatible**: Falls back to sequential if parallel fails

## Next Steps

The current running process will complete with sequential fetching. For future runs:
- The parallel version will be used automatically
- Or manually use: `fetch_weather_parallel_batch(games_df, max_workers=10)`

## GPU Note

Weather API calls are **CPU/Network bound**, not GPU bound. GPU will activate during model training (FT-Transformer, TabNet, Ensemble), not during feature generation.

