# Why Mock Data is Generated

## Issue

When running live game predictions, the system generates mock injury data instead of fetching real data.

## Root Cause

1. **Default Behavior**: `generate_all_features()` defaults to `use_mock_injuries=True`
2. **Historical Data Limitation**: Real injury APIs (ESPN, NFL.com) don't provide reliable historical data
3. **Fallback Logic**: When real data isn't available, the system falls back to mock data generation

## Why Mock Data Exists

Mock data was created for **validation and testing purposes** when:
- Historical injury data isn't available from APIs
- We need to test the feature pipeline
- We want to ensure feature generation works even without real data

## For Live Predictions

For **real-world predictions** (like tonight's game), we should:

1. ✅ **Fetch current injury data** from ESPN API or NFL.com scraping
2. ✅ **Disable mock data** (`use_mock_injuries=False`)
3. ✅ **Use real weather data** (Open-Meteo API)
4. ✅ **Use real odds data** (The Odds API)

## Solution

The prediction script has been updated to:
- Set `use_mock_injuries=False` when generating features for live games
- Try to fetch current injuries first (ESPN API)
- Fall back to historical injuries only if current data unavailable
- Never use mock data for live game predictions

## Current Status

✅ **Fixed**: Prediction script now uses `use_mock_injuries=False` for live predictions
✅ **Fixed**: Attempts to fetch current injuries before falling back
⚠️ **Note**: If real injury APIs fail, injury features will be skipped (not mocked)

