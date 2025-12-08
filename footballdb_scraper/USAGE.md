# FootballDB Scraper - Usage Guide

## Quick Commands

### Check Progress
```bash
# From project root
cd /home/dp/Documents/predictionV2
source venv/bin/activate
python footballdb_scraper/scripts/check_progress.py

# Or from footballdb_scraper directory
cd footballdb_scraper
python scripts/check_progress.py
```

### Scrape Player Splits
```bash
# Single year
python footballdb_scraper/scripts/scrape_splits.py --year 2024 --priority-only

# Historical range
python footballdb_scraper/scripts/scrape_splits.py --start-year 2015 --end-year 2024 --priority-only

# Surface splits only (highest value)
python footballdb_scraper/scripts/scrape_splits.py --surface-only --start-year 2009 --end-year 2024
```

### Scrape Coaches
```bash
python footballdb_scraper/scripts/scrape_coaches.py
```

### Export Features
```bash
python footballdb_scraper/scripts/export_features.py
```

## Using the Convenience Script

From the project root:
```bash
./footballdb_scraper/run.sh check_progress
./footballdb_scraper/run.sh scrape_splits --year 2024 --priority-only
./footballdb_scraper/run.sh scrape_coaches
```

## Common Issues

**Import errors**: Make sure you're running from the project root (`/home/dp/Documents/predictionV2`) and have activated the virtual environment:
```bash
source venv/bin/activate
```

**Path errors**: Always use full paths relative to project root:
- ✅ `python footballdb_scraper/scripts/check_progress.py`
- ❌ `python footballdb_scraper/s` (incomplete path)

## Current Rate Limiting

- **Rate**: 0.3-0.7 seconds between requests (~3x faster than before)
- **Cache**: 24-hour expiry
- **Retries**: Automatic retry on 429, 500, 502, 503, 504 errors



