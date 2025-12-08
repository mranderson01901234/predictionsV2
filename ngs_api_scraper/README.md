# Next Gen Stats API Scraper

A comprehensive scraper for NFL Next Gen Stats public JSON APIs.

## Overview

This scraper provides direct access to Next Gen Stats data via their public JSON APIs. This is significantly better than scraping HTML or using nflverse (which may lag behind).

**Key Advantages:**
- Real-time data (nflverse updates nightly)
- More granular data (game-level, play-level)
- Additional metrics not in nflverse
- Direct access to pass/route/carry charts
- Weekly granularity (nflverse only has season totals)

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Direct API Access

```python
from scrapers.base_client import NGSClient

client = NGSClient()

# Get current passing stats
passing = client.get_passing_stats(2024, 'REG')
for p in passing[:5]:
    print(f"{p['playerName']}: CPOE={p['completionPercentageAboveExpectation']:.2f}")

# Get specific week
week_10 = client.get_passing_stats(2024, 'REG', week=10)
```

### Full Season Scrape

```python
from scrapers.statboard_scraper import StatboardScraper

scraper = StatboardScraper()
data = scraper.scrape_season(2024, 'REG', include_weekly=True)
```

## Command Line Usage

### Full Historical Scrape (2018-2024)

```bash
python scripts/scrape_all.py --historical
```

### Current Season Only

```bash
python scripts/scrape_all.py --season 2024
```

### Season Totals Only (No Weekly Data)

```bash
python scripts/scrape_all.py --season 2024 --no-weekly
```

### Export for Model Integration

```bash
python scripts/export_for_model.py
```

## Project Structure

```
ngs_api_scraper/
├── config/
│   ├── endpoints.yaml           # All endpoint definitions
│   ├── scraping_config.yaml     # Rate limits, headers
│   └── teams.yaml               # Team ID mappings
├── scrapers/
│   ├── base_client.py           # Base API client
│   ├── statboard_scraper.py     # Passing/rushing/receiving stats
│   ├── leaders_scraper.py       # Top plays (speed, distance, etc.)
│   ├── gamecenter_scraper.py    # Per-game detailed stats
│   ├── highlights_scraper.py    # Highlight plays
│   └── charts_scraper.py        # Pass/route/carry charts
├── storage/
│   └── parquet_store.py         # Parquet file storage
├── scripts/
│   ├── scrape_all.py            # Scrape all statboards
│   └── export_for_model.py      # Export to model format
└── data/
    ├── raw/                     # Raw JSON responses
    ├── processed/               # Processed parquet files
    └── cache/                   # Response cache
```

## Available Endpoints

### Statboards (Player Aggregate Stats)
- **Passing**: CPOE, time to throw, aggressiveness, air yards
- **Rushing**: RYOE, efficiency, time to LOS
- **Receiving**: Separation, cushion, YAC above expected

### Leaders (Top Plays)
- Fastest ball carriers
- Longest ball carrier runs
- Longest tackles
- Fastest sacks
- Improbable completions
- YAC above expected
- Remarkable rushes

### Game Center
- Per-game detailed stats for all players
- Passers, rushers, receivers, pass rushers

### Highlights
- Notable plays from games

### Charts
- Pass/route/carry visualization data

## Data Availability

- **Start Season**: 2018
- **Current Season**: 2025
- **Season Types**: REG (Regular), POST (Postseason)
- **Max Weeks**: 18 (Regular), 4 (Postseason)

## Expected Data Volume

| Data Type | Records/Season | Total (2018-2024) |
|-----------|----------------|-------------------|
| Passing (season) | ~50 QBs | ~350 |
| Passing (weekly) | ~50 QBs × 18 weeks | ~6,300 |
| Rushing (season) | ~150 RBs | ~1,050 |
| Rushing (weekly) | ~150 × 18 | ~18,900 |
| Receiving (season) | ~300 WR/TEs | ~2,100 |
| Receiving (weekly) | ~300 × 18 | ~37,800 |
| **Total** | | **~66,500 records** |

Storage: ~50-100 MB in Parquet format.

## Rate Limiting

The scraper respects rate limits:
- Default: 2 requests per second
- Automatic retries with exponential backoff
- Response caching (1 hour for current season, 30 days for historical)

## Comparison with nflverse

| Feature | nflverse | Direct NGS API |
|---------|----------|----------------|
| **Update frequency** | Nightly | Real-time |
| **Data freshness** | 12-24 hours lag | Immediate |
| **Weekly granularity** | Season totals only | Week-by-week |
| **Game-level data** | Limited | Full via Game Center |
| **Historical data** | 2016+ | 2018+ |
| **Rate limiting** | N/A (pre-scraped) | 2 req/sec works |
| **Dependency** | nfl-data-py package | Just requests |

**Recommendation:** 
- Use nflverse for historical data (it's pre-cleaned)
- Use direct NGS API for current season (fresher data)
- Use NGS API for weekly data (more granular)

## License

MIT

