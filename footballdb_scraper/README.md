# FootballDB Comprehensive Scraper

A comprehensive scraper for FootballDB situational statistics, providing granular splits not readily available from other sources.

## Overview

FootballDB provides **granular situational splits** that are highly valuable for NFL prediction models:

- **Surface splits** (Grass vs Turf) - Some QBs perform 20%+ better on grass vs turf
- **Score differential splits** (17+, 9-16, 1-8, tied, trailing) - Teams perform very differently when leading vs trailing
- **Day of week splits** (Sunday/Monday/Thursday) - Thursday night games have different patterns
- **Monthly performance** - Seasonal trends
- **Down & distance aggregates** - Situational performance
- **Coach career records** - Coach experience correlates with late-game decisions

## Project Structure

```
footballdb_scraper/
├── config/
│   ├── splits.yaml              # All split type definitions
│   ├── scraping_config.yaml     # Rate limits, headers
│   └── teams.yaml               # Team slug mappings
├── scrapers/
│   ├── base_scraper.py          # Base HTML scraper
│   ├── player_splits_scraper.py # Player situational stats
│   └── coach_scraper.py         # Coach records
├── parsers/
│   └── stats_table_parser.py    # Parse HTML stat tables
├── scripts/
│   ├── scrape_splits.py         # Scrape all splits
│   ├── scrape_coaches.py        # Scrape coach data
│   └── export_features.py       # Export for model
├── data/
│   ├── raw/
│   ├── processed/
│   └── cache/
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Scrape Player Splits

**Single year, priority splits only:**
```bash
python scripts/scrape_splits.py --year 2024 --priority-only
```

**Historical data (2015-2024), priority splits:**
```bash
python scripts/scrape_splits.py --start-year 2015 --end-year 2024 --priority-only
```

**Surface splits only (highest value):**
```bash
python scripts/scrape_splits.py --surface-only --start-year 2015 --end-year 2024
```

**All splits for a year:**
```bash
python scripts/scrape_splits.py --year 2024 --all-splits
```

### Scrape Coach Data

```bash
python scripts/scrape_coaches.py
```

### Export Features

```bash
python scripts/export_features.py
```

## Priority Splits

The scraper prioritizes high-value splits:

**High Priority:**
- `surface-grass`, `surface-turf` - Surface performance differential
- `leading-by-1-to-8`, `trailing-by-1-to-8`, `tied-games` - Clutch performance
- `fourth-quarter` - Late game performance
- `thursday-games` - Short rest adjustment

**Medium Priority:**
- `home-games`, `away-games` - Location splits
- `third-down` - Situational performance
- `second-half` - Half splits
- `division-games` - Familiarity factor

## Expected Impact

| Feature | Impact |
|---------|--------|
| Surface Differential | +0.3-0.5% accuracy |
| Clutch Performance | +0.2-0.4% accuracy |
| 4th Quarter Rating | +0.2-0.3% accuracy |
| Thursday Adjustment | +0.1-0.2% accuracy |
| Coach Win Rate | +0.1-0.2% accuracy |
| Division Familiarity | +0.1-0.2% accuracy |

**Total Expected Impact:** +1-1.5% accuracy

## Data Volume

| Data Type | Records/Year | Years | Total |
|-----------|--------------|-------|-------|
| Player Splits (priority) | ~12,000 | 10 | ~120,000 |
| Player Splits (all) | ~50,000 | 10 | ~500,000 |
| Team Splits | ~5,000 | 10 | ~50,000 |
| Coaches | ~32 | 1 | ~32 |

## Rate Limiting

The scraper includes built-in rate limiting (1-2 seconds between requests) to be respectful to FootballDB servers. HTML responses are cached for 24 hours to avoid redundant requests.

## Features Generated

The feature extraction script creates:

1. **Surface Features**: Grass vs Turf performance differentials
2. **Clutch Features**: Performance when trailing by 1-8 vs overall
3. **4th Quarter Features**: Q4 performance vs overall
4. **Thursday Features**: TNF performance vs Sunday performance
5. **Coach Features**: Win rate, experience tiers, performance tiers

## Notes

- FootballDB is HTML-based (not JSON API) - requires BeautifulSoup parsing
- Rate limit to 1-2 requests/second (be respectful)
- Cache responses to avoid redundant requests
- Priority splits give 80% of value with 20% of scraping

## Execution Priority

1. **Week 1:** Scrape surface splits (Grass vs Turf) — highest value
2. **Week 2:** Scrape score differential splits — clutch performance
3. **Week 3:** Add Q4 and Thursday splits
4. **Week 4:** Add coach data and integrate into model

