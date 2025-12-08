# FootballDB Scraper - Quick Start Guide

## Installation

```bash
cd footballdb_scraper
pip install -r requirements.txt
```

## Quick Examples

### 1. Scrape Surface Splits (Highest Value)

Surface splits (Grass vs Turf) are the highest-value features:

```bash
python scripts/scrape_splits.py --surface-only --start-year 2020 --end-year 2024
```

### 2. Scrape Priority Splits for Current Season

```bash
python scripts/scrape_splits.py --year 2024 --priority-only
```

### 3. Scrape Historical Priority Splits

```bash
python scripts/scrape_splits.py --start-year 2015 --end-year 2024 --priority-only
```

### 4. Scrape Coach Data

```bash
python scripts/scrape_coaches.py
```

### 5. Export Features for Model

After scraping, export computed features:

```bash
python scripts/export_features.py
```

## Priority Splits Explained

**High Priority Splits** (scraped with `--priority-only`):
- `surface-grass`, `surface-turf` - Surface performance (highest value)
- `trailing-by-1-to-8`, `leading-by-1-to-8`, `tied-games` - Clutch performance
- `fourth-quarter` - Late game performance
- `thursday-games` - Short rest adjustment
- `home-games`, `away-games` - Location splits
- `division-games` - Familiarity factor

## Output Files

Scraped data is saved to:
- `data/raw/footballdb/player_splits/` - Player split data
- `data/raw/footballdb/team_splits/` - Team split data
- `data/raw/footballdb/coaches/` - Coach data
- `data/cache/footballdb/` - Cached HTML responses

Processed features are saved to:
- `data/processed/footballdb/` - Computed features

## Expected Scraping Time

- **Surface splits only**: ~30 minutes for 10 years
- **Priority splits**: ~2-3 hours for 10 years
- **All splits**: ~10-12 hours for 10 years

## Rate Limiting

The scraper automatically rate-limits to 1-2 seconds between requests to be respectful to FootballDB servers. HTML responses are cached for 24 hours.

## Troubleshooting

**Import errors**: Make sure you're running scripts from the `footballdb_scraper` directory or have the parent directory in your Python path.

**No data scraped**: Check that the URLs are accessible and the HTML structure hasn't changed. FootballDB may have updated their site structure.

**Rate limiting issues**: If you see 429 errors, the scraper will automatically retry with backoff. Consider increasing the rate limit intervals in `config/scraping_config.yaml`.

