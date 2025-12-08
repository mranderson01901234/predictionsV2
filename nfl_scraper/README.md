# NFL.com Data Scraper

Systematic scraper for NFL.com to extract granular data for NFL prediction model.

## Data Types Scraped

1. **Injury Reports** (HIGH VALUE)
   - Injury TYPE (Hamstring, Knee, Ankle, etc.)
   - Practice participation (DNP/Limited/Full)
   - Game status (Out/Doubtful/Questionable)

2. **Player Situational Stats** (HIGH VALUE)
   - Performance by quarter
   - Performance by point differential
   - Home vs away splits
   - Red zone performance

3. **Player Career Stats**
   - Season-by-season totals
   - Career averages (for baseline deviation)

4. **Transactions**
   - Trades
   - IR/Reserve list moves
   - Mid-season signings

## Installation

```bash
cd nfl_scraper
pip install -r requirements.txt
```

## Usage

### Scrape Everything

```bash
# Scrape everything for 2015-2024
python scripts/scrape_all.py --all --start-season 2015 --end-season 2024

# Scrape only injuries
python scripts/scrape_all.py --injuries --start-season 2020 --end-season 2024

# Scrape only player stats
python scripts/scrape_all.py --player-stats --start-season 2020 --end-season 2024

# Scrape only transactions
python scripts/scrape_all.py --transactions --start-season 2020 --end-season 2024
```

### Individual Scrapers

```bash
# Injuries only
python scripts/scrape_injuries.py --start-season 2020 --end-season 2024

# Player stats only
python scripts/scrape_player_stats.py --start-season 2020 --end-season 2024

# Backfill historical data
python scripts/backfill_historical.py --all --start-season 2015 --end-season 2024
```

## Output Files

All data is saved to `data/final/` directory:

- `injuries.parquet` - Injury records
- `player_career.parquet` - Career stats
- `player_situational.parquet` - Situational stats
- `player_game_logs.parquet` - Game-by-game stats
- `transactions.parquet` - Transactions

## Rate Limiting

The scraper respects NFL.com by:

- 1 request per 2 seconds (configurable in `config/scraping_config.yaml`)
- Caching responses for 24 hours
- Random delays between requests
- Rotating user agents
- Automatic retries with exponential backoff

## Configuration

Edit `config/scraping_config.yaml` to adjust:

- Rate limiting (requests per second)
- Cache expiry time
- Retry settings
- User agents

## Note on Dynamic Content

Some NFL.com pages load data via JavaScript. If the basic scraper fails:

1. Check if there's an underlying API endpoint
2. Use the `SeleniumInjuryScraper` variant for JavaScript-rendered content
3. Update parsers if HTML structure changed

## Expected Output

After running the full scraper:

| Dataset | Records (Est.) | Size (Est.) |
|---------|----------------|-------------|
| Injuries | ~50,000+ | 5-10 MB |
| Player Situational | ~5,000+ | 2-5 MB |
| Player Career | ~500+ | <1 MB |
| Transactions | ~20,000+ | 2-5 MB |

This data feeds directly into:

- Level 5 injury features (type, severity, practice progression)
- QB situational features (performance by quarter, score differential)
- Career baseline features (deviation from career average)
- Roster change flags (new to team, mid-season moves)

## Project Structure

```
nfl_scraper/
├── config/
│   ├── scraping_config.yaml      # Rate limits, retries, user agents
│   ├── urls.yaml                 # URL patterns for each data type
│   └── seasons.yaml              # Which seasons to scrape
├── scrapers/
│   ├── base_scraper.py           # Base class with rate limiting, retries
│   ├── injury_scraper.py         # Injury reports by week
│   ├── player_stats_scraper.py   # Player career/situational stats
│   └── transaction_scraper.py    # Trades, signings, IR moves
├── parsers/
│   ├── injury_parser.py          # Parse injury HTML tables
│   └── stats_parser.py           # Parse stats HTML tables
├── storage/
│   ├── database.py               # SQLite or Parquet storage
│   └── schemas.py                # Data schemas for each table
├── scripts/
│   ├── scrape_injuries.py        # Run injury scraper
│   ├── scrape_player_stats.py    # Run player stats scraper
│   ├── scrape_all.py             # Run complete scrape
│   └── backfill_historical.py    # Backfill 2015-2024 data
├── data/
│   ├── raw/                      # Raw HTML responses (cached)
│   ├── parsed/                   # Parsed data (JSON/CSV)
│   └── final/                    # Final cleaned datasets (Parquet)
└── requirements.txt
```

## Troubleshooting

### Cache Issues

If you need to refresh cached data, delete the cache directory:

```bash
rm -rf data/raw/cache/*
```

### Selenium Setup

If you need to use Selenium for dynamic content:

1. Install Chrome/Chromium
2. The scraper will automatically download ChromeDriver via webdriver-manager
3. Use `SeleniumInjuryScraper` instead of `InjuryScraper`

### HTML Structure Changes

If NFL.com changes their HTML structure:

1. Check the actual HTML in `data/raw/cache/`
2. Update the parsers in `parsers/` directory
3. Test with a single page first before full scrape

## License

Part of the NFL Prediction Model project.

