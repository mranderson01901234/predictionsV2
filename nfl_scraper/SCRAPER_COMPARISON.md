# Scraper Method Comparison

## Overview

We now have three methods for scraping NFL.com data:

1. **API Scraper** (Fastest, but endpoints may be limited)
2. **Playwright Scraper** (Recommended for JavaScript pages)
3. **Selenium Scraper** (Fallback option)

## Comparison Table

| Feature | API | Playwright | Selenium |
|---------|-----|-----------|----------|
| **Speed** | ⚡⚡⚡ Fastest (~0.5s) | ⚡⚡ Fast (~3-5s) | ⚡ Slow (~5-10s) |
| **Reliability** | ✅ High (if endpoints work) | ✅✅ Very High | ⚠️ Moderate |
| **JavaScript Support** | N/A (direct API) | ✅✅ Excellent | ✅ Good |
| **Network Interception** | N/A | ✅✅ Built-in | ❌ Requires proxy |
| **Setup Complexity** | ✅ Simple | ✅ Simple | ⚠️ Moderate |
| **Resource Usage** | ✅ Low | ⚠️ Medium | ⚠️ High |
| **Endpoint Discovery** | ❌ Manual | ✅✅ Automatic | ❌ Manual |
| **Best For** | Known endpoints | JavaScript pages | Legacy fallback |

## When to Use Each

### API Scraper (`APIInjuryScraper`)

**Use when:**
- ✅ API endpoints are known and working
- ✅ Need fastest performance
- ✅ Bulk scraping many weeks/seasons
- ✅ Running in production/automated environments

**Example:**
```python
from scrapers.api_injury_scraper import APIInjuryScraper

scraper = APIInjuryScraper()
records = scraper.scrape_season(2024)
```

### Playwright Scraper (`PlaywrightInjuryScraper`) ⭐ RECOMMENDED

**Use when:**
- ✅ JavaScript-rendered pages (most NFL.com pages)
- ✅ Need to discover API endpoints
- ✅ Want reliable scraping
- ✅ Need better error handling

**Example:**
```python
from scrapers.playwright_scraper import PlaywrightInjuryScraper

scraper = PlaywrightInjuryScraper(headless=True)
records = scraper.scrape_season(2024)

# Or discover endpoints
endpoints = scraper.discover_api_endpoints(2024, 1)
```

### Selenium Scraper (`SeleniumInjuryScraper`)

**Use when:**
- ⚠️ Playwright not available
- ⚠️ Legacy compatibility needed
- ⚠️ Specific Selenium features required

**Example:**
```python
from scrapers.injury_scraper import SeleniumInjuryScraper

scraper = SeleniumInjuryScraper()
records = scraper.scrape_week_selenium(2024, 1)
```

## Auto Mode (Recommended)

The `scrape_injuries.py` script supports auto mode, which tries API first, then falls back to Playwright:

```bash
# Auto mode (tries API → Playwright)
python scripts/scrape_injuries.py --method auto --start-season 2024 --end-season 2024

# Force Playwright
python scripts/scrape_injuries.py --method playwright --start-season 2024

# Force API
python scripts/scrape_injuries.py --method api --start-season 2024
```

## Performance Benchmarks

### Single Week Scrape
- **API**: ~0.5 seconds
- **Playwright**: ~3-5 seconds
- **Selenium**: ~5-10 seconds

### Full Season (18 weeks)
- **API**: ~9 seconds (if endpoints work)
- **Playwright**: ~54-90 seconds
- **Selenium**: ~90-180 seconds

### Historical Backfill (10 seasons, 180 weeks)
- **API**: ~90 seconds (if endpoints work)
- **Playwright**: ~15-30 minutes
- **Selenium**: ~30-60 minutes

## Recommendation

**Use Playwright as the primary method** because:

1. ✅ Most reliable for JavaScript pages
2. ✅ Can discover API endpoints automatically
3. ✅ Good balance of speed and reliability
4. ✅ Better error handling
5. ✅ Easier to debug (can run non-headless)

**Workflow:**
1. Use Playwright to discover API endpoints
2. Once endpoints are known, switch to API scraper for speed
3. Use Playwright as fallback if API fails

## Installation

```bash
# For API scraper (already installed)
pip install requests brotli

# For Playwright (recommended)
pip install playwright
playwright install chromium

# For Selenium (fallback)
pip install selenium webdriver-manager
```

## Example: Discovering Endpoints with Playwright

```python
from scrapers.playwright_scraper import PlaywrightInjuryScraper

scraper = PlaywrightInjuryScraper(headless=False)  # Show browser

# This intercepts all API requests
endpoints = scraper.discover_api_endpoints(2024, 1)

# Save discovered endpoints
import json
with open('discovered_endpoints.json', 'w') as f:
    json.dump(endpoints, f, indent=2)

# Then use discovered endpoints in API scraper
```

## Troubleshooting

### API Scraper Returns Empty Data
→ Use Playwright to discover correct endpoints

### Playwright Timeout Errors
→ Increase wait times or run non-headless to debug

### Selenium Slow Performance
→ Switch to Playwright (2-3x faster)

