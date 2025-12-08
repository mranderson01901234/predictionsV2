# Playwright Setup Guide

## Why Playwright?

Playwright is recommended over Selenium for NFL.com scraping because:

1. **Faster**: Executes JavaScript faster than Selenium
2. **More Reliable**: Better waiting mechanisms and error handling
3. **Network Interception**: Can capture API requests to discover endpoints
4. **Better JavaScript Support**: Handles modern SPAs better
5. **Auto-waiting**: Automatically waits for elements to be ready

## Installation

```bash
# Install Playwright Python package
pip install playwright

# Install browser binaries (Chromium)
playwright install chromium

# Or install all browsers
playwright install
```

## Usage

### Basic Injury Scraping

```python
from scrapers.playwright_scraper import PlaywrightInjuryScraper

# Initialize scraper
scraper = PlaywrightInjuryScraper(headless=True)  # Set False to see browser

# Scrape a single week
records = scraper.scrape_week(2024, 1)

# Scrape entire season
records = scraper.scrape_season(2024)
```

### Discover API Endpoints

One of Playwright's key advantages is discovering API endpoints:

```python
from scrapers.playwright_scraper import PlaywrightInjuryScraper

scraper = PlaywrightInjuryScraper(headless=False)

# This will intercept all API requests
endpoints = scraper.discover_api_endpoints(2024, 1)

for endpoint in endpoints:
    print(f"{endpoint['method']} {endpoint['url']}")
```

### Using the Discovery Script

```bash
# Run endpoint discovery
python scripts/discover_endpoints.py
```

This will:
1. Open browser (non-headless)
2. Navigate to injuries page
3. Intercept all API requests
4. Save discovered endpoints to `data/raw/discovered_endpoints.json`

## Comparison: Playwright vs Selenium

| Feature | Playwright | Selenium |
|---------|-----------|----------|
| Speed | ⚡ Faster | Slower |
| JavaScript | ✅ Excellent | Good |
| Network Interception | ✅ Built-in | ❌ Requires proxy |
| Auto-waiting | ✅ Automatic | Manual |
| Reliability | ✅ More stable | Less stable |
| Setup | Simple | More complex |

## Troubleshooting

### Browser Not Found

```bash
playwright install chromium
```

### Headless Mode Issues

If scraping fails in headless mode, try:
```python
scraper = PlaywrightInjuryScraper(headless=False)
```

### Timeout Errors

Increase wait times in `scrape_week_async()`:
```python
await self.page.wait_for_timeout(5000)  # Increase from 2000
```

## Best Practices

1. **Use headless=True for production** (faster, less resource usage)
2. **Use headless=False for debugging** (see what's happening)
3. **Enable network interception** to discover API endpoints
4. **Wait for networkidle** to ensure page is fully loaded
5. **Close browser properly** to avoid resource leaks

## Integration with Existing Scrapers

The Playwright scraper can be used alongside existing scrapers:

```python
# Try API first (fastest)
from scrapers.api_scraper import NFLAPIScraper
api_scraper = NFLAPIScraper()
injuries = api_scraper.get_injuries(2024, 1)

# Fallback to Playwright if API fails
if not injuries:
    from scrapers.playwright_scraper import PlaywrightInjuryScraper
    playwright_scraper = PlaywrightInjuryScraper()
    injuries = playwright_scraper.scrape_week(2024, 1)
```

## Performance

- **API Scraper**: ~0.5-1 second per request
- **Playwright**: ~3-5 seconds per page (includes JS rendering)
- **Selenium**: ~5-10 seconds per page

For bulk scraping, prefer API when available, Playwright as fallback.

