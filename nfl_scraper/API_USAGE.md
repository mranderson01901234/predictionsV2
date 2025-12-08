# NFL.com API Usage Guide

## Overview

The NFL.com API is available at `https://api.nfl.com/experience/v1/` and provides structured JSON data. This is more reliable than HTML scraping.

## Authentication

The API requires a Bearer token in the Authorization header:

```
Authorization: Bearer <token>
```

### Getting a Fresh Token

1. Open https://www.nfl.com in your browser
2. Open DevTools (F12) → Network tab
3. Filter by "api.nfl.com"
4. Navigate to any page that makes API calls (e.g., injuries page)
5. Find a request to `api.nfl.com`
6. Copy the `Authorization` header value
7. Add it to `config/credentials.yaml`:

```yaml
nfl_api:
  auth_token: "YOUR_BEARER_TOKEN_HERE"
```

**Note:** Tokens expire periodically. You'll need to refresh them.

## Working Endpoints

### Teams
```
GET https://api.nfl.com/experience/v1/teams?season=2025
```

Returns list of all NFL teams with:
- `id`: Team ID
- `name`: Full team name
- `abbreviation`: Team abbreviation (e.g., "KC", "BAL")
- `conference`: Conference name
- `division`: Division name

### Injuries

**Status:** Endpoint still being discovered. Try:
- `/injuries?season=2024&week=1`
- `/games/{gameId}/injuries`
- `/injury-reports?season=2024&week=1`

## Response Format

Responses are JSON and may be Brotli-compressed. The scraper handles decompression automatically.

## Usage Example

```python
from scrapers.api_scraper import NFLAPIScraper

# Initialize with your token
scraper = NFLAPIScraper(auth_token="YOUR_TOKEN")

# Get teams
teams = scraper.get_teams(season=2025)
print(f"Found {len(teams)} teams")

# Get injuries (when endpoint is confirmed)
injuries = scraper.get_injuries(season=2024, week=1)
```

## Rate Limiting

The API scraper respects rate limits:
- 0.5 requests per second (configurable)
- Automatic retries with exponential backoff
- Response caching (24 hours by default)

## Troubleshooting

### 401 Unauthorized
- Token has expired
- Extract a fresh token from browser DevTools

### Empty Responses
- Check endpoint URL is correct
- Verify token is valid
- Check response headers for error messages

### Brotli Decompression Errors
- Install: `pip install brotli`
- The scraper will fall back to requests' automatic decompression

