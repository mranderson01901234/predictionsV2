# NFL API Endpoint Test Results

## Game Details Endpoint ✅ WORKING

**Endpoint:** `GET /football/v2/experience/weekly-game-details`

**Status:** ✅ Successfully tested

**Parameters:**
- `season`: 2024
- `type`: REG
- `week`: 1
- `includeDriveChart`: false
- `includeReplays`: true
- `includeStandings`: true
- `includeTaggedVideos`: false

**Response:**
- Returns list of 16 games (week 1)
- Contains game metadata, teams, scores, venue info
- **Does NOT contain injury data directly**

**Sample Game Structure:**
```json
{
  "id": "game-id",
  "homeTeam": {...},
  "awayTeam": {...},
  "date": "2024-09-06",
  "week": 1,
  "season": 2024,
  "summary": {...},
  "extensions": []
}
```

## Injury Endpoints Testing

Testing various injury endpoint patterns to find the correct one.

### Next Steps

1. **Check game-specific endpoints** - Injuries may be per-game
2. **Try CMS endpoints** - May be in content management system
3. **Use Playwright** - Navigate to actual injuries page and intercept requests
4. **Check game extensions** - May be nested in game data

## Recommendations

Since the game details endpoint doesn't contain injuries, we should:

1. **Use Playwright** to navigate to `/injuries` page and capture the actual API call
2. **Check game-by-game** - Injuries may be at `/games/{gameId}/injuries`
3. **Use HTML scraping** as fallback - Parse the injuries page HTML after JavaScript renders

