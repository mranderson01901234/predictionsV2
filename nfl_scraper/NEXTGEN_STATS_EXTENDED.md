# Next Gen Stats - Extended Endpoint Discovery

## ✅ All Discovered Endpoints

### Category 1: Top Plays Leaders (7 endpoints) ✅
**Base URL:** `https://nextgenstats.nfl.com/api/leaders`

1. **Fastest Ball Carriers** - `speed/ballCarrier`
2. **Longest Ball Carrier Runs** - `distance/ballCarrier`
3. **Longest Tackles** - `distance/tackle`
4. **Fastest Sacks** - `time/sack`
5. **Improbable Completions** - `expectation/completion/season`
6. **YAC (Yards After Catch)** - `expectation/yac/season`
7. **Remarkable Rushes** - `expectation/ery/season`

**Status:** ✅ All implemented in `nextgen_stats_scraper.py`

---

### Category 2: Player Statistics Boards (3 endpoints) ✅ NEW

**Base URL:** `https://nextgenstats.nfl.com/api/statboard`

#### 1. Passing Stats ✅
- **Endpoint:** `GET /api/statboard/passing`
- **URL:** `https://nextgenstats.nfl.com/stats/passing#yards`
- **Parameters:**
  - `season` (int): NFL season year (2018+)
  - `seasonType` (str): 'REG' or 'POST'
  - `week` (int, optional): Week number (1-18)
- **Response Structure:**
  ```json
  {
    "season": 2024,
    "seasonType": "REG",
    "filter": {...},
    "threshold": {...},
    "stats": [
      {
        "player": {...},
        "playerName": "string",
        "position": "string",
        "teamId": "string",
        "aggressiveness": float,
        "attempts": int,
        "avgAirDistance": float,
        "avgAirYardsDifferential": float,
        "avgAirYardsToSticks": float,
        "avgCompletedAirYards": float,
        "avgIntendedAirYards": float,
        "avgTimeToThrow": float,
        "completionPercentage": float,
        "completionPercentageAboveExpectation": float,
        "completions": int,
        "expectedCompletionPercentage": float,
        "gamesPlayed": int,
        "interceptions": int,
        "maxAirDistance": float,
        ...
      }
    ]
  }
  ```
- **Historical:** ✅ Works back to 2018
- **All Teams:** ✅ Yes

#### 2. Rushing Stats ✅
- **Endpoint:** `GET /api/statboard/rushing`
- **URL:** `https://nextgenstats.nfl.com/stats/rushing#yards`
- **Parameters:** Same as passing
- **Response Structure:**
  ```json
  {
    "season": 2024,
    "seasonType": "REG",
    "filter": {...},
    "threshold": {...},
    "stats": [
      {
        "player": {...},
        "playerName": "string",
        "position": "string",
        "teamId": "string",
        "avgTimeToLos": float,
        "expectedRushYards": float,
        "rushAttempts": int,
        "rushPctOverExpected": float,
        "rushTouchdowns": int,
        "rushYards": int,
        "rushYardsOverExpected": float,
        "rushYardsOverExpectedPerAtt": float,
        "efficiency": float,
        "percentAttemptsGteEightDefenders": float,
        "avgRushYards": float,
        ...
      }
    ]
  }
  ```
- **Historical:** ✅ Works back to 2018
- **All Teams:** ✅ Yes

#### 3. Receiving Stats ✅
- **Endpoint:** `GET /api/statboard/receiving`
- **URL:** `https://nextgenstats.nfl.com/stats/receiving#yards`
- **Parameters:** Same as passing
- **Response Structure:**
  ```json
  {
    "season": 2024,
    "seasonType": "REG",
    "threshold": {...},
    "stats": [
      {
        "player": {...},
        "playerName": "string",
        "position": "string",
        "teamId": "string",
        "avgCushion": float,
        "avgExpectedYAC": float,
        "avgIntendedAirYards": float,
        "avgSeparation": float,
        "avgYAC": float,
        "avgYACAboveExpectation": float,
        "catchPercentage": float,
        "percentShareOfIntendedAirYards": float,
        "recTouchdowns": int,
        "receptions": int,
        "targets": int,
        "yards": int,
        ...
      }
    ]
  }
  ```
- **Historical:** ✅ Works back to 2018
- **All Teams:** ✅ Yes

---

### Category 3: Game Center ✅ NEW

- **Endpoint:** `GET /api/gamecenter/overview`
- **URL:** `https://nextgenstats.nfl.com/stats/game-center-index`
- **Parameters:**
  - `gameId` (str): Game ID in format `YYYYMMDDHH` (e.g., `2025120711`)
- **Response Structure:**
  ```json
  {
    "schedule": {...},
    "passers": [...],
    "rushers": [...],
    "passRushers": [...],
    "receivers": [...],
    "leaders": [...]
  }
  ```
- **Historical:** ✅ Works back to 2018
- **All Teams:** ✅ Yes (via game selection)

---

### Category 4: Highlights Play List ✅ NEW

- **Endpoint:** `GET /api/plays/highlights`
- **URL:** `https://nextgenstats.nfl.com/highlights/play-list`
- **Parameters:**
  - `limit` (int): Number of records (default: 16)
  - `season` (int): NFL season year (2018+)
  - `seasonType` (str, optional): 'REG' or 'POST'
  - `week` (int, optional): Week number (1-18) or 'all'
- **Response Structure:**
  ```json
  {
    "season": 2024,
    "highlights": [
      {
        "gameId": "string",
        "playId": "string",
        "play": {...},
        "players": [...],
        "season": int,
        "seasonType": "string",
        "teamAbbr": "string",
        "teamId": "string",
        "week": int
      }
    ],
    "total": int,
    "count": int,
    "offset": int,
    "limit": int
  }
  ```
- **Historical:** ✅ Works back to 2018
- **All Teams:** ✅ Yes

---

### Category 5: Charts List ✅ NEW

- **Endpoint:** `GET /api/content/microsite/chart`
- **URL:** `https://nextgenstats.nfl.com/charts/list/all`
- **Parameters:**
  - `count` (int): Number of charts (default: 12)
  - `week` (str): Week number (1-18) or 'all'
  - `type` (str): Chart type or 'all'
  - `teamId` (str): Team ID or 'all'
  - `esbId` (str): Player ID or 'all'
  - `season` (int): NFL season year (2018+)
  - `seasonType` (str, optional): 'REG' or 'POST'
- **Response Structure:**
  ```json
  {
    "charts": [
      {
        "imageName": "string",
        "carries": int,
        "esbId": "string",
        "firstName": "string",
        "gameId": "string",
        "headshot": "string",
        "lastName": "string",
        "playerName": "string",
        "position": "string",
        "rushingYards": int,
        "season": int,
        "seasonType": "string",
        "teamId": "string",
        "timestamp": "string",
        "touchdowns": int,
        ...
      }
    ],
    "count": int,
    "offset": int,
    "total": int,
    "season": int,
    "teamId": "string",
    "type": "string",
    "week": "string",
    "seasonType": "string"
  }
  ```
- **Historical:** ✅ Works back to 2018
- **All Teams:** ✅ Yes

---

## Summary

### Total Endpoints Discovered: **13**
- ✅ **7 Top Plays Leaders** (implemented)
- ✅ **3 Player Statistics Boards** (NEW - passing, rushing, receiving)
- ✅ **1 Game Center** (NEW)
- ✅ **1 Highlights** (NEW)
- ✅ **1 Charts** (NEW)

### Common Features:
- ✅ All endpoints work back to 2018
- ✅ All support filtering by season, seasonType (REG/POST), and week
- ✅ All return comprehensive player/play data
- ✅ All are public APIs (no AWS signature required)
- ✅ All require browser-like headers (`Accept`, `Origin`, `Referer`, `User-Agent`)

### Next Steps:
1. Create scraper classes for new endpoint categories
2. Add data schemas for new data types
3. Implement storage handlers for new data structures
4. Create execution scripts for batch scraping

