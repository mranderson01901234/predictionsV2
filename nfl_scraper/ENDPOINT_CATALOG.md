# NFL.com API Endpoint Catalog

## ✅ Working Endpoints

### 1. Teams ✅
**Endpoint:** `GET /experience/v1/teams?season=2025`

**Status:** ✅ Fully functional

**Response:**
- List of 32 teams
- Includes: `id`, `abbreviation`, `fullName`, `conferenceAbbr`, `divisionFullName`, `location`, `nickName`

**Usage:**
```python
from scrapers.api_scraper import NFLAPIScraper
scraper = NFLAPIScraper()
teams = scraper.get_teams(season=2025)  # ✅ Works!
```

**Data Quality:** ⭐⭐⭐⭐⭐ Excellent

---

### 2. Injuries ✅ **CRITICAL**
**Endpoint:** `GET /football/v2/injuries?season=2024&week=1`

**Status:** ✅ Fully functional

**Response Structure:**
```json
{
  "injuries": [
    {
      "season": 2024,
      "week": 1,
      "team": {"fullName": "Baltimore Ravens"},
      "person": {
        "id": "...",
        "displayName": "Zay Flowers",
        "gsisId": "00-0039064"
      },
      "injuries": ["Knee"],
      "injuryStatus": "OUT",
      "practiceDays": [
        {"date": "2025-01-07", "status": "DIDNOT"},
        {"date": "2025-01-08", "status": "DIDNOT"},
        {"date": "2025-01-09", "status": "DIDNOT"}
      ],
      "practiceStatus": "DIDNOT",
      "position": "WR"
    }
  ],
  "pagination": {...}
}
```

**Key Data Extracted:**
- ✅ Injury TYPE (Knee, Hamstring, etc.)
- ✅ Practice participation by day (Wed/Thu/Fri)
- ✅ Game status (Out/Doubtful/Questionable)
- ✅ Player identification
- ✅ Team information

**Usage:**
```python
from scrapers.api_injury_scraper import APIInjuryScraper
scraper = APIInjuryScraper()
records = scraper.scrape_week(2024, 1)  # ✅ Works!
```

**Data Quality:** ⭐⭐⭐⭐⭐ Perfect for prediction model

---

### 3. Game Details ✅
**Endpoint:** `GET /football/v2/experience/weekly-game-details?season=2024&type=REG&week=1`

**Status:** ✅ Functional

**Response:**
- List of 16 games per week
- Includes: `id`, `homeTeam`, `awayTeam`, `date`, `venue`, `summary`, `status`

**Key Fields:**
- Game ID (for linking to other data)
- Teams (home/away)
- Date/time
- Venue information
- Game status (SCHEDULED, FINAL, etc.)
- Summary (scores, attendance, weather)

**Usage:**
```python
from scrapers.api_scraper import NFLAPIScraper
scraper = NFLAPIScraper()
# Need to update API scraper to support football/v2 base
games = scraper.fetch_api('football/v2/experience/weekly-game-details',
                          params={'season': 2024, 'type': 'REG', 'week': 1})
```

**Data Quality:** ⭐⭐⭐⭐ Good (no player stats, but good game metadata)

---

### 4. Standings ✅
**Endpoint:** `GET /football/v2/standings?season=2024&seasonType=REG`

**Status:** ✅ Fully functional

**Response Structure:**
```json
{
  "season": 2024,
  "seasonType": "REG",
  "weeks": [
    {
      "week": 1,
      "standings": [
        {
          "team": {"fullName": "Arizona Cardinals"},
          "overall": {
            "wins": 0,
            "losses": 1,
            "points": {"for": 28, "against": 34},
            "streak": {"type": "L", "length": 1}
          },
          "home": {"wins": 0, "losses": 0, "points": {...}},
          "road": {"wins": 0, "losses": 1, "points": {...}},
          "last5": {"wins": 0, "losses": 1, "points": {...}},
          "division": {"rank": 3, "wins": 0, "losses": 0},
          "conference": {"rank": 11, "wins": 0, "losses": 0},
          "closeGames": {"wins": 0, "losses": 1}
        }
      ]
    }
  ]
}
```

**Key Data Extracted:**
- ✅ Overall record (W-L-T)
- ✅ Home/road splits
- ✅ Points for/against
- ✅ Last 5 games record
- ✅ Division/conference rank
- ✅ Win/loss streaks
- ✅ Close games record

**Usage:**
```python
from scrapers.api_scraper import NFLAPIScraper
scraper = NFLAPIScraper()
# Need to add standings method
standings = scraper.fetch_api('football/v2/standings',
                              params={'season': 2024, 'seasonType': 'REG'})
```

**Data Quality:** ⭐⭐⭐⭐⭐ Excellent for team form features

---

### 5. Week Info by Date ✅
**Endpoint:** `GET /football/v2/weeks/date/2024-09-08`

**Status:** ✅ Functional

**Response:**
```json
{
  "season": 2024,
  "seasonType": "REG",
  "week": 1,
  "byeTeams": [],
  "dateBegin": "2024-08-28",
  "dateEnd": "2024-09-11",
  "weekType": "REG"
}
```

**Usage:** Convert dates to week numbers

**Data Quality:** ⭐⭐⭐⭐ Good

---

### 6. Team Info ✅
**Endpoint:** `GET /football/v2/teams/{teamId}`

**Status:** ✅ Functional

**Response:**
- Team metadata: `bio`, `currentCoach`, `yearEstablished`, `colors`

**Data Quality:** ⭐⭐⭐ Good (metadata only)

---

## ❌ Not Found Endpoints

### Player Stats
- `/football/v2/players/{playerId}/stats` - 404
- `/football/v2/players/{playerId}/stats/career` - 404
- `/football/v2/players/{playerId}/stats/situational` - 404
- `/experience/v1/players/{playerId}/stats` - 404

**Note:** Player stats may require:
- Different player ID format
- Different endpoint structure
- HTML scraping from player pages

### Transactions
- `/football/v2/transactions` - 404
- `/football/v2/transactions/trades` - 404

**Note:** Transactions may be in CMS or require HTML scraping

### Game Stats
- `/football/v2/games/{gameId}/stats` - 404
- `/football/v2/games/{gameId}/boxscore` - 404
- `/football/v2/games/{gameId}/play-by-play` - 404

**Note:** Game stats may be in summary section or require different endpoint

---

## 📊 Data Extraction Validation

### ✅ Validated Endpoints

1. **Injuries** ✅
   - ✅ Fetched 20 records for Week 1
   - ✅ All fields parsed correctly
   - ✅ Practice days extracted
   - ✅ Saved to Parquet successfully

2. **Standings** ✅
   - ✅ Fetched 18 weeks of data
   - ✅ 32 teams per week
   - ✅ All stat categories extracted
   - ✅ Home/road splits working

3. **Game Details** ✅
   - ✅ Fetched 16 games per week
   - ✅ Game IDs extracted
   - ✅ Team info extracted
   - ✅ Dates extracted

4. **Teams** ✅
   - ✅ Fetched 32 teams
   - ✅ All team info extracted

---

## 🎯 Priority Endpoints for Prediction Model

### High Priority (Working) ✅
1. **Injuries** - Critical for Level 5 features
2. **Standings** - Team form, streaks, home/road splits
3. **Game Details** - Game metadata, teams, dates

### Medium Priority (Need Discovery)
1. **Player Stats** - Situational stats, career baselines
2. **Game Stats** - Box scores, play-by-play
3. **Transactions** - Roster changes

### Low Priority
1. **Team Info** - Metadata only
2. **Week Info** - Date conversion utility

---

## 📝 Next Steps

1. ✅ **Injuries** - Fully implemented
2. ⚠️ **Standings** - Add to API scraper
3. ⚠️ **Game Details** - Update API scraper for football/v2 base
4. 🔍 **Player Stats** - Continue discovery (may need HTML scraping)
5. 🔍 **Transactions** - Continue discovery (may need HTML scraping)

