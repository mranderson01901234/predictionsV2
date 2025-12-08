# Next Gen Stats API Discovery

## ✅ Discovered Endpoints

### Base URL
`https://nextgenstats.nfl.com/api/leaders/{metric}/{category}`

### Working Endpoints (Public)

1. **Fastest Ball Carriers** ✅
   - **Endpoint:** `/speed/ballCarrier`
   - **Status:** ✅ Public (200 OK)
   - **URL Pattern:** `https://nextgenstats.nfl.com/stats/top-plays/fastest-ball-carriers`
   - **Parameters:**
     - `limit`: Number of records (default: 20)
     - `season`: Year (e.g., 2024, 2018)
     - `seasonType`: 'REG' or 'POST'
     - `week`: Week number (1-18) or 'all'

2. **Longest Tackles** ✅
   - **Endpoint:** `/distance/tackle`
   - **Status:** ✅ Public (200 OK)
   - **URL Pattern:** `https://nextgenstats.nfl.com/stats/top-plays/longest-tackles/{season}/{seasonType}/{week}`

### Endpoints Requiring AWS Signature (403)

3. **Longest Plays** ⚠️
   - **Endpoint:** `/distance/play`
   - **Status:** ⚠️ Requires AWS Signature (403)
   - **URL Pattern:** `https://nextgenstats.nfl.com/stats/top-plays/longest-plays/{season}/{seasonType}/{week}`

4. **Fastest Sacks** ⚠️
   - **Endpoint:** `/speed/sack`
   - **Status:** ⚠️ Requires AWS Signature (403)
   - **URL Pattern:** `https://nextgenstats.nfl.com/stats/top-plays/fastest-sacks/{season}/{seasonType}/{week}`

## 📊 Data Structure

### Response Format
```json
{
  "season": 2024,
  "seasonType": "REG",
  "leaders": [
    {
      "playerId": "...",
      "playerName": "...",
      "team": "...",
      "position": "...",
      "value": 22.05,  // Speed in mph or distance in yards
      "gameId": "...",
      "week": 1,
      "playId": "...",
      // ... more fields
    }
  ]
}
```

### Sample Record Fields (Fastest Ball Carriers)
- `playerId`: Player identifier
- `playerName`: Player name
- `team`: Team abbreviation
- `position`: Player position
- `value`: Speed in mph
- `gameId`: Game identifier
- `week`: Week number
- `playId`: Play identifier
- Additional metadata fields

## 🔍 Remaining Categories to Discover

The user mentioned **7 total categories**. We've found:
1. ✅ Fastest Ball Carriers (`speed/ballCarrier`)
2. ✅ Longest Tackles (`distance/tackle`)
3. ⚠️ Longest Plays (`distance/play`) - requires auth
4. ⚠️ Fastest Sacks (`speed/sack`) - requires auth

**Still need to discover 3 more categories:**
- Possibly: Fastest Throws, Deepest Throws, Longest Runs, Fastest Receivers, etc.

## 📝 Next Steps

1. ✅ **Working endpoints** - Implement scrapers for public endpoints
2. ⚠️ **AWS Signature** - Implement AWS signature for protected endpoints
3. 🔍 **Discover remaining** - Find the other 3 categories
4. ✅ **Historical data** - Test back to 2018
5. ✅ **Week filtering** - Validate week parameter works

## 🎯 Implementation Priority

### High Priority (Public)
1. Fastest Ball Carriers - ✅ Ready to implement
2. Longest Tackles - ✅ Ready to implement

### Medium Priority (Requires Auth)
3. Longest Plays - ⚠️ Need AWS signature
4. Fastest Sacks - ⚠️ Need AWS signature

### Low Priority
5-7. Remaining categories - Need discovery

