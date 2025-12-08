# NFL Injury Data Sources

## Currently Implemented Sources

### 1. **ESPN API** (Unofficial)
- **Endpoint**: `https://site.api.espn.com/apis/site/v2/sports/football/nfl/injuries`
- **Status**: ✅ Implemented
- **Coverage**: Current week injuries
- **Pros**: Free, no API key needed, real-time data
- **Cons**: Unofficial API (may break), limited historical data
- **Implementation**: `ingestion/nfl/injuries_phase2.py::_fetch_espn_injuries()`

### 2. **NFL.com Scraping**
- **URL**: `https://www.nfl.com/injuries/` and `https://www.nfl.com/teams/{team}/injuries`
- **Status**: ✅ Implemented (partial)
- **Coverage**: Current week official injury reports
- **Pros**: Official source, comprehensive
- **Cons**: Web scraping (fragile, may break), rate limiting
- **Implementation**: `ingestion/nfl/injuries_phase2.py::_scrape_nfl_injuries()`

### 3. **nflverse** (nfl_data_py)
- **Status**: ⚠️ Placeholder (not fully implemented)
- **Coverage**: Unknown (may not have injury data)
- **Pros**: Free, official NFL data community
- **Cons**: May not include injury reports
- **Implementation**: `ingestion/nfl/injuries_phase2.py::_fetch_nflverse_injuries()`

## Additional Real-Time Sources (Not Yet Implemented)

### 4. **GridIron Data API**
- **Website**: https://www.gridirondata.com/
- **Status**: ❌ Not implemented
- **Coverage**: Real-time injury reports, player stats, projections
- **Pros**: Comprehensive, updated daily, official data
- **Cons**: **Paid service** (pricing varies)
- **API**: REST API with authentication
- **Recommendation**: Consider if budget allows

### 5. **BALLDONTLIE NFL API**
- **Website**: https://nfl.balldontlie.io/
- **Status**: ❌ Not implemented
- **Coverage**: Player injuries, stats, schedules
- **Pros**: Free tier available, REST API
- **Cons**: May have rate limits on free tier
- **API**: `https://nfl.balldontlie.io/api/v1/injuries`
- **Recommendation**: ✅ **Good candidate for free implementation**

### 6. **OpticOdds NFL API**
- **Website**: https://opticodds.com/sports/nfl
- **Status**: ❌ Not implemented
- **Coverage**: Real-time odds, player props, injury reports
- **Pros**: Comprehensive sports data
- **Cons**: **Paid service**
- **Recommendation**: Consider if budget allows

### 7. **SportsDataIO**
- **Website**: https://sportsdata.io/
- **Status**: ❌ Not implemented
- **Coverage**: Official NFL injury reports, player data
- **Pros**: Official data, comprehensive
- **Cons**: **Paid service** (subscription-based)
- **Recommendation**: Consider for production use

### 8. **The Odds API** (Already Integrated)
- **Status**: ✅ Implemented (for odds, not injuries)
- **Note**: May have injury data in premium tiers
- **Current Use**: Odds/spreads/totals
- **Recommendation**: Check if injury data available in current plan

## Recommended Implementation Priority

### High Priority (Free/Open)
1. ✅ **BALLDONTLIE NFL API** - Free tier, REST API, easy integration
2. ✅ **Improve NFL.com scraping** - Official source, already partially implemented
3. ✅ **Enhance ESPN API** - Already working, improve error handling

### Medium Priority (Free/Open)
4. ⚠️ **nflverse** - Check if injury data actually available
5. ⚠️ **The Odds API** - Check premium tier for injury data

### Low Priority (Paid)
6. 💰 **GridIron Data** - If budget allows
7. 💰 **OpticOdds** - If budget allows
8. 💰 **SportsDataIO** - If budget allows

## Implementation Plan

### Phase 1: Add BALLDONTLIE API (Free)
- Add `_fetch_balldontlie_injuries()` method
- Integrate into `fetch_current_injuries()` priority list
- Test with current week data

### Phase 2: Improve NFL.com Scraping
- Complete HTML parsing implementation
- Add better error handling
- Add caching for rate limiting

### Phase 3: Multi-Source Aggregation
- Combine data from multiple sources
- Resolve conflicts (prefer official sources)
- Merge duplicate entries

## Current Status

**For Live Predictions:**
- ✅ ESPN API: Working (current week)
- ⚠️ NFL.com: Partial implementation
- ❌ Other sources: Not yet implemented

**For Historical Data:**
- ⚠️ Limited historical data available
- Mock data used for validation/testing
- Real historical APIs are rare/paid

## Next Steps

1. Implement BALLDONTLIE API integration
2. Improve NFL.com scraping robustness
3. Add multi-source aggregation
4. Consider paid APIs if free sources insufficient

