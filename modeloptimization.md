# NFL Prediction Model Optimization — Cursor Prompts

**Usage**: Copy each phase's prompt into Cursor AI chat (Cmd/Ctrl + L) with your project open.

---

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1: ENSEMBLE TRAINING + CALIBRATION + REST FEATURES (Week 1-2)
# ═══════════════════════════════════════════════════════════════════════════════

## Cursor Prompt — Phase 1

```
You are helping me optimize an NFL prediction model. This is Phase 1 of a 5-phase optimization plan.

## CONTEXT

Current state from audit:
- GBM baseline achieves 59.3% accuracy / 13.29% ROI on 182 games
- Stacking Ensemble is IMPLEMENTED but NEVER TRAINED
- Calibration is broken: 50-60% confidence = 47.6% accuracy (worse than coin flip)
- Rest/bye week features are NOT implemented (but schedule data is available)

Repository structure:
```
models/
├── architectures/
│   ├── ft_transformer.py      # Implemented
│   ├── tabnet.py              # Implemented
│   └── stacking_ensemble.py   # Implemented but not trained
├── training/
│   ├── trainer.py             # Baseline trainer (LR, GBM)
│   └── train_advanced.py      # Advanced trainer (FT-T, TabNet, Ensemble)
├── artifacts/
│   └── nfl_baseline/
│       └── gbm.pkl            # Only trained model
features/
├── nfl/
│   ├── generate_features.py   # Main feature pipeline
│   └── team_features.py       # Team performance features
├── core/                      # EMPTY - missing generic features
```

## PHASE 1 TASKS

### Task 1.1: Train the Stacking Ensemble

Analyze the existing code in `models/training/train_advanced.py` and `models/architectures/stacking_ensemble.py`.

Then:
1. Identify why the ensemble hasn't been trained (missing dependencies? config issues?)
2. Create or fix the training script to train all base models:
   - Logistic Regression
   - XGBoost/GBM
   - FT-Transformer
   - TabNet (optional, can skip if causing issues)
3. Train the stacking ensemble with logistic regression meta-learner
4. Save all artifacts to `models/artifacts/nfl_ensemble/`
5. Create an evaluation script that compares ensemble vs individual models

Deliverables:
- Working `train_ensemble.py` script (or fixed `train_advanced.py`)
- Trained ensemble artifacts
- Performance comparison table (accuracy, Brier score, ROI)

### Task 1.2: Fix Probability Calibration

Current calibration results:
```
50-60% confidence → 47.6% accuracy ❌
60-70% confidence → 47.8% accuracy ❌
70-80% confidence → 60.0% accuracy ✓
80%+   confidence → 70.0% accuracy ✓
```

The model's probability estimates are not calibrated — lower confidence bins perform WORSE than chance.

Create or update `models/calibration/calibrator.py`:

1. Implement multiple calibration methods:
   - Platt scaling (logistic regression on probabilities)
   - Isotonic regression (non-parametric, recommended)
   - Temperature scaling (single parameter)

2. Create calibration validation:
   - Split data: Train calibrator on validation set, evaluate on test set
   - Generate calibration curve (reliability diagram)
   - Ensure MONOTONIC accuracy across confidence bins

3. Implement calibration wrapper:
```python
class CalibratedModel:
    def __init__(self, base_model, calibrator):
        self.base_model = base_model
        self.calibrator = calibrator
    
    def predict_proba(self, X):
        raw_probs = self.base_model.predict_proba(X)[:, 1]
        calibrated_probs = self.calibrator.predict(raw_probs)
        return calibrated_probs
```

4. Target calibration results:
```
50-60% confidence → 52%+ accuracy
60-70% confidence → 58%+ accuracy
70-80% confidence → 65%+ accuracy
80%+   confidence → 72%+ accuracy
```

Deliverables:
- `models/calibration/calibrator.py` with multiple methods
- `models/calibration/validation.py` with reliability diagram
- Calibrated model artifacts
- Before/after calibration comparison

### Task 1.3: Implement Rest/Schedule Features

The schedule data is already ingested via nflverse but rest-related features are not computed.

Create `features/nfl/schedule_features.py`:

1. Calculate for each team entering a game:
   - `days_rest`: Days since last game (typically 7, but varies)
   - `is_short_week`: Less than 6 days rest (Thursday games)
   - `is_bye_week_return`: Coming off bye (10+ days rest)
   - `opponent_days_rest`: Opponent's rest days
   - `rest_advantage`: days_rest - opponent_days_rest
   - `consecutive_road_games`: Count of consecutive away games
   - `is_back_to_back_road`: Second consecutive road game

2. Calculate travel-related features:
   - `is_home`: Home team indicator
   - `travel_timezone_diff`: Timezone difference for away team
   - `is_cross_country`: 3+ timezone difference

3. Calculate schedule context:
   - `is_divisional_game`: Playing division rival
   - `is_primetime`: SNF, MNF, TNF indicator
   - `week_of_season`: Week number (early vs late season)
   - `is_playoff_implication`: Late season, both teams in contention

Implementation requirements:
- Must use ONLY data available BEFORE the game (no leakage)
- Handle season openers (no previous game)
- Handle bye weeks correctly
- Integrate into main `generate_features.py` pipeline

Deliverables:
- `features/nfl/schedule_features.py` with all functions
- Updated `features/nfl/generate_features.py` to include schedule features
- Unit tests for edge cases (season opener, bye weeks, playoffs)
- Feature importance analysis showing signal strength

### Task 1.4: Create Phase 1 Validation Script

Create `scripts/validate_phase1.py`:

1. Train ensemble on 2015-2022 data
2. Validate calibration on 2023 data  
3. Test on 2024 data (held out)
4. Generate report with:
   - Ensemble vs baseline comparison
   - Calibration curves (before/after)
   - Accuracy by confidence tier
   - ROI simulation at different confidence thresholds
   - Feature importance for new schedule features

Success criteria for Phase 1:
- [ ] Ensemble outperforms best single model by 1%+
- [ ] All confidence tiers show accuracy >= (bin_midpoint - 3%)
- [ ] Rest features show positive feature importance
- [ ] 2024 test accuracy >= 60%

## OUTPUT FORMAT

For each task, provide:
1. Complete, runnable code files
2. Clear docstrings and comments
3. Example usage / CLI commands
4. Expected outputs and success metrics

Start with Task 1.1 (Train Ensemble) first, then proceed sequentially.
```

---

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2: INJURY DATA + LIVE ODDS API (Week 2-3)
# ═══════════════════════════════════════════════════════════════════════════════

## Cursor Prompt — Phase 2

```
You are helping me optimize an NFL prediction model. This is Phase 2 of a 5-phase optimization plan.

## CONTEXT

Phase 1 completed:
- Stacking ensemble trained and operational
- Calibration fixed (monotonic accuracy across tiers)
- Rest/schedule features integrated
- Current accuracy: ~61% (up from 59.3%)

Phase 2 goal: Integrate high-impact external data sources
- Injury data (estimated +1-2% accuracy)
- Live odds API (enables real-time predictions, accurate edge calculation)

Repository structure:
```
ingestion/
├── nfl/
│   ├── schedule.py            # ✅ Working
│   ├── play_by_play.py        # ✅ Working
│   ├── team_stats.py          # ✅ Working
│   ├── injuries.py            # ❌ DOES NOT EXIST
│   └── odds_api.py            # ❌ DOES NOT EXIST
features/
├── nfl/
│   ├── injury_features.py     # ❌ DOES NOT EXIST
│   └── ...
config/
├── credentials.yaml           # ❌ DOES NOT EXIST (for API keys)
```

## PHASE 2 TASKS

### Task 2.1: Implement Injury Data Ingestion

Create `ingestion/nfl/injuries.py`:

**Data Source Priority:**
1. nflverse (check if injury data available via nfl_data_py)
2. ESPN API (unofficial but comprehensive)
3. NFL.com scraping (fallback)

**Required functionality:**

```python
class InjuryIngestion:
    """
    Ingest NFL injury report data from multiple sources.
    
    Injury reports are released:
    - Wednesday: First practice report
    - Thursday: Second practice report
    - Friday: Final injury designations
    - Saturday: Game-day inactives (for Sunday games)
    """
    
    def __init__(self, source: str = 'auto'):
        """
        Args:
            source: 'nflverse', 'espn', 'nfl_scrape', or 'auto' (try in order)
        """
        pass
    
    def fetch_current_injuries(self) -> pd.DataFrame:
        """
        Fetch current week's injury report.
        
        Returns DataFrame with columns:
        - season, week
        - team (abbreviation)
        - player_id, player_name
        - position (QB, WR, RB, etc.)
        - injury_type (knee, hamstring, concussion, illness, etc.)
        - practice_status (DNP, Limited, Full)
        - game_status (Out, Doubtful, Questionable, Probable, None)
        """
        pass
    
    def fetch_historical_injuries(self, seasons: list) -> pd.DataFrame:
        """
        Fetch historical injury data for backtesting.
        Need injury status AS OF game day (not after).
        """
        pass
    
    def get_team_injuries(self, team: str, week: int, season: int) -> pd.DataFrame:
        """Get injuries for a specific team entering a specific game."""
        pass
```

**ESPN API approach (if nflverse doesn't have injuries):**
```python
def _fetch_espn_injuries(self) -> pd.DataFrame:
    """
    ESPN injury API endpoint (unofficial):
    https://site.api.espn.com/apis/site/v2/sports/football/nfl/injuries
    
    Parse JSON response to extract injury data.
    """
    url = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/injuries"
    # Implementation...
```

**NFL.com scraping fallback:**
```python
def _scrape_nfl_injuries(self, team: str) -> pd.DataFrame:
    """
    Scrape official injury report from NFL.com
    URL pattern: https://www.nfl.com/teams/{team}/injuries
    
    Use BeautifulSoup to parse HTML table.
    """
    pass
```

Deliverables:
- `ingestion/nfl/injuries.py` with multi-source support
- Unit tests for each data source
- Historical injury data for 2022-2024 seasons (for backtesting)
- Data quality report (coverage %, missing data)

### Task 2.2: Implement Injury Features

Create `features/nfl/injury_features.py`:

**Position Impact Weights:**
```python
POSITION_WEIGHTS = {
    # Offense
    'QB': 10.0,    # Most important position by far
    'LT': 3.5,     # Protects QB blind side
    'RT': 2.5,
    'LG': 2.0,
    'RG': 2.0,
    'C': 2.5,
    'WR': 2.0,     # Depends on depth
    'RB': 1.5,
    'TE': 1.5,
    'FB': 0.5,
    
    # Defense
    'EDGE': 2.5,   # Pass rushers
    'DT': 2.0,
    'DE': 2.0,
    'LB': 1.5,
    'CB': 2.5,     # Cover receivers
    'S': 1.5,
    
    # Special Teams
    'K': 1.0,
    'P': 0.5,
    'LS': 0.3,
}
```

**Feature calculations:**
```python
def calculate_injury_features(injuries_df: pd.DataFrame, team: str, 
                               opponent: str, week: int) -> dict:
    """
    Calculate injury-related features for a matchup.
    
    Returns:
    - team_players_out: Count of players with 'Out' status
    - team_players_questionable: Count of 'Questionable' players
    - team_weighted_injury_impact: Position-weighted injury score
    - team_qb_status: 0=healthy, 1=questionable, 2=out
    - team_oline_injuries: Count of O-line injuries
    - team_skill_position_injuries: WR/RB/TE injuries
    - team_secondary_injuries: CB/S injuries
    - opponent_* : Same features for opponent
    - injury_advantage: team_impact - opponent_impact (negative = disadvantage)
    """
    pass

def calculate_qb_injury_status(injuries_df: pd.DataFrame, team: str) -> int:
    """
    Determine starting QB status.
    
    Returns:
    - 0: Starting QB healthy/full practice
    - 1: Starting QB questionable/limited
    - 2: Starting QB out (backup starting)
    
    This is the single most important injury feature.
    """
    pass

def calculate_oline_health(injuries_df: pd.DataFrame, team: str) -> dict:
    """
    Calculate offensive line health score.
    
    O-line injuries compound — multiple injuries worse than sum of parts.
    
    Returns:
    - oline_injuries_count: Raw count
    - oline_health_score: 0-100 scale (100 = fully healthy)
    - oline_is_compromised: 1 if 2+ starters out
    """
    pass
```

**Integration with existing pipeline:**
- Update `features/nfl/generate_features.py` to include injury features
- Handle missing injury data gracefully (use neutral values)
- Ensure no data leakage (use injury status as of game day, not after)

Deliverables:
- `features/nfl/injury_features.py` with all functions
- Updated feature pipeline integration
- Feature importance analysis
- Backtesting validation (does injury data improve predictions?)

### Task 2.3: Integrate The Odds API

Create `ingestion/nfl/odds_api.py`:

**API Details:**
- Provider: The Odds API (https://the-odds-api.com/)
- Free tier: 500 requests/month
- Endpoint: `https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds`

**Implementation:**
```python
import requests
from datetime import datetime
import pandas as pd
from typing import Optional
import json
import os

class OddsAPIClient:
    """
    Client for The Odds API.
    
    Free tier: 500 requests/month
    Caches responses to minimize API calls.
    """
    
    BASE_URL = "https://api.the-odds-api.com/v4"
    
    def __init__(self, api_key: Optional[str] = None, cache_dir: str = "data/odds_cache"):
        """
        Args:
            api_key: API key (or set ODDS_API_KEY env var)
            cache_dir: Directory for caching responses
        """
        self.api_key = api_key or os.environ.get('ODDS_API_KEY')
        if not self.api_key:
            raise ValueError("API key required. Set ODDS_API_KEY env var or pass api_key.")
        self.cache_dir = cache_dir
        self.requests_remaining = None
        self.requests_used = None
    
    def get_nfl_odds(self, 
                     markets: list = ['spreads', 'h2h', 'totals'],
                     bookmakers: list = None) -> pd.DataFrame:
        """
        Fetch current NFL odds.
        
        Args:
            markets: List of markets ('spreads', 'h2h', 'totals')
            bookmakers: List of bookmakers (None = all available)
        
        Returns:
            DataFrame with columns:
            - game_id, home_team, away_team, commence_time
            - bookmaker, market, outcome_name, price, point
        """
        pass
    
    def get_historical_odds(self, 
                            date: str,
                            markets: list = ['spreads']) -> pd.DataFrame:
        """
        Fetch historical odds (requires paid plan).
        For backtesting, we may need to use cached/CSV data instead.
        """
        pass
    
    def get_best_odds(self, 
                      odds_df: pd.DataFrame, 
                      team: str, 
                      market: str = 'spreads') -> dict:
        """
        Find best available odds for a team across all bookmakers.
        
        Returns:
            {
                'bookmaker': 'fanduel',
                'price': -108,
                'point': -3.5,  # For spreads
            }
        """
        pass
    
    def calculate_implied_probability(self, american_odds: int) -> float:
        """Convert American odds to implied probability."""
        if american_odds < 0:
            return abs(american_odds) / (abs(american_odds) + 100)
        else:
            return 100 / (american_odds + 100)
    
    def calculate_edge(self, model_prob: float, market_prob: float) -> float:
        """
        Calculate betting edge.
        
        Edge = Model Probability - Market Implied Probability
        Positive edge = model sees value the market doesn't
        """
        return model_prob - market_prob
    
    def _cache_response(self, cache_key: str, data: dict):
        """Cache API response to reduce API calls."""
        pass
    
    def _get_cached(self, cache_key: str) -> Optional[dict]:
        """Retrieve cached response if fresh."""
        pass
```

**Credentials management:**
Create `config/credentials.yaml.example`:
```yaml
# Copy to credentials.yaml and fill in your keys
# DO NOT commit credentials.yaml to git

odds_api:
  api_key: "your-api-key-here"

# Future APIs
sportsdata:
  api_key: ""
espn:
  # No key needed, but rate limit yourself
```

Add to `.gitignore`:
```
config/credentials.yaml
```

**Rate limiting and caching:**
- Cache responses for 1 hour (odds don't change that fast)
- Log remaining API calls after each request
- Warn when < 50 requests remaining

Deliverables:
- `ingestion/nfl/odds_api.py` with full implementation
- `config/credentials.yaml.example` template
- Caching layer to minimize API calls
- Integration with prediction pipeline
- CLI tool: `python -m ingestion.nfl.odds_api --fetch-current`

### Task 2.4: Create Phase 2 Validation Script

Create `scripts/validate_phase2.py`:

1. Validate injury data coverage:
   - % of games with injury data available
   - Compare predictions with/without injury features
   
2. Validate odds API integration:
   - Fetch current week odds successfully
   - Calculate edge for model predictions vs market
   
3. Run full evaluation:
   - Train model with new features on 2015-2022
   - Test on 2023-2024 combined (400+ games)
   
4. Generate report:
   - Feature importance (do injuries matter?)
   - Accuracy improvement from Phase 1
   - Edge distribution (how often does model find value?)

Success criteria for Phase 2:
- [ ] Injury data available for 80%+ of games
- [ ] Injury features show positive importance
- [ ] Odds API fetches data successfully
- [ ] Model accuracy >= 62%
- [ ] Model finds positive edge on 30%+ of games

## OUTPUT FORMAT

For each task, provide:
1. Complete, runnable code files
2. Error handling for API failures, missing data
3. Logging for debugging
4. Example usage and CLI commands

Start with Task 2.1 (Injury Ingestion) first.
```

---

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3: WEATHER FEATURES + 2024 VALIDATION (Week 3-4)
# ═══════════════════════════════════════════════════════════════════════════════

## Cursor Prompt — Phase 3

```
You are helping me optimize an NFL prediction model. This is Phase 3 of a 5-phase optimization plan.

## CONTEXT

Completed:
- Phase 1: Ensemble trained, calibration fixed, rest features added (~61% accuracy)
- Phase 2: Injury data integrated, live odds API working (~62% accuracy)

Phase 3 goals:
- Add weather features (affects outdoor games, ~50% of games)
- Comprehensive validation on 2024 season (held-out data)
- Ensure no data leakage in full pipeline

Repository additions from previous phases:
```
ingestion/nfl/
├── injuries.py       # ✅ From Phase 2
├── odds_api.py       # ✅ From Phase 2
├── weather.py        # ❌ DOES NOT EXIST
features/nfl/
├── injury_features.py    # ✅ From Phase 2
├── schedule_features.py  # ✅ From Phase 1
├── weather_features.py   # ❌ DOES NOT EXIST
```

## PHASE 3 TASKS

### Task 3.1: Implement Weather Data Ingestion

Create `ingestion/nfl/weather.py`:

**Data Source:** Open-Meteo API (free, no API key required)
- Historical weather: `https://archive-api.open-meteo.com/v1/archive`
- Forecast weather: `https://api.open-meteo.com/v1/forecast`

**Stadium Data:**
```python
# All 32 NFL team stadiums with coordinates and dome status
STADIUMS = {
    # AFC East
    'BUF': {'name': 'Highmark Stadium', 'lat': 42.7738, 'lon': -78.7870, 'dome': False, 'timezone': 'America/New_York'},
    'MIA': {'name': 'Hard Rock Stadium', 'lat': 25.9580, 'lon': -80.2389, 'dome': False, 'timezone': 'America/New_York'},
    'NE': {'name': 'Gillette Stadium', 'lat': 42.0909, 'lon': -71.2643, 'dome': False, 'timezone': 'America/New_York'},
    'NYJ': {'name': 'MetLife Stadium', 'lat': 40.8135, 'lon': -74.0745, 'dome': False, 'timezone': 'America/New_York'},
    
    # AFC North
    'BAL': {'name': 'M&T Bank Stadium', 'lat': 39.2780, 'lon': -76.6227, 'dome': False, 'timezone': 'America/New_York'},
    'CIN': {'name': 'Paycor Stadium', 'lat': 39.0955, 'lon': -84.5161, 'dome': False, 'timezone': 'America/New_York'},
    'CLE': {'name': 'Cleveland Browns Stadium', 'lat': 41.5061, 'lon': -81.6995, 'dome': False, 'timezone': 'America/New_York'},
    'PIT': {'name': 'Acrisure Stadium', 'lat': 40.4468, 'lon': -80.0158, 'dome': False, 'timezone': 'America/New_York'},
    
    # AFC South
    'HOU': {'name': 'NRG Stadium', 'lat': 29.6847, 'lon': -95.4107, 'dome': True, 'timezone': 'America/Chicago'},
    'IND': {'name': 'Lucas Oil Stadium', 'lat': 39.7601, 'lon': -86.1639, 'dome': True, 'timezone': 'America/Indiana/Indianapolis'},
    'JAX': {'name': 'EverBank Stadium', 'lat': 30.3239, 'lon': -81.6373, 'dome': False, 'timezone': 'America/New_York'},
    'TEN': {'name': 'Nissan Stadium', 'lat': 36.1665, 'lon': -86.7713, 'dome': False, 'timezone': 'America/Chicago'},
    
    # AFC West
    'DEN': {'name': 'Empower Field', 'lat': 39.7439, 'lon': -105.0201, 'dome': False, 'timezone': 'America/Denver'},
    'KC': {'name': 'GEHA Field', 'lat': 39.0489, 'lon': -94.4839, 'dome': False, 'timezone': 'America/Chicago'},
    'LV': {'name': 'Allegiant Stadium', 'lat': 36.0909, 'lon': -115.1833, 'dome': True, 'timezone': 'America/Los_Angeles'},
    'LAC': {'name': 'SoFi Stadium', 'lat': 33.9535, 'lon': -118.3392, 'dome': True, 'timezone': 'America/Los_Angeles'},
    
    # NFC East
    'DAL': {'name': 'AT&T Stadium', 'lat': 32.7473, 'lon': -97.0945, 'dome': True, 'timezone': 'America/Chicago'},
    'NYG': {'name': 'MetLife Stadium', 'lat': 40.8135, 'lon': -74.0745, 'dome': False, 'timezone': 'America/New_York'},
    'PHI': {'name': 'Lincoln Financial Field', 'lat': 39.9008, 'lon': -75.1675, 'dome': False, 'timezone': 'America/New_York'},
    'WAS': {'name': 'Commanders Field', 'lat': 38.9076, 'lon': -76.8645, 'dome': False, 'timezone': 'America/New_York'},
    
    # NFC North
    'CHI': {'name': 'Soldier Field', 'lat': 41.8623, 'lon': -87.6167, 'dome': False, 'timezone': 'America/Chicago'},
    'DET': {'name': 'Ford Field', 'lat': 42.3400, 'lon': -83.0456, 'dome': True, 'timezone': 'America/Detroit'},
    'GB': {'name': 'Lambeau Field', 'lat': 44.5013, 'lon': -88.0622, 'dome': False, 'timezone': 'America/Chicago'},
    'MIN': {'name': 'U.S. Bank Stadium', 'lat': 44.9737, 'lon': -93.2577, 'dome': True, 'timezone': 'America/Chicago'},
    
    # NFC South
    'ATL': {'name': 'Mercedes-Benz Stadium', 'lat': 33.7554, 'lon': -84.4010, 'dome': True, 'timezone': 'America/New_York'},
    'CAR': {'name': 'Bank of America Stadium', 'lat': 35.2258, 'lon': -80.8528, 'dome': False, 'timezone': 'America/New_York'},
    'NO': {'name': 'Caesars Superdome', 'lat': 29.9511, 'lon': -90.0812, 'dome': True, 'timezone': 'America/Chicago'},
    'TB': {'name': 'Raymond James Stadium', 'lat': 27.9759, 'lon': -82.5033, 'dome': False, 'timezone': 'America/New_York'},
    
    # NFC West
    'ARI': {'name': 'State Farm Stadium', 'lat': 33.5276, 'lon': -112.2626, 'dome': True, 'timezone': 'America/Phoenix'},
    'LAR': {'name': 'SoFi Stadium', 'lat': 33.9535, 'lon': -118.3392, 'dome': True, 'timezone': 'America/Los_Angeles'},
    'SF': {'name': "Levi's Stadium", 'lat': 37.4033, 'lon': -121.9694, 'dome': False, 'timezone': 'America/Los_Angeles'},
    'SEA': {'name': 'Lumen Field', 'lat': 47.5952, 'lon': -122.3316, 'dome': False, 'timezone': 'America/Los_Angeles'},
}

# Teams with indoor/dome stadiums (weather doesn't affect game)
DOME_TEAMS = [team for team, info in STADIUMS.items() if info['dome']]
# ['HOU', 'IND', 'LV', 'LAC', 'DAL', 'DET', 'MIN', 'ATL', 'NO', 'ARI', 'LAR']
```

**Implementation:**
```python
class WeatherIngestion:
    """
    Fetch weather data for NFL games.
    
    Uses Open-Meteo API:
    - Free, no API key required
    - Historical data available back to 1940
    - Hourly resolution
    """
    
    def __init__(self, cache_dir: str = "data/weather_cache"):
        self.cache_dir = cache_dir
    
    def get_game_weather(self, 
                         home_team: str, 
                         game_datetime: datetime,
                         use_cache: bool = True) -> dict:
        """
        Get weather conditions for a game.
        
        Args:
            home_team: Home team abbreviation
            game_datetime: Game start time (UTC or local)
        
        Returns:
            {
                'temperature_f': 45,
                'feels_like_f': 38,
                'wind_speed_mph': 12,
                'wind_gust_mph': 22,
                'wind_direction': 'NW',
                'precipitation_prob': 20,
                'precipitation_inches': 0.0,
                'humidity_pct': 65,
                'visibility_miles': 10,
                'weather_code': 3,  # WMO code
                'weather_description': 'Overcast',
                'is_dome': False,
            }
        """
        # Check if dome stadium
        stadium = STADIUMS.get(home_team)
        if stadium and stadium['dome']:
            return self._get_dome_conditions()
        
        # Fetch from Open-Meteo
        if game_datetime < datetime.now():
            return self._fetch_historical(stadium, game_datetime)
        else:
            return self._fetch_forecast(stadium, game_datetime)
    
    def _fetch_historical(self, stadium: dict, game_datetime: datetime) -> dict:
        """
        Fetch historical weather from Open-Meteo archive.
        
        Endpoint: https://archive-api.open-meteo.com/v1/archive
        """
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            'latitude': stadium['lat'],
            'longitude': stadium['lon'],
            'start_date': game_datetime.strftime('%Y-%m-%d'),
            'end_date': game_datetime.strftime('%Y-%m-%d'),
            'hourly': 'temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,wind_gusts_10m,weather_code',
            'temperature_unit': 'fahrenheit',
            'wind_speed_unit': 'mph',
            'precipitation_unit': 'inch',
            'timezone': stadium['timezone'],
        }
        # Implementation...
        pass
    
    def _fetch_forecast(self, stadium: dict, game_datetime: datetime) -> dict:
        """
        Fetch forecast weather from Open-Meteo.
        
        Endpoint: https://api.open-meteo.com/v1/forecast
        """
        pass
    
    def _get_dome_conditions(self) -> dict:
        """Return standard dome conditions (controlled environment)."""
        return {
            'temperature_f': 72,
            'feels_like_f': 72,
            'wind_speed_mph': 0,
            'wind_gust_mph': 0,
            'precipitation_prob': 0,
            'humidity_pct': 50,
            'is_dome': True,
            'weather_description': 'Indoor/Dome',
        }
    
    def fetch_season_weather(self, season: int) -> pd.DataFrame:
        """
        Fetch weather for all games in a season (for backtesting).
        
        Batch fetch to minimize API calls.
        Cache results for reuse.
        """
        pass
```

Deliverables:
- `ingestion/nfl/weather.py` with full implementation
- Stadium coordinates for all 32 teams
- Historical weather data for 2022-2024 seasons
- Caching layer for API responses

### Task 3.2: Implement Weather Features

Create `features/nfl/weather_features.py`:

**Core features:**
```python
def calculate_weather_features(weather_data: dict) -> dict:
    """
    Calculate weather-related features for a game.
    
    Weather primarily affects:
    1. Passing game (wind, precipitation)
    2. Kicking game (wind)
    3. Player performance (extreme temps)
    
    Returns:
        {
            # Binary indicators
            'is_dome': 0/1,
            'is_cold_game': 0/1,        # < 32°F
            'is_freezing_game': 0/1,    # < 20°F
            'is_hot_game': 0/1,         # > 85°F
            'is_windy': 0/1,            # > 15 mph
            'is_very_windy': 0/1,       # > 25 mph
            'is_precipitation': 0/1,    # Rain/snow likely
            
            # Continuous features
            'temperature_f': float,
            'wind_speed_mph': float,
            'precipitation_prob': float,
            
            # Composite scores
            'passing_conditions_score': float,  # 0-100, higher = better for passing
            'kicking_conditions_score': float,  # 0-100, higher = better for kicking
            'weather_advantage_home': float,    # Home team acclimation advantage
        }
    """
    if weather_data.get('is_dome'):
        return _get_neutral_weather_features()
    
    temp = weather_data.get('temperature_f', 60)
    wind = weather_data.get('wind_speed_mph', 0)
    precip = weather_data.get('precipitation_prob', 0)
    
    # Calculate passing conditions (wind and precipitation hurt passing)
    passing_score = 100
    passing_score -= min(wind * 2, 40)  # Wind penalty (max -40)
    passing_score -= min(precip * 0.3, 30)  # Precipitation penalty (max -30)
    passing_score = max(passing_score, 0)
    
    # Calculate kicking conditions (wind is primary factor)
    kicking_score = 100
    kicking_score -= min(wind * 3, 60)  # Wind penalty (max -60)
    kicking_score -= min(precip * 0.2, 20)  # Precipitation penalty (max -20)
    kicking_score = max(kicking_score, 0)
    
    return {
        'is_dome': 0,
        'is_cold_game': 1 if temp < 32 else 0,
        'is_freezing_game': 1 if temp < 20 else 0,
        'is_hot_game': 1 if temp > 85 else 0,
        'is_windy': 1 if wind > 15 else 0,
        'is_very_windy': 1 if wind > 25 else 0,
        'is_precipitation': 1 if precip > 50 else 0,
        'temperature_f': temp,
        'wind_speed_mph': wind,
        'precipitation_prob': precip,
        'passing_conditions_score': passing_score,
        'kicking_conditions_score': kicking_score,
    }


def calculate_weather_matchup_features(weather: dict, 
                                        home_team: str, 
                                        away_team: str) -> dict:
    """
    Calculate team-specific weather advantages.
    
    Some teams are built for bad weather (run-heavy, strong defense).
    Some teams rely on passing (hurt by wind/precipitation).
    
    Returns:
        {
            'home_weather_advantage': float,  # Home team's weather acclimation
            'pass_heavy_team_disadvantage': float,  # For pass-first teams in bad weather
        }
    """
    # Teams that play in cold/bad weather regularly (acclimated)
    COLD_WEATHER_TEAMS = ['BUF', 'GB', 'CHI', 'NE', 'NYG', 'NYJ', 'CLE', 'PIT', 'DEN']
    
    # Teams with pass-heavy offenses (hurt by wind/precipitation)
    # This would ideally be calculated from team stats, not hardcoded
    
    pass
```

**Integration:**
- Add weather features to main feature pipeline
- Ensure historical weather available for backtesting
- Handle missing weather data gracefully

Deliverables:
- `features/nfl/weather_features.py`
- Integration with `generate_features.py`
- Analysis: weather feature importance by game type

### Task 3.3: Comprehensive 2024 Validation

Create `scripts/validate_2024.py`:

**Validation requirements:**
1. 2024 season must be completely held out (never seen during training)
2. Simulate week-by-week predictions (no future data leakage)
3. Compare model predictions vs actual outcomes AND vs market

```python
"""
2024 Season Validation Script

This script validates the model on the complete 2024 NFL season,
simulating real-world deployment conditions.
"""

import pandas as pd
from datetime import datetime
from typing import Dict, List

def validate_2024_season(model_path: str) -> Dict:
    """
    Run comprehensive validation on 2024 season.
    
    Process:
    1. Load model trained on 2015-2023 data
    2. For each week 1-18:
       a. Generate features using only data available before that week
       b. Make predictions
       c. Compare to actual outcomes
       d. Compare to betting market (closing lines)
    3. Calculate overall metrics
    
    Returns:
        {
            'overall_accuracy': float,
            'overall_roi': float,
            'accuracy_by_week': dict,
            'accuracy_by_confidence': dict,
            'vs_market_performance': dict,
            'calibration_results': dict,
        }
    """
    pass


def simulate_weekly_predictions(model, week: int, season: int = 2024) -> pd.DataFrame:
    """
    Simulate predictions for a single week.
    
    CRITICAL: Only use data available BEFORE this week.
    - Team stats through week-1
    - Injury report as of Friday before game
    - Weather forecast (not actual)
    - Odds available before game
    
    Returns DataFrame with:
    - game_id, home_team, away_team
    - model_home_prob, model_pick
    - market_implied_prob, market_pick
    - actual_winner, model_correct, market_correct
    """
    pass


def calculate_roi(predictions_df: pd.DataFrame, 
                  confidence_threshold: float = 0.0,
                  edge_threshold: float = 0.0) -> dict:
    """
    Calculate ROI assuming flat betting on model picks.
    
    Args:
        confidence_threshold: Only bet when model confidence > threshold
        edge_threshold: Only bet when model edge vs market > threshold
    
    Returns:
        {
            'total_bets': int,
            'wins': int,
            'losses': int,
            'win_rate': float,
            'roi': float,
            'units_won': float,
        }
    """
    pass


def generate_validation_report(results: dict) -> str:
    """
    Generate markdown report of validation results.
    """
    report = f"""
# 2024 Season Validation Report

## Overall Performance
- **Accuracy**: {results['overall_accuracy']:.1%}
- **ROI**: {results['overall_roi']:.2%}
- **Games Predicted**: {results['total_games']}

## Performance by Confidence Tier
| Confidence | Games | Accuracy | ROI |
|------------|-------|----------|-----|
"""
    # Add rows...
    return report
```

**Success criteria:**
- [ ] 2024 accuracy within 3% of 2023 test accuracy (no overfitting)
- [ ] Calibration holds on 2024 data
- [ ] Positive ROI on 70%+ confidence picks
- [ ] Model beats market on 55%+ of games

Deliverables:
- `scripts/validate_2024.py`
- Validation report (markdown)
- Week-by-week accuracy chart
- Calibration curve for 2024

### Task 3.4: Data Leakage Audit

Create `scripts/audit_data_leakage.py`:

**Audit checks:**
```python
"""
Data Leakage Audit Script

Ensures no future information leaks into model training or prediction.
"""

def audit_feature_pipeline(game_id: str, features: dict) -> List[str]:
    """
    Audit features for a single game for potential leakage.
    
    Checks:
    1. Team stats only include games BEFORE this game
    2. Injury data is from pre-game report (not post-game)
    3. Weather is forecast (for future games) or game-day (for historical)
    4. Odds are pre-game closing lines (not live/post-game)
    5. No outcome information (score, winner) in features
    
    Returns:
        List of leakage warnings (empty if clean)
    """
    warnings = []
    
    # Check team stats timing
    # Check injury report date
    # Check weather data date
    # Check odds timestamp
    
    return warnings


def audit_train_test_split(train_df: pd.DataFrame, 
                           test_df: pd.DataFrame) -> List[str]:
    """
    Audit train/test split for temporal leakage.
    
    Checks:
    1. All training games occur BEFORE all test games
    2. No game appears in both sets
    3. Rolling features don't peek into future
    """
    pass


def audit_full_pipeline():
    """
    Run full audit on entire pipeline.
    
    1. Load sample of games from each season
    2. Regenerate features
    3. Check each feature for leakage
    4. Generate audit report
    """
    pass
```

Deliverables:
- `scripts/audit_data_leakage.py`
- Audit report confirming no leakage
- Documentation of temporal boundaries

## OUTPUT FORMAT

For each task:
1. Complete, runnable code
2. Unit tests for critical functions
3. Example outputs
4. Integration with existing pipeline

Start with Task 3.1 (Weather Ingestion).
```

---

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 4: HYPERPARAMETER TUNING + FEATURE SELECTION (Week 5-6)
# ═══════════════════════════════════════════════════════════════════════════════

## Cursor Prompt — Phase 4

```
You are helping me optimize an NFL prediction model. This is Phase 4 of a 5-phase optimization plan.

## CONTEXT

Completed phases:
- Phase 1: Ensemble + calibration + rest features (~61%)
- Phase 2: Injury data + odds API (~62%)
- Phase 3: Weather features + 2024 validation (~63%)

Phase 4 goals:
- Hyperparameter optimization for all models
- Feature selection to reduce overfitting
- Advanced ensemble tuning
- Target: 64-66% accuracy

Current model configuration (defaults, not optimized):
```yaml
# FT-Transformer
ft_transformer:
  d_model: 64
  n_heads: 4
  n_layers: 3
  d_ff: 256
  dropout: 0.1
  learning_rate: 0.0001

# XGBoost
xgboost:
  n_estimators: 100
  max_depth: 3
  learning_rate: 0.1
  subsample: 1.0
  colsample_bytree: 1.0

# Stacking Ensemble
ensemble:
  meta_model: logistic_regression
  use_probabilities: true
  include_original_features: false
```

## PHASE 4 TASKS

### Task 4.1: Implement Hyperparameter Optimization with Optuna

Create `models/tuning/hyperopt.py`:

```python
"""
Hyperparameter optimization using Optuna.

Optuna advantages:
- Efficient pruning of bad trials
- Built-in visualization
- Supports all model types
"""

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import pandas as pd
import numpy as np
from typing import Dict, Any, Callable
import joblib

class ModelOptimizer:
    """
    Hyperparameter optimizer for NFL prediction models.
    
    Usage:
        optimizer = ModelOptimizer(X_train, y_train, X_val, y_val)
        best_params = optimizer.optimize_xgboost(n_trials=100)
    """
    
    def __init__(self, 
                 X_train: pd.DataFrame, 
                 y_train: pd.Series,
                 X_val: pd.DataFrame, 
                 y_val: pd.Series,
                 metric: str = 'accuracy'):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.metric = metric
    
    def optimize_xgboost(self, n_trials: int = 100) -> Dict[str, Any]:
        """
        Optimize XGBoost hyperparameters.
        
        Search space:
        - n_estimators: 50-500
        - max_depth: 2-10
        - learning_rate: 0.01-0.3
        - subsample: 0.6-1.0
        - colsample_bytree: 0.6-1.0
        - min_child_weight: 1-10
        - reg_alpha: 0-1 (L1)
        - reg_lambda: 0-1 (L2)
        """
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 2, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
                'random_state': 42,
                'use_label_encoder': False,
                'eval_metric': 'logloss',
            }
            
            from xgboost import XGBClassifier
            model = XGBClassifier(**params)
            model.fit(
                self.X_train, self.y_train,
                eval_set=[(self.X_val, self.y_val)],
                early_stopping_rounds=50,
                verbose=False
            )
            
            preds = model.predict(self.X_val)
            accuracy = (preds == self.y_val).mean()
            
            return accuracy
        
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(n_startup_trials=10)
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        return study.best_params
    
    def optimize_ft_transformer(self, n_trials: int = 50) -> Dict[str, Any]:
        """
        Optimize FT-Transformer hyperparameters.
        
        Search space:
        - d_model: 32, 64, 128, 256
        - n_heads: 2, 4, 8
        - n_layers: 2, 3, 4, 6
        - d_ff_multiplier: 2, 4
        - dropout: 0.0-0.3
        - learning_rate: 1e-5 to 1e-3
        - weight_decay: 1e-6 to 1e-3
        - batch_size: 32, 64, 128
        """
        def objective(trial):
            params = {
                'd_model': trial.suggest_categorical('d_model', [32, 64, 128, 256]),
                'n_heads': trial.suggest_categorical('n_heads', [2, 4, 8]),
                'n_layers': trial.suggest_int('n_layers', 2, 6),
                'd_ff_multiplier': trial.suggest_categorical('d_ff_multiplier', [2, 4]),
                'dropout': trial.suggest_float('dropout', 0.0, 0.3),
                'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
                'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
                'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
            }
            
            # Ensure d_model is divisible by n_heads
            if params['d_model'] % params['n_heads'] != 0:
                return 0.5  # Invalid config, return bad score
            
            # Train FT-Transformer with these params
            # (Implementation depends on your FT-Transformer class)
            
            return accuracy
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        return study.best_params
    
    def optimize_ensemble_weights(self, 
                                   base_predictions: Dict[str, np.ndarray],
                                   n_trials: int = 100) -> Dict[str, float]:
        """
        Optimize ensemble weights for combining base models.
        
        Instead of using a meta-learner, directly optimize weights.
        
        Args:
            base_predictions: Dict of model_name -> predicted_probabilities
        """
        def objective(trial):
            weights = {}
            for model_name in base_predictions.keys():
                weights[model_name] = trial.suggest_float(
                    f'weight_{model_name}', 0.0, 1.0
                )
            
            # Normalize weights
            total = sum(weights.values())
            weights = {k: v/total for k, v in weights.items()}
            
            # Calculate weighted ensemble prediction
            ensemble_pred = np.zeros_like(list(base_predictions.values())[0])
            for model_name, preds in base_predictions.items():
                ensemble_pred += weights[model_name] * preds
            
            # Evaluate
            binary_preds = (ensemble_pred > 0.5).astype(int)
            accuracy = (binary_preds == self.y_val).mean()
            
            return accuracy
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        # Extract and normalize final weights
        best_weights = {
            k.replace('weight_', ''): v 
            for k, v in study.best_params.items()
        }
        total = sum(best_weights.values())
        best_weights = {k: v/total for k, v in best_weights.items()}
        
        return best_weights
    
    def run_full_optimization(self, n_trials_per_model: int = 50) -> Dict:
        """
        Run optimization for all models and ensemble.
        
        Returns:
            {
                'xgboost': {...params...},
                'ft_transformer': {...params...},
                'ensemble_weights': {...weights...},
                'expected_accuracy': float,
            }
        """
        results = {}
        
        print("Optimizing XGBoost...")
        results['xgboost'] = self.optimize_xgboost(n_trials_per_model)
        
        print("Optimizing FT-Transformer...")
        results['ft_transformer'] = self.optimize_ft_transformer(n_trials_per_model)
        
        # Train models with optimized params, then optimize ensemble
        # ...
        
        return results


def save_optimization_results(results: Dict, path: str):
    """Save optimization results to YAML for reproducibility."""
    import yaml
    with open(path, 'w') as f:
        yaml.dump(results, f)


def visualize_optimization(study: optuna.Study, output_dir: str):
    """Generate Optuna visualization plots."""
    from optuna.visualization import (
        plot_optimization_history,
        plot_param_importances,
        plot_parallel_coordinate,
    )
    
    # Save plots
    fig = plot_optimization_history(study)
    fig.write_html(f"{output_dir}/optimization_history.html")
    
    fig = plot_param_importances(study)
    fig.write_html(f"{output_dir}/param_importances.html")
```

Deliverables:
- `models/tuning/hyperopt.py` with all optimizers
- Optimized config file: `config/models/optimized.yaml`
- Optimization visualizations
- Before/after performance comparison

### Task 4.2: Implement Feature Selection

Create `features/selection/selector.py`:

```python
"""
Feature selection to reduce overfitting and improve generalization.

Methods:
1. SHAP-based importance (model-agnostic)
2. Permutation importance
3. Recursive feature elimination (RFE)
4. Correlation filtering
"""

import numpy as np
import pandas as pd
from typing import List, Tuple
import shap

class FeatureSelector:
    """
    Feature selection for NFL prediction model.
    
    Goal: Reduce features from ~100 to ~30-50 most predictive
    """
    
    def __init__(self, model, X: pd.DataFrame, y: pd.Series):
        self.model = model
        self.X = X
        self.y = y
        self.feature_names = X.columns.tolist()
    
    def shap_importance(self, n_samples: int = 1000) -> pd.DataFrame:
        """
        Calculate SHAP-based feature importance.
        
        SHAP (SHapley Additive exPlanations) provides:
        - Global feature importance
        - Direction of impact (positive/negative)
        - Interaction effects
        
        Returns:
            DataFrame with columns: feature, importance, std
        """
        # Sample for speed
        if len(self.X) > n_samples:
            X_sample = self.X.sample(n_samples, random_state=42)
        else:
            X_sample = self.X
        
        # Calculate SHAP values
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X_sample)
        
        # If binary classification, take positive class
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        # Calculate importance (mean absolute SHAP value)
        importance = np.abs(shap_values).mean(axis=0)
        std = np.abs(shap_values).std(axis=0)
        
        df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance,
            'std': std,
        }).sort_values('importance', ascending=False)
        
        return df
    
    def permutation_importance(self, n_repeats: int = 10) -> pd.DataFrame:
        """
        Calculate permutation importance.
        
        Measures accuracy drop when feature is shuffled.
        More robust to correlation than SHAP.
        """
        from sklearn.inspection import permutation_importance
        
        result = permutation_importance(
            self.model, self.X, self.y, 
            n_repeats=n_repeats, 
            random_state=42,
            n_jobs=-1
        )
        
        df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': result.importances_mean,
            'std': result.importances_std,
        }).sort_values('importance', ascending=False)
        
        return df
    
    def correlation_filter(self, threshold: float = 0.95) -> List[str]:
        """
        Remove highly correlated features.
        
        When two features have correlation > threshold,
        keep the one with higher target correlation.
        
        Returns:
            List of features to DROP
        """
        corr_matrix = self.X.corr().abs()
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Find pairs with high correlation
        to_drop = set()
        for column in upper_tri.columns:
            correlated = upper_tri.index[upper_tri[column] > threshold].tolist()
            if correlated:
                # Keep feature with higher target correlation
                target_corrs = self.X[correlated + [column]].corrwith(self.y).abs()
                worst = target_corrs.idxmin()
                to_drop.add(worst)
        
        return list(to_drop)
    
    def select_top_features(self, 
                            n_features: int = 50,
                            method: str = 'shap') -> List[str]:
        """
        Select top N features by importance.
        
        Args:
            n_features: Number of features to keep
            method: 'shap' or 'permutation'
        
        Returns:
            List of feature names to KEEP
        """
        if method == 'shap':
            importance_df = self.shap_importance()
        else:
            importance_df = self.permutation_importance()
        
        return importance_df.head(n_features)['feature'].tolist()
    
    def recursive_elimination(self, 
                              min_features: int = 20,
                              step: int = 5) -> Tuple[List[str], pd.DataFrame]:
        """
        Recursive Feature Elimination with cross-validation.
        
        Iteratively removes least important features and tracks performance.
        
        Returns:
            (selected_features, performance_by_n_features)
        """
        from sklearn.feature_selection import RFECV
        from sklearn.model_selection import TimeSeriesSplit
        
        # Use time series CV to respect temporal order
        cv = TimeSeriesSplit(n_splits=5)
        
        selector = RFECV(
            estimator=self.model,
            step=step,
            cv=cv,
            scoring='accuracy',
            min_features_to_select=min_features,
            n_jobs=-1
        )
        
        selector.fit(self.X, self.y)
        
        selected = self.X.columns[selector.support_].tolist()
        
        # Performance curve
        performance = pd.DataFrame({
            'n_features': range(min_features, len(self.feature_names), step),
            'accuracy': selector.cv_results_['mean_test_score'],
        })
        
        return selected, performance
    
    def get_final_feature_set(self) -> List[str]:
        """
        Comprehensive feature selection pipeline.
        
        1. Remove highly correlated features (>0.95)
        2. Calculate SHAP importance
        3. Select top 50 by importance
        4. Validate with permutation importance
        
        Returns:
            Final list of features to use
        """
        # Step 1: Correlation filter
        to_drop = self.correlation_filter(threshold=0.95)
        X_filtered = self.X.drop(columns=to_drop)
        
        print(f"Dropped {len(to_drop)} correlated features")
        
        # Step 2: SHAP importance on filtered features
        self.X = X_filtered
        self.feature_names = X_filtered.columns.tolist()
        
        shap_df = self.shap_importance()
        
        # Step 3: Select top 50
        top_features = shap_df.head(50)['feature'].tolist()
        
        # Step 4: Validate these are also important by permutation
        perm_df = self.permutation_importance()
        perm_top = set(perm_df.head(60)['feature'].tolist())
        
        # Keep features that are in both top lists
        final_features = [f for f in top_features if f in perm_top]
        
        # If too few, add more from SHAP
        if len(final_features) < 30:
            for f in top_features:
                if f not in final_features:
                    final_features.append(f)
                if len(final_features) >= 40:
                    break
        
        print(f"Selected {len(final_features)} final features")
        
        return final_features


def visualize_feature_importance(importance_df: pd.DataFrame, 
                                  output_path: str,
                                  top_n: int = 30):
    """Generate feature importance visualization."""
    import matplotlib.pyplot as plt
    
    top = importance_df.head(top_n)
    
    fig, ax = plt.subplots(figsize=(10, 12))
    ax.barh(top['feature'], top['importance'])
    ax.set_xlabel('Importance')
    ax.set_title(f'Top {top_n} Features by Importance')
    ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
```

Deliverables:
- `features/selection/selector.py`
- Feature importance report
- Selected feature list saved to config
- Before/after feature count comparison

### Task 4.3: Implement Cross-Validation Framework

Create `models/validation/cv.py`:

```python
"""
Time-series cross-validation for NFL predictions.

Standard k-fold CV is WRONG for time series data!
Must use temporal splits to avoid data leakage.
"""

import pandas as pd
import numpy as np
from typing import Generator, Tuple, List
from sklearn.model_selection import TimeSeriesSplit

class NFLCrossValidator:
    """
    Cross-validation specifically designed for NFL game prediction.
    
    Approaches:
    1. Season-based CV: Train on N seasons, validate on season N+1
    2. Walk-forward CV: Expanding window, always predict next week
    3. Blocked CV: Train on seasons 1-3, validate on 4, then 2-4 validate on 5, etc.
    """
    
    def __init__(self, data: pd.DataFrame, date_column: str = 'gameday'):
        """
        Args:
            data: DataFrame with all games
            date_column: Column containing game date
        """
        self.data = data.sort_values(date_column)
        self.date_column = date_column
    
    def season_cv(self, 
                  n_train_seasons: int = 5,
                  n_test_seasons: int = 1) -> Generator[Tuple[pd.Index, pd.Index], None, None]:
        """
        Season-based cross-validation.
        
        Example with n_train=5, n_test=1:
        - Fold 1: Train 2015-2019, Test 2020
        - Fold 2: Train 2016-2020, Test 2021
        - Fold 3: Train 2017-2021, Test 2022
        - etc.
        
        Yields:
            (train_indices, test_indices)
        """
        seasons = sorted(self.data['season'].unique())
        
        for i in range(len(seasons) - n_train_seasons - n_test_seasons + 1):
            train_seasons = seasons[i:i+n_train_seasons]
            test_seasons = seasons[i+n_train_seasons:i+n_train_seasons+n_test_seasons]
            
            train_mask = self.data['season'].isin(train_seasons)
            test_mask = self.data['season'].isin(test_seasons)
            
            yield self.data[train_mask].index, self.data[test_mask].index
    
    def walk_forward_cv(self, 
                        min_train_weeks: int = 100,
                        step_weeks: int = 4) -> Generator[Tuple[pd.Index, pd.Index], None, None]:
        """
        Walk-forward cross-validation.
        
        Start with min_train_weeks, predict next step_weeks,
        then expand training window and repeat.
        
        This most closely simulates real-world deployment.
        """
        # Create week index
        self.data = self.data.sort_values(self.date_column).reset_index(drop=True)
        
        n_samples = len(self.data)
        
        train_end = min_train_weeks
        while train_end + step_weeks <= n_samples:
            train_idx = self.data.index[:train_end]
            test_idx = self.data.index[train_end:train_end+step_weeks]
            
            yield train_idx, test_idx
            
            train_end += step_weeks
    
    def evaluate_with_cv(self, 
                         model_class,
                         model_params: dict,
                         X: pd.DataFrame,
                         y: pd.Series,
                         cv_method: str = 'season') -> pd.DataFrame:
        """
        Evaluate model with cross-validation.
        
        Returns:
            DataFrame with fold-by-fold results
        """
        results = []
        
        if cv_method == 'season':
            cv_splits = self.season_cv()
        else:
            cv_splits = self.walk_forward_cv()
        
        for fold, (train_idx, test_idx) in enumerate(cv_splits):
            X_train, X_test = X.loc[train_idx], X.loc[test_idx]
            y_train, y_test = y.loc[train_idx], y.loc[test_idx]
            
            # Train model
            model = model_class(**model_params)
            model.fit(X_train, y_train)
            
            # Evaluate
            preds = model.predict(X_test)
            probs = model.predict_proba(X_test)[:, 1]
            
            accuracy = (preds == y_test).mean()
            
            results.append({
                'fold': fold,
                'train_size': len(train_idx),
                'test_size': len(test_idx),
                'accuracy': accuracy,
            })
        
        return pd.DataFrame(results)
```

Deliverables:
- `models/validation/cv.py`
- CV results for each model
- Variance analysis (are results stable across folds?)

### Task 4.4: Create Phase 4 Optimization Script

Create `scripts/run_phase4_optimization.py`:

```python
"""
Phase 4: Full hyperparameter optimization pipeline.

Usage:
    python scripts/run_phase4_optimization.py --trials 100

Output:
    - Optimized model configs
    - Selected feature list
    - Performance comparison report
"""

import argparse
from models.tuning.hyperopt import ModelOptimizer
from features.selection.selector import FeatureSelector
from models.validation.cv import NFLCrossValidator

def main(n_trials: int = 100):
    # 1. Load data
    # 2. Run feature selection
    # 3. Optimize each model
    # 4. Optimize ensemble weights
    # 5. Validate with CV
    # 6. Save results
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials', type=int, default=100)
    args = parser.parse_args()
    main(args.trials)
```

Success criteria for Phase 4:
- [ ] Hyperparameters optimized for all models
- [ ] Feature count reduced from ~100 to ~40-50
- [ ] CV accuracy variance < 3% across folds
- [ ] Final accuracy: 64-66%
- [ ] All configs saved for reproducibility

## OUTPUT FORMAT

Provide complete, runnable code for each task with:
1. Full implementations
2. CLI interfaces where appropriate
3. Visualization outputs
4. Config file templates

Start with Task 4.1 (Optuna Optimization).
```

---

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 5: PRODUCTION API + UI PREPARATION (Week 7+)
# ═══════════════════════════════════════════════════════════════════════════════

## Cursor Prompt — Phase 5

```
You are helping me optimize an NFL prediction model. This is Phase 5 of a 5-phase optimization plan.

## CONTEXT

Completed phases:
- Phase 1: Ensemble + calibration (~61%)
- Phase 2: Injury + odds integration (~62%)
- Phase 3: Weather + validation (~63%)
- Phase 4: Hyperparameter tuning + feature selection (~65%)

Phase 5 goals:
- Create production-ready API
- Prepare infrastructure for UI integration
- Implement monitoring and logging
- Set up automated weekly predictions

Final model specs:
- Accuracy: 65%+ (validated on 300+ games)
- ROI: 15%+ (on 70%+ confidence picks)
- Calibration: Monotonic across all tiers
- Features: ~45 optimized features
- Models: Stacking ensemble (XGBoost + FT-Transformer + LR)

## PHASE 5 TASKS

### Task 5.1: Create Production API with FastAPI

Create `api/main.py`:

```python
"""
NFL Prediction API

Production-ready API for serving predictions.

Features:
- GET /predictions/current - Current week predictions
- GET /predictions/game/{game_id} - Single game prediction
- GET /performance - Model performance stats
- POST /predictions/batch - Batch predictions (admin)
- GET /health - Health check

Authentication: API key required for all endpoints
Rate limiting: 100 requests/minute per key
"""

from fastapi import FastAPI, Depends, HTTPException, Security
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import pandas as pd

app = FastAPI(
    title="NFL Prediction API",
    description="AI-powered NFL game predictions with 65%+ accuracy",
    version="1.0.0"
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Key authentication
API_KEY_HEADER = APIKeyHeader(name="X-API-Key")

async def verify_api_key(api_key: str = Security(API_KEY_HEADER)):
    # In production, check against database
    if api_key != "your-secret-key":
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key


# Pydantic models for request/response
class GamePrediction(BaseModel):
    game_id: str
    home_team: str
    away_team: str
    game_time: datetime
    
    # Prediction
    predicted_winner: str
    home_win_probability: float
    confidence_tier: str  # "high", "medium", "low"
    
    # Market comparison
    market_spread: float
    market_implied_prob: float
    model_edge: float
    
    # Key factors
    top_factors: List[dict]  # [{name, impact, direction}]


class PerformanceStats(BaseModel):
    season: int
    total_predictions: int
    accuracy: float
    roi: float
    accuracy_by_tier: dict
    recent_streak: int


class PredictionResponse(BaseModel):
    week: int
    season: int
    generated_at: datetime
    predictions: List[GamePrediction]


# Endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_version": "1.0.0",
        "last_updated": datetime.now().isoformat(),
    }


@app.get("/predictions/current", response_model=PredictionResponse)
async def get_current_predictions(api_key: str = Depends(verify_api_key)):
    """
    Get predictions for current NFL week.
    
    Returns predictions for all games in the current week,
    sorted by confidence level (highest first).
    """
    # Load predictions from cache/database
    predictions = load_current_predictions()
    
    return PredictionResponse(
        week=get_current_week(),
        season=2025,
        generated_at=datetime.now(),
        predictions=predictions
    )


@app.get("/predictions/game/{game_id}", response_model=GamePrediction)
async def get_game_prediction(game_id: str, api_key: str = Depends(verify_api_key)):
    """
    Get detailed prediction for a single game.
    
    Includes:
    - Win probability
    - Confidence tier
    - Model edge vs market
    - Top factors driving prediction
    """
    prediction = load_game_prediction(game_id)
    
    if not prediction:
        raise HTTPException(status_code=404, detail="Game not found")
    
    return prediction


@app.get("/performance", response_model=PerformanceStats)
async def get_performance(
    season: Optional[int] = None,
    api_key: str = Depends(verify_api_key)
):
    """
    Get model performance statistics.
    
    Returns accuracy, ROI, and breakdown by confidence tier.
    """
    stats = calculate_performance_stats(season or 2025)
    return stats


@app.get("/predictions/week/{week}", response_model=PredictionResponse)
async def get_week_predictions(
    week: int,
    season: int = 2025,
    api_key: str = Depends(verify_api_key)
):
    """Get predictions for a specific week."""
    predictions = load_week_predictions(season, week)
    
    return PredictionResponse(
        week=week,
        season=season,
        generated_at=datetime.now(),
        predictions=predictions
    )


# Admin endpoints (require admin API key)
@app.post("/admin/generate-predictions")
async def generate_predictions(api_key: str = Depends(verify_api_key)):
    """
    Trigger prediction generation for current week.
    
    This is called by the weekly cron job or manually by admin.
    """
    # Run prediction pipeline
    # Save to database
    # Return summary
    pass


# Helper functions
def load_current_predictions() -> List[GamePrediction]:
    """Load predictions from database/cache."""
    pass

def load_game_prediction(game_id: str) -> GamePrediction:
    """Load single game prediction."""
    pass

def get_current_week() -> int:
    """Determine current NFL week."""
    pass

def calculate_performance_stats(season: int) -> PerformanceStats:
    """Calculate performance metrics."""
    pass
```

**Additional API files:**

Create `api/models.py` (Pydantic models)
Create `api/database.py` (Database connection)
Create `api/cache.py` (Redis caching)
Create `api/auth.py` (Authentication logic)

Deliverables:
- Complete FastAPI application
- Pydantic models for all endpoints
- Authentication middleware
- Rate limiting
- OpenAPI documentation (auto-generated)

### Task 5.2: Set Up Database Layer

Create `api/database.py`:

```python
"""
Database layer for NFL Predictions API.

Uses PostgreSQL for:
- Predictions storage
- User management
- Performance tracking
- Audit logs

Options:
- Supabase (recommended - free tier available)
- Railway PostgreSQL
- Local PostgreSQL for development
"""

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./predictions.db")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Prediction(Base):
    """Store each prediction made by the model."""
    __tablename__ = "predictions"
    
    id = Column(Integer, primary_key=True, index=True)
    game_id = Column(String, unique=True, index=True)
    season = Column(Integer, index=True)
    week = Column(Integer, index=True)
    
    # Game info
    home_team = Column(String)
    away_team = Column(String)
    game_time = Column(DateTime)
    
    # Prediction
    home_win_prob = Column(Float)
    predicted_winner = Column(String)
    confidence_tier = Column(String)
    
    # Market data
    market_spread = Column(Float)
    market_implied_prob = Column(Float)
    model_edge = Column(Float)
    
    # Features used (for debugging)
    features_json = Column(JSON)
    
    # Outcome (filled after game)
    actual_winner = Column(String, nullable=True)
    was_correct = Column(Boolean, nullable=True)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    model_version = Column(String)


class PerformanceLog(Base):
    """Track model performance over time."""
    __tablename__ = "performance_logs"
    
    id = Column(Integer, primary_key=True)
    date = Column(DateTime, index=True)
    season = Column(Integer)
    week = Column(Integer)
    
    total_predictions = Column(Integer)
    correct_predictions = Column(Integer)
    accuracy = Column(Float)
    
    # By confidence tier
    high_conf_total = Column(Integer)
    high_conf_correct = Column(Integer)
    medium_conf_total = Column(Integer)
    medium_conf_correct = Column(Integer)
    low_conf_total = Column(Integer)
    low_conf_correct = Column(Integer)
    
    # ROI (assuming flat betting)
    roi = Column(Float)
    units_won = Column(Float)


class APIKey(Base):
    """API key management."""
    __tablename__ = "api_keys"
    
    id = Column(Integer, primary_key=True)
    key = Column(String, unique=True, index=True)
    user_email = Column(String)
    tier = Column(String)  # "free", "pro", "enterprise"
    
    requests_today = Column(Integer, default=0)
    requests_limit = Column(Integer, default=100)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    last_used = Column(DateTime)
    is_active = Column(Boolean, default=True)


# Database operations
def get_db():
    """Dependency for FastAPI."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def save_predictions(db, predictions: list):
    """Save batch of predictions to database."""
    for pred in predictions:
        db_pred = Prediction(**pred)
        db.add(db_pred)
    db.commit()


def update_outcomes(db, results: dict):
    """Update predictions with actual outcomes."""
    for game_id, winner in results.items():
        pred = db.query(Prediction).filter(Prediction.game_id == game_id).first()
        if pred:
            pred.actual_winner = winner
            pred.was_correct = (pred.predicted_winner == winner)
    db.commit()


def get_performance_stats(db, season: int, week: int = None):
    """Calculate performance statistics."""
    query = db.query(Prediction).filter(
        Prediction.season == season,
        Prediction.actual_winner.isnot(None)
    )
    
    if week:
        query = query.filter(Prediction.week == week)
    
    predictions = query.all()
    
    total = len(predictions)
    correct = sum(1 for p in predictions if p.was_correct)
    
    return {
        'total': total,
        'correct': correct,
        'accuracy': correct / total if total > 0 else 0,
    }
```

Deliverables:
- SQLAlchemy models
- Migration scripts (Alembic)
- Database operations
- Connection pooling setup

### Task 5.3: Create Weekly Prediction Pipeline

Create `scripts/weekly_pipeline.py`:

```python
"""
Weekly Prediction Pipeline

Automated script to run every Tuesday/Wednesday:
1. Fetch latest data (schedule, odds, injuries, weather)
2. Generate features for current week games
3. Run model predictions
4. Save to database
5. Send notifications (optional)

Schedule with cron or cloud scheduler:
    0 8 * * 3 python scripts/weekly_pipeline.py  # Wednesday 8am
"""

import logging
from datetime import datetime
from typing import Dict, List
import pandas as pd

from ingestion.nfl.schedule import ScheduleIngestion
from ingestion.nfl.odds_api import OddsAPIClient
from ingestion.nfl.injuries import InjuryIngestion
from ingestion.nfl.weather import WeatherIngestion
from features.nfl.generate_features import generate_game_features
from models.inference.predictor import EnsemblePredictor
from api.database import SessionLocal, save_predictions

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WeeklyPipeline:
    """
    Orchestrates weekly prediction generation.
    """
    
    def __init__(self):
        self.schedule = ScheduleIngestion()
        self.odds = OddsAPIClient()
        self.injuries = InjuryIngestion()
        self.weather = WeatherIngestion()
        self.predictor = EnsemblePredictor.load('models/artifacts/nfl_ensemble')
    
    def run(self, week: int = None, season: int = 2025) -> List[Dict]:
        """
        Run full prediction pipeline for a week.
        
        Args:
            week: NFL week number (auto-detect if None)
            season: NFL season year
        
        Returns:
            List of prediction dictionaries
        """
        logger.info(f"Starting weekly pipeline for {season} week {week}")
        
        # Step 1: Get current week's games
        games = self.schedule.get_week_games(season, week)
        logger.info(f"Found {len(games)} games")
        
        predictions = []
        
        for _, game in games.iterrows():
            try:
                pred = self._predict_game(game)
                predictions.append(pred)
            except Exception as e:
                logger.error(f"Error predicting {game['game_id']}: {e}")
        
        # Step 2: Save to database
        db = SessionLocal()
        try:
            save_predictions(db, predictions)
            logger.info(f"Saved {len(predictions)} predictions")
        finally:
            db.close()
        
        # Step 3: Send notifications (if configured)
        high_conf_preds = [p for p in predictions if p['confidence_tier'] == 'high']
        if high_conf_preds:
            self._send_notifications(high_conf_preds)
        
        return predictions
    
    def _predict_game(self, game: pd.Series) -> Dict:
        """Generate prediction for a single game."""
        game_id = game['game_id']
        home_team = game['home_team']
        away_team = game['away_team']
        game_time = game['gameday']
        
        # Fetch current data
        odds = self.odds.get_game_odds(game_id)
        home_injuries = self.injuries.get_team_injuries(home_team)
        away_injuries = self.injuries.get_team_injuries(away_team)
        weather = self.weather.get_game_weather(home_team, game_time)
        
        # Generate features
        features = generate_game_features(
            game=game,
            odds=odds,
            home_injuries=home_injuries,
            away_injuries=away_injuries,
            weather=weather
        )
        
        # Run prediction
        home_prob = self.predictor.predict_proba(features)
        
        # Determine confidence tier
        confidence = max(home_prob, 1 - home_prob)
        if confidence >= 0.70:
            tier = 'high'
        elif confidence >= 0.60:
            tier = 'medium'
        else:
            tier = 'low'
        
        # Calculate edge vs market
        market_prob = odds.get('implied_home_prob', 0.5)
        edge = home_prob - market_prob
        
        return {
            'game_id': game_id,
            'season': game['season'],
            'week': game['week'],
            'home_team': home_team,
            'away_team': away_team,
            'game_time': game_time,
            'home_win_prob': home_prob,
            'predicted_winner': home_team if home_prob > 0.5 else away_team,
            'confidence_tier': tier,
            'market_spread': odds.get('spread', 0),
            'market_implied_prob': market_prob,
            'model_edge': edge,
            'features_json': features.to_dict(),
            'model_version': '1.0.0',
        }
    
    def _send_notifications(self, predictions: List[Dict]):
        """Send notifications for high-confidence picks."""
        # Implement email/push notifications
        pass


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--week', type=int, default=None)
    parser.add_argument('--season', type=int, default=2025)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    
    pipeline = WeeklyPipeline()
    
    if args.dry_run:
        logger.info("Dry run - not saving to database")
    
    predictions = pipeline.run(week=args.week, season=args.season)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Generated {len(predictions)} predictions")
    
    high = sum(1 for p in predictions if p['confidence_tier'] == 'high')
    med = sum(1 for p in predictions if p['confidence_tier'] == 'medium')
    low = sum(1 for p in predictions if p['confidence_tier'] == 'low')
    
    print(f"High confidence: {high}")
    print(f"Medium confidence: {med}")
    print(f"Low confidence: {low}")
    print(f"{'='*50}\n")


if __name__ == '__main__':
    main()
```

Deliverables:
- Complete weekly pipeline script
- Cron job configuration
- Error handling and logging
- Notification integration (optional)

### Task 5.4: Create Deployment Configuration

Create deployment configs:

**`docker/Dockerfile`:**
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Run API
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**`docker/docker-compose.yml`:**
```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:pass@db:5432/nfl
      - ODDS_API_KEY=${ODDS_API_KEY}
    depends_on:
      - db
      - redis
  
  db:
    image: postgres:15
    environment:
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=pass
      - POSTGRES_DB=nfl
    volumes:
      - postgres_data:/var/lib/postgresql/data
  
  redis:
    image: redis:7
    ports:
      - "6379:6379"

volumes:
  postgres_data:
```

**`railway.toml`** (for Railway deployment):
```toml
[build]
builder = "dockerfile"
dockerfilePath = "docker/Dockerfile"

[deploy]
healthcheckPath = "/health"
healthcheckTimeout = 100
restartPolicyType = "on_failure"
```

Deliverables:
- Docker configuration
- docker-compose for local development
- Railway/Render deployment config
- Environment variable documentation

### Task 5.5: Create API Client SDK

Create `sdk/python/nfl_predictions.py`:

```python
"""
Python SDK for NFL Predictions API

Usage:
    from nfl_predictions import NFLPredictionsClient
    
    client = NFLPredictionsClient(api_key="your-key")
    predictions = client.get_current_predictions()
    
    for pred in predictions:
        if pred.confidence_tier == "high":
            print(f"{pred.predicted_winner} ({pred.home_win_probability:.1%})")
"""

import requests
from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime


@dataclass
class GamePrediction:
    game_id: str
    home_team: str
    away_team: str
    game_time: datetime
    predicted_winner: str
    home_win_probability: float
    confidence_tier: str
    market_spread: float
    model_edge: float


class NFLPredictionsClient:
    """
    Client for NFL Predictions API.
    """
    
    def __init__(self, api_key: str, base_url: str = "https://api.yourservice.com"):
        self.api_key = api_key
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers['X-API-Key'] = api_key
    
    def get_current_predictions(self) -> List[GamePrediction]:
        """Get predictions for current NFL week."""
        response = self.session.get(f"{self.base_url}/predictions/current")
        response.raise_for_status()
        
        data = response.json()
        return [GamePrediction(**p) for p in data['predictions']]
    
    def get_game_prediction(self, game_id: str) -> GamePrediction:
        """Get prediction for a specific game."""
        response = self.session.get(f"{self.base_url}/predictions/game/{game_id}")
        response.raise_for_status()
        
        return GamePrediction(**response.json())
    
    def get_performance(self, season: int = None) -> dict:
        """Get model performance statistics."""
        params = {'season': season} if season else {}
        response = self.session.get(f"{self.base_url}/performance", params=params)
        response.raise_for_status()
        
        return response.json()
    
    def get_high_confidence_picks(self) -> List[GamePrediction]:
        """Get only high-confidence predictions."""
        all_preds = self.get_current_predictions()
        return [p for p in all_preds if p.confidence_tier == 'high']
```

Deliverables:
- Python SDK
- JavaScript/TypeScript SDK (for frontend)
- SDK documentation
- Example usage scripts

### Task 5.6: API Documentation

Create `docs/api.md`:

```markdown
# NFL Predictions API Documentation

## Authentication

All endpoints require an API key passed in the `X-API-Key` header.

```bash
curl -H "X-API-Key: your-key" https://api.yourservice.com/predictions/current
```

## Endpoints

### GET /predictions/current

Get predictions for the current NFL week.

**Response:**
```json
{
  "week": 14,
  "season": 2025,
  "generated_at": "2025-12-03T08:00:00Z",
  "predictions": [
    {
      "game_id": "2025_14_KC_BAL",
      "home_team": "BAL",
      "away_team": "KC",
      "game_time": "2025-12-07T20:20:00Z",
      "predicted_winner": "BAL",
      "home_win_probability": 0.72,
      "confidence_tier": "high",
      "market_spread": -3.5,
      "market_implied_prob": 0.52,
      "model_edge": 0.20,
      "top_factors": [
        {"name": "home_offense_epa", "impact": 0.15, "direction": "positive"},
        {"name": "away_injuries_impact", "impact": 0.12, "direction": "negative"}
      ]
    }
  ]
}
```

### GET /predictions/game/{game_id}

Get detailed prediction for a single game.

### GET /performance

Get model performance statistics.

**Query Parameters:**
- `season` (optional): Filter by season year

**Response:**
```json
{
  "season": 2025,
  "total_predictions": 182,
  "accuracy": 0.648,
  "roi": 0.156,
  "accuracy_by_tier": {
    "high": 0.72,
    "medium": 0.61,
    "low": 0.54
  }
}
```

## Rate Limits

- Free tier: 100 requests/day
- Pro tier: 1000 requests/day
- Enterprise: Unlimited

## Error Codes

- 401: Invalid API key
- 403: Rate limit exceeded
- 404: Resource not found
- 500: Server error
```

Deliverables:
- Complete API documentation
- OpenAPI/Swagger spec
- Example requests for each endpoint
- Error handling guide

## SUCCESS CRITERIA FOR PHASE 5

- [ ] FastAPI application running and tested
- [ ] Database schema deployed (Supabase/Railway)
- [ ] Weekly pipeline automated (cron job)
- [ ] Docker deployment working
- [ ] API documentation complete
- [ ] Python SDK published
- [ ] Health monitoring in place
- [ ] Ready for UI integration

## OUTPUT FORMAT

For each task:
1. Complete, production-ready code
2. Environment variable documentation
3. Deployment instructions
4. Testing commands

Start with Task 5.1 (FastAPI Application).
```

---

# Summary

These 5 prompts cover:

| Phase | Focus | Expected Outcome |
|-------|-------|------------------|
| 1 | Ensemble + Calibration + Rest | 61% accuracy |
| 2 | Injuries + Odds API | 62% accuracy |
| 3 | Weather + 2024 Validation | 63% accuracy |
| 4 | Hyperparameter Tuning + Feature Selection | 65% accuracy |
| 5 | Production API + Deployment | Ship-ready |

Copy each prompt into Cursor when you're ready to start that phase. Each builds on the previous, so complete them in order.
