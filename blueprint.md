# Universal Sports Prediction Pipeline - Architecture Blueprint

## 1. Project Overview

### Purpose

A **market-aware, data-rich, multi-sport prediction engine** designed to identify edges versus betting markets. The system will initially target NFL games, with a clear path to extend to other sports (NBA, MLB, NHL, etc.).

### Core Principles

- **Data Ingestion**: Leverage free/open data sources (nflfastR, nflverse, public APIs, scraped odds archives) to build comprehensive datasets without proprietary data costs.

- **Feature Engineering**: Extract meaningful signals from:
  - Play-by-play data (EPA, success rate, CPOE)
  - Player-level metrics (QB efficiency, positional usage, injury impact)
  - Team-level aggregates (offensive/defensive splits, situational performance)
  - Market signals (closing lines, line movement, market efficiency proxies)

- **Market-Aware Modeling**: Models explicitly incorporate:
  - Opening/closing spreads and totals
  - Moneyline odds
  - Market-derived probabilities
  - Outputs residual edges vs. market expectations (not just raw predictions)

- **Evaluation Focus**: Dual-metric evaluation:
  - **Predictive Quality**: Brier score, log loss, calibration metrics
  - **ROI vs. Closing Line**: Simulated betting performance using closing line value (CLV) as the benchmark

### Design Philosophy

- **Sport-Agnostic Core**: Core pipeline, models, and evaluation logic work across sports
- **Sport-Specific Adapters**: Each sport implements ingestion and feature adapters that map to the core schema
- **Extensibility**: New sports can be added by implementing adapters without modifying core logic
- **Reproducibility**: All data, features, and models are versioned and reproducible

---

## 2. Directory Structure

```
predictionV2/
├── docs/                          # Design docs, feature catalogs, README-level documentation
│   ├── blueprint.md              # This document
│   ├── feature_catalog.md        # Feature definitions and sources
│   ├── model_specs.md            # Model architectures and hyperparameters
│   └── data_sources.md           # Data source documentation and schemas
│
├── config/                        # Configuration files (YAML/JSON)
│   ├── data/                     # Data source configs
│   │   ├── nfl.yaml
│   │   ├── nba.yaml
│   │   └── shared.yaml
│   ├── features/                 # Feature engineering configs
│   │   ├── nfl_features.yaml
│   │   └── shared_features.yaml
│   ├── models/                   # Model configs
│   │   ├── nfl_baseline.yaml
│   │   └── nfl_advanced.yaml
│   └── evaluation/               # Evaluation configs
│       └── backtest_config.yaml
│
├── data/                          # Data storage (organized by sport)
│   ├── nfl/
│   │   ├── raw/                  # Raw scraped/downloaded data
│   │   │   ├── schedules/
│   │   │   ├── play_by_play/
│   │   │   ├── rosters/
│   │   │   ├── injuries/
│   │   │   └── odds/
│   │   ├── staged/               # Cleaned, normalized, but not yet feature-engineered
│   │   │   ├── games.parquet
│   │   │   ├── team_stats.parquet
│   │   │   ├── player_stats.parquet
│   │   │   ├── plays.parquet
│   │   │   └── markets.parquet
│   │   └── processed/            # Feature-engineered datasets ready for modeling
│   │       ├── game_features.parquet
│   │       └── training_sets/
│   ├── nba/                      # Same structure for NBA (future)
│   └── mlb/                      # Same structure for MLB (future)
│
├── ingestion/                     # Data ingestion modules
│   ├── __init__.py
│   ├── base.py                   # Base classes/interfaces for ingesters
│   ├── nfl/
│   │   ├── __init__.py
│   │   ├── schedule.py           # NFL schedule/results scraper
│   │   ├── play_by_play.py       # nflfastR/nflverse play-by-play loader
│   │   ├── rosters.py            # Roster/player data loader
│   │   ├── injuries.py           # Injury report scraper
│   │   └── odds.py               # Historical odds scraper/loader
│   ├── nba/                      # NBA ingestion (future)
│   └── mlb/                      # MLB ingestion (future)
│
├── features/                      # Feature engineering core + sport adapters
│   ├── __init__.py
│   ├── base.py                   # Base feature generator interfaces
│   ├── core/                     # Sport-agnostic feature generators
│   │   ├── __init__.py
│   │   ├── market_features.py    # Market-derived features (spread, total, movement)
│   │   ├── rest_features.py     # Rest/travel features
│   │   └── weather_features.py   # Weather approximations
│   ├── nfl/
│   │   ├── __init__.py
│   │   ├── team_features.py      # Team-level EPA, success rate, etc.
│   │   ├── player_features.py    # QB metrics, positional usage
│   │   ├── injury_features.py    # Injury burden by position group
│   │   └── matchup_features.py   # Offense vs defense matchup features
│   └── nba/                      # NBA feature adapters (future)
│
├── models/                        # Model definitions, training, ensembles
│   ├── __init__.py
│   ├── base.py                   # Base model interfaces
│   ├── architectures/
│   │   ├── logistic_regression.py
│   │   ├── gradient_boosting.py
│   │   └── ensemble.py
│   ├── training/
│   │   ├── trainer.py            # Training orchestration
│   │   └── calibration.py        # Platt scaling, isotonic regression
│   └── registry.py               # Model versioning and registry
│
├── eval/                          # Evaluation, backtesting, ROI analysis
│   ├── __init__.py
│   ├── metrics.py                # Brier, log loss, calibration metrics
│   ├── backtest.py               # Historical backtesting engine
│   ├── roi.py                    # ROI vs closing line analysis
│   ├── diagnostics.py            # Calibration plots, feature importance
│   └── reports.py                # Generate evaluation reports
│
├── orchestration/                 # Pipeline definitions, scheduling, CLI
│   ├── __init__.py
│   ├── pipelines/
│   │   ├── ingest_pipeline.py   # Data ingestion pipeline
│   │   ├── feature_pipeline.py  # Feature generation pipeline
│   │   ├── train_pipeline.py    # Model training pipeline
│   │   └── eval_pipeline.py     # Evaluation pipeline
│   ├── scheduler.py              # Pipeline scheduling (if needed)
│   └── cli.py                    # Command-line interface entry points
│
├── sports/                        # Sport-specific coordination modules
│   ├── __init__.py
│   ├── base.py                   # Base sport interface
│   ├── nfl/
│   │   ├── __init__.py
│   │   ├── coordinator.py        # NFL-specific orchestration
│   │   └── schemas.py            # NFL-specific data schemas
│   ├── nba/                      # NBA coordinator (future)
│   └── mlb/                      # MLB coordinator (future)
│
├── exploration/                   # Notebooks and scratch analysis (non-production)
│   ├── notebooks/
│   │   ├── data_exploration.ipynb
│   │   ├── feature_analysis.ipynb
│   │   └── model_experiments.ipynb
│   └── scripts/                  # One-off analysis scripts
│
├── tests/                         # Unit and integration tests
│   ├── unit/
│   ├── integration/
│   └── fixtures/
│
├── requirements.txt               # Python dependencies
├── setup.py                       # Package setup
└── README.md                      # Project README
```

---

## 3. Core Concepts (Sport-Agnostic)

### Shared Entities

The core pipeline operates on a set of sport-agnostic entities that map to structured data:

#### `Game`
- **Fields**: `game_id`, `sport`, `season`, `week/date`, `home_team`, `away_team`, `home_score`, `away_score`, `venue`, `date_time`
- **Purpose**: Represents a single game/match
- **Storage**: `data/<sport>/staged/games.parquet`

#### `TeamGameStats`
- **Fields**: `game_id`, `team`, `is_home`, `points_for`, `points_against`, `yards_offense`, `yards_defense`, `turnovers`, `penalties`, `time_of_possession`
- **Purpose**: Team-level aggregate statistics per game
- **Storage**: `data/<sport>/staged/team_stats.parquet`

#### `PlayerGameStats`
- **Fields**: `game_id`, `player_id`, `team`, `position`, `stats_dict` (sport-specific)
- **Purpose**: Player-level statistics per game
- **Storage**: `data/<sport>/staged/player_stats.parquet`

#### `PlayByPlay`
- **Fields**: `game_id`, `play_id`, `quarter/period`, `time`, `down`, `distance`, `yard_line`, `play_type`, `outcome`, `epa`, `success` (sport-specific)
- **Purpose**: Play-by-play granular data for advanced metrics
- **Storage**: `data/<sport>/staged/plays.parquet`

#### `MarketSnapshot`
- **Fields**: `game_id`, `timestamp`, `source`, `spread`, `total`, `moneyline_home`, `moneyline_away`, `is_closing_line`
- **Purpose**: Betting market data (opening/closing lines, movement)
- **Storage**: `data/<sport>/staged/markets.parquet`

#### `InjuryReport`
- **Fields**: `game_id`, `team`, `player_id`, `position`, `injury_type`, `status` (questionable/probable/out), `report_date`
- **Purpose**: Injury information affecting game outcomes
- **Storage**: `data/<sport>/staged/injuries.parquet`

### Data Flow

1. **Raw → Staged**: Ingestion modules download/scrape data and normalize to the core schema
2. **Staged → Processed**: Feature engineering transforms staged data into model-ready features
3. **Processed → Models**: Training and evaluation consume processed feature sets

### Core Interfaces

- **`Ingester`**: Base class for sport-specific ingestion modules
- **`FeatureGenerator`**: Base class for feature engineering modules
- **`Model`**: Base class for prediction models
- **`Evaluator`**: Base class for evaluation metrics and backtesting

---

## 4. NFL Adapter (First-Class, Template for Others)

### NFL Ingestion Scope

#### Schedule/Results
- **Source**: NFL.com, ESPN, or nflverse
- **Data**: Regular season and playoff schedules, final scores, game metadata
- **Module**: `ingestion/nfl/schedule.py`

#### Play-by-Play
- **Source**: nflfastR/nflverse (R package, Python wrapper, or direct CSV downloads)
- **Data**: Every play with EPA, success rate, CPOE, down/distance, field position
- **Key Metrics**: 
  - EPA/play (offense and defense)
  - Success rate (offense and defense)
  - Pass vs run splits
  - Situational splits (red zone, third down, etc.)
- **Module**: `ingestion/nfl/play_by_play.py`

#### Team/Player Stats
- **Source**: nflverse, Pro Football Reference (scraped), or NFL.com
- **Data**: 
  - Team: Points, yards, turnovers, penalties, time of possession
  - Player: Passing yards/TDs, rushing yards/TDs, receiving yards/TDs, targets/carries
- **Module**: `ingestion/nfl/rosters.py`

#### Injury Reports
- **Source**: Free injury report sites (NFL.com official reports, or scraped from news sites)
- **Data**: Player name, position, injury type, status (questionable/probable/out), report date
- **Module**: `ingestion/nfl/injuries.py`

#### Odds
- **Source**: Historical odds archives (free sources like odds-api.com historical, or scraped archives)
- **Data**: Opening/closing spreads, totals, moneylines with timestamps
- **Module**: `ingestion/nfl/odds.py`

### NFL Feature Adapter Outputs

#### Team Strength Metrics
- **EPA-based**: Offensive EPA/play, defensive EPA/play (overall, pass, run)
- **Success Rate**: Offensive/defensive success rate splits
- **Turnovers**: Turnover rate (giveaways/takeaways per game)
- **Pressure**: Pass rush pressure rate (if available)
- **Red Zone**: Red zone efficiency (TD rate, EPA in red zone)
- **Module**: `features/nfl/team_features.py`

#### Player-Level Features
- **QB Metrics**: EPA/play, CPOE, air yards, completion rate, sack rate
- **OL Proxies**: Sack rate allowed, pressure rate allowed, run blocking grades (if available)
- **DL Proxies**: Sacks, pressures, run stop rate
- **Skill Player Usage**: Target share, carry share, snap counts
- **Module**: `features/nfl/player_features.py`

#### Market Features
- **Closing Line**: Closing spread, total, moneyline-derived probabilities
- **Line Movement**: Change from opening to closing line
- **Market Efficiency Proxies**: Historical market accuracy, sharp vs public indicators (if available)
- **Module**: `features/core/market_features.py`

#### Context Features
- **Rest**: Days since last game, bye week indicator, short week indicator
- **Travel**: Distance traveled, time zone changes
- **Home Field**: Home/away indicator, dome/outdoor, altitude
- **Weather**: Wind speed, temperature, precipitation (approximated from historical weather data)
- **Module**: `features/core/rest_features.py`, `features/core/weather_features.py`

#### Injury Burden Metrics
- **Positional Impact**: 
  - QB injury status (binary or severity score)
  - OL injury count/severity
  - WR/CB injury count/severity
  - RB/DL injury count/severity
- **Module**: `features/nfl/injury_features.py`

#### Matchup Features
- **Offense vs Defense**: 
  - Offensive pass EPA vs opponent pass defense EPA
  - Offensive run EPA vs opponent run defense EPA
  - Offensive red zone efficiency vs opponent red zone defense
- **Module**: `features/nfl/matchup_features.py`

### Extension Pattern for Other Sports

Other sports (NBA, MLB, etc.) will:
1. Implement `Ingester` subclasses in `ingestion/<sport>/`
2. Implement `FeatureGenerator` subclasses in `features/<sport>/`
3. Map sport-specific metrics to the core schema
4. Reuse core models, evaluation, and orchestration logic

---

## 5. Implementation Phases (NFL-first)

### Phase 0 – Skeleton & Infrastructure

#### Objectives
- Establish directory structure (empty directories, placeholder files)
- Define configuration conventions (YAML/JSON schemas)
- Define core data schemas and game ID conventions
- Set up Python package structure and dependencies

#### Deliverables
- Complete directory tree as specified in Section 2
- `config/` templates for data, features, models, evaluation
- Core schema definitions (Pydantic models or JSON schemas)
- Game ID convention: `<sport>_<season>_<week>_<away_team>_<home_team>` (e.g., `nfl_2023_01_KC_DET`)
- `requirements.txt` with initial dependencies
- `README.md` with project overview

#### Exit Criteria
- Directory structure in place
- Config schemas documented
- Core entity schemas defined
- No code implementation yet (planning only)

---

### Phase 1 – Baseline NFL Model (Team-Level, Market-Aware)

#### Objectives
- Build end-to-end pipeline from data ingestion to model evaluation
- Establish baseline predictive performance
- Demonstrate market-aware modeling and ROI evaluation

#### Data Sources
- **Schedule/Results**: NFL.com or nflverse (2018-2023 seasons minimum)
- **Team Stats**: Basic aggregates (points, yards, turnovers) from nflverse or Pro Football Reference
- **Odds**: Historical opening/closing lines from free archives (2018-2023)

#### Features
- **Team Form**:
  - Win rate (last 4, 8, 16 games)
  - Point differential (last 4, 8, 16 games)
  - Points for/against (rolling averages)
  - Turnover differential
- **Market Features**:
  - Closing spread
  - Closing total
  - Moneyline-derived win probability
  - Line movement (opening to closing)

#### Models
- **Ensemble**:
  - Logistic regression (win probability)
  - Gradient boosting (XGBoost/LightGBM) for:
    - Win probability
    - Margin vs spread residual
- **Outputs**:
  - Predicted win probability (home team)
  - Predicted margin
  - Predicted total
  - Edge vs market (predicted prob - market prob)

#### Evaluation Scope
- **Historical Backtest**: 2018-2023 seasons (train on 2018-2021, validate on 2022, test on 2023)
- **Metrics**:
  - Accuracy (win prediction)
  - Brier score
  - Log loss
  - MAE vs spread
  - Calibration plots
- **ROI Simulation**:
  - Simple betting rules (bet when edge > threshold)
  - ROI vs closing line
  - Kelly criterion sizing (optional)

#### Exit Criteria
- Working E2E pipeline for NFL
- Calibrated probabilities (Brier < 0.25, log loss < 0.7)
- Positive ROI vs closing line on test set (or clear understanding of why not)
- Documentation of baseline performance

---

### Phase 2 – Advanced Metrics & Player-Level Structure

#### Objectives
- Integrate play-by-play data and advanced metrics (EPA, success rate)
- Add player-level features and injury impact
- Improve predictive performance over Phase 1 baseline

#### Data Sources
- **Play-by-Play**: nflfastR/nflverse (2018-2023)
- **Player Stats**: nflverse rosters and player stats
- **Injuries**: Scraped injury reports (2020-2023 minimum, if available earlier)

#### Features
- **EPA/Success Rate Splits**:
  - Offensive EPA/play (overall, pass, run)
  - Defensive EPA/play (overall, pass, run)
  - Success rate splits
  - Situational splits (red zone, third down, two-minute)
- **QB Metrics**:
  - EPA/play (passing)
  - CPOE (completion percentage over expected)
  - Air yards per attempt
  - Sack rate
- **OL/DL Proxies**:
  - Sack rate allowed (OL)
  - Pressure rate (DL)
  - Run blocking proxies
- **Skill Player Usage**:
  - Target share (WR/TE)
  - Carry share (RB)
  - Snap counts
- **Injury Burden**:
  - QB injury status (binary)
  - OL injury count/severity
  - WR/CB injury count
  - RB/DL injury count

#### Models
- Retrain Phase 1 models with enriched feature set
- Add calibration layer (Platt scaling or isotonic regression)
- Optionally experiment with:
  - Feature importance analysis
  - Feature selection
  - Hyperparameter tuning

#### Evaluation Scope
- **Comparison**: Phase 2 vs Phase 1 on same test set
- **Metrics**:
  - Predictive metrics (Brier, log loss, accuracy)
  - Calibration improvement
  - ROI vs closing line
- **Feature Analysis**:
  - Feature importance rankings
  - Ablation studies (which features matter most)

#### Exit Criteria
- Demonstrated improvement over Phase 1:
  - Lower Brier score and log loss
  - Better calibration
  - Higher ROI vs closing line (or explanation of trade-offs)
- Feature importance documented
- Player-level and injury features validated as useful

---

### Phase 3 – Context, Matchups, and Robust Market Edge

#### Objectives
- Add contextual features (rest, travel, weather)
- Implement matchup-specific features
- Improve robustness across different game contexts
- Optimize for market edge detection

#### Features
- **Rest Context**:
  - Days since last game
  - Bye week indicator
  - Short week indicator (Thursday games)
  - Rest advantage (home rest days - away rest days)
- **Travel Context**:
  - Distance traveled
  - Time zone changes
  - East coast team playing west coast (and vice versa)
- **Weather Context**:
  - Wind speed (approximated from historical weather)
  - Temperature
  - Precipitation flag
  - Dome vs outdoor
- **Matchup Features**:
  - Offensive pass EPA vs opponent pass defense EPA
  - Offensive run EPA vs opponent run defense EPA
  - Offensive red zone efficiency vs opponent red zone defense
  - QB EPA vs opponent pass defense EPA
- **Market Context**:
  - Historical market accuracy for similar games
  - Line range indicators (big favorite vs small dog)

#### Models
- Retrain with full feature set
- Optionally segment models by context:
  - Indoor vs outdoor
  - High wind vs low wind
  - Big favorite vs small favorite
- Experiment with multi-output models:
  - Joint prediction of win prob + margin + total
- Advanced calibration:
  - Context-aware calibration
  - Time-decay weighting (recent games more important)

#### Evaluation Scope
- **Segmented Performance**:
  - Performance by weather conditions
  - Performance by rest advantage
  - Performance by line ranges
- **Robustness Checks**:
  - Stability across seasons
  - Performance on edge cases (playoffs, primetime games)
- **ROI Analysis**:
  - ROI by context segment
  - Optimal betting thresholds per context

#### Exit Criteria
- Stable, context-aware ROI improvements
- Well-understood limitations (where model performs well vs poorly)
- Robust performance across different game contexts
- Documentation of context-specific insights

---

### Phase 4 – Multi-Sport Generalization

#### Objectives
- Extend pipeline to at least one non-NFL sport (NBA recommended first)
- Validate sport-agnostic core design
- Demonstrate reusability of models and evaluation logic

#### Sport Adapter Implementation (NBA Example)
- **Ingestion** (`ingestion/nba/`):
  - Schedule/results scraper
  - Play-by-play loader (nba_api or Basketball Reference)
  - Player stats loader
  - Odds loader
- **Features** (`features/nba/`):
  - Team features: Offensive/defensive rating, pace, efficiency splits
  - Player features: Usage rate, PER, on/off court metrics
  - Matchup features: Offensive rating vs opponent defensive rating
- **Schemas** (`sports/nba/schemas.py`):
  - Map NBA entities to core schema

#### Core Reuse
- Reuse `models/` architectures (with NBA-specific configs)
- Reuse `eval/` backtesting and ROI logic
- Reuse `orchestration/` pipeline structure

#### Evaluation Scope
- NBA backtest over multiple seasons
- Compare NFL vs NBA model performance
- Validate that core abstractions work across sports

#### Exit Criteria
- At least one non-NFL sport running through the same pipeline
- Working backtest for the new sport
- Documentation of sport-specific adaptations
- Validation that core design is truly sport-agnostic

---

## 6. Operational & Monitoring Plan

### Data Freshness Policies

#### In-Season
- **Ingestion Frequency**: 
  - Daily during season (schedule updates, injury reports, odds)
  - Weekly for play-by-play (after games complete)
- **Staleness Thresholds**:
  - Schedule data: Alert if > 24 hours old during season
  - Injury reports: Alert if > 12 hours old on game days
  - Odds: Alert if > 6 hours old on game days
- **Failure Handling**:
  - Retry logic with exponential backoff
  - Alert on persistent failures
  - Fallback to cached data if available

#### Off-Season
- **Historical Data**: Batch updates for previous seasons (quarterly or as needed)
- **Roster Updates**: Weekly during off-season for trades/drafts

### Model Lifecycle

#### Retraining Cadence
- **In-Season**: 
  - Weekly retraining (after each week's games)
  - Or bi-weekly if computational cost is high
- **Off-Season**:
  - Full retrain before season start
  - Periodic retrains as new data becomes available

#### Versioning
- **Model Registry**: Track model versions with:
  - Training date
  - Feature set version
  - Hyperparameters
  - Performance metrics
- **Promotion**: 
  - Validate new model vs current model on holdout set
  - Promote if improvement exceeds threshold
  - Keep previous version as fallback

### Monitoring

#### Calibration Drift
- **Tracking**: 
  - Weekly calibration metrics (Brier score, log loss)
  - Calibration plots over time
- **Alerts**: 
  - Alert if calibration degrades beyond threshold (e.g., Brier > 0.30)
  - Alert if calibration drift exceeds historical variance

#### Market Edge Tracking
- **Metrics**:
  - Weekly ROI vs closing line
  - Edge distribution (how often we find edges)
  - Average edge size
- **Alerts**:
  - Alert if ROI drops below threshold for N consecutive weeks
  - Alert if edge detection rate drops significantly

#### Performance Degradation
- **Checks**:
  - Compare recent predictions vs historical performance
  - Track feature importance shifts
  - Monitor prediction confidence distributions
- **Alerts**:
  - Alert if accuracy drops > 5% vs baseline
  - Alert if feature importance shifts dramatically

### Operational Tools (Conceptual)

- **Dashboard**: Real-time monitoring of data freshness, model performance, ROI
- **Alerting**: Email/Slack alerts for thresholds exceeded
- **Logging**: Structured logging for all pipeline steps
- **Audit Trail**: Track all predictions, model versions, and data versions used

---

## 7. Scope & Next Steps

### Document Scope

This blueprint is **architecture and planning only**. It defines:
- System structure and organization
- Data flow and entity relationships
- Feature engineering approach
- Model strategy
- Evaluation methodology
- Implementation phases

**No code implementation** is included in this document. Future tasks will:
- Implement ingestion modules per sport (`ingestion/nfl/`, etc.)
- Implement feature generators per sport (`features/nfl/`, etc.)
- Implement model training, calibration, and evaluation (`models/`, `eval/`)
- Build orchestration pipelines (`orchestration/`)
- Create configuration files (`config/`)

### Next Cursor Task Ideas

The following tasks can be turned into separate Cursor prompts for implementation:

1. **Phase 0 Setup**
   - Create directory structure
   - Set up Python package with `setup.py` and `requirements.txt`
   - Define core schemas (Pydantic models for `Game`, `TeamGameStats`, etc.)
   - Create config file templates (YAML schemas)

2. **Phase 1 - NFL Schedule Ingestion**
   - Implement `ingestion/nfl/schedule.py` to fetch NFL schedules/results
   - Normalize to core `Game` schema
   - Store in `data/nfl/staged/games.parquet`

3. **Phase 1 - NFL Team Stats Ingestion**
   - Implement `ingestion/nfl/rosters.py` to fetch team-level stats
   - Normalize to `TeamGameStats` schema
   - Store in `data/nfl/staged/team_stats.parquet`

4. **Phase 1 - NFL Odds Ingestion**
   - Implement `ingestion/nfl/odds.py` to fetch historical odds
   - Normalize to `MarketSnapshot` schema
   - Store in `data/nfl/staged/markets.parquet`

5. **Phase 1 - Baseline Feature Engineering**
   - Implement `features/nfl/team_features.py` for team form features
   - Implement `features/core/market_features.py` for market features
   - Create feature generation pipeline

6. **Phase 1 - Baseline Model Training**
   - Implement logistic regression model in `models/architectures/logistic_regression.py`
   - Implement gradient boosting model in `models/architectures/gradient_boosting.py`
   - Implement ensemble in `models/architectures/ensemble.py`
   - Create training pipeline

7. **Phase 1 - Evaluation Framework**
   - Implement `eval/metrics.py` (Brier, log loss, calibration)
   - Implement `eval/backtest.py` for historical backtesting
   - Implement `eval/roi.py` for ROI vs closing line analysis
   - Create evaluation reports

8. **Phase 2 - Play-by-Play Ingestion**
   - Implement `ingestion/nfl/play_by_play.py` using nflfastR/nflverse
   - Extract EPA, success rate, CPOE metrics
   - Store in `data/nfl/staged/plays.parquet`

9. **Phase 2 - Advanced Feature Engineering**
   - Implement EPA/success rate feature generators
   - Implement QB metrics features
   - Implement injury burden features
   - Extend feature pipeline

10. **Phase 3 - Context Features**
    - Implement rest/travel features (`features/core/rest_features.py`)
    - Implement weather approximations (`features/core/weather_features.py`)
    - Implement matchup features (`features/nfl/matchup_features.py`)

11. **Phase 4 - NBA Adapter**
    - Implement NBA ingestion modules
    - Implement NBA feature adapters
    - Validate multi-sport generalization

Each task should be self-contained and implementable independently, following the architecture defined in this blueprint.

