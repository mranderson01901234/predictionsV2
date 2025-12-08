# Predictr Codebase Comprehensive Audit Report

**Date**: 2025-01-XX  
**Auditor**: AI Assistant  
**Codebase Version**: Current (as of audit date)  
**Scope**: Complete codebase exploration and documentation

---

## Executive Summary

### Tech Stack Overview

**Backend**: Python 3.10+ (ML/Data Pipeline)
- **Framework**: Custom orchestration (no web framework)
- **ML Stack**: scikit-learn, XGBoost, PyTorch (for FT-Transformer/TabNet)
- **Data Processing**: pandas, pyarrow (Parquet)
- **Testing**: pytest

**Frontend**: Next.js 16 (React 19)
- **Framework**: Next.js 16.0.7 with App Router
- **Language**: TypeScript (strict mode)
- **Styling**: Tailwind CSS v4 with custom glassmorphism design system
- **UI Components**: Radix UI (shadcn/ui style)
- **Animations**: Framer Motion v12
- **Charts**: Recharts v3
- **Build Tool**: Next.js built-in (Turbopack/Webpack)

**Architecture**: Monorepo (Python backend + Next.js frontend)

### Key Strengths

1. **Well-Structured ML Pipeline**: Clear separation of ingestion → features → models → evaluation
2. **Comprehensive Feature Engineering**: Multiple feature sets (baseline, EPA, QB, rolling windows)
3. **Model Ensemble Architecture**: Stacking ensemble with multiple base models
4. **Market-Aware Design**: Explicit incorporation of betting market data
5. **Strong Documentation**: Extensive docs/ directory with implementation summaries
6. **Modern Frontend**: Clean React architecture with TypeScript

### Critical Gaps

1. **No Backend API**: Frontend uses mock data only - no connection to Python backend
2. **No Real-Time Updates**: No WebSocket/SSE implementation for live game data
3. **No AI/LLM Integration**: "AI Insights" are hardcoded template strings, not actual AI
4. **No Authentication**: No user management or protected routes
5. **No Deployment Configuration**: No Docker, CI/CD, or production deployment setup
6. **Limited Testing**: Frontend has no tests; backend tests exist but coverage unknown

### Recommended Priorities

1. **HIGH**: Build REST API to connect frontend to Python backend
2. **HIGH**: Implement real-time data updates (WebSocket or polling)
3. **MEDIUM**: Add actual AI/LLM integration for insights generation
4. **MEDIUM**: Set up CI/CD pipeline and deployment configuration
5. **LOW**: Add frontend testing framework

---

## Phase 1: Repository Structure

### Project Type
**Monorepo** with separate Python backend and Next.js frontend

### Directory Structure

```
predictionV2/
├── artifacts/              # Trained models and predictions
│   ├── models/            # Saved model artifacts (.pkl files)
│   └── predictions/       # Historical prediction outputs
├── config/                # YAML configuration files
│   ├── credentials.yaml   # API keys (gitignored)
│   ├── data/              # Data source configs
│   ├── models/            # Model hyperparameter configs
│   ├── evaluation/        # Backtest/evaluation configs
│   └── snapshots/         # Frozen metric snapshots
├── data/                  # Data storage
│   ├── cache/             # Cached API responses
│   ├── nfl/               # NFL-specific data
│   │   ├── raw/          # Raw ingested data
│   │   ├── staged/       # Cleaned/normalized data
│   │   └── processed/   # Feature-engineered data
│   └── odds_cache/       # Cached odds data
├── docs/                  # Documentation (extensive)
│   ├── reports/          # Evaluation reports
│   └── [many .md files]  # Implementation summaries
├── eval/                  # Evaluation modules
│   ├── backtest.py       # ROI calculation
│   ├── metrics.py        # Accuracy, Brier, log loss
│   └── reports.py        # Report generation
├── features/              # Feature engineering
│   ├── core/             # Sport-agnostic features
│   ├── nfl/              # NFL-specific features
│   └── registry.py       # Feature registry
├── footballdb_scraper/    # Web scraper for FootballDB
├── ingestion/             # Data ingestion modules
│   └── nfl/              # NFL ingestion scripts
├── logs/                  # Logs and validation outputs
├── models/                # Model architectures
│   ├── architectures/    # Model implementations
│   ├── training/         # Training pipeline
│   └── calibration/      # Probability calibration
├── nfl_scraper/          # NFL.com scraper
├── ngs_api_scraper/      # Next Gen Stats API scraper
├── orchestration/        # Pipeline orchestration
│   └── pipelines/       # End-to-end pipelines
├── scripts/              # Utility scripts
├── tests/                # Backend test suite (pytest)
├── venv/                 # Python virtual environment
└── web/                  # Next.js frontend
    ├── src/
    │   ├── app/         # Next.js App Router pages
    │   ├── components/  # React components
    │   └── lib/         # Utilities and mock data
    ├── public/          # Static assets
    └── package.json     # Frontend dependencies
```

### Entry Points

**Backend**:
- `python -m ingestion.nfl.run_phase1a` - Phase 1A ingestion
- `python -m orchestration.pipelines.phase1c_pipeline` - Phase 1C training
- `python -m models.training.trainer` - Model training CLI
- `python scripts/simulate_real_world_prediction.py` - Generate predictions

**Frontend**:
- `web/src/app/page.tsx` - Root (redirects to /games)
- `web/src/app/games/page.tsx` - Main live dashboard
- `npm run dev` - Development server

### Configuration Files

- `pyproject.toml` - Python project config (pytest, black, isort)
- `requirements.txt` - Python dependencies
- `Makefile` - Build automation
- `web/package.json` - Frontend dependencies
- `web/tsconfig.json` - TypeScript config (strict mode)
- `web/next.config.ts` - Next.js config (React Compiler enabled)
- `web/components.json` - shadcn/ui config
- `config/credentials.yaml.example` - API keys template

---

## Phase 2: Technology Stack Discovery

### Frontend

**Framework**: Next.js 16.0.7
- **React Version**: 19.2.0
- **App Router**: Yes (using App Router, not Pages Router)
- **React Compiler**: Enabled (`reactCompiler: true`)

**Build Tool**: Next.js built-in (Turbopack in dev, Webpack in production)

**Styling**:
- **Framework**: Tailwind CSS v4 (`@tailwindcss/postcss`)
- **Approach**: Utility-first with custom design system
- **Design System**: Glassmorphism dark theme with neon accents
- **CSS Variables**: Extensive custom property system in `globals.css`
- **Color Palette**: Dark theme with cyan/blue, green, purple neon accents

**State Management**: 
- **React State**: useState, useEffect (no global state library)
- **Server Components**: Next.js Server Components for data fetching
- **Client Components**: "use client" directive for interactivity

**UI Component Library**: 
- **Base**: Radix UI primitives
- **Style**: shadcn/ui (New York style)
- **Icons**: Lucide React v0.556.0

**Animation Library**: Framer Motion v12.23.25

**Data Fetching**: 
- **Current**: Mock data only (`web/src/lib/mock_data.ts`)
- **No API Integration**: No fetch/axios calls to backend
- **No React Query/SWR**: Not implemented

**Real-Time Updates**: 
- **None**: No WebSocket, SSE, or polling implementation
- **Time Updates**: `setInterval` for clock updates only

**TypeScript**: 
- **Enabled**: Yes, strict mode
- **Config**: `tsconfig.json` with strict: true

### Backend

**Language**: Python 3.10+ (tested on 3.10, 3.11, 3.12)

**Framework**: 
- **No Web Framework**: Custom orchestration scripts only
- **No API Server**: No Flask/FastAPI/Django implementation

**ML Stack**:
- **scikit-learn**: >=1.0.0 (Logistic Regression, ensemble)
- **XGBoost**: >=1.5.0 (Gradient Boosting)
- **PyTorch**: >=2.0.0 (FT-Transformer, TabNet, MLP meta-model)
- **pytorch-tabnet**: >=4.0 (optional, falls back if unavailable)

**Data Processing**:
- **pandas**: >=2.0.0
- **pyarrow**: >=12.0.0 (Parquet file support)
- **numpy**: (via dependencies)

**Data Sources**:
- **nfl-data-py**: >=0.3.0 (nflverse Python wrapper)

**Web Scraping**:
- **beautifulsoup4**: >=4.12.0 (injury reports)
- **requests**: >=2.31.0 (HTTP requests)
- **Playwright**: (for nfl_scraper)

**Configuration**:
- **pyyaml**: >=6.0 (YAML config files)

**Testing**:
- **pytest**: >=7.0.0

**Database**: 
- **None**: File-based storage (Parquet files)
- **No ORM**: Direct pandas DataFrame operations

**API Style**: 
- **None**: No REST/GraphQL/tRPC API exists

**Authentication**: 
- **None**: No authentication system

**Hosting/Deployment**: 
- **None**: No deployment configuration found

### External Services

**Sports Data Providers**:
1. **nflverse (nfl-data-py)**: 
   - Schedule and results data
   - Play-by-play data (EPA included)
   - Free/open source

2. **The Odds API** (`the-odds-api.com`):
   - Betting odds data
   - API key required (free tier: 500 requests/month)
   - Config: `config/credentials.yaml`

3. **NFL.com**:
   - Injury reports (scraped)
   - Module: `ingestion/nfl/injuries.py`

4. **Next Gen Stats API**:
   - Advanced player metrics
   - Scraper: `ngs_api_scraper/`

5. **FootballDB**:
   - Historical data
   - Scraper: `footballdb_scraper/`

**Analytics Services**: None

**Payment Providers**: None

**AI/LLM Providers**: **NONE** (see Phase 7)

**Monitoring/Logging Services**: None (Python logging only)

---

## Phase 3: Data Architecture

### Data Sources

#### 1. NFL Schedule and Results
- **Source**: nflverse via `nfl-data-py`
- **Authentication**: None (public data)
- **Endpoints**: `nfl.import_schedules([seasons])`
- **Data Provided**: 
  - Season, week, date, teams, scores
  - Game IDs in format: `nfl_{season}_{week}_{away}_{home}`
- **Frequency**: Historical data (2015-present)
- **Location**: `ingestion/nfl/schedule.py`

#### 2. Betting Odds
- **Source Options**:
  - The Odds API (`the-odds-api.com`)
  - CSV file input (manual)
  - nflverse (if available)
- **Authentication**: API key in `config/credentials.yaml`
- **Endpoints**: 
  - Odds API: `/v4/sports/americanfootball_nfl/odds`
  - CSV: Manual file upload
- **Data Provided**: 
  - Opening/closing spreads and totals
  - Moneylines (if available)
- **Frequency**: Historical + current week
- **Location**: `ingestion/nfl/odds.py`

#### 3. Play-by-Play Data
- **Source**: nflverse/nflfastR
- **Authentication**: None (public)
- **Data Provided**: 
  - EPA (Expected Points Added) per play
  - Success rate
  - Play type (pass/run)
  - Down, distance, yardline
- **Frequency**: Historical (1999-present)
- **Location**: `ingestion/nfl/play_by_play.py`

#### 4. Injury Reports
- **Source**: NFL.com (scraped)
- **Authentication**: None (public website)
- **Method**: Web scraping with BeautifulSoup
- **Data Provided**: 
  - Player name, position, injury type
  - Status (questionable/probable/out)
  - Report date
- **Frequency**: Weekly during season
- **Location**: `ingestion/nfl/injuries.py`

#### 5. Next Gen Stats
- **Source**: NFL.com Next Gen Stats API
- **Authentication**: None (public API)
- **Endpoints**: Various (documented in `nfl_scraper/ENDPOINT_CATALOG.md`)
- **Data Provided**: 
  - Advanced player metrics
  - CPOE (Completion Percentage Over Expected)
  - Air yards, time to throw, etc.
- **Frequency**: Historical + current
- **Location**: `ngs_api_scraper/`

### Data Models / Types

#### Game Data Structure
```typescript
// Frontend (web/src/lib/mock_data.ts)
interface Game {
    game_id: string;           // Format: "nfl_2025_14_IND_JAX"
    season: number;
    week: number;
    date: string;              // ISO 8601
    home_team: string;         // 3-letter code
    away_team: string;
    home_score: number;
    away_score: number;
    status: 'Scheduled' | 'Live' | 'Final';
    quarter?: number;
    time_remaining?: string;
    possession?: 'home' | 'away';
    home_record?: string;
    away_record?: string;
}
```

#### Market Data Structure
```typescript
interface MarketSnapshot {
    game_id: string;
    bookmaker: string;
    spread_home: number;        // From home perspective (negative = home favored)
    total: number;
    spread_home_open?: number;
    total_open?: number;
}
```

#### Prediction Structure
```typescript
interface Prediction {
    game_id: string;
    win_prob_home: number;      // 0-1 probability
    predicted_spread: number;   // From home perspective
    predicted_total: number;
    confidence_score: number;    // 0-100
    edge_spread: number;        // Model - Market
    edge_total: number;
}
```

#### Backend Data Schema (Parquet)
**games.parquet**:
- `game_id`, `season`, `week`, `date`, `home_team`, `away_team`, `home_score`, `away_score`

**markets.parquet**:
- `game_id`, `season`, `week`, `close_spread`, `close_total`, `open_spread`, `open_total`

**game_features_baseline.parquet**:
- All game/market columns plus feature columns (see Feature Registry)

### State Shape

**Frontend State**:
- **No Global State**: Component-level useState only
- **Server Components**: Data fetched server-side (currently mock)
- **Client State**: Selected game ID, UI state (sidebar open/closed)

**Backend State**:
- **File-Based**: All state persisted in Parquet files
- **No Database**: No in-memory state or database
- **Caching**: File-based cache in `data/cache/` and `data/odds_cache/`

### Data Flow

```
External Sources (nflverse, APIs, scrapers)
    ↓
Ingestion Modules (ingestion/nfl/*.py)
    ↓
Raw Data (data/nfl/raw/*.parquet)
    ↓
Staged Data (data/nfl/staged/*.parquet)
    ↓
Feature Engineering (features/nfl/*.py)
    ↓
Processed Features (data/nfl/processed/*.parquet)
    ↓
Model Training (models/training/trainer.py)
    ↓
Trained Models (artifacts/models/*.pkl)
    ↓
Prediction Scripts (scripts/simulate_real_world_prediction.py)
    ↓
[NO CONNECTION TO FRONTEND]
    ↓
Frontend Mock Data (web/src/lib/mock_data.ts)
```

**CRITICAL GAP**: No API layer connecting backend predictions to frontend.

---

## Phase 4: The Prediction Algorithm

### Algorithm Location

**Primary Prediction Script**: `scripts/simulate_real_world_prediction.py`
- **Function**: `simulate_prediction()`
- **Entry Point**: CLI (`python scripts/simulate_real_world_prediction.py --game-id <id>`)

**Model Training**: `models/training/trainer.py`
- **Function**: `run_advanced_training_pipeline()`
- **Entry Point**: CLI (`python -m models.training.trainer`)

**Model Architectures**: `models/architectures/`
- `stacking_ensemble.py` - Main ensemble model
- `logistic_regression.py` - Baseline model
- `gradient_boosting.py` - XGBoost model
- `ft_transformer.py` - Deep learning model
- `tabnet.py` - TabNet model
- `market_baseline.py` - Market-only baseline

### Algorithm Type

**Ensemble Model**: Stacking ensemble with meta-learner
- **Base Models**: Logistic Regression, Gradient Boosting (XGBoost), optionally FT-Transformer/TabNet
- **Meta-Model**: Logistic Regression or MLP (neural network)
- **Training**: Base models trained independently, then meta-model trained on their predictions

### Inputs

**Feature Categories** (from `features/registry.py`):

1. **Baseline Features** (Phase 1):
   - `win_rate_last{N}` - Rolling win rate (N = 4, 8, 16 games)
   - `pdiff_last{N}` - Rolling point differential
   - `points_for_last{N}` - Rolling points scored
   - `points_against_last{N}` - Rolling points allowed
   - `turnover_diff_last{N}` - Rolling turnover differential
   - All features computed separately for home/away teams

2. **EPA Features** (Phase 2):
   - `epa_offensive_epa_per_play` - Offensive EPA per play
   - `epa_offensive_pass_epa` - Passing EPA
   - `epa_offensive_run_epa` - Rushing EPA
   - `epa_defensive_epa_per_play_allowed` - Defensive EPA allowed
   - `epa_offensive_success_rate` - Success rate (EPA > 0)

3. **Rolling EPA Features** (Phase 2B):
   - `roll_epa_off_epa_last{N}` - Rolling offensive EPA (N = 3, 5, 8 games)
   - `roll_epa_def_epa_allowed_last{N}` - Rolling defensive EPA allowed

4. **QB Features** (Phase 2B):
   - `qb_qb_epa_per_dropback` - QB EPA per dropback
   - `qb_qb_cpoe` - Completion percentage over expected
   - `qb_qb_sack_rate` - Sack rate

**Feature Exclusion** (not used as inputs):
- `game_id`, `season`, `week`, `date`, `home_team`, `away_team`
- `home_score`, `away_score`, `home_win` (target variable)
- `close_spread`, `close_total`, `open_spread`, `open_total` (used for evaluation, not features)

**Data Sources for Features**:
- Team form: Historical game results (from `games.parquet`)
- EPA: Play-by-play data (from nflverse, EPA pre-calculated)
- QB metrics: Play-by-play data + player identification
- Market data: Odds (used for evaluation, not features in baseline)

### Processing

#### Step 1: Feature Engineering
**Location**: `features/nfl/*.py`

1. **Team Form Features** (`team_form_features.py`):
   - Compute rolling statistics (win rate, point differential, etc.)
   - Windows: 4, 8, 16 games
   - **CRITICAL**: Excludes current game to prevent leakage

2. **EPA Features** (`epa_features.py`):
   - Group play-by-play data by game and team
   - Calculate mean EPA per play (overall, pass, run)
   - Calculate success rates
   - Compute situational splits (3rd down, red zone, etc.)

3. **Rolling EPA** (`rolling_epa_features.py`):
   - Compute rolling windows of EPA metrics
   - **CRITICAL**: Excludes current game

4. **QB Features** (`qb_features.py`):
   - Identify starting QB for each game
   - Calculate QB-specific metrics (EPA/dropback, CPOE, sack rate)

#### Step 2: Model Training

**Base Models**:
1. **Logistic Regression** (`logistic_regression.py`):
   - sklearn `LogisticRegression`
   - Default: C=1.0, max_iter=1000
   - Output: Home win probability

2. **Gradient Boosting** (`gradient_boosting.py`):
   - XGBoost `XGBClassifier`
   - Default: n_estimators=100, max_depth=3, learning_rate=0.1
   - Output: Home win probability

3. **Stacking Ensemble** (`stacking_ensemble.py`):
   - Combines base model predictions
   - Meta-model: Logistic Regression or MLP
   - **Formula**: 
     ```
     base_preds = [lr.predict_proba(X), gbm.predict_proba(X), ...]
     meta_features = stack(base_preds)  # Optionally includes original features
     ensemble_prob = meta_model.predict_proba(meta_features)
     ```

#### Step 3: Prediction

**Process** (`simulate_real_world_prediction.py`):
1. Load trained ensemble model from `artifacts/models/nfl_stacked_ensemble_v2/ensemble_v1.pkl`
2. Load game features from feature table (`data/nfl/processed/game_features_baseline.parquet`)
3. Extract feature columns (exclude metadata)
4. Run `model.predict_proba(X)` → returns home win probability
5. Convert probability to spread estimate
6. Calculate edge vs market

### Outputs

**Primary Output**: Home team win probability (0-1)

**Derived Outputs**:

1. **Predicted Spread** (`predict_spread()`):
   ```python
   # Formula: spread ≈ -3 * logit(prob)
   logit = log(prob / (1 - prob))
   predicted_spread = -3 * logit
   ```
   **Location**: `scripts/simulate_real_world_prediction.py:281-292`

2. **Confidence Score**:
   ```python
   confidence = max(prob, 1 - prob)  # Distance from 0.5
   ```

3. **Edge Calculation**:
   ```python
   edge_spread = predicted_spread - market_spread
   edge_prob = predicted_prob - market_implied_prob
   ```

4. **Recommended Side**:
   - If `edge_spread > threshold`: Bet home team
   - If `edge_spread < -threshold`: Bet away team
   - Otherwise: No bet

### EPA (Expected Points Added)

**EPA Source**: **Ingested, not calculated**
- EPA comes pre-calculated from nflverse/nflfastR play-by-play data
- The codebase does NOT calculate EPA internally
- EPA is used as a feature input only

**EPA Usage**:
- Aggregated to team-level metrics (offensive EPA/play, defensive EPA allowed)
- Used in rolling windows (last 3, 5, 8 games)
- Split by play type (pass vs run)
- Situational splits (3rd down, red zone, etc.)

**Location**: `features/nfl/epa_features.py` (aggregation only, not calculation)

### Model Performance

**Evaluation Metrics** (`eval/metrics.py`):
- **Accuracy**: `np.mean(y_true == y_pred)`
- **Brier Score**: `np.mean((p_pred - y_true) ** 2)`
- **Log Loss**: Binary cross-entropy
- **Calibration**: Binned calibration buckets

**ROI Calculation** (`eval/backtest.py`):
- Simulates betting strategy vs closing line
- Edge threshold filtering (default: 0.05 = 5%)
- Unit bet size (default: 1.0)
- Returns: ROI, win rate, profit/loss

**Backtesting**:
- Location: `eval/backtest.py`
- Method: Chronological splits (train on past, test on future)
- Validation: 2022 season
- Test: 2023 season
- Holdout: 2025 season (if available)

**Model Validation**:
- Location: `tests/test_backtest.py`, `tests/test_metrics.py`
- Leakage tests: `tests/test_leak_free_test_2025_summary.md`
- Calibration validation: `models/calibration/`

**Claimed Metrics**:
- Documentation references accuracy and ROI metrics
- Specific numbers found in `docs/reports/` directory
- Need to verify against actual evaluation outputs

---

## Phase 5: UI Component Inventory

### Page Structure

| Route | Component File | Purpose | Data Dependencies |
|-------|---------------|---------|-------------------|
| `/` | `web/src/app/page.tsx` | Root redirect | None (redirects to /games) |
| `/games` | `web/src/app/games/page.tsx` | Main live dashboard | `getGames()`, `getFullGameDetails()` |
| `/games/[id]` | `web/src/app/games/[id]/page.tsx` | Game detail page | `getGameDetails(id)` |
| `/predictions` | `web/src/app/predictions/page.tsx` | Predictions overview | Mock predictions |
| `/betting` | `web/src/app/betting/page.tsx` | Betting recommendations | Mock betting data |
| `/confidence` | `web/src/app/confidence/page.tsx` | Confidence pool | Mock confidence data |
| `/performance` | `web/src/app/performance/page.tsx` | Model performance | Mock performance metrics |
| `/qb-rankings` | `web/src/app/qb-rankings/page.tsx` | QB rankings | Mock QB data |
| `/receivers` | `web/src/app/receivers/page.tsx` | Receiving leaders | Mock receiver data |
| `/power-ratings` | `web/src/app/power-ratings/page.tsx` | Team power ratings | Mock team ratings |
| `/pricing` | `web/src/app/pricing/page.tsx` | Pricing page | None |

### Component Hierarchy

**Live Dashboard** (`/games`):
```
LiveDashboard (web/src/components/live/LiveDashboard.tsx)
├── GameStrip (web/src/components/live/GameStrip.tsx)
│   └── GameTile (multiple)
├── HeroGameCard (web/src/components/live/HeroGameCard.tsx)
│   ├── Team logos
│   ├── Score display
│   └── AI summary badge
├── TeamStatsGrid (web/src/components/live/TeamStatsGrid.tsx)
│   ├── GeneralStats
│   ├── DownsStats
│   └── NegativePlaysStats
├── ScoringSummaryCard (web/src/components/live/ScoringSummaryCard.tsx)
│   └── Quarter-by-quarter scores
└── AIIntelligenceRail (web/src/components/live/AIIntelligenceRail.tsx)
    ├── Model recommendation
    ├── Key factors
    └── Model reasoning (hardcoded templates)
```

**Game Detail Page** (`/games/[id]`):
```
GameDetailPage
├── PredictionCard (web/src/components/game-detail/PredictionCard.tsx)
└── AnalyticsChart (web/src/components/game-detail/AnalyticsChart.tsx)
    └── WinProbabilityChart (Recharts)
```

### Component Catalog

| Component | File | Props | State | Data Source |
|-----------|------|-------|-------|-------------|
| LiveDashboard | `live/LiveDashboard.tsx` | `games`, `initialDetails` | `selectedGameId`, `isLoading` | Mock data |
| GameStrip | `live/GameStrip.tsx` | `games`, `selectedGameId`, `onSelect` | None | Mock data |
| HeroGameCard | `live/HeroGameCard.tsx` | `game` | None | Mock data |
| TeamStatsGrid | `live/TeamStatsGrid.tsx` | `game` | None | Mock data |
| ScoringSummaryCard | `live/ScoringSummaryCard.tsx` | `game` | None | Mock data |
| AIIntelligenceRail | `live/AIIntelligenceRail.tsx` | `game` | None | Mock data + hardcoded templates |
| PredictionCard | `game-detail/PredictionCard.tsx` | `prediction`, `market` | None | Mock data |
| WinProbabilityChart | `game-detail/WinProbabilityChart.tsx` | `winProbability` | None | Mock data |
| DashboardCard | `live/DashboardCard.tsx` | `title`, `children`, `variant` | None | None (wrapper) |
| QuarterbackPerformanceCard | `live/QuarterbackPerformanceCard.tsx` | `qb`, `team` | None | Mock data |
| PreGameLineCard | `live/PreGameLineCard.tsx` | `game` | None | Mock data |

### Design System

**Color Palette** (`web/src/app/globals.css`):
- **Background**: `#050508` (deep space black)
- **Foreground**: `#fafafa` (pure white)
- **Neon Blue**: `#4db8d9` (primary accent)
- **Neon Green**: `#4ade80` (success)
- **Neon Purple**: `#a78bfa` (tertiary)
- **Glass Backgrounds**: `rgba(255, 255, 255, 0.02-0.08)` (varying opacity)
- **Glass Borders**: `rgba(255, 255, 255, 0.06-0.15)`

**Typography**:
- **Font Family**: System fonts (sans-serif stack)
- **Scale**: Tailwind default (text-xs to text-6xl)
- **Weights**: 400 (normal), 500 (medium), 600 (semibold), 700 (bold)

**Spacing System**:
- Tailwind default spacing scale (0.25rem increments)
- Custom: `--radius: 0.625rem` (10px border radius)

**Border/Radius Patterns**:
- **Default Radius**: `0.625rem` (10px)
- **Glass Borders**: Subtle white borders with opacity
- **Neon Borders**: Colored borders for emphasis

**Shadow System**:
- **Glow Effects**: `--glow-blue`, `--glow-green`, `--glow-purple`
- **Box Shadows**: Tailwind defaults + custom glow utilities

**Animation Patterns**:
- **Framer Motion**: Page transitions, component animations
- **CSS Animations**: Pulse effects, scan lines, glow effects
- **Transitions**: `duration-300` standard

**Dark/Light Mode**:
- **Dark Mode Only**: No light mode implementation
- **CSS Variables**: Dark theme variables defined, light theme placeholders exist but unused

---

## Phase 6: Real-Time Architecture

### Live Data Updates

**Current Implementation**: **NONE**

**What Exists**:
- `setInterval` for clock updates (every 60 seconds) in `LiveDashboard.tsx`
- Mock data with `status: 'Live'` flag
- No actual real-time data fetching

**What's Missing**:
- WebSocket connection
- Server-Sent Events (SSE)
- Polling mechanism for game updates
- Backend API endpoint for live data

### Update Frequency

**Not Applicable** (no real-time implementation)

**If Implemented** (recommendations):
- Game score updates: Every 10-30 seconds during live games
- Prediction updates: Every 60 seconds (or on significant game events)
- Market updates: Every 60 seconds

### Connection Management

**Not Applicable** (no connections exist)

**If Implemented** (recommendations):
- WebSocket: Auto-reconnect with exponential backoff
- Polling: Retry logic with jitter
- Error handling: Graceful degradation to cached data

---

## Phase 7: Existing AI/LLM Integration

### Search Results

**NO AI/LLM INTEGRATION FOUND**

**Search Performed**:
- Grep for: `openai`, `anthropic`, `gemini`, `gpt`, `claude`, `llm`, `AI`
- Result: 0 matches (excluding UI text like "AI Insights" label)

### "AI Insights" Panel

**Location**: `web/src/components/live/AIIntelligenceRail.tsx`

**Implementation**: **Hardcoded Template Strings**

**Functions**:
1. `generateKeyFactors()` - Lines 123-174
   - Checks game stats (EPA, QBR, turnovers)
   - Returns hardcoded messages based on thresholds
   - Example: `"${team} offensive EPA +${epa.toFixed(2)}/play"`

2. `generateModelReasoning()` - Lines 177-243
   - Generates paragraph strings based on game state
   - Uses conditional logic to build narrative
   - Example: `"The model identifies ${team} as the stronger play..."`

**No AI/LLM Calls**: Zero API calls to OpenAI, Anthropic, or any LLM provider

**No System Prompts**: No prompt engineering or LLM integration

**No Streaming**: No streaming implementation

**Conclusion**: The "AI Insights" are purely deterministic template-based text generation, not actual AI/LLM output.

---

## Phase 8: Testing & Quality

### Test Coverage

**Backend Tests** (`tests/`):
- **Framework**: pytest
- **Test Files**: 22 test files found
- **Coverage**: Unknown (no coverage report found)
- **Test Types**:
  - Unit tests: `test_metrics.py`, `test_epa_features.py`
  - Integration tests: `test_phase1_sample_pipeline.py`, `test_phase2_pipelines_e2e.py`
  - Regression tests: `test_metric_regression.py`
  - Sanity checks: `test_backtest_sanity.py`

**Frontend Tests**: **NONE**
- No test files found in `web/src/`
- No Jest/Vitest configuration
- No React Testing Library setup

### Code Quality

**Linting**:
- **Backend**: No ESLint/Pylint config found (may use defaults)
- **Frontend**: ESLint configured (`web/eslint.config.mjs`)
  - Uses `eslint-config-next`
  - Next.js recommended rules

**Formatting**:
- **Backend**: Black (line-length: 100)
  - Config: `pyproject.toml`
- **Frontend**: No Prettier config found (may use editor defaults)

**Type Coverage**:
- **Backend**: No type hints enforced (Python)
- **Frontend**: TypeScript strict mode enabled
  - `strict: true` in `tsconfig.json`
  - Good type coverage in components

**Pre-commit Hooks**: **NONE**
- No `.git/hooks/` configuration found
- No Husky/pre-commit setup

### CI/CD

**Pipeline Configuration**: **NONE**
- No `.github/workflows/` directory
- No `.gitlab-ci.yml`
- No `.circleci/config.yml`
- No CI/CD configuration found

**Build Process**:
- **Backend**: Manual (`make` commands or Python scripts)
- **Frontend**: `npm run build` (Next.js)

**Deployment Targets**: **NONE**
- No Docker configuration
- No Kubernetes manifests
- No deployment scripts
- No environment-specific configs

**Environment Management**: **NONE**
- No `.env.example` files
- No environment variable documentation
- Credentials in `config/credentials.yaml` (gitignored)

---

## Phase 9: Performance Considerations

### Bundle Analysis

**Not Performed** (would require build)

**Code Splitting Strategy**:
- Next.js App Router: Automatic code splitting by route
- Dynamic imports: Not observed (may exist)

**Lazy Loading Patterns**:
- Next.js: Automatic lazy loading for routes
- Component-level: Not observed

**Tree Shaking**: Next.js handles automatically

### Runtime Performance

**Memoization Patterns**:
- **Not Observed**: No `useMemo`, `useCallback`, `React.memo` usage found
- **Potential Issue**: Re-renders may be inefficient

**Virtualization**: **NONE**
- No virtualized lists for long data sets
- Game lists are small (<20 games), likely not needed

**Image Optimization**: **NONE**
- Images use external URLs (ESPN CDN)
- No Next.js Image component usage
- No image optimization

**Caching Strategies**: **NONE**
- No service worker
- No cache headers configured
- No React Query/SWR caching

### Lighthouse / Core Web Vitals

**Not Analyzed** (would require running application)

**Potential Issues**:
- External image URLs (no optimization)
- No caching strategy
- No code splitting optimization observed

---

## Phase 10: Security & Environment

### Environment Variables

**Backend** (`config/credentials.yaml`):
- `odds_api.api_key` - The Odds API key
- `sportsdata.api_key` - (Future) SportsData API key
- `thesportsdb.api_key` - (Future) TheSportsDB API key

**Frontend**: **NONE**
- No environment variables used
- No `.env` files
- All configuration hardcoded

**Required vs Optional**:
- `odds_api.api_key`: Required if using Odds API
- Others: Optional (future integrations)

### API Security

**API Key Management**:
- Stored in `config/credentials.yaml` (gitignored)
- Template: `config/credentials.yaml.example`
- **Risk**: No encryption at rest
- **Risk**: No key rotation mechanism

**CORS Configuration**: **N/A** (no API server)

**Rate Limiting**: **NONE**
- No rate limiting implementation
- Scrapers may hit rate limits (no handling observed)

**Input Validation**: **PARTIAL**
- Backend: Some validation in ingestion modules
- Frontend: TypeScript provides compile-time validation
- Runtime validation: Not observed

### Authentication

**Authentication Provider**: **NONE**
- No auth system
- No user management
- No protected routes
- No session management

**Protected Routes**: **NONE**
- All routes publicly accessible
- No authentication middleware

---

## Phase 11: Technical Debt & Issues

### Code Smells

**Duplicated Code**:
1. **Spread-to-Probability Conversion**: 
   - `eval/backtest.py:18-35` (spread_to_implied_probability)
   - `models/architectures/market_baseline.py:54-77` (spread_to_probability)
   - **Same formula, different names**

2. **Feature Exclusion Lists**:
   - `models/training/trainer.py:130-144` (exclude_cols)
   - `scripts/simulate_real_world_prediction.py:215-229` (exclude_cols)
   - **Duplicated logic**

**Overly Complex Functions**:
- `models/training/trainer.py:837-1079` (`run_advanced_training_pipeline`) - 242 lines
- `models/architectures/stacking_ensemble.py:172-268` (`fit`) - Complex validation logic

**Unused Exports / Dead Code**:
- Many components in `web/src/components/` may be unused
- Need to verify actual usage

**Inconsistent Patterns**:
- Some modules use `sys.path.insert(0, ...)`, others use relative imports
- Mix of YAML and Python config loading

**TODO/FIXME Comments**: **Not Searched** (would require grep)

### Dependency Health

**Outdated Packages**: **Not Analyzed** (would require `npm audit` and `pip list --outdated`)

**Deprecated Dependencies**: **Unknown**

**Security Vulnerabilities**: **Not Analyzed** (would require `npm audit` and `pip-audit`)

**Unnecessary Dependencies**: **Unknown**

### Architecture Concerns

**Tight Coupling**:
- Frontend tightly coupled to mock data structure
- Backend tightly coupled to Parquet file structure
- No abstraction layer between data and UI

**Missing Abstractions**:
- **No API Layer**: Direct file access from scripts
- **No Service Layer**: Business logic mixed with data access
- **No Repository Pattern**: Direct pandas operations

**Scalability Bottlenecks**:
- **File-Based Storage**: Parquet files don't scale to multiple users
- **No Caching**: Repeated file reads
- **No Database**: Can't handle concurrent access

**Maintainability Issues**:
- **No API Contract**: Frontend and backend have no formal interface
- **Mock Data Duplication**: Frontend types duplicate backend schemas
- **Configuration Scattered**: Configs in multiple YAML files

---

## Phase 12: Documentation Status

### Existing Documentation

**README Quality**: **GOOD**
- `README.md`: Clear setup instructions, project structure, usage examples
- `web/README.md`: Basic Next.js setup (default)

**Inline Code Comments**: **MIXED**
- Some functions well-documented (docstrings)
- Some functions lack documentation
- Type hints: Partial (Python), Good (TypeScript)

**JSDoc / TSDoc Usage**: **NONE**
- No JSDoc comments in frontend
- TypeScript types provide some documentation

**API Documentation**: **NONE**
- No API exists to document
- `docs/live_output_contract.md` defines future API contract

**Architecture Decision Records**: **NONE**
- No ADR directory
- Decisions documented in `docs/` but not formalized

### Missing Documentation

**Should Be Documented But Isn't**:
1. **API Integration Guide**: How to connect frontend to backend (doesn't exist yet)
2. **Deployment Guide**: How to deploy to production (no deployment exists)
3. **Environment Setup**: Complete environment variable documentation
4. **Model Architecture Details**: Deep dive into ensemble construction
5. **Feature Engineering Pipeline**: Step-by-step feature generation
6. **Testing Guide**: How to run tests, what's tested, coverage goals
7. **Contributing Guide**: Code style, PR process, etc.

---

## Deliverables

### 1. Executive Summary

See top of document.

### 2. Detailed Findings

See Phases 1-12 above.

### 3. Algorithm Documentation

#### Complete Prediction Algorithm Writeup

**Model Architecture**: Stacking Ensemble

**Base Models**:
1. **Logistic Regression**:
   - Algorithm: sklearn `LogisticRegression`
   - Hyperparameters: C=1.0, max_iter=1000, random_state=42
   - Output: Home win probability (0-1)

2. **Gradient Boosting**:
   - Algorithm: XGBoost `XGBClassifier`
   - Hyperparameters: n_estimators=100, max_depth=3, learning_rate=0.1
   - Output: Home win probability (0-1)

**Meta-Model**:
- Type: Logistic Regression (default) or MLP (neural network)
- Input: Stacked predictions from base models
- Optional: Original features (if `include_features=True`)
- Output: Final ensemble probability

**Training Process**:
1. Split data chronologically (train: 2015-2021, val: 2022, test: 2023)
2. Train base models on training set
3. Get base model predictions on training set
4. Train meta-model on stacked predictions
5. Evaluate on validation/test sets

**Prediction Process**:
1. Load trained ensemble model
2. Load game features from feature table
3. Extract feature columns (exclude metadata)
4. Get base model predictions
5. Stack predictions → meta-model input
6. Meta-model outputs final probability

**Formulas**:

1. **Spread to Probability**:
   ```
   p = 1 / (1 + exp(spread / 3))
   ```
   - Location: `models/architectures/market_baseline.py:76`
   - Approximates: 3-point spread ≈ 60% win probability

2. **Probability to Spread**:
   ```
   spread = -3 * log(prob / (1 - prob))
   ```
   - Location: `scripts/simulate_real_world_prediction.py:291`
   - Inverse of spread-to-probability formula

3. **Edge Calculation**:
   ```
   edge_spread = predicted_spread - market_spread
   edge_prob = predicted_prob - market_implied_prob
   ```

4. **Confidence Score**:
   ```
   confidence = max(prob, 1 - prob)
   ```
   - Distance from 0.5 (coin flip)

**Input Specifications**:
- Feature matrix: `(n_samples, n_features)`
- Features: See Phase 4 "Inputs" section
- Target: Binary (1 = home win, 0 = away win)

**Output Specifications**:
- Primary: Home win probability (0-1 float)
- Derived: Predicted spread, confidence, edge vs market

### 4. Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    External Data Sources                     │
├─────────────────────────────────────────────────────────────┤
│  nflverse API  │  Odds API  │  NFL.com  │  Next Gen Stats  │
└────────┬───────────┬───────────┬───────────┬───────────────┘
         │           │           │           │
         ▼           ▼           ▼           ▼
┌─────────────────────────────────────────────────────────────┐
│              Ingestion Layer (Python)                       │
├─────────────────────────────────────────────────────────────┤
│  schedule.py  │  odds.py  │  injuries.py  │  play_by_play.py│
└────────┬───────────┬───────────┬───────────┬───────────────┘
         │           │           │           │
         ▼           ▼           ▼           ▼
┌─────────────────────────────────────────────────────────────┐
│              Raw Data Storage (Parquet)                      │
├─────────────────────────────────────────────────────────────┤
│  data/nfl/raw/schedules.parquet                             │
│  data/nfl/raw/odds.parquet                                  │
│  data/nfl/raw/injuries.parquet                              │
│  data/nfl/raw/plays.parquet                                 │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              Staging Layer                                  │
├─────────────────────────────────────────────────────────────┤
│  join_games_markets.py                                      │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│         Staged Data (Parquet)                               │
├─────────────────────────────────────────────────────────────┤
│  data/nfl/staged/games.parquet                               │
│  data/nfl/staged/markets.parquet                             │
│  data/nfl/staged/games_markets.parquet                      │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│         Feature Engineering Layer                           │
├─────────────────────────────────────────────────────────────┤
│  team_form_features.py  │  epa_features.py  │  qb_features.py│
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│         Processed Features (Parquet)                        │
├─────────────────────────────────────────────────────────────┤
│  data/nfl/processed/game_features_baseline.parquet          │
│  data/nfl/processed/game_features_phase2.parquet           │
│  data/nfl/processed/game_features_phase2b.parquet          │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│         Model Training                                      │
├─────────────────────────────────────────────────────────────┤
│  models/training/trainer.py                                 │
│  ├── Train base models (LR, GBM)                            │
│  ├── Train meta-model (stacking)                            │
│  └── Save to artifacts/models/                              │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│         Trained Models (Pickle)                              │
├─────────────────────────────────────────────────────────────┤
│  artifacts/models/nfl_stacked_ensemble_v2/ensemble_v1.pkl   │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│         Prediction Scripts                                  │
├─────────────────────────────────────────────────────────────┤
│  scripts/simulate_real_world_prediction.py                  │
│  ├── Load model                                             │
│  ├── Load features                                          │
│  ├── Predict probability                                    │
│  └── Calculate spread/edge                                  │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
                    [NO CONNECTION]
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│         Frontend (Next.js)                                  │
├─────────────────────────────────────────────────────────────┤
│  web/src/lib/mock_data.ts  ← Currently using mock data only │
│  web/src/components/live/  ← UI components                  │
└─────────────────────────────────────────────────────────────┘
```

**CRITICAL GAP**: No API layer connecting backend predictions to frontend.

### 5. Risk Assessment

#### Technical Risks

1. **HIGH**: No Backend API
   - **Impact**: Frontend cannot display real predictions
   - **Mitigation**: Build REST API (FastAPI/Flask) to serve predictions

2. **HIGH**: File-Based Storage
   - **Impact**: Doesn't scale, no concurrent access
   - **Mitigation**: Migrate to database (PostgreSQL/SQLite)

3. **MEDIUM**: No Real-Time Updates
   - **Impact**: Live dashboard shows stale data
   - **Mitigation**: Implement WebSocket or polling

4. **MEDIUM**: No Authentication
   - **Impact**: No user management, no access control
   - **Mitigation**: Add auth system (NextAuth.js, Auth0)

5. **LOW**: Mock Data in Production
   - **Impact**: Misleading user experience
   - **Mitigation**: Connect to real backend

#### Security Concerns

1. **MEDIUM**: API Keys in Plain Text
   - **Impact**: Credentials exposed if repo compromised
   - **Mitigation**: Use environment variables, encrypt at rest

2. **LOW**: No Input Validation
   - **Impact**: Potential injection attacks (if API built)
   - **Mitigation**: Add validation layer

3. **LOW**: No Rate Limiting
   - **Impact**: API abuse, scraping rate limits
   - **Mitigation**: Add rate limiting middleware

#### Scalability Concerns

1. **HIGH**: Parquet File Storage
   - **Impact**: Can't handle multiple concurrent users
   - **Mitigation**: Database migration

2. **MEDIUM**: No Caching
   - **Impact**: Repeated file reads, slow performance
   - **Mitigation**: Add Redis/memory cache

3. **LOW**: No Load Balancing
   - **Impact**: Single point of failure
   - **Mitigation**: Deploy with load balancer (future)

#### Dependencies on External Services

1. **HIGH**: nflverse API
   - **Risk**: Service downtime, API changes
   - **Mitigation**: Cache data, have backup sources

2. **MEDIUM**: The Odds API
   - **Risk**: Rate limits, API key expiration
   - **Mitigation**: Cache aggressively, monitor usage

3. **LOW**: NFL.com Scrapers
   - **Risk**: Website changes break scrapers
   - **Mitigation**: Monitor, update scrapers regularly

---

## Conclusion

This codebase represents a **well-structured ML pipeline** with a **modern frontend**, but has **critical gaps** preventing production deployment:

1. **No Backend API** - Frontend and backend are disconnected
2. **No Real-Time Updates** - Live dashboard shows static data
3. **No AI Integration** - "AI Insights" are templates, not AI
4. **No Deployment Setup** - No CI/CD, Docker, or production config

**Recommended Next Steps**:
1. Build REST API to connect frontend to backend
2. Implement real-time data updates
3. Add actual AI/LLM integration for insights
4. Set up deployment pipeline
5. Add authentication and user management

The ML pipeline is **production-ready**; the infrastructure to serve it is **not**.

---

**End of Audit Report**

