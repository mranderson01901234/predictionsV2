# Universal Sports Prediction Pipeline

A market-aware, data-rich, multi-sport prediction engine designed to identify edges versus betting markets. Initially targeting NFL games, with extensibility to other sports.

## Project Status

**Phase 1A**: NFL Schedule and Odds Ingestion ✅ Complete

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Run Phase 1A Pipeline

```bash
# Option 1: Using nflverse (if odds data available)
python -m ingestion.nfl.run_phase1a

# Option 2: Using CSV file with odds data
python -m ingestion.nfl.run_phase1a --odds-csv data/nfl/raw/odds.csv
```

### Generate Odds Template

If you need to manually populate odds data:

```bash
python -m ingestion.nfl.generate_odds_template
```

This creates a CSV template at `data/nfl/raw/odds_template.csv` that you can fill in with historical odds.

### Run Tests

```bash
pytest tests/test_phase1a_ingestion.py -v
```

## Project Structure

```
predictionV2/
├── config/              # Configuration files
├── data/                # Data storage (raw, staged, processed)
├── docs/                # Documentation
├── ingestion/           # Data ingestion modules
├── tests/               # Test files
└── requirements.txt     # Python dependencies
```

See `blueprint.md` for complete architecture documentation.

## Phase 1A Deliverables

After running the pipeline, you should have:

- `data/nfl/staged/games.parquet` - NFL schedules and results (2015-2024)
- `data/nfl/staged/markets.parquet` - Historical betting odds
- `data/nfl/staged/games_markets.parquet` - Joined games and markets data

## Data Sources

See `docs/data_sources.md` for detailed information about:
- NFL schedule data sources (nflverse)
- Betting odds data sources
- Data schemas and validation
- Usage instructions

## Phase 1C: Baseline Model Training & Evaluation

Run the complete Phase 1C pipeline:

```bash
source venv/bin/activate
python -m orchestration.pipelines.phase1c_pipeline
```

This will:
1. Train logistic regression, gradient boosting, and ensemble models on 2015-2021 data
2. Evaluate on 2022 (validation) and 2023 (test) seasons
3. Calculate ROI vs closing line
4. Generate evaluation report at `docs/reports/nfl_baseline_phase1c.md`

### Phase 1C Deliverables

After running the pipeline:
- Trained models saved in `models/artifacts/nfl_baseline/`
- Evaluation report: `docs/reports/nfl_baseline_phase1c.md`
- Metrics: Accuracy, Brier score, log loss, calibration, ROI

## Phase 1D: Baseline Model Sanity Check & Market Comparison

Run the Phase 1D sanity check pipeline:

```bash
source venv/bin/activate
python -m orchestration.pipelines.phase1d_pipeline
```

This will:
1. Create market-only baseline model
2. Compare all models (logit, GBM, ensemble) vs market baseline
3. Run season-by-season stability analysis
4. Validate ROI calculation with synthetic tests
5. Generate sanity check report at `docs/reports/nfl_baseline_phase1d.md`

### Phase 1D Deliverables

After running the pipeline:
- Market baseline model implementation
- Comprehensive ROI/edge calculation audit
- Season-by-season stability metrics
- Sanity check report comparing models vs market baseline

## Next Steps

Phase 2 will add:
- Play-by-play data (EPA, success rate)
- Player-level features
- Injury impact metrics
- Advanced matchup features

See `blueprint.md` for the complete roadmap.

