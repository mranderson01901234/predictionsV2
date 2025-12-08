# Phase B Readiness Audit Report

**Date**: 2024-01-XX  
**Audit Scope**: Data acquisition, model architecture, repository structure  
**Reference Documents**: `roadmap_phaseA_phaseB.md`, `blueprint.md`, `phasebdevroadmap.md`

---

## Executive Summary

The repository is **partially ready** for Phase B implementation. Phase A advanced models (FT-Transformer, TabNet) and stacking ensemble are already implemented, but several data acquisition gaps and structural improvements are needed before Phase B can begin effectively.

**Key Findings**:
- ✅ Advanced models (FT-Transformer, TabNet) exist
- ✅ Stacking ensemble supports multiple models
- ⚠️ Missing free API integrations (TheSportsDB, Sportsipy)
- ⚠️ Missing injuries ingestion module
- ⚠️ Odds data acquisition relies on CSV fallbacks
- ⚠️ Trainer module needs extension for advanced models

---

## 1. Data Acquisition Modules Audit

### 1.1 Missing API-Based Data Sources

#### **Priority: HIGH**

**Issue**: The repository currently relies primarily on `nflverse` (nfl-data-py) for data acquisition. While nflverse is excellent, `phasebdevroadmap.md` mentions integrating free APIs like TheSportsDB and Sportsipy for broader data coverage and redundancy.

**Missing Modules**:

1. **TheSportsDB Integration** (`ingestion/nfl/thesportsdb.py`)
   - **Purpose**: Free JSON API for team rosters, player stats, event results, historical data
   - **Coverage**: NFL, NBA, MLB, soccer, etc.
   - **Status**: Not implemented
   - **Action Required**:
     ```python
     # Suggested structure:
     # ingestion/nfl/thesportsdb.py
     def fetch_thesportsdb_schedules(seasons: List[int], api_key: Optional[str] = None) -> pd.DataFrame:
         """Fetch NFL schedules from TheSportsDB API."""
         # Implementation needed
     ```

2. **Sportsipy Integration** (`ingestion/nfl/sportsipy.py`)
   - **Purpose**: Python library that scrapes Sports-Reference.com for comprehensive stats
   - **Coverage**: Historical stats, player data, team performance
   - **Status**: Not implemented
   - **Action Required**:
     ```python
     # Suggested structure:
     # ingestion/nfl/sportsipy.py
     def fetch_sportsipy_team_stats(seasons: List[int]) -> pd.DataFrame:
         """Fetch team stats from Sportsipy (Sports-Reference)."""
         # Implementation needed
     ```

**Files to Create**:
- `ingestion/nfl/thesportsdb.py` (new)
- `ingestion/nfl/sportsipy.py` (new)
- `config/data/thesportsdb.yaml` (new, for API key config)

**Dependencies to Add**:
- `requests>=2.31.0` (for TheSportsDB API calls)
- `sportsipy>=0.15.0` (for Sportsipy integration)

**References**:
- `phasebdevroadmap.md` lines 9-23
- `blueprint.md` lines 240-250 (mentions multiple data sources)

---

### 1.2 Missing Injuries Module

#### **Priority: MEDIUM**

**Issue**: `blueprint.md` specifies an injuries ingestion module (`ingestion/nfl/injuries.py`), but this file does not exist.

**Missing Module**:
- **File**: `ingestion/nfl/injuries.py`
- **Purpose**: Scrape/fetch NFL injury reports (player status, position, injury type)
- **Status**: Not implemented
- **Data Source Options**:
  - NFL.com official injury reports (requires scraping)
  - TheSportsDB API (if available)
  - Sportsipy (if available)

**Action Required**:
```python
# ingestion/nfl/injuries.py
def fetch_nfl_injuries(seasons: List[int]) -> pd.DataFrame:
    """
    Fetch NFL injury reports.
    
    Returns DataFrame with columns:
    - game_id, team, player_id, position, injury_type, status, report_date
    """
    # Implementation needed
```

**References**:
- `blueprint.md` lines 247-250
- `blueprint.md` lines 201-204 (InjuryReport schema)

---

### 1.3 Odds Data Acquisition Issues

#### **Priority: HIGH**

**Issue**: Current odds ingestion (`ingestion/nfl/odds.py`) relies on CSV fallbacks or nflverse schedule data (which may not include odds). No free API-based odds fetching is implemented.

**Current State**:
- `ingestion/nfl/odds.py` lines 84-145: `fetch_nflverse_odds()` attempts to extract odds from nflverse schedule data
- Lines 311-323: Falls back to CSV file if nflverse doesn't have odds
- `config/data/nfl.yaml` line 9: `source: "manual_scrape"` indicates manual process

**Problems**:
1. **No API Integration**: No integration with free odds APIs (e.g., odds-api.com historical, TheSportsDB)
2. **Brittle Fallback**: CSV fallback requires manual data collection
3. **Limited Coverage**: nflverse may not have historical odds for all seasons

**Action Required**:

1. **Add Free Odds API Integration**:
   ```python
   # ingestion/nfl/odds_api.py (new)
   def fetch_odds_api_historical(seasons: List[int], api_key: Optional[str] = None) -> pd.DataFrame:
       """Fetch historical odds from odds-api.com or similar free API."""
       # Implementation needed
   ```

2. **Update `ingestion/nfl/odds.py`**:
   - Add API-based fetching as primary method
   - Keep CSV fallback for backward compatibility
   - Update `ingest_nfl_odds()` to try API first, then CSV, then nflverse

**Files to Modify**:
- `ingestion/nfl/odds.py` (lines 278-345): Add API integration
- `config/data/nfl.yaml`: Update odds source configuration

**References**:
- `ingestion/nfl/odds.py` lines 84-145, 278-345
- `config/data/nfl.yaml` lines 8-12
- `docs/data_sources.md` lines 99-103

---

### 1.4 Deprecated or Hardcoded Data Sources

#### **Priority: LOW-MEDIUM**

**Findings**:

1. **Team Stats Placeholders** (`ingestion/nfl/team_stats.py`):
   - Lines 212, 218, 223: Turnovers and yards fields default to 0 if not available
   - **Issue**: These are placeholders that should be filled from play-by-play data
   - **Action**: Document that these are Phase 1 limitations, will be filled in Phase 2
   - **Status**: Acceptable for Phase 1, but should be addressed before Phase B

2. **Odds CSV Dependency** (`ingestion/nfl/odds.py`):
   - Lines 311-323: Requires CSV file if nflverse doesn't have odds
   - **Issue**: Hardcoded dependency on manual CSV files
   - **Action**: Add API integration (see 1.3 above)

3. **No Hardcoded URLs Found**: No deprecated API endpoints or hardcoded URLs detected in codebase

**Files to Review**:
- `ingestion/nfl/team_stats.py` lines 212, 218, 223
- `ingestion/nfl/odds.py` lines 311-323

---

## 2. Model Architecture Audit

### 2.1 Ensemble Expansion Support

#### **Priority: LOW** (Already Supports Expansion)

**Current State**: ✅ **GOOD**

The repository has **two ensemble implementations**:

1. **Simple Ensemble** (`models/architectures/ensemble.py`):
   - **Limitation**: Hardcoded to 2 models (logit + GBM)
   - **Lines 24-35**: Takes `logit_model` and `gbm_model` as constructor args
   - **Lines 50-67**: Simple weighted average: `p = w * p_gbm + (1-w) * p_logit`
   - **Status**: Works for baseline, but not extensible

2. **Stacking Ensemble** (`models/architectures/stacking_ensemble.py`):
   - **Status**: ✅ **Supports multiple models**
   - **Lines 88-108**: Constructor accepts `base_models` dict (any BaseModel instances)
   - **Lines 123-125**: `add_model(name, model)` method for dynamic addition
   - **Lines 127-130**: `remove_model(name)` method for removal
   - **Lines 132-145**: `_get_base_predictions()` collects predictions from all base models
   - **Supports**: Logistic regression, GBM, FT-Transformer, TabNet (any BaseModel)

**Assessment**: Stacking ensemble **already supports expansion**. No structural changes needed.

**Recommendation**: 
- Use `StackingEnsemble` for Phase A/B advanced model ensembles
- Consider deprecating simple `EnsembleModel` or refactoring it to use `StackingEnsemble` internally

**References**:
- `models/architectures/ensemble.py` (simple, 2-model)
- `models/architectures/stacking_ensemble.py` (multi-model, extensible)
- `models/base.py` (BaseModel interface)

---

### 2.2 Trainer Module Extension Needs

#### **Priority: MEDIUM**

**Issue**: `models/training/trainer.py` only supports baseline models (logistic regression, GBM, simple ensemble). Advanced models (FT-Transformer, TabNet) are trained via separate `train_advanced.py` module.

**Current State**:

1. **Baseline Trainer** (`models/training/trainer.py`):
   - **Lines 173-270**: `train_models()` only trains LR, GBM, simple Ensemble
   - **Lines 17-19**: Imports only baseline models
   - **Status**: Does not support FT-Transformer or TabNet

2. **Advanced Trainer** (`models/training/train_advanced.py`):
   - **Lines 113-152**: `train_tabnet()` function
   - **Lines 155-233**: `train_ensemble()` supports advanced models
   - **Status**: Separate module, not integrated with main trainer

**Problem**: Two separate training paths create inconsistency and make experiment config integration harder.

**Action Required**:

1. **Unify Training Interface**:
   ```python
   # models/training/trainer.py
   # Add support for advanced models
   from models.architectures.ft_transformer import FTTransformerModel
   from models.architectures.tabnet import TabNetModel
   from models.architectures.stacking_ensemble import StackingEnsemble
   
   def train_model(
       model_type: str,
       X_train, y_train, X_val, y_val,
       config: dict,
       artifacts_dir: Path
   ) -> BaseModel:
       """Unified training function for all model types."""
       if model_type == "lr":
           return train_logistic_regression(...)
       elif model_type == "gbm":
           return train_gradient_boosting(...)
       elif model_type == "ft_transformer":
           return train_ft_transformer(...)
       elif model_type == "tabnet":
           return train_tabnet(...)
       elif model_type == "ensemble":
           return train_stacking_ensemble(...)
   ```

2. **Integrate with Experiment Configs**:
   - Update `train_models()` to read model type from experiment config
   - Support `experiment_schema.md` model block structure

**Files to Modify**:
- `models/training/trainer.py`: Add advanced model support
- Consider merging `train_advanced.py` functionality into `trainer.py`

**References**:
- `models/training/trainer.py` lines 173-270
- `models/training/train_advanced.py` lines 113-233
- `docs/experiment_schema.md` Section 3 (Model Block)

---

### 2.3 BaseModel Interface Compliance

#### **Priority: LOW** (Already Compliant)

**Status**: ✅ **All models implement BaseModel interface**

- `models/base.py`: Defines `BaseModel` ABC with `fit()`, `predict_proba()`, `save()`, `load()`
- All model classes inherit from `BaseModel`:
  - `LogisticRegressionModel` ✅
  - `GradientBoostingModel` ✅
  - `FTTransformerModel` ✅ (exists)
  - `TabNetModel` ✅ (exists)
  - `EnsembleModel` ✅
  - `StackingEnsemble` ✅

**No action needed**.

---

## 3. Repository Structure Audit

### 3.1 Alignment with Blueprint/Roadmap

#### **Priority: LOW** (Structure is Good)

**Assessment**: ✅ **Repository structure aligns well with `blueprint.md` and `roadmap_phaseA_phaseB.md`**

**Directory Structure Comparison**:

| Blueprint/Roadmap Expectation | Actual Location | Status |
|-------------------------------|-----------------|--------|
| `ingestion/nfl/` | ✅ `ingestion/nfl/` | ✅ Match |
| `features/nfl/` | ✅ `features/nfl/` | ✅ Match |
| `models/architectures/` | ✅ `models/architectures/` | ✅ Match |
| `models/training/` | ✅ `models/training/` | ✅ Match |
| `eval/` | ✅ `eval/` | ✅ Match |
| `orchestration/pipelines/` | ✅ `orchestration/pipelines/` | ✅ Match |
| `config/models/` | ✅ `config/models/` | ✅ Match |
| `config/data/` | ✅ `config/data/` | ✅ Match |

**Advanced Models Already Exist**:
- ✅ `models/architectures/ft_transformer.py` (exists)
- ✅ `models/architectures/tabnet.py` (exists)
- ✅ `models/architectures/stacking_ensemble.py` (exists)
- ✅ `models/calibration.py` (exists)

**Conclusion**: Structure is **ready for Phase B**. No restructuring needed.

---

### 3.2 Missing Directories/Modules

#### **Priority: LOW**

**Findings**:

1. **No `ingestion/base.py`**: Blueprint mentions base ingester interface, but not critical
2. **No `features/base.py`**: Blueprint mentions base feature generator interface, but not critical
3. **No `sports/` directory**: Blueprint mentions sport-specific coordinators, but Phase B is NFL-only

**Recommendation**: These can be added later if multi-sport support is needed. Not blocking for Phase B.

---

## 4. Prioritized Action Items

### Priority 1: CRITICAL (Block Phase B)

1. **Unify Training Interface** (`models/training/trainer.py`)
   - **File**: `models/training/trainer.py`
   - **Action**: Extend `train_models()` to support FT-Transformer, TabNet, StackingEnsemble
   - **Lines to Modify**: 173-270
   - **Code Changes**:
     ```python
     # Add imports
     from models.architectures.ft_transformer import FTTransformerModel
     from models.architectures.tabnet import TabNetModel
     from models.architectures.stacking_ensemble import StackingEnsemble
     
     # Extend train_models() or create train_model() unified function
     # Support model_type from config: "lr", "gbm", "ft_transformer", "tabnet", "ensemble"
     ```
   - **Estimated Effort**: 4-6 hours
   - **Dependencies**: None

2. **Integrate Experiment Configs with Trainer**
   - **File**: `models/training/trainer.py`
   - **Action**: Load model config from experiment YAML (per `experiment_schema.md`)
   - **Lines to Modify**: 26-30 (config loading), 173-270 (training logic)
   - **Code Changes**:
     ```python
     def load_experiment_config(config_path: Path) -> dict:
         """Load experiment config YAML."""
         # Implementation
     
     def train_from_experiment_config(config_path: Path):
         """Train models from experiment config."""
         config = load_experiment_config(config_path)
         # Train models per config
     ```
   - **Estimated Effort**: 3-4 hours
   - **Dependencies**: Experiment config schema finalized

---

### Priority 2: HIGH (Important for Robustness)

3. **Add Free Odds API Integration** (`ingestion/nfl/odds.py`)
   - **File**: `ingestion/nfl/odds.py`
   - **Action**: Add API-based odds fetching (odds-api.com, TheSportsDB, etc.)
   - **Lines to Modify**: 278-345 (`ingest_nfl_odds()`)
   - **Code Changes**:
     ```python
     def fetch_odds_api(seasons: List[int], api_key: Optional[str] = None) -> pd.DataFrame:
         """Fetch odds from free API."""
         # Try odds-api.com or similar
         # Return DataFrame with odds data
         pass
     
     # Update ingest_nfl_odds() to try API first:
     # 1. Try API
     # 2. Try CSV (fallback)
     # 3. Try nflverse (fallback)
     ```
   - **Estimated Effort**: 6-8 hours
   - **Dependencies**: API key (if required), API documentation

4. **Create Injuries Ingestion Module** (`ingestion/nfl/injuries.py`)
   - **File**: `ingestion/nfl/injuries.py` (new)
   - **Action**: Implement injury report fetching
   - **Code Changes**:
     ```python
     def fetch_nfl_injuries(seasons: List[int]) -> pd.DataFrame:
         """Fetch NFL injury reports."""
         # Scrape NFL.com or use API
         # Return DataFrame with columns: game_id, team, player_id, position, injury_type, status, report_date
         pass
     ```
   - **Estimated Effort**: 8-10 hours (scraping can be complex)
   - **Dependencies**: Scraping library (BeautifulSoup, Scrapy) or API access

---

### Priority 3: MEDIUM (Nice to Have)

5. **Add TheSportsDB Integration** (`ingestion/nfl/thesportsdb.py`)
   - **File**: `ingestion/nfl/thesportsdb.py` (new)
   - **Action**: Implement TheSportsDB API client
   - **Estimated Effort**: 4-6 hours
   - **Dependencies**: TheSportsDB API key (free tier available)

6. **Add Sportsipy Integration** (`ingestion/nfl/sportsipy.py`)
   - **File**: `ingestion/nfl/sportsipy.py` (new)
   - **Action**: Implement Sportsipy wrapper for historical stats
   - **Estimated Effort**: 3-4 hours
   - **Dependencies**: `sportsipy` package, rate limiting

7. **Refactor Simple Ensemble** (`models/architectures/ensemble.py`)
   - **File**: `models/architectures/ensemble.py`
   - **Action**: Refactor to use `StackingEnsemble` internally or deprecate
   - **Estimated Effort**: 2-3 hours
   - **Dependencies**: None

---

### Priority 4: LOW (Documentation/Polish)

8. **Document Team Stats Placeholders** (`ingestion/nfl/team_stats.py`)
   - **File**: `ingestion/nfl/team_stats.py`
   - **Action**: Add comments explaining that turnovers/yards are Phase 1 placeholders
   - **Lines to Modify**: 212, 218, 223
   - **Estimated Effort**: 15 minutes

9. **Update Requirements.txt**
   - **File**: `requirements.txt`
   - **Action**: Add `requests`, `sportsipy` if implementing API integrations
   - **Estimated Effort**: 5 minutes

---

## 5. Summary & Recommendations

### Phase B Readiness Score: **75/100**

**Strengths**:
- ✅ Advanced models already implemented
- ✅ Stacking ensemble supports expansion
- ✅ Repository structure aligns with blueprint
- ✅ BaseModel interface is well-defined

**Gaps**:
- ⚠️ Trainer module needs extension for advanced models
- ⚠️ Missing free API integrations (odds, injuries)
- ⚠️ Experiment config integration incomplete

### Recommended Path Forward

1. **Before Phase B**:
   - Complete Priority 1 items (unify training interface, experiment config integration)
   - These are **blocking** for Phase B to work smoothly

2. **During Phase B** (can be done in parallel):
   - Complete Priority 2 items (odds API, injuries module)
   - These improve robustness but don't block evaluation engine

3. **After Phase B** (future enhancements):
   - Complete Priority 3 items (TheSportsDB, Sportsipy)
   - These add redundancy and broader data coverage

### Critical Path for Phase B

**Minimum Required**:
1. ✅ Advanced models exist (FT-Transformer, TabNet)
2. ✅ Stacking ensemble exists
3. ⚠️ **Trainer extension** (Priority 1) ← **BLOCKER**
4. ⚠️ **Experiment config integration** (Priority 1) ← **BLOCKER**

**Conclusion**: Phase B can begin **after completing Priority 1 items**. Priority 2-4 items can be done in parallel or deferred.

---

*End of Audit Report*



