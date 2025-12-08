# Predictr Python API

FastAPI server exposing ML predictions for the Next.js frontend.

## Setup

1. Install dependencies:
```bash
cd /path/to/predictionV2
pip install -r api/requirements.txt
```

2. Ensure model artifacts exist:
- Model: `artifacts/models/nfl_stacked_ensemble_v2/ensemble_v1.pkl`
- Features: `data/nfl/processed/game_features_baseline.parquet`

3. Run locally:
```bash
python -m api.run_local
```

The API will be available at `http://localhost:8000`

## Endpoints

- `GET /health` - Health check
- `GET /predictions/{game_id}` - Get prediction for a specific game
- `GET /predictions/week/{season}/{week}` - Get all predictions for a week
- `GET /model/performance` - Model performance metrics
- `GET /model/methodology` - Model architecture and methodology
- `GET /features/{game_id}` - Raw feature values for debugging

## Testing

```bash
# Health check
curl http://localhost:8000/health

# Get prediction
curl http://localhost:8000/predictions/nfl_2024_14_IND_JAX

# Get week predictions
curl http://localhost:8000/predictions/week/2024/14
```

## Deployment

The API is designed to be serverless-ready. See commented Modal deployment code in `api/main.py`.

For production, update CORS origins in `api/main.py` to include your frontend domain.

