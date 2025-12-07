# Prediction Simulation Report - 2025 Weeks 1-13

**Date**: 2025-12-07 18:15:12

## Summary

- **Games Analyzed**: 182
- **Games with Results**: 182
- **Accuracy**: 59.3% (108/182)
- **Spread MAE**: 13.70 points
- **Spread RMSE**: 17.11 points
- **Brier Score**: 0.3623
- **Log Loss**: 1.3493
- **Mean Calibration Error**: 0.3278

## ROI Analysis (assuming -110 odds)

| Confidence Threshold | Bets | Wins | Win Rate | ROI |
|---------------------|------|------|----------|-----|
| All Games | 182 | 108 | 59.3% | 13.29% |
| ≥60% | 176 | 102 | 58.0% | 10.64% |
| ≥70% | 163 | 93 | 57.1% | 8.92% |
| ≥80% | 150 | 87 | 58.0% | 10.73% |

## Weekly Breakdown

| Week | Games | Correct | Accuracy |
|------|-------|---------|----------|
| 1 | 15 | 8 | 53.3% |
| 2 | 15 | 9 | 60.0% |
| 3 | 15 | 10 | 66.7% |
| 4 | 15 | 10 | 66.7% |
| 5 | 13 | 8 | 61.5% |
| 6 | 14 | 7 | 50.0% |
| 7 | 14 | 9 | 64.3% |
| 8 | 13 | 10 | 76.9% |
| 9 | 13 | 3 | 23.1% |
| 10 | 13 | 7 | 53.8% |
| 11 | 14 | 8 | 57.1% |
| 12 | 13 | 10 | 76.9% |
| 13 | 15 | 9 | 60.0% |

## Leakage Check

- **Future scores in features**: True
- **Result column present**: False
- **Spread column present**: False
- **Future-looking features**: False

⚠ Potential leakage detected - scores are in dataframe but excluded from features (expected)

