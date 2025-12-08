# NFL Prediction Model Backtest Report
## Seasons 2023-2024 (Proper Holdout - No Leakage)

**Generated:** 2025-12-07 18:47:12

**Training Configuration:**
- Training seasons: [2015, 2016, 2017, 2018, 2019, 2020, 2021]
- Validation season: 2022
- Test seasons: 2023-2024 (proper holdout, no leakage)

---

## Overall Summary

- **Total Games:** 533
- **Correct Predictions:** 333
- **Accuracy:** 62.48%
- **Brier Score:** 0.2295
- **Log Loss:** 0.6508
- **Mean Spread Error:** 10.84 points
- **RMSE Spread:** 14.14 points

## Market Comparison

- **Market Accuracy:** 31.33%
- **Model Accuracy:** 62.48%
- **Difference:** +31.14%

- **Market Brier Score:** 0.4814
- **Model Brier Score:** 0.2295
- **Difference:** -0.2519

## ROI vs Market

### Edge Threshold: 3%
- **Total Bets:** 323
- **Wins:** 0
- **Losses:** 0
- **ROI:** 38.08%
- **Profit:** $123.00

### Edge Threshold: 5%
- **Total Bets:** 320
- **Wins:** 0
- **Losses:** 0
- **ROI:** 38.12%
- **Profit:** $122.00

## Season-by-Season Performance

| Season | Games | Accuracy | Brier Score | Mean Spread Error |
|--------|-------|----------|-------------|-------------------|
| 2023 | 267 | 59.18% | 0.2372 | 10.89 |
| 2024 | 266 | 65.79% | 0.2217 | 10.80 |

## Confidence Analysis

| Confidence | Correct | Total | Accuracy | Mean Spread Error |
|------------|---------|-------|----------|-------------------|
| <50% | 0 | 0 | nan% | nan |
| 50-60% | 112 | 208 | 53.85% | 10.43 |
| 60-70% | 120 | 177 | 67.80% | 11.07 |
| 70-80% | 80 | 126 | 63.49% | 10.83 |
| 80%+ | 21 | 22 | 95.45% | 12.97 |

## Calibration Analysis

| Bin Range | Predicted | Actual | Count | Error |
|-----------|-----------|--------|-------|-------|
| [0.10, 0.20] | 0.181 | 0.000 | 1 | 0.181 |
| [0.20, 0.30] | 0.263 | 0.370 | 27 | 0.107 |
| [0.30, 0.40] | 0.349 | 0.426 | 68 | 0.077 |
| [0.40, 0.50] | 0.452 | 0.415 | 106 | 0.037 |
| [0.50, 0.60] | 0.551 | 0.490 | 102 | 0.061 |
| [0.60, 0.70] | 0.654 | 0.743 | 109 | 0.089 |
| [0.70, 0.80] | 0.751 | 0.636 | 99 | 0.114 |
| [0.80, 0.90] | 0.820 | 0.952 | 21 | 0.132 |

## Home vs Away Analysis

- **Home Wins Predicted:** 331 (62.10%)
- **Home Wins Actual:** 297 (55.72%)