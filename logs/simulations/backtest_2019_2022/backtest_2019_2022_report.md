# NFL Prediction Model Backtest Report
## Seasons 2019-2022

**Generated:** 2025-12-07 18:44:49

---

## Overall Summary

- **Total Games:** 1017
- **Correct Predictions:** 672
- **Accuracy:** 66.08%
- **Brier Score:** 0.2152
- **Log Loss:** 0.6193
- **Mean Spread Error:** 10.28 points
- **RMSE Spread:** 13.34 points

## Market Comparison

- **Market Accuracy:** 35.10%
- **Model Accuracy:** 66.08%
- **Difference:** +30.97%

- **Market Brier Score:** 0.4928
- **Model Brier Score:** 0.2152
- **Difference:** -0.2776

## ROI vs Market

### Edge Threshold: 3%
- **Total Bets:** 616
- **Wins:** 0
- **Losses:** 0
- **ROI:** 29.87%
- **Profit:** $184.00

### Edge Threshold: 5%
- **Total Bets:** 606
- **Wins:** 0
- **Losses:** 0
- **ROI:** 30.36%
- **Profit:** $184.00

## Season-by-Season Performance

| Season | Games | Accuracy | Brier Score | Mean Spread Error |
|--------|-------|----------|-------------|-------------------|
| 2019 | 235 | 66.81% | 0.2072 | 10.51 |
| 2020 | 251 | 67.73% | 0.2114 | 10.31 |
| 2021 | 264 | 64.39% | 0.2200 | 11.38 |
| 2022 | 267 | 65.54% | 0.2211 | 8.97 |

## Confidence Analysis

| Confidence | Correct | Total | Accuracy | Mean Spread Error |
|------------|---------|-------|----------|-------------------|
| <50% | 0 | 0 | nan% | nan |
| 50-60% | 206 | 364 | 56.59% | 9.50 |
| 60-70% | 217 | 330 | 65.76% | 9.82 |
| 70-80% | 213 | 285 | 74.74% | 11.23 |
| 80%+ | 36 | 38 | 94.74% | 14.56 |

## Calibration Analysis

| Bin Range | Predicted | Actual | Count | Error |
|-----------|-----------|--------|-------|-------|
| [0.10, 0.20] | 0.196 | 0.000 | 4 | 0.196 |
| [0.20, 0.30] | 0.266 | 0.261 | 88 | 0.005 |
| [0.30, 0.40] | 0.349 | 0.309 | 123 | 0.040 |
| [0.40, 0.50] | 0.454 | 0.367 | 188 | 0.087 |
| [0.50, 0.60] | 0.547 | 0.494 | 176 | 0.053 |
| [0.60, 0.70] | 0.651 | 0.638 | 207 | 0.014 |
| [0.70, 0.80] | 0.751 | 0.751 | 197 | 0.001 |
| [0.80, 0.90] | 0.819 | 0.941 | 34 | 0.123 |

## Home vs Away Analysis

- **Home Wins Predicted:** 614 (60.37%)
- **Home Wins Actual:** 529 (52.02%)