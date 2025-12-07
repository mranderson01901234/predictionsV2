# NFL Baseline Model Evaluation Report (Phase 1C)

## Dataset

- **Training Seasons**: [2015, 2016, 2017, 2018, 2019, 2020, 2021]
- **Validation Season**: 2022
- **Test Season**: 2023

## Models Evaluated

1. Logistic Regression
2. Gradient Boosting
3. Ensemble (blended)

## Logistic Regression

### Validation Set

- **Games**: 267
- **Accuracy**: 0.6592
- **Brier Score**: 0.2214
- **Log Loss**: 0.6342
- **Mean Calibration Error**: 0.0508

#### Calibration

| Bin | Predicted | Actual | Count | Error |
|-----|-----------|--------|-------|-------|
| 2.0 | 0.191 | 0.000 | 1.0 | 0.191 |
| 3.0 | 0.259 | 0.333 | 15.0 | 0.074 |
| 4.0 | 0.361 | 0.343 | 35.0 | 0.019 |
| 5.0 | 0.453 | 0.396 | 48.0 | 0.057 |
| 6.0 | 0.551 | 0.596 | 57.0 | 0.046 |
| 7.0 | 0.646 | 0.654 | 52.0 | 0.008 |
| 8.0 | 0.739 | 0.745 | 47.0 | 0.005 |
| 9.0 | 0.827 | 0.833 | 12.0 | 0.007 |

#### ROI vs Closing Line

**Edge Threshold: 3%**
- Bets: 164
- Win Rate: 68.29%
- ROI: 36.59%

**Edge Threshold: 5%**
- Bets: 163
- Win Rate: 68.71%
- ROI: 37.42%

### Test Set

- **Games**: 267
- **Accuracy**: 0.6292
- **Brier Score**: 0.2344
- **Log Loss**: 0.6629
- **Mean Calibration Error**: 0.1000

#### ROI vs Closing Line

**Edge Threshold: 3%**
- Bets: 162
- Win Rate: 68.52%
- ROI: 37.04%

**Edge Threshold: 5%**
- Bets: 160
- Win Rate: 68.12%
- ROI: 36.25%

---

## Gradient Boosting

### Validation Set

- **Games**: 267
- **Accuracy**: 0.6479
- **Brier Score**: 0.2299
- **Log Loss**: 0.6578
- **Mean Calibration Error**: 0.1180

#### Calibration

| Bin | Predicted | Actual | Count | Error |
|-----|-----------|--------|-------|-------|
| 2.0 | 0.179 | 0.250 | 4.0 | 0.071 |
| 3.0 | 0.253 | 0.379 | 29.0 | 0.127 |
| 4.0 | 0.356 | 0.353 | 34.0 | 0.003 |
| 5.0 | 0.455 | 0.449 | 49.0 | 0.006 |
| 6.0 | 0.555 | 0.677 | 31.0 | 0.122 |
| 7.0 | 0.654 | 0.571 | 42.0 | 0.082 |
| 8.0 | 0.752 | 0.814 | 43.0 | 0.062 |
| 9.0 | 0.845 | 0.667 | 33.0 | 0.179 |
| 10.0 | 0.910 | 0.500 | 2.0 | 0.410 |

#### ROI vs Closing Line

**Edge Threshold: 3%**
- Bets: 163
- Win Rate: 69.94%
- ROI: 39.88%

**Edge Threshold: 5%**
- Bets: 162
- Win Rate: 70.37%
- ROI: 40.74%

### Test Set

- **Games**: 267
- **Accuracy**: 0.5955
- **Brier Score**: 0.2500
- **Log Loss**: 0.7034
- **Mean Calibration Error**: 0.1474

#### ROI vs Closing Line

**Edge Threshold: 3%**
- Bets: 168
- Win Rate: 66.67%
- ROI: 33.33%

**Edge Threshold: 5%**
- Bets: 164
- Win Rate: 67.68%
- ROI: 35.37%

---

## Ensemble

### Validation Set

- **Games**: 267
- **Accuracy**: 0.6479
- **Brier Score**: 0.2232
- **Log Loss**: 0.6395
- **Mean Calibration Error**: 0.0699

#### Calibration

| Bin | Predicted | Actual | Count | Error |
|-----|-----------|--------|-------|-------|
| 2.0 | 0.182 | 0.000 | 1.0 | 0.182 |
| 3.0 | 0.261 | 0.350 | 20.0 | 0.089 |
| 4.0 | 0.357 | 0.286 | 35.0 | 0.071 |
| 5.0 | 0.448 | 0.480 | 50.0 | 0.032 |
| 6.0 | 0.552 | 0.568 | 44.0 | 0.016 |
| 7.0 | 0.646 | 0.635 | 52.0 | 0.011 |
| 8.0 | 0.751 | 0.795 | 44.0 | 0.044 |
| 9.0 | 0.828 | 0.714 | 21.0 | 0.114 |

#### ROI vs Closing Line

**Edge Threshold: 3%**
- Bets: 167
- Win Rate: 68.26%
- ROI: 36.53%

**Edge Threshold: 5%**
- Bets: 165
- Win Rate: 68.48%
- ROI: 36.97%

### Test Set

- **Games**: 267
- **Accuracy**: 0.6180
- **Brier Score**: 0.2395
- **Log Loss**: 0.6747
- **Mean Calibration Error**: 0.0893

#### ROI vs Closing Line

**Edge Threshold: 3%**
- Bets: 163
- Win Rate: 68.71%
- ROI: 37.42%

**Edge Threshold: 5%**
- Bets: 161
- Win Rate: 68.94%
- ROI: 37.89%

---

## Market Baseline

### Validation Set

- **Games**: 267
- **Accuracy**: 0.3408
- **Brier Score**: 0.4780
- **Log Loss**: 1.4880
- **Mean Calibration Error**: 0.4156

#### Calibration

| Bin | Predicted | Actual | Count | Error |
|-----|-----------|--------|-------|-------|
| 1.0 | 0.043 | 0.875 | 48.0 | 0.832 |
| 2.0 | 0.138 | 0.657 | 35.0 | 0.519 |
| 3.0 | 0.252 | 0.622 | 45.0 | 0.370 |
| 4.0 | 0.315 | 0.593 | 27.0 | 0.277 |
| 5.0 | 0.417 | 0.308 | 13.0 | 0.110 |
| 6.0 | 0.583 | 0.467 | 15.0 | 0.116 |
| 7.0 | 0.682 | 0.421 | 19.0 | 0.261 |
| 8.0 | 0.755 | 0.400 | 30.0 | 0.355 |
| 9.0 | 0.877 | 0.250 | 16.0 | 0.627 |
| 10.0 | 0.953 | 0.263 | 19.0 | 0.690 |

#### ROI vs Closing Line

**Edge Threshold: 3%**
- Bets: 0
- Win Rate: 0.00%
- ROI: 0.00%

**Edge Threshold: 5%**
- Bets: 0
- Win Rate: 0.00%
- ROI: 0.00%

### Test Set

- **Games**: 267
- **Accuracy**: 0.3371
- **Brier Score**: 0.4589
- **Log Loss**: 1.4282
- **Mean Calibration Error**: 0.3981

#### ROI vs Closing Line

**Edge Threshold: 3%**
- Bets: 0
- Win Rate: 0.00%
- ROI: 0.00%

**Edge Threshold: 5%**
- Bets: 0
- Win Rate: 0.00%
- ROI: 0.00%

---

## Model Comparison

### Test Set Performance

| Model | Accuracy | Brier | Log Loss | Calibration Error |
|-------|----------|-------|----------|-------------------|
| Logistic Regression | 0.6292 | 0.2344 | 0.6629 | 0.1000 |
| Gradient Boosting | 0.5955 | 0.2500 | 0.7034 | 0.1474 |
| Ensemble | 0.6180 | 0.2395 | 0.6747 | 0.0893 |
| Market Baseline | 0.3371 | 0.4589 | 1.4282 | 0.3981 |

---

*Report generated by Phase 1C evaluation pipeline*