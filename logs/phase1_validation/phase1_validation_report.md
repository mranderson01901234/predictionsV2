# Phase 1 Validation Report

Generated: 2025-12-07 18:53:43

---

## Summary

- **Test Accuracy**: 54.89%
- **Test Brier Score**: 0.2428
- **Validation Accuracy**: 56.55%

## Success Criteria

- **✗ Ensemble beats baseline by 1%+**: -8.01% improvement
- **✓ All confidence tiers accurate**: PASS
- **? Rest features show positive importance**: TODO
- **✗ Test accuracy >= 60%**: 54.89%

## Detailed Results

### Test Set Performance

- **Games**: 266
- **Accuracy**: 0.5489
- **Brier Score**: 0.2428
- **Log Loss**: 0.6786

### Calibration Metrics

- **ECE (Expected Calibration Error)**: 0.0000
- **MCE (Maximum Calibration Error)**: 0.0157
- **Brier Score**: 0.2428

### Accuracy by Confidence Tier

| Confidence | Games | Accuracy | Required | Status |
|------------|-------|----------|----------|--------|
| 50%-60% | N/A | 54.89% | 52.00% | ✓ |

### ROI Analysis

**Edge Threshold ≥ 3%**:
- ROI: -41.75%
- Number of Bets: 103
- Win Rate: 29.13%

**Edge Threshold ≥ 5%**:
- ROI: -41.75%
- Number of Bets: 103
- Win Rate: 29.13%

---

## Files Generated

- Calibration plots: `/home/dp/Documents/predictionV2/logs/phase1_validation/calibration_test/`
- Reliability diagrams: `/home/dp/Documents/predictionV2/logs/phase1_validation/calibration_test/reliability_diagram.png`
- Accuracy by confidence: `/home/dp/Documents/predictionV2/logs/phase1_validation/calibration_test/accuracy_by_confidence.png`