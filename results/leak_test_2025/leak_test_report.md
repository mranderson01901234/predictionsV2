# Leak-Free Test Report: 2025 Weeks 1-13

**Generated**: 2025-12-07 21:19:20

---

## Executive Summary

This report verifies that the model evaluation on 2025 Weeks 1-13 is completely leak-free.

### Test Set Performance

- **Accuracy**: 0.5714 (57.14%)
- **Brier Score**: 0.3581
- **Log Loss**: 1.3039
- **Mean Calibration Error**: 0.2114
- **Number of Games**: 182

### Leakage Audit

- **Total Checks**: 12
- **Passed**: 12
- **Failed**: 0

**Status**: ✓ ALL CHECKS PASSED

### Data Splits

- **Training**: 2015-2024 (2458 games)
- **Validation**: 2024 (266 games)
- **Test**: 2025 Weeks 1-13 (182 games)

---

## Detailed Checks

- **no_test_in_train**: ✓ PASS
- **no_test_in_val**: ✓ PASS
- **test_is_correct**: ✓ PASS
- **no_train_test_overlap**: ✓ PASS
- **no_val_test_overlap**: ✓ PASS
- **train_before_test**: ✓ PASS
- **rolling_features_exist**: ✓ PASS
- **rolling_features_correct**: ✓ PASS
- **no_post_game_features**: ✓ PASS
- **reasonable_nan_counts**: ✓ PASS
- **has_calibration**: ✓ PASS
- **model_predicts**: ✓ PASS
