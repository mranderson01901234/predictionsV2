# Prediction Test Report - First 10 Games of 2025

**Date**: 2025-12-07 18:13:40

## Summary

- **Games Analyzed**: 10
- **Games with Results**: 10
- **Accuracy**: 40.0% (4/10)
- **Spread MAE**: 14.93 points
- **Brier Score**: 0.4783
- **Log Loss**: 1.6412

## Leakage Check

- **Future scores in features**: True
- **Result column present**: False
- **Spread column present**: False
- **Future-looking features**: False

## Detailed Results

| Week | Away | Home | Predicted | Prob | Spread | Actual | Spread | Correct |
|------|------|------|-----------|------|--------|--------|--------|---------|
| 1 | ARI | NO | NO | 0.765 | +7.4 | ARI | -7.0 | ✗ |
| 1 | BAL | BUF | BUF | 0.566 | +1.8 | BUF | +1.0 | ✓ |
| 1 | CAR | JAX | CAR | 0.238 | -7.3 | JAX | +16.0 | ✗ |
| 1 | CIN | CLE | CIN | 0.037 | -13.0 | CIN | -1.0 | ✓ |
| 1 | DAL | PHI | PHI | 0.992 | +13.8 | PHI | +4.0 | ✓ |
| 1 | DET | GB | GB | 0.969 | +13.1 | GB | +14.0 | ✓ |
| 1 | KC | LAC | KC | 0.030 | -13.2 | LAC | +6.0 | ✗ |
| 1 | LV | NE | NE | 0.967 | +13.1 | LV | -7.0 | ✗ |
| 1 | MIA | IND | MIA | 0.010 | -13.7 | IND | +25.0 | ✗ |
| 1 | MIN | CHI | CHI | 0.756 | +7.2 | MIN | -3.0 | ✗ |
