# UIED 10-Fold Results

AquaCLIP-QA was trained with the official UIED tt10 split files, 800 train / 200 test per fold, using CUDA.

## AquaCLIP-QA Summary

- Mean SRCC: 0.5993 ? 0.0602
- Mean PLCC: 0.6495 ? 0.0529
- Mean NRMSE: 0.1755 ? 0.0108
- Mean epochs: 55.4 ? 15.1

## Reference Comparison

| Method | SRCC | PLCC | Source |
|---|---:|---:|---|
| BRISQUE | 0.4650 | 0.4960 | UIF paper |
| NIQE | 0.3260 | 0.3440 | UIF paper |
| UCIQE | 0.2520 | 0.2980 | UIF paper |
| UIQM | 0.2760 | 0.2680 | UIF paper |
| UIF | 0.7330 | 0.7570 | UIF paper |
| AquaCLIP-QA | 0.5993 | 0.6495 | ours, UIED 10-fold |

Note: UIF paper numbers are reported from the original paper table; protocol details should be double-checked before making strict claims.
