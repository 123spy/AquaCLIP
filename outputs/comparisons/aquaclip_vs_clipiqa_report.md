# AquaCLIP-QA vs. Classic CLIP-IQA Comparison

## Comparison protocol

The classic CLIP-IQA results are read from `D:/实验/CLIP_IQA/result`.
For a fair direct comparison, CLIP-IQA predictions are matched to the exact AquaCLIP-QA in-domain test images by `image_path`.

Important protocol note:

- Classic CLIP-IQA is a zero-shot `Good photo.` probability.
- AquaCLIP-QA is supervised on each dataset train split and evaluated on the held-out test split.
- Therefore, this table compares performance on the same test images, but not the same training regime.

## Same Test Subset Results

| Dataset | Test images | AquaCLIP SRCC | CLIP-IQA SRCC | SRCC Gain | AquaCLIP PLCC | CLIP-IQA PLCC | AquaCLIP NRMSE | CLIP-IQA NRMSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| UID2021 | 180 | 0.7801 | 0.1761 | +0.6040 | 0.7930 | 0.1429 | 0.1497 | 0.3640 |
| UWIQA | 178 | 0.8246 | 0.5249 | +0.2997 | 0.8318 | 0.5136 | 0.1084 | 0.3937 |
| UIQD | 1263 | 0.9651 | 0.6778 | +0.2874 | 0.9698 | 0.7728 | 0.0621 | 0.2572 |
| SAUD2 | 480 | 0.6582 | 0.1967 | +0.4615 | 0.6665 | 0.2075 | 0.1825 | 0.3654 |
| SOTA | 1440 | 0.8927 | 0.0965 | +0.7962 | 0.9124 | 0.0906 | 0.0945 | 0.3725 |
| Mean | 3541 total | 0.8241 | 0.3344 | +0.4897 | 0.8347 | 0.3455 | 0.1194 | 0.3506 |

## Classic CLIP-IQA Full-Dataset Background

These are the original full-dataset CLIP-IQA results from `D:/实验/CLIP_IQA/result/summary.csv`.

| Dataset | Images | Full SRCC | Logistic PLCC | Logistic NRMSE |
|---|---:|---:|---:|---:|
| UID | 900 | 0.2692 | 0.2902 | 0.2284 |
| UWIQA | 890 | 0.5345 | 0.5396 | 0.1603 |
| UIQD | 6316 | 0.6495 | 0.8211 | 0.1389 |
| SAUD2 | 2400 | 0.1693 | 0.1826 | 0.2303 |
| SOTA | 7200 | 0.1170 | 0.1237 | 0.2083 |

## Main Observation

AquaCLIP-QA is substantially stronger than classic CLIP-IQA on the matched test subsets. The largest gains appear on SOTA and UID2021, where classic CLIP-IQA has weak underwater-domain ranking. UIQD is the strongest dataset for classic CLIP-IQA, but AquaCLIP-QA still improves SRCC from 0.6778 to 0.9651 on the matched test split.

This supports the paper narrative: generic CLIP-IQA provides useful but unstable visual quality priors for underwater images, while AquaCLIP-QA improves underwater IQA by adding supervised physics-calibrated fusion and reliability/attribute explanations.
