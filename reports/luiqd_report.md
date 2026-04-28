# LUIQD Results

AquaCLIP-QA was trained on `data/manifests/luiqd.csv` with CUDA. The manifest contains 51,350 train images and 12,820 test images.

## AquaCLIP-QA

- SRCC: 0.8795
- PLCC: 0.8726
- KRCC: 0.7073
- NRMSE: 0.0675
- RMSE on MOS 0-95 scale: 6.4089
- Reliability mean: CLIP 0.4222, Attr 0.2393, Phys 0.3386

## Reference Comparison

| Method | SRCC | KRCC | PLCC | RMSE | Source |
|---|---:|---:|---:|---:|---|
| PAUQA-Net | 0.8745 | 0.6991 | 0.8635 | 6.0703 | PAUQA/LUIQD paper |
| AquaCLIP-QA | 0.8795 | 0.7073 | 0.8726 | 6.4089 | ours, LUIQD train/test manifest |

Note: PAUQA-Net numbers are public paper values. Verify exact protocol before claiming strict superiority.
