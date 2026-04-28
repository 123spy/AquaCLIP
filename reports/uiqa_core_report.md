# UWIQA + UIQD Core Benchmark

Protocol: UWIQA and UIQD official/local test split only. External IQA scores are evaluated as NR scores; SRCC uses the method direction and PLCC/NRMSE use linear mapping to normalized MOS.

## Mean Comparison
| Category | Method | Status | N | SRCC | PLCC | NRMSE | Time(s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Ours | AquaCLIP-QA | completed | 1441 | 0.8949 | 0.9008 | 0.0830 | 19.2 |
| Underwater-specific UIQA | QUIQ | partial_existing | 645 | 0.7273 | 0.6947 | 0.1572 | 2613.4 |
| Deep NR-IQA | MUSIQ | completed_existing | 1441 | 0.6984 | 0.7169 | 0.1437 | 276.2 |
| Deep NR-IQA | TOPIQ-NR | completed_existing | 1441 | 0.6589 | 0.6788 | 0.1491 | 202.9 |
| VLM / CLIP-IQA | LIQE | completed_existing | 1429 | 0.6373 | 0.6928 | 0.1510 | 60.6 |
| Traditional underwater metric | UCIQE | completed | 1441 | 0.6223 | 0.6165 | 0.1683 | 109.1 |
| Deep NR-IQA | DBCNN | completed | 1441 | 0.6205 | 0.6559 | 0.1570 | 448.2 |
| Deep NR-IQA | TReS | completed | 1441 | 0.6152 | 0.6338 | 0.1515 | 939.4 |
| Traditional NR-IQA | NIQE | completed | 1441 | 0.6014 | 0.5254 | 0.1747 | 70.6 |
| VLM / CLIP-IQA | CLIP-IQA classic | completed_existing | 1441 | 0.6013 | 0.6432 | 0.1598 | 0.0 |
| Deep NR-IQA | HyperIQA | completed | 1441 | 0.5936 | 0.6489 | 0.1561 | 81.2 |
| VLM / CLIP-IQA | QualiCLIP | completed | 1441 | 0.5816 | 0.6381 | 0.1557 | 288.7 |
| VLM / CLIP-IQA | QualiCLIP+ | completed | 1441 | 0.5811 | 0.6269 | 0.1602 | 264.0 |
| Traditional NR-IQA | BRISQUE | completed | 1441 | 0.5655 | 0.5495 | 0.1756 | 46.9 |
| Deep NR-IQA | PaQ-2-PiQ | completed | 1441 | 0.5598 | 0.4950 | 0.1846 | 94.7 |
| VLM / CLIP-IQA | CLIP-IQA+ ViT-L/14 512 | partial | 675 | 0.5454 | 0.5766 | 0.1671 | 5901.5 |
| VLM / CLIP-IQA | CLIP-IQA pyiqa | completed_existing | 1441 | 0.5188 | 0.5604 | 0.1780 | 149.3 |
| VLM / CLIP-IQA | MA-CLIP | completed_with_failures | 1429 | 0.5178 | 0.5992 | 0.1713 | 265.6 |
| Deep NR-IQA | ARNIQA-SPAQ | completed | 1441 | 0.4578 | 0.4799 | 0.1850 | 211.9 |
| Deep NR-IQA | MANIQA | completed_existing | 1441 | 0.4341 | 0.4934 | 0.1872 | 1136.7 |
| Traditional underwater metric | UIQM | completed | 1441 | 0.2564 | 0.3126 | 0.2048 | 292.9 |

## Notes
- AquaCLIP-QA time here is cached-feature training/evaluation wall time observed in this run, not end-to-end feature extraction time.
- Traditional metrics and PUIQA-style handcrafted features are CPU-bound.
- CLIP-IQA+ ViT-L/14 and QUIQ are kept as partial runs because full evaluation was too slow in this environment.
- PIGUIQA code URL was unavailable; PUIQA official code was downloaded and a reproduction runner was written, but full feature extraction did not finish within one hour.
