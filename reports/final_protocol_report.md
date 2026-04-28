# Final Protocol Results

Train datasets: LUIQD + UIQD + UWIQA.

Test datasets: local datasets with prepared manifests/features outside the training set.

CPU-only or unavailable methods are excluded from this GPU-only protocol.

## Mean Comparison

| category | display_name | method | n | srcc | plcc_linear | nrmse_linear | elapsed_sec |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Ours | AquaCLIP-QA | AquaCLIP-QA | 11500 | 0.4421 | 0.4599 | 0.1974 | 0.0076 |
| Ours | AquaCLIP-QA-E2E | AquaCLIP-QA-E2E | 11500 | 0.4421 | 0.4599 | 0.1974 | 229.6236 |
| VLM-IQA | LIQE | liqe | 11461 | 0.3265 | 0.3344 | 0.2104 | 162.3364 |
| Deep NR-IQA | MUSIQ | musiq | 11500 | 0.2683 | 0.2533 | 0.2167 | 171.8911 |
| Deep NR-IQA | TOPIQ-NR | topiq_nr | 11500 | 0.2389 | 0.2385 | 0.2180 | 203.1847 |
| Deep NR-IQA | HyperIQA | hyperiqa | 11500 | 0.2307 | 0.2294 | 0.2186 | 253.5372 |
| Deep NR-IQA | TReS | tres | 2700 | 0.2090 | 0.1991 | 0.2313 | 729.1661 |
| Deep NR-IQA | DBCNN | dbcnn | 11500 | 0.1931 | 0.1868 | 0.2195 | 220.5264 |
| VLM-IQA | CLIP-IQA | clipiqa | 11500 | 0.1890 | 0.1915 | 0.2208 | 253.0249 |
| Underwater UIQA | ATUIQP | ATUIQP | 11500 | 0.1676 | 0.1825 | 0.2170 | 155.9655 |
| Underwater metric | UCIQE | UCIQE | 11500 | 0.1669 | 0.1909 | 0.2145 | 145.5511 |
| Deep NR-IQA | PaQ-2-PiQ | paq2piq | 11500 | 0.1667 | 0.1426 | 0.2217 | 96.6254 |
| Traditional NR-IQA | NIQE | niqe | 11500 | 0.1318 | 0.1270 | 0.2231 | 163.8741 |
| Underwater metric | UIQM | UIQM | 11500 | 0.0817 | 0.1502 | 0.2222 | 413.5253 |

## Per-Dataset Comparison

### saud2

| category | display_name | method | n | srcc | plcc_linear | nrmse_linear | elapsed_sec |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Ours | AquaCLIP-QA | AquaCLIP-QA | 2400 | 0.2314 | 0.2477 | 0.2270 | 0.0045 |
| Ours | AquaCLIP-QA-E2E | AquaCLIP-QA-E2E | 2400 | 0.2314 | 0.2477 | 0.2270 | 224.6532 |
| VLM-IQA | LIQE | liqe | 2400 | 0.1921 | 0.2061 | 0.2292 | 127.4951 |
| VLM-IQA | CLIP-IQA | clipiqa | 2400 | 0.1422 | 0.1438 | 0.2318 | 163.6779 |
| Deep NR-IQA | MUSIQ | musiq | 2400 | 0.1323 | 0.1281 | 0.2323 | 109.1066 |
| Deep NR-IQA | TReS | tres | 1800 | 0.1282 | 0.1344 | 0.2323 | 972.6684 |
| Deep NR-IQA | TOPIQ-NR | topiq_nr | 2400 | 0.1267 | 0.1373 | 0.2320 | 107.7886 |
| Deep NR-IQA | HyperIQA | hyperiqa | 2400 | 0.1226 | 0.1259 | 0.2324 | 209.8561 |
| Underwater UIQA | ATUIQP | ATUIQP | 2400 | 0.1087 | 0.0929 | 0.2333 | 116.3508 |
| Deep NR-IQA | PaQ-2-PiQ | paq2piq | 2400 | 0.0887 | 0.1061 | 0.2329 | 61.1231 |
| Deep NR-IQA | DBCNN | dbcnn | 2400 | 0.0650 | 0.0762 | 0.2336 | 115.0599 |
| Traditional NR-IQA | NIQE | niqe | 2400 | 0.0100 | 0.0659 | 0.2338 | 119.8851 |
| Underwater metric | UCIQE | UCIQE | 2400 | -0.0515 | 0.0013 | 0.2343 | 82.5539 |
| Underwater metric | UIQM | UIQM | 2400 | -0.0691 | 0.2094 | 0.2291 | 177.2753 |

### sota

| category | display_name | method | n | srcc | plcc_linear | nrmse_linear | elapsed_sec |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Ours | AquaCLIP-QA | AquaCLIP-QA | 7200 | 0.4564 | 0.4800 | 0.1841 | 0.0140 |
| Ours | AquaCLIP-QA-E2E | AquaCLIP-QA-E2E | 7200 | 0.4564 | 0.4800 | 0.1841 | 547.8533 |
| VLM-IQA | LIQE | liqe | 7200 | 0.2771 | 0.2746 | 0.2018 | 413.3784 |
| Deep NR-IQA | MUSIQ | musiq | 7200 | 0.2211 | 0.2117 | 0.2051 | 460.3378 |
| Deep NR-IQA | HyperIQA | hyperiqa | 7200 | 0.1984 | 0.1911 | 0.2060 | 638.4023 |
| Deep NR-IQA | TOPIQ-NR | topiq_nr | 7200 | 0.1877 | 0.1822 | 0.2064 | 567.8388 |
| Deep NR-IQA | DBCNN | dbcnn | 7200 | 0.1299 | 0.1047 | 0.2087 | 620.1740 |
| VLM-IQA | CLIP-IQA | clipiqa | 7200 | 0.1264 | 0.1329 | 0.2080 | 685.3622 |
| Underwater metric | UIQM | UIQM | 7200 | 0.1163 | 0.0770 | 0.2093 | 1104.0602 |
| Traditional NR-IQA | NIQE | niqe | 7200 | 0.0972 | 0.0839 | 0.2092 | 424.9824 |
| Deep NR-IQA | PaQ-2-PiQ | paq2piq | 7200 | 0.0676 | 0.0531 | 0.2096 | 260.9196 |
| Underwater metric | UCIQE | UCIQE | 7200 | 0.0161 | 0.0077 | 0.2099 | 402.7438 |
| Underwater UIQA | ATUIQP | ATUIQP | 7200 | -0.0012 | 0.0015 | 0.2099 | 410.6184 |

### uid2021

| category | display_name | method | n | srcc | plcc_linear | nrmse_linear | elapsed_sec |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Ours | AquaCLIP-QA | AquaCLIP-QA | 900 | 0.5858 | 0.6029 | 0.1904 | 0.0024 |
| Ours | AquaCLIP-QA-E2E | AquaCLIP-QA-E2E | 900 | 0.5858 | 0.6029 | 0.1904 | 58.7809 |
| Underwater metric | UCIQE | UCIQE | 900 | 0.5035 | 0.5433 | 0.2004 | 24.1076 |
| VLM-IQA | LIQE | liqe | 900 | 0.4889 | 0.5027 | 0.2063 | 45.2403 |
| Underwater UIQA | ATUIQP | ATUIQP | 900 | 0.4563 | 0.4856 | 0.2087 | 47.7710 |
| Deep NR-IQA | MUSIQ | musiq | 900 | 0.4109 | 0.3889 | 0.2199 | 34.4888 |
| Deep NR-IQA | DBCNN | dbcnn | 900 | 0.3798 | 0.3803 | 0.2208 | 34.9250 |
| Underwater metric | UIQM | UIQM | 900 | 0.3505 | 0.2184 | 0.2329 | 55.2802 |
| Deep NR-IQA | PaQ-2-PiQ | paq2piq | 900 | 0.3430 | 0.3154 | 0.2265 | 19.0726 |
| Deep NR-IQA | TOPIQ-NR | topiq_nr | 900 | 0.3353 | 0.3328 | 0.2251 | 33.3716 |
| Deep NR-IQA | HyperIQA | hyperiqa | 900 | 0.3170 | 0.3170 | 0.2264 | 74.5883 |
| Deep NR-IQA | TReS | tres | 900 | 0.2898 | 0.2639 | 0.2302 | 485.6637 |
| VLM-IQA | CLIP-IQA | clipiqa | 900 | 0.2361 | 0.2450 | 0.2314 | 54.3196 |
| Traditional NR-IQA | NIQE | niqe | 900 | 0.1656 | 0.1377 | 0.2364 | 42.9254 |

### uied

| category | display_name | method | n | srcc | plcc_linear | nrmse_linear | elapsed_sec |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Ours | AquaCLIP-QA | AquaCLIP-QA | 1000 | 0.4950 | 0.5090 | 0.1881 | 0.0095 |
| Ours | AquaCLIP-QA-E2E | AquaCLIP-QA-E2E | 1000 | 0.4950 | 0.5090 | 0.1881 | 87.2071 |
| VLM-IQA | LIQE | liqe | 961 | 0.3480 | 0.3543 | 0.2042 | 63.2316 |
| Deep NR-IQA | MUSIQ | musiq | 1000 | 0.3087 | 0.2847 | 0.2095 | 83.6311 |
| Deep NR-IQA | TOPIQ-NR | topiq_nr | 1000 | 0.3060 | 0.3016 | 0.2084 | 103.7396 |
| Deep NR-IQA | HyperIQA | hyperiqa | 1000 | 0.2849 | 0.2835 | 0.2096 | 91.3019 |
| Traditional NR-IQA | NIQE | niqe | 1000 | 0.2543 | 0.2206 | 0.2132 | 67.7034 |
| VLM-IQA | CLIP-IQA | clipiqa | 1000 | 0.2514 | 0.2445 | 0.2119 | 108.7398 |
| Underwater metric | UCIQE | UCIQE | 1000 | 0.1994 | 0.2114 | 0.2136 | 72.7991 |
| Deep NR-IQA | DBCNN | dbcnn | 1000 | 0.1976 | 0.1858 | 0.2148 | 111.9465 |
| Deep NR-IQA | PaQ-2-PiQ | paq2piq | 1000 | 0.1674 | 0.0959 | 0.2176 | 45.3861 |
| Underwater UIQA | ATUIQP | ATUIQP | 1000 | 0.1066 | 0.1498 | 0.2161 | 49.1218 |
| Underwater metric | UIQM | UIQM | 1000 | -0.0709 | 0.0962 | 0.2175 | 317.4853 |

## Method Status

| method | category | display_name | pred_rows | summary_rows | failures | status | error |
| --- | --- | --- | --- | --- | --- | --- | --- |
| aquaclip_qa | Ours | AquaCLIP-QA | 11500 | 5 | 0 | complete |  |
| aquaclip_qa_e2e_timing | Ours | AquaCLIP-QA-E2E | 11500 | 5 | 0 | complete |  |
| atuiqp | Underwater UIQA | ATUIQP | 11500 | 5 | 0 | complete |  |
| clipiqa | VLM-IQA | CLIP-IQA | 11500 | 5 | 0 | complete |  |
| dbcnn | Deep NR-IQA | DBCNN | 11500 | 5 | 0 | complete |  |
| hyperiqa | Deep NR-IQA | HyperIQA | 11500 | 5 | 0 | complete |  |
| liqe | VLM-IQA | LIQE | 11461 | 5 | 39 | complete |  |
| musiq | Deep NR-IQA | MUSIQ | 11500 | 5 | 0 | complete |  |
| niqe | Traditional NR-IQA | NIQE | 11500 | 5 | 0 | complete |  |
| paq2piq | Deep NR-IQA | PaQ-2-PiQ | 11500 | 5 | 0 | complete |  |
| topiq_nr | Deep NR-IQA | TOPIQ-NR | 11500 | 5 | 0 | complete |  |
| tres | Deep NR-IQA | TReS | 2700 | 3 | 0 | complete |  |
| uciqe | Underwater metric | UCIQE | 11500 | 5 | 0 | complete |  |
| uiqm | Underwater metric | UIQM | 11500 | 5 | 0 | complete |  |
