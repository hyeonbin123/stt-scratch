# runB-lr1e-3-avg10 (fp16-autocast, left 128, checkpoint outputs/stage2/runB-lr1e-3/avg10.pt)

| set | mode | CER_ns hangul-only % | 95 % interval | CER_ns all % | CER_ws hangul-only % | ratio |
|---|---|---|---|---|---|---|
| dev_mix | full | 22.90 | 21.78-24.07 | 24.15 | 20.02 |  |
| dev_mix | chunk4_160ms | 98.69 | 97.06-100.38 | 100.03 | 86.76 | 4.310 |
| dev_mix | chunk8_320ms | 94.67 | 93.22-96.08 | 95.63 | 84.14 | 4.134 |
| dev_mix | chunk16_640ms | 77.16 | 75.89-78.43 | 78.48 | 67.61 | 3.370 |
| dev_mix | chunk32_1280ms | 52.26 | 51.22-53.33 | 54.25 | 45.06 | 2.282 |
| dev_mix | head layer3 (full) | 25.05 | | 25.99 | | |
| dev_mix | head layer6 (full) | 21.01 | | 22.37 | | |
| dev_mix | head layer9 (full) | 21.60 | | 22.89 | | |
| dev_h | full | 20.37 | 19.79-20.93 | 20.37 | 18.24 |  |
| dev_h | chunk4_160ms | 98.18 | 97.42-99.04 | 98.18 | 85.79 | 4.819 |
| dev_h | chunk8_320ms | 96.85 | 95.85-97.84 | 96.85 | 85.98 | 4.754 |
| dev_h | chunk16_640ms | 75.05 | 74.08-75.94 | 75.05 | 66.77 | 3.684 |
| dev_h | chunk32_1280ms | 47.93 | 47.14-48.67 | 47.93 | 42.17 | 2.353 |
| dev_h | head layer3 (full) | 13.00 | | 13.00 | | |
| dev_h | head layer6 (full) | 11.94 | | 11.94 | | |
| dev_h | head layer9 (full) | 15.34 | | 15.34 | | |

train sample (1000 utts, seed 0, full context, no augmentation): CER_ns 0.03 %, ratio to dev_h full 0.001 (overfitting flag if < 0.5)

fp16 streaming check (chunk8 left128, 200 dev_h utts, 42969 frames): max |dlog p| 2.40e+00, max |dp| 1.92e-01, argmax agreement 99.935 %

stability: attention logit std per block [2.2, 1.8, 2.5, 1.5, 2.0, 2.0, 1.9, 1.4, 1.5, 1.8, 1.6, 1.9], QK-norm max |gain| q [1.36, 1.08, 1.44, 1.12, 1.57, 1.44, 1.44, 1.41, 1.2, 1.46, 1.28, 1.38]
