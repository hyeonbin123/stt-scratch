# runA-avg10 (fp16-autocast, left 128, checkpoint outputs/stage2/runA/avg10.pt)

| set | mode | CER_ns hangul-only % | 95 % interval | CER_ns all % | CER_ws hangul-only % | ratio |
|---|---|---|---|---|---|---|
| dev_mix | full | 22.05 | 20.93-23.18 | 23.33 | 19.59 |  |
| dev_mix | chunk4_160ms | 28.04 | 26.81-29.38 | 29.00 | 25.27 | 1.272 |
| dev_mix | chunk8_320ms | 25.99 | 24.81-27.19 | 27.06 | 23.37 | 1.179 |
| dev_mix | chunk16_640ms | 24.73 | 23.54-25.93 | 25.88 | 22.07 | 1.121 |
| dev_mix | chunk32_1280ms | 23.58 | 22.45-24.75 | 24.85 | 20.98 | 1.069 |
| dev_mix | head layer3 (full) | 26.71 | | 27.84 | | |
| dev_mix | head layer6 (full) | 21.89 | | 23.11 | | |
| dev_mix | head layer9 (full) | 21.35 | | 22.68 | | |
| dev_h | full | 17.83 | 17.28-18.37 | 17.83 | 16.25 |  |
| dev_h | chunk4_160ms | 21.98 | 21.44-22.56 | 21.98 | 20.29 | 1.233 |
| dev_h | chunk8_320ms | 20.54 | 19.96-21.09 | 20.54 | 18.79 | 1.152 |
| dev_h | chunk16_640ms | 19.57 | 19.01-20.13 | 19.57 | 17.85 | 1.097 |
| dev_h | chunk32_1280ms | 18.83 | 18.30-19.38 | 18.83 | 17.20 | 1.056 |
| dev_h | head layer3 (full) | 13.96 | | 13.96 | | |
| dev_h | head layer6 (full) | 12.18 | | 12.18 | | |
| dev_h | head layer9 (full) | 14.07 | | 14.07 | | |

train sample (1000 utts, seed 0, full context, no augmentation): CER_ns 0.14 %, ratio to dev_h full 0.008 (overfitting flag if < 0.5)

fp16 streaming check (chunk8 left128, 200 dev_h utts, 42969 frames): max |dlog p| 2.27e+00, max |dp| 1.72e-01, argmax agreement 99.991 %

stability: attention logit std per block [2.3, 2.2, 2.2, 1.9, 2.3, 2.0, 2.0, 1.8, 1.4, 2.0, 2.2, 1.7], QK-norm max |gain| q [1.31, 1.33, 1.2, 1.33, 1.43, 1.45, 1.6, 1.89, 1.52, 1.7, 1.49, 1.41]
