# control-avg10 (fp16-autocast, left 128, checkpoint outputs/stage4/control/avg10.pt)

| set | mode | CER_ns hangul-only % | 95 % interval | CER_ns all % | CER_ws hangul-only % | ratio |
|---|---|---|---|---|---|---|
| dev_mix | full | 22.37 | 21.24-23.53 | 23.70 | 19.98 |  |
| dev_mix | chunk8_320ms | 26.81 | 25.54-28.12 | 27.83 | 24.17 | 1.198 |
| dev_mix | chunk16_640ms | 25.47 | 24.22-26.67 | 26.61 | 22.88 | 1.138 |
| dev_mix | head layer6 (full) | 22.69 | | 23.87 | | |
| dev_h | full | 16.32 | 15.78-16.85 | 16.32 | 15.32 |  |
| dev_h | chunk8_320ms | 18.58 | 18.03-19.15 | 18.58 | 17.71 | 1.139 |
| dev_h | chunk16_640ms | 17.80 | 17.25-18.35 | 17.80 | 16.81 | 1.091 |
| dev_h | head layer6 (full) | 12.16 | | 12.16 | | |

train sample (1000 utts, seed 0, full context, no augmentation): CER_ns 0.30 %, ratio to dev_h full 0.018 (overfitting flag if < 0.5)

fp16 streaming check (chunk8 left128, 200 dev_h utts, 42969 frames): max |dlog p| 6.75e+00, max |dp| 6.91e-01, argmax agreement 99.979 %

stability: attention logit std per block [2.9, 1.9, 1.6, 1.9, 2.2, 1.9, 1.8, 1.5, 2.3, 2.0, 2.2, 2.5], QK-norm max |gain| q [1.39, 1.33, 1.27, 1.58, 1.79, 1.33, 1.43, 1.25, 1.61, 1.51, 1.58, 1.63]
