# trimtail-avg10 (fp16-autocast, left 128, checkpoint outputs/stage4/trimtail/avg10.pt)

| set | mode | CER_ns hangul-only % | 95 % interval | CER_ns all % | CER_ws hangul-only % | ratio |
|---|---|---|---|---|---|---|
| dev_mix | full | 22.39 | 21.29-23.60 | 23.82 | 19.93 |  |
| dev_mix | chunk8_320ms | 26.89 | 25.65-28.21 | 27.93 | 24.09 | 1.201 |
| dev_mix | chunk16_640ms | 25.35 | 24.14-26.59 | 26.57 | 22.66 | 1.132 |
| dev_mix | head layer6 (full) | 22.58 | | 23.81 | | |
| dev_h | full | 16.09 | 15.54-16.61 | 16.09 | 15.20 |  |
| dev_h | chunk8_320ms | 18.76 | 18.17-19.34 | 18.76 | 17.80 | 1.166 |
| dev_h | chunk16_640ms | 17.93 | 17.37-18.47 | 17.93 | 16.84 | 1.114 |
| dev_h | head layer6 (full) | 12.26 | | 12.26 | | |

train sample (1000 utts, seed 0, full context, no augmentation): CER_ns 0.32 %, ratio to dev_h full 0.020 (overfitting flag if < 0.5)

fp16 streaming check (chunk8 left128, 200 dev_h utts, 42969 frames): max |dlog p| 1.40e+00, max |dp| 2.66e-01, argmax agreement 99.986 %

stability: attention logit std per block [2.9, 1.9, 1.5, 1.9, 2.3, 1.9, 1.9, 1.4, 2.3, 1.9, 2.1, 2.6], QK-norm max |gain| q [1.4, 1.32, 1.25, 1.64, 1.8, 1.35, 1.42, 1.22, 1.63, 1.52, 1.56, 1.62]
