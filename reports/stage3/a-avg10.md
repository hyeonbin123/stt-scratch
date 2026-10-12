# a-avg10 (fp16-autocast, left 128, checkpoint outputs\stage3\a/avg10.pt)

| set | mode | CER_ns hangul-only % | 95 % interval | CER_ns all % | CER_ws hangul-only % | ratio |
|---|---|---|---|---|---|---|
| dev_mix | full | 22.32 | 21.22-23.48 | 23.77 | 19.83 |  |
| dev_mix | chunk8_320ms | 26.61 | 25.36-27.87 | 27.70 | 23.86 | 1.193 |
| dev_mix | chunk16_640ms | 25.30 | 24.12-26.50 | 26.47 | 22.52 | 1.133 |
| dev_mix | head layer6 (full) | 22.65 | | 23.88 | | |
| dev_h | full | 16.47 | 15.92-16.99 | 16.47 | 15.43 |  |
| dev_h | chunk8_320ms | 18.66 | 18.09-19.23 | 18.66 | 17.71 | 1.133 |
| dev_h | chunk16_640ms | 17.92 | 17.36-18.47 | 17.92 | 16.87 | 1.088 |
| dev_h | head layer6 (full) | 12.33 | | 12.33 | | |

train sample (1000 utts, seed 0, full context, no augmentation): CER_ns 0.31 %, ratio to dev_h full 0.019 (overfitting flag if < 0.5)
