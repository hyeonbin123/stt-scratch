# b-avg10 (fp16-autocast, left 128, checkpoint outputs\stage3\b/avg10.pt)

| set | mode | CER_ns hangul-only % | 95 % interval | CER_ns all % | CER_ws hangul-only % | ratio |
|---|---|---|---|---|---|---|
| dev_mix | full | 22.49 | 21.27-23.70 | 23.72 | 19.98 |  |
| dev_mix | chunk8_320ms | 26.42 | 25.09-27.72 | 27.48 | 23.62 | 1.175 |
| dev_mix | chunk16_640ms | 25.09 | 23.83-26.32 | 26.18 | 22.23 | 1.116 |
| dev_mix | head layer3 (full) | 26.47 | | 27.47 | | |
| dev_mix | head layer6 (full) | 23.16 | | 24.30 | | |
| dev_mix | head layer9 (full) | 22.39 | | 23.58 | | |
| dev_h | full | 16.97 | 16.42-17.48 | 16.97 | 15.51 |  |
| dev_h | chunk8_320ms | 19.17 | 18.58-19.73 | 19.17 | 17.72 | 1.129 |
| dev_h | chunk16_640ms | 18.33 | 17.75-18.85 | 18.33 | 16.80 | 1.080 |
| dev_h | head layer3 (full) | 13.52 | | 13.52 | | |
| dev_h | head layer6 (full) | 12.87 | | 12.87 | | |
| dev_h | head layer9 (full) | 14.25 | | 14.25 | | |

train sample (1000 utts, seed 0, full context, no augmentation): CER_ns 0.10 %, ratio to dev_h full 0.006 (overfitting flag if < 0.5)
