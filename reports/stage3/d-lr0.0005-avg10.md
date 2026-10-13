# d-lr0.0005-avg10 (fp16-autocast, left 128, checkpoint outputs\stage3\d-lr0.0005/avg10.pt)

| set | mode | CER_ns hangul-only % | 95 % interval | CER_ns all % | CER_ws hangul-only % | ratio |
|---|---|---|---|---|---|---|
| dev_mix | full | 22.24 | 21.12-23.44 | 23.54 | 19.69 |  |
| dev_mix | chunk8_320ms | 26.83 | 25.61-28.14 | 27.87 | 23.86 | 1.206 |
| dev_mix | chunk16_640ms | 25.25 | 24.04-26.52 | 26.37 | 22.37 | 1.135 |
| dev_mix | head layer3 (full) | 30.05 | | 30.83 | | |
| dev_mix | head layer6 (full) | 22.71 | | 23.79 | | |
| dev_mix | head layer9 (full) | 20.99 | | 22.23 | | |
| dev_h | full | 26.49 | 25.89-27.12 | 26.49 | 23.19 |  |
| dev_h | chunk8_320ms | 27.54 | 26.92-28.16 | 27.54 | 24.30 | 1.040 |
| dev_h | chunk16_640ms | 27.25 | 26.66-27.90 | 27.25 | 23.96 | 1.029 |
| dev_h | head layer3 (full) | 15.46 | | 15.46 | | |
| dev_h | head layer6 (full) | 11.32 | | 11.32 | | |
| dev_h | head layer9 (full) | 12.77 | | 12.77 | | |

train sample (1000 utts, seed 0, full context, no augmentation): CER_ns 0.46 %, ratio to dev_h full 0.017 (overfitting flag if < 0.5)
