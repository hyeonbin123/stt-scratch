| run | LR | 이어 학습 | epoch | 학습 계산 h | 건너뛴 step | NaN | 최저 scale(step 300 이후) | 발산 |
|---|---|---|---|---|---|---|---|---|
| stage1-attempt3/lr0.001_seed1 | 0.001 |  | 5 | 0.43 | 10 | 0 | 64 | - |
| stage1-attempt3/lr0.001_seed2 | 0.001 |  | 5 | 0.42 | 9 | 0 | 128 | - |
| stage1-attempt3/lr0.001_seed3 | 0.001 |  | 5 | 0.42 | 10 | 0 | 64 | - |
| stage1-attempt3/lr0.002_seed1 | 0.002 |  | 5 | 0.43 | 10 | 0 | 64 | - |
| stage1-attempt3/lr0.002_seed2 | 0.002 |  | 5 | 0.42 | 9 | 0 | 128 | - |
| stage1-attempt3/lr0.002_seed3 | 0.002 |  | 5 | 0.43 | 10 | 0 | 64 | - |
| stage2/runA | 0.002 |  | 100 | 8.92 | 34 | 0 | 64 | - |
| stage2/runB | 0.002 |  | 3 | 0.27 | 10 | 0 | 64 | dev_loss_not_decreasing |
| stage2/runB-lr1e-3 | 0.001 |  | 100 | 7.60 | 33 | 0 | 64 | - |
| stage3/a | 0.002 |  | 100 | 7.16 | 34 | 0 | 64 | - |
| stage3/b | 0.002 |  | 100 | 15.09 | 35 | 0 | 32 | - |
| stage3/d | 0.002 |  | 3 | 0.41 | 10 | 0 | 64 | dev_loss_not_decreasing |
| stage3/d-lr0.0005 | 0.0005 |  | 49 | 7.29 | 36 | 0 | 64 | - |
| stage3/d-lr0.001 | 0.001 |  | 3 | 0.44 | 10 | 0 | 64 | dev_loss_not_decreasing |
| stage4/control | 0.001 | 예 | 15 | 1.39 | 9 | 0 | 256 | - |
| stage4/peakfirst | 0.001 | 예 | 15 | 1.14 | 9 | 0 | 512 | - |
| stage4/trimtail | 0.001 | 예 | 15 | 1.26 | 10 | 0 | 512 | - |

학습 계산 합계 53.52 h (epoch 학습 시간만; dev 평가·일시 중지·측정·재개로 버린 구간 제외), 발산 3런
