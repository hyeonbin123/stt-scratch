# 봉인 test (started 2026-10-14 02:10:45, checkpoint outputs/stage4/trimtail/avg10.pt sha256 6502406c19860e46e751026fb2bae00c86d0517fcff12686b20457a1dbd9884b; finished 2026-10-14 03:18:45)

checkpoint `outputs/stage4/trimtail/avg10.pt` (SHA-256 6502406c1986...), greedy, fp16 autocast; streaming = real cache streaming chunk8 left128

| set | n | full CER_ns all | full CER_ns 한글 전용 | full CER_ws 한글 전용 | streaming CER_ns 한글 전용 | streaming CER_ws 한글 전용 | 출력 불가능 문자 하한 all / 한글 전용 |
|---|---|---|---|---|---|---|---|
| test_zeroth | 457 | 3.67 (2.97-4.42) | 3.67 (2.97-4.42) | 3.57 (2.92-4.30) | 5.35 (4.55-6.26) | 5.31 (4.53-6.13) | 0.01 / 0.01 |
| test_h | 1280 | 15.55 (15.09-16.02) | 15.55 (15.09-16.02) | 14.34 (13.92-14.75) | 18.08 (17.56-18.55) | 16.81 (16.36-17.22) | 0.05 / 0.05 |
| test_fleurs | 382 | 19.73 (18.34-20.99) | 14.62 (13.47-15.89) | 12.97 (11.99-14.02) | 17.74 (16.59-19.07) | 15.90 (14.90-17.08) | 4.80 / 0.14 |
| test_cv | 472 | 28.19 (26.62-29.64) | 28.19 (26.62-29.64) | 25.39 (24.03-26.69) | 33.35 (31.63-34.92) | 30.07 (28.63-31.46) | 0.37 / 0.37 |

H7.1: {'zeroth_full_cer_ns': 3.6654914973040236, 'anchors': {'whisper-turbo zero-shot': 4.48, 'whisper-turbo + LoRA': 1.96}, 'met': True}
H7.2: {'fleurs_hangul_only': 14.621893178212586, 'fleurs_in_12_30': True, 'cv_hangul_only': 28.187988482106128, 'cv_in_15_40': True, 'test_h_above_official': True}
