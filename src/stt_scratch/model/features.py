"""파형 → 특징 단계: log-mel, 속도 교란, SpecAugment, TrimTail. 모두 학습 장치에서 돈다.

배치 log-mel(LogMel), 프레임 단위로 정확한 스트리밍 log-mel(StreamingFbank), 학습용 증강(속도 교란,
SpecAugment)과 4단계 TrimTail을 둔다. 전역 정규화는 모델 안(normalize_mel)에서 한다.
프레임: 25 ms 창, 10 ms 이동, center=False, 16 kHz → n_mel = floor((samples - 400) / 160) + 1.
스트리밍도 같은 프레임 격자를 쓴다(StreamingFbank는 마지막 완성 프레임 뒤의 표본을 남겨 둔다). 320 ms
패킷(5,120표본)은 240표본이 이월된 상태에서 정확히 32프레임을 내므로, 스트림의 첫 패킷에는
FIRST_PACKET_EXTRA = 240표본을 더 실어야 한다(5,360). 그러지 않으면 첫 인코더 청크가 두 번째 패킷까지
기다린다.
"""

# [설계 이유] 학습·평가·서비스가 같은 LogMel 한 구현을 쓰므로 오프라인과 스트리밍의 특징이 프레임 단위로
# 같다(tests/test_model.py test_streaming_fbank_matches_offline). 증강도 학습 장치(GPU)에서 돌려 데이터 로더를
# 가볍게 둔다(data.collate는 파형 목록만 넘긴다).
# [연결] train.prepare_batch(LogMel·speed_perturb·trim_tail),
# conformer.ChunkConformerCTC.forward(spec_augment), evaluate.decode_streaming과
# service.session(StreamingFbank, FIRST_PACKET_EXTRA).

from __future__ import annotations

# torchaudio: MelSpectrogram과 resample을 검증된 구현으로 쓰고, GPU 텐서에 그대로 적용하려고 가져온다.
import torch
import torchaudio

# 특징 규격: 16 kHz, 창 400표본(25 ms), 이동 160표본(10 ms), 80 mel. 서브샘플링 4배 뒤 인코더는 25 fps(40
# ms)가 된다.
# LOG_EPS: log(0)을 막는 작은 값. 무음·패딩 구간의 log-mel은 log(1e-6) 근처가 되며, trim_tail도 이 값을 패딩
# 값으로 쓴다.
SAMPLE_RATE = 16000
N_FFT = 400
HOP = 160
N_MELS = 80
LOG_EPS = 1e-6
# 첫 패킷에만 붙이는 240표본(창 길이 400 − 이동 160). 첫 패킷이 5,120표본이면 프레임이 30개만 나오고 첫 인코더
# 청크(32프레임)가 다음 패킷까지 320 ms 늦어진다. 5,360표본을 보내면 첫 패킷부터 32프레임이 나오고 240표본이
# 이월되어 이후 5,120표본 패킷마다 정확히 32프레임이 나온다(규칙 변경 9).
FIRST_PACKET_EXTRA = N_FFT - HOP


# center=False 조건의 mel 프레임 수: floor((표본 수 − 400) / 160) + 1, 창 하나도 안 차면 0. StreamingFbank,
# 테스트, scripts/prepare_data.py의 CTC 길이 계산과 같은 식이다.
def n_mel_frames(n_samples: int) -> int:
    return max(0, (n_samples - N_FFT) // HOP + 1)


# center=False를 쓰는 이유: center=True는 신호 양 끝을 반사 패딩하므로 끝 쪽 프레임 값이 스트림이 어디서
# 끝나는지에 따라 달라진다. center=False면 프레임 i는 표본 160i .. 160i+399만 쓰므로 음성이 패킷으로 나뉘어
# 들어와도 오프라인과 같은 값이 나온다.
# 정규화(평균·표준편차)는 여기서 하지 않고 모델 안(normalize_mel)에서 하며, 그 통계는 config와 버퍼로
# 체크포인트에 함께 저장된다.
class LogMel(torch.nn.Module):
    """패딩된 파형 배치의 log-mel. (B, T, 80)과 발화별 유효 프레임 수를 돌려준다."""

    def __init__(self) -> None:
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=N_FFT,
            win_length=N_FFT,
            hop_length=HOP,
            n_mels=N_MELS,
            center=False,
            power=2.0,
        )

    def forward(self, wav: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # 학습·평가는 fp16 autocast 안에서 이 모듈을 부르지만, 파워 스펙트럼은 값의 범위가 넓어 fp16에서
        # 넘치거나 0으로 사라질 수 있으므로 autocast를 끄고 fp32로 계산한다.
        with torch.autocast(device_type=wav.device.type, enabled=False):
            mel = self.mel(wav.float())  # (B, 80, T)
            mel = torch.log(mel + LOG_EPS).transpose(1, 2)
        # 배치 텐서는 가장 긴 발화 길이로 계산되므로, 각 발화의 실제 길이로 유효 프레임 수를 따로 낸다. 그
        # 뒤의 프레임은 0 패딩이 섞인 것이라 모델이 마스크로 무시한다.
        frames = torch.clamp((lengths - N_FFT) // HOP + 1, min=0)
        return mel, frames


# 패킷 단위로 들어오는 음성에서 '지금까지 완성된 프레임'만 내보낸다. 같은 LogMel을 쓰므로 프레임 값이
# 오프라인 계산과 같다.
# 호출 위치: evaluate.decode_streaming(실제 캐시 스트리밍 평가), service.session.StreamSession(서비스).
class StreamingFbank:
    """프레임 단위로 정확한 스트리밍 log-mel.

    마지막 완성 프레임 다음 프레임이 시작하는 표본부터(240~399표본, 스트림이 160표본 격자에 맞춰지면 정확히
    240)를 carry로 남겨 다음 패킷과 이어 붙이므로, 청크의 프레임이 오프라인 프레임과 같다.
    """

    def __init__(self, logmel: LogMel) -> None:
        self.logmel = logmel
        self.carry = torch.zeros(0)

    # 스트림을 새로 시작할 때 carry를 비운다(세션 flush 뒤). 끝점에서는 부르지 않는다: carry는 다음 세그먼트의
    # 음성이다(규칙 변경 14).
    def reset(self) -> None:
        self.carry = torch.zeros(0)

    def push(self, samples: torch.Tensor) -> torch.Tensor:
        """samples: 1차원 float 파형 조각. 지금까지 완성된 새 프레임 (T_new, 80)을 돌려준다."""
        # 이월 표본 + 새 표본으로 만들 수 있는 완성 프레임 수 n을 센다. 아직 창 하나(400표본)도 안 되면 전부
        # carry로 남기고 빈 (0, 80)을 돌려준다.
        buf = torch.cat([self.carry.to(samples.device), samples])
        n = n_mel_frames(buf.numel())
        if n == 0:
            self.carry = buf
            return buf.new_zeros((0, N_MELS))
        # n개 프레임에 꼭 필요한 표본(마지막 프레임 끝까지)만 LogMel에 넣어 오프라인과 같은 프레임 n개를 얻고,
        # 프레임 n이 시작할 위치(n × 160)부터를 다음 호출로 넘긴다. 이월 길이는 240~399표본이며 160표본 격자에
        # 맞춰지면 항상 240이다.
        used = (n - 1) * HOP + N_FFT
        mel, _ = self.logmel(buf[:used].unsqueeze(0), torch.tensor([used], device=buf.device))
        self.carry = buf[n * HOP :]
        return mel[0]


# 4단계 방출 지연 실험의 증강이다. 끝이 잘린 음성으로도 모든 토큰을 내도록 학습시켜 토큰 방출을 앞당기려는
# 방법이다.
# 입력: mel (B, T, 80), 유효 프레임 수, 최대 절단 프레임 수(train.py가 --trimtail-ms ÷ 10으로 넘김, 4단계는
# 200 ms = 20프레임), 발화별 최소 프레임 수(CTC 길이 조건), Python random 객체. 출력: 잘린 mel과 새 유효
# 프레임 수.
# 잘린 프레임을 패딩 값으로 덮는 이유: 유효 길이만 줄이면 합성곱이 경계 근처에서 잘린 프레임 값을 읽을 수
# 있다. 값 자체를 지워 잘린 소리가 합성곱을 거쳐 모델에 들어가지 않게 한다.
# 결과: TrimTail이 채택됐지만 FTD P50 감소는 10 ms라 H4(80 ms)는 미충족이었다(docs/experiments.md 4단계 결과).
def trim_tail(
    mel: torch.Tensor, frames: torch.Tensor, max_trim: int, min_frames: torch.Tensor, rng
) -> tuple[torch.Tensor, torch.Tensor]:
    """TrimTail(arXiv 2211.00522): 발화마다 끝의 mel 프레임을 U{0..max_trim}개 잘라 낸다.

    남는 길이는 min_frames(CTC 길이 조건) 아래로 내려가지 않는다. 잘린 프레임은 패딩 값 log(LOG_EPS)로
    덮어써, 잘린 소리가 합성곱을 거쳐 모델에 들어가지 않게 한다.
    """
    out = mel.clone()
    new = frames.clone()
    pad_value = float(torch.log(torch.tensor(LOG_EPS)))
    for i in range(mel.shape[0]):
        # 절단량은 U{0..max_trim}에서 뽑되, 남는 프레임이 CTC 최소 프레임 아래로 내려가지 않게 상한을 둔다(CTC
        # 손실이 무한대가 되는 것을 막는다).
        cut = min(rng.randint(0, max_trim), max(0, int(frames[i]) - int(min_frames[i])))
        if cut > 0:
            new[i] = frames[i] - cut
            out[i, int(new[i]) :] = pad_value
    return out, new


# 속도 교란: 원래 16 kHz 파형을 16 kHz × factor로 녹음된 것처럼 보고 16 kHz로 다시 샘플링한다.
# 길이·템포·음높이가 함께 바뀐다(factor 0.9 → 약 1.11배 길고 낮은 소리).
# 호출 위치: train.prepare_batch가 발화마다 (0.9, 1.0, 1.1) 중 하나를 골라 GPU에서 부른다. 배치 예산의 10 %
# 여유는 data.DurationBucketSampler에 있다.
def speed_perturb(wav: torch.Tensor, factor: float) -> torch.Tensor:
    """리샘플링으로 템포·음높이를 바꾼다(ESPnet/icefall 관행). factor 0.9면 음성이 길어진다."""
    if factor == 1.0:
        return wav
    return torchaudio.functional.resample(wav, orig_freq=int(SAMPLE_RATE * factor), new_freq=SAMPLE_RATE)


# 입력은 정규화된 mel이라 0으로 지우는 것은 '전역 평균값'으로 덮는 것과 같다.
# 호출 위치: conformer.ChunkConformerCTC.forward(augment=True이고 train 모드일 때만). CR-CTC(3단계 (d))는 같은
# 배치를 두 번 이어 붙여 넣으므로 행마다 마스크를 따로 뽑는 이 구현 덕분에 두 뷰의 마스크가 달라진다.
# generator를 주지 않으면 torch 전역 난수를 쓴다(학습 시드로 재현). 테스트는 generator를 넘겨 결정적으로
# 돌린다.
def spec_augment(
    mel: torch.Tensor,
    frames: torch.Tensor,
    freq_masks: int,
    freq_width: int,
    time_masks: int,
    time_ratio: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """SpecAugment: 발화마다 무작위 주파수 대역과 시간 구간을 0으로 지운다(정규화된 mel에 건다).

    폭 w ~ U{0..max}, 시작 ~ U{0..size-w}(양끝 포함)이라 마지막 bin·프레임도 가려질 수 있다.
    """
    b, t, f = mel.shape
    out = mel.clone()
    for i in range(b):
        valid = int(frames[i])
        # 주파수 마스크: 폭 w ~ U{0..27}, 시작 f0 ~ U{0..80−w}(양끝 포함이라 마지막 bin도 가려질 수 있다).
        for _ in range(freq_masks):
            w = int(torch.randint(0, min(f, freq_width) + 1, (1,), generator=generator))
            f0 = int(torch.randint(0, f - w + 1, (1,), generator=generator))
            out[i, :, f0 : f0 + w] = 0.0
        # 시간 마스크: 폭 상한을 유효 길이의 5 %로 두어 짧은 발화가 통째로 지워지지 않게 하고, 시작 위치는
        # 유효 프레임 안에서만 뽑는다(패딩 구간은 이미 무시되므로 가려도 의미가 없다).
        max_w = max(1, int(valid * time_ratio))
        for _ in range(time_masks):
            w = int(torch.randint(0, min(valid, max_w) + 1, (1,), generator=generator))
            t0 = int(torch.randint(0, max(0, valid - w) + 1, (1,), generator=generator))
            out[i, t0 : t0 + w, :] = 0.0
    return out
