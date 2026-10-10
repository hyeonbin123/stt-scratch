"""Log-mel features, global normalization, speed perturbation and SpecAugment, all on the training device.

Frames: 25 ms window, 10 ms hop, center=False, 16 kHz -> n_mel = floor((samples - 400) / 160) + 1.
Streaming uses the same frame grid: 320 ms of audio (5,120 samples) carries exactly 32 frames when the
previous 240 samples are carried over (see StreamingFbank).
"""

from __future__ import annotations

import torch
import torchaudio

SAMPLE_RATE = 16000
N_FFT = 400
HOP = 160
N_MELS = 80
LOG_EPS = 1e-6


def n_mel_frames(n_samples: int) -> int:
    return max(0, (n_samples - N_FFT) // HOP + 1)


class LogMel(torch.nn.Module):
    """Batched log-mel on padded waveforms. Returns (B, T, 80) and the valid frame count per item."""

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
        with torch.autocast(device_type=wav.device.type, enabled=False):
            mel = self.mel(wav.float())  # (B, 80, T)
            mel = torch.log(mel + LOG_EPS).transpose(1, 2)
        frames = torch.clamp((lengths - N_FFT) // HOP + 1, min=0)
        return mel, frames


class StreamingFbank:
    """Frame-exact streaming log-mel: carries the last 240 samples so chunk frames equal offline frames."""

    def __init__(self, logmel: LogMel) -> None:
        self.logmel = logmel
        self.carry = torch.zeros(0)

    def reset(self) -> None:
        self.carry = torch.zeros(0)

    def push(self, samples: torch.Tensor) -> torch.Tensor:
        """samples: 1-D float waveform chunk. Returns (T_new, 80) frames that are complete so far."""
        buf = torch.cat([self.carry.to(samples.device), samples])
        n = n_mel_frames(buf.numel())
        if n == 0:
            self.carry = buf
            return buf.new_zeros((0, N_MELS))
        used = (n - 1) * HOP + N_FFT
        mel, _ = self.logmel(buf[:used].unsqueeze(0), torch.tensor([used], device=buf.device))
        self.carry = buf[n * HOP :]
        return mel[0]


def speed_perturb(wav: torch.Tensor, factor: float) -> torch.Tensor:
    """Tempo/pitch change by resampling (ESPnet/icefall practice): factor 0.9 makes the audio longer."""
    if factor == 1.0:
        return wav
    return torchaudio.functional.resample(wav, orig_freq=int(SAMPLE_RATE * factor), new_freq=SAMPLE_RATE)


def spec_augment(
    mel: torch.Tensor,
    frames: torch.Tensor,
    freq_masks: int,
    freq_width: int,
    time_masks: int,
    time_ratio: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Zero out random frequency bands and time spans per utterance (masks the normalized mel)."""
    b, t, f = mel.shape
    out = mel.clone()
    for i in range(b):
        valid = int(frames[i])
        for _ in range(freq_masks):
            w = int(torch.randint(0, freq_width + 1, (1,), generator=generator))
            f0 = int(torch.randint(0, max(1, f - w), (1,), generator=generator))
            out[i, :, f0 : f0 + w] = 0.0
        max_w = max(1, int(valid * time_ratio))
        for _ in range(time_masks):
            w = int(torch.randint(0, max_w + 1, (1,), generator=generator))
            t0 = int(torch.randint(0, max(1, valid - w), (1,), generator=generator))
            out[i, t0 : t0 + w, :] = 0.0
    return out
