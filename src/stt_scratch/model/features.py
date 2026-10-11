"""Log-mel features, global normalization, speed perturbation and SpecAugment, all on the training device.

Frames: 25 ms window, 10 ms hop, center=False, 16 kHz -> n_mel = floor((samples - 400) / 160) + 1.
Streaming uses the same frame grid (StreamingFbank keeps the samples after the last complete frame). A
320 ms packet (5,120 samples) yields exactly 32 frames once 240 samples are carried over, so the first
packet of a stream should carry FIRST_PACKET_EXTRA = 240 extra samples (5,360); otherwise the first
encoder chunk waits for the second packet.
"""

from __future__ import annotations

import torch
import torchaudio

SAMPLE_RATE = 16000
N_FFT = 400
HOP = 160
N_MELS = 80
LOG_EPS = 1e-6
FIRST_PACKET_EXTRA = N_FFT - HOP  # 240 samples: makes every 320 ms packet (5,120 samples) carry 32 frames


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
    """Frame-exact streaming log-mel: carries the samples after the last complete frame (240..399, exactly
    240 once the stream is on the 160-sample grid) so the chunk frames equal the offline frames."""

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


def trim_tail(
    mel: torch.Tensor, frames: torch.Tensor, max_trim: int, min_frames: torch.Tensor, rng
) -> tuple[torch.Tensor, torch.Tensor]:
    """TrimTail (arXiv 2211.00522): drop U{0..max_trim} trailing mel frames per utterance, never below
    min_frames (the CTC length condition). The dropped frames are overwritten with the padding value
    log(LOG_EPS) so that no trimmed audio reaches the model through the convolutions."""
    out = mel.clone()
    new = frames.clone()
    pad_value = float(torch.log(torch.tensor(LOG_EPS)))
    for i in range(mel.shape[0]):
        cut = min(rng.randint(0, max_trim), max(0, int(frames[i]) - int(min_frames[i])))
        if cut > 0:
            new[i] = frames[i] - cut
            out[i, int(new[i]) :] = pad_value
    return out, new


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
    """Zero out random frequency bands and time spans per utterance (masks the normalized mel).

    Width w ~ U{0..max}, start ~ U{0..size-w} inclusive, so the last bin/frame can be masked too."""
    b, t, f = mel.shape
    out = mel.clone()
    for i in range(b):
        valid = int(frames[i])
        for _ in range(freq_masks):
            w = int(torch.randint(0, min(f, freq_width) + 1, (1,), generator=generator))
            f0 = int(torch.randint(0, f - w + 1, (1,), generator=generator))
            out[i, :, f0 : f0 + w] = 0.0
        max_w = max(1, int(valid * time_ratio))
        for _ in range(time_masks):
            w = int(torch.randint(0, min(valid, max_w) + 1, (1,), generator=generator))
            t0 = int(torch.randint(0, max(0, valid - w) + 1, (1,), generator=generator))
            out[i, t0 : t0 + w, :] = 0.0
    return out
