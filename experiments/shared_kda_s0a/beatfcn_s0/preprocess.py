"""Audited official PupuM2D magnitude-log-mel recipe, 24 kHz, 25 Hz targets.

Keep each 10.24 s official input at exactly 1024 mel frames, 256 time patches.
Final input is zero-padded; its final partial 40 ms cell is retained with a
truncated valid interval. Window-local padding is the published reflect392.
"""
from __future__ import annotations
import hashlib
import numpy as np
import torch
from torch.nn import functional as F


SR = 24000
NFFT = 1024
HOP = 240
NMEL = 128
PAD = 392
MEAN = -4.089994845986366
STD = 2.0242277159094813
WINDOW_SAMPLES = 245760  # 10.24 s
CELL_SAMPLES = 960     # 40 ms, four mel hops
PREPROCESS_RECIPE = {'version': 1, 'sample_rate': SR, 'n_fft': NFFT, 'hop': HOP,
    'reflect_pad': PAD, 'center': False, 'window': 'periodic Hann1024',
    'magnitude': 'sqrt(real^2+imag^2+1e-9)', 'filter': 'librosa Slaney area mel128 fmin0 fmax12000',
    'log': 'natural log(clamp_min1e-5)', 'mean': MEAN, 'std': STD, 'std_epsilon': 1e-8,
    'window_samples': WINDOW_SAMPLES, 'tail': 'zero-pad input; preserve truncated last interval',
    'aggregation': 'none: preserve 8 frequency patch tokens'}


def slaney_mel_filter():
    def hz_to_mel(f):
        f = np.asarray(f, dtype=float)
        return np.where(f >= 1000, 15 + np.log(np.maximum(f, 1e-20) / 1000) / (np.log(6.4) / 27), f / (200 / 3))
    def mel_to_hz(m):
        m = np.asarray(m)
        return np.where(m >= 15, 1000 * np.exp((np.log(6.4) / 27) * (m - 15)), (200 / 3) * m)
    hz = mel_to_hz(np.linspace(hz_to_mel(0), hz_to_mel(12000), NMEL + 2))
    freqs = np.fft.rfftfreq(NFFT, 1 / SR)
    ramps, fdiff = hz[:, None] - freqs[None, :], np.diff(hz)
    weights = np.zeros((NMEL, len(freqs)), np.float32)
    for i in range(NMEL):
        weights[i] = np.maximum(0, np.minimum(-ramps[i] / fdiff[i], ramps[i + 2] / fdiff[i + 1]))
    weights *= (2 / (hz[2:NMEL + 2] - hz[:NMEL]))[:, None]
    return torch.from_numpy(weights)


def waveform_mel(wave):
    wave = np.asarray(wave, dtype=np.float32)
    if len(wave) < 393 or len(wave) % CELL_SAMPLES:
        raise ValueError('Teacher waveform needs >=393 samples, multiple of960')
    x = F.pad(torch.from_numpy(wave)[None, None], (PAD, PAD), mode='reflect').squeeze(1)
    spec = torch.stft(x, n_fft=NFFT, hop_length=HOP, win_length=NFFT, window=torch.hann_window(NFFT), center=False, return_complex=True)
    mag = torch.sqrt(spec.real.square() + spec.imag.square() + 1e-9)
    lm = (torch.log(torch.clamp(slaney_mel_filter() @ mag, min=1e-5)) - MEAN) / (STD + 1e-8)
    mel = lm.transpose(1, 2)[:, None]
    if mel.shape[2] != len(wave) // HOP or not torch.isfinite(mel).all():
        raise ValueError('Malformed teacher mel')
    return mel

