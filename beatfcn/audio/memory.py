"""Frozen PupuM2D tap extraction over native windows (FIRST_CANDIDATE §2.1, §2.6).

Input is the cached Pupu-normalized 10 ms log-mel [T,128] (mel10). Native
windows are 1024 frames (10.24 s) every 512 frames; each keeps only its centre
512 frames, so every 40 ms cell of the song comes from exactly one window
centre. Song edges are padded with the mel of digital silence, matching the
official zero-padded input; no attention mask is applied, as in pretraining.
Taps are block outputs (pre final LayerNorm) as [cells, 8 bands, 1024].
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import numpy as np
import torch
from .pupu import FrozenLargeTeacher, make_rope
from .preprocess import waveform_mel, PREPROCESS_RECIPE

WINDOW = 1024  # mel frames, 10.24 s
HOP = 512  # 5.12 s
EDGE = (WINDOW - HOP) // 2  # 256 frames dropped on each side
FRAMES_PER_CELL = 4  # one 40 ms time patch
BANDS = 8
SUMMARY_CELLS = 50  # 2 s global summary buckets


def silence_row() -> np.ndarray:
    return waveform_mel(np.zeros(960 * 8, np.float32))[0, 0, 0].numpy()


@dataclass(frozen=True)
class WindowPlan:
    starts: tuple[int, ...]  # window start frame (may be negative)
    cells: int  # song length in 40 ms cells (last partial cell kept)


def plan(n_frames: int) -> WindowPlan:
    cells = -(-n_frames // FRAMES_PER_CELL)
    starts, s = [], -EDGE
    while s + EDGE < cells * FRAMES_PER_CELL:
        starts.append(s)
        s += HOP
    return WindowPlan(tuple(starts), cells)


def window_batch(mel: np.ndarray, starts, pad_row: np.ndarray) -> torch.Tensor:
    """[N,1,1024,128] float32 windows; out-of-range frames are digital silence."""
    out = np.broadcast_to(pad_row, (len(starts), WINDOW, mel.shape[1])).copy()
    for i, s in enumerate(starts):
        a, b = max(s, 0), min(s + WINDOW, mel.shape[0])
        if b > a:
            out[i, a - s:b - s] = mel[a:b]
    return torch.from_numpy(out)[:, None]


class TapEncoder(torch.nn.Module):
    """Runs the frozen encoder once per window and returns requested taps (centre only)."""

    def __init__(self, teacher: FrozenLargeTeacher):
        super().__init__()
        self.teacher = teacher.eval().requires_grad_(False)

    @torch.inference_mode()
    def forward(self, windows: torch.Tensor, layers) -> dict[int, torch.Tensor]:
        p = self.teacher.patch_embed.proj.weight
        windows = windows.to(device=p.device, dtype=p.dtype)
        grid = (WINDOW // FRAMES_PER_CELL, BANDS)
        rope = make_rope(grid, self.teacher.head_dim, device=windows.device)
        x = self.teacher.patch_embed(windows)
        want, out, keep = set(layers), {}, slice(EDGE // FRAMES_PER_CELL, (EDGE + HOP) // FRAMES_PER_CELL)
        for l, block in enumerate(self.teacher.teacher.blocks, 1):
            x = block(x, rope)
            if l in want:
                out[l] = x.reshape(x.shape[0], *grid, -1)[:, keep]
            if l >= max(want):
                break
        return out


def encode_song(encoder: TapEncoder, mel: np.ndarray, layers, *, batch: int = 8, pad_row=None,
                out_dtype=torch.float16) -> dict[int, torch.Tensor]:
    """Per tap: [cells, 8, 1024] in out_dtype on CPU, covering the whole song."""
    pad_row = silence_row() if pad_row is None else pad_row
    p = plan(mel.shape[0])
    parts = {l: [] for l in layers}
    for i in range(0, len(p.starts), batch):
        taps = encoder(window_batch(mel, p.starts[i:i + batch], pad_row), layers)
        for l, t in taps.items():
            parts[l].append(t.reshape(-1, BANDS, t.shape[-1]).to('cpu', out_dtype))
    return {l: torch.cat(v)[:p.cells] for l, v in parts.items()}


def summary_means(tap: torch.Tensor) -> torch.Tensor:
    """2 s deterministic time means keeping 8 bands: [ceil(cells/50), 8, D] float32."""
    n = -(-tap.shape[0] // SUMMARY_CELLS)
    x = torch.zeros(n * SUMMARY_CELLS, *tap.shape[1:], dtype=torch.float32)
    x[:tap.shape[0]] = tap.float()
    counts = torch.full((n,), SUMMARY_CELLS, dtype=torch.float32)
    counts[-1] = tap.shape[0] - (n - 1) * SUMMARY_CELLS
    return x.reshape(n, SUMMARY_CELLS, *tap.shape[1:]).sum(1) / counts[:, None, None]


def cache_key(audio_sha256: str, teacher_sha256: str, layers, dtype: str, lora: str | None = None) -> str:
    """Exact dependency key (§2.6): audio, preprocessing, windowing, frozen weights, taps, precision."""
    payload = repr((audio_sha256, PREPROCESS_RECIPE, WINDOW, HOP, 'silence-pad-no-mask', teacher_sha256,
                    tuple(sorted(layers)), dtype, lora))
    return hashlib.sha256(payload.encode()).hexdigest()[:32]
