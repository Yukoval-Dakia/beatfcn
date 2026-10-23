"""S2: short joint adaptation with LoRA on PupuM2D blocks 21-24 (design §10.3, FIRST_CANDIDATE §2.5).

Only the top tap changes under LoRA; taps below 21 stay cached. For each microbatch the windows that
cover every cell the model reads at the top tap (16 s local audio, reference cells, hold end-block
detail) are re-encoded: blocks 1-20 under no_grad, blocks 21-24 with LoRA and autograd. Their centre
cells pass through the same frozen PCA as the cache (no quantisation needed online) and replace the
cached top-tap slices in the batch. Nothing human-conditioned enters the encoder.
"""
from __future__ import annotations
import math
import numpy as np
import torch
from torch import nn
from ..audio import memory as M
from ..model import features as FE
from ..model.batch import END_BLOCK_CELLS

CENTRE = M.HOP // M.FRAMES_PER_CELL  # 128 cells per window centre


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank=8, alpha=16, dropout=0.05):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.a = nn.Parameter(torch.empty(rank, base.in_features, dtype=torch.float32))
        self.b = nn.Parameter(torch.zeros(base.out_features, rank, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))
        self.scale, self.drop = alpha / rank, nn.Dropout(dropout)

    def forward(self, x):
        lora = (self.drop(x).float() @ self.a.T) @ self.b.T
        return self.base(x) + (lora * self.scale).to(x.dtype)


def add_lora(teacher, layers=(21, 22, 23, 24), rank=8, alpha=16, dropout=0.05):
    """Wrap fused qkv, attention output, SwiGLU fc1/fc2 of the given 1-based blocks. Returns LoRA params."""
    params = []
    for l in layers:
        blk = teacher.teacher.blocks[l - 1]
        for parent, name in ((blk.attn, 'qkv'), (blk.attn, 'proj'), (blk.mlp, 'fc1'), (blk.mlp, 'fc2')):
            w = LoRALinear(getattr(parent, name), rank, alpha, dropout).to(getattr(parent, name).weight.device)
            setattr(parent, name, w)
            params += [w.a, w.b]
    return params


def encode_top(teacher, windows, frozen_until=20, top=24):
    """[W,1,1024,128] mel windows -> top-tap centre cells [W,128,8,D]; blocks <= frozen_until run without grad."""
    p = teacher.patch_embed.proj.weight
    windows = windows.to(device=p.device, dtype=p.dtype)
    grid = (M.WINDOW // M.FRAMES_PER_CELL, M.BANDS)
    rope = M.make_rope(grid, teacher.head_dim, device=windows.device)
    with torch.no_grad():
        x = teacher.patch_embed(windows)
        for blk in teacher.teacher.blocks[:frozen_until]:
            x = blk(x, rope)
    for blk in teacher.teacher.blocks[frozen_until:top]:
        x = blk(x, rope)
    keep = slice(M.EDGE // M.FRAMES_PER_CELL, (M.EDGE + M.HOP) // M.FRAMES_PER_CELL)
    return x.reshape(x.shape[0], *grid, -1)[:, keep]


def plan_windows(examples, bt, n_cells_by_song):
    """Per microbatch: the window list and, for every top-tap cell the batch reads, its row in the
    flattened (window, centre-cell) table (-1 when outside the song)."""
    keyed = {}  # (song_index, window k) -> row
    rows = []

    def row_of(si, cell):
        if cell < 0 or cell >= n_cells_by_song[si]:
            return -1
        k = cell // CENTRE
        if (si, k) not in keyed:
            keyed[(si, k)] = len(rows)
            rows.append((si, k))
        return keyed[(si, k)] * CENTRE + cell % CENTRE
    B = len(examples)
    local = np.full((B, FE.LOCAL_CELLS), -1, np.int64)
    R = bt['refs'].shape[1]
    ref = np.full((B, R), -1, np.int64)
    for b, ex in enumerate(examples):
        si = b
        for c in range(FE.LOCAL_CELLS):
            local[b, c] = row_of(si, ex.song_cell0 + c)
        for r, rowv in enumerate(ex.refs):
            ref[b, r] = row_of(si, int(round((ex.c0_us + rowv[0] * FE.SEC) / FE.CELL_US)))
    hold = np.full((len(bt['hold_index']), END_BLOCK_CELLS), -1, np.int64)
    for h, (b, g, n) in enumerate(bt['hold_index'].tolist()):
        start = (int(bt['hold_onset_us'][h]) // FE.BLOCK_US + int(bt['hold_block'][h])) * END_BLOCK_CELLS
        for c in range(END_BLOCK_CELLS):
            hold[h, c] = row_of(b, start + c)
    return rows, torch.from_numpy(local), torch.from_numpy(ref), torch.from_numpy(hold)


def mel_windows(rows, mels, pad_row):
    starts = [k * M.HOP - M.EDGE for _, k in rows]
    out = [M.window_batch(mels[si], [s], pad_row) for (si, _), s in zip(rows, starts)]
    return torch.cat(out) if out else torch.zeros(0, 1, M.WINDOW, 128)


def attach_top(bt, teacher, windows, idx_local, idx_ref, idx_hold, pca_mean, pca_comp, tap_index):
    """Replace the batch's top-tap slices with LoRA-encoded, PCA-projected features (with grad)."""
    top = encode_top(teacher, windows)  # [W,128,8,1024]
    flat = top.float() - pca_mean  # centred like the cache (fit_i8); pca_comp=None: full width
    if pca_comp is not None:
        flat = flat @ pca_comp  # [W,128,8,Dp]
    flat = flat.reshape(-1, *flat.shape[2:])
    zero = flat.new_zeros(1, *flat.shape[1:])
    table = torch.cat([flat, zero])  # index -1 -> last row (zeros)

    def take(idx):
        i = idx.to(table.device).clone()
        i[i < 0] = table.shape[0] - 1
        return table[i.reshape(-1)].reshape(*idx.shape, *table.shape[1:])
    audio = bt['audio'].clone()
    audio[:, tap_index] = take(idx_local).to(audio.dtype)
    bt['audio'] = audio
    if idx_ref.numel():
        ra = bt['ref_audio'].clone()
        ra[:, :, tap_index] = take(idx_ref).to(ra.dtype)
        bt['ref_audio'] = ra
    if idx_hold.numel():
        bt['hold_detail'] = take(idx_hold).to(bt['hold_detail'].dtype)
    return bt
