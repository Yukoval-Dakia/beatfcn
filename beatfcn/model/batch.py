"""Collate CoreExamples + song caches into padded torch tensors (P1).

Song caches (any array-like, typically np.memmap): taps {layer: [cells, 8, Dt]},
summary [S, 8, 1024] (2 s means of H20), mel10 [frames, 128]. Only slices a core
needs are read: 16 s of taps and mel, the reference cells, and for each target
hold its 16 candidate end blocks (summary) plus the true end block's detail
(top tap only, revealed AFTER the block prediction, design §4.3).
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
from . import features as F

MAX_NOTES = 8
END_BLOCK_CELLS = F.BLOCK_US // F.CELL_US  # 50
FRAMES_LOCAL = F.LOCAL_CELLS * F.FRAMES_PER_CELL  # 1600


@dataclass
class SongCache:
    taps: dict  # layer -> [cells, 8, Dt]
    summary: np.ndarray  # [S, 8, 1024]
    mel10: np.ndarray  # [frames, 128]
    duration_us: int


def _slice(arr, start, n, fill=0.0, dtype=np.float32):
    """Rows [start, start+n) of arr with zero (or fill) outside; returns ndarray and validity.
    Int8 PCA taps (anything with .raw) are sliced raw: dequantisation happens on the device (dequant)."""
    src = getattr(arr, 'raw', arr)
    if src is not arr:
        dtype = np.int8
    out = np.full((n, *arr.shape[1:]), fill, dtype)
    a, b = max(start, 0), min(start + n, arr.shape[0])
    if b > a:
        out[a - start:b - start] = np.asarray(src[a:b], dtype)
    valid = np.zeros(n, bool)
    valid[max(a - start, 0):max(b - start, 0)] = True
    return out, valid


def pool_summary(summary: np.ndarray, max_tokens: int = F.MAX_SUMMARY):
    """Adaptive bucket width for long songs (design §5.1); returns tokens and their centre time (s)."""
    s = np.asarray(summary, np.float32)
    k = -(-len(s) // max_tokens)
    n = -(-len(s) // k)
    pad = np.zeros((n * k, *s.shape[1:]), np.float32)
    pad[:len(s)] = s
    cnt = np.minimum(k, len(s) - np.arange(n) * k)
    tokens = pad.reshape(n, k, *s.shape[1:]).sum(1) / cnt[:, None, None]
    centre = (np.arange(n) * k + cnt / 2) * F.BLOCK_US / F.SEC
    return tokens, centre, k


BUCKET = {'G': 32, 'K': 64, 'R': 64, 'A': 16, 'S': 64, 'H': 64}  # variable dims rounded up: few distinct batch shapes
# One fixed shape for CUDA-graph training (collate_meta(static=STATIC)); cores beyond a limit are skipped
# and counted. Limits cover >99% of cores (measured: G<=121, K<=218, R<=256, A<=26, holds/batch<=360).
STATIC_CORE = {'G': 128, 'K': 256, 'R': 256, 'A': 32, 'holds': 32}  # per-core caps the sampler enforces (CoreSampler.limits)
STATIC_BATCH = 32
STATIC = {'G': 128, 'K': 256, 'R': 256, 'A': 32, 'S': 256, 'H': STATIC_CORE['holds'] * STATIC_BATCH}  # H cannot overflow


def _bucket(n, q):
    return -(-max(n, 1) // q) * q


def _size(n, key, static):
    if static is None:
        return _bucket(n, BUCKET[key])
    if n > static[key]:
        raise F.SkipExample(f'static shape overflow: {key}={n} > {static[key]}')
    return static[key]


@dataclass
class SongInfo:
    """What collate_meta needs to know about a core's song without touching its audio arrays."""
    key: object  # whatever attach_audio's get_song accepts (a sha, or an index into a local list)
    n_cells: int
    n_summary: int

    @staticmethod
    def of(song: SongCache, key, tap):
        return SongInfo(key, song.taps[tap].shape[0], song.summary.shape[0])


def collate_meta(examples: list[F.CoreExample], infos: list[SongInfo], taps: tuple[int, ...], top_tap: int,
                 Dt: int, D_sum: int, quant: bool, static: dict | None = None) -> dict:
    """Every batch tensor that does not need audio (small: ~50 KB per core), plus `_audio`, the recipe
    attach_audio() uses to fill the audio tensors on any device from resident song caches. Workers run this;
    the audio never crosses process boundaries."""
    B = len(examples)
    G = _size(max([len(e.groups) for e in examples] + [1]), 'G', static)
    K = _size(max([len(e.known) for e in examples] + [1]), 'K', static)
    R = _size(max([len(e.refs) for e in examples] + [1]), 'R', static)
    A = _size(max([len(e.copy_times) for e in examples] + [1]), 'A', static)
    max_tokens = F.MAX_SUMMARY if static is None else static['S']  # static: long songs pool coarser (same rule)
    sk = [-(-i.n_summary // max_tokens) for i in infos]  # summary pooling factor per core
    sn = [-(-i.n_summary // k) for i, k in zip(infos, sk)]  # pooled token count per core
    S = _size(max(sn), 'S', static)
    out = {
        'mode': torch.tensor([e.mode for e in examples]),
        'audio_valid': torch.zeros(B, F.LOCAL_CELLS, dtype=torch.bool), 'cell_state': torch.zeros(B, F.LOCAL_CELLS, dtype=torch.long),
        'summary_t': torch.zeros(B, S), 'summary_stats': torch.zeros(B, S, 6), 'summary_valid': torch.zeros(B, S, dtype=torch.bool),
        'known': torch.zeros(B, K, 9), 'known_valid': torch.zeros(B, K, dtype=torch.bool),
        'refs': torch.zeros(B, R, 9), 'refs_valid': torch.zeros(B, R, dtype=torch.bool),
        'controls': torch.full((B, len(F.CONTROL_FIELDS)), -1, dtype=torch.long),
        'allowed_types': torch.zeros(B, 4, dtype=torch.bool), 'lines': torch.zeros(B, dtype=torch.long),
        'bin_ok': torch.zeros(B, F.CORE_BINS, 10, dtype=torch.bool),
        'copy_t': torch.zeros(B, A), 'copy_valid': torch.zeros(B, A, dtype=torch.bool),
        'right_rows': torch.zeros(B, G + 1, 4),
        'grp_valid': torch.zeros(B, G, dtype=torch.bool), 'grp_t': torch.zeros(B, G), 'grp_bin': torch.zeros(B, G, dtype=torch.long),
        'grp_res': torch.zeros(B, G, dtype=torch.long), 'grp_copy': torch.full((B, G), -1, dtype=torch.long),
        'grp_n': torch.zeros(B, G, dtype=torch.long), 'busy': torch.zeros(B, G, 4, dtype=torch.bool), 'right_new': torch.zeros(B, G, 4),
        'note_type': torch.zeros(B, G, MAX_NOTES, dtype=torch.long), 'note_lane': torch.full((B, G, MAX_NOTES), -1, dtype=torch.long),
        'note_line': torch.full((B, G, MAX_NOTES), -1, dtype=torch.long), 'note_side': torch.full((B, G, MAX_NOTES), -1, dtype=torch.long),
        'note_x': torch.zeros(B, G, MAX_NOTES), 'note_dur': torch.zeros(B, G, MAX_NOTES),
        'cell_sel': torch.zeros(B, F.CORE_US // F.CELL_US, dtype=torch.long),
    }
    out['max_notes'] = max([len(tg['notes']) for e in examples for tg in e.groups] + [0])  # host int (unused by the dense decoder)
    recipe = {'keys': [i.key for i in infos], 'cell0': [e.song_cell0 for e in examples], 'summary_k': sk, 'summary_n': sn,
              'ref_reads': [], 'holds': [], 'taps': tuple(taps), 'top': top_tap, 'Dt': Dt, 'D_sum': D_sum, 'quant': quant,
              'R': R, 'S': S}
    hold_index, hold = [], {'block': [], 'coarse': [], 'fine': [], 'ok_lo': [], 'ok_hi': [], 'onset_us': []}
    for b, (ex, info) in enumerate(zip(examples, infos)):
        a, bb = max(ex.song_cell0, 0), min(ex.song_cell0 + F.LOCAL_CELLS, info.n_cells)
        out['audio_valid'][b, max(a - ex.song_cell0, 0):max(bb - ex.song_cell0, 0)] = True
        out['cell_state'][b] = torch.from_numpy(ex.cell_state)
        k, n = sk[b], sn[b]
        centre = (np.arange(n) * k + np.minimum(k, info.n_summary - np.arange(n) * k) / 2) * F.BLOCK_US / F.SEC
        out['summary_t'][b, :n] = torch.from_numpy(centre - ex.c0_us / F.SEC)
        stats = ex.summary_stats
        if k > 1:  # pool stats like the audio summary
            stats = np.stack([stats[i:i + k].mean(0) for i in range(0, len(stats), k)])
        ns = min(len(stats), n)
        out['summary_stats'][b, :ns] = torch.from_numpy(stats[:ns])
        out['summary_valid'][b, :n] = True
        out['known'][b, :len(ex.known)] = torch.from_numpy(ex.known)
        out['known_valid'][b, :len(ex.known)] = True
        out['refs'][b, :len(ex.refs)] = torch.from_numpy(ex.refs)
        out['refs_valid'][b, :len(ex.refs)] = True
        if len(ex.refs):
            cells = np.rint((ex.c0_us + ex.refs[:, 0].astype(np.float64) * F.SEC) / F.CELL_US).astype(np.int64)
            order = np.argsort(cells, kind='stable')
            breaks = np.flatnonzero(np.diff(cells[order]) > 200) + 1  # one contiguous read per reference cluster
            for grp in np.split(order, breaks):
                lo, hi = int(cells[grp].min()), int(cells[grp].max()) + 1
                recipe['ref_reads'].append((b, grp.tolist(), lo, hi, (cells[grp] - lo).tolist()))
        out['controls'][b] = torch.from_numpy(ex.controls)
        out['allowed_types'][b] = torch.from_numpy(ex.allowed_types)
        out['lines'][b] = ex.lines
        out['bin_ok'][b] = torch.from_numpy(ex.bin_ok)
        out['copy_t'][b, :len(ex.copy_times)] = torch.from_numpy(ex.copy_times / F.SEC).float()
        out['copy_valid'][b, :len(ex.copy_times)] = True
        out['right_rows'][b, :len(ex.right)] = torch.from_numpy(ex.right)
        # selection labels per core cell: any head (target or observed) -> 1; cells outside the audio -> -1
        core_cells = F.CORE_US // F.CELL_US
        lab = torch.zeros(core_cells, dtype=torch.long)
        for tg in ex.groups:
            lab[tg['t_rel_us'] // F.CELL_US] = 1
        for row in ex.known:
            t = row[0] * F.SEC
            if 0 <= t < F.CORE_US and row[7] == 0:
                lab[int(t // F.CELL_US)] = 1
        h0 = F.HALO_US // F.CELL_US
        lab[~out['audio_valid'][b, h0:h0 + core_cells]] = -1
        out['cell_sel'][b] = lab
        for g, tg in enumerate(ex.groups):
            if len(tg['notes']) > MAX_NOTES:
                raise F.SkipExample('group has more notes than MAX_NOTES')
            out['grp_valid'][b, g] = True
            out['grp_t'][b, g] = tg['t_rel_us'] / F.SEC
            out['grp_bin'][b, g], out['grp_res'][b, g], out['grp_copy'][b, g] = tg['bin'], tg['residual'], tg['copy']
            out['grp_n'][b, g] = len(tg['notes'])
            out['busy'][b, g] = torch.from_numpy(tg['busy_lanes'])
            out['right_new'][b, g] = torch.from_numpy(tg['right_new'])
            onset_abs = ex.c0_us + tg['t_rel_us']
            for n_, nt in enumerate(tg['notes']):
                out['note_type'][b, g, n_] = nt['type']
                out['note_lane'][b, g, n_], out['note_line'][b, g, n_] = nt['lane'], nt['line']
                out['note_side'][b, g, n_], out['note_x'][b, g, n_] = nt['side'], nt['x']
                if 'end_rel_us' not in nt:
                    continue
                end_abs = onset_abs + nt['end_rel_us']
                b0 = onset_abs // F.BLOCK_US
                blk = end_abs // F.BLOCK_US - b0
                if blk >= F.MAX_END_BLOCKS:
                    raise F.SkipExample('hold longer than the endpoint block range')
                out['note_dur'][b, g, n_] = nt['end_rel_us'] / F.SEC
                hold_index.append((b, g, n_))
                hold['block'].append(blk)
                hold['coarse'].append(end_abs % F.BLOCK_US // F.BIN_US)
                hold['fine'].append(end_abs % F.BIN_US // F.MS)
                lo, hi = nt['end_ok_us']
                hold['ok_lo'].append(onset_abs + lo)
                hold['ok_hi'].append(onset_abs + hi)
                hold['onset_us'].append(onset_abs)
                recipe['holds'].append((b, int(b0), int((b0 + blk) * END_BLOCK_CELLS)))
    H, Hp = len(hold_index), _size(len(hold_index), 'H', static)
    pad = Hp - H  # padded holds: index (0,0,0), zero targets/blocks/detail, hold_valid False (masked everywhere)
    out['hold_index'] = torch.tensor(hold_index + [(0, 0, 0)] * pad, dtype=torch.long).reshape(-1, 3)
    out['hold_valid'] = torch.arange(Hp) < H
    for key in ('block', 'coarse', 'fine', 'ok_lo', 'ok_hi', 'onset_us'):
        out['hold_' + key] = torch.tensor(hold[key] + [0] * pad, dtype=torch.long)
    recipe['Hp'] = Hp
    out['_audio'] = recipe
    return out


def _dslice(t, start, n, out):
    """out[:] = rows [start, start+n) of t, zero outside t (same semantics as _slice, any device)."""
    a, b = max(start, 0), min(start + n, t.shape[0])
    out.zero_()
    if b > a:
        out[a - start:b - start] = t[a:b].to(out.dtype)
    return out


def attach_audio(bt: dict, get_song, device='cpu', scales=None) -> dict:
    """Fill the audio tensors of a collate_meta batch from song caches: get_song(key) -> object with .taps
    (layer -> tensor [cells,8,Dt], int8 raw when quantised), .summary [S,8,D] and .mel10 [frames,128] on
    `device`. scales: {layer: per-dim scale} for int8 taps (tap_scale / dequant). Runs on the GPU in
    training, so only ~50 KB of metadata per core leaves the worker processes."""
    r = bt.pop('_audio')
    taps, top, Dt, quant = r['taps'], r['top'], r['Dt'], r['quant']
    B, R, S, Hp = len(r['keys']), r['R'], r['S'], r['Hp']
    tdt = torch.int8 if quant else torch.float32
    dev = torch.device(device)
    audio = torch.zeros(B, len(taps), F.LOCAL_CELLS, 8, Dt, dtype=tdt, device=dev)
    mel = torch.zeros(B, FRAMES_LOCAL, 128, dtype=torch.float16, device=dev)
    summary = torch.zeros(B, S, 8, r['D_sum'], dtype=torch.float16, device=dev)
    ref_audio = torch.zeros(B, R, len(taps), 8, Dt, dtype=tdt, device=dev)
    blocks = torch.zeros(Hp, F.MAX_END_BLOCKS, 8, r['D_sum'], dtype=torch.float16, device=dev)
    detail = torch.zeros(Hp, END_BLOCK_CELLS, 8, Dt, dtype=tdt, device=dev)
    songs = [get_song(k) for k in r['keys']]
    for b, (song, c0) in enumerate(zip(songs, r['cell0'])):
        for j, l in enumerate(taps):
            _dslice(song.taps[l], c0, F.LOCAL_CELLS, audio[b, j])
        _dslice(song.mel10, c0 * F.FRAMES_PER_CELL, FRAMES_LOCAL, mel[b])
        k, n = r['summary_k'][b], r['summary_n'][b]
        if k == 1:
            summary[b, :n] = song.summary[:n].to(torch.float16)
        else:  # adaptive bucket width for long songs (== pool_summary)
            sm = song.summary.float()
            padded = torch.zeros(n * k, *sm.shape[1:], dtype=torch.float32, device=sm.device)
            padded[:sm.shape[0]] = sm
            cnt = torch.clamp(sm.shape[0] - torch.arange(n, device=sm.device) * k, max=k).float()
            summary[b, :n] = (padded.reshape(n, k, *sm.shape[1:]).sum(1) / cnt[:, None, None]).to(torch.float16)
    for b, grp, lo, hi, rel in r['ref_reads']:
        gi = torch.tensor(grp, device=dev)
        ri = torch.tensor(rel, device=dev)
        for j, l in enumerate(taps):
            span = torch.zeros(hi - lo, 8, Dt, dtype=tdt, device=dev)
            _dslice(songs[b].taps[l], lo, hi - lo, span)
            ref_audio[b, gi, j] = span[ri]
    for h, (b, b0, det0) in enumerate(r['holds']):
        _dslice(songs[b].summary, b0, F.MAX_END_BLOCKS, blocks[h])
        _dslice(songs[b].taps[top], det0, END_BLOCK_CELLS, detail[h])
    bt['audio'], bt['mel'], bt['summary'], bt['ref_audio'] = audio, mel, summary, ref_audio
    bt['hold_blocks'], bt['hold_detail'] = blocks, detail
    if quant:  # per-tap dequantisation scale [T, Dt]; applied on the device by dequant()
        sc = [torch.broadcast_to(torch.as_tensor(np.asarray(scales[l], np.float32)), (Dt,)) for l in taps]
        bt['tap_scale'] = torch.stack(sc).clone().to(dev)
        bt['tap_top'] = taps.index(top)
    return bt


class _CpuSong:
    """Torch views of a SongCache for attach_audio on the CPU (int8 taps stay raw)."""

    def __init__(self, song: SongCache, taps):
        self.taps = {l: torch.from_numpy(np.asarray(getattr(song.taps[l], 'raw', song.taps[l]))) for l in taps}
        self.summary = torch.from_numpy(np.asarray(song.summary))
        self.mel10 = torch.from_numpy(np.asarray(song.mel10))


def collate(examples: list[F.CoreExample], songs: list[SongCache], taps: tuple[int, ...], top_tap: int) -> dict:
    """Metadata + audio in one call on the CPU (tests, evaluation, sampling): the same two functions the
    training loop runs split across worker processes and the GPU."""
    quant = all(hasattr(s.taps[l], 'raw') for s in songs for l in taps)
    if not quant and any(hasattr(s.taps[l], 'raw') for s in songs for l in taps):
        raise ValueError('mixed int8 / float tap caches in one batch')
    infos = [SongInfo.of(s, b, taps[0]) for b, s in enumerate(songs)]
    bt = collate_meta(examples, infos, taps, top_tap, songs[0].taps[taps[0]].shape[-1], songs[0].summary.shape[-1], quant)
    scales = {l: songs[0].taps[l].scale for l in taps} if quant else None
    cpu = [_CpuSong(s, taps) for s in songs]
    return attach_audio(bt, lambda b: cpu[b], 'cpu', scales)


def dequant(bt: dict) -> dict:
    """Device-side expansion of the compact transport format: int8 taps x scale, float16 -> float32.
    Idempotent; collate keeps batches small so worker IPC, pinning and H2D copies stay cheap."""
    sc = bt.pop('tap_scale', None)
    if sc is not None:
        top = bt.pop('tap_top', None)
        bt['audio'] = bt['audio'].float() * sc[None, :, None, None, :]
        bt['ref_audio'] = bt['ref_audio'].float() * sc[None, None, :, None, :]
        if top is not None:
            bt['hold_detail'] = bt['hold_detail'].float() * sc[top]
    for k, v in bt.items():
        if torch.is_tensor(v) and v.dtype == torch.float16:
            bt[k] = v.float()
    return bt
