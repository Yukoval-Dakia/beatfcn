"""Collate CoreExamples + song caches into padded torch tensors (P1 v2.1).

Song caches (any array-like, typically np.memmap or device-resident tensors): taps {layer: [cells, 8, Dt]},
summary [S, 8, 1024] (2 s means, hold end-block candidates), mel10 [frames, 128] (S2 only). A core reads
16 s of local taps, the reference cells, for each target hold its 16 candidate end blocks (summary) plus
the true end block's detail (top tap only, revealed AFTER the block prediction), and the whole-song
stream: up to MAX_STREAM_CELLS cells of every tap (a window centred on the core beyond that) that the
model's block encoder turns into 400 ms song tokens (FIRST_CANDIDATE v2.1 §3.2).
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
from . import features as F

MAX_NOTES = 8
END_BLOCK_CELLS = F.BLOCK_US // F.CELL_US  # 50
FRAMES_LOCAL = F.LOCAL_CELLS * F.FRAMES_PER_CELL  # 1600
CORE_CELLS = F.CORE_US // F.CELL_US  # 200
MS_PER_CELL = F.CELL_US // F.MS  # 40
STREAM_BUCKET = 2000  # song-stream crop rounded up to this many song cells (80 s): few distinct shapes
SONG_TAPS = 2  # the block encoder reads the last SONG_TAPS taps (the deeper, more semantic layers)
PREV_ONSETS = 4  # most recent onsets fed to the rhythm interval kernel, per Y row
NO_ONSET = -1000.0  # prev_onsets padding (seconds): "no context"


@dataclass
class SongCache:
    taps: dict  # layer -> [cells, 8, Dt]
    summary: np.ndarray  # [S, 8, 1024]
    mel10: np.ndarray  # [frames, 128]
    duration_us: int


def _slice(arr, start, n, fill=0.0, dtype=np.float32):
    """Rows [start, start+n) of arr with zero (or fill) outside; returns ndarray and validity.
    Int8 taps (anything with .raw) are sliced raw: dequantisation happens on the device (dequant)."""
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


BUCKET = {'G': 32, 'K': 64, 'R': 64, 'A': 16, 'H': 64, 'C': STREAM_BUCKET}  # variable dims rounded up
# One fixed shape for the legacy static path (collate_meta(static=STATIC)); cores beyond a limit are skipped.
STATIC_CORE = {'G': 128, 'K': 256, 'R': 256, 'A': 32, 'holds': 32}  # per-core caps the sampler enforces (CoreSampler.limits)
STATIC_BATCH = 32
STATIC = {'G': 128, 'K': 256, 'R': 256, 'A': 32, 'H': STATIC_CORE['holds'] * STATIC_BATCH, 'C': F.MAX_STREAM_CELLS}


# CUDA-graph shape tiers (train.graph.TieredStep): a dynamic batch is padded up to the smallest tier that
# holds it; batches beyond the last tier run eagerly. Padding only adds rows every loss already masks
# (valid flags false, hold_valid false, blk_valid false), so no sample is dropped.
TIERS = [  # micro-batch 8, from 1000 random 8-core batches of the validation cores (2026-10-23): tier 0 holds
    # ~95% of them on the row axes (largest seen: G 160, K 256, R 256, H 320, N 521); C (song-stream window in
    # 40 ms song cells; stored at STREAM_STRIDE) 8000 = 5.3 min covers most songs, longer ones are windowed.
    {'G': 192, 'K': 256, 'R': 256, 'A': 16, 'H': 320, 'C': 8000},
    {'G': 256, 'K': 384, 'R': 256, 'A': 32, 'H': 640, 'C': 12000},
    {'G': 320, 'K': 512, 'R': 256, 'A': 32, 'H': 1024, 'C': 12000},  # C never selects tier 2 (its slots are eager)
]
# which padded axis each batch key grows along (everything else has a fixed shape per micro-batch size)
AXIS = {**{k: ('G', 1) for k in ('grp_valid', 'grp_t', 'grp_cell', 'grp_off', 'grp_copy', 'grp_n', 'busy', 'right_new',
                                 'note_type', 'note_lane', 'note_line', 'note_side', 'note_x', 'note_dur')},
        'right_rows': ('G1', 1), 'prev_onsets': ('G1', 1),
        'known': ('K', 1), 'known_valid': ('K', 1),
        'refs': ('R', 1), 'refs_valid': ('R', 1), 'ref_audio': ('R', 1),
        'copy_t': ('A', 1), 'copy_valid': ('A', 1),
        'song_cells': ('CS', 2),  # stored cells = C // STREAM_STRIDE
        **{k: ('NB', 1) for k in ('blk_valid', 'blk_t', 'blk_cov', 'blk_grid', 'blk_counts')},
        **{k: ('H', 0) for k in ('hold_index', 'hold_valid', 'hold_block', 'hold_cell', 'hold_off', 'hold_ok_lo',
                                 'hold_ok_hi', 'hold_onset_us', 'hold_prev', 'hold_blocks', 'hold_detail', 'hold_inner_index')},
        'inner_index': ('N', 0)}  # compact_inner: valid-group rows; N = B x G x 3/4 per tier, padded with an invalid slot
PAD_VALUE = {'grp_copy': -1, 'note_lane': -1, 'note_line': -1, 'note_side': -1, 'controls': -1, 'prev_onsets': NO_ONSET,
             'hold_prev': NO_ONSET}


def batch_dims(bt) -> dict:
    d = {'G': bt['grp_t'].shape[1], 'K': bt['known'].shape[1], 'R': bt['refs'].shape[1], 'A': bt['copy_t'].shape[1],
         'H': bt['hold_index'].shape[0], 'C': bt['blk_valid'].shape[1] * F.BLOCK_CELLS, 'B': bt['mode'].shape[0]}
    if 'inner_index' in bt:
        d['N'] = bt['inner_index'].shape[0]
    return d


def tier_n(tier, B):
    return B * tier['G'] * 3 // 4


def tier_of(dims: dict, tiers=TIERS):
    """Index of the smallest tier holding `dims`, or None (eager fallback). With compaction ('N' in dims)
    the tier must leave the last group slot free (its index is the padding sentinel) and hold N rows."""
    compact = 'N' in dims
    for i, t in enumerate(tiers):
        if not all(dims[k] <= t[k] for k in t):
            continue
        if compact and (dims['G'] >= t['G'] or dims['N'] > tier_n(t, dims['B'])):
            continue
        return i
    return None


def tier_shape(key, shape, tier, B=None):
    """Static shape of batch key `key` (dynamic shape `shape`) in `tier`."""
    if key not in AXIS:
        return tuple(shape)
    name, ax = AXIS[key]
    if name == 'G1':
        n = tier['G'] + 1
    elif name == 'N':
        n = tier_n(tier, B)
    elif name == 'NB':
        n = tier['C'] // F.BLOCK_CELLS
    elif name == 'CS':
        n = tier['C'] // F.STREAM_STRIDE
    else:
        n = tier[name]
    out = list(shape)
    out[ax] = n
    return tuple(out)


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


def stream_crop(n_cells: int, song_cell0: int, C: int):
    """Song-stream window of C cells: the whole song when it fits, else C cells centred on the core,
    aligned to block boundaries. Returns (start cell, valid cells)."""
    if n_cells <= C:
        return 0, n_cells
    centre = song_cell0 + F.LOCAL_CELLS // 2
    start = min(max(centre - C // 2, 0), n_cells - C)
    start -= start % F.BLOCK_CELLS
    return start, min(C, n_cells - start)


def collate_meta(examples: list[F.CoreExample], infos: list[SongInfo], taps: tuple[int, ...], top_tap: int,
                 Dt: int, D_sum: int, quant: bool, static: dict | None = None) -> dict:
    """Every batch tensor that does not need audio (small per core), plus `_audio`, the recipe attach_audio()
    uses to fill the audio tensors on any device from resident song caches. Workers run this; the audio
    never crosses process boundaries."""
    B = len(examples)
    G = _size(max([len(e.groups) for e in examples] + [1]), 'G', static)
    K = _size(max([len(e.known) for e in examples] + [1]), 'K', static)
    R = _size(max([len(e.refs) for e in examples] + [1]), 'R', static)
    A = _size(max([len(e.copy_times) for e in examples] + [1]), 'A', static)
    C = min(_size(max(i.n_cells for i in infos), 'C', static), F.MAX_STREAM_CELLS)
    NB = C // F.BLOCK_CELLS
    out = {
        'mode': torch.tensor([e.mode for e in examples]),
        'audio_valid': torch.zeros(B, F.LOCAL_CELLS, dtype=torch.bool), 'cell_state': torch.zeros(B, F.LOCAL_CELLS, dtype=torch.long),
        'known': torch.zeros(B, K, 9), 'known_valid': torch.zeros(B, K, dtype=torch.bool),
        'refs': torch.zeros(B, R, 9), 'refs_valid': torch.zeros(B, R, dtype=torch.bool),
        'controls': torch.full((B, len(F.CONTROL_FIELDS)), -1, dtype=torch.long),
        'allowed_types': torch.zeros(B, 4, dtype=torch.bool), 'lines': torch.zeros(B, dtype=torch.long),
        'bin_ok': torch.zeros(B, F.CORE_BINS, 10, dtype=torch.bool),
        'copy_t': torch.zeros(B, A), 'copy_valid': torch.zeros(B, A, dtype=torch.bool),
        'right_rows': torch.zeros(B, G + 1, 4),
        'prev_onsets': torch.full((B, G + 1, PREV_ONSETS), NO_ONSET),
        'blk_valid': torch.zeros(B, NB, dtype=torch.bool), 'blk_t': torch.zeros(B, NB), 'blk_cov': torch.zeros(B, NB, 3),
        'blk_grid': torch.zeros(B, NB, F.BLOCK_CELLS, 4, 2, dtype=torch.uint8), 'blk_counts': torch.zeros(B, NB, 4),
        'grp_valid': torch.zeros(B, G, dtype=torch.bool), 'grp_t': torch.zeros(B, G),
        'grp_cell': torch.zeros(B, G, dtype=torch.long), 'grp_off': torch.zeros(B, G, dtype=torch.long),
        'grp_copy': torch.full((B, G), -1, dtype=torch.long),
        'grp_n': torch.zeros(B, G, dtype=torch.long), 'busy': torch.zeros(B, G, 4, dtype=torch.bool), 'right_new': torch.zeros(B, G, 4),
        'note_type': torch.zeros(B, G, MAX_NOTES, dtype=torch.long), 'note_lane': torch.full((B, G, MAX_NOTES), -1, dtype=torch.long),
        'note_line': torch.full((B, G, MAX_NOTES), -1, dtype=torch.long), 'note_side': torch.full((B, G, MAX_NOTES), -1, dtype=torch.long),
        'note_x': torch.zeros(B, G, MAX_NOTES), 'note_dur': torch.zeros(B, G, MAX_NOTES),
        'cell_sel': torch.zeros(B, CORE_CELLS, dtype=torch.long),
    }
    out['max_notes'] = max([len(tg['notes']) for e in examples for tg in e.groups] + [0])  # host int (unused by the dense decoder)
    recipe = {'keys': [i.key for i in infos], 'cell0': [e.song_cell0 for e in examples], 'crops': [],
              'ref_reads': [], 'holds': [], 'taps': tuple(taps), 'top': top_tap, 'Dt': Dt, 'D_sum': D_sum, 'quant': quant,
              'R': R, 'C': C}
    hold_index, hold = [], {'block': [], 'cell': [], 'off': [], 'ok_lo': [], 'ok_hi': [], 'onset_us': []}
    hold_prev = []  # per hold: its onset then the most recent earlier onsets (absolute seconds; rhythm kernel)
    blk_arange = np.arange(NB, dtype=np.float64)
    for b, (ex, info) in enumerate(zip(examples, infos)):
        a, bb = max(ex.song_cell0, 0), min(ex.song_cell0 + F.LOCAL_CELLS, info.n_cells)
        out['audio_valid'][b, max(a - ex.song_cell0, 0):max(bb - ex.song_cell0, 0)] = True
        out['cell_state'][b] = torch.from_numpy(ex.cell_state)
        # whole-song stream window (the same cells attach_audio copies for the block encoder)
        start, n_valid = stream_crop(info.n_cells, ex.song_cell0, C)
        recipe['crops'].append((start, n_valid))
        b0 = start // F.BLOCK_CELLS
        nbv = max(0, min(NB, len(ex.blk_cov) - b0))
        out['blk_valid'][b, :nbv] = True
        out['blk_t'][b] = torch.from_numpy(((start + (blk_arange + 0.5) * F.BLOCK_CELLS) * F.CELL_US - ex.c0_us) / F.SEC).float()
        out['blk_cov'][b, :nbv] = torch.from_numpy(ex.blk_cov[b0:b0 + nbv])
        out['blk_grid'][b, :nbv] = torch.from_numpy(ex.blk_grid[b0:b0 + nbv])
        out['blk_counts'][b, :nbv] = torch.from_numpy(ex.blk_counts[b0:b0 + nbv])
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
                recipe['ref_reads'].append((b, torch.from_numpy(grp.copy()), lo, hi, torch.from_numpy((cells[grp] - lo).copy())))
        out['controls'][b] = torch.from_numpy(ex.controls)
        out['allowed_types'][b] = torch.from_numpy(ex.allowed_types)
        out['lines'][b] = ex.lines
        out['bin_ok'][b] = torch.from_numpy(ex.bin_ok)
        out['copy_t'][b, :len(ex.copy_times)] = torch.from_numpy(ex.copy_times / F.SEC).float()
        out['copy_valid'][b, :len(ex.copy_times)] = True
        out['right_rows'][b, :len(ex.right)] = torch.from_numpy(ex.right)
        # selection labels per core cell: any head (target or observed) -> 1; cells outside the audio -> -1
        lab = torch.zeros(CORE_CELLS, dtype=torch.long)
        for tg in ex.groups:
            lab[tg['t_rel_us'] // F.CELL_US] = 1
        for row in ex.known:
            t = row[0] * F.SEC
            if 0 <= t < F.CORE_US and row[7] == 0:
                lab[int(t // F.CELL_US)] = 1
        h0 = F.HALO_US // F.CELL_US
        lab[~out['audio_valid'][b, h0:h0 + CORE_CELLS]] = -1
        out['cell_sel'][b] = lab
        # rhythm context per Y row: the most recent onsets among known onsets BEFORE the core and the earlier
        # targets (never a later target; v2.1 §3.3 item 3), most recent first
        hist = sorted({float(t) for t in ex.known[:, 0] if t < 0})  # distinct onsets (chords share a time)
        all_onsets = sorted(set(hist) | {tg['t_rel_us'] / F.SEC for tg in ex.groups})  # for hold ends: every onset <= the hold's
        for i in range(len(ex.groups) + 1):
            recent = hist[-PREV_ONSETS:][::-1]
            if recent:
                out['prev_onsets'][b, i, :len(recent)] = torch.tensor(recent)
            if i < len(ex.groups):
                hist.append(ex.groups[i]['t_rel_us'] / F.SEC)
        for g, tg in enumerate(ex.groups):
            if len(tg['notes']) > MAX_NOTES:
                raise F.SkipExample('group has more notes than MAX_NOTES')
            out['grp_valid'][b, g] = True
            out['grp_t'][b, g] = tg['t_rel_us'] / F.SEC
            out['grp_cell'][b, g] = tg['t_rel_us'] // F.CELL_US
            out['grp_off'][b, g] = tg['t_rel_us'] % F.CELL_US // F.MS
            out['grp_copy'][b, g] = tg['copy']
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
                b0h = onset_abs // F.BLOCK_US
                blk = end_abs // F.BLOCK_US - b0h
                if blk >= F.MAX_END_BLOCKS:
                    raise F.SkipExample('hold longer than the endpoint block range')
                out['note_dur'][b, g, n_] = nt['end_rel_us'] / F.SEC
                hold_index.append((b, g, n_))
                hold['block'].append(blk)
                hold['cell'].append(end_abs % F.BLOCK_US // F.CELL_US)
                hold['off'].append(end_abs % F.CELL_US // F.MS)
                lo, hi = nt['end_ok_us']
                hold['ok_lo'].append(onset_abs + lo)
                hold['ok_hi'].append(onset_abs + hi)
                hold['onset_us'].append(onset_abs)
                t_on = tg['t_rel_us'] / F.SEC
                prev = [t for t in all_onsets if t <= t_on][-PREV_ONSETS:][::-1]  # the hold's own onset first
                hold_prev.append([p + ex.c0_us / F.SEC for p in prev] + [NO_ONSET] * (PREV_ONSETS - len(prev)))
                recipe['holds'].append((b, int(b0h), int((b0h + blk) * END_BLOCK_CELLS)))
    H, Hp = len(hold_index), _size(len(hold_index), 'H', static)
    pad = Hp - H  # padded holds: index (0,0,0), zero targets/blocks/detail, hold_valid False (masked everywhere)
    out['hold_index'] = torch.tensor(hold_index + [(0, 0, 0)] * pad, dtype=torch.long).reshape(-1, 3)
    out['hold_valid'] = torch.arange(Hp) < H
    for key in ('block', 'cell', 'off', 'ok_lo', 'ok_hi', 'onset_us'):
        out['hold_' + key] = torch.tensor(hold[key] + [0] * pad, dtype=torch.long)
    out['hold_prev'] = torch.tensor(hold_prev + [[NO_ONSET] * PREV_ONSETS] * pad, dtype=torch.float32).reshape(-1, PREV_ONSETS)
    recipe['Hp'] = Hp
    out['_audio'] = recipe
    prepare_inner_indices(out)
    return out


def prepare_inner_indices(bt):
    """CPU-side indices for optional compaction of padded GRU rows; no labels are removed."""
    if bt['grp_valid'].device.type != 'cpu':
        raise ValueError('inner indices must be prepared on CPU before device transfer')
    valid = bt['grp_valid'].flatten()
    indices = torch.where(valid)[0]
    if not len(indices):
        indices = torch.zeros(1, dtype=torch.long)  # masked dummy keeps the empty-batch loss well-defined
    inverse = torch.zeros(len(valid), dtype=torch.long)
    inverse[indices] = torch.arange(len(indices))
    hi = bt['hold_index']
    bt['inner_index'] = indices
    bt['hold_inner_index'] = inverse[hi[:, 0] * bt['grp_valid'].shape[1] + hi[:, 1]]
    return bt


def _dslice(t, start, n, out):
    """out[:] = rows [start, start+n) of t, zero outside t (same semantics as _slice, any device)."""
    a, b = max(start, 0), min(start + n, t.shape[0])
    out.zero_()
    if b > a:
        out[a - start:b - start] = t[a:b].to(out.dtype)
    return out


def _dslice_strided(t, start, n, stride, out):
    """out[i] = t[start + i * stride] for start + i * stride < n_rows, zero beyond (start >= 0)."""
    out.zero_()
    b = min(start + n, t.shape[0])
    if b > start:
        src = t[start:b:stride]
        out[:src.shape[0]] = src.to(out.dtype)
    return out


def _gather_rows(t, starts, n, out, out_idx, dev):
    """out[out_idx[i]] = t[starts[i] : starts[i] + n] (zero beyond t's rows), as one gather per call."""
    idx = starts.to(dev)[:, None] + torch.arange(n, device=dev)[None]  # [k, n]
    ok = (idx >= 0) & (idx < t.shape[0])
    rows = t[idx.clamp(0, max(t.shape[0] - 1, 0))].to(out.dtype)  # [k, n, ...]
    rows = rows.masked_fill(~ok[(...,) + (None,) * (rows.dim() - 2)], 0)
    out[out_idx] = rows
    return out


def attach_audio(bt: dict, get_song, device='cpu', scales=None, mel=False) -> dict:
    """Fill the audio tensors of a collate_meta batch from song caches: get_song(key) -> object with .taps
    (layer -> tensor [cells,8,Dt], int8 raw when quantised), .summary [S,8,D] and .mel10 [frames,128] on
    `device`. scales: {layer: per-dim scale} (or a stacked [T, Dt] device tensor) for int8 taps. Runs on
    the GPU in training, so only metadata leaves the worker processes. mel: also attach the local mel10
    window (S2 and diagnostics; the v2.1 model does not read it)."""
    r = bt.pop('_audio')
    taps, top, Dt, quant = r['taps'], r['top'], r['Dt'], r['quant']
    B, R, C, Hp = len(r['keys']), r['R'], r['C'], r['Hp']
    tdt = torch.int8 if quant else torch.float32
    dev = torch.device(device)
    audio = torch.zeros(B, len(taps), F.LOCAL_CELLS, 8, Dt, dtype=tdt, device=dev)
    song_taps = taps[-SONG_TAPS:]
    song_cells = torch.zeros(B, len(song_taps), C // F.STREAM_STRIDE, 8, Dt, dtype=tdt, device=dev)
    ref_audio = torch.zeros(B, R, len(taps), 8, Dt, dtype=tdt, device=dev)
    blocks = torch.zeros(Hp, F.MAX_END_BLOCKS, 8, r['D_sum'], dtype=torch.float16, device=dev)
    detail = torch.zeros(Hp, END_BLOCK_CELLS, 8, Dt, dtype=tdt, device=dev)
    songs = [get_song(k) for k in r['keys']]
    for b, (song, c0, (start, _)) in enumerate(zip(songs, r['cell0'], r['crops'])):
        for j, l in enumerate(taps):
            _dslice(song.taps[l], c0, F.LOCAL_CELLS, audio[b, j])
        for j, l in enumerate(song_taps):
            _dslice_strided(song.taps[l], start, C, F.STREAM_STRIDE, song_cells[b, j])
    if mel:
        m = torch.zeros(B, FRAMES_LOCAL, 128, dtype=torch.float16, device=dev)
        for b, (song, c0) in enumerate(zip(songs, r['cell0'])):
            _dslice(song.mel10, c0 * F.FRAMES_PER_CELL, FRAMES_LOCAL, m[b])
        bt['mel'] = m
    for b, grp, lo, hi, rel in r['ref_reads']:
        gi = torch.as_tensor(grp).to(dev, non_blocking=True)
        ri = torch.as_tensor(rel).to(dev, non_blocking=True)
        for j, l in enumerate(taps):
            span = torch.zeros(hi - lo, 8, Dt, dtype=tdt, device=dev)
            _dslice(songs[b].taps[l], lo, hi - lo, span)
            ref_audio[b, gi, j] = span[ri]
    if r['holds']:  # one gather per core for the block rows and one for the detail cells (a Python loop with two
        # device copies per hold cost ~0.1 s per micro-batch on the main thread and starved the GPU)
        hb = torch.tensor([h[0] for h in r['holds']])
        b0s = torch.tensor([h[1] for h in r['holds']])
        d0s = torch.tensor([h[2] for h in r['holds']])
        for b in range(B):
            sel = torch.nonzero(hb == b)[:, 0]
            if not len(sel):
                continue
            _gather_rows(songs[b].summary, b0s[sel], F.MAX_END_BLOCKS, blocks, sel.to(dev), dev)
            _gather_rows(songs[b].taps[top], d0s[sel], END_BLOCK_CELLS, detail, sel.to(dev), dev)
    bt['audio'], bt['song_cells'], bt['ref_audio'] = audio, song_cells, ref_audio
    bt['hold_blocks'], bt['hold_detail'] = blocks, detail
    if quant:  # per-tap dequantisation scale [T, Dt]; applied on the device by dequant()
        if torch.is_tensor(scales):  # already stacked [T, Dt] on the device (training loop caches it)
            bt['tap_scale'] = scales
        else:
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


def collate(examples: list[F.CoreExample], songs: list[SongCache], taps: tuple[int, ...], top_tap: int, mel=False) -> dict:
    """Metadata + audio in one call on the CPU (tests, evaluation, sampling): the same two functions the
    training loop runs split across worker processes and the GPU."""
    quant = all(hasattr(s.taps[l], 'raw') for s in songs for l in taps)
    if not quant and any(hasattr(s.taps[l], 'raw') for s in songs for l in taps):
        raise ValueError('mixed int8 / float tap caches in one batch')
    infos = [SongInfo.of(s, b, taps[0]) for b, s in enumerate(songs)]
    bt = collate_meta(examples, infos, taps, top_tap, songs[0].taps[taps[0]].shape[-1], songs[0].summary.shape[-1], quant)
    scales = {l: songs[0].taps[l].scale for l in taps} if quant else None
    cpu = [_CpuSong(s, taps) for s in songs]
    return attach_audio(bt, lambda b: cpu[b], 'cpu', scales, mel=mel)


def dequant(bt: dict) -> dict:
    """Device-side expansion of the compact transport format: int8 local taps x scale, float16 -> float32.
    song_cells stay int8: the block encoder consumes them directly (Int8MixProj). Idempotent."""
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
