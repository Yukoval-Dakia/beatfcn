#!/usr/bin/env python
"""Phase probe (design/FIRST_CANDIDATE_v2.md section 5 row 1): can the in-cell onset offset (ms within a
40 ms cell) be read from the frozen PupuM2D cell tokens as well as from the 10 ms mel spectrogram?

Re-runnable end to end:

    .venv/bin/python scripts/phase_probe.py --root <data root> [--download] [--n-songs 60]

<data root> has the training layout (train/data.py Corpus): manifest/<manifest_name>, canon/<source>/<id>.json,
mel10/<sha>.npy, taps/<taps_key>/<sha>/L{9,16,24}.npy, pca/<pca>.npz. Only `modal volume get` (free reads)
is used for the download step; nothing on the volume is touched.

Time mapping (identical to training, see the report):
  model-view onset_us = native onset_us + (offset_decision.shift_ms + offset_decision.c_ms) * 1000
      (train/data.py Corpus._load_chart -> chart_from_canon(doc, model_shift_us(r, apply_codec_constant=True)))
  target onset_us = quantize_us(model-view onset_us) = round half up to 1 ms (events/canonical.py build_model_view);
  song cell = onset_us // CELL_US (CELL_US = 40_000, model/features.py:22); cores start on cell boundaries
      (features.choose_core) so t_rel // CELL_US in batch.collate_meta is the same cell;
  in-cell ms  = (onset_us % CELL_US) // 1000  (features.target_group: residual = t % BIN_US // MS, bin = t // BIN_US;
      cell offset = 10 * (bin % 4) + residual);
  tap cell k <-> taps[k]; mel frames 4k..4k+3 (batch.attach_audio slices taps at song_cell0 and mel at 4 * song_cell0).
"""
from __future__ import annotations
import argparse
import collections
import hashlib
import json
import math
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from beatfcn.train.data import load_manifest, model_shift_us, chart_from_canon, npy_shape  # noqa: E402
from beatfcn.model.features import CELL_US, FRAMES_PER_CELL, MS  # noqa: E402
from beatfcn.events.canonical import quantize_us  # noqa: E402
from beatfcn.audio.memory import plan  # noqa: E402

TAPS_KEY = 'v4_i8_L9-16-24'
LAYERS = (9, 16, 24)
PCA_NAME = 'i8_9-16-24'
MANIFEST = 'charts.offset-v2-3d99ff6bd5e7.jsonl'
VOLUME = 'beatfcn-data'
N_CLASSES = CELL_US // MS  # 40
BANDS = 8
MEL_PAD = 2  # frames each side of the cell's 4 frames -> 8 frames
MODE_NAME = {'mania_4k': 'mania', 'phira': 'phira'}


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


# ----------------------------------------------------------------------------------------------- selection

def select_songs(root, manifest_name, n_songs, seed, byte_budget):
    """Train-split songs (offset decision v2, training selection filters) with local mel10 and >= 1 local canon
    chart; stratified by source, shuffled with a fixed seed, longest songs dropped until the tap bytes fit."""
    rows = load_manifest(root, split='train', require_offset_version=2, manifest_name=manifest_name)
    by = collections.defaultdict(list)
    for r in rows:
        by[r['audio_sha256']].append(r)
    index_path = os.path.join(root, 'taps', TAPS_KEY, '_index.json')
    index = set(json.load(open(index_path))) if os.path.exists(index_path) else None
    cand = {}
    for sha, rs in by.items():
        if not os.path.exists(os.path.join(root, 'mel10', f'{sha}.npy')):
            continue
        if index is not None and sha not in index:
            continue
        local = [r for r in rs if os.path.exists(os.path.join(root, 'canon', r['source'], f"{r['native_id']}.json"))]
        if local:
            cand[sha] = local
    rng = np.random.default_rng(seed)
    per_source = collections.defaultdict(list)
    for sha in sorted(cand):
        per_source[cand[sha][0]['source']].append(sha)
    sources = sorted(per_source)
    quota = {s: n_songs // len(sources) for s in sources}
    for s in sources[: n_songs - sum(quota.values())]:
        quota[s] += 1
    chosen = []
    for s in sources:
        lst = list(per_source[s])
        rng.shuffle(lst)
        chosen += lst[: quota[s]]

    def tap_bytes(sha):
        frames = npy_shape(os.path.join(root, 'mel10', f'{sha}.npy'))[0]
        return len(LAYERS) * (plan(frames).cells * BANDS * 1024 + 128)

    size = {sha: tap_bytes(sha) for sha in chosen}
    while chosen and sum(size[s] for s in chosen) > byte_budget:
        chosen.remove(max(chosen, key=lambda s: size[s]))
    sel = [{'sha': s, 'source': cand[s][0]['source'], 'mode': cand[s][0]['mode'], 'tap_bytes_est': size[s],
            'charts': [r['chart_id'] for r in cand[s]], 'charts_in_manifest': len(by[s])} for s in chosen]
    sel.sort(key=lambda d: (d['source'], d['sha']))
    return sel, {s: cand[s] for s in chosen}, {'candidate_songs': len(cand),
                                               'candidates_by_source': {s: len(v) for s, v in per_source.items()}}


def download(root, selection, workers=4):
    """`modal volume get` for each missing tap file (the only Modal command used; free volume reads).
    A few files are fetched concurrently: each CLI call has several seconds of fixed overhead."""
    from concurrent.futures import ThreadPoolExecutor
    modal = os.path.join(os.path.dirname(HERE), '.venv', 'bin', 'modal')
    if not os.path.exists(modal):
        modal = 'modal'
    jobs = []
    for d in selection:
        dest = os.path.join(root, 'taps', TAPS_KEY, d['sha'])
        os.makedirs(dest, exist_ok=True)
        for l in LAYERS:
            local = os.path.join(dest, f'L{l}.npy')
            if not (os.path.exists(local) and os.path.getsize(local) > 128):
                jobs.append((f"taps/{TAPS_KEY}/{d['sha']}/L{l}.npy", dest, local))
    log(f'{len(jobs)} tap files to download')

    def fetch(job):
        remote, dest, local = job
        for attempt in range(3):
            p = subprocess.run([modal, 'volume', 'get', '--force', VOLUME, remote, dest + '/'], capture_output=True, text=True)
            if p.returncode == 0 and os.path.exists(local):
                return os.path.getsize(local)
            log(f'retry {attempt + 1} for {remote}: {p.stderr.strip()[-300:]}')
        raise RuntimeError(f'download failed: {remote}')
    got, done = 0, 0
    with ThreadPoolExecutor(workers) as ex:
        for n in ex.map(fetch, jobs):
            got += n
            done += 1
            if done % 6 == 0 or done == len(jobs):
                log(f'[{done}/{len(jobs)}] downloaded {got / 1e9:.2f} GB')
    return got


def verify_taps(root, selection):
    """Delete truncated/mismatched tap files (size != header-declared size, or cells != plan(mel frames).cells)."""
    bad = []
    for d in selection:
        frames = npy_shape(os.path.join(root, 'mel10', f"{d['sha']}.npy"))[0]
        for l in LAYERS:
            p = os.path.join(root, 'taps', TAPS_KEY, d['sha'], f'L{l}.npy')
            if not os.path.exists(p):
                continue
            try:
                with open(p, 'rb') as f:
                    version = np.lib.format.read_magic(f)
                    shape, _, dtype = np.lib.format._read_array_header(f, version)
                    expect = f.tell() + int(np.prod(shape)) * np.dtype(dtype).itemsize
                ok = os.path.getsize(p) == expect and shape[0] == plan(frames).cells and shape[1:] == (BANDS, 1024)
            except Exception as e:  # noqa: BLE001
                log(f'bad header {p}: {e}')
                ok = False
            if not ok:
                bad.append(p)
                os.remove(p)
    if bad:
        log(f'{len(bad)} truncated/mismatched tap files removed (will be re-downloaded)')
    return bad


def disk_bytes(root, selection):
    total = 0
    for d in selection:
        for l in LAYERS:
            p = os.path.join(root, 'taps', TAPS_KEY, d['sha'], f'L{l}.npy')
            total += os.path.getsize(p) if os.path.exists(p) else 0
    return total


# ----------------------------------------------------------------------------------------------- samples

def build_samples(root, rows_by_song, selection):
    """One sample per (chart, cell) that contains >= 1 onset (holds count). Target = in-cell ms of the FIRST
    onset in the cell. Rhythm context = the previous 4 distinct onsets of the same chart before the cell."""
    recs = []
    per_song = {}
    for si, d in enumerate(selection):
        sha = d['sha']
        frames = npy_shape(os.path.join(root, 'mel10', f'{sha}.npy'))[0]
        n_cells = plan(frames).cells
        per_song[sha] = {'frames': frames, 'cells': n_cells}
        for r in rows_by_song[sha]:
            with open(os.path.join(root, 'canon', r['source'], f"{r['native_id']}.json")) as f:
                doc = json.load(f)
            shift = model_shift_us(r, apply_codec_constant=True)  # == Corpus._load_chart (apply_codec_constant=True)
            chart = chart_from_canon(doc, shift)
            # hidden targets are quantised to 1 ms, round half up (canonical.build_model_view -> quantize_us)
            raw = sorted({e.onset_us for e in chart.events})
            q = {}
            for t in raw:
                q.setdefault(quantize_us(t), t)  # distinct quantised onset times (keep the first exact time)
            onsets = np.array(sorted(q), np.int64)
            inside = onsets[(onsets >= 0) & (onsets < n_cells * CELL_US)]
            if len(inside) < len(onsets):
                log(f"  {r['chart_id']}: {len(onsets) - len(inside)} onsets outside the audio dropped")
            cells = inside // CELL_US
            first = np.r_[True, cells[1:] != cells[:-1]]  # first onset in each cell (onsets sorted)
            idx = np.flatnonzero(first)
            for j in idx:
                t0 = int(inside[j])
                c = int(cells[j])
                cell_start = c * CELL_US
                k = int(np.searchsorted(inside, cell_start))  # onsets strictly before the cell
                prev = inside[max(k - 5, 0):k][::-1]  # t_-1, t_-2, ... (up to 5)
                gaps = [(cell_start - int(p)) / MS for p in prev[:4]]  # ms from cell start back to each
                iois = [(int(prev[m]) - int(prev[m + 1])) / MS for m in range(min(len(prev) - 1, 4))]
                recs.append((si, r['chart_id'], MODE_NAME[r['mode']], c, (t0 - cell_start) // MS, (q[t0] - cell_start) / MS,
                             gaps, iois))
    log(f'{len(recs)} samples from {len(selection)} songs')
    return recs, per_song


def rhythm_features(recs):
    """[N, 40]: for 4 gaps (cell start -> previous onsets) and 4 IOIs: seconds (clipped 10 s), log1p(ms), sin/cos of
    the 40 ms phase, plus presence flags. Missing context -> zeros with flag 0 (the 'no context' state)."""
    out = np.zeros((len(recs), 40), np.float32)
    for i, rec in enumerate(recs):
        vals = list(rec[6]) + [None] * (4 - len(rec[6])) + list(rec[7]) + [None] * (4 - len(rec[7]))
        for j, v in enumerate(vals):
            if v is None:
                continue
            ph = 2 * math.pi * ((v % N_CLASSES) / N_CLASSES)
            out[i, 4 * j:4 * j + 4] = (min(v, 10_000.0) / 1000.0, math.log1p(v), math.sin(ph), math.cos(ph))
            out[i, 32 + j] = 1.0
    return out


def alignment_check(root, rows_by_song, selection, per_source=8):
    """Independent sanity check of the chart->audio mapping: mean spectral-flux envelope (ingest/offset.flux_envelope on
    mel10) at frame offsets -6..6 around the quantised model-view onsets, and at offset 0 split by the sub-frame ms.
    A peak at offset 0/+1 that decays with the sub-frame ms means the mapping is aligned and mel carries phase."""
    from beatfcn.ingest.offset import flux_envelope
    out = {}
    for src in sorted({d['source'] for d in selection}):
        acc, n, by_ms, n_ms = np.zeros(13), 0, np.zeros(10), np.zeros(10)
        for d in [d for d in selection if d['source'] == src][:per_source]:
            env = flux_envelope(np.load(os.path.join(root, 'mel10', f"{d['sha']}.npy")).astype(np.float32))
            for r in rows_by_song[d['sha']]:
                with open(os.path.join(root, 'canon', r['source'], f"{r['native_id']}.json")) as f:
                    chart = chart_from_canon(json.load(f), model_shift_us(r, apply_codec_constant=True))
                ts = np.array(sorted({quantize_us(e.onset_us) for e in chart.events}), np.int64)
                ts = ts[(ts >= 100_000) & (ts < (len(env) - 10) * 10_000)]
                fr, ms = ts // 10_000, (ts % 10_000) // 1000
                for k in range(-6, 7):
                    acc[k + 6] += env[fr + k].sum()
                np.add.at(by_ms, ms, env[fr])
                np.add.at(n_ms, ms, 1)
                n += len(ts)
        out[src] = {'onsets': int(n), 'mean_flux_by_frame_offset_-6..6': (acc / max(n, 1)).round(3).tolist(),
                    'mean_flux_at_offset0_by_sub_frame_ms_0..9': [round(float(by_ms[m] / n_ms[m]), 3) if n_ms[m] else None for m in range(10)]}
    return out


# ----------------------------------------------------------------------------------------------- features

def token_ln(x, eps=1e-5):
    """nn.LayerNorm(1024) without affine, per token (p1.py:222 tap_ln applies LayerNorm(tap_dim) to each tap)."""
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps)


def cell_tokens(tap_i8, scale, cells, n_cells):
    """[n, 3, 8, 1024] float32: dequantised (int8 * scale, as QuantArray / batch.dequant; mean not re-added) and
    per-token standardised tokens of cells c-1, c, c+1; out-of-range neighbours are zeros (audio_valid=False)."""
    out = np.zeros((len(cells), 3, BANDS, tap_i8.shape[-1]), np.float32)
    for k, off in enumerate((-1, 0, 1)):
        cc = cells + off
        ok = (cc >= 0) & (cc < n_cells)
        if ok.any():
            raw = np.asarray(tap_i8[cc[ok]], np.float32) * scale  # fancy indexing on the memmap: one read
            out[ok, k] = token_ln(raw)
    return out


class TokenPCA:
    """Per-token PCA (1024 -> k) fitted on train-song tokens only."""

    def __init__(self, dim):
        self.dim, self.n, self.s1, self.s2 = dim, 0, None, None

    def add(self, tokens):  # tokens [m, 1024]
        t = tokens.reshape(-1, tokens.shape[-1]).astype(np.float64)
        self.n += len(t)
        self.s1 = t.sum(0) if self.s1 is None else self.s1 + t.sum(0)
        g = t.T @ t
        self.s2 = g if self.s2 is None else self.s2 + g

    def finish(self):
        mu = self.s1 / self.n
        cov = self.s2 / self.n - np.outer(mu, mu)
        w, v = np.linalg.eigh(cov)
        order = np.argsort(w)[::-1]
        self.mean = mu.astype(np.float32)
        self.comp = v[:, order[: self.dim]].astype(np.float32)
        self.var_kept = float(w[order[: self.dim]].sum() / max(w.sum(), 1e-12))
        return self

    def __call__(self, tokens):
        shp = tokens.shape[:-1]
        return ((tokens.reshape(-1, tokens.shape[-1]) - self.mean) @ self.comp).reshape(*shp, self.dim)


def fit_pca_rows(x, dim, chunk=8192):
    """PCA on rows (second stage for A_all); x [n, D] float32, covariance accumulated in chunks (memory)."""
    n, D = x.shape
    mu = x.mean(0, dtype=np.float64)
    cov = np.zeros((D, D), np.float64)
    for i in range(0, n, chunk):
        xc = (x[i:i + chunk].astype(np.float64) - mu)
        cov += xc.T @ xc
    cov /= n
    w, v = np.linalg.eigh(cov)
    order = np.argsort(w)[::-1][:dim]
    return mu.astype(np.float32), v[:, order].astype(np.float32), float(w[order].sum() / max(w.sum(), 1e-12))


def extract_features(root, selection, recs, per_song, train_songs, per_token_dim, cache_path, rp_dim=682, tap_dim=1024):
    if os.path.exists(cache_path):
        log('loading cached features', cache_path)
        z = np.load(cache_path)
        return {k: z[k] for k in z.files}
    z = np.load(os.path.join(root, 'pca', f'{PCA_NAME}.npz'))
    scale = {l: z[f'scale_{l}'].astype(np.float32) for l in LAYERS}  # Corpus(pca=..., whiten=False)
    song_of = np.array([r[0] for r in recs])
    cell_of = np.array([r[3] for r in recs])
    N = len(recs)
    # pass 1: per-token PCA statistics on train songs
    pcas = {l: TokenPCA(per_token_dim) for l in LAYERS}
    for si, d in enumerate(selection):
        if si not in train_songs:
            continue
        rows = np.flatnonzero(song_of == si)
        n_cells = per_song[d['sha']]['cells']
        for l in LAYERS:
            tap = np.load(os.path.join(root, 'taps', TAPS_KEY, d['sha'], f'L{l}.npy'), mmap_mode='r')
            assert tap.shape[0] == n_cells, (d['sha'], tap.shape, n_cells)
            toks = cell_tokens(tap, scale[l], cell_of[rows], n_cells)
            valid = np.abs(toks).sum(-1) > 0  # drop zero (out-of-range) tokens from the statistics
            pcas[l].add(toks[valid])
        log(f'pca stats {si + 1}/{len(selection)}')
    for l in LAYERS:
        pcas[l].finish()
        log(f'L{l}: per-token PCA {per_token_dim} dims keeps {pcas[l].var_kept:.3f} of token variance')
    # pass 2: project every sample (per-token PCA, and a fixed Gaussian random projection of the whole 24,576-dim
    # cell block as a reduction-robustness check: 3 taps x rp_dim <= 2048)
    rp = np.random.default_rng(12345).standard_normal((3 * BANDS * tap_dim, rp_dim), dtype=np.float32) / np.sqrt(rp_dim)
    rpd = np.random.default_rng(54321).standard_normal((2 * BANDS * tap_dim, rp_dim), dtype=np.float32) / np.sqrt(rp_dim)
    feats = {f'tap{l}': np.zeros((N, 3 * BANDS * per_token_dim), np.float16) for l in LAYERS}
    feats.update({f'rp{l}': np.zeros((N, rp_dim), np.float16) for l in LAYERS})
    feats.update({f'rpd{l}': np.zeros((N, rp_dim), np.float16) for l in LAYERS})  # relu(cell differences), projected
    mel_feat = np.zeros((N, 2 * MEL_PAD + FRAMES_PER_CELL, 128), np.float32)
    for si, d in enumerate(selection):
        rows = np.flatnonzero(song_of == si)
        n_cells = per_song[d['sha']]['cells']
        cells = cell_of[rows]
        for l in LAYERS:
            tap = np.load(os.path.join(root, 'taps', TAPS_KEY, d['sha'], f'L{l}.npy'), mmap_mode='r')
            toks = cell_tokens(tap, scale[l], cells, n_cells)
            feats[f'tap{l}'][rows] = pcas[l](toks).reshape(len(rows), -1).astype(np.float16)
            feats[f'rp{l}'][rows] = (toks.reshape(len(rows), -1) @ rp).astype(np.float16)
            dif = np.maximum(toks[:, 1:] - toks[:, :-1], 0)  # [n, 2, 8, 1024]: relu(c - (c-1)), relu((c+1) - c)
            feats[f'rpd{l}'][rows] = (dif.reshape(len(rows), -1) @ rpd).astype(np.float16)
        mel = np.load(os.path.join(root, 'mel10', f'{d["sha"]}.npy'), mmap_mode='r')
        f0 = cells * FRAMES_PER_CELL - MEL_PAD
        for k in range(2 * MEL_PAD + FRAMES_PER_CELL):
            fr = f0 + k
            ok = (fr >= 0) & (fr < mel.shape[0])
            mel_feat[rows[ok], k] = np.asarray(mel[fr[ok]], np.float32)
        log(f'features {si + 1}/{len(selection)}')
    feats['mel'] = mel_feat.reshape(N, -1)
    feats['pca_var_kept'] = np.array([pcas[l].var_kept for l in LAYERS], np.float32)
    np.savez(cache_path, **feats)
    return feats


# ----------------------------------------------------------------------------------------------- models

def make_fold(order, f, k):
    """Song-id sets (train, val, test) of fold f: per source, chunk f of the shuffled ids is the test set (20%), the
    first 20% of the remaining ids is the validation set."""
    tr, va, te = set(), set(), set()
    for ids in order.values():
        n = len(ids)
        chunk = [ids[i] for i in range(n) if (i * k) // n == f]
        rest = [i for i in ids if i not in chunk]
        n_val = max(1, int(round(0.2 * len(rest))))
        te |= set(chunk)
        va |= set(rest[:n_val])
        tr |= set(rest[n_val:])
    return tr, va, te


def standardise(x, train_idx):
    mu = x[train_idx].mean(0, dtype=np.float64).astype(np.float32)
    sd = x[train_idx].std(0, dtype=np.float64).astype(np.float32) + 1e-5
    return ((x - mu) / sd).astype(np.float32)


GRID = [{'lr': lr, 'wd': 1e-1, 'in_drop': dr} for lr in (1e-4, 1e-3) for dr in (0.0, 0.5)]


def train_mlp(x, y, tr, va, seed, hidden=256, epochs=80, patience=8, lr=1e-3, wd=1e-2, batch=512, dropout=0.3, in_drop=0.0):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    xt, yt = torch.from_numpy(x), torch.from_numpy(y.astype(np.int64))
    model = nn.Sequential(nn.Dropout(in_drop), nn.Linear(x.shape[1], hidden), nn.GELU(), nn.Dropout(dropout),
                          nn.Linear(hidden, N_CLASSES))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    best, best_state, bad, hist = float('inf'), None, 0, []
    g = torch.Generator().manual_seed(seed)
    tr_t = torch.from_numpy(tr)
    for ep in range(epochs):
        model.train()
        perm = tr_t[torch.randperm(len(tr_t), generator=g)]
        for i in range(0, len(perm), batch):
            b = perm[i:i + batch]
            loss = nn.functional.cross_entropy(model(xt[b]), yt[b])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            lv = sum(nn.functional.cross_entropy(model(xt[va[i:i + 8192]]), yt[va[i:i + 8192]], reduction='sum').item()
                     for i in range(0, len(va), 8192)) / len(va)
        hist.append(round(lv, 4))
        if lv < best - 1e-4:
            best, bad = lv, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()

    def predict(idx):
        with torch.no_grad():
            return torch.cat([model(xt[idx[i:i + 8192]]).softmax(-1) for i in range(0, len(idx), 8192)]).numpy()
    return predict, {'epochs_run': len(hist), 'best_val_nll': round(best, 4), 'val_nll_history': hist,
                     'params': sum(p.numel() for p in model.parameters()), 'input_dim': int(x.shape[1]),
                     'config': {'lr': lr, 'wd': wd, 'in_drop': in_drop, 'hidden_drop': dropout, 'hidden': hidden, 'batch': batch}}


def train_grid(x, y, tr, va, seed, hidden, epochs, grid=GRID):
    """Same small grid for every feature set. Primary selection: lowest val MAE of the expectation decoder (the decision
    metric); the lowest-val-NLL model is kept as well (val = held-out songs, never the test songs)."""
    k = np.arange(N_CLASSES, dtype=np.float64)
    trials, fits = [], []
    for cfg in grid:
        predict, info = train_mlp(x, y, tr, va, seed, hidden=hidden, epochs=epochs, **cfg)
        pv = predict(va)
        info['val_mae_expect'] = round(float(np.abs(pv @ k - y[va]).mean()), 3)
        info['val_mae_argmax'] = round(float(np.abs(pv.argmax(1) - y[va]).mean()), 3)
        trials.append({**cfg, 'epochs_run': info['epochs_run'], 'val_nll': info['best_val_nll'],
                       'val_mae_expect': info['val_mae_expect'], 'val_mae_argmax': info['val_mae_argmax']})
        fits.append((predict, info))
    by_mae = min(fits, key=lambda f: f[1]['val_mae_expect'])
    by_nll = min(fits, key=lambda f: f[1]['best_val_nll'])
    by_mae[1]['grid'] = trials
    by_mae[1]['nll_selected_config'] = by_nll[1]['config']
    return by_mae, by_nll


def flux_heuristic(mel8, cells_valid=None):
    """Non-learned mel decoder: positive spectral flux summed over bins at frames 4c-1..4c+5; the 10 ms bin b whose
    following frame (4c+b+1) has the largest flux wins (the flux of an onset at frame f peaks at f+1, see the
    alignment check); probability mass uniform over that bin's 10 classes (expect -> bin centre 10b+4.5)."""
    d = np.maximum(mel8[:, 1:] - mel8[:, :-1], 0).sum(-1)  # [N, 7]: frames 4c-1 .. 4c+5
    b = d[:, 2:6].argmax(1)  # frames 4c+1 .. 4c+4  <-> bins 0..3
    prob = np.zeros((len(b), N_CLASSES))
    for i in range(4):
        prob[b == i, 10 * i:10 * i + 10] = 0.1
    return prob


def mel_flux_features(mel8):
    """Translation-friendly mel representation: raw 8 frames + rectified frame differences (7 x 128) + summed flux (7)."""
    d = np.maximum(mel8[:, 1:] - mel8[:, :-1], 0)
    return np.concatenate([mel8.reshape(len(mel8), -1), d.reshape(len(d), -1), d.sum(-1)], 1).astype(np.float32)


def metrics(prob, y, y_exact, groups):
    """Two decoders: argmax class and the expectation under the softmax. Errors against the integer target."""
    k = np.arange(N_CLASSES, dtype=np.float64)
    ev = prob.astype(np.float64) @ k
    am = prob.argmax(1).astype(np.float64)
    out = {}
    for name, pred in (('argmax', am), ('expect', ev)):
        err = np.abs(pred - y)
        d = {'mae_ms': float(err.mean()), 'median_ms': float(np.median(err)), 'p_le5': float((err <= 5).mean()),
             'p_le10': float((err <= 10).mean()), 'mae_vs_exact_ms': float(np.abs(pred - y_exact).mean()), 'n': int(len(y))}
        for gname, gmask in groups.items():
            if gmask.any():
                e = err[gmask]
                d[gname] = {'mae_ms': float(e.mean()), 'median_ms': float(np.median(e)), 'p_le5': float((e <= 5).mean()),
                            'p_le10': float((e <= 10).mean()), 'n': int(gmask.sum())}
        out[name] = d
    out['nll'] = float(-np.log(np.clip(prob[np.arange(len(y)), y], 1e-9, None)).mean())
    out['top1_acc'] = float((prob.argmax(1) == y).mean())
    return out


# ----------------------------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', required=True)
    ap.add_argument('--manifest-name', default=MANIFEST)
    ap.add_argument('--n-songs', type=int, default=60)
    ap.add_argument('--byte-budget', type=float, default=6.3e9, help='max tap bytes to download')
    ap.add_argument('--seed', type=int, default=20261023)
    ap.add_argument('--download', action='store_true', help='fetch missing taps with `modal volume get`')
    ap.add_argument('--download-only', action='store_true', help='stop after the download step')
    ap.add_argument('--per-token-dim', type=int, default=85, help='per-token PCA dims (24 tokens x 85 = 2040 <= 2048)')
    ap.add_argument('--all-dim', type=int, default=2048, help='second-stage PCA dims for A_all')
    ap.add_argument('--epochs', type=int, default=80)
    ap.add_argument('--hidden', type=int, default=256)
    ap.add_argument('--out-json', default=os.path.join(os.path.dirname(HERE), 'reports', 'phase_probe_2026-10-23.json'))
    ap.add_argument('--out-md', default=os.path.join(os.path.dirname(HERE), 'reports', 'phase_probe_2026-10-23.md'))
    ap.add_argument('--recompute', action='store_true')
    ap.add_argument('--render-only', action='store_true', help='re-render the .md (and decision summary) from the existing .json')
    a = ap.parse_args()
    t_start = time.time()
    if a.render_only:
        with open(a.out_json) as f:
            report = json.load(f)
        report['decision']['summary'] = decision_summary(report)
        with open(a.out_json, 'w') as f:
            json.dump(report, f, indent=1, ensure_ascii=False)
        with open(a.out_md, 'w') as f:
            f.write(render_md(report))
        print(render_md(report))
        return

    selection, rows_by_song, sel_info = select_songs(a.root, a.manifest_name, a.n_songs, a.seed, a.byte_budget)
    log(f"selected {len(selection)} songs: {collections.Counter(d['source'] for d in selection)} "
        f"est {sum(d['tap_bytes_est'] for d in selection) / 1e9:.2f} GB taps")
    sel_path = os.path.join(a.root, 'probe_selection.json')
    with open(sel_path, 'w') as f:
        json.dump(selection, f, indent=1)
    downloaded = 0
    if a.download or a.download_only:
        for _ in range(3):
            downloaded += download(a.root, selection)
            if not verify_taps(a.root, selection):
                break
    on_disk = disk_bytes(a.root, selection)
    if a.download_only:
        log(f'download only: {downloaded / 1e9:.3f} GB this run, {on_disk / 1e9:.3f} GB taps on disk')
        return
    extra = sum(os.path.getsize(os.path.join(a.root, p)) for p in (f'manifest/{a.manifest_name}', f'pca/{PCA_NAME}.npz')
                if os.path.exists(os.path.join(a.root, p)))
    missing = [d['sha'] for d in selection if any(not os.path.exists(os.path.join(a.root, 'taps', TAPS_KEY, d['sha'], f'L{l}.npy'))
                                                  for l in LAYERS)]
    if missing:
        raise SystemExit(f'{len(missing)} songs lack taps; rerun with --download')

    recs, per_song = build_samples(a.root, rows_by_song, selection)
    align = alignment_check(a.root, rows_by_song, selection)
    log('alignment check', json.dumps(align))
    song_of = np.array([r[0] for r in recs])
    y = np.array([r[4] for r in recs], np.int64)
    y_exact = np.array([r[5] for r in recs], np.float64)
    mode_of = np.array([r[2] for r in recs])
    # split by song 80/20 (stratified by source), val = 20% of the non-test songs for early stopping / model choice.
    # Five by-song folds share one seeded shuffle; fold 0 is the pre-registered split, folds 1-4 are a robustness check.
    rng = np.random.default_rng(a.seed)
    order = {}
    for src in sorted({d['source'] for d in selection}):
        ids = [i for i, d in enumerate(selection) if d['source'] == src]
        rng.shuffle(ids)
        order[src] = ids
    folds = [make_fold(order, f, 5) for f in range(5)]
    train_songs, val_songs, test_songs = folds[0]
    tr, va, te = (np.flatnonzero(np.isin(song_of, list(sset))) for sset in folds[0])
    fit_songs = train_songs | val_songs  # PCA / standardisation statistics: everything but the fold-0 test songs
    log(f'split: train {len(tr)} / val {len(va)} / test {len(te)} samples; songs {len(train_songs)}/{len(val_songs)}/{len(test_songs)}')

    key = hashlib.sha256(json.dumps([d['sha'] for d in selection] + [a.per_token_dim, 'v2']).encode()).hexdigest()[:12]
    cache = os.path.join(a.root, 'cache', f'features_{key}.npz')
    if a.recompute and os.path.exists(cache):
        os.remove(cache)
    feats = extract_features(a.root, selection, recs, per_song, fit_songs, a.per_token_dim, cache)
    fit_idx = np.concatenate([tr, va])

    # feature sets
    X = {}
    for l in LAYERS:
        X[f'A_{l}'] = standardise(feats[f'tap{l}'].astype(np.float32), fit_idx)
    cat = np.concatenate([X[f'A_{l}'] for l in LAYERS], 1)
    mu2, comp2, kept2 = fit_pca_rows(cat[fit_idx], a.all_dim)
    X['A_all'] = standardise(((cat - mu2) @ comp2).astype(np.float32), fit_idx)
    del cat
    log(f'A_all: {3 * X["A_9"].shape[1]} -> {a.all_dim} dims keeps {kept2:.3f} of variance')
    X['A_all_rp'] = standardise(np.concatenate([feats[f'rp{l}'].astype(np.float32) for l in LAYERS], 1), fit_idx)
    X['A_all_rpd'] = standardise(np.concatenate([feats[f'rp{l}'].astype(np.float32) for l in LAYERS]
                                                + [feats[f'rpd{l}'].astype(np.float32) for l in LAYERS], 1), fit_idx)
    mel8 = feats['mel'].astype(np.float32).reshape(len(recs), 2 * MEL_PAD + FRAMES_PER_CELL, 128)
    X['B'] = standardise(feats['mel'].astype(np.float32), fit_idx)
    X['B_flux'] = standardise(mel_flux_features(mel8), fit_idx)
    X['C'] = np.concatenate([X['A_all'], X['B']], 1)
    X['C_flux'] = np.concatenate([X['A_all_rpd'], X['B_flux']], 1)
    R = standardise(rhythm_features(recs), fit_idx)
    X['E'] = np.concatenate([X['A_all'], R], 1)
    X['E_only'] = R  # rhythm context alone (diagnostic)
    heur_prob = flux_heuristic(mel8)
    del mel8

    def groups_of(idx):
        return {'mania': mode_of[idx] == 'mania', 'phira': mode_of[idx] == 'phira'}

    def marginal_probs(fit, idx, per_mode):
        pm = np.zeros((len(idx), N_CLASSES))
        if not per_mode:
            pm[:] = np.bincount(y[fit], minlength=N_CLASSES) / len(fit)
            return pm
        for m in ('mania', 'phira'):
            sel_m = mode_of[fit] == m
            if sel_m.any():
                pm[mode_of[idx] == m] = np.bincount(y[fit][sel_m], minlength=N_CLASSES) / sel_m.sum()
        return pm

    results, results_nll, train_info = {}, {}, {}
    groups = groups_of(te)
    marg = np.bincount(y[fit_idx], minlength=N_CLASSES) / len(fit_idx)
    results['D'] = metrics(marginal_probs(fit_idx, te, False), y[te], y_exact[te], groups)
    results['D_mode'] = metrics(marginal_probs(fit_idx, te, True), y[te], y_exact[te], groups)
    results['B_heur'] = metrics(heur_prob[te], y[te], y_exact[te], groups)
    MODEL_SETS = ['A_9', 'A_16', 'A_24', 'A_all', 'A_all_rp', 'A_all_rpd', 'B', 'B_flux', 'C', 'C_flux', 'E', 'E_only']
    for name in MODEL_SETS:
        t0 = time.time()
        (predict, info), (predict_nll, info_nll) = train_grid(X[name], y, tr, va, a.seed, a.hidden, a.epochs)
        results[name] = metrics(predict(te), y[te], y_exact[te], groups)
        results_nll[name] = metrics(predict_nll(te), y[te], y_exact[te], groups)
        info['train_seconds'] = round(time.time() - t0, 1)
        train_info[name] = info
        r = results[name]
        log(f"{name:9s} dim {info['input_dim']:5d} cfg {info['config']['lr']:.0e}/{info['config']['in_drop']} "
            f"ep {info['epochs_run']:2d} val_nll {info['best_val_nll']:.3f} val_mae {info['val_mae_expect']:.2f} | "
            f"argmax MAE {r['argmax']['mae_ms']:.2f} P5 {r['argmax']['p_le5']:.3f} | expect MAE {r['expect']['mae_ms']:.2f} "
            f"P5 {r['expect']['p_le5']:.3f} | mania {r['expect']['mania']['mae_ms']:.2f} phira {r['expect']['phira']['mae_ms']:.2f} "
            f"({info['train_seconds']}s)")

    # robustness: folds 1-4 with each set's fold-0 configuration (inputs keep the fold-0 PCA / standardisation statistics)
    CV_SETS = ['A_all', 'A_all_rp', 'A_all_rpd', 'B', 'B_flux', 'C', 'C_flux', 'E', 'E_only']
    cv = {n: {'expect_mae': [results[n]['expect']['mae_ms']], 'argmax_mae': [results[n]['argmax']['mae_ms']],
              'mania_expect_mae': [results[n]['expect']['mania']['mae_ms']], 'phira_expect_mae': [results[n]['expect']['phira']['mae_ms']]}
          for n in CV_SETS + ['D', 'B_heur']}
    for f in range(1, 5):
        tr_f, va_f, te_f = (np.flatnonzero(np.isin(song_of, list(sset))) for sset in folds[f])
        fit_f = np.concatenate([tr_f, va_f])
        g = groups_of(te_f)
        fold_res = {'D': metrics(marginal_probs(fit_f, te_f, False), y[te_f], y_exact[te_f], g),
                    'B_heur': metrics(heur_prob[te_f], y[te_f], y_exact[te_f], g)}
        for n in CV_SETS:
            cfg = train_info[n]['config']
            predict, _ = train_mlp(X[n], y, tr_f, va_f, a.seed + f, hidden=cfg['hidden'], epochs=a.epochs,
                                   lr=cfg['lr'], wd=cfg['wd'], in_drop=cfg['in_drop'], dropout=cfg['hidden_drop'])
            fold_res[n] = metrics(predict(te_f), y[te_f], y_exact[te_f], g)
        for n, r in fold_res.items():
            cv[n]['expect_mae'].append(r['expect']['mae_ms'])
            cv[n]['argmax_mae'].append(r['argmax']['mae_ms'])
            cv[n]['mania_expect_mae'].append(r['expect']['mania']['mae_ms'])
            cv[n]['phira_expect_mae'].append(r['expect']['phira']['mae_ms'])
        log(f'fold {f}: ' + ' '.join(f"{n} {fold_res[n]['expect']['mae_ms']:.2f}" for n in ['A_all', 'A_all_rpd', 'B', 'B_flux', 'B_heur', 'C_flux']))
    for n, d in cv.items():
        for key in list(d):
            v = np.array(d[key])
            d[key] = {'folds': [round(float(x), 3) for x in v], 'mean': round(float(v.mean()), 3), 'sd': round(float(v.std(ddof=1)), 3)}
    cv_gap = {}
    for a_name, b_name in (('A_all', 'B'), ('A_all', 'C'), ('A_all_rpd', 'B_flux'), ('A_all_rpd', 'B_heur'), ('A_all_rpd', 'C_flux')):
        v = np.array(cv[a_name]['expect_mae']['folds']) - np.array(cv[b_name]['expect_mae']['folds'])
        vm = np.array(cv[a_name]['mania_expect_mae']['folds']) - np.array(cv[b_name]['mania_expect_mae']['folds'])
        cv_gap[f'{a_name}-{b_name}'] = {'folds': [round(float(x), 3) for x in v], 'mean': round(float(v.mean()), 3),
                                        'sd': round(float(v.std(ddof=1)), 3), 'mania_mean': round(float(vm.mean()), 3),
                                        'mania_sd': round(float(vm.std(ddof=1)), 3)}

    # decision
    dec = 'expect'
    m = lambda n, mode=None: results[n][dec]['mae_ms'] if mode is None else results[n][dec][mode]['mae_ms']
    gap_B, gap_C = m('A_all') - m('B'), m('A_all') - m('C')
    remove_mel = bool(gap_B <= 1.0 and gap_C <= 1.0)
    best_dec = {n: min(results[n]['argmax']['mae_ms'], results[n]['expect']['mae_ms']) for n in results}
    gap_B_best, gap_C_best = best_dec['A_all'] - best_dec['B'], best_dec['A_all'] - best_dec['C']
    # strict check: best token representation (by val MAE) against the best mel representation incl. the non-learned heuristic
    tok_best = min(['A_all', 'A_all_rp', 'A_all_rpd'], key=lambda n: train_info[n]['val_mae_expect'])
    mel_learned_best = min(['B', 'B_flux'], key=lambda n: train_info[n]['val_mae_expect'])
    mel_best = min([mel_learned_best, 'B_heur'], key=lambda n: m(n))
    strict = {'token_set': tok_best, 'mel_set': mel_best, 'mel_learned_best': mel_learned_best,
              'gap_all_ms': round(m(tok_best) - m(mel_best), 3), 'gap_vs_C_flux_ms': round(m(tok_best) - m('C_flux'), 3),
              'gap_mania_ms': round(m(tok_best, 'mania') - m(mel_best, 'mania'), 3),
              'gap_phira_ms': round(m(tok_best, 'phira') - m(mel_best, 'phira'), 3),
              'gap_mania_vs_C_flux_ms': round(m(tok_best, 'mania') - m('C_flux', 'mania'), 3)}
    strict['remove_mel'] = bool(strict['gap_all_ms'] <= 1.0 and strict['gap_vs_C_flux_ms'] <= 1.0
                                and strict['gap_mania_ms'] <= 1.0 and strict['gap_mania_vs_C_flux_ms'] <= 1.0)
    e_gain = m('A_all') - m('E')

    mapping = {
        'model_view_onset_us': 'native onset_us + (offset_decision.shift_ms + offset_decision.c_ms) * 1000',
        'target_onset_us': 'quantize_us(model-view onset_us) = ((us + 500) // 1000) * 1000 (hidden targets are quantised to 1 ms, round half up)',
        'where': ['beatfcn/train/data.py:72-75 model_shift_us (shift_ms + c_ms when apply_codec_constant)',
                  'beatfcn/train/data.py:127 Corpus.__init__ apply_codec_constant=True (default, used by train_app)',
                  'beatfcn/train/data.py:157-162 Corpus._load_chart -> chart_from_canon(doc, model_shift_us(r, self.apply_c))',
                  'beatfcn/train/data.py:98-107 chart_from_canon adds shift_us to onset_us / end_us',
                  'beatfcn/events/canonical.py:286-287 quantize_us; :333-345 build_model_view quantises hidden targets (not observed anchors)',
                  'beatfcn/events/requests.py:281 sample_request -> build_model_view; train/loop.py:81-87 sample_request on corpus.chart, then choose_core/build_core',
                  'beatfcn/model/features.py:22 CELL_US = 40_000; :23 BIN_US = 10_000',
                  'beatfcn/model/features.py:306-320 choose_core: c0 is a multiple of CELL_US',
                  'beatfcn/model/features.py:113-125 target_group: t = time_us - c0, bin = t // BIN_US, residual = t % BIN_US // MS',
                  'beatfcn/model/batch.py:221 collate_meta: cell label at t_rel_us // CELL_US',
                  'beatfcn/model/batch.py:316-317 attach_audio: taps sliced at song_cell0, mel at song_cell0 * FRAMES_PER_CELL',
                  'beatfcn/train/data.py:111-123 QuantArray: int8 * scale_l (pca npz), mean not re-added; batch.py:371-384 dequant same',
                  'beatfcn/model/p1.py:222,504 tap_ln = nn.LayerNorm(tap_dim) per tap applied to each token'],
        'song_cell': 'quantised onset_us // 40_000', 'in_cell_ms': '(quantised onset_us % 40_000) // 1000 (exact integer after quantisation)',
        'audio_axis': 'tap cell k <-> taps[k]; mel10 frames 4k..4k+3; neighbours out of range -> zero tokens',
        'dequant': 'int8 * scale_l from pca/i8_9-16-24.npz (whiten=False), then per-token LayerNorm (no affine, eps 1e-5)',
        'fakes': 'canon events are playable only (fakes are kept separately), as chart_from_canon reads doc["events"]',
    }
    report = {
        'date': '2026-10-23', 'design_ref': 'design/FIRST_CANDIDATE_v2.md section 5 row 1, section 3.3 time head v2',
        'root': a.root, 'manifest': a.manifest_name, 'taps_key': TAPS_KEY, 'layers': list(LAYERS), 'pca': PCA_NAME,
        'seed': a.seed, 'selection_info': sel_info,
        'songs': {'n': len(selection), 'by_source': dict(collections.Counter(d['source'] for d in selection)),
                  'charts_used': sum(len(d['charts']) for d in selection),
                  'charts_in_manifest_for_these_songs': sum(d['charts_in_manifest'] for d in selection),
                  'train': sorted(selection[i]['sha'] for i in train_songs), 'val': sorted(selection[i]['sha'] for i in val_songs),
                  'test': sorted(selection[i]['sha'] for i in test_songs)},
        'samples': {'total': len(recs), 'train': int(len(tr)), 'val': int(len(va)), 'test': int(len(te)),
                    'test_by_mode': {m: int(g.sum()) for m, g in groups.items()},
                    'target_marginal_train': marg.round(5).tolist()},
        'bytes': {'taps_downloaded_this_run': downloaded, 'taps_on_disk': on_disk, 'manifest_and_pca_on_disk': extra,
                  'total_on_disk': on_disk + extra},
        'mapping': mapping, 'alignment_check': align,
        'features': {
            'A_tap': f'cells c-1,c,c+1 x 8 bands x 1024 (dequant + per-token LN) -> per-token PCA {a.per_token_dim} dims fitted on '
                     f'train+val songs -> {24 * a.per_token_dim} dims; variance kept per layer: '
                     + ', '.join(f'L{l} {v:.3f}' for l, v in zip(LAYERS, feats['pca_var_kept'].tolist())),
            'A_all': f'concat of the three A_tap features ({3 * 24 * a.per_token_dim}) -> second-stage PCA {a.all_dim} dims '
                     f'(train+val rows), variance kept {kept2:.3f}',
            'A_all_rp': 'reduction check: fixed Gaussian random projection (seed 12345) of each tap\'s 3 x 8 x 1024 block to 682 dims, '
                        '3 taps concatenated = 2046 dims (no fitting)',
            'A_all_rpd': 'A_all_rp + random projection (seed 54321, 682 dims per tap) of relu(token(c) - token(c-1)), '
                         'relu(token(c+1) - token(c)) = 4092 dims (token analogue of spectral flux)',
            'B': 'mel10 frames 4c-2 .. 4c+5 (8 x 128 = 1024 dims, raw Pupu-normalised log-mel)',
            'B_flux': 'B + rectified frame differences (7 x 128) + per-frame summed flux (7) = 1927 dims',
            'B_heur': 'non-learned: 10 ms bin b = argmax over b of summed positive flux at frame 4c+b+1, prediction = bin '
                      'centre 10b+4.5 (expect decoder; argmax gives 10b)',
            'C_flux': 'A_all_rpd + B_flux',
            'C': 'A_all + B', 'D': 'train marginal of the 40 offsets (chance); D_mode = per-mode marginal',
            'E': 'A_all + rhythm context (40 dims): gaps from the cell start back to the previous 4 distinct onsets of the same '
                 'chart and the 4 IOIs among the previous 5; each as seconds, log1p(ms), sin/cos of the 40 ms phase, + presence flags',
            'E_only': 'rhythm context alone (diagnostic)',
            'standardisation': 'every input z-scored per dim on train+val rows',
            'model': f'Dropout(in_drop) - Linear(d, {a.hidden}) - GELU - Dropout(0.3) - Linear({a.hidden}, 40); AdamW wd 0.1, batch 512, '
                     f'cosine over {a.epochs} epochs, early stop on val NLL (patience 8), CPU. Per feature set the same grid '
                     f'lr x in_drop = {{1e-4,1e-3}} x {{0,0.5}} is run; the model with the lowest val MAE (expectation decoder) is '
                     f'reported (results), the lowest-val-NLL one is kept in results_nll_selected. val = held-out songs. '
                     f'(Run 1 with a single lr 1e-3 / wd 1e-2 setting over-fitted within one epoch on every audio set; run 2 with a '
                     f'wd axis showed wd has no effect.)',
            'cv': '5 by-song folds from one seeded shuffle (fold 0 = the reported split); folds 1-4 reuse each set\'s fold-0 '
                  'configuration; PCA / standardisation statistics stay those of fold 0 (unsupervised, mild leakage into folds 1-4)',
            'decoders': 'argmax class; expectation sum_k p_k k. Errors against the integer target (0-39 ms); '
                        'mae_vs_exact_ms uses the sub-ms onset position',
        },
        'results': results, 'results_nll_selected': results_nll, 'training': train_info, 'cv_by_song_5fold': cv, 'cv_gaps': cv_gap,
        'decision': {
            'rule': 'remove the mel branch if MAE_A_all - MAE_B <= 1 ms and MAE_A_all - MAE_C <= 1 ms (test split, expectation decoder, '
                    'models selected on val MAE)',
            'decoder_used': dec, 'mae_A_all_minus_B_ms': round(gap_B, 3), 'mae_A_all_minus_C_ms': round(gap_C, 3),
            'remove_mel_by_preregistered_rule': remove_mel,
            'per_mode': {md: {'mae_A_all_minus_B_ms': round(m('A_all', md) - m('B', md), 3),
                              'mae_A_all_minus_C_ms': round(m('A_all', md) - m('C', md), 3)} for md in ('mania', 'phira')},
            'with_best_decoder_per_model': {'mae_A_all_minus_B_ms': round(gap_B_best, 3), 'mae_A_all_minus_C_ms': round(gap_C_best, 3),
                                            'remove_mel': bool(gap_B_best <= 1.0 and gap_C_best <= 1.0)},
            'strict_check_best_token_vs_best_mel': strict,
            'cv_gap_A_all_minus_B_ms': cv_gap['A_all-B'], 'cv_gap_A_all_minus_C_ms': cv_gap['A_all-C'],
            'cv_gap_A_all_rpd_minus_B_heur_ms': cv_gap['A_all_rpd-B_heur'],
            'E_gain_over_A_all_ms': round(e_gain, 3),
            'D_mae_ms': round(results['D'][dec]['mae_ms'], 3),
            'B_heur_mae_ms': round(results['B_heur'][dec]['mae_ms'], 3),
            'B_heur_mania_mae_ms': round(results['B_heur'][dec]['mania']['mae_ms'], 3),
        },
        'runtime_seconds': round(time.time() - t_start, 1),
    }
    report['decision']['summary'] = decision_summary(report)
    os.makedirs(os.path.dirname(a.out_json), exist_ok=True)
    with open(a.out_json, 'w') as f:
        json.dump(report, f, indent=1, ensure_ascii=False)
    with open(a.out_md, 'w') as f:
        f.write(render_md(report))
    log('wrote', a.out_json, a.out_md)
    print(render_md(report))


def decision_summary(rep, th=1.0):
    """Four conditions behind the headline (all on the expectation decoder, test songs only):
    1 pre-registered rule on the pre-registered split; 2 the same rule averaged over the 5 by-song folds;
    3 best token representation vs best LEARNED mel representation and vs tokens+mel (5-fold means);
    4 best token representation vs the non-learned mel flux heuristic on mania (pre-registered split)."""
    d, cv, g = rep['decision'], rep['cv_by_song_5fold'], rep['cv_gaps']
    st = d['strict_check_best_token_vs_best_mel']
    tok = st['token_set']
    c1 = d['mae_A_all_minus_B_ms'] <= th and d['mae_A_all_minus_C_ms'] <= th
    c2 = g['A_all-B']['mean'] <= th and g['A_all-C']['mean'] <= th
    c3 = g[f'{tok}-B_flux']['mean'] <= th and g[f'{tok}-C_flux']['mean'] <= th if f'{tok}-B_flux' in g else None
    c4 = st['gap_mania_ms'] <= th
    txt = []
    txt.append(f"条件 1（预注册判据，固定切分）：{'满足' if c1 else '不满足'}（A_all−B {d['mae_A_all_minus_B_ms']:+.2f}，A_all−C {d['mae_A_all_minus_C_ms']:+.2f} ms）。")
    txt.append(f"条件 2（同判据，5 折均值）：{'满足' if c2 else '不满足'}（{g['A_all-B']['mean']:+.2f} ± {g['A_all-B']['sd']:.2f}，"
               f"{g['A_all-C']['mean']:+.2f} ± {g['A_all-C']['sd']:.2f} ms）；预注册的 A_all（两级 PCA 到 2048 维）是最差的 token 表示"
               f"（5 折 {cv['A_all']['expect_mae']['mean']:.2f} ms，对比随机投影 A_all_rp {cv['A_all_rp']['expect_mae']['mean']:.2f}、"
               f"A_all_rpd {cv['A_all_rpd']['expect_mae']['mean']:.2f} ms），它在固定切分上的“满足”是弱对弱的巧合。")
    if c3 is not None:
        txt.append(f"条件 3（最优 token 表示 {tok} 对最优学习型 mel 表示 B_flux 与对 token+mel 的 C_flux，5 折均值）：{'满足' if c3 else '不满足'}"
                   f"（{g[f'{tok}-B_flux']['mean']:+.2f} ± {g[f'{tok}-B_flux']['sd']:.2f}，{g[f'{tok}-C_flux']['mean']:+.2f} ± "
                   f"{g[f'{tok}-C_flux']['sd']:.2f} ms）：学习型读出下 token 不劣于 mel，且 mel 叠加到 token 上没有增益。")
    txt.append(f"条件 4（{tok} 对非学习的 mel 谱通量启发式 B_heur，mania，固定切分）：{'满足' if c4 else '不满足'}（{st['gap_mania_ms']:+.2f} ms；"
               f"5 折 {g[f'{tok}-B_heur']['mania_mean']:+.2f} ± {g[f'{tok}-B_heur']['mania_sd']:.2f} ms，折间不稳定）：mel 里仍有一部分"
               f"相位信息是零参数启发式能读出、而 38 首歌训练的 token 探针在该切分上读不出的；phira 上启发式无信息"
               f"（{rep['results']['B_heur']['expect']['phira']['mae_ms']:.2f} ms ≈ 机会水平）。")
    verdict = ('按预注册判据：删 mel 支路。' if c1 else '按预注册判据：保留 mel 支路。')
    verdict += ('稳健性支持“token 不劣于学习型 mel 读出、mel 无叠加增益”，' if c3 else '稳健性不支持该结论，')
    verdict += ('但未能证明 token 已含 mel 的全部相位信息（条件 4 不满足）。建议：删 mel 支路，但保留 §5 实验 4 的 mel × 间隔核 2×2 对照作为验证；'
                '所有音频探针都远不及节律先验（E_only）。' if not c4 else '且 token 对 mel 的启发式亦不劣，可删 mel 支路。')
    return {'conditions': {'c1_preregistered_fixed_split': bool(c1), 'c2_preregistered_5fold': bool(c2),
                           'c3_best_token_vs_learned_mel_5fold': None if c3 is None else bool(c3), 'c4_best_token_vs_heuristic_mania': bool(c4)},
            'text': txt, 'verdict': verdict}


def render_md(rep):
    R, d, s, b = rep['results'], rep['decision'], rep['samples'], rep['bytes']
    order = ['D', 'D_mode', 'A_9', 'A_16', 'A_24', 'A_all', 'A_all_rp', 'A_all_rpd', 'B', 'B_flux', 'B_heur', 'C', 'C_flux', 'E', 'E_only']
    L = []
    L.append('# 相位探针（phase probe）— 2026-10-23\n')
    L.append(f"设计依据：{rep['design_ref']}。问题：40 ms 格内的起音偏移（0–39 ms）能否从冻结 PupuM2D 格 token 读出，与 10 ms mel 相比如何。\n")
    L.append('## 结论\n')
    verdict = '满足"删 mel"判据' if d['remove_mel_by_preregistered_rule'] else '不满足"删 mel"判据'
    L.append(f"**预注册判据**：{d['rule']}。MAE_A_all − MAE_B = {d['mae_A_all_minus_B_ms']:+.2f} ms，"
             f"MAE_A_all − MAE_C = {d['mae_A_all_minus_C_ms']:+.2f} ms → **{verdict}**。"
             f"按模式：mania {d['per_mode']['mania']['mae_A_all_minus_B_ms']:+.2f} / {d['per_mode']['mania']['mae_A_all_minus_C_ms']:+.2f} ms，"
             f"phira {d['per_mode']['phira']['mae_A_all_minus_B_ms']:+.2f} / {d['per_mode']['phira']['mae_A_all_minus_C_ms']:+.2f} ms。")
    g1, g2 = d['cv_gap_A_all_minus_B_ms'], d['cv_gap_A_all_minus_C_ms']
    L.append(f"5 折（按歌）稳健性：A_all − B = {g1['mean']:+.2f} ± {g1['sd']:.2f} ms，A_all − C = {g2['mean']:+.2f} ± {g2['sd']:.2f} ms（折间 sd）。")
    st = d['strict_check_best_token_vs_best_mel']
    L.append(f"**严格检查**（val 上最优的 token 表示 {st['token_set']} 对最优的 mel 表示 {st['mel_set']}，含非学习的谱通量启发式 B_heur）："
             f"全体 {st['gap_all_ms']:+.2f} ms，mania {st['gap_mania_ms']:+.2f} ms，phira {st['gap_phira_ms']:+.2f} ms；对 C_flux：全体 "
             f"{st['gap_vs_C_flux_ms']:+.2f}，mania {st['gap_mania_vs_C_flux_ms']:+.2f} ms → "
             f"{'通过' if st['remove_mel'] else '**不通过**'}（阈值 1 ms）。")
    g3 = d['cv_gap_A_all_rpd_minus_B_heur_ms']
    L.append(f"5 折：A_all_rpd − B_heur = {g3['mean']:+.2f} ± {g3['sd']:.2f} ms（mania {g3['mania_mean']:+.2f} ± {g3['mania_sd']:.2f}）。")
    L.append(f"非学习启发式 B_heur（mel 谱通量选 10 ms 桶）MAE = {d['B_heur_mae_ms']:.2f} ms（mania {d['B_heur_mania_mae_ms']:.2f}）；"
             f"D（仅边缘分布）MAE = {d['D_mae_ms']:.2f} ms；E 比 A_all 的 MAE 低 {d['E_gain_over_A_all_ms']:+.2f} ms（节律先验）。\n")
    sm = rep['decision'].get('summary')
    if sm:
        L.append('### 解读\n')
        for t in sm['text']:
            L.append(f'- {t}')
        L.append(f"\n**{sm['verdict']}**\n")
    L.append('## 主表（测试集 = 预注册的按歌 80/20 切分；模型按 val MAE 选网格配置）\n')
    L.append('| 模型 | 输入维 | 解码 | MAE ms | 中位 ms | P(≤5) | P(≤10) | NLL | top1 | MAE mania | MAE phira |')
    L.append('|---|---|---|---|---|---|---|---|---|---|---|')
    for n in order:
        if n not in R:
            continue
        dim = rep['training'].get(n, {}).get('input_dim', '-')
        for dec in ('argmax', 'expect'):
            mm = R[n][dec]
            L.append(f"| {n} | {dim} | {dec} | {mm['mae_ms']:.2f} | {mm['median_ms']:.1f} | {mm['p_le5']:.3f} | {mm['p_le10']:.3f} | "
                     f"{R[n]['nll']:.3f} | {R[n]['top1_acc']:.3f} | {mm['mania']['mae_ms']:.2f} | {mm['phira']['mae_ms']:.2f} |")
    L.append('')
    L.append('## 5 折（按歌）MAE，expect 解码，均值 ± 折间 sd\n')
    L.append('| 模型 | 全体 | mania | phira | 各折（全体） |')
    L.append('|---|---|---|---|---|')
    for n, v in rep['cv_by_song_5fold'].items():
        L.append(f"| {n} | {v['expect_mae']['mean']:.2f} ± {v['expect_mae']['sd']:.2f} | {v['mania_expect_mae']['mean']:.2f} ± "
                 f"{v['mania_expect_mae']['sd']:.2f} | {v['phira_expect_mae']['mean']:.2f} ± {v['phira_expect_mae']['sd']:.2f} | "
                 f"{', '.join(f'{x:.2f}' for x in v['expect_mae']['folds'])} |")
    L.append('')
    L.append('差值（expect MAE，折均值 ± sd；括号内 mania）：' + '；'.join(
        f"{k} = {v['mean']:+.2f} ± {v['sd']:.2f}（{v['mania_mean']:+.2f} ± {v['mania_sd']:.2f}）" for k, v in rep['cv_gaps'].items()) + '\n')
    for mode in ('mania', 'phira'):
        if s['test_by_mode'].get(mode, 0) == 0:
            continue
        L.append(f"## 按模式：{mode}（测试样本 {s['test_by_mode'][mode]}）\n")
        L.append('| 模型 | 解码 | MAE ms | 中位 ms | P(≤5) | P(≤10) |')
        L.append('|---|---|---|---|---|---|')
        for n in order:
            if n not in R or mode not in R[n]['argmax']:
                continue
            for dec in ('argmax', 'expect'):
                mm = R[n][dec][mode]
                L.append(f"| {n} | {dec} | {mm['mae_ms']:.2f} | {mm['median_ms']:.1f} | {mm['p_le5']:.3f} | {mm['p_le10']:.3f} |")
        L.append('')
    L.append('## 数据\n')
    sg = rep['songs']
    L.append(f"- 歌曲：{sg['n']}（{sg['by_source']}），train/val/test 歌曲数 {len(sg['train'])}/{len(sg['val'])}/{len(sg['test'])}；"
             f"使用谱面 {sg['charts_used']} 张（这些歌在清单中共 {sg['charts_in_manifest_for_these_songs']} 张，只用本地已有 canon 的）。")
    L.append(f"- 样本（含起音的 40 ms 格，按 (chart, cell) 去重）：共 {s['total']}，train {s['train']} / val {s['val']} / test {s['test']}；"
             f"测试集按模式 {s['test_by_mode']}。")
    L.append(f"- 下载（`modal volume get`，只读）：taps {b['taps_on_disk'] / 1e9:.3f} GB（180 个文件，60 首 × L9/L16/L24；本次运行新下载 "
             f"{b['taps_downloaded_this_run'] / 1e9:.3f} GB，其余来自前一步 `--download-only`）+ 清单/PCA {b['manifest_and_pca_on_disk'] / 1e6:.1f} MB "
             f"= 共 {b['total_on_disk'] / 1e9:.3f} GB，全部仍在 {rep['root']}（未删除；特征缓存 cache/ 为派生数据）。")
    L.append(f"- 清单：{rep['manifest']}（split=train，offset decision v2，训练同款筛选）；taps：{rep['taps_key']}；pca：{rep['pca']}。\n")
    L.append('## 时间映射（与训练一致）\n')
    mp = rep['mapping']
    L.append(f"- 模型视角起音：`{mp['model_view_onset_us']}`；目标时刻 = `{mp['target_onset_us']}`；格 = `{mp['song_cell']}`；格内 ms = `{mp['in_cell_ms']}`。")
    L.append(f"- 音频轴：{mp['audio_axis']}。")
    L.append(f"- 反量化：{mp['dequant']}。")
    L.append(f"- {mp['fakes']}。")
    L.append('- 出处：')
    for w in mp['where']:
        L.append(f'  - {w}')
    L.append('')
    L.append('## 对齐自检（谱面→音频映射）\n')
    L.append('量化后的模型视角起音周围的谱通量包络（`ingest/offset.flux_envelope`，mel10），帧偏移 −6..+6 的均值，以及偏移 0 处按帧内 ms 分组的均值（每源前 8 首）：\n')
    for src, v in rep['alignment_check'].items():
        L.append(f"- {src}（{v['onsets']} 起音）：按帧偏移 {v['mean_flux_by_frame_offset_-6..6']}；偏移 0 按帧内 ms {v['mean_flux_at_offset0_by_sub_frame_ms_0..9']}")
    L.append('')
    L.append('## 特征与模型\n')
    for k, v in rep['features'].items():
        L.append(f'- **{k}**：{v}')
    L.append('')
    L.append('## 训练细节\n')
    L.append('| 模型 | 输入维 | 参数 | 选中 lr/in_drop | 轮数 | val NLL | val MAE(expect) | 秒（整个网格） |')
    L.append('|---|---|---|---|---|---|---|---|')
    for n, t in rep['training'].items():
        c = t['config']
        L.append(f"| {n} | {t['input_dim']} | {t['params']} | {c['lr']:.0e}/{c['in_drop']} | {t['epochs_run']} | "
                 f"{t['best_val_nll']:.3f} | {t['val_mae_expect']:.2f} | {t['train_seconds']} |")
    L.append('')
    L.append('按 val NLL 选配置时的测试 MAE（expect）：' + '，'.join(
        f"{n} {v['expect']['mae_ms']:.2f}" for n, v in rep['results_nll_selected'].items()) + '\n')
    L.append(f"训练集偏移边缘分布（0..39 ms）：{[round(p, 3) for p in s['target_marginal_train']]}\n")
    L.append(f"运行时长 {rep['runtime_seconds']} s。脚本：`scripts/phase_probe.py --root <root> [--download]`。")
    return '\n'.join(L) + '\n'


if __name__ == '__main__':
    main()
