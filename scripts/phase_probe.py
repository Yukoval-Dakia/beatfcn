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
    feats = {f'tap{l}': np.zeros((N, 3 * BANDS * per_token_dim), np.float16) for l in LAYERS}
    feats.update({f'rp{l}': np.zeros((N, rp_dim), np.float16) for l in LAYERS})
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

def standardise(x, train_idx):
    mu = x[train_idx].mean(0, dtype=np.float64).astype(np.float32)
    sd = x[train_idx].std(0, dtype=np.float64).astype(np.float32) + 1e-5
    return ((x - mu) / sd).astype(np.float32)


def train_mlp(x, y, tr, va, seed, hidden=256, epochs=80, patience=8, lr=1e-3, wd=1e-2, batch=512, dropout=0.1):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    xt, yt = torch.from_numpy(x), torch.from_numpy(y.astype(np.int64))
    model = nn.Sequential(nn.Linear(x.shape[1], hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, N_CLASSES))
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
                     'params': sum(p.numel() for p in model.parameters()), 'input_dim': int(x.shape[1])}


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
    a = ap.parse_args()
    t_start = time.time()

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
    song_of = np.array([r[0] for r in recs])
    y = np.array([r[4] for r in recs], np.int64)
    y_exact = np.array([r[5] for r in recs], np.float64)
    mode_of = np.array([r[2] for r in recs])
    # split by song 80/20 (stratified by source), val = 10% of the train songs for early stopping
    rng = np.random.default_rng(a.seed)
    test_songs, val_songs, train_songs = set(), set(), set()
    for src in sorted({d['source'] for d in selection}):
        ids = [i for i, d in enumerate(selection) if d['source'] == src]
        rng.shuffle(ids)
        n_test = int(round(0.2 * len(ids)))
        test_songs |= set(ids[:n_test])
        rest = ids[n_test:]
        n_val = max(1, int(round(0.1 * len(rest))))
        val_songs |= set(rest[:n_val])
        train_songs |= set(rest[n_val:])
    tr = np.flatnonzero(np.isin(song_of, list(train_songs)))
    va = np.flatnonzero(np.isin(song_of, list(val_songs)))
    te = np.flatnonzero(np.isin(song_of, list(test_songs)))
    fit_songs = train_songs | val_songs  # PCA / standardisation statistics: everything but test
    log(f'split: train {len(tr)} / val {len(va)} / test {len(te)} samples; songs {len(train_songs)}/{len(val_songs)}/{len(test_songs)}')

    key = hashlib.sha256(json.dumps([d['sha'] for d in selection] + [a.per_token_dim]).encode()).hexdigest()[:12]
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
    X['B'] = standardise(feats['mel'].astype(np.float32), fit_idx)
    X['C'] = np.concatenate([X['A_all'], X['B']], 1)
    R = standardise(rhythm_features(recs), fit_idx)
    X['E'] = np.concatenate([X['A_all'], R], 1)
    X['E_only'] = R  # rhythm context alone (diagnostic)

    results, train_info = {}, {}
    groups = {'mania': mode_of[te] == 'mania', 'phira': mode_of[te] == 'phira'}
    # D: marginal distribution of the train split (chance baseline); D_mode: per-mode marginal
    marg = np.bincount(y[fit_idx], minlength=N_CLASSES) / len(fit_idx)
    results['D'] = metrics(np.tile(marg, (len(te), 1)), y[te], y_exact[te], groups)
    pm = np.zeros((len(te), N_CLASSES))
    for m in ('mania', 'phira'):
        sel_m = mode_of[fit_idx] == m
        if sel_m.any():
            pm[groups[m]] = np.bincount(y[fit_idx][sel_m], minlength=N_CLASSES) / sel_m.sum()
    results['D_mode'] = metrics(pm, y[te], y_exact[te], groups)
    for name in ['A_9', 'A_16', 'A_24', 'A_all', 'A_all_rp', 'B', 'C', 'E', 'E_only']:
        t0 = time.time()
        predict, info = train_mlp(X[name], y, tr, va, a.seed, hidden=a.hidden, epochs=a.epochs)
        results[name] = metrics(predict(te), y[te], y_exact[te], groups)
        info['train_seconds'] = round(time.time() - t0, 1)
        train_info[name] = info
        r = results[name]
        log(f"{name:7s} dim {info['input_dim']:5d} ep {info['epochs_run']:2d} val_nll {info['best_val_nll']:.3f} | "
            f"argmax MAE {r['argmax']['mae_ms']:.2f} P5 {r['argmax']['p_le5']:.3f} | expect MAE {r['expect']['mae_ms']:.2f} "
            f"P5 {r['expect']['p_le5']:.3f} ({info['train_seconds']}s)")

    # decision
    dec = 'expect'
    best_dec = {n: min(results[n]['argmax']['mae_ms'], results[n]['expect']['mae_ms']) for n in results}
    gap_B = results['A_all'][dec]['mae_ms'] - results['B'][dec]['mae_ms']
    gap_C = results['A_all'][dec]['mae_ms'] - results['C'][dec]['mae_ms']
    gap_B_best, gap_C_best = best_dec['A_all'] - best_dec['B'], best_dec['A_all'] - best_dec['C']
    remove_mel = bool(gap_B <= 1.0 and gap_C <= 1.0)
    remove_mel_best = bool(gap_B_best <= 1.0 and gap_C_best <= 1.0)
    e_gain = results['A_all'][dec]['mae_ms'] - results['E'][dec]['mae_ms']
    gap_B_rp = results['A_all_rp'][dec]['mae_ms'] - results['B'][dec]['mae_ms']
    gap_C_rp = results['A_all_rp'][dec]['mae_ms'] - results['C'][dec]['mae_ms']

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
        'mapping': mapping,
        'features': {
            'A_tap': f'cells c-1,c,c+1 x 8 bands x 1024 (dequant + per-token LN) -> per-token PCA {a.per_token_dim} dims fitted on '
                     f'train+val songs -> {24 * a.per_token_dim} dims; variance kept per layer: '
                     + ', '.join(f'L{l} {v:.3f}' for l, v in zip(LAYERS, feats['pca_var_kept'].tolist())),
            'A_all': f'concat of the three A_tap features ({3 * 24 * a.per_token_dim}) -> second-stage PCA {a.all_dim} dims '
                     f'(train+val rows), variance kept {kept2:.3f}',
            'A_all_rp': 'reduction check: fixed Gaussian random projection (seed 12345) of each tap\'s 3 x 8 x 1024 block to 682 dims, '
                        '3 taps concatenated = 2046 dims (no fitting)',
            'B': 'mel10 frames 4c-2 .. 4c+5 (8 x 128 = 1024 dims, raw Pupu-normalised log-mel)',
            'C': 'A_all + B', 'D': 'train marginal of the 40 offsets (chance); D_mode = per-mode marginal',
            'E': 'A_all + rhythm context (40 dims): gaps from the cell start back to the previous 4 distinct onsets of the same '
                 'chart and the 4 IOIs among the previous 5; each as seconds, log1p(ms), sin/cos of the 40 ms phase, + presence flags',
            'E_only': 'rhythm context alone (diagnostic)',
            'standardisation': 'every input z-scored per dim on train+val rows',
            'model': f'Linear({"d"}, {a.hidden}) - GELU - Dropout(0.1) - Linear({a.hidden}, 40); AdamW lr 1e-3 wd 1e-2, batch 512, '
                     f'cosine over {a.epochs} epochs, early stop on val NLL (patience 8), CPU',
            'decoders': 'argmax class; expectation sum_k p_k k. Errors against the integer target (0-39 ms); '
                        'mae_vs_exact_ms uses the sub-ms onset position',
        },
        'results': results, 'training': train_info,
        'decision': {
            'rule': 'remove the mel branch if MAE_A_all - MAE_B <= 1 ms and MAE_A_all - MAE_C <= 1 ms',
            'decoder_used': dec, 'mae_A_all_minus_B_ms': round(gap_B, 3), 'mae_A_all_minus_C_ms': round(gap_C, 3),
            'remove_mel': remove_mel,
            'with_best_decoder_per_model': {'mae_A_all_minus_B_ms': round(gap_B_best, 3), 'mae_A_all_minus_C_ms': round(gap_C_best, 3),
                                            'remove_mel': remove_mel_best},
            'with_A_all_rp': {'mae_A_all_rp_minus_B_ms': round(gap_B_rp, 3), 'mae_A_all_rp_minus_C_ms': round(gap_C_rp, 3),
                              'remove_mel': bool(gap_B_rp <= 1.0 and gap_C_rp <= 1.0)},
            'E_gain_over_A_all_ms': round(e_gain, 3),
            'D_mae_ms': round(results['D'][dec]['mae_ms'], 3),
        },
        'runtime_seconds': round(time.time() - t_start, 1),
    }
    os.makedirs(os.path.dirname(a.out_json), exist_ok=True)
    with open(a.out_json, 'w') as f:
        json.dump(report, f, indent=1, ensure_ascii=False)
    with open(a.out_md, 'w') as f:
        f.write(render_md(report))
    log('wrote', a.out_json, a.out_md)
    print(render_md(report))


def render_md(rep):
    R, d, s, b = rep['results'], rep['decision'], rep['samples'], rep['bytes']
    order = ['D', 'D_mode', 'A_9', 'A_16', 'A_24', 'A_all', 'A_all_rp', 'B', 'C', 'E', 'E_only']
    L = []
    L.append('# 相位探针（phase probe）— 2026-10-23\n')
    L.append(f"设计依据：{rep['design_ref']}。问题：40 ms 格内的起音偏移（0–39 ms）能否从冻结 PupuM2D 格 token 读出，与 10 ms mel 相比如何。\n")
    L.append('## 结论\n')
    verdict = '**删除 mel 支路**' if d['remove_mel'] else '**保留 mel 支路**'
    L.append(f"判据：{d['rule']}（解码器：{d['decoder_used']}）。MAE_A_all − MAE_B = {d['mae_A_all_minus_B_ms']:+.2f} ms，"
             f"MAE_A_all − MAE_C = {d['mae_A_all_minus_C_ms']:+.2f} ms → {verdict}。")
    bd = d['with_best_decoder_per_model']
    L.append(f"（每个模型取各自更优解码器时：{bd['mae_A_all_minus_B_ms']:+.2f} / {bd['mae_A_all_minus_C_ms']:+.2f} ms → "
             f"{'删' if bd['remove_mel'] else '保留'}。）")
    rp = d['with_A_all_rp']
    L.append(f"（随机投影版 A_all_rp 替代 A_all 时：{rp['mae_A_all_rp_minus_B_ms']:+.2f} / {rp['mae_A_all_rp_minus_C_ms']:+.2f} ms → "
             f"{'删' if rp['remove_mel'] else '保留'}。）")
    L.append(f"节律网格先验的价值：E 比 A_all 的 MAE 低 {d['E_gain_over_A_all_ms']:+.2f} ms；D（仅边缘分布）MAE = {d['D_mae_ms']:.2f} ms。\n")
    L.append('## 主表（测试集，按歌 80/20 切分）\n')
    L.append('| 模型 | 输入维 | 解码 | MAE ms | 中位 ms | P(≤5) | P(≤10) | NLL | top1 |')
    L.append('|---|---|---|---|---|---|---|---|---|')
    for n in order:
        if n not in R:
            continue
        dim = rep['training'].get(n, {}).get('input_dim', '-')
        for dec in ('argmax', 'expect'):
            m = R[n][dec]
            L.append(f"| {n} | {dim} | {dec} | {m['mae_ms']:.2f} | {m['median_ms']:.1f} | {m['p_le5']:.3f} | {m['p_le10']:.3f} | "
                     f"{R[n]['nll']:.3f} | {R[n]['top1_acc']:.3f} |")
    L.append('')
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
                m = R[n][dec][mode]
                L.append(f"| {n} | {dec} | {m['mae_ms']:.2f} | {m['median_ms']:.1f} | {m['p_le5']:.3f} | {m['p_le10']:.3f} |")
        L.append('')
    L.append('## 数据\n')
    sg = rep['songs']
    L.append(f"- 歌曲：{sg['n']}（{sg['by_source']}），train/val/test 歌曲数 {len(sg['train'])}/{len(sg['val'])}/{len(sg['test'])}；"
             f"使用谱面 {sg['charts_used']} 张（这些歌在清单中共 {sg['charts_in_manifest_for_these_songs']} 张，只用本地已有 canon 的）。")
    L.append(f"- 样本（含起音的 40 ms 格，按 (chart, cell) 去重）：共 {s['total']}，train {s['train']} / val {s['val']} / test {s['test']}；"
             f"测试集按模式 {s['test_by_mode']}。")
    L.append(f"- 下载：本次下载 taps {b['taps_downloaded_this_run'] / 1e9:.3f} GB；磁盘上 taps {b['taps_on_disk'] / 1e9:.3f} GB + "
             f"清单/PCA {b['manifest_and_pca_on_disk'] / 1e6:.1f} MB = 共 {b['total_on_disk'] / 1e9:.3f} GB（未删除）。")
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
    L.append('## 特征与模型\n')
    for k, v in rep['features'].items():
        L.append(f'- **{k}**：{v}')
    L.append('')
    L.append('## 训练细节\n')
    L.append('| 模型 | 输入维 | 参数 | 轮数 | 最优 val NLL | 秒 |')
    L.append('|---|---|---|---|---|---|')
    for n, t in rep['training'].items():
        L.append(f"| {n} | {t['input_dim']} | {t['params']} | {t['epochs_run']} | {t['best_val_nll']:.3f} | {t['train_seconds']} |")
    L.append('')
    L.append(f"训练集偏移边缘分布（0..39 ms）：{[round(p, 3) for p in s['target_marginal_train']]}\n")
    L.append(f"运行时长 {rep['runtime_seconds']} s。脚本：`scripts/phase_probe.py --root <root> [--download]`。")
    return '\n'.join(L) + '\n'


if __name__ == '__main__':
    main()
