"""Modal app: frozen PupuM2D weights, encoder throughput, tap extraction.

Run from the repo root, e.g.:
  .venv/bin/modal run beatfcn/cloud/encode_app.py::fetch_teacher
  .venv/bin/modal run beatfcn/cloud/encode_app.py::bench_encode
"""
from __future__ import annotations
import json
import os
import time
from pathlib import Path
import modal

_HERE = Path(__file__).resolve()
REPO = _HERE.parents[2] if len(_HERE.parents) > 2 else _HERE.parent  # container: /root/encode_app.py
TEACHER_SHA256 = '6a97605541ba9d3680f6758e2ae19019736f03e5db14e8bbfbd71746b19304f4'
TEACHER_BYTES = 1_612_597_160
WEIGHTS = '/weights'
DATA = '/data'

app = modal.App('beatfcn-encode')
weights = modal.Volume.from_name('beatfcn-weights', create_if_missing=True)
data = modal.Volume.from_name('beatfcn-data', create_if_missing=True)

cpu_image = (modal.Image.debian_slim(python_version='3.12').apt_install('curl')
             .add_local_dir(REPO / 'experiments/shared_kda_s0a/teacher_fetch', '/opt/teacher_fetch', copy=True))
gpu_image = (modal.Image.debian_slim(python_version='3.12').apt_install('ffmpeg')
             .pip_install('torch==2.7.1', index_url='https://download.pytorch.org/whl/cu126')
             .pip_install('numpy==2.2.6', 'safetensors==0.5.3', 'pyyaml==6.0.3')
             .add_local_python_source('beatfcn'))


def _sha256(path, chunk=16 << 20):
    import hashlib
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


@app.function(image=cpu_image, volumes={WEIGHTS: weights}, cpu=1.0, memory=2048, timeout=1800)
def fetch_teacher():
    """Pinned-revision range download inside Modal; exact SHA256 or stop."""
    import shutil
    import subprocess
    dest = Path(WEIGHTS) / 'teacher_encoder.safetensors'
    if dest.exists() and dest.stat().st_size == TEACHER_BYTES and _sha256(dest) == TEACHER_SHA256:
        return {'status': 'present', 'sha256': TEACHER_SHA256}
    work = Path('/tmp/tf')
    shutil.copytree('/opt/teacher_fetch', work, dirs_exist_ok=True)
    (work / 'weights').mkdir(exist_ok=True)
    t = time.time()
    subprocess.run(['python', str(work / 'src/download_teacher.py')], check=True)
    got = work / 'weights/teacher_encoder.safetensors'
    digest = _sha256(got)
    if digest != TEACHER_SHA256 or got.stat().st_size != TEACHER_BYTES:
        raise RuntimeError(f'teacher mismatch {digest} {got.stat().st_size}; refusing')
    shutil.copy2(got, dest)
    shutil.copy2(work / 'reports/teacher_download_manifest.json', Path(WEIGHTS) / 'teacher_download_manifest.json')
    weights.commit()
    return {'status': 'downloaded', 'sha256': digest, 'seconds': round(time.time() - t, 1)}


def load_encoder(device='cuda'):
    import torch
    from beatfcn.audio.pupu import load_teacher
    from beatfcn.audio.memory import TapEncoder
    path = Path(WEIGHTS) / 'teacher_encoder.safetensors'
    if _sha256(path) != TEACHER_SHA256:
        raise RuntimeError('teacher checkpoint hash mismatch')
    return TapEncoder(load_teacher(path, dtype=torch.bfloat16, device=device))


@app.function(image=gpu_image, gpu='L40S', volumes={WEIGHTS: weights}, cpu=4.0, memory=16384, timeout=900)
def bench_encode(batches=(4, 8, 16, 32), repeats: int = 6):
    """Encoder throughput (windows/s) and peak memory on synthetic mel windows."""
    import torch
    t0 = time.time()
    enc = load_encoder()
    load_s = time.time() - t0
    rows = []
    for b in batches:
        x = torch.randn(b, 1, 1024, 128)
        torch.cuda.reset_peak_memory_stats()
        enc(x, [8, 16, 24])
        torch.cuda.synchronize()
        t = time.time()
        for _ in range(repeats):
            enc(x, [8, 16, 24])
        torch.cuda.synchronize()
        dt = (time.time() - t) / repeats
        rows.append({'batch': b, 'seconds_per_batch': round(dt, 4), 'windows_per_s': round(b / dt, 1),
                     'peak_allocated_gb': round(torch.cuda.max_memory_allocated() / 1e9, 2)})
    report = {'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__, 'load_seconds': round(load_s, 1),
              'layers': [8, 16, 24], 'dtype': 'bfloat16', 'rows': rows}
    print(json.dumps(report, indent=1))
    return report


PRICE_PER_HOUR = {'L4': 0.7992, 'A10': 1.1016, 'L40S': 1.9512, 'A100-40GB': 2.0988, 'A100-80GB': 2.4984, 'H100': 3.9492}


@app.local_entrypoint()
def compare_gpus(gpus: str = 'L4,A10,L40S,A100-40GB,H100'):
    """Same benchmark on several GPU types in parallel; rank by windows per dollar (GPU list price only)."""
    names = [g.strip() for g in gpus.split(',') if g.strip()]
    calls = {g: bench_encode.with_options(gpu=g).spawn() for g in names}
    table = []
    for g, call in calls.items():
        try:
            r = call.get()
        except Exception as exc:  # record, never retry automatically
            table.append({'gpu': g, 'error': f'{type(exc).__name__}: {exc}'[:200]})
            continue
        best = max(r['rows'], key=lambda x: x['windows_per_s'])
        table.append({'gpu': g, 'device': r['gpu'], 'best_batch': best['batch'], 'windows_per_s': best['windows_per_s'],
                      'peak_gb': best['peak_allocated_gb'],
                      'windows_per_dollar': round(best['windows_per_s'] * 3600 / PRICE_PER_HOUR[g])})
    out = REPO / 'reports' / 'gpu_encode_benchmark.json'
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(table, indent=1))
    print(json.dumps(table, indent=1))


GPU = 'H100'  # measured best windows per dollar (reports/gpu_encode_benchmark.json)


def _ap(scores, labels):
    import numpy as np
    if labels.sum() == 0:
        return None
    o = np.argsort(-scores, kind='stable')
    y = labels[o]
    tp = np.cumsum(y)
    prec = tp / np.arange(1, len(y) + 1)
    return float((prec * y).sum() / y.sum())


def _song_labels(row, n_cells):
    """Per 40 ms cell: any (model-view) head in the cell; plus flux-peak cells from mel10."""
    import numpy as np
    from beatfcn.train.data import Corpus
    from beatfcn.ingest import offset as OFF
    corpus = Corpus(DATA)
    corpus.train_rows([row])
    chart = corpus.chart(row['chart_id'])
    y = np.zeros(n_cells, np.float32)
    for e in chart.events:
        c = e.onset_us // 40_000
        if 0 <= c < n_cells:
            y[c] = 1
    mel = np.asarray(corpus.mel(row['audio_sha256']), np.float32)
    env = OFF.flux_envelope(mel)
    peak = np.zeros(n_cells, bool)
    pk = np.where((env[1:-1] > env[:-2]) & (env[1:-1] >= env[2:]) & (env[1:-1] > OFF.PEAK_MIN))[0] + 1
    peak[np.clip(pk // 4, 0, n_cells - 1)] = True
    return mel, y, peak


@app.function(image=gpu_image, gpu=GPU, volumes={WEIGHTS: weights, DATA: data}, cpu=4.0, memory=32768, timeout=3600)
def layer_probe(train_rows: list, val_rows: list, epochs: int = 2, batch: int = 16, lr: float = 1e-3):
    """S0b: one linear onset probe per layer (24) on frozen full-band features (8 x 1024 per 40 ms cell).

    Metrics on val songs: AP over all cells (onset), and AP restricted to cells holding a spectral-flux
    peak (selection proxy: was this salient sound charted?). Reported per mode.
    """
    import numpy as np
    import torch
    from torch.nn import functional as F
    from beatfcn.audio import memory as M
    t0 = time.time()
    enc = load_encoder()
    layers = list(range(1, 25))
    pad = M.silence_row()
    W = torch.zeros(24, 8 * 1024, device='cuda', requires_grad=True)
    b = torch.zeros(24, device='cuda', requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=lr)

    def windows(row):
        n_frames = np.load(f"{DATA}/mel10/{row['audio_sha256']}.npy", mmap_mode='r').shape[0]
        p = M.plan(n_frames)
        mel, y, peak = _song_labels(row, p.cells)
        for i in range(0, len(p.starts), batch):
            st = p.starts[i:i + batch]
            cells = np.concatenate([np.arange((s + M.EDGE) // 4, (s + M.EDGE) // 4 + M.HOP // 4) for s in st])
            ok = cells < p.cells
            taps = enc(M.window_batch(mel, st, pad), layers)
            feats = torch.stack([taps[l].reshape(-1, 8 * 1024) for l in layers])  # [24, N, 8192]
            yield feats[:, torch.from_numpy(ok).cuda()], y[cells[ok]], peak[cells[ok]]

    def logits(feats):
        x = F.layer_norm(feats.float(), (feats.shape[-1],))
        return torch.einsum('lnd,ld->ln', x, W) + b[:, None]

    rng = np.random.default_rng(0)
    pos = np.mean([r.get('n_events', 0) / max(r['audio_sec'] * 25, 1) for r in train_rows])
    pw = torch.tensor(min(max((1 - pos) / max(pos, 1e-3), 1.0), 20.0), device='cuda')
    n_train = 0
    for ep in range(epochs):
        for k in rng.permutation(len(train_rows)):
            for feats, y, _ in windows(train_rows[k]):
                yt = torch.from_numpy(y).cuda()[None].expand(24, -1)
                loss = F.binary_cross_entropy_with_logits(logits(feats), yt, pos_weight=pw)
                opt.zero_grad()
                (loss * 24).backward()
                opt.step()
                n_train += yt.shape[1]
    t_train = time.time() - t0
    scores, labels, peaks, modes = [], [], [], []
    with torch.no_grad():
        for row in val_rows:
            for feats, y, peak in windows(row):
                scores.append(logits(feats).cpu().numpy())
                labels.append(y)
                peaks.append(peak)
                modes.append(np.full(len(y), row['mode'], dtype=object))
    S, Y, Pk, Md = np.concatenate(scores, 1), np.concatenate(labels), np.concatenate(peaks), np.concatenate(modes)
    rows = []
    for li, l in enumerate(layers):
        r = {'layer': l, 'ap_onset': _ap(S[li], Y), 'ap_select_on_peaks': _ap(S[li][Pk], Y[Pk])}
        for mode in sorted(set(Md)):
            mm = Md == mode
            r[f'ap_onset_{mode}'] = _ap(S[li][mm], Y[mm])
            r[f'ap_select_{mode}'] = _ap(S[li][mm & Pk], Y[mm & Pk])
        rows.append(r)
    report = {'gpu': torch.cuda.get_device_name(), 'train_songs': len(train_rows), 'val_songs': len(val_rows),
              'train_cells_seen': n_train, 'val_cells': int(len(Y)), 'val_pos_rate': float(Y.mean()),
              'val_peak_cells': int(Pk.sum()), 'val_pos_rate_on_peaks': float(Y[Pk].mean()) if Pk.any() else None,
              'pos_weight': float(pw), 'epochs': epochs, 'seconds_total': round(time.time() - t0, 1),
              'seconds_train': round(t_train, 1), 'layers': rows}
    os.makedirs(f'{DATA}/reports', exist_ok=True)
    with open(f'{DATA}/reports/layer_probe.json', 'w') as f:
        json.dump(report, f, indent=1)
    data.commit()
    return report


@app.function(image=gpu_image, gpu=GPU, volumes={WEIGHTS: weights, DATA: data}, cpu=8.0, memory=32768, timeout=3600)
def encode_songs(shas: list, layers: list, taps_key: str, batch: int = 32, pca: str = '', quant: bool = False,
                 group: int = 4, mode: str = 'auto', writers: int = 8, probe_batches: int = 16,
                 min_util: float = 0.0, check_after_s: float = 20.0):
    """Write taps/{key}/{sha}/L{l}.npy and S20.npy (2 s means of layer 20, full width, float16).

    pca: frozen stats in /data/pca. With comp_{l}: PCA projection; without (fit_i8): full width, centred.
    quant: int8 with scale_{l}. Pipeline and modes: beatfcn/audio/encode_pipeline.py. mode='auto' times
    'compile-graph' and 'graph' on synthetic windows in this container and keeps the faster one.
    """
    import numpy as np
    import torch
    from beatfcn.audio import memory as M
    from beatfcn.audio import encode_pipeline as P
    from beatfcn.cloud.gpu_util import GpuUtil
    mon, t0 = GpuUtil(), time.time()
    want = sorted(set(layers) | {20})

    def todo(sha):
        base = f'{DATA}/taps/{taps_key}/{sha}'
        return not (all(os.path.exists(f'{base}/L{l}.npy') for l in layers) and os.path.exists(f'{base}/S20.npy'))
    work = [sha for sha in shas if todo(sha)]
    if not work:
        mon.close()
        return {'songs': 0, 'windows': 0, 'seconds': round(time.time() - t0, 1)}
    mean, comp, scale = {}, {}, {}
    if pca:
        z = np.load(f'{DATA}/pca/{pca}.npz')
        mean = {l: torch.from_numpy(z[f'mean_{l}']).cuda() for l in layers}
        comp = {l: torch.from_numpy(z[f'comp_{l}']).cuda() for l in layers if f'comp_{l}' in z}
        if quant:
            scale = {l: torch.from_numpy(z[f'scale_{l}']).cuda() for l in layers}
    qf = P.quantiser(layers, mean, scale, comp)
    fwd = P.tap_forward(load_encoder(), want)
    shape, probe = (1, M.WINDOW, 128), {}
    with torch.inference_mode():
        if mode == 'auto':
            x = torch.randn(batch, *shape).to(torch.bfloat16).pin_memory()
            best = None
            for m in ('compile-graph', 'graph'):
                try:
                    ts = time.time()
                    r = P.TapRunner(fwd, want, layers, qf, batch, shape, 'cuda', mode=m)
                    r(x, None)
                    torch.cuda.synchronize()
                    setup, _ = time.time() - ts, mon.take()
                    t = time.time()
                    for _ in range(probe_batches):
                        r(x, None)
                    torch.cuda.synchronize()
                    wps = batch * probe_batches / (time.time() - t)
                    probe[m] = {'windows_per_s': round(wps, 1), 'util': mon.take()['util_mean'], 'setup_s': round(setup, 1)}
                    if best is None or wps > best[1]:
                        best = (r, wps, m)
                except Exception as exc:  # a mode that cannot be captured/compiled here is skipped
                    probe[m] = {'error': repr(exc)[:300]}
            if best is None:
                raise RuntimeError(f'no encoder mode works: {probe}')
            runner, mode = best[0], best[2]
        else:
            runner = P.TapRunner(fwd, want, layers, qf, batch, shape, 'cuda', mode=mode)

        def save(sha, arrays):
            base = f'{DATA}/taps/{taps_key}/{sha}'
            os.makedirs(base, exist_ok=True)
            for name, a in arrays:  # S20 last: its presence marks a finished song
                np.save(f'{base}/{name}.npy', a)
        util = {}

        def guard(secs, wins):  # cost guard: stop early when the GPU is not kept busy enough
            if not min_util or secs < check_after_s:
                return None
            u = mon.peek()
            return None if u is None or u >= min_util else f'steady GPU util {u}% < {min_util}% after {secs:.0f} s ({wins / secs:.1f} windows/s)'
        stats = P.run(work, lambda sha: np.load(f'{DATA}/mel10/{sha}.npy'), save, runner, layers,
                      group=group, writers=writers, device='cuda', on_first_group=mon.take,
                      on_steady_end=lambda: util.update(mon.take()), guard=guard)
    data.commit()  # Modal also commits in the background every few seconds; never from the main loop
    mon.close()
    return {**stats, 'seconds': round(time.time() - t0, 1), 'mode': mode, 'probe': probe, 'steady_util': util.get('util_mean'),
            'gpu': torch.cuda.get_device_name()}


@app.function(image=gpu_image, volumes={DATA: data}, cpu=4.0, memory=32768, timeout=3600)
def fit_i8(src_key: str = 'v2_L9-16-24', layers: list = (9, 16, 24), songs: int = 40, seed: int = 20261023):
    """Per-dimension int8 stats for full-width taps: mean_{l}, scale_{l} = 6 sigma / 127 (no comp_{l}).

    Fit on TRAIN songs of the full-width float16 cache; saved to pca/i8_{layers}.npz.
    """
    import numpy as np
    from concurrent.futures import ThreadPoolExecutor
    import json as _json
    rows = [_json.loads(l) for l in open(f'{DATA}/manifest/charts.jsonl')]
    train = {r['audio_sha256'] for r in rows if r.get('split') == 'train'}
    have = sorted(s for s in os.listdir(f'{DATA}/taps/{src_key}') if s in train)
    rng = np.random.default_rng(seed)
    pick = [have[i] for i in rng.choice(len(have), size=min(songs, len(have)), replace=False)]
    out, report = {}, {'src_key': src_key, 'songs': len(pick), 'layers': list(layers), 'sigmas': 6}
    with ThreadPoolExecutor(8) as ex:
        for l in layers:
            s1 = s2 = n = 0
            for a in ex.map(lambda sha: np.load(f'{DATA}/taps/{src_key}/{sha}/L{l}.npy'), pick):
                x = a.astype(np.float64).reshape(-1, a.shape[-1])
                s1, s2, n = s1 + x.sum(0), s2 + (x * x).sum(0), n + len(x)
            mu = s1 / n
            sd = np.sqrt(np.maximum(s2 / n - mu * mu, 1e-12))
            out[f'mean_{l}'], out[f'scale_{l}'] = mu.astype(np.float32), (6 * sd / 127).astype(np.float32)
            report[f'L{l}'] = {'cells': int(n), 'sd_median': float(np.median(sd)), 'sd_max': float(sd.max())}
    name = 'i8_' + '-'.join(str(l) for l in layers)
    os.makedirs(f'{DATA}/pca', exist_ok=True)
    np.savez(f'{DATA}/pca/{name}.npz', **out)
    data.commit()
    return {'name': name, **report}


@app.local_entrypoint()
def i8(src_key: str = 'v2_L9-16-24', layers: str = '9,16,24', songs: int = 40):
    print(json.dumps(fit_i8.remote(src_key, [int(x) for x in layers.split(',')], songs), indent=1))


@app.local_entrypoint()
def run_probe(rows_json: str, out: str = 'reports/layer_probe.json', epochs: int = 2):
    rows = json.loads(Path(rows_json).read_text())
    r = layer_probe.remote(rows['train'], rows['val'], epochs=epochs)
    Path(out).write_text(json.dumps(r, indent=1))
    best = sorted(r['layers'], key=lambda x: -(x['ap_onset'] or 0))[:5]
    print(json.dumps({k: r[k] for k in ('gpu', 'train_songs', 'val_songs', 'val_pos_rate', 'val_pos_rate_on_peaks', 'seconds_total')}))
    for x in r['layers']:
        print(x['layer'], round(x['ap_onset'] or 0, 3), round(x['ap_select_on_peaks'] or 0, 3))


@app.local_entrypoint()
def encode(shas_file: str, taps_key: str = 'v1_L8-16-24', layers: str = '8,16,24', per_container: int = 200, pca: str = '',
           quant: bool = False, gpu: str = '', batch: int = 32, limit: int = 0, group: int = 4, offset: int = 0,
           mode: str = 'auto', writers: int = 8, cpu: float = 0, timeout: int = 0, min_util: float = 0.0,
           check_after_s: float = 20.0):
    """Encode songs listed (one sha per line) in parallel containers; idempotent. gpu overrides the
    container GPU (encoding is volume-bound, so cheaper GPUs can win per song); limit: first N songs only."""
    shas = [s.strip() for s in Path(shas_file).read_text().split() if s.strip()]
    shas = shas[offset:offset + limit] if limit else shas[offset:]
    ls = [int(x) for x in layers.split(',')]
    chunks = [shas[i:i + per_container] for i in range(0, len(shas), per_container)]
    opts = {k: v for k, v in (('gpu', gpu), ('cpu', cpu), ('timeout', timeout)) if v}  # timeout: hard cost cap per container
    fn = encode_songs.with_options(**opts) if opts else encode_songs
    t = time.time()
    res = list(fn.starmap([(c, ls, taps_key, batch, pca, quant, group, mode, writers, 16, min_util, check_after_s)
                           for c in chunks]))
    keys = ('mode', 'probe', 'steady_util', 'steady_windows_per_s', 'steady_host_s', 'flush_s', 'write_MBps_per_thread', 'gpu',
            'aborted', 'fallback_buffers')
    print(json.dumps({'chunks': len(chunks), 'songs': sum(r['songs'] for r in res), 'windows': sum(r['windows'] for r in res),
                      'container_s': round(sum(r['seconds'] for r in res), 1), 'wall_s': round(time.time() - t, 1),
                      'per_container': [{k: r.get(k) for k in keys} for r in res]}, indent=1))


@app.function(image=gpu_image, gpu=GPU, volumes={DATA: data}, cpu=8.0, memory=65536, timeout=3600)
def fit_pca(src_key: str = 'v2_L9-16-24', layers: list = (9, 16, 24), dims: int = 256, songs: int = 150,
            cells_per_song: int = 300, seed: int = 20261023):
    """Frozen per-tap PCA (shared over the 8 bands) fit on TRAIN songs' cached taps; saved to pca/{key}.npz.

    Fixed after fitting, so its outputs are cacheable (FIRST_CANDIDATE §2.6). Reports retained variance.
    """
    import numpy as np
    import torch
    import json as _json
    rows = [_json.loads(l) for l in open(f'{DATA}/manifest/charts.jsonl')]
    train = {r['audio_sha256'] for r in rows if r.get('split') == 'train'}
    have = sorted(s for s in os.listdir(f'{DATA}/taps/{src_key}') if s in train)
    rng = np.random.default_rng(seed)
    pick = [have[i] for i in rng.choice(len(have), size=min(songs, len(have)), replace=False)]
    out, report = {}, {'src_key': src_key, 'songs': len(pick), 'dims': dims, 'layers': list(layers)}
    for l in layers:
        xs = []
        for sha in pick:
            a = np.load(f'{DATA}/taps/{src_key}/{sha}/L{l}.npy')  # one sequential read; random mmap reads are slow on volumes
            idx = np.sort(rng.choice(a.shape[0], size=min(cells_per_song, a.shape[0]), replace=False))
            xs.append(a[idx].astype(np.float32).reshape(-1, a.shape[-1]))
        X = torch.from_numpy(np.concatenate(xs)).cuda()
        mu = X.mean(0)
        Xc = X - mu
        cov = Xc.T @ Xc / (len(Xc) - 1)
        evals, evecs = torch.linalg.eigh(cov.double())
        evals, evecs = evals.flip(0), evecs.flip(1)
        comp = evecs[:, :dims].float()
        kept = float(evals[:dims].sum() / evals.sum())
        rec = Xc @ comp @ comp.T
        rel = float(((rec - Xc).pow(2).sum(1) / Xc.pow(2).sum(1).clamp_min(1e-9)).mean())
        out[f'mean_{l}'], out[f'comp_{l}'] = mu.cpu().numpy(), comp.cpu().numpy()
        # int8 storage scale per component: 4 sigma maps to 127 (components are decorrelated, sigma = sqrt(eigenvalue))
        out[f'scale_{l}'] = (4 * evals[:dims].clamp_min(1e-12).sqrt() / 127).float().cpu().numpy()
        report[f'L{l}'] = {'vectors': len(X), 'variance_kept': round(kept, 4), 'mean_rel_recon_err': round(rel, 4),
                           'variance_kept_128': round(float(evals[:128].sum() / evals.sum()), 4),
                           'variance_kept_512': round(float(evals[:512].sum() / evals.sum()), 4)}
    key = f'pca{dims}_' + '-'.join(map(str, layers))
    os.makedirs(f'{DATA}/pca', exist_ok=True)
    np.savez(f'{DATA}/pca/{key}.npz', **out)
    with open(f'{DATA}/pca/{key}.json', 'w') as f:
        _json.dump(report, f, indent=1)
    data.commit()
    return report


@app.local_entrypoint()
def pca(dims: int = 512):
    r = fit_pca.remote(dims=dims)
    Path('reports/pca_fit.json').write_text(json.dumps(r, indent=1))
    print(json.dumps(r, indent=1))


@app.function(image=gpu_image, gpu='L40S', volumes={WEIGHTS: weights}, cpu=4.0, memory=32768, timeout=1800)
def bench_encode2(batches=(16, 32), repeats: int = 20, modes=('eager', 'default', 'max-autotune-no-cudagraphs')):
    """Steady-state teacher throughput for the cache layers {9,16,20,24}: eager vs torch.compile modes.
    Reports windows/s, GPU utilisation and the max |compiled - eager| on the deepest tap."""
    import torch
    from beatfcn.audio import memory as M
    from beatfcn.audio.pupu import make_rope
    from beatfcn.cloud.gpu_util import GpuUtil
    enc = load_encoder()
    teacher = enc.teacher
    want = (9, 16, 20, 24)
    grid = (M.WINDOW // M.FRAMES_PER_CELL, M.BANDS)
    rope = make_rope(grid, teacher.head_dim, device='cuda')
    keep = slice(M.EDGE // M.FRAMES_PER_CELL, (M.EDGE + M.HOP) // M.FRAMES_PER_CELL)

    def fwd(w):
        x = teacher.patch_embed(w)
        out = []
        for l, block in enumerate(teacher.teacher.blocks, 1):
            x = block(x, rope)
            if l in want:
                out.append(x.reshape(x.shape[0], *grid, -1)[:, keep])
            if l >= max(want):
                break
        return out

    mon, rows, ref = GpuUtil(), [], {}
    for mode in modes:
        f = fwd if mode == 'eager' else torch.compile(fwd, mode=mode)
        for b in batches:
            g = torch.Generator().manual_seed(b)
            x = torch.randn(b, 1, 1024, 128, generator=g).to('cuda', torch.bfloat16)
            with torch.inference_mode():
                t = time.time()
                y = f(x)
                torch.cuda.synchronize()
                first = time.time() - t
                for _ in range(3):
                    f(x)
                torch.cuda.synchronize()
                mon.take()
                t = time.time()
                for _ in range(repeats):
                    y = f(x)
                torch.cuda.synchronize()
                dt = (time.time() - t) / repeats
                util = mon.take()
            deep = y[-1].float()
            if mode == 'eager':
                ref[b] = deep
            err = float((deep - ref[b]).abs().max()) if b in ref else None
            rel = float(((deep - ref[b]).norm() / ref[b].norm())) if b in ref else None
            rows.append({'mode': mode, 'batch': b, 'windows_per_s': round(b / dt, 1), 'first_call_s': round(first, 1),
                         'util': util['util_mean'], 'max_abs_diff_L24': err, 'rel_diff_L24': rel})
            print(json.dumps(rows[-1]), flush=True)
    mon.close()
    return {'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__, 'rows': rows}


@app.local_entrypoint()
def bench2(gpus: str = 'L40S,H100'):
    calls = {g: bench_encode2.with_options(gpu=g).spawn() for g in gpus.split(',')}
    for g, c in calls.items():
        print(json.dumps(c.get(), indent=1))
