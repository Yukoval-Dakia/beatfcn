"""Modal app: P1 training (candidate B) on cached frozen taps.

  .venv/bin/modal run --detach beatfcn/cloud/train_app.py::main --run-id p1-pilot-001 --steps 3000

Data workers build requests/cores on CPU in parallel (torch DataLoader over an IterableDataset);
the GPU only runs the event model. Logs: /data/runs/{run}/log.jsonl; best checkpoint: best.pt.
"""
from __future__ import annotations
import json
import math
import os
import time
from dataclasses import asdict, replace
from pathlib import Path
import modal

_HERE = Path(__file__).resolve()
REPO = _HERE.parents[2] if len(_HERE.parents) > 2 else _HERE.parent
DATA = '/data'
app = modal.App('beatfcn-train')
POOL, RESIDENCE = 6, 16  # per worker and mode: resident songs, draws (of up to 4 cores) each before eviction
data = modal.Volume.from_name('beatfcn-data', create_if_missing=True)
image = (modal.Image.debian_slim(python_version='3.12')
         .pip_install('torch==2.14.1', index_url='https://download.pytorch.org/whl/cu130')
         .pip_install('numpy==2.2.6', 'safetensors==0.5.3', 'pyyaml==6.0.3', 'jsonschema==4.26.0')
         .add_local_python_source('beatfcn'))


def _audit_corpus(corpus_args, rows):
    from beatfcn.train.data import Corpus
    c = Corpus(**corpus_args)
    c.train_rows(rows)
    return c


def _stream(rows, corpus_args, seed, micro_batch, taps, top_tap, meta=True, static=None):
    """Training micro-batches. meta=True (training): workers build only the metadata (collate_meta,
    ~50 KB per core; songs are never read by workers) and the main process attaches audio from a
    GPU-resident SongPool. meta=False: full CPU collate with whole songs read in the worker."""
    import torch
    from beatfcn.train.data import Corpus
    from beatfcn.train.loop import CoreSampler
    from beatfcn.model.batch import collate, collate_meta, STATIC_CORE

    class Stream(torch.utils.data.IterableDataset):
        def __iter__(self):
            info = torch.utils.data.get_worker_info()
            wid = info.id if info else 0
            corpus = Corpus(**corpus_args)
            song_cache = corpus.song_info if meta else (lambda sha: corpus.song(sha, in_memory=True))
            smp = CoreSampler(corpus.train_rows(rows), corpus.chart, song_cache, seed * 1000 + wid,
                              salient=corpus.salient_us, pool=POOL, residence=RESIDENCE,
                              limits=STATIC_CORE if static else None)
            from beatfcn.model import features as F
            t_w = time.time()
            Dt, quant = corpus.tap_dim, corpus.quant
            first = True
            while True:
                items = []
                while len(items) < micro_batch:  # 4 cores per song bucket -> 2+ songs per microbatch
                    items += smp.draw(min(4, micro_batch - len(items)))
                try:
                    if meta:
                        bt = collate_meta([a for a, _ in items], [b for _, b in items], taps, top_tap, Dt, 1024, quant, static)
                    else:
                        bt = collate([a for a, _ in items], [b for _, b in items], taps, top_tap)
                except F.SkipExample as exc:  # never let one core take the worker (and the run) down
                    key = 'collate:' + str(exc)[:40]
                    smp.skipped[key] = smp.skipped.get(key, 0) + 1
                    continue
                if first:  # one line per worker in the container log: where start-up time went
                    print(json.dumps({'worker': wid, 'first_batch_s': round(time.time() - t_w, 1), 'skipped': dict(smp.skipped)}), flush=True)
                    first = False
                yield bt, dict(smp.skipped)
    return Stream()


def _pair_stream(rows, corpus_args, seed, n_pairs, taps):
    """Counterfactual support-swap batches from same-recording, same-mode chart pairs (train only)."""
    import torch

    class Stream(torch.utils.data.IterableDataset):
        def __iter__(self):
            import numpy as np
            from collections import defaultdict
            from beatfcn.model import cf
            from beatfcn.train.data import Corpus
            info = torch.utils.data.get_worker_info()
            rng = np.random.default_rng(seed * 7919 + (info.id if info else 0))
            corpus = Corpus(**corpus_args)
            corpus.train_rows(rows)
            groups = defaultdict(list)
            for r in rows:
                groups[(r['audio_sha256'], r['mode'])].append(r)
            keys = [k for k, v in groups.items() if len(v) >= 2]
            heads = {}

            def head_array(cid):
                if cid not in heads:
                    heads[cid] = np.unique(np.array([e.onset_us for e in corpus.chart(cid).events], np.int64))
                return heads[cid]
            while True:
                pairs, songs = [], []
                for _ in range(50 * n_pairs):
                    k = keys[rng.integers(len(keys))]
                    a, b = rng.choice(len(groups[k]), size=2, replace=False)
                    ra, rb = groups[k][a], groups[k][b]
                    try:
                        p = cf.build_pair(head_array(ra['chart_id']), head_array(rb['chart_id']), int(ra['audio_sec'] * 1e6), rng)
                    except ValueError:
                        p = None
                    if p:
                        pairs.append(p)
                        songs.append(corpus.song(k[0]))
                    if len(pairs) == n_pairs:
                        break
                if pairs:
                    yield cf.collate_pairs(pairs, songs, taps, None)
    return Stream()


def _pair_audit(rows, corpus, n_try=600, seed=0):
    """Design §3.4 gate: eligible song groups and independent pair tasks among train charts."""
    import numpy as np
    from collections import defaultdict
    from beatfcn.model import cf
    groups = defaultdict(list)
    for r in rows:
        groups[(r['audio_sha256'], r['mode'])].append(r)
    multi = [k for k, v in groups.items() if len(v) >= 2]
    rng = np.random.default_rng(seed)
    ok_groups, ok = set(), 0
    if not multi:  # no song with two charts: nothing to pair
        return {'groups_with_2plus_charts': 0, 'sampled': 0, 'success_rate': 0.0, 'eligible_groups_seen': 0,
                'est_eligible_pairs': 0, 'by_mode': {'mania_4k': 0, 'phira': 0}}
    for _ in range(n_try):
        k = multi[rng.integers(len(multi))]
        a, b = rng.choice(len(groups[k]), size=2, replace=False)
        try:
            ha = np.unique([e.onset_us for e in corpus.chart(groups[k][a]['chart_id']).events])
            hb = np.unique([e.onset_us for e in corpus.chart(groups[k][b]['chart_id']).events])
        except ValueError:
            continue
        if cf.build_pair(ha, hb, int(groups[k][a]['audio_sec'] * 1e6), rng):
            ok += 1
            ok_groups.add(k)
    rate = ok / max(n_try, 1)
    total_pairs = sum(len(v) * (len(v) - 1) // 2 for k, v in groups.items() if len(v) >= 2)
    return {'groups_with_2plus_charts': len(multi), 'sampled': n_try, 'success_rate': round(rate, 3),
            'eligible_groups_seen': len(ok_groups), 'est_eligible_pairs': int(rate * total_pairs),
            'by_mode': {m: sum(1 for k in multi if k[1] == m) for m in ('mania_4k', 'phira')}}


def _cached_songs(taps_key: str) -> set:
    """Songs with a complete cache. Reads taps/<key>/_index.json when present; otherwise lists the
    directory once (8.7k entries + exists checks = ~160 s of FUSE round-trips on the volume, measured)
    and writes the index for every later launch."""
    tdir = f'{DATA}/taps/{taps_key}'
    idx = f'{tdir}/_index.json'
    if os.path.exists(idx):
        with open(idx) as f:
            return set(json.load(f))
    have = sorted(s for s in os.listdir(tdir) if not s.startswith('_') and os.path.exists(f'{tdir}/{s}/S20.npy'))
    with open(idx + '.tmp', 'w') as f:
        json.dump(have, f)
    os.replace(idx + '.tmp', idx)
    return set(have)


def _select_rows(taps_key, taps, pca, whiten, songs_file, max_train_songs, val_songs_file):
    """Manifest rows usable with this cache, split into train / val as train() sees them."""
    from beatfcn.train.data import Corpus, load_manifest
    have = _cached_songs(taps_key)
    probe = Corpus(root=DATA, taps_key=taps_key, layers=taps, pca=pca, whiten=whiten)
    rows = [r for r in load_manifest(DATA) if r['audio_sha256'] in have]
    n0 = len(rows)
    rows = [r for r in rows if probe.usable(r)]
    train_rows = [r for r in rows if r['split'] == 'train']
    if songs_file:  # restrict to a fixed song list (pilot / equivalence checks)
        allowed = set(open(songs_file).read().split())
        train_rows = [r for r in train_rows if r['audio_sha256'] in allowed]
    if max_train_songs:
        keep = sorted({r['audio_sha256'] for r in train_rows})[:max_train_songs]
        train_rows = [r for r in train_rows if r['audio_sha256'] in set(keep)]
    val_rows = [r for r in rows if r['split'] == 'val']
    if val_songs_file:  # fixed validation songs so runs with different caches stay comparable
        vallow = set(open(val_songs_file).read().split())
        val_rows = [r for r in val_rows if r['audio_sha256'] in vallow]
    info = {'dup_collapsed_charts': sum(1 for r in rows if not r.get('kept')), 'dup_ineligible_dropped': n0 - len(rows)}
    return train_rows, val_rows, info


def _prep_key(**kw):
    import hashlib
    return hashlib.sha256(json.dumps(kw, sort_keys=True, default=str).encode()).hexdigest()[:16]


@app.function(image=image, volumes={DATA: data}, cpu=8.0, memory=32768, timeout=3600)
def prep_train(taps_key: str, taps: tuple, pca: str = '', whiten: bool = False, songs_file: str = '', max_train_songs: int = 0,
               val_songs_file: str = '', val_cores: int = 256, audit: bool = True):
    """CPU-only start-up work for train(): the fixed validation cores and the counterfactual pair audit,
    saved under /data/prep/<key>.pt. train() loads them, so the GPU container starts stepping within
    seconds instead of idling ~2 minutes (measured: 68 s val build + 54 s audit) per launch."""
    import torch
    from beatfcn.train.data import Corpus
    from beatfcn.train.loop import CoreSampler
    t0 = time.time()
    key = _prep_key(taps_key=taps_key, taps=list(taps), pca=pca, whiten=whiten, songs_file=songs_file,
                    max_train_songs=max_train_songs, val_songs_file=val_songs_file, val_cores=val_cores, audit=audit)
    path = f'{DATA}/prep/{key}.pt'
    if os.path.exists(path):
        return {'key': key, 'cached': True}
    train_rows, val_rows, info = _select_rows(taps_key, taps, pca, whiten, songs_file, max_train_songs, val_songs_file)
    corpus = Corpus(root=DATA, taps_key=taps_key, layers=taps, pca=pca, whiten=whiten)
    vsmp = CoreSampler(corpus.train_rows(val_rows), corpus.chart, corpus.song, 7, salient=corpus.salient_us)
    val = vsmp.fixed(val_cores, 20261022)
    t_val = time.time() - t0
    corpus.train_rows(train_rows)  # registers the train charts the audit loads
    aud = _pair_audit(train_rows, corpus) if audit else None
    os.makedirs(f'{DATA}/prep', exist_ok=True)
    torch.save({'val': val, 'audit': aud, 'n_train': len(train_rows), 'n_val': len(val_rows), 'info': info,
                'val_build_s': round(t_val, 1), 'audit_s': round(time.time() - t0 - t_val, 1)}, path)
    data.commit()
    return {'key': key, 'cached': False, 'val_cores': len(val), 'val_build_s': round(t_val, 1),
            'audit_s': round(time.time() - t0 - t_val, 1), 'audit': aud}


@app.function(image=image, gpu='L40S', volumes={DATA: data}, cpu=12.0, memory=49152, timeout=24 * 3600)
def train(run_id: str, steps: int = 3000, micro_batch: int = 8, accumulation: int = 4, taps_key: str = 'v1_L8-16-24',
          taps: tuple = (8, 16, 24), workers: int = 16, val_cores: int = 256, max_train_songs: int = 0,
          lr: float = 2e-4, warmup: int = 500, validate_every: int = 500, tap_dim: int = 1024, songs_file: str = '',
          variant: dict | None = None, cf_every: int = 4, cf_pairs: int = 8, cf_start: int = 2000, val_songs_file: str = '',
          pca: str = '', cf_force: bool = False, stop_at: int = 0, whiten: bool = False, also: list | None = None,
          device: str = 'cuda', min_util: float = 0.0, check_after_s: float = 120.0, prep: str = '',
          pool_songs: int = 160, prefetch: int = 6, graph: bool = False):
    """stop_at: pause at this step (checkpointed) while the LR schedule still spans `steps`; a later call
    with the same run_id resumes from last.pt (S1 compares both variants at stop_at, then extends the winner).

    also: [{'run_id': ..., 'variant': {...}}, ...] further models trained in this container on the SAME
    micro-batches (paired comparison; the data path, CPUs and container overheads are paid once). Each run
    keeps its own optimiser, checkpoints and log under /data/runs/{run_id}.

    min_util: cost guard. From check_after_s seconds into training, a log interval whose GPU utilisation is
    below min_util checkpoints every run (resumable) and stops.

    graph: one fixed batch shape (batch.STATIC) and the whole optimiser step replayed as a CUDA graph
    (train.graph.GraphedStep); accumulation must be 1."""
    import numpy as np
    import torch
    from beatfcn.model.p1 import P1Config, P1Model
    from beatfcn.train.data import Corpus
    from beatfcn.train.loop import CoreSampler, TrainConfig, evaluate, lr_at, merge, summarize, to_device
    from beatfcn.cloud.gpu_util import GpuUtil

    class Run:
        def __init__(self, rid, var):
            self.id, self.dir = rid, f'{DATA}/runs/{rid}'
            os.makedirs(self.dir, exist_ok=True)
            self.logf = open(f'{self.dir}/log.jsonl', 'a')
            self.mc = P1Config(taps=tuple(taps), top_tap=max(taps), tap_dim=tap_dim, **(var or {}))
            self.acc, self.best, self.start, self.use_cf = {}, None, 0, False

        def log(self, rec):
            rec['ts'] = time.time()
            self.logf.write(json.dumps(rec) + '\n')
            self.logf.flush()
            print(self.id, json.dumps(rec)[:600], flush=True)
    runs = [Run(run_id, variant)] + [Run(a['run_id'], a.get('variant')) for a in (also or [])]
    log_all = lambda rec: [r.log(dict(rec)) for r in runs]

    train_rows, val_rows, info = _select_rows(taps_key, taps, pca, whiten, songs_file, max_train_songs, val_songs_file)
    dup_rows = info['dup_collapsed_charts']
    need_audit = any(bool((v or {}).get('selection')) for v in [variant] + [a.get('variant') for a in (also or [])])
    prep_path = f'{DATA}/prep/{prep}.pt' if prep else ''
    prepped = torch.load(prep_path, weights_only=False) if prep_path and os.path.exists(prep_path) else None
    corpus_args = {'root': DATA, 'taps_key': taps_key, 'layers': taps, 'pca': pca, 'whiten': whiten}
    tc = TrainConfig(steps=steps, micro_batch=micro_batch, accumulation=accumulation, lr=lr, warmup=warmup,
                     validate_every=validate_every, val_cores=val_cores)
    corpus = Corpus(**corpus_args)
    # the training stream starts first: its workers warm up while validation cores are built
    from beatfcn.model.batch import STATIC
    if graph and accumulation != 1:
        raise ValueError('graph=True needs accumulation=1 (one replay per optimiser step)')
    resume_step = 0
    if os.path.exists(f'{runs[0].dir}/last.pt'):  # resumed runs must not replay the sample sequence from step 0
        resume_step = int(torch.load(f'{runs[0].dir}/last.pt', map_location='cpu', weights_only=False)['step'])
    stream = torch.utils.data.DataLoader(_stream(train_rows, corpus_args, tc.seed + 7919 * resume_step, micro_batch, tuple(taps),
                                                 max(taps), static=STATIC if graph else None),
                                         batch_size=None, num_workers=workers, prefetch_factor=4, persistent_workers=True,
                                         pin_memory=device == 'cuda')
    from collections import deque
    from beatfcn.model.batch import attach_audio
    from beatfcn.train.pool import SongPool
    pool = SongPool(corpus.load_numpy, tuple(taps), device, capacity=pool_songs)
    raw_it, queue = iter(stream), deque()

    def next_batch():
        """Prefetch `prefetch` metadata batches so their songs load in the background, then attach audio
        on the device for the oldest one."""
        while len(queue) < prefetch:
            b, sk = next(raw_it)
            pool.request(b['_audio']['keys'])
            queue.append((b, sk))
        b, sk = queue.popleft()
        keys = list(b['_audio']['keys'])
        attach_audio(b, pool.get, device, corpus.scale if corpus.quant else None)
        pool.release(keys)
        return b, sk

    class _It:
        def __next__(self):
            return next_batch()
    it = _It()
    t0 = time.time()
    if prepped is not None:
        if prepped['n_train'] != len(train_rows) or prepped['n_val'] != len(val_rows):
            raise RuntimeError(f'prep {prep} was built for a different row selection')
        val, val_build_s = prepped['val'], 0.0
    else:
        vsmp = CoreSampler(corpus.train_rows(val_rows), corpus.chart, corpus.song, 7, salient=corpus.salient_us)
        val = vsmp.fixed(val_cores, 20261022)
        val_build_s = round(time.time() - t0, 1)
    log_all({'event': 'data', 'train_charts': len(train_rows), 'train_songs': len({r['audio_sha256'] for r in train_rows}),
             'val_charts': len(val_rows), 'val_cores': len(val), 'val_build_s': val_build_s, 'prep': prep or None,
             'dup_collapsed_charts': dup_rows, 'dup_ineligible_dropped': info['dup_ineligible_dropped'],
             'paired_runs': [r.id for r in runs], 'modes': {m: sum(r['mode'] == m for r in train_rows) for m in ('mania_4k', 'phira')}})
    dev = torch.device(device)
    cuda = dev.type == 'cuda'
    for r in runs:
        torch.manual_seed(tc.seed)
        r.model = P1Model(r.mc).to(dev)
        decay = [p for p in r.model.parameters() if p.ndim >= 2]
        other = [p for p in r.model.parameters() if p.ndim < 2]
        lr0 = torch.tensor(tc.lr, device=dev) if graph else tc.lr  # capturable AdamW needs a tensor LR
        r.opt = torch.optim.AdamW([{'params': decay, 'weight_decay': tc.weight_decay}, {'params': other, 'weight_decay': 0.0}],
                                  lr=lr0, betas=(0.9, 0.95), fused=cuda, capturable=graph and cuda)
        r.log({'event': 'start', 'model': asdict(r.mc), 'train': asdict(tc), 'params': sum(p.numel() for p in r.model.parameters()),
               'gpu': torch.cuda.get_device_name() if cuda else device})
        if os.path.exists(f'{r.dir}/last.pt'):  # resume after preemption / restart (same run id)
            ck = torch.load(f'{r.dir}/last.pt', map_location=dev, weights_only=False)
            r.model.load_state_dict(ck['model'])
            r.opt.load_state_dict(ck['opt'])
            r.start, r.best = ck['step'], ck.get('best')
            r.log({'event': 'resume', 'step': r.start, 'best': r.best})
    if len({r.start for r in runs}) != 1:
        raise RuntimeError(f'paired runs must resume from the same step: {[(r.id, r.start) for r in runs]}')
    start_step = runs[0].start
    if any(r.mc.selection for r in runs):
        audit = (prepped or {}).get('audit') or _pair_audit(train_rows, _audit_corpus(corpus_args, train_rows))
        cf_ok = cf_force or (audit['eligible_groups_seen'] >= 50 and audit['est_eligible_pairs'] >= 300)
        for r in runs:
            if r.mc.selection:
                r.use_cf = cf_ok
                r.log({'event': 'cf_audit', **audit, 'cf_enabled': cf_ok})
        if cf_ok:
            cf_it = iter(torch.utils.data.DataLoader(_pair_stream(train_rows, corpus_args, tc.seed, cf_pairs, tuple(taps)),
                                                     batch_size=None, num_workers=4, prefetch_factor=2, persistent_workers=True))
    mon = GpuUtil()  # nvidia-smi utilisation per log interval (what the Modal dashboard shows)
    from beatfcn.train.graph import GraphedStep
    gs = GraphedStep(runs, tc.clip, dev, capture=cuda, before_capture=pool.wait_idle) if graph else None
    examples, wait_s, gpu_s, t_loop, aborted = 0, 0.0, 0.0, time.time(), None
    end = min(stop_at, tc.steps) if stop_at else tc.steps

    def checkpoint(step):
        for r in runs:
            torch.save({'model': r.model.state_dict(), 'opt': r.opt.state_dict(), 'config': asdict(r.mc), 'data': corpus_args,
                        'step': step, 'best': r.best}, f'{r.dir}/last.pt.tmp')
            os.replace(f'{r.dir}/last.pt.tmp', f'{r.dir}/last.pt')
        data.commit()
    def cf_step(step):
        # counterfactual support-swap loss (eager, variable shapes); its grads accumulate before the step
        from beatfcn.model import cf as CF
        cbt = to_device(next(cf_it), dev)
        for r in cf_runs:
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=cuda):
                l_cf, cf_acc = CF.cf_loss(r.model, dict(cbt))
            (0.1 * min(1.0, (step - cf_start + 1) / 1000) * l_cf).backward()
            merge(r.acc, {'cf': (l_cf.detach(), cbt['audio'].shape[0], cf_acc.detach() * cbt['audio'].shape[0])})
    cf_runs = [r for r in runs if r.use_cf]
    for step in range(start_step, end):
        lr = lr_at(step, tc)
        for r in runs:
            for g in r.opt.param_groups:
                if torch.is_tensor(g['lr']):
                    g['lr'].fill_(lr)
                else:
                    g['lr'] = lr
        w_sel = 0.2 * min(1.0, (step + 1) / 1000)
        cf_due = bool(cf_runs) and step >= cf_start and step % cf_every == 0
        if gs is not None:  # graphed: fill static inputs, eager CF grads, one replay = fwd + bwd + clip + step
            tw = time.time()
            bt, skipped = next(it)
            gs.fill(bt, w_sel)
            tg = time.time()
            wait_s += tg - tw
            if cf_due:
                cf_step(step)
            mets, norms = gs.step()
            for r, met, gn in zip(runs, mets, norms):
                merge(r.acc, met)
                r.gn = gn
            gpu_s += time.time() - tg
            examples += bt['mode'].shape[0]
        else:
            for _ in range(tc.accumulation):
                tw = time.time()
                bt, skipped = next(it)
                bt = to_device(bt, dev)
                tg = time.time()
                wait_s += tg - tw
                for r in runs:  # same micro-batch for every paired run
                    b = dict(bt)
                    b['w_select'] = w_sel
                    with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=cuda):
                        loss, met = r.model(b)
                    (loss / tc.accumulation).backward()
                    merge(r.acc, met)
                gpu_s += time.time() - tg  # host-side issue time (no per-micro-batch sync: the device runs ahead)
                examples += bt['mode'].shape[0]
            if cf_due:
                cf_step(step)
            for r in runs:
                r.gn = torch.nn.utils.clip_grad_norm_(r.model.parameters(), tc.clip)
                r.opt.step()
                r.opt.zero_grad(set_to_none=True)
        if (step + 1) % tc.log_every == 0:
            shared = {'step': step + 1, 'lr': lr_at(step, tc), 'examples': examples, 'data_wait_s': round(wait_s, 1),
                      'gpu_s': round(gpu_s, 1), 'elapsed_s': round(time.time() - t0, 1),
                      'pool': {k: (round(v, 1) if isinstance(v, float) else v) for k, v in pool.stats.items()},
                      'pool_gb': round(pool.resident_bytes() / 1e9, 2),
                      'peak_gb': round(torch.cuda.max_memory_allocated() / 1e9, 2) if cuda else None,
                      'gpu_util': mon.take()['util_mean'], 'skipped': skipped}
            for r in runs:
                r.log({'event': 'train', **shared, 'grad_norm': float(r.gn), 'fields': summarize(r.acc)})
                r.acc = {}
            u = shared['gpu_util']
            if min_util and u is not None and time.time() - t_loop >= check_after_s and u < min_util:
                aborted = f'GPU util {u}% < {min_util}% at step {step + 1}'
                log_all({'event': 'aborted', 'reason': aborted, 'step': step + 1})
                checkpoint(step + 1)
                break
        if (step + 1) % tc.validate_every == 0 or step + 1 in (tc.steps, end):
            for r in runs:
                v = evaluate(r.model, val, corpus.song, r.mc, dev, micro_batch)
                r.log({'event': 'val', 'step': step + 1, **v})
                if r.best is None or v['total'] < r.best:
                    r.best = v['total']
                    torch.save({'model': r.model.state_dict(), 'config': asdict(r.mc), 'data': corpus_args, 'step': step + 1,
                                'val': v}, f'{r.dir}/best.pt')
            checkpoint(step + 1)
    data.commit()
    return {'runs': {r.id: r.best for r in runs}, 'elapsed_s': round(time.time() - t0, 1), 'examples': examples,
            'aborted': aborted}


@app.local_entrypoint()
def main(run_id: str, steps: int = 3000, micro_batch: int = 8, accumulation: int = 4, max_train_songs: int = 0,
         validate_every: int = 500, val_cores: int = 256, warmup: int = 500, taps_key: str = 'v1_L8-16-24',
         taps: str = '8,16,24', tap_dim: int = 1024, songs_file: str = '', variant: str = '{}', cf_start: int = 2000,
         val_songs_file: str = '', pca: str = ''):
    print(train.remote(run_id, steps=steps, micro_batch=micro_batch, accumulation=accumulation, taps_key=taps_key,
                       taps=tuple(int(x) for x in taps.split(',')), max_train_songs=max_train_songs,
                       validate_every=validate_every, val_cores=val_cores, warmup=warmup, tap_dim=tap_dim,
                       songs_file=songs_file, variant=json.loads(variant), cf_start=cf_start, val_songs_file=val_songs_file,
                       pca=pca))


@app.function(image=image, volumes={DATA: data}, cpu=8.0, memory=32768, timeout=3600)
def diagnose(n_cores: int = 3000, taps_key: str = 'v1_L8-16-24', select: bool = False):
    """Legality audit on real data: check_targets reasons, then a small model forward on legal cores."""
    import collections
    import traceback
    import numpy as np
    import torch
    from beatfcn.model import features as F
    from beatfcn.model.batch import collate
    from beatfcn.model.p1 import P1Config, P1Model
    from beatfcn.train.data import Corpus, load_manifest
    from beatfcn.train.loop import CoreSampler
    have = _cached_songs(taps_key)
    rows = [r for r in load_manifest(DATA, select=select) if r['audio_sha256'] in have]
    corpus = Corpus(DATA, taps_key)
    smp = CoreSampler(corpus.train_rows(rows), corpus.chart, corpus.song, 1)
    reasons, examples, legal = collections.Counter(), [], []
    for k in range(n_cores):
        row = smp.charts[smp.rng.integers(len(smp.charts))]
        try:
            chart = corpus.chart(row['chart_id'])
            from beatfcn.events.requests import sample_request
            s = sample_request(chart, seed=k, audio_id=row['song'], audio_duration_us=row['duration_us'])
            if not s.ok:
                reasons['model_view_quarantine'] += 1
                continue
            c0 = F.choose_core(s, np.random.default_rng([k, 1]), row['duration_us'])
            ex = F.build_core(s, chart.events, c0, row['duration_us'])
            bad = F.check_targets(ex)
            for b in bad:
                reasons[b] += 1
                if len(examples) < 30:
                    examples.append({'chart': row['chart_id'], 'reason': b, 'c0': c0, 'kind': s.recipe['kind'],
                                     'groups': [{kk: (v if not isinstance(v, np.ndarray) else v.tolist()) for kk, v in g.items()}
                                                for g in ex.groups][:3]})
            if not bad:
                legal.append((ex, row['song']))
                reasons['legal'] += 1
        except Exception as exc:
            reasons[f'{type(exc).__name__}:{str(exc)[:60]}'] += 1
    torch.manual_seed(0)
    m = P1Model(P1Config(d=64, layers=1, heads=4, ffn=96, ptr=32, gru=48)).eval()
    model_fail = collections.Counter()
    with torch.no_grad():
        for i in range(0, min(len(legal), 600), 4):
            chunk = legal[i:i + 4]
            try:
                m(collate([a for a, _ in chunk], [corpus.song(b) for _, b in chunk], m.c.taps, m.c.top_tap))
                model_fail['ok'] += 1
            except Exception as exc:
                model_fail[str(exc)[:120]] += 1
    out = {'rows': len(rows), 'cores_tried': n_cores, 'reasons': dict(reasons), 'model_forward': dict(model_fail),
           'examples': examples[:12]}
    print(json.dumps(out)[:4000])
    return out


@app.local_entrypoint()
def audit(n_cores: int = 3000):
    r = diagnose.remote(n_cores)
    Path('reports/legality_audit_dry.json').write_text(json.dumps(r, indent=1))
    print(json.dumps({k: r[k] for k in ('rows', 'reasons', 'model_forward')}, indent=1))


@app.function(image=image, gpu='L40S', volumes={DATA: data}, cpu=8.0, memory=32768, timeout=4 * 3600)
def eval_generation(run_id: str, n_cores: int = 200, taps_key: str = 'v1_L8-16-24', greedy: bool = False, seed: int = 20261022,
                    split: str = 'val', pca: str = ''):
    """Generate every fixed val core with the run's best checkpoint and score against hidden targets."""
    import collections
    import numpy as np
    import torch
    from beatfcn.events.requests import sample_request
    from beatfcn.model import features as F
    from beatfcn.model.p1 import P1Config, P1Model
    from beatfcn.model.sample import generate_core, core_context, match_f1
    from beatfcn.train.data import Corpus, load_manifest
    ck = torch.load(f'{DATA}/runs/{run_id}/best.pt', map_location='cuda', weights_only=False)
    cfg = P1Config(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in ck['config'].items()})
    model = P1Model(cfg).cuda().eval()
    model.load_state_dict(ck['model'])
    have = _cached_songs(taps_key)
    rows = [r for r in load_manifest(DATA, split=split) if r['audio_sha256'] in have]
    corpus = Corpus(DATA, taps_key, layers=cfg.taps, pca=pca)
    trows = corpus.train_rows(rows)
    rng = np.random.default_rng(seed)
    tol = (10_000, 20_000, 50_000)
    agg = collections.defaultdict(lambda: collections.Counter())
    t0, done, illegal = time.time(), 0, 0
    while done < n_cores:
        row = trows[rng.integers(len(trows))]
        try:
            chart = corpus.chart(row['chart_id'])
            s = sample_request(chart, seed=int(rng.integers(2 ** 31)), audio_id=row['song'], audio_duration_us=row['duration_us'])
            if not s.ok:
                continue
            c0 = F.choose_core(s, np.random.default_rng(done), row['duration_us'])
            li = F.line_index_of(chart.events)
            ex = F.build_core(s, chart.events, c0, row['duration_us'], line_index=li)
            if F.check_targets(ex):
                continue
        except (F.SkipExample, ValueError, AssertionError):
            continue
        song = corpus.song(row['song'])
        holds, runs, heads = core_context(s, ex, li)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            gen = generate_core(model, ex, song, holds, runs, heads, greedy=greedy, seed=done, device='cuda')
        mode = row['mode']
        gt = [g['t_rel_us'] for g in ex.groups for _ in g['notes']]
        pr = [g['t_rel_us'] for g in gen for _ in g['notes']]
        a = agg[mode]
        for tl in tol:
            tp, npred, ntrue, _ = match_f1(pr, gt, tl)
            a[f'tp{tl // 1000}'] += tp
        a['pred'] += len(pr)
        a['true'] += len(gt)
        a['cores'] += 1
        a['empty_pred'] += not gen
        a['empty_true'] += not ex.groups
        gh = [g['t_rel_us'] for g in ex.groups for n in g['notes'] if 'end_rel_us' in n]
        ph = [g['t_rel_us'] for g in gen for n in g['notes'] if 'end_rel_us' in n]
        tp, _, _, _ = match_f1(ph, gh, 50_000)
        a['hold_tp50'] += tp
        a['hold_pred'] += len(ph)
        a['hold_true'] += len(gh)
        # group-level attribute agreement for exactly matched (<=20 ms) group times
        gtimes = {g['t_rel_us']: g for g in ex.groups}
        for g in gen:
            near = min(gtimes, key=lambda t: abs(t - g['t_rel_us']), default=None)
            if near is not None and abs(near - g['t_rel_us']) <= 20_000:
                tg = gtimes[near]
                a['groups_matched'] += 1
                a['size_equal'] += len(tg['notes']) == len(g['notes'])
                if mode == 'mania_4k':
                    a['lanes_equal'] += sorted(n['lane'] for n in tg['notes']) == sorted(n['lane'] for n in g['notes'])
                a['types_equal'] += sorted(n['type'] for n in tg['notes']) == sorted(n['type'] for n in g['notes'])
        illegal += 0  # generation is mask-constrained; independent validator lands with runtime/constraints
        done += 1
    out = {'run': run_id, 'step': ck['step'], 'cores': done, 'greedy': greedy, 'seconds': round(time.time() - t0, 1)}
    for mode, a in agg.items():
        f = {}
        for tl in tol:
            k = tl // 1000
            p = a[f'tp{k}'] / max(a['pred'], 1)
            r = a[f'tp{k}'] / max(a['true'], 1)
            f[f'f1_{k}ms'] = round(2 * p * r / max(p + r, 1e-9), 4)
            f[f'p_{k}ms'], f[f'r_{k}ms'] = round(p, 4), round(r, 4)
        hp, hr = a['hold_tp50'] / max(a['hold_pred'], 1), a['hold_tp50'] / max(a['hold_true'], 1)
        out[mode] = {**f, 'density_ratio': round(a['pred'] / max(a['true'], 1), 3), 'cores': a['cores'],
                     'empty_pred_rate': round(a['empty_pred'] / max(a['cores'], 1), 3),
                     'empty_true_rate': round(a['empty_true'] / max(a['cores'], 1), 3),
                     'hold_head_f1_50ms': round(2 * hp * hr / max(hp + hr, 1e-9), 4),
                     'group_size_acc': round(a['size_equal'] / max(a['groups_matched'], 1), 4),
                     'group_type_acc': round(a['types_equal'] / max(a['groups_matched'], 1), 4),
                     'group_lane_acc': round(a['lanes_equal'] / max(a['groups_matched'], 1), 4) if mode == 'mania_4k' else None}
    out['split'] = split
    with open(f'{DATA}/runs/{run_id}/eval_generation_{split}{"_greedy" if greedy else ""}.json', 'w') as fo:
        json.dump(out, fo, indent=1)
    data.commit()
    return out


@app.local_entrypoint()
def evaluate(run_id: str, n_cores: int = 200, greedy: bool = False, taps_key: str = 'v1_L8-16-24', split: str = 'val',
             pca: str = ''):
    r = eval_generation.remote(run_id, n_cores=n_cores, greedy=greedy, taps_key=taps_key, split=split, pca=pca)
    Path(f'reports/eval_{run_id}_{split}{"_greedy" if greedy else ""}.json').write_text(json.dumps(r, indent=1))
    print(json.dumps(r, indent=1))


weights_vol = modal.Volume.from_name('beatfcn-weights', create_if_missing=True)


def _s2_stream(rows, corpus_args, seed, micro_batch, taps, top_tap):
    import torch
    from beatfcn.train.data import Corpus
    from beatfcn.train.loop import CoreSampler
    from beatfcn.model.batch import collate
    from beatfcn.audio import memory as M
    from beatfcn.train import s2 as S2

    class Stream(torch.utils.data.IterableDataset):
        def __iter__(self):
            info = torch.utils.data.get_worker_info()
            corpus = Corpus(**corpus_args)
            smp = CoreSampler(corpus.train_rows(rows), corpus.chart, lambda sha: corpus.song(sha, in_memory=True),
                              seed * 1000 + (info.id if info else 0), salient=corpus.salient_us, pool=POOL,
                              residence=RESIDENCE)
            pad = M.silence_row()
            while True:
                items = []
                while len(items) < micro_batch:
                    items += smp.draw(min(4, micro_batch - len(items)))
                exs, songs = [a for a, _ in items], [b for _, b in items]
                bt = collate(exs, songs, taps, top_tap)
                cells = [-(-s.mel10.shape[0] // M.FRAMES_PER_CELL) for s in songs]
                plan, il, ir, ih = S2.plan_windows(exs, bt, cells)
                win = S2.mel_windows(plan, [s.mel10 for s in songs], pad).half()
                yield bt, win, il, ir, ih, dict(smp.skipped)
    return Stream()


@app.function(image=image, gpu='H100', volumes={DATA: data, '/weights': weights_vol}, cpu=16.0, memory=65536, timeout=24 * 3600)
def train_s2(run_id: str, init_run: str, steps: int = 3000, micro_batch: int = 8, accumulation: int = 4,
             taps_key: str = 'v3_pca512q_L9-16-24', taps: tuple = (9, 16, 24), pca: str = 'pca512_9-16-24',
             workers: int = 14, val_cores: int = 256, lr_event: float = 5e-5, lr_lora: float = 5e-6,
             validate_every: int = 500):
    import numpy as np
    import torch
    from beatfcn.audio.pupu import load_teacher
    from beatfcn.audio import memory as M
    from beatfcn.model.batch import collate
    from beatfcn.model.p1 import P1Config, P1Model
    from beatfcn.train import s2 as S2
    from beatfcn.train.data import Corpus, load_manifest
    from beatfcn.train.loop import CoreSampler, merge, summarize, to_device
    run = f'{DATA}/runs/{run_id}'
    os.makedirs(run, exist_ok=True)
    logf = open(f'{run}/log.jsonl', 'a')

    def log(rec):
        rec['ts'] = time.time()
        logf.write(json.dumps(rec) + '\n')
        logf.flush()
        print(json.dumps(rec)[:600], flush=True)
    dev = torch.device('cuda')
    ck = torch.load(f'{DATA}/runs/{init_run}/best.pt', map_location='cuda', weights_only=False)
    mc = P1Config(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in ck['config'].items()})
    model = P1Model(mc).to(dev)
    model.load_state_dict(ck['model'])
    teacher = load_teacher('/weights/teacher_encoder.safetensors', dtype=torch.bfloat16, device='cuda')
    lora = S2.add_lora(teacher)
    z = np.load(f'{DATA}/pca/{pca}.npz')
    mu = torch.from_numpy(z[f'mean_{mc.top_tap}']).cuda()
    comp = torch.from_numpy(z[f'comp_{mc.top_tap}']).cuda() if f'comp_{mc.top_tap}' in z else None  # fit_i8: centred only
    ti = mc.taps.index(mc.top_tap)
    have = _cached_songs(taps_key)
    corpus_args = {'root': DATA, 'taps_key': taps_key, 'layers': taps, 'pca': pca}
    corpus = Corpus(**corpus_args)
    rows = [r for r in load_manifest(DATA) if r['audio_sha256'] in have and corpus.usable(r)]
    train_rows = [r for r in rows if r['split'] == 'train']
    val_rows = [r for r in rows if r['split'] == 'val']
    vsmp = CoreSampler(corpus.train_rows(val_rows), corpus.chart, corpus.song, 7, salient=corpus.salient_us)
    val = vsmp.fixed(val_cores, 20261022)
    opt = torch.optim.AdamW([{'params': [p for p in model.parameters()], 'lr': lr_event, 'weight_decay': 0.01},
                             {'params': lora, 'lr': lr_lora, 'weight_decay': 0.0}], betas=(0.9, 0.95))
    log({'event': 'start', 'init_run': init_run, 'init_step': ck['step'], 'init_val': ck['val'].get('total'),
         'lora_params': sum(p.numel() for p in lora), 'train_charts': len(train_rows), 'val_cores': len(val)})
    pad = M.silence_row()

    def run_batch(bt, win, il, ir, ih):
        bt = to_device(bt, dev)
        bt = S2.attach_top(bt, teacher, win.to(dev), il, ir, ih, mu, comp, ti)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            return model(bt)

    def validate():
        model.eval()
        teacher.eval()
        acc = {}
        with torch.no_grad():
            for i in range(0, len(val), micro_batch):
                chunk = val[i:i + micro_batch]
                exs, songs = [a for a, _ in chunk], [corpus.song(s) for _, s in chunk]
                bt = collate(exs, songs, mc.taps, mc.top_tap)
                plan, il, ir, ih = S2.plan_windows(exs, bt, [-(-s.mel10.shape[0] // 4) for s in songs])
                _, met = run_batch(bt, S2.mel_windows(plan, [s.mel10 for s in songs], pad).half(), il, ir, ih)
                merge(acc, met)
        model.train()
        out = summarize(acc)
        out['total'] = round(sum(v['loss'] for v in out.values()), 4)
        return out
    v0 = validate()
    log({'event': 'val', 'step': 0, **v0})
    best = v0['total']
    stream = torch.utils.data.DataLoader(_s2_stream(train_rows, corpus_args, 20261024, micro_batch, mc.taps, mc.top_tap),
                                         batch_size=None, num_workers=workers, prefetch_factor=2, persistent_workers=True)
    it, acc, t0 = iter(stream), {}, time.time()
    model.train()
    for step in range(steps):
        warm = min(1.0, (step + 1) / 100)
        for g, base in zip(opt.param_groups, (lr_event, lr_lora)):
            g['lr'] = base * warm * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / steps)))
        for _ in range(accumulation):
            bt, win, il, ir, ih, skipped = next(it)
            loss, met = run_batch(bt, win, il, ir, ih)
            (loss / accumulation).backward()
            merge(acc, met)
        gn = torch.nn.utils.clip_grad_norm_(list(model.parameters()) + lora, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if (step + 1) % 25 == 0:
            log({'event': 'train', 'step': step + 1, 'grad_norm': float(gn), 'elapsed_s': round(time.time() - t0, 1),
                 'peak_gb': round(torch.cuda.max_memory_allocated() / 1e9, 2), 'fields': summarize(acc)})
            acc = {}
        if (step + 1) % validate_every == 0 or step + 1 == steps:
            v = validate()
            log({'event': 'val', 'step': step + 1, **v})
            if v['total'] < best:
                best = v['total']
                torch.save({'model': model.state_dict(), 'lora': {f'{i}': p.detach().cpu() for i, p in enumerate(lora)},
                            'config': ck['config'], 'step': step + 1, 'val': v, 'init_run': init_run}, f'{run}/best.pt')
            data.commit()
    return {'run': run_id, 'init_val': v0['total'], 'best_val': best, 'improved': best < v0['total']}


@app.local_entrypoint()
def s2(run_id: str, init_run: str, steps: int = 3000):
    print(train_s2.remote(run_id, init_run, steps=steps))


@app.local_entrypoint()
def sweep(configs: str):
    """Several training runs inside ONE app (avoids the app-create rate limit). configs: JSON list of
    {"run_id": ..., **train kwargs}; variant may be a dict. Optional "resources": {"gpu", "cpu", "memory"}
    overrides the container shape (cost-first runs use fewer CPUs / cheaper GPUs)."""
    cfgs = json.loads(Path(configs).read_text()) if configs.endswith('.json') else json.loads(configs)
    calls = []
    for c in cfgs:
        res = c.pop('resources', None)
        if c.pop('prep', False):  # CPU-only start-up work first; the GPU container then starts stepping at once
            need_audit = any(bool((v or {}).get('selection')) for v in [c.get('variant')] + [a.get('variant') for a in c.get('also', [])])
            pk = {k: c.get(k, d) for k, d in (('pca', ''), ('whiten', False), ('songs_file', ''), ('max_train_songs', 0),
                                              ('val_songs_file', ''), ('val_cores', 256))}
            r = prep_train.remote(c['taps_key'], tuple(c['taps']), audit=need_audit, **pk)
            print(c['run_id'], 'prep', json.dumps(r)[:300])
            c['prep'] = r['key']
        fn = train.with_options(**res) if res else train
        calls.append((c['run_id'], fn.spawn(**{k: (tuple(v) if k == 'taps' else v) for k, v in c.items()})))
    for rid, call in calls:
        try:
            print(rid, call.get())
        except Exception as exc:
            print(rid, 'FAILED', repr(exc)[:300])


@app.function(image=image, gpu='L40S', volumes={DATA: data}, cpu=8.0, memory=32768, timeout=4 * 3600)
def eval_controls(run_id: str, n_cores: int = 80, taps_key: str = 'v3_pca512q_L9-16-24', pca: str = 'pca512_9-16-24',
                  seed: int = 20261025):
    """Control sweep (design §13.2 item 5): same core generated at low/mid/high bands of one control;
    report how often the generated statistic moves in the requested direction."""
    import dataclasses
    import numpy as np
    import torch
    from beatfcn.events.requests import sample_request
    from beatfcn.events import controls as C
    from beatfcn.model import features as F
    from beatfcn.model.p1 import P1Config, P1Model
    from beatfcn.model.sample import generate_core, core_context
    from beatfcn.train.data import Corpus, load_manifest
    ck = torch.load(f'{DATA}/runs/{run_id}/best.pt', map_location='cuda', weights_only=False)
    cfg = P1Config(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in ck['config'].items()})
    model = P1Model(cfg).cuda().eval()
    model.load_state_dict(ck['model'])
    have = _cached_songs(taps_key)
    corpus = Corpus(DATA, taps_key, layers=cfg.taps, pca=pca)
    trows = corpus.train_rows([r for r in load_manifest(DATA, split='val') if r['audio_sha256'] in have])
    sweeps = {'density_notes_per_second': [(1.0, 1.5), (3.0, 4.0), (8.0, 10.0)],
              'hold_head_ratio': [(0.0, 0.02), (0.1, 0.2), (0.4, 0.5)]}
    rng = np.random.default_rng(seed)
    res = {f: {'monotonic': 0, 'cores': 0, 'stats': []} for f in sweeps}
    tries = 0
    while min(v['cores'] for v in res.values()) < n_cores and tries < 20 * n_cores:
        tries += 1
        row = trows[rng.integers(len(trows))]
        try:
            chart = corpus.chart(row['chart_id'])
            s = sample_request(chart, seed=int(rng.integers(2 ** 31)), audio_id=row['song'], audio_duration_us=row['duration_us'],
                               mix={'from_scratch': 1.0}, empty_p=0.0, boundary_p=0.0, control_dropout=False)
            c0 = F.choose_core(s, np.random.default_rng(tries), row['duration_us'])
        except (F.SkipExample, ValueError, AssertionError):
            continue
        song = corpus.song(row['song'])
        li = F.line_index_of(chart.events)
        for field, bands in sweeps.items():
            if res[field]['cores'] >= n_cores:
                continue
            vals = []
            for band in bands:
                req = {**s.request, 'controls': {**s.request['controls'], field: band}}
                s2 = dataclasses.replace(s, request=req)
                ex = F.build_core(s2, chart.events, c0, row['duration_us'], line_index=li)
                ex.groups = []
                holds, runs, heads = core_context(s2, ex, li)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    gen = generate_core(model, ex, song, holds, runs, heads, seed=tries, device='cuda')
                heads_n = sum(len(g['notes']) for g in gen)
                holds_n = sum('end_rel_us' in n for g in gen for n in g['notes'])
                vals.append(heads_n / 8.0 if field.startswith('density') else holds_n / max(heads_n, 1))
            res[field]['cores'] += 1
            res[field]['monotonic'] += int(vals[0] <= vals[1] <= vals[2] and vals[0] < vals[2])
            res[field]['stats'].append(vals)
    out = {'run': run_id, 'step': ck['step']}
    for f, v in res.items():
        st = np.array(v['stats']) if v['stats'] else np.zeros((0, 3))
        out[f] = {'cores': v['cores'], 'direction_correct': round(v['monotonic'] / max(v['cores'], 1), 3),
                  'mean_by_band': [round(float(x), 3) for x in st.mean(0)] if len(st) else None, 'bands': sweeps[f]}
    with open(f'{DATA}/runs/{run_id}/eval_controls.json', 'w') as fo:
        json.dump(out, fo, indent=1)
    data.commit()
    return out


@app.local_entrypoint()
def controls(run_id: str, n_cores: int = 80):
    r = eval_controls.remote(run_id, n_cores=n_cores)
    Path(f'reports/controls_{run_id}.json').write_text(json.dumps(r, indent=1))
    print(json.dumps(r, indent=1))


@app.function(image=image, volumes={DATA: data}, cpu=2.0, memory=8192, timeout=1800)
def profile_data(taps_key: str = 'v3_pca512q_L9-16-24', taps: tuple = (9, 16, 24), pca: str = 'pca512_9-16-24',
                 n_batches: int = 12, micro_batch: int = 8, pool: int = POOL):
    """Single-process cProfile of the training data path (draw -> collate), to find the CPU hot spots."""
    import cProfile
    import io
    import pstats
    from beatfcn.model.batch import collate
    from beatfcn.train.data import Corpus, load_manifest
    from beatfcn.train.loop import CoreSampler
    corpus = Corpus(root=DATA, taps_key=taps_key, layers=taps, pca=pca)
    rows = [r for r in load_manifest(DATA) if r['split'] == 'train']
    load = (lambda sha: corpus.song(sha, in_memory=True)) if pool else corpus.song
    smp = CoreSampler(corpus.train_rows(rows), corpus.chart, load, 1, salient=corpus.salient_us, pool=pool,
                      residence=RESIDENCE)
    for _ in range(2):  # warm caches the same way a long-running worker would
        items = smp.draw(4)
    pr = cProfile.Profile()
    t0 = time.time()
    pr.enable()
    for _ in range(n_batches):
        items = []
        while len(items) < micro_batch:
            items += smp.draw(min(4, micro_batch - len(items)))
        collate([a for a, _ in items], [b for _, b in items], taps, max(taps))
    pr.disable()
    dt = time.time() - t0
    out = io.StringIO()
    pstats.Stats(pr, stream=out).sort_stats('cumulative').print_stats(45)
    return {'s_per_core': dt / (n_batches * micro_batch), 'skipped': dict(smp.skipped), 'profile': out.getvalue()[-9000:]}


@app.local_entrypoint()
def profile(n_batches: int = 12, pool: int = POOL):
    r = profile_data.remote(n_batches=n_batches, pool=pool)
    print(json.dumps({k: v for k, v in r.items() if k != 'profile'}))
    print(r['profile'])


@app.function(image=image, volumes={DATA: data}, cpu=16.0, memory=65536, timeout=1800)
def prep_bench_batches(taps_key: str = 'v4_i8_L9-16-24', taps: tuple = (9, 16, 24), pca: str = 'i8_9-16-24',
                       n_batches: int = 12, workers: int = 14, out: str = 'bench/meta_static_v4.pt'):
    """CPU-only: metadata batches (static shape, no audio) from the real worker stream, saved for the GPU
    bench. Also reports the metadata path's sustained rate (cores/s with `workers` processes)."""
    import torch
    from beatfcn.model.batch import STATIC
    from beatfcn.train.data import load_manifest
    have = _cached_songs(taps_key)
    rows = [r for r in load_manifest(DATA) if r['audio_sha256'] in have and r['split'] == 'train']
    corpus_args = {'root': DATA, 'taps_key': taps_key, 'layers': tuple(taps), 'pca': pca}
    loader = torch.utils.data.DataLoader(_stream(rows, corpus_args, 3, 32, tuple(taps), max(taps), static=STATIC), batch_size=None,
                                         num_workers=workers, prefetch_factor=2)
    it = iter(loader)
    t0 = time.time()
    first, skipped = next(it)
    t1 = time.time()
    print(json.dumps({'first_batch_s': round(t1 - t0, 1), 'skipped': skipped}), flush=True)
    batches = [first]
    for i in range(n_batches - 1):
        b, skipped = next(it)
        batches.append(b)
        print(json.dumps({'batch': i + 2, 'elapsed_s': round(time.time() - t1, 1), 'skipped': skipped}), flush=True)
    t2 = time.time()
    os.makedirs(os.path.dirname(f'{DATA}/{out}'), exist_ok=True)
    torch.save({'batches': batches, 'taps': list(taps), 'taps_key': taps_key, 'pca': pca}, f'{DATA}/{out}')
    data.commit()
    songs = sorted({k for b in batches for k in b['_audio']['keys']})
    stats = {'first_s': round(t1 - t0, 1), 'cores_per_s_after_first': round(32 * (n_batches - 1) / max(t2 - t1, 1e-6), 1),
             'songs': len(songs), 'skipped': skipped, 'bytes_per_batch': sum(v.numel() * v.element_size() for v in first.values() if torch.is_tensor(v))}
    print(json.dumps(stats))
    return stats


@app.function(image=image, gpu='H100', volumes={DATA: data}, cpu=4.0, memory=32768, timeout=1800)
def bench_graph(batches: str = 'bench/meta_static_v4.pt', tap_dim: int = 1024, steps: int = 20, variant: dict | None = None,
                paired: bool = True, pool_songs: int = 160):
    """GPU: data path (SongPool load + attach) and eager vs CUDA-graph step on static metadata batches;
    checks graphed parameters against the eager model after the same steps (same init, same batches)."""
    import copy
    import torch
    from beatfcn.cloud.gpu_util import GpuUtil
    from beatfcn.model.batch import attach_audio
    from beatfcn.model.p1 import P1Config, P1Model
    from beatfcn.train.data import Corpus
    from beatfcn.train.graph import GraphedStep
    from beatfcn.train.loop import to_device, merge, summarize
    from beatfcn.train.pool import SongPool
    z = torch.load(f'{DATA}/{batches}', weights_only=False)
    taps, metas = tuple(z['taps']), z['batches']
    corpus = Corpus(root=DATA, taps_key=z['taps_key'], layers=taps, pca=z['pca'])
    dev = torch.device('cuda')
    out = {'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__, 'n_batches': len(metas)}
    # --- data path: pool loads + attach
    pool = SongPool(corpus.load_numpy, taps, 'cuda', capacity=pool_songs)
    songs = sorted({k for b in metas for k in b['_audio']['keys']})
    t = time.time()
    pool.request(songs)
    for sha in songs:
        pool.get(sha)
    torch.cuda.synchronize()
    out['pool'] = {'songs': len(songs), 'load_wall_s': round(time.time() - t, 1), 'gb': round(pool.resident_bytes() / 1e9, 2),
                   'MB_per_s': round(pool.resident_bytes() / 1e6 / max(time.time() - t, 1e-6), 1)}
    t = time.time()
    full = []
    for b in metas:
        bb = copy.deepcopy(b)
        attach_audio(bb, pool.get, 'cuda', corpus.scale if corpus.quant else None)
        full.append(bb)
    torch.cuda.synchronize()
    out['attach_ms_per_batch'] = round(1000 * (time.time() - t) / len(metas), 1)
    print(json.dumps(out), flush=True)  # partial results survive a later failure

    class Run:
        def __init__(self, model, opt):
            self.model, self.opt = model, opt

    def make(graph, seed=0, dropout=None):
        runs = []
        for i in range(2 if paired else 1):
            torch.manual_seed(seed)
            extra = {} if dropout is None else {'dropout': dropout}
            mc = P1Config(taps=taps, top_tap=max(taps), tap_dim=tap_dim, **{'packed': False, **(variant or {}), **extra, 'selection': bool(i)})
            m = P1Model(mc).to(dev)
            lr = torch.tensor(1e-4, device=dev) if graph else 1e-4
            runs.append(Run(m, torch.optim.AdamW(m.parameters(), lr=lr, fused=True, capturable=graph)))
        return runs
    mon = GpuUtil()
    torch.manual_seed(1)
    # --- eager reference
    runs_e = make(False)
    gs_e = GraphedStep(runs_e, 1.0, dev, capture=False)
    t = time.time()
    for i in range(steps):
        gs_e.fill(full[i % len(full)], 0.2)
        gs_e.step()
    torch.cuda.synchronize()
    out['eager'] = {'s_per_step': round((time.time() - t) / steps, 3), 'util': mon.take()['util_mean']}
    print(json.dumps(out), flush=True)
    # --- graphed
    torch.manual_seed(1)
    runs_g = make(True)
    gs_g = GraphedStep(runs_g, 1.0, dev, capture=True)
    torch.cuda.synchronize()
    t = time.time()
    for i in range(gs_g.warmup + 1):  # 3 eager warm-ups + capture (+ replay) on the first batches
        gs_g.fill(full[i % len(full)], 0.2)
        gs_g.step()
    torch.cuda.synchronize()
    out['graph_setup_s'] = round(time.time() - t, 1)
    mon.take()
    torch.cuda.reset_peak_memory_stats()
    t = time.time()
    for i in range(gs_g.warmup + 1, gs_g.warmup + 1 + steps):
        gs_g.fill(full[i % len(full)], 0.2)
        gs_g.step()
    torch.cuda.synchronize()
    out['graph'] = {'s_per_step': round((time.time() - t) / steps, 3), 'util': mon.take()['util_mean'],
                    'peak_gb': round(torch.cuda.max_memory_allocated() / 1e9, 2)}
    print(json.dumps(out), flush=True)
    # --- numerics: fresh models, identical steps eager vs graphed on the same batches
    torch.manual_seed(2)
    ra, rb = make(False, seed=5, dropout=0.0), make(True, seed=5, dropout=0.0)  # deterministic comparison
    ga, gb = GraphedStep(ra, 1.0, dev, capture=False), GraphedStep(rb, 1.0, dev, capture=True)
    n_check = gb.warmup + 4
    for i in range(n_check):
        ga.fill(full[i % len(full)], 0.2)
        ga.step()
        gb.fill(full[i % len(full)], 0.2)
        gb.step()
    torch.cuda.synchronize()
    diffs = []
    for a, b in zip(ra, rb):
        pa = torch.cat([p.detach().float().flatten() for p in a.model.parameters()])
        pb = torch.cat([p.detach().float().flatten() for p in b.model.parameters()])
        diffs.append(float((pa - pb).abs().max()))
    out['numerics'] = {'steps': n_check, 'max_abs_param_diff_per_run': diffs, 'threshold': 1e-4, 'ok': all(d < 1e-4 for d in diffs)}
    mon.close()
    print(json.dumps(out))
    return out


@app.local_entrypoint()
def graph_bench(gpus: str = 'H100', steps: int = 20, paired: bool = True):
    calls = {g: bench_graph.with_options(gpu=g, timeout=1200).spawn(steps=steps, paired=paired) for g in gpus.split(',')}
    for g, c in calls.items():
        print(json.dumps(c.get(), indent=1))


@app.function(image=image, volumes={DATA: data}, cpu=2.0, memory=8192, timeout=900)
def diag_worker(taps_key: str = 'v4_i8_L9-16-24', taps: tuple = (9, 16, 24), pca: str = 'i8_9-16-24', draws: int = 8):
    """Stage timings of one worker's start-up on the volume (what a DataLoader worker does before its first batch)."""
    from beatfcn.model.batch import STATIC, STATIC_CORE, collate_meta
    from beatfcn.train.data import Corpus, load_manifest
    from beatfcn.train.loop import CoreSampler
    t = {}
    t0 = time.time()
    have = _cached_songs(taps_key)
    t['listdir_exists_s'] = round(time.time() - t0, 1)
    t0 = time.time()
    rows = [r for r in load_manifest(DATA) if r['audio_sha256'] in have and r['split'] == 'train']
    t['manifest_s'] = round(time.time() - t0, 1)
    t0 = time.time()
    corpus = Corpus(root=DATA, taps_key=taps_key, layers=tuple(taps), pca=pca)
    smp = CoreSampler(corpus.train_rows(rows), corpus.chart, corpus.song_info, 1, salient=corpus.salient_us, pool=POOL,
                      residence=RESIDENCE, limits=STATIC_CORE)
    t['corpus_sampler_s'] = round(time.time() - t0, 1)
    t0 = time.time()
    Dt, quant = corpus.tap_dim, corpus.quant
    t['tap_dim_s'] = round(time.time() - t0, 1)
    sha = rows[0]['audio_sha256']
    t0 = time.time(); corpus.song_info(sha); t['song_info_s'] = round(time.time() - t0, 2)
    t0 = time.time(); corpus.salient_us(sha); t['salient_s'] = round(time.time() - t0, 2)
    t0 = time.time(); corpus.chart(rows[0]['chart_id']); t['chart_s'] = round(time.time() - t0, 2)
    per = []
    items = []
    for i in range(draws):
        t0 = time.time()
        items += smp.draw(4)
        per.append(round(time.time() - t0, 2))
    t['draw_s'] = per
    t0 = time.time()
    bt = collate_meta([a for a, _ in items[:32]], [b for _, b in items[:32]], tuple(taps), max(taps), Dt, 1024, quant, STATIC)
    t['collate_meta_s'] = round(time.time() - t0, 2)
    t['skipped'] = dict(smp.skipped)
    t['cores'] = len(items)
    print(json.dumps(t))
    return t


@app.local_entrypoint()
def diag():
    print(json.dumps(diag_worker.remote(), indent=1))


@app.local_entrypoint()
def prep_bench(n_batches: int = 12, workers: int = 14):
    print(prep_bench_batches.remote(n_batches=n_batches, workers=workers))


@app.local_entrypoint()
def bench_gpu(gpus: str = 'L40S,H100,L4,A10', timeout: int = 600):
    calls = {g: bench_train_gpu.with_options(gpu=g, timeout=timeout).spawn() for g in gpus.split(',')}
    for g, c in calls.items():
        print(json.dumps(c.get(), indent=1))
