"""Corpus ingestion on Modal (CPU only). Rules: design/DATA_RULES.md. Volume layout: DATA_RULES section 6.

Run from the repo root:
  .venv/bin/modal run --detach -m beatfcn.cloud.ingest_app --osu-shards 0,1 --phira-limit 50 --out <json>   # dry run
  .venv/bin/modal run --detach -m beatfcn.cloud.ingest_app --out reports/ingest_stats.json               # full
Every stage is idempotent: raw downloads, mel10 and per-item rows are skipped when present (rows carry
ROW_VERSION); finalize/stamp are cheap and always recomputed.
"""
from __future__ import annotations
import json
import re
import os
import time
import modal

ROOT = '/data'
HF_REPO = 'project-riz/osu-beatmaps'
HF_REV = 'b82117a5de13b7df5290482aed7cffc49cb76e6d'
N_OSU_SHARDS = 205
PHIRA_DL_CONTAINERS, PHIRA_DL_THREADS = 2, 4   # at most 8 requests in flight to phira.5wyxi.com
UA = 'BeatFCN-research-ingest/1.0 (one-time corpus snapshot; bounded concurrency)'
# (cpu physical cores, memory GiB) per stage, for the cost estimate
RES = {'osu_shard': (1.0, 6), 'phira_download': (0.5, 1), 'phira_process': (1.0, 6), 'finalize': (2.0, 8),
       'stamp': (1.0, 2), 'volume_usage': (0.5, 1), 'run_all': (0.25, 1)}
PRICE_CPU, PRICE_MEM = 0.0000131, 0.00000222

app = modal.App('beatfcn-ingest')
vol = modal.Volume.from_name('beatfcn-data', create_if_missing=True, version=2)
image = (modal.Image.debian_slim(python_version='3.12')
         .apt_install('ffmpeg')
         .pip_install('torch==2.6.0', index_url='https://download.pytorch.org/whl/cpu')
         .pip_install('numpy==2.2.6', 'pyyaml', 'jsonschema', 'requests')
         .env({'OMP_NUM_THREADS': '2', 'PYTHONUNBUFFERED': '1'})
         .add_local_python_source('beatfcn'))
FN = dict(image=image, volumes={ROOT: vol}, scaledown_window=10)


def _p(*a):
    return os.path.join(ROOT, *a)


def _row_file_ok(path, version):
    try:
        with open(path) as f:
            first = f.readline()
        return bool(first) and json.loads(first).get('row_version') == version
    except (OSError, ValueError):
        return False


def _done(stage: str, res: dict, n: int | None = None) -> dict:
    """Persist one call's timing to work/acct so cost accounting survives orchestrator restarts."""
    import uuid
    rec = {'stage': stage, 't0': res['t0'], 't1': res['t1'], 'cpu': RES[stage][0], 'mem_gib': RES[stage][1],
           'skipped': bool(res.get('skipped')), 'n': n}
    d = _p('work', 'acct', stage)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, f'{int(res["t0"] * 1000)}_{uuid.uuid4().hex[:8]}.json'), 'w') as f:
        json.dump(rec, f)
    vol.commit()
    return res


def _recorded_cost() -> dict:
    import glob
    out = {}
    for f in glob.glob(_p('work', 'acct', '*', '*.json')):
        r = json.load(open(f))
        s = out.setdefault(r['stage'], {'calls': 0, 'container_s': 0.0, 'cost_usd': 0.0})
        dt = r['t1'] - r['t0']
        s['calls'] += 1
        s['container_s'] += dt
        s['cost_usd'] += dt * (r['cpu'] * PRICE_CPU + r['mem_gib'] * PRICE_MEM)
    return {k: {'calls': v['calls'], 'container_hours': round(v['container_s'] / 3600, 3),
                'cost_usd': round(v['cost_usd'], 3)} for k, v in sorted(out.items())}

def _dl_records() -> dict:
    import glob
    recs = [json.load(open(f)) for f in glob.glob(_p('work', 'phira_dl', '*.json'))]
    if not recs:
        return {'n': 0}
    secs = sorted(r['secs'] for r in recs)
    return {'n': len(recs), 'bytes': sum(r['bytes'] for r in recs), 'retries': sum(r['retries'] for r in recs),
            'secs_p50': round(secs[len(secs) // 2], 2), 'secs_p95': round(secs[int(len(secs) * .95)], 2)}

def _shard_oid(i: int) -> str:
    import requests
    for attempt in range(5):
        try:
            r = requests.post(f'https://huggingface.co/api/datasets/{HF_REPO}/paths-info/{HF_REV}',
                             data={'paths': f'original/data-{i:06d}.tar'}, timeout=60, headers={'User-Agent': UA})
            r.raise_for_status()
            return r.json()[0]['lfs']['oid']
        except Exception as e:  # noqa: BLE001
            print(f'paths-info shard {i} attempt {attempt}: {e!r}', flush=True)
            time.sleep(10 * (attempt + 1))
    raise RuntimeError(f'cannot fetch LFS oid of shard {i}')


def _download_verified(url: str, path: str, sha_hex: str) -> float:
    """Download to local disk with HTTP Range resume; verify the LFS SHA256. Returns seconds."""
    import hashlib
    import requests
    t0, h = time.time(), hashlib.sha256()
    open(path, 'wb').close()
    for attempt in range(10):
        have = os.path.getsize(path)
        try:
            hdr = {'User-Agent': UA, **({'Range': f'bytes={have}-'} if have else {})}
            with requests.get(url, stream=True, timeout=(30, 120), headers=hdr) as r:
                if have and r.status_code != 206:
                    raise RuntimeError(f'no range support (HTTP {r.status_code})')
                r.raise_for_status()
                with open(path, 'ab') as f:
                    for chunk in r.iter_content(8 << 20):
                        f.write(chunk)
                        h.update(chunk)
            if h.hexdigest() == sha_hex:
                return round(time.time() - t0, 1)
            raise RuntimeError('sha256 mismatch after complete download')
        except Exception as e:  # noqa: BLE001
            print(f'download {url[-20:]} attempt {attempt}: {e!r}', flush=True)
            if 'mismatch' in str(e) or 'range' in str(e):
                open(path, 'wb').close()
                h = hashlib.sha256()
            time.sleep(min(10 * (attempt + 1), 60))
    raise RuntimeError(f'download failed: {url}')


# ---------------------------------------------------------------- osu
@app.function(cpu=RES['osu_shard'][0], memory=RES['osu_shard'][1] * 1024, timeout=4 * 3600, max_containers=64,
              retries=1, **FN)
def osu_shard(i: int) -> dict:
    import tarfile
    import tempfile
    import requests
    import torch
    from beatfcn.ingest import pipeline as P
    torch.set_num_threads(2)
    t0 = time.time()
    vol.reload()
    out = _p('work', 'rows', 'osu', f'{i:06d}.jsonl')
    if _row_file_ok(out, P.ROW_VERSION):
        return _done('osu_shard', {'shard': i, 'skipped': True, 't0': t0, 't1': time.time()})
    url = f'https://huggingface.co/datasets/{HF_REPO}/resolve/{HF_REV}/original/data-{i:06d}.tar'
    st = {'shard': i, 'samples': 0, 'samples_wanted': 0, 'beatmaps_total': 0, 'beatmaps_mania': 0,
          'beatmaps_wanted': 0, 'bytes': 0, 'audio_errors': 0, 'mel_new': 0}
    rows = []

    def handle(key, parts, tmpdir):
        st['samples'] += 1
        meta = json.loads(parts['json']) if 'json' in parts else None
        if meta is None:
            return
        bms = meta.get('beatmaps') or []
        st['beatmaps_total'] += len(bms)
        st['beatmaps_mania'] += sum(str(b.get('mode')) in ('3', '3.0') for b in bms)
        wanted = [b for b in bms if P.osu_wanted(b)]
        if not wanted:
            return
        st['samples_wanted'] += 1
        st['beatmaps_wanted'] += len(wanted)
        meta = dict(meta, beatmaps=wanted)
        audio = parts.get('mp3')
        try:
            if audio is None:
                raise ValueError('sample has no audio member')
            ainfo = P.ensure_audio(ROOT, audio, os.path.join('raw', 'osu', 'audio'), tmpdir)
            st['mel_new'] += not ainfo['mel_cached']
        except Exception as e:  # noqa: BLE001 - fail closed per sample
            st['audio_errors'] += 1
            for bm in wanted:
                bid = str(bm.get('beatmap_id'))
                P._atomic_write(_p('raw', 'osu', 'charts', f'{bid}.osu'), (bm.get('content') or '').encode())
                rows.append(P.failed_row({'chart_id': f'osu:{bid}', 'source': 'osu', 'native_id': bid,
                                          'title': bm.get('title') or '', 'title_alt': bm.get('title') or '',
                                          'set_id': str(bm.get('beatmapset_id')), 'shard': i},
                                         'audio_decode_error', f'{type(e).__name__}: {e}',
                                         {'audio_sha256': P.sha256(audio) if audio else None}))
            return
        rows.extend(P.osu_beatmap_rows(ROOT, meta, ainfo, i))

    with tempfile.TemporaryDirectory() as tmpdir:
        tar_path = os.path.join(tmpdir, 'shard.tar')
        st['download_s'] = _download_verified(url, tar_path, _shard_oid(i))
        cur, parts = None, {}
        with tarfile.open(tar_path, mode='r|') as tf:   # sequential read from local disk, one sample in RAM
            for m in tf:
                if not m.isfile():
                    continue
                name = m.name.rsplit('/', 1)[-1]
                key, _, ext = name.partition('.')
                data = tf.extractfile(m).read()
                st['bytes'] += len(data)
                if key != cur and cur is not None:
                    handle(cur, parts, tmpdir)
                    parts = {}
                cur = key
                parts[ext] = data
            if cur is not None:
                handle(cur, parts, tmpdir)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    body = ''.join(json.dumps(r, ensure_ascii=False, separators=(',', ':')) + '\n' for r in rows)
    if not rows:
        body = json.dumps({'row_version': P.ROW_VERSION, 'empty_shard': i}) + '\n'
    P._atomic_write(out, body.encode())
    P._atomic_write(_p('work', 'osu_shard_stats', f'{i:06d}.json'), json.dumps(st).encode())
    vol.commit()
    return _done('osu_shard', {**st, 'rows': len(rows), 't0': t0, 't1': time.time()})


# ---------------------------------------------------------------- phira
@app.function(cpu=RES['phira_download'][0], memory=RES['phira_download'][1] * 1024, timeout=12 * 3600,
              max_containers=PHIRA_DL_CONTAINERS, **FN)
def phira_download(batch: list) -> dict:
    import random
    import threading
    import zipfile
    from concurrent.futures import ThreadPoolExecutor
    import requests
    t0 = time.time()
    vol.reload()
    sess = requests.Session()
    sess.headers['User-Agent'] = UA
    zdir = _p('raw', 'phira', 'zips')
    os.makedirs(zdir, exist_ok=True)
    os.makedirs(_p('work', 'phira_dl'), exist_ok=True)

    def one(c):
        pid, dst = str(c['id']), os.path.join(zdir, f'{c["id"]}.zip')
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            return {'id': pid, 'status': 'cached', 'bytes': os.path.getsize(dst), 'retries': 0, 'secs': 0}
        s0, retries, last = time.time(), 0, ''
        for attempt in range(6):
            try:
                with sess.get(c['file'], stream=True, timeout=(20, 120)) as r:
                    if r.status_code in (403, 404, 410):
                        return {'id': pid, 'status': f'http_{r.status_code}', 'bytes': 0, 'retries': retries,
                                'secs': time.time() - s0}
                    if r.status_code in (429, 500, 502, 503, 504):
                        retries += 1
                        last = f'http_{r.status_code}'
                        ra = r.headers.get('Retry-After')
                        time.sleep(min(float(ra), 300) if ra and ra.isdigit() else min(5 * 2 ** attempt, 160) + random.random() * 3)
                        continue
                    r.raise_for_status()
                    part = f'{dst}.{os.getpid()}.{threading.get_ident()}.part'
                    n = 0
                    with open(part, 'wb') as f:
                        for chunk in r.iter_content(1 << 20):
                            f.write(chunk)
                            n += len(chunk)
                if not zipfile.is_zipfile(part):
                    os.unlink(part)
                    return {'id': pid, 'status': 'not_zip', 'bytes': n, 'retries': retries, 'secs': time.time() - s0}
                os.replace(part, dst)
                rec = {'id': pid, 'status': 'downloaded', 'bytes': n, 'retries': retries, 'secs': time.time() - s0}
                with open(_p('work', 'phira_dl', f'{pid}.json'), 'w') as f:
                    json.dump(rec, f)
                return rec
            except (requests.RequestException, OSError) as e:
                retries += 1
                last = type(e).__name__
                time.sleep(min(5 * 2 ** attempt, 160) + random.random() * 3)
        return {'id': pid, 'status': f'failed_{last}', 'bytes': 0, 'retries': retries, 'secs': time.time() - s0}

    with ThreadPoolExecutor(PHIRA_DL_THREADS) as ex:
        res = list(ex.map(one, batch))
    vol.commit()
    return _done('phira_download', {'items': res, 't0': t0, 't1': time.time()}, n=len(res))


@app.function(cpu=RES['phira_process'][0], memory=RES['phira_process'][1] * 1024, timeout=4 * 3600,
              max_containers=64, retries=1, **FN)
def phira_process(batch: list) -> dict:
    import tempfile
    from collections import Counter
    import torch
    from beatfcn.ingest import pipeline as P
    torch.set_num_threads(2)
    t0 = time.time()
    vol.reload()
    done = Counter()
    with tempfile.TemporaryDirectory() as tmpdir:
        for c in batch:
            pid = str(c['id'])
            out = _p('work', 'rows', 'phira', f'{pid}.json')
            if _row_file_ok(out, P.ROW_VERSION):
                done['skipped'] += 1
                continue
            zp = _p('raw', 'phira', 'zips', f'{pid}.zip')
            if not os.path.exists(zp):
                continue   # download failure is recorded by the download stage
            try:
                row = P.phira_zip_row(ROOT, zp, c, tmpdir)
            except Exception as e:  # noqa: BLE001
                row = P.failed_row({'chart_id': f'phira:{pid}', 'source': 'phira', 'native_id': pid,
                                    'title': c.get('name') or ''}, 'process_error', f'{type(e).__name__}: {e}')
            P._atomic_write(out, (json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n').encode())
            done[row['status']] += 1
    vol.commit()
    return _done('phira_process', {'counts': dict(done), 't0': t0, 't1': time.time()})


# ---------------------------------------------------------------- pass 2
@app.function(cpu=RES['finalize'][0], memory=RES['finalize'][1] * 1024, timeout=3 * 3600, **FN)
def finalize(phira_candidates: list, download_status: dict) -> dict:
    import glob
    from beatfcn.ingest import pipeline as P
    t0 = time.time()
    vol.reload()
    rows = []
    for f in sorted(glob.glob(_p('work', 'rows', 'osu', '*.jsonl'))):
        for line in open(f):
            r = json.loads(line)
            if 'chart_id' in r:
                rows.append(r)
    seen = {r['chart_id'] for r in rows}
    dup_osu = len(rows) - len(seen)
    if dup_osu:   # same beatmap in two samples: keep the first occurrence
        uniq, ids = [], set()
        for r in rows:
            if r['chart_id'] not in ids:
                ids.add(r['chart_id'])
                uniq.append(r)
        rows = uniq
    for c in phira_candidates:
        pid = str(c['id'])
        f = _p('work', 'rows', 'phira', f'{pid}.json')
        if os.path.exists(f):
            rows.append(json.loads(open(f).read()))
        else:
            st = download_status.get(pid, 'not_attempted')
            rows.append(P.failed_row({'chart_id': f'phira:{pid}', 'source': 'phira', 'native_id': pid,
                                      'title': c.get('name') or ''}, f'download_{st}'))
    rows, ctab = P.finalize_rows(rows)
    os.makedirs(_p('manifest'), exist_ok=True)
    P._atomic_write(_p('manifest', 'charts.jsonl'), ''.join(
        json.dumps(P.manifest_row(r), ensure_ascii=False, separators=(',', ':')) + '\n' for r in rows).encode())
    shard_stats = [json.load(open(f)) for f in sorted(glob.glob(_p('work', 'osu_shard_stats', '*.json')))]
    osu_scan = {k: sum(s[k] for s in shard_stats) for k in ('samples', 'samples_wanted', 'beatmaps_total',
                                                             'beatmaps_mania', 'beatmaps_wanted', 'bytes', 'audio_errors')}
    osu_scan['shards_done'] = len(shard_stats)
    osu_scan['duplicate_beatmap_rows_dropped'] = dup_osu
    ff = __import__('subprocess').run(['ffmpeg', '-version'], capture_output=True, text=True).stdout.split('\n')[0]
    stats = P.corpus_stats(rows, ctab, {'osu_scan': osu_scan, 'ffmpeg': ff, 'hf_revision': HF_REV})
    stamps = {r['chart_id']: {k: r.get(k) for k in ('split', 'song_group', 'offset_decision', 'offset_disagreement',
                                                    'timing_unreliable', 'quarantined', 'quarantine_reasons', 'kept')}
              for r in rows if r['status'] == 'parsed'}
    vol.commit()
    return _done('finalize', {'stats': stats, 'stamps': stamps, 't0': t0, 't1': time.time()}, n=len(rows))


@app.function(cpu=RES['stamp'][0], memory=RES['stamp'][1] * 1024, timeout=3600, max_containers=64, **FN)
def stamp(items: list) -> dict:
    from beatfcn.ingest import pipeline as P
    t0 = time.time()
    vol.reload()
    n = miss = 0
    for cid, st in items:
        src, nid = cid.split(':', 1)
        p = _p('canon', src, f'{nid}.json')
        try:
            d = json.load(open(p))
        except OSError:
            miss += 1
            continue
        if d.get('ingest') == st:
            continue
        d['ingest'] = st
        P._atomic_write(p, json.dumps(d, ensure_ascii=False, separators=(',', ':')).encode())
        n += 1
    vol.commit()
    return _done('stamp', {'stamped': n, 'missing': miss, 't0': t0, 't1': time.time()})


@app.function(cpu=RES['volume_usage'][0], memory=RES['volume_usage'][1] * 1024, timeout=3600, **FN)
def volume_usage() -> dict:
    t0 = time.time()
    vol.reload()
    removed = 0
    for dp, _, fs in os.walk(ROOT):   # runs only after every writer stage has finished
        for f in fs:
            if f.endswith('.part'):
                os.unlink(os.path.join(dp, f))
                removed += 1
    vol.commit()
    use = {}
    for dp, _, fs in os.walk(ROOT):
        rel = os.path.relpath(dp, ROOT).split(os.sep)
        key = '/'.join(rel[:3] if rel[0] == 'raw' else rel[:2] if rel[0] in ('canon', 'work') else rel[:1])
        b = use.setdefault(key, [0, 0])
        for f in fs:
            try:
                b[0] += os.path.getsize(os.path.join(dp, f))
                b[1] += 1
            except OSError:
                pass
    return _done('volume_usage', {'usage': {k: {'bytes': v[0], 'gib': round(v[0] / 2 ** 30, 3), 'files': v[1]}
                                            for k, v in sorted(use.items()) if v[1]},
                                  'stale_part_files_removed': removed, 't0': t0, 't1': time.time()})


# ---------------------------------------------------------------- orchestration
def _acct(name, results):
    secs = sum(r['t1'] - r['t0'] for r in results if isinstance(r, dict) and 't0' in r)
    cpu, mem = RES[name]
    return {'calls': len(results), 'container_hours': round(secs / 3600, 3),
            'cost_usd': round(secs * (cpu * PRICE_CPU + mem * PRICE_MEM), 3), 'cpu': cpu, 'mem_gib': mem}


@app.function(cpu=RES['run_all'][0], memory=RES['run_all'][1] * 1024, timeout=24 * 3600, nonpreemptible=True, **FN)
async def run_all(cands: list, shards: list) -> dict:
    import asyncio
    from collections import Counter
    T0 = time.time()
    stages, errors = {}, []
    os.makedirs(_p('manifest'), exist_ok=True)
    with open(_p('manifest', 'phira_candidates.jsonl'), 'w') as f:
        for c in cands:
            f.write(json.dumps(c, ensure_ascii=False) + '\n')
    await vol.commit.aio()

    async def osu_stage():
        t = time.time()
        res = []
        async for r in osu_shard.map.aio(shards, return_exceptions=True):
            if isinstance(r, BaseException):
                errors.append(f'osu_shard: {r!r}'[:300])
            else:
                res.append(r)
                print(f'[osu] shard {r["shard"]} done ({len(res)}/{len(shards)})', flush=True)
        stages['osu_shard'] = {**_acct('osu_shard', res), 'wall_s': round(time.time() - t),
                               'failed_shards': len(shards) - len(res),
                               'skipped': sum(bool(r.get('skipped')) for r in res)}

    dl_status = {}

    async def phira_stage():
        t = time.time()
        batches = [cands[k:k + 50] for k in range(0, len(cands), 50)]
        res = []
        async for r in phira_download.map.aio(batches, return_exceptions=True):
            if isinstance(r, BaseException):
                errors.append(f'phira_download: {r!r}'[:300])
                continue
            res.append(r)
            for it in r['items']:
                dl_status[it['id']] = it['status']
            print(f'[phira-dl] {len(dl_status)}/{len(cands)} {dict(Counter(dl_status.values()))}', flush=True)
        items = [it for r in res for it in r['items']]
        got = [it for it in items if it['status'] == 'downloaded']
        stages['phira_download'] = {**_acct('phira_download', res), 'wall_s': round(time.time() - t),
                                    'status_counts': dict(Counter(it['status'] for it in items)),
                                    'bytes_downloaded': sum(it['bytes'] for it in got),
                                    'bytes_total': sum(it['bytes'] for it in items),
                                    'retries': sum(it['retries'] for it in items),
                                    'mean_secs_per_zip': round(sum(it['secs'] for it in got) / max(len(got), 1), 2),
                                    'recorded_downloads_all_invocations': _dl_records()}
        t = time.time()
        ok = [c for c in cands if dl_status.get(str(c['id'])) in ('downloaded', 'cached')]
        res = []
        async for r in phira_process.map.aio([ok[k:k + 25] for k in range(0, len(ok), 25)], return_exceptions=True):
            if isinstance(r, BaseException):
                errors.append(f'phira_process: {r!r}'[:300])
            else:
                res.append(r)
        stages['phira_process'] = {**_acct('phira_process', res), 'wall_s': round(time.time() - t),
                                   'counts': dict(sum((Counter(r['counts']) for r in res), Counter()))}
        print(f'[phira] processed {stages["phira_process"]["counts"]}', flush=True)

    await asyncio.gather(osu_stage(), phira_stage())
    t = time.time()
    fin = await finalize.remote.aio(cands, dl_status)
    stages['finalize'] = {**_acct('finalize', [fin]), 'wall_s': round(time.time() - t)}
    t = time.time()
    items = list(fin['stamps'].items())
    res = [r async for r in stamp.map.aio([items[k:k + 300] for k in range(0, len(items), 300)])]
    stages['stamp'] = {**_acct('stamp', res), 'wall_s': round(time.time() - t),
                       'stamped': sum(r['stamped'] for r in res), 'missing': sum(r['missing'] for r in res)}
    t = time.time()
    vu = await volume_usage.remote.aio()
    stages['volume_usage'] = {**_acct('volume_usage', [vu]), 'wall_s': round(time.time() - t)}
    stages['run_all'] = {**_acct('run_all', [{'t0': T0, 't1': time.time()}]), 'wall_s': round(time.time() - T0)}
    stats = fin['stats']
    stats['phira_candidates'] = len(cands)
    stats['phira_download'] = stages['phira_download']
    stats['volume_usage'] = vu['usage']
    stats['stages'] = stages
    rec = _recorded_cost()
    rec['run_all'] = {'calls': 1, **{k: stages['run_all'][k] for k in ('container_hours', 'cost_usd')}}
    stats['recorded_cost_all_invocations'] = rec
    stats['total_container_hours'] = round(sum(s['container_hours'] for s in stages.values()), 3)
    stats['total_cost_usd_estimate'] = round(sum(s['cost_usd'] for s in stages.values()), 2)
    stats['cost_note'] = ('sum over inputs of (cpu*$0.0000131/core-s + GiB*$0.00000222/s) at the reserved size; '
                          'excludes container start-up and <=10 s idle scale-down per container')
    stats['errors'] = errors
    stats['wall_clock_s'] = round(time.time() - T0)
    with open(_p('manifest', 'stats.json'), 'w') as f:
        json.dump(stats, f, ensure_ascii=False, indent=1)
    await vol.commit.aio()
    return stats


@app.local_entrypoint()
def main(osu_shards: str = '', phira_limit: int = 0, out: str = 'reports/ingest_stats.json'):
    cands = []
    with open('phira_all_charts.jsonl') as f:
        for line in f:
            r = json.loads(line)
            tag = re.split(r'[\s.]', str(r.get('level') or '?').strip().upper())[0]
            if (r.get('rating') or 0) >= 0.90 and (r.get('ratingCount') or 0) >= 100 and tag in ('EZ', 'HD', 'IN', 'AT'):  # DATA_RULES v4
                cands.append({k: r.get(k) for k in ('id', 'name', 'level', 'charter', 'composer', 'file', 'rating',
                                                    'ratingCount', 'chartUpdated')})
    if phira_limit:
        cands = cands[:phira_limit]
    shards = [int(x) for x in osu_shards.split(',')] if osu_shards else list(range(N_OSU_SHARDS))
    print(f'phira candidates {len(cands)}, osu shards {len(shards)}')
    stats = run_all.remote(cands, shards)
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    with open(out, 'w') as f:
        json.dump(stats, f, ensure_ascii=False, indent=1)
    print(json.dumps({k: stats[k] for k in ('total_container_hours', 'total_cost_usd_estimate', 'wall_clock_s',
                                            'errors')}, indent=1))
