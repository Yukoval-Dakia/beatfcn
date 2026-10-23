"""Corpus ingestion steps shared by the Modal app (beatfcn/cloud/ingest_app.py) and tests.

Pass 1 (per item, parallel): store raw bytes, decode audio once per SHA256 -> mel10, parse with the P0
parsers, validate, offset + timing diagnostics, write canon/{source}/{id}.json and a draft row
(work/rows/...). Pass 2 (finalize, one container): song groups + split, per-(source, codec) train
constant c, v1 offset decision, disagreement flags, manifest/charts.jsonl and stats.
Every item fails closed with a reason code; nothing here repairs a chart.
"""
from __future__ import annotations
from collections import Counter, defaultdict
import hashlib
import json
import os
import re
import subprocess
import tempfile
import numpy as np

from beatfcn.events.canonical import UnsupportedFormat, build_model_view
from beatfcn.events import validate as V
from beatfcn.events.group_split import group_rows, split_of, work_key
from . import offset as OFF
from .timing import timing_reliability

ROW_VERSION = 3
SEED = 20261022
SR = 24000
CODEC_EXT = {'mp3': 'mp3', 'vorbis': 'ogg', 'opus': 'opus', 'flac': 'flac', 'aac': 'm4a', 'alac': 'm4a',
             'wavpack': 'wv'}


# ---------------------------------------------------------------- audio
def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def probe_codec(path: str) -> str:
    out = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'a:0', '-show_entries', 'stream=codec_name',
                          '-of', 'csv=p=0', path], capture_output=True, text=True, timeout=120)
    c = out.stdout.strip().split('\n')[0].split(',')[0].strip()
    return c or 'unknown'


def codec_ext(codec: str) -> str:
    if codec.startswith('pcm_'):
        return 'wav'
    return CODEC_EXT.get(codec, codec if re.fullmatch(r'[a-z0-9]{1,8}', codec) else 'bin')


def decode(path: str) -> np.ndarray:
    raw = subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-i', path, '-vn', '-f', 'f32le', '-ac', '1',
                          '-ar', str(SR), 'pipe:1'], capture_output=True, check=True, timeout=1800).stdout
    return np.frombuffer(raw, dtype='<f4').copy()


def mel10(wave: np.ndarray) -> np.ndarray:
    from beatfcn.audio.preprocess import waveform_mel, CELL_SAMPLES
    w = wave[: len(wave) // CELL_SAMPLES * CELL_SAMPLES]
    return waveform_mel(w)[0, 0].numpy().astype(np.float16)


def ensure_audio(root: str, audio: bytes, raw_dir: str | None, tmpdir: str) -> dict:
    """Probe codec, optionally store raw audio once, compute mel10 once. Returns audio info + env."""
    sha = sha256(audio)
    tmp = os.path.join(tmpdir, f'{sha}.audio')
    with open(tmp, 'wb') as f:
        f.write(audio)
    try:
        codec = probe_codec(tmp)
        ext = codec_ext(codec)
        info = {'audio_sha256': sha, 'codec': codec, 'audio_ext': ext, 'audio_bytes': len(audio)}
        if raw_dir:
            dst = os.path.join(root, raw_dir, f'{sha}.{ext}')
            if not os.path.exists(dst):
                _atomic_write(dst, audio)
        mp = os.path.join(root, 'mel10', f'{sha}.npy')
        meta = mp[:-4] + '.json'
        if os.path.exists(mp) and os.path.exists(meta):
            mel = np.load(mp)
            info['audio_sec'] = json.load(open(meta))['audio_sec']
            info['mel_cached'] = True
        else:
            wave = decode(tmp)
            info['audio_sec'] = round(len(wave) / SR, 4)
            mel = mel10(wave)
            del wave
            os.makedirs(os.path.dirname(mp), exist_ok=True)
            part = mp + f'.{os.getpid()}.part'
            with open(part, 'wb') as f:
                np.save(f, mel)
            os.replace(part, mp)
            _atomic_write(meta, json.dumps({'audio_sec': info['audio_sec'], 'codec': codec, 'frames': int(len(mel)),
                                            'hop_sec': 0.01, 'dtype': 'float16'}).encode())
            info['mel_cached'] = False
        info['env'] = OFF.flux_envelope(mel)
        return info
    finally:
        os.unlink(tmp)


def _atomic_write(path: str, data: bytes):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    part = f'{path}.{os.getpid()}.part'
    with open(part, 'wb') as f:
        f.write(data)
    os.replace(part, path)


# ---------------------------------------------------------------- charts
def chart_validation(chart, audio_us: int) -> list:
    """Chart-level checks reused from P0: model view (1 ms quantization, collisions, types) and the
    request event checks (README 2) against the decoded audio length; osu sidecar byte identity."""
    view = build_model_view(chart.mode, (), chart.events)
    out = [i['code'] for i in V.model_view_issues(view, ())]
    for e in chart.events:
        out += [i['code'] for i in V._check_event(e.event_id, e.to_json(), chart.mode, audio_us)]
    if chart.source_format == 'osu_v14':
        from beatfcn.events.osu_mania import export_osu
        if export_osu(chart, []) != b''.join(chart.sidecar['lines']):
            out.append('sidecar_not_byte_identical')
    return out


def fstr(x) -> str:
    """Exact 'p/q' for ordinary rationals; float repr when the rational is huge (osu beats derived from
    decimal beatLengths can exceed 4300 digits; they are recomputable exactly from the raw .osu)."""
    from fractions import Fraction
    x = Fraction(x)
    if x.denominator.bit_length() <= 64 and x.numerator.bit_length() <= 96:
        return str(x)
    return repr(float(x))


def chart_json(chart, base: dict, diag: dict) -> dict:
    def ev(e, fake=False):
        d = e.to_json()
        d['fake'] = fake
        return d
    tm = chart.timing
    return {**base, 'schema': 'beatfcn.canon.v1', 'mode': chart.mode, 'source_format': chart.source_format,
            'format_version': chart.format_version, 'source_sha256': chart.source_sha256,
            'offset_sec': fstr(chart.offset_sec),
            'timing': None if tm is None else {'offset_sec': fstr(tm.offset_sec), 'segments': [
                {'start_beat': fstr(s.start_beat), 'bpm': fstr(s.bpm), 'start_sec': fstr(s.start_sec)} for s in tm.segments]},
            'native_beats': {k: [fstr(a), fstr(b)] for k, (a, b) in chart.native_beats.items()},
            'events': [ev(e) for e in chart.events], 'fakes': [ev(e, True) for e in chart.fakes],
            'issues': [i.to_json() for i in chart.issues], 'diagnostics': diag}


def process_chart(root: str, chart, base: dict, ainfo: dict) -> dict:
    """Validate + diagnose a parsed chart; write canon JSON; return the draft row."""
    audio_us = int(round(ainfo['audio_sec'] * 1e6))
    p0q = sorted({i.code for i in chart.issues if i.severity == 'quarantine'})
    warn = sorted({i.code for i in chart.issues if i.severity != 'quarantine'})
    valq = sorted(set(chart_validation(chart, audio_us)) - set(p0q))
    off = OFF.estimate_offset([e.onset_us / 1e6 for e in chart.events], env=ainfo['env'])
    tim = timing_reliability(chart)
    diag = {'offset': off, 'timing': tim}
    cj = chart_json(chart, {**base, **_audio_fields(ainfo)}, diag)
    _atomic_write(os.path.join(root, 'canon', base['source'], f'{base["native_id"]}.json'),
                  json.dumps(cj, ensure_ascii=False, separators=(',', ':')).encode())
    return {**base, **_audio_fields(ainfo), 'row_version': ROW_VERSION, 'status': 'parsed', 'mode': chart.mode,
            'source_format': chart.source_format, 'format_version': chart.format_version,
            'n_events': len(chart.events), 'n_fakes': len(chart.fakes),
            'types': dict(Counter(e.type for e in chart.events)), 'p0_quarantine': p0q,
            'validation_quarantine': valq, 'warnings': warn, 'offset': off, 'timing': tim}


def _audio_fields(a):
    return {k: a.get(k) for k in ('audio_sha256', 'codec', 'audio_ext', 'audio_sec')}


def failed_row(base: dict, code: str, detail: str = '', ainfo: dict | None = None) -> dict:
    return {**base, **(_audio_fields(ainfo) if ainfo else {}), 'row_version': ROW_VERSION,
            'status': 'unsupported_format' if code == 'unsupported_format' else 'error',
            'status_code': code, 'status_detail': str(detail)[:300]}


# ---------------------------------------------------------------- osu
_OSU_VER = re.compile(rb'osu file format v(\d+)')


def osu_title_unicode(text: str) -> str:
    m = re.search(r'^\s*TitleUnicode\s*:(.*)$', text, re.M)
    return m.group(1).strip() if m else ''


def osu_beatmap_rows(root: str, meta: dict, ainfo: dict, shard: int) -> list:
    from beatfcn.events.osu_mania import parse_osu
    rows = []
    for bm in meta['beatmaps']:
        bid = str(bm.get('beatmap_id'))
        text = bm.get('content') or ''
        data = text.encode('utf-8')
        cpath = os.path.join(root, 'raw', 'osu', 'charts', f'{bid}.osu')
        if not os.path.exists(cpath):
            _atomic_write(cpath, data)
        base = {'chart_id': f'osu:{bid}', 'source': 'osu', 'native_id': bid,
                'title': osu_title_unicode(text) or bm.get('title') or '', 'title_alt': bm.get('title') or '',
                'artist': bm.get('artist') or '', 'charter': bm.get('creator') or '',
                'difficulty_label': bm.get('version') or '', 'set_id': str(bm.get('beatmapset_id')),
                'approved': bm.get('approved'), 'shard': shard, 'dataset_audio_hash': meta.get('audio_hash')}
        try:
            chart = parse_osu(data, source_name=f'{bid}.osu')
        except UnsupportedFormat as e:
            row = failed_row(base, 'unsupported_format', e, ainfo)
            m = _OSU_VER.search(data[:200])
            row['format_version'] = f'v{m.group(1).decode()}' if m else None
            if m and int(m.group(1)) < 14:
                row['legacy_probe'] = legacy_probe(data, ainfo)
            rows.append(row)
            continue
        except Exception as e:  # noqa: BLE001 - fail closed per chart
            rows.append(failed_row(base, 'parse_error', f'{type(e).__name__}: {e}', ainfo))
            continue
        try:
            rows.append(process_chart(root, chart, base, ainfo))
        except Exception as e:  # noqa: BLE001
            rows.append(failed_row(base, 'process_error', f'{type(e).__name__}: {e}', ainfo))
    return rows


def legacy_probe(data: bytes, ainfo: dict) -> dict | None:
    """Evidence only (never a label): parse a pre-v14 file as if v14 and measure its offset."""
    from beatfcn.events.osu_mania import parse_osu
    try:
        ch = parse_osu(_OSU_VER.sub(b'osu file format v14', data, count=1))
    except Exception as e:  # noqa: BLE001
        return {'status': 'unparseable_as_v14', 'detail': str(e)[:120]}
    off = OFF.estimate_offset([e.onset_us / 1e6 for e in ch.events], env=ainfo['env'])
    off.pop('agree_counts', None)
    return {**off, 'p0_quarantine': sorted({i.code for i in ch.issues if i.severity == 'quarantine'})}


def osu_wanted(bm: dict) -> bool:
    try:
        return int(float(bm.get('mode'))) == 3 and float(bm.get('diff_size')) == 4.0
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------- phira
def phira_zip_row(root: str, zpath: str, cand: dict, tmpdir: str) -> dict:
    import zipfile
    import yaml
    from beatfcn.events.phira import load_package
    pid = str(cand['id'])
    base = {'chart_id': f'phira:{pid}', 'source': 'phira', 'native_id': pid, 'title': cand.get('name') or '',
            'title_alt': '', 'artist': cand.get('composer') or '', 'charter': cand.get('charter') or '',
            'difficulty_label': cand.get('level') or '', 'rating': cand.get('rating'),
            'rating_count': cand.get('ratingCount')}
    try:
        with zipfile.ZipFile(zpath) as z:
            names = {n.lower().rsplit('/', 1)[-1]: n for n in z.namelist()}
            iname = names.get('info.yml') or names.get('info.yaml')
            info = yaml.safe_load(z.read(iname).decode('utf-8-sig')) if iname else {}
            info = info if isinstance(info, dict) else {}
            music = names.get(str(info.get('music') or '').lower().rsplit('/', 1)[-1])
            audio = z.read(music) if music else None
    except Exception as e:  # noqa: BLE001
        return failed_row(base, 'bad_zip', f'{type(e).__name__}: {e}')
    base['info_format'] = str(info.get('format') or '')
    try:
        chart = load_package(zpath)
    except UnsupportedFormat as e:
        chart, err = None, ('unsupported_format', e)
    except Exception as e:  # noqa: BLE001
        chart, err = None, ('parse_error', f'{type(e).__name__}: {e}')
    if audio is None:
        return failed_row(base, 'audio_missing', f'music={info.get("music")!r}')
    try:
        ainfo = ensure_audio(root, audio, None, tmpdir) if chart is not None else _audio_only(audio, tmpdir)
    except Exception as e:  # noqa: BLE001
        return failed_row(base, 'audio_decode_error', f'{type(e).__name__}: {e}', {'audio_sha256': sha256(audio)})
    if chart is None:
        return failed_row(base, err[0], err[1], ainfo)
    try:
        return process_chart(root, chart, base, ainfo)
    except Exception as e:  # noqa: BLE001
        return failed_row(base, 'process_error', f'{type(e).__name__}: {e}', ainfo)


def _audio_only(audio: bytes, tmpdir: str) -> dict:
    sha = sha256(audio)
    tmp = os.path.join(tmpdir, f'{sha}.audio')
    with open(tmp, 'wb') as f:
        f.write(audio)
    try:
        codec = probe_codec(tmp)
    finally:
        os.unlink(tmp)
    return {'audio_sha256': sha, 'codec': codec, 'audio_ext': codec_ext(codec), 'audio_sec': None}


# ---------------------------------------------------------------- pass 2
def assign_splits(rows: list, seed: int = SEED) -> tuple[dict, dict, dict]:
    """Song groups across both sources: union of audio SHA256 and normalized titles (title/work keys of
    group_split, plus the romanized osu title and each Phira name as a shared extra 'alt' key)."""
    g_rows = []
    for r in rows:
        alt = work_key(r.get('title_alt') or r.get('title'))
        g_rows.append({'id': r['chart_id'], 'name': r.get('title') or r.get('title_alt'),
                       'audio_sha256': r.get('audio_sha256'), 'alt': alt or None})
    groups, c2g = group_rows(g_rows, fp_keys=('alt',))
    g2s = {g: split_of(g, seed) for g in groups}
    return groups, c2g, g2s


def finalize_rows(rows: list, seed: int = SEED, th=OFF.V1) -> tuple[list, dict]:
    """Assign split, c, decisions and flags. Returns (manifest rows, c table)."""
    groups, c2g, g2s = assign_splits(rows, seed)
    for r in rows:
        r['song_group'] = c2g[r['chart_id']]
        r['split'] = g2s[r['song_group']]
    # c per (source, codec) on train charts with a valid diagnostic and no P0 quarantine
    pools, src_pool = defaultdict(list), defaultdict(list)
    for r in rows:
        o = r.get('offset') or {}
        if r['status'] == 'parsed' and r['split'] == 'train' and o.get('status') == 'ok' and not r['p0_quarantine']:
            pools[(r['source'], r['codec'])].append(o['delta_ms'])
            src_pool[r['source']].append(o['delta_ms'])
    ctab = {}
    for key in {(r['source'], r.get('codec')) for r in rows if r['status'] == 'parsed'}:
        d = pools.get(key, [])
        if len(d) >= 20:
            ctab[key] = {'c_ms': OFF.train_constant(d), 'n_train': len(d), 'basis': 'source_codec'}
        else:
            ctab[key] = {'c_ms': OFF.train_constant(src_pool[key[0]]) or 0, 'n_train': len(d),
                         'basis': 'source_pooled' if src_pool[key[0]] else 'default_zero'}
    for r in rows:
        if r['status'] != 'parsed':
            r['quarantine_reasons'] = [r['status_code']]
            r['quarantined'], r['kept'] = True, False
            continue
        c = ctab[(r['source'], r['codec'])]
        dec = OFF.decide(r['offset'], c['c_ms'], th)
        dec['c_basis'] = c['basis']
        r['offset_decision'] = dec
        r['timing_unreliable'] = r['timing']['unreliable']
    dis = OFF.disagreement_flags((r['chart_id'], r.get('audio_sha256'), r['offset_decision']['r_ms'])
                                 for r in rows if r['status'] == 'parsed' and r['offset'].get('status') == 'ok')
    for r in rows:
        if r['status'] != 'parsed':
            continue
        r['offset_disagreement'] = r['chart_id'] in dis
        reasons = r['p0_quarantine'] + r['validation_quarantine'] + r['offset_decision']['reasons']
        r['quarantine_reasons'] = list(dict.fromkeys(reasons))
        r['quarantined'] = bool(reasons)
        r['kept'] = not reasons
    return rows, {f'{s}|{c}': v for (s, c), v in sorted(ctab.items(), key=lambda x: (x[0][0], str(x[0][1])))}


def manifest_row(r: dict) -> dict:
    out = {k: v for k, v in r.items() if k not in ('offset', 'legacy_probe')}
    if r.get('offset') is not None:
        out['offset'] = {k: v for k, v in r['offset'].items() if k != 'agree_counts'}
    if r.get('legacy_probe') is not None:
        out['legacy_probe'] = r['legacy_probe']
    return out


def _q(x, qs=(0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)):
    x = np.asarray(x, dtype=float)
    return {f'p{int(q * 100)}': round(float(np.quantile(x, q)), 2) for q in qs} if len(x) else {}


def corpus_stats(rows: list, ctab: dict, extra: dict | None = None) -> dict:
    S = {'seed': SEED, 'row_version': ROW_VERSION, 'thresholds_offset_v1': OFF.V1, **(extra or {})}
    by_src = defaultdict(list)
    for r in rows:
        by_src[r['source']].append(r)
    src = {}
    for s, rs in sorted(by_src.items()):
        parsed = [r for r in rs if r['status'] == 'parsed']
        reasons = Counter(c for r in rs for c in r['quarantine_reasons'])
        src[s] = {'rows': len(rs), 'parsed': len(parsed),
                  'unsupported_format': sum(r['status'] == 'unsupported_format' for r in rs),
                  'unsupported_by_version': dict(Counter(str(r.get('format_version') or r.get('info_format') or '?')
                                                         for r in rs if r['status'] == 'unsupported_format')),
                  'errors_by_code': dict(Counter(r['status_code'] for r in rs if r['status'] == 'error')),
                  'quarantined': sum(r['quarantined'] for r in rs),
                  'quarantined_parsed': sum(r['quarantined'] for r in parsed),
                  'kept': sum(r['kept'] for r in rs),
                  'quarantine_reason_counts': dict(reasons.most_common()),
                  'p0_quarantine_counts': dict(Counter(c for r in parsed for c in r['p0_quarantine']).most_common()),
                  'validation_quarantine_counts': dict(Counter(c for r in parsed for c in r['validation_quarantine']).most_common()),
                  'source_formats': dict(Counter(r['source_format'] for r in parsed)),
                  'unique_audio': len({r['audio_sha256'] for r in parsed if r.get('audio_sha256')}),
                  'audio_hours_parsed': round(_hours(parsed), 2),
                  'audio_hours_kept': round(_hours([r for r in parsed if r['kept']]), 2)}
    S['per_source'] = src
    parsed = [r for r in rows if r['status'] == 'parsed']
    S['unique_audio_total'] = len({r['audio_sha256'] for r in parsed})
    S['audio_hours_total'] = round(_hours(parsed), 2)
    S['song_groups'] = {'all_rows': len({r['song_group'] for r in rows}),
                        'with_kept_chart': len({r['song_group'] for r in rows if r['kept']})}
    gsize = Counter(r['song_group'] for r in rows)
    S['song_groups']['largest_group_sizes'] = [n for _, n in gsize.most_common(5)]
    S['song_groups']['cross_source_groups'] = sum(
        1 for g, ss in _groups_sources(rows).items() if len(ss) > 1)
    split = {}
    for sp in ('train', 'val', 'test'):
        split[sp] = {}
        for s in sorted(by_src):
            kr = [r for r in by_src[s] if r['split'] == sp and r['kept']]
            split[sp][s] = {'charts_kept': len(kr), 'charts_all': sum(r['split'] == sp for r in by_src[s]),
                            'songs_kept': len({r['song_group'] for r in kr}), 'hours_kept': round(_hours(kr), 2)}
    S['splits'] = split
    # offset
    od = {}
    for key, c in ctab.items():
        s, codec = key.split('|')
        rs = [r for r in parsed if r['source'] == s and str(r['codec']) == codec]
        ok = [r for r in rs if r['offset'].get('status') == 'ok']
        acts = Counter(r['offset_decision']['action'] for r in rs)
        od[key] = {**c, 'charts': len(rs), 'diag_ok': len(ok),
                   'diag_status': dict(Counter(r['offset'].get('status') for r in rs)),
                   'delta_ms': _q([r['offset']['delta_ms'] for r in ok]),
                   'abs_r_ms': _q([abs(r['offset_decision']['r_ms']) for r in ok]),
                   'conf': _q([r['offset']['conf'] for r in ok]),
                   'frac_keep': _frac(acts['keep'], len(rs)), 'frac_shift': _frac(acts['shift'], len(rs)),
                   'frac_quarantine': _frac(acts['quarantine'], len(rs)),
                   'offset_quarantine_reasons': dict(Counter(x for r in rs for x in r['offset_decision']['reasons'])),
                   'offset_disagreement': sum(r['offset_disagreement'] for r in rs),
                   'agree_before_by_action': {a: _q([r['offset_decision']['agree_before'] for r in ok
                                                     if r['offset_decision']['action'] == a], (0.1, 0.5, 0.9))
                                              for a in ('keep', 'shift', 'quarantine')},
                   'quarantined_but_agree_before_ge_kept_median': _qa(ok)}
    S['offset'] = od
    S['offset_disagreement_charts'] = sum(r.get('offset_disagreement', False) for r in parsed)
    S['offset_disagreement_audios'] = len({r['audio_sha256'] for r in parsed if r.get('offset_disagreement')})
    S['offset_r_hist_ms'] = _rhist(parsed)
    # timing
    tm = {}
    for s in sorted(by_src):
        ps = [r for r in parsed if r['source'] == s]
        tm[s] = {'charts': len(ps), 'unreliable': sum(r['timing']['unreliable'] for r in ps),
                 'frac_unreliable': _frac(sum(r['timing']['unreliable'] for r in ps), len(ps)),
                 'trigger_counts': dict(Counter(t for r in ps for t in r['timing']['triggers'])),
                 'only_trigger_counts': dict(Counter(r['timing']['triggers'][0] for r in ps if len(r['timing']['triggers']) == 1)),
                 'snap_rate': _q([r['timing']['snap_rate'] for r in ps if r['timing']['snap_rate'] is not None]),
                 'n_segments': _q([r['timing']['n_segments'] for r in ps]),
                 'frac_unreliable_among_kept': _frac(sum(r['timing']['unreliable'] for r in ps if r['kept']),
                                                     sum(r['kept'] for r in ps))}
    S['timing'] = tm
    # types
    ty = {}
    for mode in sorted({r['mode'] for r in parsed}):
        ms = [r for r in parsed if r['mode'] == mode]
        ty[mode] = {'all': _types(ms), 'kept': _types([r for r in ms if r['kept']])}
    S['types'] = ty
    # legacy osu probe (evidence for a possible format revision)
    lp = [r['legacy_probe'] for r in rows if r.get('legacy_probe')]
    if lp:
        ok = [p for p in lp if p.get('status') == 'ok']
        S['osu_legacy_probe'] = {'charts': len(lp), 'ok': len(ok), 'delta_ms': _q([p['delta_ms'] for p in ok]),
                                 'abs_delta_ms': _q([abs(p['delta_ms']) for p in ok]),
                                 'frac_abs_le_25': _frac(sum(abs(p['delta_ms']) <= 25 for p in ok), len(ok)),
                                 'conf': _q([p['conf'] for p in ok]),
                                 'p0_quarantine_if_v14': dict(Counter(c for p in ok for c in p.get('p0_quarantine', [])))}
    return S


def _qa(ok):
    kept = [r['offset_decision']['agree_before'] for r in ok if r['offset_decision']['action'] == 'keep']
    if not kept:
        return None
    m = float(np.median(kept))
    return sum(r['offset_decision']['action'] == 'quarantine' and r['offset_decision']['agree_before'] >= m
               for r in ok)


def _rhist(parsed):
    edges = [-200, -120, -50, -25, -10, 10, 25, 50, 120, 201]
    h = {}
    for s in sorted({r['source'] for r in parsed}):
        rs = [r['offset_decision']['r_ms'] for r in parsed if r['source'] == s and r['offset_decision']['r_ms'] is not None]
        cnt, _ = np.histogram(rs, bins=edges)
        h[s] = {f'[{a},{b})': int(n) for a, b, n in zip(edges, edges[1:], cnt)}
    return h


def _groups_sources(rows):
    d = defaultdict(set)
    for r in rows:
        d[r['song_group']].add(r['source'])
    return d


def _types(rs):
    c = Counter()
    for r in rs:
        c.update(r['types'])
    tot = sum(c.values())
    return {t: {'n': n, 'frac': _frac(n, tot)} for t, n in sorted(c.items())}


def _hours(rs):
    seen = {}
    for r in rs:
        if r.get('audio_sha256') and r.get('audio_sec'):
            seen[r['audio_sha256']] = r['audio_sec']
    return sum(seen.values()) / 3600


def _frac(a, b):
    return round(a / b, 4) if b else None
