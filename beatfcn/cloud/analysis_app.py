"""Modal app: data analyses that feed design decisions.

dup_onsets: do exact duplicate Phira notes (same line/side/x/type/time) sit on stronger audio
onsets than ordinary notes? Per chart, each head time gets the onset strength (max positive
spectral flux within +-20 ms, after the chart's own global offset delta) and is classed as
dup / chord (>=2 heads at the time, no exact duplicate) / single. Strength is compared as a
within-chart percentile so loud and quiet songs are comparable.
"""
from __future__ import annotations
import json
from pathlib import Path
import modal

_HERE = Path(__file__).resolve()
REPO = _HERE.parents[2] if len(_HERE.parents) > 2 else _HERE.parent
DATA = '/data'
app = modal.App('beatfcn-analysis')
data = modal.Volume.from_name('beatfcn-data', create_if_missing=True)
image = (modal.Image.debian_slim(python_version='3.12')
         .pip_install('torch==2.7.1', index_url='https://download.pytorch.org/whl/cpu')
         .pip_install('numpy==2.2.6', 'pyyaml==6.0.3')
         .add_local_python_source('beatfcn'))


def _classify(events):
    from collections import defaultdict
    by_t = defaultdict(list)
    for e in events:
        loc = e['location']
        by_t[e['onset_us']].append((e['type'], loc.get('line_id'), loc.get('side'), loc.get('native_x')))
    out = {}
    for t, sigs in by_t.items():
        dup = len(sigs) != len(set(sigs))
        out[t] = 'dup' if dup else ('chord' if len(sigs) >= 2 else 'single')
    return out, by_t


@app.function(image=image, volumes={DATA: data}, cpu=2.0, memory=4096, timeout=1800)
def dup_chunk(rows: list) -> list:
    import numpy as np
    from beatfcn.ingest.offset import flux_envelope
    res = []
    for r in rows:
        try:
            doc = json.load(open(f"{DATA}/canon/phira/{r['native_id']}.json"))
            mel = np.load(f"{DATA}/mel10/{r['audio_sha256']}.npy").astype(np.float32)
        except Exception as exc:
            res.append({'chart_id': r['chart_id'], 'error': str(exc)[:120]})
            continue
        env = flux_envelope(mel)
        delta = ((r.get('offset') or {}).get('delta_ms') or 0) / 1000.0
        cls, by_t = _classify(doc['events'])
        times = np.array(sorted(cls), dtype=np.int64)
        fr = np.arange(len(env)) * 0.01

        def strength(t_sec):
            a = max(int((t_sec - 0.02) * 100), 0)
            b = min(int((t_sec + 0.02) * 100) + 1, len(env))
            return float(env[a:b].max()) if b > a else float('nan')
        s = np.array([strength(t / 1e6 + delta) for t in times])
        ok = np.isfinite(s)
        pct = np.full(len(s), np.nan)
        if ok.sum() >= 10:
            order = s[ok].argsort().argsort()
            pct[ok] = (order + 0.5) / ok.sum()
        kinds = np.array([cls[int(t)] for t in times])
        dup_sigs = []
        for t in times[kinds == 'dup']:
            sigs = by_t[int(t)]
            for sig in set(sigs):
                n = sigs.count(sig)
                if n > 1:
                    dup_sigs.append({'type': sig[0], 'mult': n})
        out = {'chart_id': r['chart_id'], 'name': r.get('title'), 'rating': r.get('rating'), 'n_heads': int(len(times)),
               'n_dup_times': int((kinds == 'dup').sum()), 'n_chord_times': int((kinds == 'chord').sum()),
               'delta_ms': r.get('offset', {}).get('delta_ms'), 'dup_types': dup_sigs}
        for k in ('dup', 'chord', 'single'):
            m = (kinds == k) & np.isfinite(pct)
            out[f'pct_{k}'] = float(np.median(pct[m])) if m.any() else None
            out[f'n_{k}'] = int(m.sum())
            out[f'top25_{k}'] = float((pct[m] >= 0.75).mean()) if m.any() else None
        # one 8 s example window centred on the densest duplicate cluster
        if (kinds == 'dup').any():
            dt = times[kinds == 'dup']
            centre = int(dt[np.argmax([(np.abs(dt - t) < 4_000_000).sum() for t in dt])])
            a, b = centre - 4_000_000, centre + 4_000_000
            lines = sorted({e['location'].get('line_id') for e in doc['events']}, key=lambda x: (len(str(x)), str(x)))
            li = {l: i for i, l in enumerate(lines)}
            notes = [{'t': round((e['onset_us'] - a) / 1e6, 3), 'line': li[e['location'].get('line_id')],
                      'x': e['location'].get('native_x'), 'type': e['type'], 'side': e['location'].get('side'),
                      'dup': cls[e['onset_us']] == 'dup' and by_t[e['onset_us']].count(
                          (e['type'], e['location'].get('line_id'), e['location'].get('side'), e['location'].get('native_x'))) > 1}
                     for e in doc['events'] if a <= e['onset_us'] < b]
            i0, i1 = max(int((a / 1e6 + delta) * 100), 0), min(int((b / 1e6 + delta) * 100), len(env))
            out['example'] = {'start_s': round(a / 1e6, 2), 'notes': notes,
                              'env': [round(float(v), 2) for v in env[i0:i1]], 'env_offset_s': round(max(-(a / 1e6 + delta), 0), 2)}
        res.append(out)
    return res


@app.local_entrypoint()
def dup_onsets(manifest: str, out: str = 'reports/phira_duplicate_onsets.json'):
    rows = [json.loads(l) for l in open(manifest)]
    rows = [r for r in rows if r['source'] == 'phira' and r.get('status') == 'parsed'
            and 'duplicate_object' in (r.get('quarantine_reasons') or [])]
    chunks = [rows[i:i + 20] for i in range(0, len(rows), 20)]
    res = [x for part in dup_chunk.map(chunks) for x in part]
    Path(out).write_text(json.dumps(res, ensure_ascii=False))
    import numpy as np
    good = [r for r in res if 'error' not in r and r.get('pct_dup') is not None]
    print(json.dumps({'charts': len(rows), 'analysed': len(good),
                      'median_pct': {k: float(np.median([r[f'pct_{k}'] for r in good if r[f'pct_{k}'] is not None])) for k in ('dup', 'chord', 'single')}}))


# ---------------------------------------------------------------- Phira type-combination statics
HF_BINS = (97, 128)  # ~5-12 kHz on the Slaney mel128 (cymbals, hats)
LF_BINS = (0, 20)    # below ~530 Hz (kick, bass)


def _band_env(mel, lo, hi):
    import numpy as np
    m = mel[:, lo:hi]
    d = np.concatenate([[0], np.maximum(m[1:] - m[:-1], 0).sum(1)])
    d = np.convolve(d, [.25, .5, .25], 'same')
    d = d - np.convolve(d, np.ones(51) / 51, 'same')
    return d / (d.std() + 1e-6)


def _combo(types, stacked):
    from collections import Counter
    c = Counter(types)
    if len(types) == 1:
        return 'single_' + types[0]
    if 'flick' in c:
        return 'chord_with_flick'
    key = '+'.join(f'{t}x{c[t]}' if c[t] > 1 else t for t in ('tap', 'drag', 'hold') if c[t])
    return ('stack:' if stacked else 'chord:') + key


@app.function(image=image, volumes={DATA: data}, cpu=2.0, memory=4096, timeout=1800)
def combo_chunk(rows: list) -> list:
    import numpy as np
    from collections import defaultdict
    from fractions import Fraction
    from beatfcn.ingest.offset import flux_envelope
    out = []
    for r in rows:
        try:
            doc = json.load(open(f"{DATA}/canon/phira/{r['native_id']}.json"))
            mel = np.load(f"{DATA}/mel10/{r['audio_sha256']}.npy").astype(np.float32)
        except Exception as exc:
            out.append({'chart_id': r['chart_id'], 'error': str(exc)[:100]})
            continue
        envs = {'all': flux_envelope(mel), 'hf': _band_env(mel, *HF_BINS), 'lf': _band_env(mel, *LF_BINS)}
        delta = ((r.get('offset') or {}).get('delta_ms') or 0) / 1000.0
        by_t = defaultdict(list)
        for e in doc['events']:
            loc = e['location']
            by_t[e['onset_us']].append((e['type'], loc.get('line_id'), loc.get('side'), loc.get('native_x')))
        times = sorted(by_t)

        def strength(env, t):
            a, b = max(int((t - 0.02) * 100), 0), min(int((t + 0.02) * 100) + 1, len(env))
            return float(env[a:b].max()) if b > a else np.nan
        S = {k: np.array([strength(v, t / 1e6 + delta) for t in times]) for k, v in envs.items()}
        P = {}
        for k, s in S.items():
            ok = np.isfinite(s)
            p = np.full(len(s), np.nan)
            if ok.sum() >= 10:
                p[ok] = (s[ok].argsort().argsort() + 0.5) / ok.sum()
            P[k] = p
        combos = []
        for t in times:
            sigs = by_t[t]
            combos.append(_combo([s[0] for s in sigs], len(sigs) != len(set(sigs))))
        rec = {'chart_id': r['chart_id'], 'n_times': len(times), 'timing_unreliable': r.get('timing_unreliable'),
               'combos': {}, 'tag': r.get('difficulty_label')}
        # Position relationships (x is line-local, so distances only within one line).
        def pair_key(a, b):
            return '+'.join(sorted((a, b)))
        chord_pos, run_pos = defaultdict(lambda: [0, 0, []]), defaultdict(lambda: [0, 0, []])
        for t in times:
            sigs = by_t[t]
            for i in range(len(sigs)):
                for j in range(i + 1, len(sigs)):
                    a_, b_ = sigs[i], sigs[j]
                    if a_ == b_:
                        continue  # exact stack, counted elsewhere
                    k = chord_pos[pair_key(a_[0], b_[0])]
                    k[0] += 1
                    if a_[1] == b_[1]:
                        k[1] += 1
                        if len(k[2]) < 300:
                            k[2].append(round(abs((a_[3] or 0) - (b_[3] or 0))))
        for i in range(1, len(times)):
            if times[i] - times[i - 1] > 150_000 or len(by_t[times[i]]) != 1 or len(by_t[times[i - 1]]) != 1:
                continue  # fast single-note successions only (<= 150 ms)
            a_, b_ = by_t[times[i - 1]][0], by_t[times[i]][0]
            k = run_pos[a_[0] + '>' + b_[0]]
            k[0] += 1
            if a_[1] == b_[1]:
                k[1] += 1
                if len(k[2]) < 300:
                    k[2].append(round(abs((a_[3] or 0) - (b_[3] or 0))))
        rec['chord_pos'] = {k: v for k, v in chord_pos.items()}
        rec['run_pos'] = {k: v for k, v in run_pos.items()}
        for c in set(combos):
            idx = [i for i, cc in enumerate(combos) if cc == c]
            rec['combos'][c] = {'n': len(idx), **{f'p_{k}': float(np.nanmedian(P[k][idx])) if np.isfinite(P[k][idx]).any() else None for k in P}}
        # periodicity on the chart's own beat grid (only when timing is reliable)
        tm = doc.get('timing')
        if tm and tm.get('segments') and not r.get('timing_unreliable'):
            segs = [(float(Fraction(s['start_beat'])), float(Fraction(s['bpm'])), float(Fraction(s['start_sec']))) for s in tm['segments']]

            def beat_of(sec):
                b0, bpm, s0 = segs[0]
                for sb, bp, ss in segs:
                    if ss <= sec:
                        b0, bpm, s0 = sb, bp, ss
                return b0 + (sec - s0) * bpm / 60.0
            native_sec = lambda t_us: t_us / 1e6  # canon onset is audio time; beats need chart time
            off = float(Fraction(doc.get('offset_sec', '0')))
            beats = np.array([beat_of(t / 1e6 + off) for t in times])

            def iois(mask):
                b = beats[np.asarray(mask)]
                return np.round(np.diff(b) * 4) / 4 if len(b) > 1 else np.array([])
            ctype = np.array(combos)
            flick = np.array(['flick' in c for c in combos])
            mixed = np.array([c.startswith(('chord:', 'stack:')) and 'tap' in c and 'drag' in c for c in combos])
            rec['ioi_flick'] = iois(flick).tolist()[:400]
            rec['ioi_mixed'] = iois(mixed).tolist()[:400]
            rec['phase_flick'] = (np.round((beats[flick] % 4) * 4) / 4).tolist()[:400]
            rec['phase_all'] = (np.round((beats % 4) * 4) / 4).tolist()[:2000]
            # dense blue-yellow runs: consecutive heads <= 1/4 beat apart mixing tap & drag
            types_at = [set(s[0] for s in by_t[t]) for t in times]
            runs, cur = [], 0
            for i in range(1, len(times)):
                close = beats[i] - beats[i - 1] <= 0.26
                mix = ('tap' in types_at[i] or 'tap' in types_at[i - 1]) and ('drag' in types_at[i] or 'drag' in types_at[i - 1])
                if close and mix:
                    cur += 1
                elif cur:
                    runs.append(cur + 1)
                    cur = 0
            rec['by_runs'] = runs[:300]
        out.append(rec)
    return out


@app.local_entrypoint()
def phira_combos(manifest: str, out: str = 'reports/phira_type_combos.json'):
    from beatfcn.train.data import dup_only, selected_for_training
    rows = [json.loads(l) for l in open(manifest)]
    rows = [r for r in rows if r['source'] == 'phira' and r.get('status') == 'parsed' and selected_for_training(r)
            and (r.get('kept') or dup_only(r))]
    chunks = [rows[i:i + 25] for i in range(0, len(rows), 25)]
    res = [x for part in combo_chunk.map(chunks) for x in part]
    Path(out).write_text(json.dumps(res))
    print(json.dumps({'charts': len(rows), 'ok': sum('error' not in r for r in res)}))
