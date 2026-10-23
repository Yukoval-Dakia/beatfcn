"""Sequential song rollout and independent, one-to-one generation metrics.

Phira placement is explicitly outside the evaluated contract when disabled. The legality
checker never calls decoder masks. Ground-truth future events never enter rollout inputs.
"""
from __future__ import annotations
from bisect import bisect_left, bisect_right
from collections import Counter, deque
from dataclasses import replace
import math
import time
import numpy as np
from ..events.canonical import Event, LaneLoc, LineLoc, MODE_TYPES, ModelView, lane_conflicts, oversized_stacks
from . import features as F
from .sample import generate_core, core_context, match_f1


def groups_to_events(groups, c0, mode, line_index=None, id_prefix='generated', placement=False):
    ids = {v:k for k,v in (line_index or {}).items()}
    fallback = next(iter(ids.values()), '0')
    events = []
    for gi, group in enumerate(groups):
        onset = int(c0 + group['t_rel_us'])
        for ni, note in enumerate(group['notes']):
            ty = F.TYPES[note['type']]
            if mode == 'mania_4k':
                loc = LaneLoc(int(note['lane']))
            else:
                # A context placeholder only; no physical placement claim is made in this mode.
                loc = LineLoc(ids[int(note['line'])] if placement else fallback,
                              F.SIDES[int(note['side'])] if placement else 'above',
                              float(note['x'] * F.X_HALF_WIDTH) if placement else 0.)
            end = onset + int(note.get('end_rel_us', 0))
            events.append(Event(f'{id_prefix}:{gi}:{ni}', onset, end, ty, loc))
    return events


def legality_issues(request, additions, *, placement=False):
    """Independent checks of permission, type, duration, identity and cross-core collisions."""
    from ..events.controls import union
    runs = union((r['start_us'], r['end_us']) for r in request['regions'] if r['state']=='editable')
    observed = [Event.from_json(e) for e in request['observed_events']]
    observed_ids = {e.event_id for e in observed}
    allowed = request['controls'].get('allowed_types', MODE_TYPES[request['mode']])
    issues = []
    def add(code, event_id):
        issues.append({'code':code, 'event_id':event_id})
    seen = set(observed_ids)
    for e in additions:
        if e.event_id in seen:
            add('duplicate_or_observed_id',e.event_id)
        seen.add(e.event_id)
        if e.type not in allowed or e.type not in MODE_TYPES[request['mode']]:
            add('type_not_allowed',e.event_id)
        if not 0 <= e.onset_us < request['audio_duration_us'] or e.end_us > request['audio_duration_us']:
            add('outside_audio',e.event_id)
        if (e.is_hold and e.end_us <= e.onset_us) or (not e.is_hold and e.end_us != e.onset_us):
            add('invalid_duration',e.event_id)
        a,b=e.occupancy()
        if not any(s<=a and b<=t for s,t in runs):
            add('outside_editable',e.event_id)
    if request['mode']=='mania_4k':
        for code,a,b in lane_conflicts(observed+list(additions)):
            if a not in observed_ids or b not in observed_ids:
                add(code,b)
    elif placement:
        for a,b in oversized_stacks(observed+list(additions)):
            if b not in observed_ids:
                add('oversized_stack',b)
    return issues


def generate_song(model, sample, song, *, line_index=None, seed=0, device='cpu', greedy=False,
                  core_generator=generate_core, deadline=None):
    """Roll every editable core, carrying only the model's own prior predictions.

    Returns partial output and an explicit failure if bounds or conversion fail; never substitutes
    truth, skips a failed core, crops labels, or silently counts the partial song as complete.
    """
    req=sample.request
    observed=[Event.from_json(e) for e in req['observed_events']]
    if line_index is None:
        ref_events=[Event.from_json(e) for r in req['references'] for e in r['events']]
        line_index=F.line_index_of(observed+ref_events)
        if req['mode']=='phira' and not line_index:
            line_index={'0':0}
    generated, cores, failure = [], [], None
    for c0 in range(0, req['audio_duration_us'], F.CORE_US):
        if not any(r['state']=='editable' and r['start_us'] < c0+F.CORE_US and c0 < r['end_us'] for r in req['regions']):
            continue
        live=replace(sample, hidden=tuple(generated), model_view=ModelView(req['mode'],(),(),()))
        try:
            if deadline is not None and time.perf_counter() >= deadline:
                raise ValueError('generation_deadline')
            ex=F.build_core(live, observed, c0, req['audio_duration_us'], line_index=line_index)
            holds,runs,heads=core_context(live,ex,line_index)
            diagnostics={}
            groups=core_generator(model, ex, song, holds, runs, heads, seed=seed+c0//F.CORE_US,
                                  device=device, greedy=greedy, diagnostics=diagnostics,
                                  **({'deadline':deadline} if deadline is not None else {}))
            times=[g['t_rel_us'] for g in groups]
            if any(b<=a for a,b in zip(times,times[1:])) or any(not 0<=t<F.CORE_US for t in times):
                raise ValueError('nonmonotonic_or_out_of_core_groups')
            if any(not g['notes'] for g in groups):
                raise ValueError('empty_group')
            generated += groups_to_events(groups,c0,req['mode'],line_index,f'generated:{c0}',model.c.phira_location)
            cores.append({'c0_us':c0,'groups':len(groups),**diagnostics})
            if diagnostics.get('deadline_reached'):
                raise ValueError('generation_deadline')
            if diagnostics.get('group_limit_reached'):
                raise ValueError('group_limit_reached')
        except (F.SkipExample, ValueError, KeyError, IndexError) as exc:
            failure={'c0_us':c0,'reason':f'{type(exc).__name__}:{exc}'}
            break
    issues=legality_issues(req, generated, placement=model.c.phira_location)
    return {'events':generated,'cores':cores,'complete':failure is None,'failure':failure,
            'legality':issues,'placement_checked':req['mode']=='mania_4k' or model.c.phira_location}


def hold_matches(pred, truth, head_tol, end_tol=None, match_lane=False):
    """Maximum-cardinality one-to-one matching; joint matches require BOTH endpoints."""
    pred=sorted((e for e in pred if e.is_hold),key=lambda e:e.onset_us)
    truth=sorted((e for e in truth if e.is_hold),key=lambda e:e.onset_us)
    times=[e.onset_us for e in truth]
    edges=[]
    for p in pred:
        candidates=range(bisect_left(times,p.onset_us-head_tol),bisect_right(times,p.onset_us+head_tol))
        edges.append([j for j in candidates if (end_tol is None or abs(p.end_us-truth[j].end_us)<=end_tol)
                      and (not match_lane or p.location==truth[j].location)])
    p2t,t2p={},{}
    for start in range(len(pred)):
        queue, visited, parent=deque([start]),{start},{}
        finish=None
        while queue and finish is None:
            pi=queue.popleft()
            for ti in edges[pi]:
                if ti in parent: continue
                parent[ti]=pi
                if ti not in t2p:
                    finish=ti
                    break
                nxt=t2p[ti]
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append(nxt)
        while finish is not None:
            pi=parent[finish]
            old=p2t.get(pi)
            p2t[pi],t2p[finish]=finish,pi
            finish=old
    pairs=sorted(p2t.items())
    return {'tp':len(pairs),'pred':len(pred),'true':len(truth),
            'tail_errors_us':[abs(pred[i].end_us-truth[j].end_us) for i,j in pairs]}


def score_events(pred, truth, mode):
    out={'pred':len(pred),'true':len(truth), 'density_ratio':len(pred)/len(truth) if truth else None}
    # Timing alone would credit a tap on the wrong key, or a hold replacing a tap.
    # Phira deliberately scores type/multiplicity only when placement is outside P1's contract.
    def note_key(event):
        return (event.type, event.location.lane if mode == 'mania_4k' else None)
    pred_notes, truth_notes = {}, {}
    for events, groups in ((pred, pred_notes), (truth, truth_notes)):
        for event in events:
            groups.setdefault(note_key(event), []).append(event.onset_us)
    for ms in (10,20,50):
        tp,npred,ntrue,f1=match_f1([e.onset_us for e in pred],[e.onset_us for e in truth],ms*1000)
        out[f'onset_{ms}ms']={'tp':tp,'pred':npred,'true':ntrue,'f1':f1}
        note_tp = sum(match_f1(pred_notes.get(key, []), truth_notes.get(key, []), ms * 1000)[0]
                      for key in pred_notes.keys() | truth_notes.keys())
        out[f'note_{ms}ms']={'tp':note_tp,'pred':len(pred),'true':len(truth),
                           'f1':2*note_tp/max(len(pred)+len(truth),1)}
        h=hold_matches(pred,truth,ms*1000,match_lane=mode=='mania_4k')
        joint=hold_matches(pred,truth,ms*1000,ms*1000,match_lane=mode=='mania_4k')
        for label,counts in [('hold_head',h),('hold_joint',joint)]:
            errors=counts.pop('tail_errors_us')
            counts['f1']=2*counts['tp']/max(counts['pred']+counts['true'],1)
            out[f'{label}_{ms}ms']=counts
            if label=='hold_head':
                out[f'tail_error_given_head_{ms}ms']={'matched':len(errors),
                    'mae_ms':float(np.mean(errors)/1000) if errors else None,
                    'p95_ms':float(np.quantile(errors,.95)/1000) if errors else None}
    return out


# ---------------------------------------------------------------- quality metrics (design/FIRST_CANDIDATE_v2.md §1)
# All of them score the editable union only (the onset-F1 scope); onsets are per event, as in score_events.
SEGMENT_US = 8 * F.SEC
REPEAT_GAP_US = 16 * F.SEC
IOI_BINS, IOI_MIN_MS, IOI_MAX_MS = 24, 30., 4000.
# summary[mode] mean fields: per-song dict section -> scalar fields averaged over the songs where they are not None
QUALITY_MEANS = {'onset_error_50ms': ('match_rate', 'median_ms', 'p90_ms'),
                 'density_curve': ('pearson_r', 'count_ratio'),
                 'repetition': ('p_model_given_human', 'p_model_given_not_human'),
                 'ioi_kl': ('kl_nats',),
                 'style_variance': ('hold_frac_var_generated', 'hold_frac_var_true',
                                    'chord_frac_var_generated', 'chord_frac_var_true')}


def editable_union(regions):
    from ..events.controls import union
    return union((r['start_us'], r['end_us']) for r in regions if r['state'] == 'editable')


def _inside(t, runs):
    i = bisect_right(runs, (t, float('inf'))) - 1
    return i >= 0 and runs[i][0] <= t < runs[i][1]


def _editable_onsets(events, runs):
    return sorted(e.onset_us for e in events if _inside(e.onset_us, runs))


def onset_error_distribution(pred, truth, runs, tol_us=50_000):
    """Greedy nearest one-to-one matching (smallest |error| first) of generated to true onsets within tol.
    match_rate = matched / true onsets; median / p90 of |error| over the matched pairs, in ms."""
    p, t = _editable_onsets(pred, runs), _editable_onsets(truth, runs)
    candidates = sorted((abs(x - t[j]), i, j) for i, x in enumerate(p)
                        for j in range(bisect_left(t, x - tol_us), bisect_right(t, x + tol_us)))
    used_p, used_t, errors = set(), set(), []
    for d, i, j in candidates:
        if i not in used_p and j not in used_t:
            used_p.add(i)
            used_t.add(j)
            errors.append(d)
    return {'matched': len(errors), 'pred': len(p), 'true': len(t), 'tol_ms': tol_us / 1000,
            'match_rate': len(errors) / len(t) if t else None,
            'median_ms': float(np.median(errors)) / 1000 if errors else None,
            'p90_ms': float(np.quantile(errors, .9)) / 1000 if errors else None}


def density_curve(pred, truth, runs, block_us=F.BLOCK_US, min_blocks=10):
    """Onset counts per 2 s block of the song clock, over the blocks lying wholly inside the editable union.
    pearson_r needs >= min_blocks blocks and a non-constant curve on both sides; count_ratio = all
    generated / all true onsets in the editable union (not only the whole blocks)."""
    blocks = [k for s, e in runs for k in range(-(-s // block_us), e // block_us)]
    p, t = _editable_onsets(pred, runs), _editable_onsets(truth, runs)
    def counts(onsets):
        return np.array([bisect_left(onsets, (k + 1) * block_us) - bisect_left(onsets, k * block_us) for k in blocks], float)
    g, h = counts(p), counts(t)
    r = None
    if len(blocks) >= min_blocks and g.std() > 0 and h.std() > 0:
        r = float(np.corrcoef(g, h)[0, 1])
    return {'blocks': len(blocks), 'block_s': block_us / F.SEC, 'pearson_r': r,
            'generated': len(p), 'true': len(t), 'count_ratio': len(p) / len(t) if t else None}


def _keyed_onsets(events, runs, mode):
    """Sorted onsets per matching key: lane for mania (lane equality required), one key for Phira."""
    out = {}
    for e in events:
        if _inside(e.onset_us, runs):
            out.setdefault(e.location.lane if mode == 'mania_4k' else None, []).append(e.onset_us)
    return {k: sorted(v) for k, v in out.items()}


def _segment_f1(keyed, a, b, window_us, tol_us):
    """Onset F1 between the two windows after shifting each to its own start (Δ=0); both empty -> 1."""
    tp = na = nb = 0
    for onsets in keyed.values():
        sa = [x - a for x in onsets[bisect_left(onsets, a):bisect_left(onsets, a + window_us)]]
        sb = [x - b for x in onsets[bisect_left(onsets, b):bisect_left(onsets, b + window_us)]]
        tp += match_f1(sa, sb, tol_us)[0]
        na, nb = na + len(sa), nb + len(sb)
    return 2 * tp / (na + nb) if na + nb else 1.0


def repetition_consistency(pred, truth, runs, mode, summary, *, window_us=SEGMENT_US, min_gap_us=REPEAT_GAP_US,
                           top_fraction=0.05, min_pairs=3, tol_us=20_000, reuse_f1=0.5):
    """Segment pairs come from AUDIO only: the song's resident S20 summary rows (SongCache.summary, the
    2 s band means of the frozen teacher, [S, 8, D]); 8 s windows = 4 consecutive rows at a 2 s stride,
    wholly inside the editable union; cosine of the window means; |Δt| >= 16 s; top `top_fraction` pairs
    (ceil), None unless at least `min_pairs` are kept. Chart similarity per pair = 20 ms onset F1 (lane
    equality for mania) with both windows aligned to their starts; reuse <=> F1 >= 0.5 on that chart."""
    out = {'source': 'summary_s20' if summary is not None else None, 'windows': 0, 'candidate_pairs': 0,
           'kept_pairs': 0, 'human_reuse_pairs': 0, 'human_no_reuse_pairs': 0,
           'p_model_given_human': None, 'p_model_given_not_human': None}
    if summary is None:
        return out
    if hasattr(summary, 'detach'):
        summary = summary.detach().float().cpu().numpy()
    s = np.asarray(summary, np.float32)
    s = s.reshape(s.shape[0], -1)
    rows, stride_us = window_us // F.BLOCK_US, F.BLOCK_US
    starts = [i for i in range(s.shape[0] - rows + 1)
              if any(a <= i * stride_us and (i * stride_us + window_us) <= b for a, b in runs)]
    out['windows'] = len(starts)
    if not starts:
        return out
    v = np.stack([s[i:i + rows].mean(0) for i in starts])
    v /= np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
    cos = v @ v.T
    gap = -(-min_gap_us // stride_us)
    pairs = [(-float(cos[x, y]), x, y) for x in range(len(starts)) for y in range(x + gap, len(starts))]
    out['candidate_pairs'] = len(pairs)
    keep = math.ceil(len(pairs) * top_fraction)
    if keep < min_pairs:
        return out
    pairs.sort()
    kp, kt = _keyed_onsets(pred, runs, mode), _keyed_onsets(truth, runs, mode)
    human, model = [], []
    for _, x, y in pairs[:keep]:
        a, b = starts[x] * stride_us, starts[y] * stride_us
        human.append(_segment_f1(kt, a, b, window_us, tol_us) >= reuse_f1)
        model.append(_segment_f1(kp, a, b, window_us, tol_us) >= reuse_f1)
    yes = [m for h, m in zip(human, model) if h]
    no = [m for h, m in zip(human, model) if not h]
    out.update(kept_pairs=len(human), human_reuse_pairs=len(yes), human_no_reuse_pairs=len(no),
               p_model_given_human=sum(yes) / len(yes) if yes else None,
               p_model_given_not_human=sum(no) / len(no) if no else None)
    return out


def _iois_ms(events, runs, lo_ms, hi_ms):
    out = []
    for a, b in runs:
        onsets = sorted({e.onset_us for e in events if a <= e.onset_us < b})
        out += [d / 1000 for d in np.diff(onsets) if lo_ms <= d / 1000 <= hi_ms]
    return out


def ioi_histogram_kl(pred, truth, runs, bins=IOI_BINS, lo_ms=IOI_MIN_MS, hi_ms=IOI_MAX_MS):
    """KL(true || generated) in nats between Laplace-smoothed histograms of log2(IOI ms) over `bins`
    log-spaced bins in [lo_ms, hi_ms]; IOIs between distinct onsets within one editable run; None when
    the true chart has no IOI in range."""
    edges = np.logspace(np.log2(lo_ms), np.log2(hi_ms), bins + 1, base=2)
    edges[0], edges[-1] = lo_ms, hi_ms
    p, t = _iois_ms(pred, runs, lo_ms, hi_ms), _iois_ms(truth, runs, lo_ms, hi_ms)
    def smoothed(x):
        c = np.histogram(x, edges)[0] + 1.
        return c / c.sum()
    q, r = smoothed(p), smoothed(t)
    return {'bins': bins, 'range_ms': [lo_ms, hi_ms], 'generated': len(p), 'true': len(t),
            'kl_nats': float(np.sum(r * np.log(r / q))) if t else None}


def _style_stats(events, runs, segment_us):
    """Per 8 s segment of the song clock that meets the editable union and holds >= 1 note: hold fraction
    and chord fraction (notes sharing their onset with another note of the segment)."""
    hold, chord = [], []
    last = max((b for _, b in runs), default=0)
    for k in range(-(-last // segment_us)):
        a, b = k * segment_us, (k + 1) * segment_us
        if not any(s < b and a < e for s, e in runs):
            continue
        notes = [e for e in events if a <= e.onset_us < b and _inside(e.onset_us, runs)]
        if not notes:
            continue
        onsets = Counter(e.onset_us for e in notes)
        hold.append(sum(e.is_hold for e in notes) / len(notes))
        chord.append(sum(onsets[e.onset_us] > 1 for e in notes) / len(notes))
    var = lambda x: float(np.var(x)) if len(x) >= 2 else None
    return len(hold), var(hold), var(chord)


def style_variance(pred, truth, runs, segment_us=SEGMENT_US):
    """Population variance across 8 s segments of the hold and chord fractions, for each chart (None with < 2 segments)."""
    out = {'segment_s': segment_us / F.SEC}
    for label, events in (('generated', pred), ('true', truth)):
        n, hv, cv = _style_stats(events, runs, segment_us)
        out.update({f'segments_{label}': n, f'hold_frac_var_{label}': hv, f'chord_frac_var_{label}': cv})
    return out


def quality_metrics(pred, truth, regions, mode, summary=None):
    """The §1 quality metrics of one song: matched onset error, density curve, repetition consistency,
    IOI histogram KL and segment style variance, all over the editable union of `regions`."""
    runs = editable_union(regions)
    return {'onset_error_50ms': onset_error_distribution(pred, truth, runs),
            'density_curve': density_curve(pred, truth, runs),
            'repetition': repetition_consistency(pred, truth, runs, mode, summary),
            'ioi_kl': ioi_histogram_kl(pred, truth, runs),
            'style_variance': style_variance(pred, truth, runs)}


def quality_summary(records):
    """Per-mode means of every QUALITY_MEANS scalar over the songs where it is not None, with that song count."""
    out = {}
    for section, fields in QUALITY_MEANS.items():
        agg = {}
        for name in fields:
            vals = [r['quality'][section][name] for r in records if r.get('quality', {}).get(section, {}).get(name) is not None]
            agg[name] = {'mean': float(np.mean(vals)) if vals else None, 'songs': len(vals)}
        out[section] = agg
    for name in ('kept_pairs', 'human_reuse_pairs', 'human_no_reuse_pairs'):
        out['repetition'][f'{name}_total'] = sum(r['quality']['repetition'][name] for r in records if 'quality' in r)
    return out


def evaluate_songs(model, corpus, rows, n_songs=20, seed=20261023, device='cpu', greedy=True, per_song_timeout_s=None):
    """Fixed held-out songs, one chart per song, full from-scratch rollout without oracle controls.

    Failed/illegal songs stay in the denominator. Every result retains identity, seed and raw counts.
    """
    import hashlib
    from ..events.requests import build_request
    from ..train.audit import fingerprint
    if n_songs <= 0 or not rows:
        raise ValueError('whole-song evaluation needs positive n_songs and nonempty held-out rows')
    if per_song_timeout_s is not None and per_song_timeout_s <= 0:
        raise ValueError('per-song timeout must be positive')
    if any(r['split'] not in ('val','test') for r in rows):
        raise ValueError('whole-song evaluation requires held-out rows')
    started = time.perf_counter()
    by_song={}
    for row in sorted(rows,key=lambda r:r['chart_id']):
        by_song.setdefault((row['mode'],row['audio_sha256']),[]).append(row)
    mode_keys={mode:deque(sorted((k for k in by_song if k[0]==mode),
                key=lambda key:hashlib.sha256(f'{seed}:{key}'.encode()).hexdigest()))
               for mode in sorted({k[0] for k in by_song})}
    chosen=[]
    while len(chosen)<n_songs and any(mode_keys.values()):
        for queue in mode_keys.values():
            if queue and len(chosen)<n_songs:
                chosen.append(queue.popleft())
    records=[]
    corpus.train_rows(rows)
    for i,key in enumerate(chosen):
        song_started = time.perf_counter()
        row=by_song[key][seed % len(by_song[key])]
        record={'chart_id':row['chart_id'],'song':row['audio_sha256'],'mode':row['mode'],'seed':seed+i}
        try:
            chart=corpus.chart(row['chart_id'])
            duration=int(round(row['audio_sec']*1e6))
            sample=build_request(chart,kind='from_scratch',seed=seed+i,audio_id=row['audio_sha256'],
                                 audio_duration_us=duration,empty_overlay=None,control_dropout=False)
            if not sample.ok:
                raise ValueError('model_view_quarantine')
            # Allowed types are part of the task; density/intensity inferred from the answer are not.
            sample=replace(sample,request={**sample.request,'controls':{'allowed_types':list(MODE_TYPES[row['mode']])}})
            truth=[e for g in sample.model_view.target_groups for e in g.events]
            song_cache=corpus.song(row['audio_sha256'],in_memory=True)
            rollout=generate_song(model,sample,song_cache,
                                  line_index=F.line_index_of(chart.events),seed=seed+i,device=device,greedy=greedy,
                                  deadline=None if per_song_timeout_s is None else song_started+per_song_timeout_s)
            record.update({k:v for k,v in rollout.items() if k!='events'})
            record['scores']=score_events(rollout['events'],truth,row['mode'])
            # §1 quality metrics; the repetition pairs come from the song's resident S20 summary (audio only).
            record['quality']=quality_metrics(rollout['events'],truth,sample.request['regions'],row['mode'],
                                              getattr(song_cache,'summary',None))
            record['generated_events']=[e.to_json() for e in rollout['events']]
        except (ValueError,F.SkipExample,AssertionError) as exc:
            record.update(complete=False,failure={'reason':f'{type(exc).__name__}:{exc}'},legality=[])
        record['wall_s'] = time.perf_counter() - song_started
        records.append(record)
    summary={}
    for mode in sorted({r['mode'] for r in records}):
        recs=[r for r in records if r['mode']==mode]
        s={'requested':len(recs),'completed':sum(r['complete'] for r in recs),
           'legal_completed':sum(r['complete'] and not r['legality'] for r in recs),
           'illegal_songs':sum(bool(r['legality']) for r in recs),
           'legality_reasons':dict(Counter(i['code'] for r in recs for i in r['legality'])),
           'unscored':sum('scores' not in r for r in recs)}
        for kind in ('onset','note','hold_head','hold_joint'):
            for ms in (10,20,50):
                name=f'{kind}_{ms}ms'
                counts={k:sum(r.get('scores',{}).get(name,{}).get(k,0) for r in recs) for k in ('tp','pred','true')}
                counts['f1']=2*counts['tp']/max(counts['pred']+counts['true'],1)
                s[name]=counts
        s.update(quality_summary(recs))
        summary[mode]=s
    return {'schema':'beatfcn.whole_song_eval.v2','seed':seed,'greedy':greedy,'controls':'allowed_types_only',
            'per_song_timeout_s':per_song_timeout_s,
            'wall_s':time.perf_counter()-started,
            'note_match':'type and onset; mania also requires exact lane; Phira placement is not scored',
            'quality_metrics':'design/FIRST_CANDIDATE_v2.md §1: onset_error_50ms, density_curve, repetition (audio S20 pairs), ioi_kl, style_variance; per song and per-mode means',
            'valid_for_comparison':len(records)==n_songs and all('scores' in r for r in records),
            'rows_sha256':fingerprint(rows),'requested':n_songs,'available_selected':len(chosen),
            'summary':summary,'songs':records}
