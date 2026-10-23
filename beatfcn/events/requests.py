"""Training requests from one full canonical chart (spec section 7, 3.1, 8).

Base kinds: from_scratch, continuation, middle_holes, sparse_anchors, reference (complete S -> disjoint
T). Overlays: explicit empty (natural blank or simulated user ban) and boundary hold. Masks are placed
by time on a 1 ms grid, never by hidden-note counts.

Permission resolution (after all painting): head in complete -> O; head in editable -> hidden target
unless it is a sparse anchor; a hidden hold whose occupancy leaves its editable run becomes O *whole*
(never cropped). Simulated empty removes every hidden target whose occupancy intersects it, whole, with
provenance; if it would touch an O object the mask is resampled. Derived controls are recomputed after
all of this over the editable union (controls.py) and then dropped out (20 % vector, 20 % per field).

References (kind 'reference'): 1-2 complete 4 s segments, <= 16 s, <= 128 selected heads (the span is
shortened first; positives are never relabelled), unselected only for complete coverage, excluding
+-30 ms of every head and every active hold; half from caller-provided salient candidates
(e.g. spectral-flux peaks; no audio code here), the rest uniform over non-note time.

NOT implemented: 'expand editable' alternative for boundary holds (we always observe whole), structured
missing (holds-only / motif / annotated voice), cross-chart A/B references and counterfactual pairs
(P2 intent_pairs), audio-derived salient candidates, scene templates for Phira from-scratch.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from .canonical import MODE_TYPES, Chart, Event, ModelView, build_model_view
from . import controls as C
from .validate import leakage_issues, reference_evidence_issues

KINDS = ('from_scratch', 'continuation', 'middle_holes', 'sparse_anchors', 'reference')
MASK_MIX = {'from_scratch': 0.20, 'continuation': 0.15, 'middle_holes': 0.25, 'sparse_anchors': 0.20, 'reference': 0.20}
MS = 1000
SEC = 1_000_000


class LeakError(AssertionError):
    pass


@dataclass(frozen=True)
class TrainingSample:
    request: dict
    hidden: tuple[Event, ...]
    evidence: tuple[dict, ...]
    removed: tuple[dict, ...]
    recipe: dict
    model_view: ModelView
    raw_controls: dict = field(default_factory=dict)

    @property
    def ok(self):
        return self.model_view.ok


def assert_no_leak(request, hidden, evidence=()):
    errs = leakage_issues(request, hidden, evidence)
    if errs:
        raise LeakError('; '.join(f'{e["code"]}@{e["path"]}' for e in errs))


# ---------------------------------------------------------------- regions
def paint(regions, s, e, state):
    out = []
    for a, b, st in regions:
        if b <= s or a >= e:
            out.append((a, b, st))
            continue
        if a < s:
            out.append((a, s, st))
        if b > e:
            out.append((e, b, st))
    out.append((max(s, regions[0][0]), min(e, regions[-1][1]), state))
    out.sort()
    merged = []
    for a, b, st in out:
        if merged and merged[-1][2] == st and merged[-1][1] == a:
            merged[-1] = (merged[-1][0], b, st)
        elif b > a:
            merged.append((a, b, st))
    return merged


def state_at(regions, t):
    for a, b, st in regions:
        if a <= t < b:
            return st
    raise ValueError(f'time {t} outside regions')


def _ms(rng, lo, hi):
    lo, hi = int(lo) // MS, int(hi) // MS
    return int(rng.integers(lo, hi + 1)) * MS if hi >= lo else lo * MS


def _runs(regions, state='editable'):
    return C.union((a, b) for a, b, st in regions if st == state)


def resolve(events, regions, anchors=frozenset()):
    """Split events into (observed, hidden, provenance). Raises if a head lies in an empty region."""
    runs = _runs(regions)
    obs, hid, prov = [], [], []
    for e in events:
        st = state_at(regions, e.onset_us)
        if st == 'empty':
            raise ValueError(f'{e.event_id} head inside empty region')
        if st == 'complete' or e.event_id in anchors:
            obs.append(e)
            continue
        a, b = e.occupancy()
        if not any(s <= a and b <= t for s, t in runs):
            obs.append(e)
            prov.append({'event_id': e.event_id, 'reason': 'hold_crosses_permission_boundary_observed_whole'})
            continue
        hid.append(e)
    return obs, hid, prov


# ---------------------------------------------------------------- references (spec 3.1)
def build_reference(events, span, coverage, rng, reference_id='ref0', salient_candidates_us=None,
                    max_selected=128, max_unselected=128, exclude_us=30_000, n_unselected=None):
    s, e = span
    heads = sorted(ev.onset_us for ev in events)
    sel = sorted((ev for ev in events if s <= ev.onset_us < e), key=lambda ev: ev.onset_us)
    shortened = None
    if len(sel) > max_selected:      # shorten the complete span; never drop positives inside it
        shortened, e = (s, e), sel[max_selected].onset_us
        sel = sel[:max_selected]
    ctx = [ev for ev in events if ev.is_hold and ev.onset_us < s < ev.end_us]
    unsel, src = [], []
    if coverage == 'complete':
        bad = [(h - exclude_us, h + exclude_us + 1) for h in heads]
        bad += [(ev.onset_us, ev.end_us) for ev in events if ev.is_hold]
        allowed = C.subtract([(s, e)], bad)
        n = min(max_unselected, n_unselected if n_unselected is not None else max(len(sel), 8))
        cands = sorted({int(t) for t in (salient_candidates_us or []) if C.contains(allowed, int(t))})
        k = min(len(cands), n // 2)
        if k:
            pick = rng.choice(len(cands), size=k, replace=False)
            unsel += [cands[i] for i in sorted(pick)]
            src += ['salient'] * k
        L = C.length(allowed)
        tries = 0
        while len(unsel) < n and L > 0 and tries < 50 * n:
            tries += 1
            off = int(rng.integers(0, L))
            for a, b in allowed:
                if off < b - a:
                    t = a + off
                    break
                off -= b - a
            if t not in unsel:
                unsel.append(t)
                src.append('random')
        order = np.argsort(unsel, kind='stable')
        unsel, src = [unsel[i] for i in order], [src[i] for i in order]
    return {'reference_id': reference_id, 'coverage': coverage, 'span': (s, e), 'shortened_from': shortened,
            'selected': [ev.to_json(locked=True) for ev in sel], 'context_holds': [ev.to_json(locked=True) for ev in ctx],
            'unselected_us': unsel, 'unselected_source': src,
            'salient_missing': coverage == 'complete' and not salient_candidates_us}


# ---------------------------------------------------------------- main builder
def build_request(chart: Chart, *, kind: str, seed: int, audio_id: str, audio_duration_us: int,
                  empty_overlay: str | None = None, boundary_hold: bool = False, reference_context: str | None = None,
                  salient_candidates_us=None, control_dropout: bool = True, hide_timing: bool | None = None,
                  request_id: str | None = None) -> TrainingSample:
    if kind not in KINDS:
        raise ValueError(f'kind {kind!r}')
    if chart.quarantined:
        raise ValueError(f'chart quarantined: {[i.code for i in chart.quarantine_reasons()]}')
    D, mode = int(audio_duration_us), chart.mode
    events = list(chart.events)
    for e in events:
        if not (0 <= e.onset_us < D and e.occupancy()[1] <= D):
            raise ValueError(f'{e.event_id} outside audio duration')
    rng = np.random.default_rng(seed)
    recipe = {'kind': kind, 'seed': seed, 'overlays': [], 'notes': []}
    regions = [(0, D, 'editable')]
    anchors, evidence_spans = frozenset(), []
    if kind == 'continuation':
        c = _ms(rng, 0.1 * D, 0.9 * D)
        regions = paint(regions, 0, c, 'complete')
    elif kind == 'middle_holes':
        regions = [(0, D, 'complete')]
        for _ in range(int(rng.integers(1, 4))):
            L = _ms(rng, min(2 * SEC, D // 4), max(min(16 * SEC, D // 3), min(2 * SEC, D // 4)))
            s = _ms(rng, 0, max(D - L, 0))
            regions = paint(regions, s, min(s + L, D), 'editable')
    elif kind == 'sparse_anchors':
        p = float(rng.uniform(0.03, 0.2))
        onsets = sorted({e.onset_us for e in events})
        keep = {t for t in onsets if rng.random() < p}
        anchors = frozenset(e.event_id for e in events if e.onset_us in keep)
        recipe['anchor_group_p'] = p
    elif kind == 'reference':
        ctx = reference_context or ('complete' if rng.random() < 0.5 else 'editable')
        recipe['reference_context'] = ctx
        regions = [(0, D, ctx)]
        seg = min(4 * SEC, D // 4)
        for _ in range(100):
            TL = _ms(rng, min(4 * SEC, D // 4), max(min(16 * SEC, D // 2), min(4 * SEC, D // 4)))
            ts = _ms(rng, 0, D - TL)
            free = C.subtract([(0, D)], [(ts, ts + TL)])
            nseg = int(rng.integers(1, 3))
            spans = []
            for _ in range(nseg):
                fits = [(a, b) for a, b in C.subtract(free, spans) if b - a >= seg]
                if not fits:
                    break
                a, b = fits[int(rng.integers(len(fits)))]
                s = _ms(rng, a, b - seg)
                spans.append((s, s + seg))
            if spans:
                break
        else:
            raise ValueError('cannot place disjoint reference and target spans')
        regions = paint(regions, ts, ts + TL, 'editable')
        for s, e in spans:
            regions = paint(regions, s, e, 'complete')
        evidence_spans = sorted(spans)
        recipe['target_span'] = (ts, ts + TL)
    if boundary_hold:
        runs = _runs(regions)
        cand = [e for e in events if e.is_hold and e.event_id not in anchors and e.end_us - e.onset_us >= 2 * MS
                and C.contains(runs, e.onset_us)]
        if cand:
            h = cand[int(rng.integers(len(cand)))]
            b = _ms(rng, h.onset_us + MS, h.end_us - MS)
            L = _ms(rng, SEC, 4 * SEC)
            regions = paint(regions, b, min(b + L, D), 'complete')
            recipe['overlays'].append({'type': 'boundary_hold', 'hold': h.event_id, 'complete': (b, min(b + L, D))})
        else:
            recipe['notes'].append('boundary_hold_unavailable')
    obs, hid, prov = resolve(events, regions, anchors)
    removed = []
    if empty_overlay in ('natural', 'simulated'):
        regions, obs, hid, removed, note = _empty_overlay(empty_overlay, rng, regions, obs, hid, events, D)
        recipe['overlays' if note is None else 'notes'].append(note or {'type': f'empty_{empty_overlay}'})
        if note is None:
            recipe['overlays'][-1]['span'] = next((a, b) for a, b, st in regions if st == 'empty')
    elif empty_overlay is not None:
        raise ValueError(f'empty_overlay {empty_overlay!r}')
    # final resolution must be a fixed point (no hidden hold crosses any non-editable region)
    obs2, hid2, prov2 = resolve(obs + hid, regions, anchors | {e.event_id for e in obs})
    assert [e.event_id for e in hid2] == [e.event_id for e in hid] and not prov2, 'resolution not a fixed point'
    evidence = []
    for k, sp in enumerate(evidence_spans):
        evidence.append(build_reference(events, sp, 'complete', rng, f'ref{k}', salient_candidates_us))
    bands, raw = C.derive_controls(mode, obs + hid, _runs(regions))
    controls, dropped = dict(bands), []
    if control_dropout:
        if rng.random() < 0.2:
            dropped, controls = sorted(controls), {}
        else:
            for k in sorted(controls):
                if rng.random() < 0.2:
                    dropped.append(k)
                    del controls[k]
    controls['allowed_types'] = list(MODE_TYPES[mode])
    recipe['dropped_controls'] = dropped
    if hide_timing is None:
        hide_timing = bool(rng.random() < 0.5)
    src = chart.sidecar.get('source_name', 'chart')
    obs.sort(key=lambda e: (e.onset_us, e.event_id))
    req = {'schema_version': 'beatfcn.intent.v1',
           'request_id': request_id or f'{chart.source_sha256[:12]}:{kind}:{seed}',
           'audio_id': audio_id, 'audio_duration_us': D, 'mode': mode,
           'observed_events': [e.to_json(locked=True) for e in obs],
           'regions': [{'region_id': f'r{i}-{st}', 'start_us': a, 'end_us': b, 'state': st}
                       for i, (a, b, st) in enumerate(regions)],
           'references': [{'reference_id': ev['reference_id'], 'audio_id': audio_id, 'mode': mode,
                           'span': {'start_us': ev['span'][0], 'end_us': ev['span'][1]}, 'coverage': ev['coverage'],
                           'events': ev['selected'], 'scene_ref': f'{src}#scene' if mode == 'phira' else None,
                           'native_chart_ref': src} for ev in evidence],
           'controls': controls,
           'timing_map_ref': None if hide_timing else f'{src}#timing',
           'scene_ref': f'{src}#scene' if mode == 'phira' else None,
           'hard_profile_ref': None, 'seed': int(seed)}
    recipe['permission_provenance'] = prov
    hid.sort(key=lambda e: (e.onset_us, e.event_id))
    view = build_model_view(mode, obs, hid)
    sample = TrainingSample(req, tuple(hid), tuple(evidence), tuple(removed), recipe, view, raw)
    assert_no_leak(req, hid, evidence)
    ev_errs = reference_evidence_issues(evidence, events)
    if ev_errs:
        raise AssertionError(f'reference evidence invalid: {ev_errs}')
    return sample


def _empty_overlay(how, rng, regions, obs, hid, events, D):
    runs = _runs(regions)
    occ = [e.occupancy() for e in events]
    for _ in range(20):
        if how == 'natural':
            blanks = [(a, b) for a, b in C.subtract(runs, occ) if b - a >= SEC]
            if not blanks:
                return regions, obs, hid, [], 'natural_empty_unavailable'
            a, b = blanks[int(rng.integers(len(blanks)))]
            L = _ms(rng, SEC // 2, min(b - a, 4 * SEC))
            s = _ms(rng, a + MS, b - L - MS) if b - a - L >= 2 * MS else a
            span = (s, min(s + L, b))
        else:
            fits = [(a, b) for a, b in runs if b - a >= SEC]
            if not fits:
                return regions, obs, hid, [], 'simulated_empty_unavailable'
            a, b = fits[int(rng.integers(len(fits)))]
            L = _ms(rng, SEC, min(b - a, 4 * SEC))
            s = _ms(rng, a, b - L)
            span = (s, s + L)
        if any(C.overlaps([span], *e.occupancy()) for e in obs):
            continue   # would affect a locked object: resample
        if how == 'natural' and any(C.overlaps([span], *e.occupancy()) for e in hid):
            continue
        kill = [e for e in hid if C.overlaps([span], *e.occupancy())]
        removed = [{'event': e.to_json(), 'reason': 'simulated_empty_removed_whole', 'span': span} for e in kill]
        ids = {e.event_id for e in kill}
        return paint(regions, *span, 'empty'), obs, [e for e in hid if e.event_id not in ids], removed, None
    return regions, obs, hid, [], f'{how}_empty_resample_exhausted'


def sample_request(chart: Chart, seed: int, audio_id: str, audio_duration_us: int, mix=None,
                   empty_p=0.2, boundary_p=0.2, **kw) -> TrainingSample:
    """Draw kind ~ mix and overlays (20 % empty: half natural / half simulated; 20 % boundary hold)."""
    rng = np.random.default_rng([seed, 0x5EED])
    mix = mix or MASK_MIX
    kinds = list(mix)
    kind = kinds[int(rng.choice(len(kinds), p=np.array([mix[k] for k in kinds]) / sum(mix.values())))]
    empty = None
    if rng.random() < empty_p:
        empty = 'natural' if rng.random() < 0.5 else 'simulated'
    return build_request(chart, kind=kind, seed=seed, audio_id=audio_id, audio_duration_us=audio_duration_us,
                         empty_overlay=empty, boundary_hold=bool(rng.random() < boundary_p), **kw)
