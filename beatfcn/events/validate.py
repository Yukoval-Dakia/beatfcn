"""Request/response validation: JSON Schema plus the README's extra semantic checks.

Every function returns a list of issue dicts {check, code, severity, path, detail}; nothing raises on
the first problem. `check` is the README item number (0 = JSON Schema).

Coverage of README items:
 1 regions half-open, cover [0, audio_duration_us), no overlap; unique ids.            implemented
 2 events in audio range; tap end == onset; hold end > onset; mode/type/location.     implemented
 3 native_ref present; observed events equal the imported chart (when a Chart is given) and the osu
   sidecar re-exports byte-identically. Renderer timing/position parity is NOT checkable here.
 4 complete references: events inside span; completeness vs the source chart when given; a
   native_chart_ref is required so holds started before the span have occupancy context; sparse
   references never carry unselected evidence (evidence-level check).                 implemented
 5 occupancy index (active observed holds at each editable start, all observed per lane); locked
   lane conflicts reported.                                                            implemented
 6 response additions: head and whole occupancy inside one connected editable run; no collision
   with observed or other additions; observed ids untouched; allowed_types respected.  implemented
 7 control bands lo <= hi; mode-incompatible controls/types; obviously unsatisfiable combos. implemented
 8 Phira: line_id exists in the given scene (structural only); two-finger declared via
   max_active_contacts <= 2 needs hard_profile_ref. Coordinate-domain/scene transforms NOT checked.
 9 leakage_issues(): hidden targets absent from every conditioning field.             implemented
10 model_view_issues(): groups non-empty, strictly increasing, observed exact; quarantine surfaced.
11 cf_pair_issues(): strict counterfactual pair differs only in selected bits, attributes masked,
   no shared 40 ms cells. (The A/B pair builder itself is P2 and not implemented.)
12 controls_recompute_issues(): request controls equal a recomputation over the editable union
   after augmentation. Cache-key invalidation (H20/H24/projection) is NOT in P0.
"""
from __future__ import annotations
from pathlib import Path
import json
import math
from .canonical import MODE_TYPES, Event, LaneLoc, Chart, check_groups, lane_conflicts, exact_duplicates
from . import controls as C

SCHEMA_PATH = Path(__file__).resolve().parents[2] / 'design' / 'package_2026-10-22' / 'request.schema.json'
PHIRA_ONLY = ('flick_ratio', 'drag_ratio')
MANIA_ONLY = ('jack_ratio',)
BAND_FIELDS = tuple(C.BAND_EDGES)


def issue(check, code, path='', detail='', severity='error'):
    return {'check': check, 'code': code, 'severity': severity, 'path': path, 'detail': str(detail)}


def errors(issues):
    return [i for i in issues if i['severity'] == 'error']


def load_schema(path=SCHEMA_PATH):
    return json.loads(Path(path).read_text())


def schema_issues(req, schema=None):
    try:
        import jsonschema
    except ImportError:
        return [issue(0, 'jsonschema_unavailable', detail='pip install jsonschema (requirements-p0.txt)')]
    schema = schema or load_schema()
    v = jsonschema.Draft202012Validator(schema, format_checker=jsonschema.FormatChecker())
    return [issue(0, 'schema', '/'.join(map(str, e.absolute_path)), e.message)
            for e in sorted(v.iter_errors(req), key=lambda e: list(map(str, e.absolute_path)))]


def _ev(d):
    try:
        return Event.from_json(d)
    except (KeyError, TypeError, ValueError):
        return None


def regions_of(req, state=None):
    return [(r['start_us'], r['end_us']) for r in req.get('regions', []) if state is None or r['state'] == state]


def editable_runs(req):
    """Connected editable runs (adjacent editable regions merge)."""
    return C.union(regions_of(req, 'editable'))


# ---------------------------------------------------------------- checks 1-2
def _check_regions(req):
    out, D = [], req.get('audio_duration_us')
    regs = sorted(req.get('regions', []), key=lambda r: (r.get('start_us', 0), r.get('end_us', 0)))
    ids = [r.get('region_id') for r in regs]
    if len(set(ids)) != len(ids):
        out.append(issue(1, 'duplicate_region_id', 'regions', sorted({i for i in ids if ids.count(i) > 1})))
    cur = 0
    for r in regs:
        s, e = r.get('start_us'), r.get('end_us')
        if not (isinstance(s, int) and isinstance(e, int)):
            continue
        if s >= e:
            out.append(issue(1, 'empty_or_inverted_region', r.get('region_id'), f'{s}>={e}'))
        if s > cur:
            out.append(issue(1, 'region_gap', r.get('region_id'), f'[{cur},{s}) uncovered'))
        elif s < cur:
            out.append(issue(1, 'region_overlap', r.get('region_id'), f'starts {s} < previous end {cur}'))
        cur = max(cur, e)
    if isinstance(D, int) and cur != D:
        out.append(issue(1, 'regions_do_not_cover_audio', 'regions', f'cover ends {cur}, audio {D}'))
    for name, items, key in (('observed_events', req.get('observed_events', []), 'event_id'),
                             ('references', req.get('references', []), 'reference_id')):
        ids = [x.get(key) for x in items]
        if len(set(ids)) != len(ids):
            out.append(issue(1, f'duplicate_{key}', name, sorted({i for i in ids if ids.count(i) > 1})))
    for k, ref in enumerate(req.get('references', [])):
        ids = [x.get('event_id') for x in ref.get('events', [])]
        if len(set(ids)) != len(ids):
            out.append(issue(1, 'duplicate_event_id', f'references/{k}/events'))
    return out


def _check_event(path, d, mode, D):
    out = []
    t, on, end, loc = d.get('type'), d.get('onset_us'), d.get('end_us'), d.get('location', {})
    if not isinstance(on, int) or not isinstance(end, int):
        return [issue(2, 'bad_times', path)]
    if isinstance(D, int) and not (0 <= on < D and end <= D):
        out.append(issue(2, 'event_outside_audio', path, f'[{on},{end}) vs {D}'))
    if t == 'hold' and end <= on:
        out.append(issue(2, 'non_positive_hold', path, f'{on}..{end}'))
    if t != 'hold' and end != on:
        out.append(issue(2, 'non_hold_end_ne_onset', path, f'{on}..{end}'))
    if t not in MODE_TYPES.get(mode, ()):
        out.append(issue(2, 'type_not_in_mode', path, f'{t} in {mode}'))
    want = 'lane' if mode == 'mania_4k' else 'line'
    if loc.get('kind') != want:
        out.append(issue(2, 'location_kind_mismatch', path, f'{loc.get("kind")} in {mode}'))
    if d.get('fake', False) is not False:
        out.append(issue(2, 'fake_note_in_request', path))
    return out


# ---------------------------------------------------------------- checks 3-5
def occupancy_index(req):
    """Exact constraint index: per-lane (mania) / per-line (Phira) observed occupancies, plus the
    observed holds already active at the start of every editable region (README check 5)."""
    obs = [e for e in map(_ev, req.get('observed_events', [])) if e]
    per = {}
    for e in obs:
        k = e.location.lane if isinstance(e.location, LaneLoc) else e.location.line_id
        per.setdefault(k, []).append((e.onset_us, e.occupancy()[1], e.event_id))
    for v in per.values():
        v.sort()
    active = {r['region_id']: [e.event_id for e in obs if e.is_hold and e.onset_us < r['start_us'] < e.end_us]
              for r in req.get('regions', []) if r['state'] == 'editable'}
    return {'per_location': per, 'active_at_editable_start': active}


def _check_observed(req, chart: Chart | None):
    out, mode = [], req.get('mode')
    obs = [e for e in map(_ev, req.get('observed_events', [])) if e]
    if mode == 'mania_4k':
        for code, a, b in lane_conflicts(obs):
            out.append(issue(5, 'locked_lane_conflict', b, f'{code} with {a}'))
    else:
        for a, b in exact_duplicates(obs):
            out.append(issue(5, 'locked_duplicate', b, a))
    for e in obs:
        for s, t in regions_of(req, 'empty'):
            a, b = e.occupancy()
            if a < t and s < b:
                out.append(issue(7, 'locked_content_in_empty_region', e.event_id, f'[{s},{t})', 'warning'))
    if chart is not None:
        by_ref = {e.native_ref: e for e in chart.events}
        for e in obs:
            c = by_ref.get(e.native_ref)
            if c is None:
                out.append(issue(3, 'native_ref_unresolved', e.event_id, e.native_ref))
            elif (c.onset_us, c.end_us, c.type, c.location) != (e.onset_us, e.end_us, e.type, e.location):
                out.append(issue(3, 'observed_differs_from_import', e.event_id, f'{c} vs {e}'))
        if chart.source_format == 'osu_v14':
            from .osu_mania import export_osu
            if export_osu(chart, []) != b''.join(chart.sidecar['lines']):
                out.append(issue(3, 'sidecar_not_byte_identical', chart.sidecar.get('source_name')))
    return out


def _check_references(req, ref_charts):
    out, D = [], req.get('audio_duration_us')
    for k, r in enumerate(req.get('references', [])):
        p = f'references/{k}'
        s, e = r.get('span', {}).get('start_us'), r.get('span', {}).get('end_us')
        if not (isinstance(s, int) and isinstance(e, int)) or s >= e:
            out.append(issue(4, 'bad_reference_span', p, f'{s}..{e}'))
            continue
        if r.get('audio_id') == req.get('audio_id') and isinstance(D, int) and e > D:
            out.append(issue(4, 'reference_span_outside_audio', p))
        if r.get('mode') != req.get('mode'):
            out.append(issue(2, 'reference_mode_mismatch', p, f'{r.get("mode")} vs {req.get("mode")}'))
        for j, d in enumerate(r.get('events', [])):
            out += _check_event(f'{p}/events/{j}', d, r.get('mode'), D if r.get('audio_id') == req.get('audio_id') else None)
            if isinstance(d.get('onset_us'), int) and not s <= d['onset_us'] < e:
                out.append(issue(4, 'reference_event_outside_span', f'{p}/events/{j}'))
        if r.get('coverage') == 'complete':
            if not r.get('native_chart_ref'):
                out.append(issue(4, 'complete_reference_without_native_chart', p,
                                 'cannot prove occupancy of holds started before the span'))
            ch = (ref_charts or {}).get(r.get('native_chart_ref'))
            if ch is not None:
                have = {(x['onset_us'], x['end_us'], x['type'], json.dumps(x['location'], sort_keys=True))
                        for x in r.get('events', [])}
                for ev in ch.events:
                    sig = (ev.onset_us, ev.end_us, ev.type, json.dumps(ev.location.to_json(), sort_keys=True))
                    if s <= ev.onset_us < e and sig not in have:
                        out.append(issue(4, 'complete_reference_missing_event', p, ev.event_id))
    return out


def reference_evidence_issues(evidence, all_events=(), exclude_us=30_000, max_tokens=128):
    """Check selected/unselected evidence built by requests.build_reference (README 4, spec 3.1)."""
    out = []
    holds = [(e.onset_us, e.end_us) for e in all_events if e.is_hold]
    heads = [e.onset_us for e in all_events]
    for ev in evidence:
        p, (s, e) = ev['reference_id'], ev['span']
        if ev['coverage'] != 'complete' and ev['unselected_us']:
            out.append(issue(4, 'unselected_without_complete_coverage', p))
        if len(ev['selected']) > max_tokens or len(ev['unselected_us']) > max_tokens:
            out.append(issue(4, 'reference_token_budget', p))
        for t in ev['unselected_us']:
            if not s <= t < e:
                out.append(issue(4, 'unselected_outside_span', p, t))
            if any(abs(t - h) <= exclude_us for h in heads):
                out.append(issue(4, 'unselected_near_head', p, t))
            if any(a <= t < b for a, b in holds):
                out.append(issue(4, 'unselected_in_active_hold', p, t))
    return out


# ---------------------------------------------------------------- checks 7-8
def _check_controls(req):
    out, mode, c = [], req.get('mode'), req.get('controls', {})
    for k in BAND_FIELDS:
        b = c.get(k)
        if isinstance(b, list) and len(b) == 2 and b[0] > b[1]:
            out.append(issue(7, 'band_low_gt_high', f'controls/{k}', b))
    bad = PHIRA_ONLY if mode == 'mania_4k' else MANIA_ONLY
    for k in bad:
        if k in c:
            out.append(issue(7, 'unsupported_control_for_mode', f'controls/{k}', mode))
    at = c.get('allowed_types')
    if at is not None:
        extra = [t for t in at if t not in MODE_TYPES.get(mode, ())]
        if extra:
            out.append(issue(7, 'unsupported_type_for_mode', 'controls/allowed_types', extra))
        if 'hold' not in at:
            for k in ('hold_head_ratio', 'hold_occupancy_ratio'):
                if isinstance(c.get(k), list) and c[k][0] > 0:
                    out.append(issue(7, 'unsatisfiable_hold_control', f'controls/{k}', 'hold not allowed'))
        for k, t in (('flick_ratio', 'flick'), ('drag_ratio', 'drag')):
            if isinstance(c.get(k), list) and c[k][0] > 0 and t not in at:
                out.append(issue(7, 'unsatisfiable_type_control', f'controls/{k}', f'{t} not allowed'))
    if not regions_of(req, 'editable'):
        for k in ('density_notes_per_second', 'peak_notes_per_second'):
            if isinstance(c.get(k), list) and c[k][0] > 0:
                out.append(issue(7, 'unsatisfiable_no_editable_region', f'controls/{k}'))
    curve = c.get('intensity_curve', [])
    ts = [p.get('time_us') for p in curve]
    if any(b <= a for a, b in zip(ts, ts[1:])):
        out.append(issue(7, 'intensity_curve_not_increasing', 'controls/intensity_curve'))
    D = req.get('audio_duration_us')
    if isinstance(D, int) and any(isinstance(t, int) and t >= D for t in ts):
        out.append(issue(7, 'intensity_curve_outside_audio', 'controls/intensity_curve'))
    mac = c.get('max_active_contacts')
    if isinstance(mac, int) and mac <= 2 and not req.get('hard_profile_ref'):
        out.append(issue(8, 'two_finger_without_profile', 'hard_profile_ref'))
    return out


def _check_scene(req, scene_line_ids, extra_events=()):
    if req.get('mode') != 'phira':
        return []
    if scene_line_ids is None:
        return [issue(8, 'scene_unverified', 'scene_ref', 'no scene given; line_id existence not checked', 'warning')]
    out, ids = [], set(scene_line_ids)
    evs = [('observed_events', d) for d in req.get('observed_events', [])]
    evs += [(f'references/{r.get("reference_id")}', d) for r in req.get('references', []) for d in r.get('events', [])]
    evs += [('additions', d) for d in extra_events]
    for p, d in evs:
        lid = d.get('location', {}).get('line_id')
        if lid not in ids:
            out.append(issue(8, 'unknown_line_id', f'{p}/{d.get("event_id")}', lid))
        x = d.get('location', {}).get('native_x')
        if not isinstance(x, (int, float)) or not math.isfinite(x):
            out.append(issue(8, 'bad_native_x', f'{p}/{d.get("event_id")}', x))
    return out


def validate_request(req, schema=None, chart: Chart | None = None, ref_charts=None, scene_line_ids=None):
    """All request-level checks. `chart`: imported target chart (enables check 3);
    `ref_charts`: {native_chart_ref: Chart} (enables reference completeness); `scene_line_ids`: Phira line ids."""
    out = schema_issues(req, schema)
    if not isinstance(req, dict):
        return out
    out += _check_regions(req)
    for k, d in enumerate(req.get('observed_events', [])):
        out += _check_event(f'observed_events/{k}', d, req.get('mode'), req.get('audio_duration_us'))
        if not d.get('native_ref'):
            out.append(issue(3, 'missing_native_ref', f'observed_events/{k}'))
    out += _check_observed(req, chart)
    out += _check_references(req, ref_charts)
    out += _check_controls(req)
    out += _check_scene(req, scene_line_ids)
    return out


# ---------------------------------------------------------------- check 6
def validate_response(req, resp, scene_line_ids=None):
    out, mode, D = [], req.get('mode'), req.get('audio_duration_us')
    runs = editable_runs(req)
    obs = [e for e in map(_ev, req.get('observed_events', [])) if e]
    obs_ids = {e.event_id for e in obs}
    allowed = req.get('controls', {}).get('allowed_types') or MODE_TYPES.get(mode, ())
    adds = []
    for k, d in enumerate(resp.get('additions', [])):
        p = f'additions/{k}'
        out += [dict(i, check=6) for i in _check_event(p, d, mode, D)]
        e = _ev(d)
        if e is None:
            out.append(issue(6, 'bad_addition', p))
            continue
        if e.event_id in obs_ids:
            out.append(issue(6, 'addition_id_reuses_observed', p, e.event_id))
        if e.type not in allowed:
            out.append(issue(6, 'type_not_allowed', p, e.type))
        a, b = e.occupancy()
        if not any(s <= a and b <= t for s, t in runs):
            out.append(issue(6, 'addition_outside_editable', p, f'[{a},{b})'))
        adds.append(e)
    ids = [e.event_id for e in adds]
    if len(set(ids)) != len(ids):
        out.append(issue(6, 'duplicate_addition_id', 'additions'))
    if mode == 'mania_4k':
        for code, a, b in lane_conflicts(obs + adds):
            if a in obs_ids and b in obs_ids:
                continue
            out.append(issue(6, 'addition_lane_conflict', b, f'{code} with {a}'))
    else:
        for a, b in exact_duplicates(obs + adds):
            if not (a in obs_ids and b in obs_ids):
                out.append(issue(6, 'addition_duplicates_object', b, a))
    unchanged = resp.get('unchanged_observed_event_ids')
    if unchanged is not None and set(unchanged) != obs_ids:
        out.append(issue(6, 'observed_ids_not_preserved', 'unchanged_observed_event_ids',
                         sorted(obs_ids ^ set(unchanged))))
    out += [dict(i, check=8) for i in _check_scene({'mode': mode}, scene_line_ids, resp.get('additions', []))
            if i['code'] != 'scene_unverified'] if scene_line_ids is not None else []
    return out


# ---------------------------------------------------------------- check 9
def leakage_issues(req, hidden, evidence=(), training=True):
    """Hidden targets must not appear in any conditioning field (README 9, spec 2.3/7)."""
    out, hidden = [], list(hidden)
    hid_ids = {e.event_id for e in hidden}
    hid_sig = {e.signature(): e.event_id for e in hidden}
    cond = [('observed_events', d) for d in req.get('observed_events', [])]
    cond += [(f'references/{r.get("reference_id")}', d) for r in req.get('references', []) for d in r.get('events', [])]
    cond += [(f'evidence/{ev["reference_id"]}/selected', d) for ev in evidence for d in ev.get('selected', [])]
    cond += [(f'evidence/{ev["reference_id"]}/context', d) for ev in evidence for d in ev.get('context_holds', [])]
    for p, d in cond:
        e = _ev(d)
        if d.get('event_id') in hid_ids:
            out.append(issue(9, 'hidden_id_in_condition', p, d.get('event_id')))
        elif e is not None and e.signature() in hid_sig:
            out.append(issue(9, 'hidden_content_in_condition', p, hid_sig[e.signature()]))
    spans = [(r['span']['start_us'], r['span']['end_us']) for r in req.get('references', [])]
    spans += [tuple(ev['span']) for ev in evidence]
    runs = editable_runs(req)
    for e in hidden:
        a, b = e.occupancy()
        if C.overlaps(spans, a, b):
            out.append(issue(9, 'reference_span_overlaps_hidden_target', e.event_id))
        if not any(s <= a and b <= t for s, t in runs):
            out.append(issue(9, 'hidden_target_outside_editable', e.event_id, f'[{a},{b})'))
    for ev in evidence:
        hs = [e.onset_us for e in hidden]
        if any(abs(t - h) <= 30_000 for t in ev.get('unselected_us', []) for h in hs):
            out.append(issue(9, 'unselected_derived_from_hidden', ev['reference_id']))
    if training:
        for k, b in req.get('controls', {}).items():
            if k in BAND_FIELDS and not C.is_table_band(k, b):
                out.append(issue(9, 'control_not_binned', f'controls/{k}', b))
    for k in ('hidden', 'targets', 'target_count', 'true_count', 'candidates'):
        if k in req:
            out.append(issue(9, 'forbidden_request_key', k))
    return out


# ---------------------------------------------------------------- checks 10-12
def model_view_issues(view, observed_source=()):
    out = [issue(10, i.code, i.event_id or '', i.detail) for i in view.quarantine]
    out += [issue(10, 'observed_groups', '', m) for m in check_groups(view.observed_groups)]
    out += [issue(10, 'target_groups', '', m) for m in check_groups(view.target_groups)]
    src = {e.event_id: e for e in observed_source}
    for g in view.observed_groups:
        for e in g.events:
            if e.event_id in src and src[e.event_id] != e:
                out.append(issue(10, 'observed_altered', e.event_id))
    for g in view.target_groups:
        for e in g.events:
            if not g.copy_time and e.onset_us % 1000:
                out.append(issue(10, 'target_not_quantized', e.event_id))
    return out


def cf_pair_issues(a, b, cell_us=40_000):
    """Strict CF pair: lists of candidates {time_us, attrs, selected}; README 11 / spec 3.3."""
    out = []
    if len(a) != len(b):
        return [issue(11, 'candidate_count_differs', '', f'{len(a)} vs {len(b)}')]
    for i, (x, y) in enumerate(zip(a, b)):
        if {k: v for k, v in x.items() if k != 'selected'} != {k: v for k, v in y.items() if k != 'selected'}:
            out.append(issue(11, 'non_selected_field_differs', str(i)))
        if x.get('attrs') not in (None, 'MASK'):
            out.append(issue(11, 'attributes_not_masked', str(i)))
    cells = {}
    for x in a:
        cells.setdefault(x['time_us'] // cell_us, set()).add(bool(x['selected']))
    for c, v in cells.items():
        if len(v) > 1:
            out.append(issue(11, 'shared_selection_cell', str(c)))
    if sum(bool(x['selected']) for x in a) != sum(not x['selected'] for x in a):
        out.append(issue(11, 'unbalanced_selected_counts', ''))
    return out


def controls_recompute_issues(req, hidden):
    """README 12: request controls must equal recomputation over the editable union (missing allowed)."""
    obs = [e for e in map(_ev, req.get('observed_events', [])) if e]
    bands, _ = C.derive_controls(req['mode'], obs + list(hidden), regions_of(req, 'editable'))
    out = []
    for k, b in req.get('controls', {}).items():
        if k in BAND_FIELDS and bands.get(k) != b:
            out.append(issue(12, 'control_not_recomputed', f'controls/{k}', f'{b} vs derived {bands.get(k)}'))
    return out
