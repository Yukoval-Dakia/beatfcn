"""TrainingSample -> one generation-core example (FIRST_CANDIDATE §2, design §4–5, §7).

A core owns onsets in [c0, c0+8 s). The model reads 16 s of fine audio around it
(4 s halo each side), a whole-song 2 s summary, the request's known events,
reference evidence and controls. Hidden targets of EARLIER cores in the same
gap enter as already-generated context (never as human evidence); hidden
targets at or after the core end never enter any input. Everything here is
numpy; no labels leak into the prefix arrays (tests/test_model_features.py).

Not handled in P1: remote retrieval segments, Phira scene geometry (lines are
indexed, not transformed), timing-map input (DATA_RULES §3), hard 2-finger
profiles, structured-missing tasks.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from ..events.canonical import Event, MODE_TYPES
from ..events import controls as C

MS = 1000
SEC = 1_000_000
CELL_US = 40_000
BIN_US = 10_000
CORE_US = 8 * SEC
HALO_US = 4 * SEC
LOCAL_CELLS = (CORE_US + 2 * HALO_US) // CELL_US  # 400
CORE_BINS = CORE_US // BIN_US  # 800
FRAMES_PER_CELL = 4
BLOCK_US = 2 * SEC
MAX_END_BLOCKS = 16
MAX_KNOWN = 512
MAX_REF = 256
MAX_GROUPS = 256
MAX_LINES = 64
MAX_SUMMARY = 600
MODES = ('mania_4k', 'phira')
TYPES = ('tap', 'hold', 'flick', 'drag')
SIDES = ('above', 'below')
STATES = ('editable', 'complete', 'empty')
CONTROL_FIELDS = tuple(sorted(C.BAND_EDGES))
X_HALF_WIDTH = 675.0  # Phira native_x half canvas (RPE convention); clipped to [-1, 1] after scaling


class SkipExample(ValueError):
    """Example exceeds P1 resource bounds; caller samples another core (never crops labels)."""


def _loc_fields(e: Event, line_index: dict):
    loc = e.location
    if loc.kind == 'lane':
        return loc.lane, -1, -1, 0.0
    li = line_index.get(loc.line_id)
    if li is None:
        raise SkipExample(f'line {loc.line_id} beyond {MAX_LINES} indexed lines')
    return -1, li, SIDES.index(loc.side), float(np.clip(loc.native_x / X_HALF_WIDTH, -1, 1))


def event_rows(events, c0: int, line_index: dict, flags) -> np.ndarray:
    """[N, 9]: rel_time_s, type, lane, line, side, x, hold_dur_s, flag_a, flag_b."""
    out = np.zeros((len(events), 9), np.float32)
    for i, e in enumerate(events):
        lane, line, side, x = _loc_fields(e, line_index)
        out[i] = ((e.onset_us - c0) / SEC, TYPES.index(e.type), lane, line, side, x,
                  (e.end_us - e.onset_us) / SEC, *flags(e))
    return out


def state_at(regions, t: int) -> int:
    for r in regions:
        if r['start_us'] <= t < r['end_us']:
            return STATES.index(r['state'])
    return STATES.index('complete')


def editable_runs(regions):
    return [(r['start_us'], r['end_us']) for r in regions if r['state'] == 'editable']


def line_index_of(chart_events) -> dict:
    ids = sorted({e.location.line_id for e in chart_events if e.location.kind == 'line'}, key=lambda s: (len(s), s))
    return {lid: i for i, lid in enumerate(ids[:MAX_LINES])}


@dataclass
class CoreExample:
    mode: int
    c0_us: int
    song_cell0: int  # song cell index of local cell 0 (= (c0 - 4 s) / 40 ms, may be negative)
    known: np.ndarray  # [K, 9] flags: (is_generated, is_future_of_core)
    refs: np.ndarray  # [R, 9] flags: (is_selected, 0); unselected rows have type=-1
    controls: np.ndarray  # [len(CONTROL_FIELDS)] band index or -1 missing
    allowed_types: np.ndarray  # [4] bool
    cell_state: np.ndarray  # [LOCAL_CELLS] region state per 40 ms cell (state at cell start)
    bin_ok: np.ndarray  # [CORE_BINS, 10] bool: ms inside each 10 ms bin where a NEW head is allowed
    copy_times: np.ndarray  # [A] rel us (int64) of observed heads in editable core time, usable by COPY_TIME
    summary_stats: np.ndarray  # [S, 6] per 2 s bucket: log1p heads by type (4), editable frac, empty frac
    groups: list = field(default_factory=list)  # targets, see target_group()
    lines: int = 0
    right: np.ndarray = None  # [G+1, 4] per outer row: s to next observed head, its lane/line, s to editable-run end


def target_group(g, c0, line_index, copy_list, hold_limit) -> dict:
    t = g.time_us - c0
    notes = []
    for e in g.events:
        lane, line, side, x = _loc_fields(e, line_index)
        n = {'type': TYPES.index(e.type), 'lane': lane, 'line': line, 'side': side, 'x': x}
        if e.is_hold:
            lo, hi = hold_limit(e)
            n['end_rel_us'] = e.end_us - g.time_us
            n['end_ok_us'] = (lo - g.time_us, hi - g.time_us)  # allowed (exclusive hi) relative to onset
        notes.append(n)
    copy = copy_list.index(t) if g.copy_time and t in copy_list else -1
    return {'t_rel_us': t, 'bin': t // BIN_US, 'residual': t % BIN_US // MS, 'copy': copy, 'notes': notes}


def build_core(sample, chart_events, c0: int, audio_duration_us: int, *, line_index=None) -> CoreExample:
    req, view = sample.request, sample.model_view
    mode = req['mode']
    line_index = line_index_of(chart_events) if line_index is None else line_index
    regions = req['regions']
    c1, l0, l1 = c0 + CORE_US, c0 - HALO_US, c0 + CORE_US + HALO_US
    observed = [Event.from_json(e) for e in req['observed_events']]
    hidden = list(sample.hidden)
    # Earlier hidden targets = this request's previously generated events: onset before the core and
    # occupancy touching the window (a hold that started before the halo but is still active counts).
    generated = [e for e in hidden if e.onset_us < c0 and max(e.occupancy()[1], e.onset_us + 1) > l0]

    def touches(e):
        a, b = e.occupancy()
        return e.onset_us < l1 and max(b, e.onset_us + 1) > l0
    known_obs = [e for e in observed if touches(e)]
    if len(known_obs) + len(generated) > MAX_KNOWN:
        raise SkipExample('known events exceed bound')
    known = np.concatenate([
        event_rows(known_obs, c0, line_index, lambda e: (0.0, float(e.onset_us >= c0))),
        event_rows(generated, c0, line_index, lambda e: (1.0, 0.0))]) if (known_obs or generated) else np.zeros((0, 9), np.float32)

    refs = []
    for ref in sample.evidence:
        sel = [Event.from_json(e) for e in ref['selected']]
        refs.append(event_rows(sel, c0, line_index, lambda e: (1.0, 0.0)))
        un = np.zeros((len(ref['unselected_us']), 9), np.float32)
        if len(un):
            un[:, 0] = (np.asarray(ref['unselected_us']) - c0) / SEC
            un[:, 1:6] = -1
        refs.append(un)
    refs = np.concatenate(refs) if refs else np.zeros((0, 9), np.float32)
    if len(refs) > MAX_REF:
        raise SkipExample('reference evidence exceeds bound')

    ctrl = np.full(len(CONTROL_FIELDS), -1, np.int64)
    for i, f in enumerate(CONTROL_FIELDS):
        band = req['controls'].get(f)
        if band is not None:
            edges = C.BAND_EDGES[f]
            ctrl[i] = int(np.searchsorted(edges, band[0], side='right') - 1)
    allowed = np.array([t in req['controls'].get('allowed_types', MODE_TYPES[mode]) for t in TYPES])

    cell_state = np.array([state_at(regions, l0 + k * CELL_US) for k in range(LOCAL_CELLS)], np.int64)
    runs = editable_runs(regions)
    bin_ok = np.zeros((CORE_BINS, 10), bool)
    for a, b in runs:
        a, b = max(a, c0), min(b, c1)
        if b > a:
            flat = bin_ok.reshape(-1)
            flat[(a - c0) // MS:(b - c0 + MS - 1) // MS] = True
    obs_heads = sorted({e.onset_us for e in observed if c0 <= e.onset_us < c1 and C.contains(runs, e.onset_us)})
    copy_list = [t - c0 for t in obs_heads]

    # 2 s summary over the whole song from KNOWN information only (observed heads, region states).
    nb = -(-audio_duration_us // BLOCK_US)
    stats = np.zeros((nb, 6), np.float32)
    for e in observed:
        stats[min(e.onset_us // BLOCK_US, nb - 1), TYPES.index(e.type)] += 1
    stats[:, :4] = np.log1p(stats[:, :4])
    for k in range(nb):
        s, t_end = k * BLOCK_US, min((k + 1) * BLOCK_US, audio_duration_us)
        inter = lambda st: C.length(C.intersect([(s, t_end)], [(r['start_us'], r['end_us']) for r in regions if r['state'] == st]))
        stats[k, 4] = inter('editable') / max(t_end - s, 1)
        stats[k, 5] = inter('empty') / max(t_end - s, 1)

    # Hold end limits: the editable run, and (mania) the next KNOWN head on the same lane (observed events
    # and the generated prefix; never hidden future targets, which generation does not have). Phira holds
    # on one line may overlap at different x, so only the run applies there.
    occupied = [e for e in observed + generated if e.location.kind == 'lane']

    def hold_limit(e: Event):
        run = next(((a, b) for a, b in runs if a <= e.onset_us < b), (e.onset_us, e.onset_us + 1))
        hi = run[1]
        for o in occupied:
            if e.location.kind == 'lane' and o.event_id != e.event_id and o.location == e.location and o.onset_us > e.onset_us:
                hi = min(hi, o.onset_us)
        return e.onset_us + 1, hi

    targets = [g for g in view.target_groups if c0 <= g.time_us < c1]
    if len(targets) > MAX_GROUPS:
        raise SkipExample('groups exceed bound')
    groups = [target_group(g, c0, line_index, copy_list, hold_limit) for g in targets]
    # Lanes busy at each group time (holds strictly active): observed, generated and earlier targets.
    holds = [e for e in observed + generated if e.is_hold and e.location.kind == 'lane']
    earlier = []
    for g, tg in zip(targets, groups):
        busy = np.zeros(4, bool)
        for h in holds + earlier:
            if h.onset_us < g.time_us < h.end_us:
                busy[h.location.lane] = True
        tg['busy_lanes'] = busy
        earlier += [e for e in g.events if e.is_hold and e.location.kind == 'lane']
    # Right-side lookahead (FIRST_CANDIDATE §2.4), only from observed heads and regions.
    heads = [(e.onset_us, e) for e in sorted(observed, key=lambda e: (e.onset_us, e.event_id))]
    times = [c0 - 1] + [g.time_us for g in targets]

    def right_of(t):
        nxt = next((e for u, e in heads if u > t), None)
        run_end = next((b for a, b in runs if a <= t < b), next((b for a, b in runs if a > t), t))
        dt = (nxt.onset_us - t) / SEC if nxt else 60.0
        lane, line, _, _ = _loc_fields(nxt, line_index) if nxt else (-1, -1, 0, 0.0)
        return min(dt, 60.0), lane, line, min((run_end - t) / SEC, 60.0)
    right = np.array([right_of(t) for t in times], np.float32)
    for tg, g in zip(groups, targets):
        tg['right_new'] = np.array(right_of(g.time_us - 1), np.float32)
    return CoreExample(MODES.index(mode), c0, (c0 - HALO_US) // CELL_US, known, refs, ctrl, allowed, cell_state,
                       bin_ok, np.asarray(copy_list, np.int64), stats, groups, len(line_index), right)


def choose_core(sample, rng, audio_duration_us: int) -> int:
    """Core start (multiple of 40 ms) whose span intersects the editable union."""
    runs = editable_runs(sample.request['regions'])
    total = C.length(runs)
    if not total:
        raise SkipExample('no editable time')
    off = int(rng.integers(0, total))
    for a, b in runs:
        if off < b - a:
            u = a + off
            break
        off -= b - a
    c0 = u - int(rng.integers(0, CORE_US // CELL_US)) * CELL_US
    c0 = max(0, min(c0, max(audio_duration_us - CELL_US, 0)))
    return c0 // CELL_US * CELL_US


def check_targets(ex: CoreExample) -> list[str]:
    """Same legality semantics as the model's masks (p1.py); returns reason codes, [] if all legal."""
    bad = []
    mode = MODES[ex.mode]
    mode_types = {0, 1} if mode == 'mania_4k' else {0, 1, 2, 3}
    prev = -1
    for g in ex.groups:
        t = g['t_rel_us']
        if t <= prev:
            bad.append('time_not_increasing')
        prev = t
        if g['copy'] >= 0:
            if int(ex.copy_times[g['copy']]) != t:
                bad.append('copy_mismatch')
        elif not ex.bin_ok[g['bin'], g['residual']]:
            bad.append('time_not_editable')
        used = g['busy_lanes'].copy()
        onset = ex.c0_us + t
        for n in g['notes']:
            if n['type'] not in mode_types or not ex.allowed_types[n['type']]:
                bad.append('type_not_allowed')
            if mode == 'mania_4k':
                if used[n['lane']]:
                    bad.append('lane_busy_or_repeated')
                used[n['lane']] = True
            elif not 0 <= n['line'] < max(ex.lines, 1):
                bad.append('line_out_of_range')
            if 'end_rel_us' in n:
                end = onset + n['end_rel_us']
                lo, hi = (onset + n['end_ok_us'][0], onset + n['end_ok_us'][1])
                end_ms_point = end // MS * MS
                if not (lo <= end <= hi):
                    bad.append('hold_end_outside_limits')
                elif end_ms_point < lo:
                    bad.append('hold_shorter_than_1ms')
                if (end // BLOCK_US) - (onset // BLOCK_US) >= MAX_END_BLOCKS:
                    bad.append('hold_beyond_end_blocks')
    return sorted(set(bad))
