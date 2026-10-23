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
from dataclasses import dataclass, field, replace
import numpy as np
from ..events.canonical import Event, MODE_TYPES, LaneLoc
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
BLOCK_CELLS = 10  # whole-song stream block: 400 ms of song cells (FIRST_CANDIDATE v2.1 §3.2)
STREAM_BLOCK_US = BLOCK_CELLS * CELL_US
STREAM_STRIDE = 2  # the block encoder reads every other cell (80 ms) of the last two taps: static memory x4 smaller
MAX_STREAM_CELLS = 12000  # the song stream sees at most 8 min of song cells (a window centred on the core beyond)
# context perturbation (exposure bias, v2.1 §3.3): applied to the GENERATED context only, never to observed events
PERTURB = {'drop': 0.10, 'jitter_us': 8_000, 'jitter_max_us': 25_000, 'lane': 0.03, 'hold_to_tap': 0.02, 'insert': 0.05}
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
    # whole-song chart stream, one entry per 400 ms block of the song (v2.1 §3.2): occupancy grid
    # [NB, BLOCK_CELLS, 4 lanes|types, (onset, hold active)], coverage [NB, 3] (fraction before c0, editable,
    # empty) and log1p onset counts by type [NB, 4]; built from observed events and the generated prefix only
    blk_grid: np.ndarray
    blk_cov: np.ndarray
    blk_counts: np.ndarray
    groups: list = field(default_factory=list)  # targets, see target_group()
    lines: int = 0
    right: np.ndarray = None  # [G+1, 4] per outer row: s to next observed head, its lane/line, s to editable-run end
    provenance: dict = field(default_factory=dict)  # chart/request/core identity; never a model input


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


def build_core(sample, chart_events, c0: int, audio_duration_us: int, *, line_index=None, perturb=None) -> CoreExample:
    """perturb: np.random.Generator (training only) -> the generated context is perturbed (PERTURB) so the
    model learns to read its own imperfect output; masks, hold limits and the chart stream all see the same
    perturbed context, and targets that became illegal under it are skipped by the sampler."""
    req, view = sample.request, sample.model_view
    mode = req['mode']
    line_index = line_index_of(chart_events) if line_index is None else line_index
    regions = req['regions']
    c1, l0, l1 = c0 + CORE_US, c0 - HALO_US, c0 + CORE_US + HALO_US
    observed = [Event.from_json(e) for e in req['observed_events']]
    hidden = list(sample.hidden)
    runs = editable_runs(regions)
    # Earlier hidden targets = this request's previously generated events (onset before the core). The whole
    # prefix feeds the song stream; rows see those whose occupancy touches the window (a hold that started
    # before the halo but is still active counts).
    generated_all = [e for e in hidden if e.onset_us < c0]
    if perturb is not None and generated_all:
        generated_all = perturb_context(generated_all, perturb, runs, c0, mode)
    generated = [e for e in generated_all if max(e.occupancy()[1], e.onset_us + 1) > l0]

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
    bin_ok = np.zeros((CORE_BINS, 10), bool)
    for a, b in runs:
        a, b = max(a, c0), min(b, c1)
        if b > a:
            flat = bin_ok.reshape(-1)
            flat[(a - c0) // MS:(b - c0 + MS - 1) // MS] = True
    obs_heads = sorted({e.onset_us for e in observed if c0 <= e.onset_us < c1 and C.contains(runs, e.onset_us)})
    copy_list = [t - c0 for t in obs_heads]

    # whole-song chart stream from KNOWN information only: observed events and the generated prefix
    grid, cov, counts = song_blocks(observed, generated_all, regions, c0, audio_duration_us, mode)

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
                       bin_ok, np.asarray(copy_list, np.int64), grid, cov, counts, groups, len(line_index), right)


def song_blocks(observed, generated, regions, c0: int, duration_us: int, mode: str):
    """Per 400 ms block of the whole song: occupancy grid [NB, BLOCK_CELLS, 4, 2] (channel = lane for mania,
    type for Phira; plane 0 = onset in the cell, plane 1 = hold active in the cell), coverage [NB, 3]
    (fraction of the block before c0, editable fraction, empty fraction) and log1p onset counts by type
    [NB, 4]. Only observed events and the generated prefix enter (hidden targets never do)."""
    n_cells = -(-duration_us // CELL_US)
    nb = -(-n_cells // BLOCK_CELLS)
    grid = np.zeros((nb, BLOCK_CELLS, 4, 2), np.uint8)
    counts = np.zeros((nb, 4), np.float32)
    flat = grid.reshape(nb * BLOCK_CELLS, 4, 2)
    last = nb * BLOCK_CELLS - 1
    lanes = mode == 'mania_4k'
    for e in observed + generated:
        c = e.onset_us // CELL_US
        if not 0 <= c <= last:
            continue
        ti = TYPES.index(e.type)
        ch = e.location.lane if lanes and e.location.kind == 'lane' else ti
        flat[c, ch, 0] = 1
        counts[c // BLOCK_CELLS, ti] += 1
        if e.is_hold:
            c1 = min((e.end_us - 1) // CELL_US, last)
            if c1 > c:
                flat[c + 1:c1 + 1, ch, 1] = 1
    cov = np.zeros((nb, 3), np.float32)
    starts = np.arange(nb, dtype=np.int64) * STREAM_BLOCK_US
    ends = np.minimum(starts + STREAM_BLOCK_US, duration_us)
    length = np.maximum(ends - starts, 1).astype(np.float64)
    cov[:, 0] = np.clip((c0 - starts) / length, 0, 1)
    for r in regions:
        st = STATES.index(r['state'])
        if st == 1:  # complete: the default state
            continue
        ov = np.clip(np.minimum(ends, r['end_us']) - np.maximum(starts, r['start_us']), 0, None)
        cov[:, 1 if st == 0 else 2] += ov / length
    return grid, cov, np.log1p(counts)


def perturb_context(events, rng, runs, c0: int, mode: str):
    """Exposure-bias perturbation of the generated context (PERTURB): drops, Gaussian time jitter, lane
    swaps (mania), hold -> tap flips and inserted taps at random editable times before the core."""
    out = []
    for e in events:
        if rng.random() < PERTURB['drop']:
            continue
        j = int(np.clip(rng.normal(0, PERTURB['jitter_us']), -PERTURB['jitter_max_us'], PERTURB['jitter_max_us']))
        onset = max(0, e.onset_us + j)
        end = max(onset + 1, e.end_us + j) if e.is_hold else onset
        loc, typ = e.location, e.type
        if mode == 'mania_4k' and loc.kind == 'lane' and rng.random() < PERTURB['lane']:
            loc = LaneLoc(lane=int((loc.lane + rng.integers(1, 4)) % 4))
        if e.is_hold and rng.random() < PERTURB['hold_to_tap']:
            typ, end = 'tap', onset
        out.append(replace(e, onset_us=onset, end_us=end, type=typ, location=loc))
    pre = [(a, min(b, c0)) for a, b in runs if a < c0]
    total = sum(b - a for a, b in pre)
    if total > 0:
        for i in range(int(rng.binomial(len(events), PERTURB['insert']))):
            off = int(rng.integers(0, total))
            for a, b in pre:
                if off < b - a:
                    t = a + off
                    break
                off -= b - a
            src = events[int(rng.integers(len(events)))]
            loc = LaneLoc(lane=int(rng.integers(4))) if mode == 'mania_4k' else src.location
            out.append(Event(f'perturb:{i}', int(t), int(t), 'tap', loc))
    return sorted(out, key=lambda e: (e.onset_us, e.event_id))


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
        if prev >= 0 and t // MS <= prev // MS:  # the model places onsets on a 1 ms grid, strictly increasing
            bad.append('time_same_ms')
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
