"""Immutable canonical events, charts and the model view (spec section 4.1).

Storage layer: integer microseconds on the audio clock, exact native beat
rationals (fractions.Fraction), offset, format version and an opaque frozen
native sidecar. Nothing here rounds, clamps or repairs a human chart; problems
are recorded as Issue(severity='quarantine') and the caller must skip the chart.

Model view: hidden *new* targets are quantized to 1 ms (round half up), then
grouped by time. Observed events are never quantized. A target whose exact
onset equals an observed onset keeps that exact time (COPY_TIME). Quantization
that creates a same-lane collision, a duplicate Phira object or a non-positive
hold is quarantined, never fixed.

NOT handled here: Phira position transforms through moving/parent lines (only
line_id/side/native_x are recorded), release-gap re-press rules, per-core
256-group/512-event overflow splitting (P1 runtime), renderer parity.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from fractions import Fraction
from types import MappingProxyType
from typing import Any, Iterable, Mapping
import math

TYPES = ('tap', 'hold', 'flick', 'drag')
MODE_TYPES = {'mania_4k': ('tap', 'hold'), 'phira': TYPES}
SIDES = ('above', 'below')
MANIA_LANES = 4
QUANT_US = 1000


class UnsupportedFormat(ValueError):
    """Input format/version/mode that P0 does not parse. Never guessed."""


def round_half_up(x: Fraction | int) -> int:
    return math.floor(Fraction(x) + Fraction(1, 2))


def sec_to_us(sec: Fraction) -> int:
    return round_half_up(Fraction(sec) * 1_000_000)


def freeze(x: Any) -> Any:
    """Recursively turn dict/list into MappingProxyType/tuple (opaque sidecars)."""
    if isinstance(x, dict):
        return MappingProxyType({k: freeze(v) for k, v in x.items()})
    if isinstance(x, (list, tuple)):
        return tuple(freeze(v) for v in x)
    return x


@dataclass(frozen=True, slots=True)
class LaneLoc:
    lane: int
    kind: str = 'lane'

    def __post_init__(self):
        if type(self.lane) is not int or not 0 <= self.lane < MANIA_LANES:
            raise ValueError(f'lane must be int 0..{MANIA_LANES - 1}, got {self.lane!r}')

    def to_json(self):
        return {'kind': 'lane', 'lane': self.lane}

    def sort_key(self):
        return (self.lane,)


def _line_key(line_id: str):
    return (0, int(line_id), '') if line_id.lstrip('-').isdigit() else (1, 0, line_id)


@dataclass(frozen=True, slots=True)
class LineLoc:
    line_id: str
    side: str
    native_x: float
    kind: str = 'line'

    def __post_init__(self):
        if not isinstance(self.line_id, str) or not self.line_id:
            raise ValueError('line_id must be a non-empty string')
        if self.side not in SIDES:
            raise ValueError(f'side must be one of {SIDES}')
        if not isinstance(self.native_x, float) or not math.isfinite(self.native_x):
            raise ValueError('native_x must be a finite float')

    def to_json(self):
        return {'kind': 'line', 'line_id': self.line_id, 'side': self.side, 'native_x': self.native_x}

    def sort_key(self):
        return (_line_key(self.line_id), SIDES.index(self.side), self.native_x)


def location_from_json(d: Mapping) -> LaneLoc | LineLoc:
    if d.get('kind') == 'lane':
        return LaneLoc(d['lane'])
    if d.get('kind') == 'line':
        return LineLoc(d['line_id'], d['side'], float(d['native_x']))
    raise ValueError(f'unknown location kind {d.get("kind")!r}')


@dataclass(frozen=True, slots=True)
class Event:
    """Canonical event. end_us == onset_us for non-holds; hold occupancy is [onset_us, end_us)."""
    event_id: str
    onset_us: int
    end_us: int
    type: str
    location: LaneLoc | LineLoc
    native_ref: str | None = None

    def __post_init__(self):
        if type(self.onset_us) is not int or type(self.end_us) is not int:
            raise TypeError('event times must be int microseconds')
        if self.type not in TYPES:
            raise ValueError(f'unknown event type {self.type!r}')
        if not isinstance(self.location, (LaneLoc, LineLoc)):
            raise TypeError('location must be LaneLoc or LineLoc')

    @property
    def is_hold(self):
        return self.type == 'hold'

    def occupancy(self) -> tuple[int, int]:
        """Half-open occupancy; a tap occupies only its head instant [t, t+1us)."""
        return (self.onset_us, self.end_us if self.is_hold else self.onset_us + 1)

    def signature(self):
        """Content identity used for leak/duplicate checks (ignores ids)."""
        return (self.onset_us, self.end_us, self.type, tuple(sorted(self.location.to_json().items())))

    def to_json(self, locked: bool | None = None) -> dict:
        d = {'event_id': self.event_id, 'onset_us': self.onset_us, 'end_us': self.end_us,
             'type': self.type, 'location': self.location.to_json(), 'native_ref': self.native_ref, 'fake': False}
        if locked is not None:
            d['locked'] = locked
        return d

    @staticmethod
    def from_json(d: Mapping) -> 'Event':
        return Event(d['event_id'], d['onset_us'], d['end_us'], d['type'], location_from_json(d['location']), d.get('native_ref'))


TYPE_ORDER = {t: i for i, t in enumerate(TYPES)}


def in_group_key(e: Event):
    """Deterministic serialization order: mania by lane; Phira by line/side/x/type."""
    return (*e.location.sort_key(), TYPE_ORDER[e.type], e.end_us, e.event_id)


@dataclass(frozen=True, slots=True)
class Issue:
    code: str
    severity: str  # 'quarantine' | 'warning'
    detail: str
    event_id: str | None = None

    def to_json(self):
        return {'code': self.code, 'severity': self.severity, 'detail': self.detail, 'event_id': self.event_id}


@dataclass(frozen=True, slots=True)
class BpmSegment:
    start_beat: Fraction
    bpm: Fraction
    start_sec: Fraction  # chart clock (before offset)


@dataclass(frozen=True, slots=True)
class TimingMap:
    """Piecewise-constant BPM map; audio_sec = chart_sec(beat) + offset_sec.

    Beats before the first segment extend the first BPM backwards (osu rule);
    importers that consider that undefined must quarantine before building this.
    """
    segments: tuple[BpmSegment, ...]
    offset_sec: Fraction = Fraction(0)

    @staticmethod
    def from_bpm_points(points: Iterable[tuple[Fraction, Fraction]], offset_sec=Fraction(0), first_sec=Fraction(0)) -> 'TimingMap':
        segs, sec, prev = [], Fraction(first_sec), None
        for beat, bpm in points:
            beat, bpm = Fraction(beat), Fraction(bpm)
            if bpm <= 0:
                raise ValueError('BPM must be positive')
            if prev is not None:
                if beat < prev.start_beat:
                    raise ValueError('BPM points must be sorted by beat')
                sec = prev.start_sec + (beat - prev.start_beat) * 60 / prev.bpm
            prev = BpmSegment(beat, bpm, sec)
            segs.append(prev)
        if not segs:
            raise ValueError('empty timing map')
        return TimingMap(tuple(segs), Fraction(offset_sec))

    def chart_sec(self, beat: Fraction) -> Fraction:
        beat = Fraction(beat)
        seg = self.segments[0]
        for s in self.segments:
            if s.start_beat <= beat:
                seg = s
            else:
                break
        return seg.start_sec + (beat - seg.start_beat) * 60 / seg.bpm

    def audio_sec(self, beat: Fraction) -> Fraction:
        return self.chart_sec(beat) + self.offset_sec

    def audio_us(self, beat: Fraction) -> int:
        return sec_to_us(self.audio_sec(beat))


@dataclass(frozen=True)
class Chart:
    """Imported chart. `events` are playable only; fakes/decor stay in `fakes` + sidecar."""
    mode: str
    source_format: str           # 'osu_v14' | 'phira_rpe' | 'phira_official_v3'
    format_version: str
    events: tuple[Event, ...]
    timing: TimingMap | None
    offset_sec: Fraction
    native_beats: Mapping[str, tuple[Fraction, Fraction]]
    sidecar: Mapping[str, Any]
    fakes: tuple[Event, ...] = ()
    issues: tuple[Issue, ...] = ()
    source_sha256: str = ''

    @property
    def quarantined(self) -> bool:
        return any(i.severity == 'quarantine' for i in self.issues)

    def quarantine_reasons(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == 'quarantine']

    def by_id(self) -> dict[str, Event]:
        return {e.event_id: e for e in self.events}


def lane_conflicts(events: Iterable[Event]) -> list[tuple[str, str, str]]:
    """Mania same-lane conflicts: identical heads, or a head inside another hold's [onset, end)."""
    out, by_lane = [], {}
    for e in events:
        if isinstance(e.location, LaneLoc):
            by_lane.setdefault(e.location.lane, []).append(e)
    for evs in by_lane.values():
        evs.sort(key=lambda e: (e.onset_us, e.event_id))
        for i, a in enumerate(evs):
            for b in evs[i + 1:]:
                if b.onset_us == a.onset_us:
                    out.append(('same_lane_same_onset', a.event_id, b.event_id))
                elif a.is_hold and b.onset_us < a.end_us:
                    out.append(('head_inside_hold', a.event_id, b.event_id))
                else:
                    # sorted by onset: later b cannot hit a once b.onset >= a's occupancy end
                    if b.onset_us >= (a.end_us if a.is_hold else a.onset_us):
                        break
    return out


MAX_SOURCE_STACK = 4  # DATA_RULES v4: identical stacked notes (hit-sound weight) are kept up to this count


def oversized_stacks(events: Iterable[Event], max_stack: int = MAX_SOURCE_STACK) -> list[tuple[str, str]]:
    """(first, offending) pairs for every identical-signature stack deeper than max_stack."""
    from collections import defaultdict
    groups = defaultdict(list)
    for e in events:
        groups[e.signature()].append(e.event_id)
    return [(ids[0], x) for ids in groups.values() if len(ids) > max_stack for x in ids[max_stack:]]


def exact_duplicates(events: Iterable[Event]) -> list[tuple[str, str]]:
    seen, out = {}, []
    for e in events:
        k = e.signature()
        if k in seen:
            out.append((seen[k], e.event_id))
        else:
            seen[k] = e.event_id
    return out


# ---------------------------------------------------------------- model view
def quantize_us(us: int, step: int = QUANT_US) -> int:
    return ((us + step // 2) // step) * step


@dataclass(frozen=True, slots=True)
class Group:
    time_us: int
    events: tuple[Event, ...]
    copy_time: bool = False


@dataclass(frozen=True)
class ModelView:
    mode: str
    observed_groups: tuple[Group, ...]
    target_groups: tuple[Group, ...]
    quarantine: tuple[Issue, ...]
    quantized_from: Mapping[str, tuple[int, int]] = field(default_factory=lambda: MappingProxyType({}))

    @property
    def ok(self):
        return not self.quarantine


def group_events(events: Iterable[Event], copy_times: frozenset[int] = frozenset()) -> tuple[Group, ...]:
    by_t = {}
    for e in events:
        by_t.setdefault(e.onset_us, []).append(e)
    return tuple(Group(t, tuple(sorted(by_t[t], key=in_group_key)), t in copy_times) for t in sorted(by_t))


def check_groups(groups: Iterable[Group]) -> list[str]:
    """Spec 4.1/README check 10: groups non-empty, strictly increasing, members share the group time, ordered."""
    errs, prev = [], None
    for i, g in enumerate(groups):
        if not g.events:
            errs.append(f'group {i} empty')
        if prev is not None and g.time_us <= prev:
            errs.append(f'group {i} time {g.time_us} not > {prev}')
        if any(e.onset_us != g.time_us for e in g.events):
            errs.append(f'group {i} member onset != group time')
        if list(g.events) != sorted(g.events, key=in_group_key):
            errs.append(f'group {i} not in canonical order')
        prev = g.time_us
    return errs


def build_model_view(mode: str, observed: Iterable[Event], hidden: Iterable[Event]) -> ModelView:
    """Quantize hidden targets to 1 ms (COPY_TIME for exact observed onsets), then group.

    Returns quarantine issues instead of altering anything. Observed events are passed through exactly.
    """
    observed, hidden = tuple(observed), tuple(hidden)
    anchor_times = frozenset(e.onset_us for e in observed)
    issues, qmap, targets = [], {}, []
    for e in hidden:
        on = e.onset_us if e.onset_us in anchor_times else quantize_us(e.onset_us)
        end = quantize_us(e.end_us) if e.is_hold else on
        if e.is_hold and end <= on:
            issues.append(Issue('non_positive_hold_after_quantization', 'quarantine',
                                f'{e.onset_us}->{on}, {e.end_us}->{end}', e.event_id))
            continue
        if (on, end) != (e.onset_us, e.end_us):
            qmap[e.event_id] = (e.onset_us, e.end_us)
        targets.append(Event(e.event_id, on, end, e.type, e.location, e.native_ref))
    # source conflicts are reported even if quantization would hide them; new ones are attributed to quantization
    if mode == 'mania_4k':
        pre = [(c, a, b) for c, a, b in lane_conflicts(observed + hidden)]
        post = [(c, a, b) for c, a, b in lane_conflicts(observed + tuple(targets)) if (a, b) not in {x[1:] for x in pre}]
        for src, lst in (('in_source', pre), ('after_quantization', post)):
            issues += [Issue(f'lane_collision_{src}', 'quarantine', f'{c}: {a} vs {b}', b) for c, a, b in lst]
    else:
        # identical stacks up to MAX_SOURCE_STACK are intentional (DATA_RULES v4); deeper stacks, and events
        # that quantization merged although their source signatures differed, are quarantined
        pre = oversized_stacks(observed + hidden)
        sig = {e.event_id: e.signature() for e in observed + hidden}
        post = [(a, b) for a, b in exact_duplicates(observed + tuple(targets)) if sig[a] != sig[b]]
        for src, lst in (('in_source', pre), ('after_quantization', post)):
            issues += [Issue(f'duplicate_object_{src}', 'quarantine', f'{a} vs {b}', b) for a, b in lst]
    allowed = MODE_TYPES[mode]
    for e in observed + tuple(targets):
        if e.type not in allowed:
            issues.append(Issue('type_not_in_mode', 'quarantine', f'{e.type} in {mode}', e.event_id))
    tg = group_events(targets, anchor_times)
    return ModelView(mode, group_events(observed), tg, tuple(issues), MappingProxyType(qmap))
