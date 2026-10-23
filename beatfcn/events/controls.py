"""Derived controls (spec section 8) and half-open integer interval helpers.

Scope is the union E of the request's editable regions, computed AFTER every augmentation:
- density_notes_per_second: heads (observed + kept hidden) with onset in E / |E| seconds.
- peak_notes_per_second: max over heads h in E of #heads in E with onset in [h, h + 1 s); needs |E| >= 1 s.
- hold_head_ratio: hold heads / all heads in E.
- hold_occupancy_ratio: |union of hold occupancies [onset, end) intersect E| / |E|, holds starting
  outside E included.
- chord_ratio: exact-onset groups in E with >= 2 heads / groups in E.
- jack_ratio (mania only): consecutive onset groups in E sharing a lane / transitions.
- flick_ratio, drag_ratio (Phira only): heads of that type / heads in E.
Values are emitted only as fixed bins [lo, hi) (last ratio bin closed at 1.0) so an exact local count
is never a conditioning input. A value outside the bin table, or undefined (empty E, no heads), is
left missing rather than clamped. difficulty_label, intensity_curve, reference_strength and
max_active_contacts are not derived here.
"""
from __future__ import annotations
from bisect import bisect_left
from typing import Iterable

Interval = tuple[int, int]
US = 1_000_000

RATIO_EDGES = (0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0)
BAND_EDGES = {
    'density_notes_per_second': (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0, 24.0, 32.0, 48.0, 64.0),
    'peak_notes_per_second': (0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0, 24.0, 32.0, 48.0, 64.0, 96.0),
    'hold_head_ratio': RATIO_EDGES, 'hold_occupancy_ratio': RATIO_EDGES, 'chord_ratio': RATIO_EDGES,
    'jack_ratio': RATIO_EDGES, 'flick_ratio': RATIO_EDGES, 'drag_ratio': RATIO_EDGES,
}
MODE_FIELDS = {
    'mania_4k': ('density_notes_per_second', 'peak_notes_per_second', 'hold_head_ratio', 'hold_occupancy_ratio',
                 'chord_ratio', 'jack_ratio'),
    'phira': ('density_notes_per_second', 'peak_notes_per_second', 'hold_head_ratio', 'hold_occupancy_ratio',
              'chord_ratio', 'flick_ratio', 'drag_ratio'),
}


def union(ivs: Iterable[Interval]) -> list[Interval]:
    out = []
    for s, e in sorted(i for i in ivs if i[1] > i[0]):
        if out and s <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def intersect(a: Iterable[Interval], b: Iterable[Interval]) -> list[Interval]:
    a, b, out, i, j = union(a), union(b), [], 0, 0
    while i < len(a) and j < len(b):
        s, e = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if s < e:
            out.append((s, e))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return out


def subtract(a: Iterable[Interval], b: Iterable[Interval]) -> list[Interval]:
    out, b = [], union(b)
    for s, e in union(a):
        cur = s
        for bs, be in b:
            if be <= cur or bs >= e:
                continue
            if bs > cur:
                out.append((cur, bs))
            cur = max(cur, be)
        if cur < e:
            out.append((cur, e))
    return out


def length(ivs: Iterable[Interval]) -> int:
    return sum(e - s for s, e in union(ivs))


def contains(ivs: Iterable[Interval], t: int) -> bool:
    return any(s <= t < e for s, e in ivs)


def overlaps(ivs: Iterable[Interval], s: int, e: int) -> bool:
    return any(a < e and s < b for a, b in ivs)


def band_of(field: str, v: float | None):
    if v is None:
        return None
    edges = BAND_EDGES[field]
    for lo, hi in zip(edges, edges[1:]):
        if lo <= v < hi or (hi == edges[-1] == 1.0 and v == 1.0):
            return [lo, hi]
    return None


def is_table_band(field: str, band) -> bool:
    edges = BAND_EDGES.get(field)
    return edges is not None and isinstance(band, list) and len(band) == 2 and \
        any([lo, hi] == band for lo, hi in zip(edges, edges[1:]))


def raw_controls(mode: str, events, editable: Iterable[Interval]) -> dict:
    E = union(editable)
    L = length(E)
    fields = MODE_FIELDS[mode]
    out = {k: None for k in fields}
    if L <= 0:
        return out
    heads = sorted((e for e in events if contains(E, e.onset_us)), key=lambda e: e.onset_us)
    out['density_notes_per_second'] = len(heads) * US / L
    if L >= US and heads:
        t = [h.onset_us for h in heads]
        out['peak_notes_per_second'] = float(max(bisect_left(t, x + US) - bisect_left(t, x) for x in t))
    if heads:
        out['hold_head_ratio'] = sum(h.is_hold for h in heads) / len(heads)
    out['hold_occupancy_ratio'] = length(intersect([(e.onset_us, e.end_us) for e in events if e.is_hold], E)) / L
    groups = {}
    for h in heads:
        groups.setdefault(h.onset_us, []).append(h)
    gs = [groups[k] for k in sorted(groups)]
    if gs:
        out['chord_ratio'] = sum(len(g) >= 2 for g in gs) / len(gs)
    if mode == 'mania_4k' and len(gs) >= 2:
        lanes = [{h.location.lane for h in g} for g in gs]
        out['jack_ratio'] = sum(bool(a & b) for a, b in zip(lanes, lanes[1:])) / (len(gs) - 1)
    if mode == 'phira' and heads:
        out['flick_ratio'] = sum(h.type == 'flick' for h in heads) / len(heads)
        out['drag_ratio'] = sum(h.type == 'drag' for h in heads) / len(heads)
    return out


def derive_controls(mode: str, events, editable: Iterable[Interval]) -> tuple[dict, dict]:
    """Return (bands_without_missing_fields, raw_values)."""
    raw = raw_controls(mode, events, editable)
    bands = {k: band_of(k, v) for k, v in raw.items()}
    return {k: v for k, v in bands.items() if v is not None}, raw
