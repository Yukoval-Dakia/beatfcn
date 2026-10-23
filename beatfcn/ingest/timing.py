"""Timing-map (beat grid / BPM) reliability diagnostic (design/DATA_RULES.md section 3).

Never used to create or move labels; only flags `timing_unreliable`.
- snap rate: fraction of playable heads within +-3 ms of a 1/16 or 1/12 subdivision of the chart's
  own beat grid. A head counts as on-grid if it snaps either on the global beat count or relative
  to the start of its BPM segment (osu red lines restart the bar at their own time, RPE beats are
  global). Deviation in seconds = |beat - snapped| * 60 / bpm of the head's segment.
- BPM segments: zero-length segments (several points at the same beat) are dropped, since only the
  last one takes effect. Duration of a non-final segment is measured in its own beats.
- Adjacent BPM ratio is "simple" if within 0.1 % of 1/2, 2/3, 3/4, 1, 4/3, 3/2, 2 (ratio 1 = re-sync
  red line, not a tempo change).
Flags (v1): snap < 0.9 -> timing_low_snap; a non-final segment < 1 beat -> timing_short_segment;
a non-simple adjacent ratio with > 3 segments -> timing_irregular_bpm.
Official Phigros v3 charts have one constant BPM per line (no segments); each head uses its line BPM.
"""
from __future__ import annotations
import numpy as np

GRIDS = (16, 12)
SNAP_TOL_SEC = 0.003
SIMPLE_RATIOS = (1 / 2, 2 / 3, 3 / 4, 1.0, 4 / 3, 3 / 2, 2.0)
RATIO_TOL = 1e-3
V1 = {'min_snap': 0.9, 'min_segment_beats': 1.0, 'irregular_min_segments': 4}


def is_simple_ratio(x: float) -> bool:
    return any(abs(x / s - 1) <= RATIO_TOL for s in SIMPLE_RATIOS)


def segment_metrics(segments) -> dict:
    """segments: [(start_beat, bpm)] sorted by beat (floats)."""
    segs = []
    for b, bpm in segments:
        if segs and abs(b - segs[-1][0]) < 1e-9:
            segs[-1] = (b, bpm)
        else:
            segs.append((b, bpm))
    lens = [segs[i + 1][0] - segs[i][0] for i in range(len(segs) - 1)]
    ratios = [segs[i + 1][1] / segs[i][1] for i in range(len(segs) - 1)]
    nonsimple = [round(r, 6) for r in ratios if not is_simple_ratio(r)]
    return {'n_segments': len(segs), 'min_segment_beats': round(min(lens), 6) if lens else None,
            'n_nonsimple_ratios': len(nonsimple), 'nonsimple_ratio_examples': nonsimple[:5],
            'segments': segs}


def snap_deviation_sec(beat: np.ndarray, seg_start: np.ndarray, bpm: np.ndarray) -> np.ndarray:
    best = np.full(len(beat), np.inf)
    for g in GRIDS:
        for base in (np.zeros_like(beat), seg_start):
            rel = beat - base
            best = np.minimum(best, np.abs(rel - np.round(rel * g) / g))
    return best * 60.0 / bpm


def heads_with_segments(chart):
    """(beat, segment_start_beat, bpm) per playable head, and the segment list used for metrics."""
    beats = chart.native_beats
    ev = [e for e in chart.events if e.event_id in beats]
    if chart.source_format == 'phira_official_v3':
        line_bpm = {s['line_id']: float(s['bpm']) for s in chart.sidecar['scene']}
        rows = [(float(beats[e.event_id][0]), 0.0, line_bpm[e.location.line_id]) for e in ev]
        return rows, [(0.0, b) for b in sorted(set(line_bpm.values()))[:1]]
    if chart.timing is None:
        return [], []
    segs = [(float(s.start_beat), float(s.bpm)) for s in chart.timing.segments]
    factor = {}
    if chart.source_format == 'phira_rpe':
        factor = {s['line_id']: float(s['bpmfactor']) for s in chart.sidecar['scene']}
    starts = np.array([s[0] for s in segs])
    rows = []
    for e in ev:
        b = float(beats[e.event_id][0])
        k = max(int(np.searchsorted(starts, b + 1e-9, 'right')) - 1, 0)
        f = factor.get(getattr(e.location, 'line_id', None), 1.0)
        rows.append((b, segs[k][0], segs[k][1] / f))
    return rows, segs


def timing_reliability(chart, th=V1) -> dict:
    rows, segs = heads_with_segments(chart)
    m = segment_metrics(segs) if segs else {'n_segments': 0, 'min_segment_beats': None,
                                             'n_nonsimple_ratios': 0, 'nonsimple_ratio_examples': [], 'segments': []}
    out = {k: v for k, v in m.items() if k != 'segments'}
    if chart.source_format == 'phira_official_v3':
        out['line_bpms_distinct'] = len({s['bpm'] for s in chart.sidecar['scene']})
    if rows:
        a = np.array(rows, dtype=np.float64)
        dev = snap_deviation_sec(a[:, 0], a[:, 1], a[:, 2])
        out['snap_rate'] = round(float((dev <= SNAP_TOL_SEC + 1e-9).mean()), 4)
        out['n_heads'] = len(rows)
    else:
        out['snap_rate'], out['n_heads'] = None, 0
    trig = []
    if out['snap_rate'] is None or out['snap_rate'] < th['min_snap']:
        trig.append('timing_low_snap')
    if m['min_segment_beats'] is not None and m['min_segment_beats'] < th['min_segment_beats']:
        trig.append('timing_short_segment')
    if m['n_nonsimple_ratios'] and m['n_segments'] >= th['irregular_min_segments']:
        trig.append('timing_irregular_bpm')
    out['triggers'] = trig
    out['unreliable'] = bool(trig)
    return out
