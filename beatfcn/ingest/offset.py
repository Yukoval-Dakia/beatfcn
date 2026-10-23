"""Global chart-vs-audio offset diagnostic and the v1 decision rule (design/DATA_RULES.md section 2).

Estimator input is the precomputed Pupu-normalized 10 ms log-mel [T, 128] (mel10/{sha}.npy).
Envelope: positive spectral flux, [.25 .5 .25] smoothing, 0.5 s (51-frame) local mean removed, unit std.
score(s) = mean envelope at (note head + s), s in [-200, 200] ms, 1 ms steps; delta = argmax.
delta > 0 means the notes sit EARLY relative to the audio (audio onset at head + delta).
conf = (score(delta) - median(score)) / std(score). agree(s) = fraction of heads (+s) within
+-20 ms of an envelope peak (> 0.5); the full agree curve is kept as integer counts so the
decision stage can evaluate it at any residual r without the audio.

Decision (only a global shift of the model view, never per-note snapping; native chart untouched):
  r = delta - c, c = per-(source, codec) train median of delta (rounded to 1 ms).
  |r| <= 25                                       -> keep
  25 < |r| <= 120, conf >= 3, agree(r)-agree(0) >= 0.15, agree(r) >= 0.3, not edge -> shift by r
  anything else                                    -> quarantine with every failing reason
"""
from __future__ import annotations
import numpy as np

HOP_SEC = 0.01
MAX_SHIFT_MS = 200
SHIFTS_MS = np.arange(-MAX_SHIFT_MS, MAX_SHIFT_MS + 1)
AGREE_TOL_SEC = 0.02
PEAK_MIN = 0.5
EDGE_GUARD_SEC = 0.25
MIN_HEADS = 20
V1 = {'keep_ms': 25, 'shift_max_ms': 120, 'min_conf': 3.0, 'min_gain': 0.15, 'min_agree_after': 0.3,
      'disagree_ms': 25}


def flux_envelope(mel: np.ndarray) -> np.ndarray:
    m = np.asarray(mel, dtype=np.float32)
    d = np.concatenate([[0.0], np.maximum(m[1:] - m[:-1], 0).sum(1)])
    d = np.convolve(d, [.25, .5, .25], 'same')
    d = d - np.convolve(d, np.ones(51) / 51, 'same')
    return (d / (d.std() + 1e-6)).astype(np.float64)


def _peaks(env):
    i = np.where((env[1:-1] > env[:-2]) & (env[1:-1] >= env[2:]) & (env[1:-1] > PEAK_MIN))[0] + 1
    return i * HOP_SEC


def agree_counts(heads_sec: np.ndarray, peaks_sec: np.ndarray, shifts_ms=SHIFTS_MS) -> np.ndarray:
    """For each shift s: number of heads with a peak within +-20 ms of head + s."""
    if len(peaks_sec) == 0 or len(heads_sec) == 0:
        return np.zeros(len(shifts_ms), dtype=np.int64)
    t = heads_sec[None, :] + np.asarray(shifts_ms)[:, None] / 1000.0
    j = np.searchsorted(peaks_sec, t)
    lo = peaks_sec[np.clip(j - 1, 0, len(peaks_sec) - 1)]
    hi = peaks_sec[np.clip(j, 0, len(peaks_sec) - 1)]
    near = np.minimum(np.abs(t - lo), np.abs(t - hi)) <= AGREE_TOL_SEC + 1e-9
    return near.sum(1).astype(np.int64)


def estimate_offset(heads_sec, mel=None, env=None) -> dict:
    """Offset diagnostic for one chart. `heads_sec`: playable note-head times on the audio clock."""
    env = flux_envelope(mel) if env is None else env
    dur = len(env) * HOP_SEC
    h = np.unique(np.asarray(heads_sec, dtype=np.float64))
    h = h[(h > EDGE_GUARD_SEC) & (h < dur - EDGE_GUARD_SEC)]
    if len(h) < MIN_HEADS:
        return {'status': 'too_few_notes', 'n': int(len(h))}
    fr = np.arange(len(env)) * HOP_SEC
    sc = np.interp(h[None, :] + SHIFTS_MS[:, None] / 1000.0, fr, env).mean(1)
    i = int(sc.argmax())
    ac = agree_counts(h, _peaks(env))
    zero = MAX_SHIFT_MS
    return {'status': 'ok', 'n': int(len(h)), 'delta_ms': int(SHIFTS_MS[i]),
            'conf': round(float((sc[i] - np.median(sc)) / (sc.std() + 1e-6)), 3),
            'edge': i in (0, len(SHIFTS_MS) - 1),
            'agree0': round(float(ac[zero] / len(h)), 4), 'agree_delta': round(float(ac[i] / len(h)), 4),
            'agree_counts': ac.tolist()}


def agree_at(diag: dict, shift_ms: int) -> float | None:
    k = int(shift_ms) + MAX_SHIFT_MS
    if diag.get('status') != 'ok' or not 0 <= k < len(diag['agree_counts']):
        return None
    return diag['agree_counts'][k] / diag['n']


def train_constant(deltas) -> int | None:
    d = [x for x in deltas if x is not None]
    return int(np.round(np.median(d))) if d else None


def decide(diag: dict, c_ms: int, th=V1) -> dict:
    """Apply the v1 rule. Returns action keep|shift|quarantine, shift_ms (model-view shift), reasons."""
    if diag.get('status') != 'ok':
        return {'action': 'quarantine', 'shift_ms': 0, 'c_ms': c_ms, 'r_ms': None,
                'agree_before': None, 'agree_after': None, 'reasons': [f'offset_{diag.get("status", "missing")}']}
    r = int(diag['delta_ms']) - int(c_ms)
    a0, ar = agree_at(diag, 0), agree_at(diag, r)
    out = {'c_ms': int(c_ms), 'r_ms': r, 'agree_before': round(a0, 4),
           'agree_after': None if ar is None else round(ar, 4)}
    if abs(r) <= th['keep_ms']:
        return dict(out, action='keep', shift_ms=0, reasons=[])
    reasons = []
    if abs(r) > th['shift_max_ms']:
        reasons.append('offset_residual_too_large')
    if diag['edge']:
        reasons.append('offset_search_edge')
    if diag['conf'] < th['min_conf']:
        reasons.append('offset_low_conf')
    if ar is None or ar < th['min_agree_after']:
        reasons.append('offset_low_agree_after')
    if ar is not None and ar - a0 < th['min_gain']:
        reasons.append('offset_insufficient_gain')
    if reasons:
        return dict(out, action='quarantine', shift_ms=0, reasons=reasons)
    return dict(out, action='shift', shift_ms=r, reasons=[])


def disagreement_flags(rows, th=V1) -> set:
    """rows: iterable of (chart_id, audio_sha256, r_ms or None). Returns chart ids on audios whose
    charts' residuals span more than `disagree_ms`."""
    by_audio = {}
    for cid, sha, r in rows:
        if r is not None and sha:
            by_audio.setdefault(sha, []).append((cid, r))
    out = set()
    for lst in by_audio.values():
        rs = [r for _, r in lst]
        if len(rs) > 1 and max(rs) - min(rs) > th['disagree_ms']:
            out.update(c for c, _ in lst)
    return out
