"""Synthetic chart builders for test_events_*.py (no real chart or audio bytes)."""
from fractions import Fraction
from pathlib import Path
import numpy as np
from beatfcn.events.canonical import Chart, Event, LaneLoc, LineLoc, TimingMap, freeze

FIX = Path(__file__).resolve().parent / 'fixtures'
CONTRACTS = Path(__file__).resolve().parents[1] / 'design' / 'package_2026-10-22'
DURATION = 30_000_000


def synthetic_mania(seed=0, duration=DURATION, gap=(12_000_000, 15_000_000)):
    """Collision-free 4K chart: 125 ms grid, chords, holds (some long), a natural blank `gap`."""
    rng = np.random.default_rng(seed)
    free_at, evs = [0] * 4, []
    for t in range(500_000, duration - 2_500_000, 125_000):
        if gap[0] <= t < gap[1] or rng.random() < 0.35:
            continue
        for lane in sorted(rng.choice(4, size=int(rng.integers(1, 3)), replace=False)):
            if free_at[lane] > t:
                continue
            if rng.random() < 0.2:
                end = min(t + int(rng.integers(2, 17)) * 125_000, gap[0] if t < gap[0] else duration - 500_000)
                if end <= t:
                    continue
                evs.append(Event(f'syn:{len(evs)}', t, end, 'hold', LaneLoc(int(lane)), f'syn.osu#{len(evs)}'))
                free_at[lane] = end
            else:
                evs.append(Event(f'syn:{len(evs)}', t, t, 'tap', LaneLoc(int(lane)), f'syn.osu#{len(evs)}'))
                free_at[lane] = t + 1
    return _chart('mania_4k', evs)


def synthetic_phira(seed=0, duration=DURATION):
    rng = np.random.default_rng(seed)
    evs, types = [], ('tap', 'tap', 'drag', 'flick', 'hold')
    for t in range(400_000, duration - 3_000_000, 200_000):
        if rng.random() < 0.4:
            continue
        for _ in range(int(rng.integers(1, 3))):
            ty = types[int(rng.integers(len(types)))]
            end = t + int(rng.integers(1, 10)) * 100_000 if ty == 'hold' else t
            loc = LineLoc(str(int(rng.integers(0, 3))), ('above', 'below')[int(rng.integers(2))],
                          float(rng.integers(-600, 600)))
            evs.append(Event(f'p:{len(evs)}', t, end, ty, loc, f'syn.json#{len(evs)}'))
    seen, uniq = set(), []
    for e in evs:
        if e.signature() not in seen:
            seen.add(e.signature())
            uniq.append(e)
    return _chart('phira', uniq)


def _chart(mode, evs):
    evs = tuple(sorted(evs, key=lambda e: (e.onset_us, e.event_id)))
    tm = TimingMap.from_bpm_points([(Fraction(0), Fraction(120))])
    return Chart(mode, 'synthetic', '0', evs, tm, Fraction(0), freeze({}),
                 freeze({'source_name': f'synthetic-{mode}'}), (), (), f'{mode}-synthetic-0000')
