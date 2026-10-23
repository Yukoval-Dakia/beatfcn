"""osu! file format v14, mania 4K only: parse to canonical events and merge-export additions.

Semantics (osu wiki, File formats/osu): [TimingPoints] `time,beatLength,meter,sampleSet,
sampleIndex,volume,uninherited,effects`. uninherited=1 -> BPM = 60000/beatLength;
uninherited=0 -> inherited point, negative beatLength is a slider-velocity multiplier
(-100/beatLength), never a BPM. Hit objects `x,y,time,type,hitSound,...`; type bit 128 is a
mania hold whose extras start with `endTime:`; lane = floor(x*keys/512). Hit-object times are
taken as audio-clock milliseconds (v14 has no implicit legacy offset).

Round trip: the original bytes are split with bytes.splitlines(keepends=True) and kept verbatim;
`export_osu(chart, [])` returns them unchanged; additions are inserted as new lines inside
[HitObjects] and no original line content is touched.

NOT handled: versions other than v14, modes other than mania, key counts other than 4
(UnsupportedFormat); lane clamping (out-of-range x is quarantined, not clamped); SV/scroll,
hitsounds or samples of additions (written as defaults); renderer/judge parity is not verified here.
"""
from __future__ import annotations
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import hashlib
import math
from .canonical import (Chart, Event, Issue, LaneLoc, TimingMap, UnsupportedFormat, freeze,
                        lane_conflicts, sec_to_us)

KEYS = 4
SUPPORTED_VERSIONS = (14,)


class ExportError(ValueError):
    pass


def _dec(s: str) -> Fraction:
    try:
        d = Decimal(s.strip())
    except InvalidOperation as e:
        raise ValueError(f'not a number: {s!r}') from e
    if not d.is_finite():
        raise ValueError(f'non-finite number: {s!r}')
    return Fraction(d)


def _sections(lines: list[bytes]):
    """Yield (line_index, section_name, decoded_stripped_text)."""
    sec = None
    for i, raw in enumerate(lines):
        t = raw.decode('utf-8').rstrip('\r\n')
        if i == 0:
            t = t.lstrip('﻿')
        s = t.strip()
        if s.startswith('[') and s.endswith(']'):
            sec = s[1:-1]
        yield i, sec, s


def parse_osu(data: bytes, source_name: str = 'chart.osu') -> Chart:
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError('parse_osu needs the original bytes for byte-identical round trip')
    data = bytes(data)
    lines = data.splitlines(keepends=True)
    try:
        rows = list(_sections(lines))
    except UnicodeDecodeError as e:
        raise UnsupportedFormat(f'.osu is not UTF-8: {e}') from e
    head = next((s for _, _, s in rows if s), '')
    if not head.startswith('osu file format v'):
        raise UnsupportedFormat('missing "osu file format vN" header')
    try:
        version = int(head.rsplit('v', 1)[1])
    except ValueError as e:
        raise UnsupportedFormat(f'bad header {head!r}') from e
    if version not in SUPPORTED_VERSIONS:
        raise UnsupportedFormat(f'osu file format v{version} not supported (P0 supports {SUPPORTED_VERSIONS})')
    kv = {}
    for _, sec, s in rows:
        if sec in ('General', 'Difficulty', 'Metadata') and ':' in s and not s.startswith('//'):
            k, v = s.split(':', 1)
            kv[(sec, k.strip())] = v.strip()
    if kv.get(('General', 'Mode')) != '3':
        raise UnsupportedFormat(f'Mode={kv.get(("General", "Mode"))!r}; only mania (3) supported')
    keys = kv.get(('Difficulty', 'CircleSize'))
    try:
        keys_n = _dec(keys) if keys is not None else None
    except ValueError:
        keys_n = None
    if keys_n != KEYS:
        raise UnsupportedFormat(f'CircleSize={keys!r}; only 4K mania supported')

    issues, timing_rows, sv_rows, ho = [], [], [], []
    for i, sec, s in rows:
        if not s or s.startswith('//') or s.startswith('['):
            continue
        if sec == 'TimingPoints':
            f = s.split(',')
            try:
                t, bl = _dec(f[0]), _dec(f[1])
                meter = int(f[2]) if len(f) > 2 and f[2].strip() else 4
                unin = f[6].strip() if len(f) > 6 else None
            except (ValueError, IndexError) as e:
                issues.append(Issue('bad_timing_point', 'quarantine', f'line {i + 1}: {e}'))
                continue
            if unin is None:   # pre-v6 rows lack the flag; sign is then the documented discriminator
                unin = '1' if bl > 0 else '0'
            if unin == '1':
                if bl <= 0:
                    issues.append(Issue('uninherited_nonpositive_beatlength', 'quarantine', f'line {i + 1}: beatLength={f[1]}'))
                    continue
                timing_rows.append({'line': i, 'time_ms': t, 'beat_length_ms': bl, 'meter': meter})
            elif unin == '0':
                sv = (Fraction(-100) / bl) if bl < 0 else None
                if bl >= 0:
                    issues.append(Issue('inherited_nonnegative_beatlength', 'warning', f'line {i + 1}: beatLength={f[1]} (SV undefined, ignored for timing)'))
                sv_rows.append({'line': i, 'time_ms': t, 'beat_length': bl, 'sv': sv})
            else:
                issues.append(Issue('bad_uninherited_flag', 'quarantine', f'line {i + 1}: {unin!r}'))
        elif sec == 'HitObjects':
            ho.append((i, s))
    if not timing_rows:
        issues.append(Issue('no_uninherited_timing_point', 'quarantine', 'no BPM timing point'))
    timing_rows.sort(key=lambda r: r['time_ms'])
    timing = None
    if timing_rows:
        # beat 0 at the first uninherited point; each red line restarts at its own exact time
        pts, beat = [], Fraction(0)
        for k, r in enumerate(timing_rows):
            if k:
                p = timing_rows[k - 1]
                beat += (r['time_ms'] - p['time_ms']) / p['beat_length_ms']
            pts.append((beat, Fraction(60000) / r['beat_length_ms']))
        timing = TimingMap.from_bpm_points(pts, first_sec=timing_rows[0]['time_ms'] / 1000)

    events, beats, raw_map = [], {}, {}
    for i, s in ho:
        f = s.split(',')
        eid = f'osu:L{i + 1}'
        ref = f'{source_name}#L{i + 1}'
        try:
            x, t_ms, typ = _dec(f[0]), _dec(f[2]), int(f[3])
        except (ValueError, IndexError) as e:
            issues.append(Issue('bad_hit_object', 'quarantine', f'line {i + 1}: {e}', eid))
            continue
        lane = math.floor(x * KEYS / 512)
        if not 0 <= lane < KEYS:
            issues.append(Issue('lane_out_of_range', 'quarantine', f'line {i + 1}: x={f[0]}', eid))
            continue
        is_hold, is_circle = bool(typ & 128), bool(typ & 1)
        if typ & (2 | 8) or is_hold == is_circle:
            issues.append(Issue('invalid_mania_object_type', 'quarantine', f'line {i + 1}: type={typ}', eid))
            continue
        end_ms = t_ms
        if is_hold:
            try:
                end_ms = _dec(f[5].split(':', 1)[0])
            except (ValueError, IndexError) as e:
                issues.append(Issue('bad_hold_end', 'quarantine', f'line {i + 1}: {e}', eid))
                continue
        on, end = sec_to_us(t_ms / 1000), sec_to_us(end_ms / 1000)
        if on < 0:
            issues.append(Issue('negative_time', 'quarantine', f'line {i + 1}: {f[2]}', eid))
        if is_hold and end <= on:
            issues.append(Issue('non_positive_hold', 'quarantine', f'line {i + 1}: {f[2]}..{end_ms}', eid))
            continue
        ev = Event(eid, on, end, 'hold' if is_hold else 'tap', LaneLoc(lane), ref)
        events.append(ev)
        raw_map[eid] = i
        if timing is not None:
            beats[eid] = (_beat_of(timing, t_ms), _beat_of(timing, end_ms))
    for code, a, b in lane_conflicts(events):
        issues.append(Issue('lane_collision', 'quarantine', f'{code}: {a} vs {b}', b))
    events.sort(key=lambda e: (e.onset_us, e.location.lane))
    side = {'lines': tuple(lines), 'event_line': raw_map, 'timing_rows': timing_rows, 'sv_rows': sv_rows,
            'hit_object_lines': tuple(i for i, _ in ho), 'keys': KEYS, 'source_name': source_name,
            'newline': _newline(lines)}
    return Chart('mania_4k', 'osu_v14', f'v{version}', tuple(events), timing, Fraction(0),
                 freeze(beats), freeze(side), (), tuple(issues), hashlib.sha256(data).hexdigest())


def _beat_of(tm: TimingMap, t_ms: Fraction) -> Fraction:
    sec = Fraction(t_ms) / 1000
    seg = tm.segments[0]
    for s in tm.segments:
        if s.start_sec <= sec:
            seg = s
    return seg.start_beat + (sec - seg.start_sec) * seg.bpm / 60


def _newline(lines) -> bytes:
    for ln in lines:
        if ln.endswith(b'\r\n'):
            return b'\r\n'
        if ln.endswith(b'\n'):
            return b'\n'
    return b'\r\n'


def lane_x(lane: int) -> int:
    return lane * 512 // KEYS + 512 // (2 * KEYS)


def hit_object_line(e: Event) -> bytes:
    if e.onset_us % 1000 or e.end_us % 1000:
        raise ExportError(f'{e.event_id}: osu hit objects need integer ms, got {e.onset_us}/{e.end_us} us')
    t = e.onset_us // 1000
    if e.type == 'hold':
        return f'{lane_x(e.location.lane)},192,{t},128,0,{e.end_us // 1000}:0:0:0:0:'.encode()
    return f'{lane_x(e.location.lane)},192,{t},1,0,0:0:0:0:'.encode()


def export_osu(chart: Chart, additions=()) -> bytes:
    """Merge additions into the original file. Original lines are emitted byte-for-byte, in order.

    Fails closed (ExportError) on quarantined charts, non-mania additions, non-integer-ms times,
    id clashes, or lane conflicts with existing/other added events. Nothing is rounded.
    """
    if chart.source_format != 'osu_v14':
        raise ExportError('not an osu chart')
    lines = list(chart.sidecar['lines'])
    if not additions:
        return b''.join(lines)
    if chart.quarantined:
        raise ExportError('refusing to merge into a quarantined chart')
    errs, ids = [], {e.event_id for e in chart.events}
    for a in additions:
        if a.type not in ('tap', 'hold') or not isinstance(a.location, LaneLoc):
            errs.append(f'{a.event_id}: not a mania tap/hold')
        elif a.type == 'hold' and a.end_us <= a.onset_us or a.type == 'tap' and a.end_us != a.onset_us:
            errs.append(f'{a.event_id}: bad end')
        if a.event_id in ids:
            errs.append(f'{a.event_id}: id clashes with existing event')
        ids.add(a.event_id)
    if not errs:
        errs += [f'{c}: {x} vs {y}' for c, x, y in lane_conflicts(list(chart.events) + list(additions))]
    if errs:
        raise ExportError('; '.join(errs))
    new = sorted(additions, key=lambda e: (e.onset_us, e.location.lane))
    ho_idx = list(chart.sidecar['hit_object_lines'])
    nl = chart.sidecar['newline']
    if ho_idx:
        times = {i: int(Decimal(lines[i].decode('utf-8').split(',')[2])) for i in ho_idx}
        last = ho_idx[-1]
    else:
        sec_line = next((i for i, ln in enumerate(lines) if ln.strip() == b'[HitObjects]'), None)
        if sec_line is None:
            raise ExportError('no [HitObjects] section to merge into')
        times, last = {}, sec_line
    inserts = {}  # line index -> list of new lines emitted *before* that original line; key None = after `last`
    for a in new:
        t = a.onset_us // 1000
        at = next((i for i in ho_idx if times[i] > t), None)
        inserts.setdefault(at, []).append(hit_object_line(a))
    out = []
    for i, ln in enumerate(lines):
        for x in inserts.get(i, []):
            out.append(x + nl)
        out.append(ln)
        if i == last and None in inserts:
            tail = inserts[None]
            if not (ln.endswith(b'\n') or ln.endswith(b'\r')):
                out.append(nl)  # original line content untouched; only a terminator precedes the new line
                out.append(nl.join(tail))
            else:
                out.extend(x + nl for x in tail)
    return b''.join(out)
