"""Phira charts: RPE JSON and official Phigros JSON (formatVersion 3) -> canonical events.

Time semantics (Phira docs https://teamflos.github.io/phira-docs/ , cross-read against the
renderer source TeamFlos/phira prpr@0afd166 2026-10-19, prpr/src/parse/{rpe,pgr}.rs and
prpr/src/scene/game.rs):
- RPE beat triple [a, b, c] = a + b/c beats, kept as an exact Fraction (non-int or c == 0 quarantined).
- RPE multi-BPM: piecewise BPMList (docs beat2sec); first BPM point must be at beat 0, else quarantine.
- RPE bpmfactor: docs say the line BPM is bpm / bpmfactor. The renderer has `// TODO bpmfactor`
  (ignored), and one docs example multiplies. We compute the documented division but any
  bpmfactor != 1 is quarantined (`bpmfactor_semantics_unverified`) until the user decides.
- Official v3: times are in T = 1.875 / line_bpm seconds (1/32 beat); hold end = time + holdTime;
  note kinds 1 tap, 2 drag, 3 hold, 4 flick (renderer pgr.rs). RPE kinds 1 tap, 2 hold, 3 flick, 4 drag.
- Offset: renderer computes chart_time = music_time - (chart.offset + info.yml offset), so
  audio_sec = chart_sec + META.offset/1000 (RPE, int ms) or + offset (official, seconds), plus the
  info.yml `offset` (seconds) when the package provides it. NOTE: the RPE docs prose describes the
  opposite sign for META.offset; we follow the renderer and report the discrepancy.
- above: RPE `above == 1` -> above, anything else -> below (renderer). Official: notesAbove/notesBelow.
- isFake: 0/false playable, 1/true fake (kept only in `fakes` + sidecar). Other values quarantined
  (docs: "1 fake, others real"; renderer: "non-zero fake").

NOT handled: PEC / PBC / formatVersion != 3 / concatenated JSON (UnsupportedFormat); position
transforms through moving, rotating or parent lines (only line_id, side, raw native_x recorded);
speed/floorPosition/visibility semantics; attachUI lines; renderer parity golden cases.
"""
from __future__ import annotations
from decimal import Decimal
from fractions import Fraction
import hashlib
import json
from .canonical import (Chart, Event, Issue, LineLoc, TimingMap, UnsupportedFormat, exact_duplicates,
                        freeze, sec_to_us)

RPE_TYPES = {1: 'tap', 2: 'hold', 3: 'flick', 4: 'drag'}
PGR_TYPES = {1: 'tap', 2: 'drag', 3: 'hold', 4: 'flick'}
RPE_NOTE_FIELDS = {'type', 'above', 'startTime', 'endTime', 'positionX', 'yOffset', 'alpha', 'hitsound', 'size',
                   'speed', 'isFake', 'visibleTime', 'tint', 'color', 'tintHitEffects', 'judgeArea'}
PGR_NOTE_FIELDS = {'type', 'time', 'positionX', 'holdTime', 'speed', 'floorPosition'}
RPE_LINE_FIELDS = {'Group', 'Name', 'Texture', 'anchor', 'eventLayers', 'extended', 'father', 'isCover', 'notes',
                   'numOfNotes', 'zOrder', 'attachUI', 'isGif', 'posControl', 'sizeControl', 'skewControl',
                   'yControl', 'alphaControl', 'bpmfactor', 'rotateWithFather', 'scaleOnNotes', 'appearanceOnAttach',
                   'GroupID'}
PGR_LINE_FIELDS = {'bpm', 'notesAbove', 'notesBelow', 'speedEvents', 'judgeLineMoveEvents',
                   'judgeLineRotateEvents', 'judgeLineDisappearEvents', 'numOfNotes', 'numOfNotesAbove',
                   'numOfNotesBelow'}


def _is_int(x):
    return type(x) is int


def _num(x) -> Fraction | None:
    if type(x) is bool:
        return None
    if isinstance(x, (int, Decimal)) and (not isinstance(x, Decimal) or x.is_finite()):
        return Fraction(x)
    return None


def beat_triple(t) -> Fraction:
    if not (isinstance(t, list) and len(t) == 3 and all(_is_int(v) for v in t)):
        raise ValueError(f'beat must be [int, int, int], got {t!r}')
    if t[2] == 0:
        raise ValueError(f'beat denominator is 0: {t!r}')
    return t[0] + Fraction(t[1], t[2])


def _load_json(data):
    if isinstance(data, (bytes, bytearray)):
        try:
            text = bytes(data).decode('utf-8-sig')
        except UnicodeDecodeError as e:
            raise UnsupportedFormat(f'chart is not UTF-8: {e}') from e
    else:
        text = data.lstrip('﻿')
    if not text.lstrip().startswith('{'):
        raise UnsupportedFormat('non-JSON chart (PEC/PBC?): needs a verified converter, not implemented in P0')
    try:
        return json.loads(text, parse_float=Decimal), text
    except json.JSONDecodeError as e:
        raise UnsupportedFormat(f'malformed or concatenated JSON: {e}') from e


def _offset_from_info(info_offset) -> Fraction:
    if info_offset is None:
        return Fraction(0)
    f = Fraction(Decimal(str(info_offset)))
    return f


def parse_phira(data, source_name: str = 'chart.json', info_offset_sec=None) -> Chart:
    """Parse RPE or official v3 JSON. `info_offset_sec` is info.yml `offset` (seconds) if known."""
    d, text = _load_json(data)
    if not isinstance(d, dict):
        raise UnsupportedFormat('chart JSON root is not an object')
    raw = text.encode('utf-8') if isinstance(data, str) else bytes(data)
    info_off = _offset_from_info(info_offset_sec)
    if 'formatVersion' in d:
        if d['formatVersion'] != 3:
            raise UnsupportedFormat(f'official formatVersion {d["formatVersion"]!r} not supported (P0: 3 only)')
        return _parse_pgr(d, raw, source_name, info_off)
    if 'BPMList' in d and 'judgeLineList' in d:
        return _parse_rpe(d, raw, source_name, info_off)
    raise UnsupportedFormat('JSON is neither RPE (BPMList+judgeLineList) nor official (formatVersion)')


def _finish(fmt, version, events, fakes, issues, timing, offset, beats, scene, raw, source_name, extra=None):
    for a, b in exact_duplicates(events):
        issues.append(Issue('duplicate_object', 'quarantine', f'{a} vs {b}', b))
    for e in events:
        if e.onset_us < 0:
            issues.append(Issue('negative_time', 'quarantine', f'onset {e.onset_us} us', e.event_id))
    key = lambda e: (e.onset_us, *e.location.sort_key(), e.type, e.event_id)
    side = {'raw': raw, 'source_name': source_name, 'scene': scene, **(extra or {})}
    return Chart('phira', fmt, version, tuple(sorted(events, key=key)), timing, offset, freeze(beats),
                 freeze(side), tuple(sorted(fakes, key=key)), tuple(issues), hashlib.sha256(raw).hexdigest())


def _parse_rpe(d, raw, source_name, info_off) -> Chart:
    issues = []
    meta = d.get('META') if isinstance(d.get('META'), dict) else {}
    if not isinstance(d.get('META'), dict):
        issues.append(Issue('missing_meta', 'quarantine', 'META object missing; offset unknown'))
    off_ms = meta.get('offset', 0)
    if not _is_int(off_ms):
        issues.append(Issue('meta_offset_not_int', 'quarantine', f'META.offset={off_ms!r} (renderer reads i32 ms)'))
        off_ms = 0
    offset = Fraction(off_ms, 1000) + info_off
    pts = []
    for k, b in enumerate(d.get('BPMList') or []):
        try:
            beat, bpm = beat_triple(b.get('startTime')), _num(b.get('bpm'))
        except (ValueError, AttributeError) as e:
            issues.append(Issue('bad_bpm_point', 'quarantine', f'BPMList[{k}]: {e}'))
            continue
        if bpm is None or bpm <= 0:
            issues.append(Issue('bad_bpm_point', 'quarantine', f'BPMList[{k}].bpm={b.get("bpm")!r}'))
            continue
        pts.append((beat, bpm))
    if not pts:
        issues.append(Issue('no_bpm', 'quarantine', 'empty or invalid BPMList'))
    if [p[0] for p in pts] != sorted(p[0] for p in pts):
        issues.append(Issue('bpmlist_unsorted', 'warning', 'BPMList sorted by startTime (documented loader behaviour)'))
        pts.sort(key=lambda p: p[0])
    if len({p[0] for p in pts}) != len(pts):
        issues.append(Issue('bpmlist_duplicate_beat', 'quarantine', 'two BPM points at the same beat'))
    if pts and pts[0][0] != 0:
        issues.append(Issue('bpm_first_not_zero', 'quarantine', f'first BPM point at beat {pts[0][0]}'))
    timing = TimingMap.from_bpm_points(pts, offset) if pts else None
    events, fakes, beats, lines_scene = [], [], {}, []
    lines = d.get('judgeLineList')
    if not isinstance(lines, list):
        issues.append(Issue('no_judge_lines', 'quarantine', 'judgeLineList missing'))
        lines = []
    tmaps = {}
    for li, line in enumerate(lines):
        if not isinstance(line, dict):
            issues.append(Issue('bad_line', 'quarantine', f'line {li} not an object'))
            continue
        unknown = sorted(set(line) - RPE_LINE_FIELDS)
        if unknown:
            issues.append(Issue('unknown_line_fields', 'warning', f'line {li}: {unknown}'))
        father = line.get('father', -1)
        if not _is_int(father) or father != -1 and not 0 <= father < len(lines) or father == li:
            issues.append(Issue('bad_father', 'warning', f'line {li}: father={father!r} (positions only; ignored in P0)'))
        bf = _num(line.get('bpmfactor', 1))
        if bf is None or bf <= 0:
            issues.append(Issue('bad_bpmfactor', 'quarantine', f'line {li}: bpmfactor={line.get("bpmfactor")!r}'))
            bf = Fraction(1)
        elif bf != 1:
            issues.append(Issue('bpmfactor_semantics_unverified', 'quarantine',
                                f'line {li}: bpmfactor={bf}; docs divide BPM, renderer ignores it'))
        if timing is not None and bf not in tmaps:
            tmaps[bf] = TimingMap.from_bpm_points([(b, bpm / bf) for b, bpm in pts], offset)
        lines_scene.append({'line_id': str(li), 'father': father, 'bpmfactor': str(bf),
                            'fields': {k: v for k, v in line.items() if k != 'notes'}})
        for ni, n in enumerate(line.get('notes') or []):
            eid, ref = f'rpe:{li}:{ni}', f'{source_name}#judgeLineList/{li}/notes/{ni}'
            if not isinstance(n, dict):
                issues.append(Issue('bad_note', 'quarantine', 'note not an object', eid))
                continue
            unknown = sorted(set(n) - RPE_NOTE_FIELDS)
            if unknown:
                issues.append(Issue('unknown_note_fields', 'quarantine', f'{unknown}', eid))
            typ = RPE_TYPES.get(n.get('type')) if _is_int(n.get('type')) else None
            if typ is None:
                issues.append(Issue('unknown_note_type', 'quarantine', f'type={n.get("type")!r}', eid))
                continue
            fake = n.get('isFake', 0)
            if fake not in (0, 1) or not isinstance(fake, (int, bool)):
                issues.append(Issue('ambiguous_is_fake', 'quarantine', f'isFake={fake!r}', eid))
                continue
            try:
                b0 = beat_triple(n.get('startTime'))
                b1 = beat_triple(n.get('endTime')) if typ == 'hold' else b0
            except ValueError as e:
                issues.append(Issue('bad_note_time', 'quarantine', str(e), eid))
                continue
            x = _num(n.get('positionX'))
            if x is None:
                issues.append(Issue('bad_position_x', 'quarantine', f'positionX={n.get("positionX")!r}', eid))
                continue
            above = n.get('above', 1)
            if typ == 'hold' and b1 <= b0:
                issues.append(Issue('non_positive_hold', 'quarantine', f'{b0}..{b1} beats', eid))
                continue
            if timing is None:
                continue
            tm = tmaps[bf]
            on, end = tm.audio_us(b0), tm.audio_us(b1)
            ev = Event(eid, on, end, typ, LineLoc(str(li), 'above' if above == 1 else 'below', float(x)), ref)
            (fakes if fake else events).append(ev)
            beats[eid] = (b0, b1)
    return _finish('phira_rpe', str(meta.get('RPEVersion', 'unknown')), events, fakes, issues, timing, offset,
                   beats, lines_scene, raw, source_name, {'meta': meta, 'bpm_points': [(str(b), str(v)) for b, v in pts]})


def _parse_pgr(d, raw, source_name, info_off) -> Chart:
    issues = []
    off = _num(d.get('offset', 0))
    if off is None:
        issues.append(Issue('bad_offset', 'quarantine', f'offset={d.get("offset")!r}'))
        off = Fraction(0)
    offset = off + info_off
    events, beats, scene = [], {}, []
    lines = d.get('judgeLineList')
    if not isinstance(lines, list):
        issues.append(Issue('no_judge_lines', 'quarantine', 'judgeLineList missing'))
        lines = []
    bpms = set()
    for li, line in enumerate(lines):
        if not isinstance(line, dict):
            issues.append(Issue('bad_line', 'quarantine', f'line {li} not an object'))
            continue
        unknown = sorted(set(line) - PGR_LINE_FIELDS)
        if unknown:
            issues.append(Issue('unknown_line_fields', 'warning', f'line {li}: {unknown}'))
        bpm = _num(line.get('bpm'))
        if bpm is None or bpm <= 0:
            issues.append(Issue('bad_line_bpm', 'quarantine', f'line {li}: bpm={line.get("bpm")!r}'))
            continue
        bpms.add(bpm)
        unit = Fraction(15, 8) / bpm  # seconds per T (1/32 beat)
        scene.append({'line_id': str(li), 'bpm': str(bpm),
                      'fields': {k: v for k, v in line.items() if k not in ('notesAbove', 'notesBelow')}})
        for side, key in (('above', 'notesAbove'), ('below', 'notesBelow')):
            for ni, n in enumerate(line.get(key) or []):
                eid, ref = f'pgr:{li}:{side}:{ni}', f'{source_name}#judgeLineList/{li}/{key}/{ni}'
                if not isinstance(n, dict):
                    issues.append(Issue('bad_note', 'quarantine', 'note not an object', eid))
                    continue
                unknown = sorted(set(n) - PGR_NOTE_FIELDS)
                if unknown:
                    issues.append(Issue('unknown_note_fields', 'quarantine', f'{unknown}', eid))
                typ = PGR_TYPES.get(n.get('type')) if _is_int(n.get('type')) else None
                t, ht, x = _num(n.get('time')), _num(n.get('holdTime', 0)), _num(n.get('positionX'))
                if typ is None or t is None or ht is None or x is None:
                    issues.append(Issue('bad_note', 'quarantine', f'type/time/holdTime/positionX invalid: {dict(n)!r}'[:200], eid))
                    continue
                if typ == 'hold' and ht <= 0:
                    issues.append(Issue('non_positive_hold', 'quarantine', f'holdTime={ht}', eid))
                    continue
                if typ != 'hold' and ht != 0:
                    issues.append(Issue('holdtime_on_non_hold', 'warning', f'holdTime={ht} ignored', eid))
                end_t = t + ht if typ == 'hold' else t
                on, end = sec_to_us(t * unit + offset), sec_to_us(end_t * unit + offset)
                events.append(Event(eid, on, end, typ, LineLoc(str(li), side, float(x)), ref))
                beats[eid] = (t / 32, end_t / 32)
    timing = TimingMap.from_bpm_points([(0, next(iter(bpms)))], offset) if len(bpms) == 1 else None
    return _finish('phira_official_v3', '3', events, [], issues, timing, offset, beats, scene, raw, source_name,
                   {'line_bpms_distinct': len(bpms)})


def load_package(zip_path, chart_name: str | None = None) -> Chart:
    """Read a Phira .zip without extracting it: info.yml (chart, offset, format) + chart file.

    info.yml `format` of pec/pbc -> UnsupportedFormat. Requires PyYAML.
    """
    import zipfile
    import yaml
    with zipfile.ZipFile(zip_path) as z:
        names = {n.lower().rsplit('/', 1)[-1]: n for n in z.namelist()}
        info_name = names.get('info.yml') or names.get('info.yaml')
        info = yaml.safe_load(z.read(info_name).decode('utf-8-sig')) if info_name else {}
        info = info if isinstance(info, dict) else {}
        fmt = str(info.get('format') or '').lower()
        if fmt in ('pec', 'pbc'):
            raise UnsupportedFormat(f'info.yml format={fmt}')
        name = chart_name or info.get('chart')
        if not name:
            raise UnsupportedFormat('no chart file named in info.yml')
        member = names.get(str(name).lower().rsplit('/', 1)[-1])
        if member is None:
            raise UnsupportedFormat(f'chart file {name!r} not in package')
        return parse_phira(z.read(member), source_name=str(name), info_offset_sec=info.get('offset', 0))
