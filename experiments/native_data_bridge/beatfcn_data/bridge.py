"""Architecture-neutral, fail-closed, addition-only native osu4K target conversion.

Never import a model or infer intent from an unknown/editable blank. Provenance
and supervision are deliberately separate from the allowlisted model view.
"""
from __future__ import annotations
import copy
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path

VERSION = 'beatfcn.osu4k.bridge.v1'
RECIPES = ('middle_gap', 'sparse_anchors', 'complete_reference', 'multiple_gaps', 'explicit_blank')
EVENT_FIELDS = ('event_id', 'onset_us', 'end_us', 'type', 'location', 'native_ref')
MODEL_EVENT_FIELDS = ('onset_us', 'end_us', 'type', 'location')

class DataError(ValueError):
    """Invalid, corrupted, or inconsistent input; never silently repair it."""

class Unsupported(DataError):
    """Well-defined operation for which source evidence or implementation is absent."""

def require(ok, reason):
    if not ok:
        raise DataError(reason)

def canonical_bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()

def digest(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()

def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]

def decimal_us(text, scale=1000):
    """Exact native decimal -> microseconds. Never round native objects."""
    try:
        d = Decimal(str(text))
        require(d.is_finite(), 'nonfinite native time')
        q = d * scale
        require(q == q.to_integral_value(), 'native time not representable in exact microseconds')
        return int(q)
    except (InvalidOperation, ValueError) as exc:
        raise DataError('invalid native decimal time') from exc

def _project(event, fields=EVENT_FIELDS):
    return {k: copy.deepcopy(event[k]) for k in fields}

def event_id(chart_hash, object_index):
    # Full digest identifies immutable bytes; ordinal distinguishes identical rows.
    # No truncated hashes or random IDs; no fallback renaming of musical collisions.
    return f'osu:{chart_hash}:{object_index}'

def parse_osu(raw: bytes, expected_sha=None, duration_us=None):
    actual_sha = hashlib.sha256(raw).hexdigest()
    if expected_sha is not None:
        require(actual_sha == expected_sha, 'chart SHA256 mismatch')
    try:
        lines = raw.decode('utf-8-sig').splitlines()
    except UnicodeDecodeError as exc:
        raise Unsupported('non-UTF8 osu source requires an explicit encoding adapter') from exc
    require(lines and lines[0].startswith('osu file format v'), 'missing osu format header')
    version = int(lines[0].split('v')[-1])
    if version < 10 or version > 14:
        raise Unsupported(f'unvalidated osu format version {version}')
    sections = defaultdict(list)
    section = ''
    for line_number, raw_line in enumerate(lines, 1):
        line = raw_line.strip()
        if line.startswith('[') and line.endswith(']'):
            section = line[1:-1]
        elif line and not line.startswith('//'):
            sections[section].append((line_number, raw_line))
    def kv(section):
        return {s.split(':', 1)[0].strip(): s.split(':', 1)[1].strip()
                for _, s in sections[section] if ':' in s}
    general, difficulty = kv('General'), kv('Difficulty')
    require(general.get('Mode') == '3', 'not native osu!mania Mode 3')
    require(Decimal(difficulty.get('CircleSize', '-1')) == 4, 'not native 4K')
    timing = []
    for line_number, line in sections['TimingPoints']:
        p = line.split(',')
        require(len(p) >= 2, 'incomplete timing point')
        offset = Decimal(p[0]); length = Decimal(p[1])
        require(offset.is_finite() and length.is_finite(), 'nonfinite timing point')
        flag = int(p[6]) if len(p) > 6 else 1
        require(flag in (0, 1), 'unknown timing inheritance flag')
        require((flag == 1 and length > 0) or (flag == 0 and length < 0), 'timing sign/flag mismatch')
        timing.append({'native_line_number': line_number, 'native_line': line,
                       'offset_ms_decimal': p[0], 'beat_length_ms_decimal': p[1],
                       'uninherited': flag, 'role': 'music_timing_reference' if flag else 'sv_or_control'})
    require(any(t['uninherited'] for t in timing), 'missing positive uninherited timing')
    events, sidecar, used_ids = [], {}, set()
    for index, (line_number, line) in enumerate(sections['HitObjects']):
        p = line.split(',')
        require(len(p) >= 5, 'incomplete hit object')
        x, flags = int(p[0]), int(p[3])
        require(0 <= x <= 512, 'native x outside supported lane domain')
        basic = flags & (1 | 2 | 8 | 128)
        require(basic in (1, 128), 'unsupported or ambiguous playable object flags')
        onset = decimal_us(p[2])
        hold = basic == 128
        require(not hold or len(p) >= 6, 'missing hold endpoint')
        end = decimal_us(p[5].split(':')[0]) if hold else onset
        require(onset >= 0 and (end > onset if hold else end == onset), 'invalid native event span')
        require(duration_us is None or (onset < duration_us and end <= duration_us), 'event outside audio')
        eid = event_id(actual_sha, index)
        require(eid not in used_ids, 'event ID identity collision')
        used_ids.add(eid)
        events.append({'event_id': eid, 'onset_us': onset, 'end_us': end,
                       'type': 'hold' if hold else 'tap',
                       'location': {'kind': 'lane', 'lane': min(3, x * 4 // 512)},
                       'native_ref': f'HitObjects/{index}'})
        sidecar[eid] = {'native_line_number': line_number, 'native_line': line,
                        'native_onset_ms_decimal': p[2],
                        'native_end_ms_decimal': p[5].split(':')[0] if hold else p[2],
                        'native_x': x, 'native_flags': flags}
    require(events, 'empty source chart')
    require(all(a['onset_us'] <= b['onset_us'] for a, b in zip(events, events[1:])), 'unsorted native onsets')
    validate_collisions(events)
    return {'chart_sha256': actual_sha, 'format_version': version, 'events': events,
            'native_sidecar': sidecar, 'timing_points': timing, 'general': general,
            'audio_lead_in_ms_decimal': general.get('AudioLeadIn', '0'),
            'native_beat_storage': 'osu stores audio-relative time, not beat rationals'}

def validate_collisions(events):
    by_lane = defaultdict(list)
    for e in events:
        by_lane[e['location']['lane']].append(e)
    for lane, notes in by_lane.items():
        notes.sort(key=lambda e: (e['onset_us'], e['end_us'], e['event_id']))
        last_onset, active_end = -1, -1
        for e in notes:
            require(e['onset_us'] != last_onset, f'same-lane onset collision: lane {lane}')
            require(e['onset_us'] >= active_end, f'same-lane hold overlap: lane {lane}')
            last_onset = e['onset_us']
            active_end = max(active_end, e['end_us'])

def span_intersects(event, start, end):
    if event['type'] == 'hold':
        return event['onset_us'] < end and event['end_us'] > start
    return start <= event['onset_us'] < end

def spans_cover(start, end, spans):
    """A tap at a right endpoint belongs to the next span; hold end is half-open."""
    if end == start:
        return any(a <= start < b for a, b in spans)
    cursor = start
    for a, b in sorted(spans):
        if a <= cursor < b:
            cursor = max(cursor, b)
            if cursor >= end:
                return True
    return False

def make_regions(duration, start, end, recipe):
    cuts = {0, duration, start, end}
    if recipe == 'complete_reference': cuts.add(start + 4_000_000)
    if recipe == 'multiple_gaps': cuts.update((start + 4_000_000, start + 6_000_000))
    if recipe == 'explicit_blank': cuts.update((start + 5_000_000, start + 6_000_000))
    def state(a):
        if not start <= a < end: return 'complete'
        if recipe == 'complete_reference' and a < start + 4_000_000: return 'complete'
        if recipe == 'multiple_gaps' and start + 4_000_000 <= a < start + 6_000_000: return 'complete'
        if recipe == 'explicit_blank' and start + 5_000_000 <= a < start + 6_000_000: return 'empty'
        return 'editable'
    cuts = sorted(cuts)
    return [{'region_id': f'region-{i}', 'start_us': a, 'end_us': b, 'state': state(a)}
            for i, (a, b) in enumerate(zip(cuts, cuts[1:])) if a < b]

def _can_add(event, regions):
    return spans_cover(event['onset_us'], event['end_us'],
                       [(r['start_us'], r['end_us']) for r in regions if r['state'] == 'editable'])

def _empty_intersections(event, regions):
    return [r['region_id'] for r in regions if r['state'] == 'empty' and span_intersects(event, r['start_us'], r['end_us'])]

def signed_reference(reference, all_events, chart_hash):
    """Source-complete training assumption, not human-certified musical negatives.

    This bounded sampler is a deterministic 40 ms non-note grid, NOT the
    specification's half spectral-flux/half random sampler. It never uses target
    events outside the support to choose evidence, except overlapping hold
    occupancy/head safety at the actual support boundary.
    """
    start, end = reference['span']['start_us'], reference['span']['end_us']
    selected = [{'event_id': e['event_id'], 'time_us': e['onset_us'], 'selected': True}
                for e in reference['events']]
    if reference['coverage'] != 'complete':
        return {'reference_id': reference['reference_id'], 'coverage': 'sparse',
                'selected': selected, 'unselected': [], 'coverage_proof': None,
                'carry_holds': [], 'sampler': 'none_sparse_positive_only'}
    heads = [e['onset_us'] for e in all_events if start - 30_000 <= e['onset_us'] <= end + 30_000]
    holds = [e for e in all_events if e['type'] == 'hold' and span_intersects(e, start, end)]
    candidates = [t for t in range(((start + 39_999) // 40_000) * 40_000, end, 40_000)
                  if all(abs(t - h) > 30_000 for h in heads)
                  and not any(e['onset_us'] <= t < e['end_us'] for e in holds)]
    # Prefer the support interior; do not expose any head outside reference span
    # by letting its ±30ms exclusion alter a boundary label.
    candidates = [t for t in candidates if start + 30_000 <= t < end - 30_000]
    candidates = candidates[:128]
    carry = [_project(e) for e in holds if e['onset_us'] < start]
    return {'reference_id': reference['reference_id'], 'coverage': 'complete',
            'selected': selected, 'unselected': [{'time_us': t, 'selected': False,
                'meaning': 'not_selected_in_this_complete_source_chart'} for t in candidates],
            'carry_holds': carry,
            'coverage_proof': {'kind': 'constructed_complete_source_coverage',
                'chart_sha256': chart_hash, 'span': copy.deepcopy(reference['span']),
                'all_onset_event_ids': [e['event_id'] for e in reference['events']],
                'native_source_full_file_required': True, 'human_alignment_reviewed': False},
            'sampler': 'deterministic_40ms_non_note_grid_not_spectral_flux'}

def quantize_ms(time_us):
    # Half-up integer rounding, fixed across platforms. Originals never pass here.
    return ((time_us + 500) // 1000) * 1000

def model_target_view(targets, observed, regions):
    anchor_times = sorted({e['onset_us'] for e in observed})
    out = []
    for e in targets:
        m = _project(e)
        copy_time = e['onset_us'] if e['onset_us'] in anchor_times else None
        m['onset_us'] = e['onset_us'] if copy_time is not None else quantize_ms(e['onset_us'])
        m['end_us'] = quantize_ms(e['end_us']) if e['type'] == 'hold' else m['onset_us']
        require(m['type'] != 'hold' or m['end_us'] > m['onset_us'], 'quantization produced nonpositive hold')
        require(_can_add(m, regions), 'quantization crosses permission boundary')
        m['time_encoding'] = {'kind': 'COPY_TIME', 'time_us': copy_time} if copy_time is not None else {'kind': 'FREE_TIME'}
        out.append(m)
    validate_collisions([*observed, *out])
    return out

def make_segments(regions, targets, observed, window, max_groups=256, max_events=512):
    """Core ownership <=8s; adaptively split on time, never discard complete groups."""
    output = []
    def segment(start, end, rid):
        events = sorted([e for e in targets if start <= e['onset_us'] < end],
                        key=lambda e: (e['onset_us'], e['location']['lane'], e['event_id']))
        grouped = defaultdict(list)
        for e in events: grouped[e['onset_us']].append(e)
        if len(events) > max_events or len(grouped) > max_groups:
            mid = ((start + end) // 2 // 1000) * 1000
            if mid <= start or mid >= end:
                raise Unsupported('resource_exhausted: unsplittable onset group')
            segment(start, mid, rid); segment(mid, end, rid); return
        known_times = sorted({e['onset_us'] for e in observed if start <= e['onset_us'] < end})
        groups = []
        for onset, members in grouped.items():
            require(1 <= len(members) <= 4, 'nonempty 4K group size violation')
            uses_copy = any(e['time_encoding']['kind'] == 'COPY_TIME' for e in members)
            if uses_copy:
                require(onset in known_times, 'COPY_TIME anchor outside current core')
                time = {'kind': 'COPY_TIME', 'anchor_index': known_times.index(onset), 'onset_us': onset}
            else:
                delta = onset - start
                require(delta % 1000 == 0, 'free onset not on relative 1ms grid')
                time = {'kind': 'FREE_TIME', 'coarse_10ms': delta // 10_000,
                        'residual_ms': delta % 10_000 // 1000, 'onset_us': onset}
            notes = []
            for event in members:
                n = {k: copy.deepcopy(event[k]) for k in ('event_id', *MODEL_EVENT_FIELDS)}
                n['action'] = 'CONTINUE'
                if event['type'] == 'hold':
                    ep = event['end_us']
                    n['endpoint_target'] = {'block_2s': ep // 2_000_000,
                        'coarse_10ms': ep % 2_000_000 // 10_000,
                        'residual_ms': ep % 10_000 // 1000,
                        'reveal_detail_only_after': 'block_2s_prediction'}
                notes.append(n)
            groups.append({'outer_action': 'NEW_GROUP', 'time': time, 'notes': notes,
                           'inner_stop': 'END_GROUP'})
        # Only known immutable holds belong to the prefix. Previously generated
        # target carry is supervision/state from earlier owned groups, separate.
        carry_o = [_project(e, MODEL_EVENT_FIELDS) for e in observed
                   if e['type'] == 'hold' and e['onset_us'] < start < e['end_us']]
        carry_y = [_project(e, MODEL_EVENT_FIELDS) for e in targets
                   if e['type'] == 'hold' and e['onset_us'] < start < e['end_us']]
        output.append({'segment_id': f'{rid}:{start}:{end}', 'region_id': rid,
            'start_us': start, 'end_us': end, 'known_anchor_times_us': known_times,
            'observed_active_holds': carry_o, 'past_generated_active_holds': carry_y,
            'groups': groups, 'outer_stop': 'END_GAP'})
    for r in regions:
        if r['state'] != 'editable': continue
        a, b = max(r['start_us'], window[0]), min(r['end_us'], window[1])
        while a < b:
            e = min(a + 8_000_000, b)
            segment(a, e, r['region_id']); a = e
    return output

def build_artifact(manifest_request, source, split_row, source_files, recipe):
    if recipe not in RECIPES:
        raise Unsupported(f'unsupported task kind: {recipe}')
    require(manifest_request['split'] == source['split'] == split_row['split'] == 'train', 'non-train source denied')
    for field in ('chart_key', 'song_group', 'audio_version_group'):
        require(manifest_request[field] == source[field] == split_row[field], f'{field} provenance mismatch')
    for field in ('chart_sha256', 'audio_sha256'):
        require(manifest_request[field] == source[field], f'{field} manifest mismatch')
    require(source['structural_pass'], 'source structural audit failed')
    require(file_sha(source['audio_path']) == source['audio_sha256'], 'audio SHA256 mismatch')
    duration = decimal_us(source['audio_duration_seconds'], 1_000_000)
    parsed = parse_osu(Path(source['chart_path']).read_bytes(), source['chart_sha256'], duration)
    # Verify the original AudioFilename points to the exact same source bytes.
    audio_ref = Path(source['chart_path']).parent / parsed['general']['AudioFilename']
    require(audio_ref.is_file() and file_sha(audio_ref) == source['audio_sha256'], 'native chart/audio pairing mismatch')
    start = decimal_us(manifest_request['start_seconds'], 1_000_000)
    end = decimal_us(manifest_request['end_seconds'], 1_000_000)
    require(0 <= start < end <= duration, 'requested interval out of bounds')
    require(end - start == 12_000_000, 'this recipe pack requires exactly 12-second source intervals')
    regions = make_regions(duration, start, end, recipe)
    observed, native_targets, suppressed = [], [], []
    for e in parsed['events']:
        onset_in_window = start <= e['onset_us'] < end
        empty = _empty_intersections(e, regions)
        if empty:
            if not onset_in_window:
                raise Unsupported('blank overlay conflicts with an immutable outside-window hold')
            suppressed.append({'event': _project(e), 'reason': 'explicit_blank_remove_whole_hidden_object',
                               'empty_region_ids': empty})
            continue
        temporal_anchor = recipe == 'sparse_anchors' and (e['onset_us'] - start) % 2_000_000 < 300_000
        if onset_in_window and _can_add(e, regions) and not temporal_anchor:
            native_targets.append(_project(e))
        else:
            observed.append(dict(_project(e), locked=True))
    references = []
    if recipe in ('complete_reference', 'sparse_anchors'):
        ref_end = start + 4_000_000 if recipe == 'complete_reference' else end
        pool = parsed['events'] if recipe == 'complete_reference' else observed
        ref_events = [_project(e) for e in pool if start <= e['onset_us'] < ref_end]
        while len(ref_events) > 128:
            ref_end = start + ((ref_end - start) // 2 // 1000) * 1000
            require(ref_end > start, 'reference cannot be shortened without dropping a full group')
            ref_events = [_project(e) for e in pool if start <= e['onset_us'] < ref_end]
        references.append({'reference_id': f"{manifest_request['request_id']}:support",
                           'audio_id': source['audio_sha256'], 'mode': 'mania_4k',
                           'span': {'start_us': start, 'end_us': ref_end},
                           'coverage': 'complete' if recipe == 'complete_reference' else 'sparse',
                           'events': ref_events, 'native_chart_ref': f"sha256:{source['chart_sha256']}"})
    request = {'schema_version': 'beatfcn.intent.v1', 'request_id': manifest_request['request_id'],
               'audio_id': source['audio_sha256'], 'audio_duration_us': duration,
               'mode': 'mania_4k', 'observed_events': observed, 'regions': regions,
               'references': references, 'controls': {}, 'timing_map_ref': None,
               'scene_ref': None, 'seed': 20261022, 'requested_candidates': 1}
    signed = [signed_reference(r, parsed['events'], source['chart_sha256']) for r in references]
    model_targets = model_target_view(native_targets, observed, regions)
    segments = make_segments(regions, model_targets, observed, (start, end))
    provenance = {k: copy.deepcopy(source[k]) for k in ('chart_key', 'song_group', 'audio_version_group',
        'chart_path', 'chart_sha256', 'audio_path', 'audio_sha256', 'source_url', 'dataset',
        'dataset_revision', 'source_shard', 'source_sample_key', 'source_rank_status',
        'human_authorship_evidence', 'rights_status', 'execution_gate')}
    provenance.update({'split': 'train', 'split_row': split_row, 'source_files': source_files,
        'original_request': manifest_request, 'native_format_version': parsed['format_version'],
        'audio_duration_provenance': 'existing full decode metadata; not re-estimated by bridge',
        'native_timing_points': parsed['timing_points'],
        'audio_lead_in_ms_decimal': parsed['audio_lead_in_ms_decimal'],
        'time_policy': 'native audio-relative milliseconds; never add AudioLeadIn or snap to BPM',
        'immutable_original_sha256': parsed['chart_sha256'],
        'native_sidecar_policy': 'original full .osu bytes retained at verified chart_path; native_ref resolves HitObjects ordinal',
        'canonical_source_event_digest': digest(parsed['events']),
        'independent_audio_alignment_review': 'not_performed', 'gold': False,
        'mask_origin': 'deterministic_constructed_training_task_not_real_edit_session',
        'reference_completion_basis': 'published_full_chart_assumed_complete_for_constructed_support_not_independently_reviewed',
        'timing_map_visible_to_model': False,
        'release_gap_us': 0, 'native_renderer_validation': 'not_performed'})
    result = {'bridge_version': VERSION, 'status': 'supported_local_candidate', 'recipe': recipe,
        'window': {'start_us': start, 'end_us': end}, 'request': request,
        'reference_evidence': signed,
        'supervision': {'native_additions': native_targets, 'model_additions': model_targets,
                        'suppressed_originals': suppressed, 'segments': segments,
                        'selection_bce': None, 'counterfactual': None},
        'provenance': provenance,
        'unsupported': ['phira', 'real_human_edit_session', 'same_audio_two_chart_counterfactual',
                        'spectral_flux_reference_sampler', 'full_song_from_scratch_rollout',
                        'full_decoder_or_training', 'independent_human_alignment_gold']}
    validate_artifact(result, original_events=parsed['events'])
    return result

def _validate_event(e, duration, locked=False):
    require(set(e) >= set(EVENT_FIELDS), 'incomplete canonical event')
    require(isinstance(e['event_id'], str) and bool(e['event_id']), 'bad event ID')
    require(e['type'] in ('tap', 'hold'), 'unsupported event type')
    require(e['location'].get('kind') == 'lane' and type(e['location'].get('lane')) is int
            and 0 <= e['location']['lane'] < 4, 'invalid 4K lane')
    a, b = e['onset_us'], e['end_us']
    require(type(a) is int and type(b) is int and 0 <= a < duration and b <= duration, 'bad event bounds')
    require(b > a if e['type'] == 'hold' else b == a, 'bad event length')
    require(isinstance(e['native_ref'], str) and e['native_ref'].startswith('HitObjects/'), 'missing native reference')
    if locked: require(e.get('locked') is True, 'observed object must be immutable')

def validate_request(request):
    # This bridge supports a strict subset of the supplied JSON schema. It does
    # not claim to replace the schema for arbitrary Phira or external requests.
    allowed = {'schema_version', 'request_id', 'audio_id', 'audio_duration_us', 'mode',
               'observed_events', 'regions', 'references', 'controls', 'timing_map_ref',
               'scene_ref', 'seed', 'requested_candidates'}
    require(set(request) == allowed, 'unexpected or missing request fields')
    require(request['schema_version'] == 'beatfcn.intent.v1' and request['mode'] == 'mania_4k', 'unsupported request schema/mode')
    require(request['timing_map_ref'] is None and request['scene_ref'] is None, 'unsupported timing/scene input')
    require(request['controls'] == {}, 'derived controls not implemented; must remain missing')
    duration = request['audio_duration_us']
    require(type(duration) is int and duration > 0, 'bad audio duration')
    cursor = 0; region_ids = set()
    for r in request['regions']:
        require(set(r) == {'region_id', 'start_us', 'end_us', 'state'}, 'invalid region fields')
        require(r['region_id'] not in region_ids, 'duplicate region ID'); region_ids.add(r['region_id'])
        require(type(r['start_us']) is int and type(r['end_us']) is int
                and r['start_us'] == cursor < r['end_us'] <= duration, 'regions must tile audio without gaps/overlap')
        require(r['state'] in ('editable', 'complete', 'empty'), 'bad region state')
        cursor = r['end_us']
    require(cursor == duration, 'regions do not cover full audio')
    seen = set()
    for e in request['observed_events']:
        _validate_event(e, duration, locked=True)
        require(e['event_id'] not in seen, 'duplicate observed ID'); seen.add(e['event_id'])
        require(not _empty_intersections(e, request['regions']), 'locked original conflicts with empty region')
    validate_collisions(request['observed_events'])
    ref_ids = set()
    for ref in request['references']:
        require(ref['reference_id'] not in ref_ids, 'duplicate reference ID'); ref_ids.add(ref['reference_id'])
        require(ref['coverage'] in ('complete', 'sparse'), 'unknown reference coverage')
        require(ref['mode'] == 'mania_4k' and ref['audio_id'] == request['audio_id'], 'this subset requires same-audio 4K references')
        a, b = ref['span']['start_us'], ref['span']['end_us']
        require(0 <= a < b <= duration, 'reference bounds invalid')
        require(len(ref['events']) <= 128, 'reference positive cap exceeded')
        ids = set()
        for e in ref['events']:
            _validate_event(e, duration)
            require(a <= e['onset_us'] < b, 'reference head outside support')
            require(e['event_id'] not in ids, 'duplicate reference event'); ids.add(e['event_id'])
            require(e['event_id'] in seen, 'reference event is not an exposed original')
        if ref['coverage'] == 'complete':
            require(not any(a < r['end_us'] and b > r['start_us'] and r['state'] == 'editable'
                            for r in request['regions']), 'complete support overlaps editable target')
    return True

def validate_artifact(artifact, original_events=None):
    require(artifact['bridge_version'] == VERSION, 'unknown bridge version')
    require(artifact['status'] == 'supported_local_candidate', 'unsupported artifact')
    require(artifact['recipe'] in RECIPES, 'unknown construction recipe')
    request, sup, prov = artifact['request'], artifact['supervision'], artifact['provenance']
    validate_request(request)
    require(prov['split'] == prov['split_row']['split'] == 'train', 'non-train artifact')
    require(prov['gold'] is False and prov['independent_audio_alignment_review'] == 'not_performed', 'unsupported gold claim')
    require(prov['timing_map_visible_to_model'] is False, 'unexpected oracle timing map')
    require(sup['selection_bce'] is None and sup['counterfactual'] is None, 'unimplemented labels supplied')
    original = {e['event_id']: _project(e) for e in original_events} if original_events is not None else None
    if original is not None:
        require(digest(original_events) == prov['canonical_source_event_digest'], 'canonical source mismatch')
        require(len(original) == len(original_events), 'duplicate original ID')
    partition = {}; obs = request['observed_events']; targets = sup['native_additions']
    for category, events in (('observed', obs), ('target', targets), ('suppressed', [x['event'] for x in sup['suppressed_originals']])):
        for e in events:
            _validate_event(e, request['audio_duration_us'], locked=category == 'observed')
            eid = e['event_id']; require(eid not in partition, 'source event belongs to multiple roles'); partition[eid] = category
            if original is not None:
                require(eid in original and _project(e) == original[eid], 'immutable native object changed')
            if category == 'target':
                require(_can_add(e, request['regions']), 'native target outside editable permissions')
                require(artifact['window']['start_us'] <= e['onset_us'] < artifact['window']['end_us'], 'target ownership outside request')
    if original is not None:
        require(set(partition) == set(original), 'source event omitted or invented')
    for s in sup['suppressed_originals']:
        require(s['reason'] == 'explicit_blank_remove_whole_hidden_object', 'unlicensed target suppression')
        require(s['empty_region_ids'] == _empty_intersections(s['event'], request['regions'])
                and bool(s['empty_region_ids']), 'suppressed object does not intersect explicit blank')
    validate_collisions([*obs, *targets])
    expected_model = model_target_view(targets, obs, request['regions'])
    require(expected_model == sup['model_additions'], 'model view is not native-derived quantized view')
    expected_segments = make_segments(request['regions'], expected_model, obs,
                                      (artifact['window']['start_us'], artifact['window']['end_us']))
    require(expected_segments == sup['segments'], 'onset groups/endpoint/carry ownership mismatch')
    target_ids = {e['event_id'] for e in targets}
    ref_evidence = {r['reference_id']: r for r in artifact['reference_evidence']}
    require(len(ref_evidence) == len(artifact['reference_evidence']) == len(request['references']), 'reference evidence mismatch')
    all_events = original_events if original_events is not None else [*obs, *targets, *[x['event'] for x in sup['suppressed_originals']]]
    for ref in request['references']:
        require(not ({e['event_id'] for e in ref['events']} & target_ids), 'hidden target leaked into reference')
        # Source order, not category order, is needed for exact coverage IDs.
        ordered = sorted(all_events, key=lambda e: int(e['native_ref'].split('/')[-1]))
        if ref['coverage'] == 'complete':
            a, b = ref['span']['start_us'], ref['span']['end_us']
            require(ref['events'] == [_project(e) for e in ordered if a <= e['onset_us'] < b], 'complete reference omitted an original onset')
        else:
            known = {e['event_id']: e for e in obs}
            require(all(e == _project(known[e['event_id']]) for e in ref['events']), 'sparse reference rewrote an original')
        require(ref_evidence[ref['reference_id']] == signed_reference(ref, ordered, prov['chart_sha256']), 'selected/unselected evidence mismatch')
    return True

def model_condition(artifact):
    """Only permitted conditioning; NO labels, source files, IDs of songs, counts,
    native sidecars, oracle timing maps, endpoint targets or hidden candidates.
    audio_id is an audio-only routing key, not an embedding feature. Native
    event IDs (which encode source ordinals) are deliberately removed.
    """
    request = artifact['request']
    refs = []
    evidence_by_id = {e['reference_id']: e for e in artifact['reference_evidence']}
    for reference_index, r in enumerate(request['references']):
        ev = evidence_by_id[r['reference_id']]
        refs.append({'reference_id': f'reference-{reference_index}', 'span': copy.deepcopy(r['span']),
            'coverage': r['coverage'], 'events': [_project(e, MODEL_EVENT_FIELDS) for e in r['events']],
            'selected_times_us': [x['time_us'] for x in ev['selected']],
            'unselected_times_us': [x['time_us'] for x in ev['unselected']],
            'carry_holds': [_project(e, MODEL_EVENT_FIELDS) for e in ev['carry_holds']]})
    return {'audio_id': request['audio_id'], 'audio_duration_us': request['audio_duration_us'],
            'mode': request['mode'], 'regions': copy.deepcopy(request['regions']),
            'observed_events': [_project(e, MODEL_EVENT_FIELDS) for e in request['observed_events']],
            'references': refs, 'controls': {}, 'timing_map': None,
            'window': copy.deepcopy(artifact['window'])}

def completed_group_attributes(group):
    """No outer hidden state, endpoint reveal request, or future-field leakage."""
    return {'onset_us': group['time']['onset_us'],
            'notes': [{k: copy.deepcopy(n[k]) for k in MODEL_EVENT_FIELDS} for n in group['notes']]}

def teacher_forcing_steps(artifact):
    """Explicit two-object records: input (only completed past), target (loss).

    Endpoint block/coarse/fine are loss targets, not input fields. This is data
    only, not an executable decoder or its attention mask. Consumer must keep
    target separate; do not concatenate these records as model features.
    """
    prefix = model_condition(artifact)
    earlier_groups = []
    for segment in artifact['supervision']['segments']:
        groups = segment['groups']
        for i in range(len(groups) + 1):
            yield {'input': {'condition': copy.deepcopy(prefix),
                'segment': {k: copy.deepcopy(segment[k]) for k in ('segment_id', 'start_us', 'end_us', 'known_anchor_times_us', 'observed_active_holds')},
                'completed_previous_groups': copy.deepcopy(earlier_groups),
                'current_gap_past_groups': [completed_group_attributes(g) for g in groups[:i]],
                'past_generated_active_holds': copy.deepcopy(segment['past_generated_active_holds'])},
                'target': copy.deepcopy(groups[i]) if i < len(groups) else {'outer_action': 'END_GAP'}}
        earlier_groups.extend(completed_group_attributes(g) for g in groups)

def _source_path(path, source_root=None):
    path = Path(path)
    if path.is_absolute():
        return path
    require(source_root is not None, 'relative source refs require source_root')
    root = Path(source_root).resolve()
    resolved = (root / path).resolve()
    require(resolved == root or root in resolved.parents, 'source ref escapes source_root')
    return resolved

def verify_source_files(artifact, source_root=None):
    p = artifact['provenance']
    require(p['split'] == 'train', 'refuse non-train source')
    for label, f in p['source_files'].items():
        require(_source_path(f['path'], source_root).is_file(), f'missing {label} source file')
        require(file_sha(_source_path(f['path'], source_root)) == f['sha256'], f'{label} source manifest hash mismatch')
    require(file_sha(_source_path(p['chart_path'], source_root)) == p['chart_sha256'] == p['immutable_original_sha256'], 'immutable chart changed')
    require(file_sha(_source_path(p['audio_path'], source_root)) == p['audio_sha256'], 'immutable audio changed')
    split_rows = rows(_source_path(p['source_files']['split_manifest']['path'], source_root))
    exact = [x for x in split_rows if x['chart_key'] == p['chart_key']]
    require(len(exact) == 1 and exact[0] == p['split_row'], 'split provenance row changed')
    require(all(x['split'] not in ('validation', 'test', 'sealed_test') for x in split_rows
                if x['song_group'] == p['song_group']), 'song group crosses held-out split')
    parsed = parse_osu(_source_path(p['chart_path'], source_root).read_bytes(), p['chart_sha256'], artifact['request']['audio_duration_us'])
    native_audio = _source_path(p['chart_path'], source_root).parent / parsed['general']['AudioFilename']
    require(native_audio.is_file() and file_sha(native_audio) == p['audio_sha256'], 'native audio source pairing changed')
    require(parsed['timing_points'] == p['native_timing_points'], 'native timing sidecar changed')
    require(parsed['audio_lead_in_ms_decimal'] == p['audio_lead_in_ms_decimal'], 'AudioLeadIn sidecar changed')
    return parsed

def load_artifact(path, verify_sources=True, source_root=None):
    path = Path(path)
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8') as f:
        artifact = json.load(f)
    parsed = verify_source_files(artifact, source_root) if verify_sources else None
    validate_artifact(artifact, parsed['events'] if parsed else None)
    return artifact

def resolve_native_sidecar(artifact, event_id_to_resolve, source_root=None):
    """Returns immutable original row without rewriting chart data."""
    parsed = verify_source_files(artifact, source_root)
    require(event_id_to_resolve in parsed['native_sidecar'], 'unknown source event ID')
    return copy.deepcopy(parsed['native_sidecar'][event_id_to_resolve])

def restore_original_bytes(artifact, source_root=None):
    """Read the hash-verified original bytes for exact roundtrip; no generated patch."""
    p = artifact['provenance']
    raw = _source_path(p['chart_path'], source_root).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == p['immutable_original_sha256'], 'immutable original missing or changed')
    return raw

def build_all(source_root, output_root):
    source_root, output_root = Path(source_root).resolve(), Path(output_root).resolve()
    # Never allow the CLI to write into a source tree or S0a. The package is an
    # offline staging area; integration must be a separate reviewed operation.
    package_root = Path(__file__).resolve().parents[1]
    require(output_root == package_root or package_root in output_root.parents, 'output must remain inside beatfcn_s0b_data staging directory')
    files = {'request_manifest': source_root / 'requests_32.jsonl',
             'shortlist': source_root / 'shortlist.jsonl',
             'split_manifest': source_root / 'rhythm_clean_v1/manifests/splits.jsonl'}
    source_files = {key: {'path': str(path), 'sha256': file_sha(path)} for key, path in files.items()}
    requests, shortlist, splits = rows(files['request_manifest']), rows(files['shortlist']), rows(files['split_manifest'])
    require(len(requests) == 32, 'expected exactly 32 source requests')
    for key in ('request_id', 'song_group', 'audio_sha256'):
        require(len({r[key] for r in requests}) == 32, f'not one unique {key} per request')
    source_by_key = {s['chart_key']: s for s in shortlist}
    split_by_key = {s['chart_key']: s for s in splits}
    require(len(source_by_key) == len(shortlist), 'duplicate shortlist chart key')
    require(len(split_by_key) == len(splits), 'duplicate split chart key')
    (output_root / 'requests').mkdir(parents=True, exist_ok=True)
    entries, failures, counts = [], [], Counter()
    for i, request in enumerate(requests):
        recipe = RECIPES[i % len(RECIPES)]
        try:
            source, split = source_by_key[request['chart_key']], split_by_key[request['chart_key']]
            require(all(x['split'] not in ('validation', 'test', 'sealed_test') for x in splits
                        if x['song_group'] == request['song_group']), 'selected song group crosses held-out split')
            artifact = build_artifact(request, source, split, source_files, recipe)
            filename = f"requests/{request['request_id']}.json.gz"
            data = gzip.compress(canonical_bytes(artifact), compresslevel=9, mtime=0)
            (output_root / filename).write_bytes(data)
            sup, inp = artifact['supervision'], artifact['request']
            entry = {'request_id': request['request_id'], 'path': filename,
                     'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data),
                     'recipe': recipe, 'song_group': request['song_group'], 'split': 'train',
                     'observed_events': len(inp['observed_events']),
                     'target_events': len(sup['native_additions']),
                     'target_holds': sum(e['type'] == 'hold' for e in sup['native_additions']),
                     'groups': sum(len(s['groups']) for s in sup['segments']),
                     'segments': len(sup['segments']),
                     'suppressed_whole_objects': len(sup['suppressed_originals']),
                     'reference_selected': sum(len(e['selected']) for e in artifact['reference_evidence']),
                     'reference_unselected': sum(len(e['unselected']) for e in artifact['reference_evidence'])}
            entries.append(entry)
            counts.update({k: entry[k] for k in ('observed_events', 'target_events', 'target_holds', 'groups', 'segments', 'suppressed_whole_objects', 'reference_selected', 'reference_unselected')})
        except (DataError, KeyError, OSError) as exc:
            failures.append({'request_id': request['request_id'], 'recipe': recipe,
                             'status': 'unsupported_or_quarantined', 'reason': str(exc)})
    manifest = {'bridge_version': VERSION, 'entries': entries, 'unsupported': failures,
                'source_files': source_files, 'source_requests': 32,
                'supported_requests': len(entries), 'unique_train_songs': len({x['song_group'] for x in entries}),
                'totals': dict(counts), 'recipe_counts': dict(Counter(x['recipe'] for x in entries)),
                'source_audio_seconds': 384, 'human_alignment_reviewed': False,
                'gold': False, 'training_performed': False, 'upload_performed': False,
                'rights_verified': False, 'sealed_test_or_validation_bodies_read': False}
    (output_root / 'manifest.json').write_bytes(canonical_bytes(manifest) + b'\n')
    return manifest

def endpoint_teacher_forcing_fields(note):
    """Conditional hold-end labels, not audio features or a decoder.

    First predict a block using only the chosen head/type/location. Only after
    that decision may the consumer fetch that block's details. Teacher forcing
    substitutes gold choices at their corresponding subsequent stages only.
    """
    require(note['type'] == 'hold' and 'endpoint_target' in note, 'endpoint fields require an atomic hold target')
    endpoint = note['endpoint_target']
    head = {k: copy.deepcopy(note[k]) for k in ('onset_us', 'type', 'location')}
    block, coarse, fine = endpoint['block_2s'], endpoint['coarse_10ms'], endpoint['residual_ms']
    require(note['end_us'] == block * 2_000_000 + coarse * 10_000 + fine * 1000,
            'hold endpoint hierarchy mismatch')
    return [
        {'stage': 'block_2s', 'input': {'chosen_note_head': copy.deepcopy(head)}, 'target': block},
        {'stage': 'coarse_10ms', 'input': {'chosen_note_head': copy.deepcopy(head),
            'chosen_block_2s': block, 'detail_audio_block': block}, 'target': coarse},
        {'stage': 'residual_ms', 'input': {'chosen_note_head': copy.deepcopy(head),
            'chosen_block_2s': block, 'chosen_coarse_10ms': coarse,
            'detail_audio_block': block}, 'target': fine},
    ]
