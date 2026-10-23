"""Corpus access for training/probes: manifest rows, canon charts, song caches (DATA_RULES §6).

Model-view time = native time + offset_decision.shift_ms + the per-(source,codec) constant c
(DATA_RULES v2: osu c = +14 ms for both mp3 and vorbis, Phira -2..-6 ms; export subtracts c).
Native charts on the Volume are never modified. Charts whose shifted events leave the audio
are skipped by the request builder rather than clipped.
"""
from __future__ import annotations
from dataclasses import replace
from fractions import Fraction
import json
import os
import numpy as np
from ..events.canonical import Chart, Event
from ..model.batch import SongCache

DATA = '/data'


DUP_REASONS = {'duplicate_object', 'duplicate_object_in_source'}
DUP_MAX_STACK = 4  # higher stacks at one spot are performance/visual (DATA_RULES v4)


def dup_only(r) -> bool:
    """Quarantined solely for exact duplicate notes (kept as written when stacks are <= DUP_MAX_STACK)."""
    reasons = set(r.get('quarantine_reasons') or [])
    return r.get('status') == 'parsed' and bool(reasons) and reasons <= DUP_REASONS


def load_manifest(root=DATA, split=None, kept_only=True, select=True, dup_collapse=True):
    rows = []
    with open(os.path.join(root, 'manifest', 'charts.jsonl')) as f:
        for line in f:
            r = json.loads(line)
            if kept_only and not r.get('kept') and not (dup_collapse and dup_only(r)):
                continue
            if split and r.get('split') != split:
                continue
            if select and not selected_for_training(r):
                continue
            rows.append(r)
    return rows


PHIRA_MIN_RATING = 0.90
PHIRA_MIN_VOTES = 100
OSU_APPROVED = (1, 2)  # ranked, approved; loved (4) excluded (DATA_RULES §1)
# Phira difficulty whitelist (DATA_RULES v4, user decision): only the standard EZ/HD/IN/AT tags. SP charts
# (notes for performance/drawing), 4K/5K/6K mania imitation, ADOFAI and every custom tag are excluded.
PHIRA_TAGS = frozenset({'EZ', 'HD', 'IN', 'AT'})


def phira_tag(r) -> str:
    import re
    return re.split(r'[\s.]', (r.get('difficulty_label') or '?').strip().upper())[0]


def selected_for_training(r) -> bool:
    """Quality selection on top of the ingested superset (DATA_RULES §1, 宁缺毋滥)."""
    if r.get('source') == 'phira':
        return (float(r.get('rating') or 0) >= PHIRA_MIN_RATING and int(float(r.get('rating_count') or 0)) >= PHIRA_MIN_VOTES
                and phira_tag(r) in PHIRA_TAGS)
    if r.get('source') == 'osu':
        return int(r.get('approved') or 0) in OSU_APPROVED
    return False


def model_shift_us(row, apply_codec_constant=False) -> int:
    d = row.get('offset_decision') or {}
    ms = int(d.get('shift_ms') or 0) + (int(d.get('c_ms') or 0) if apply_codec_constant else 0)
    return ms * 1000


def check_stacks(events, max_stack=DUP_MAX_STACK):
    """Exact duplicates (onset, end, type, location) are intentional stacking (hit-sound weight, impact;
    DATA_RULES v4) and are KEPT: with Phira placement left to the charter, a stack and a spread chord of
    the same types are the same model target. Raises ValueError above max_stack (performance stacks)."""
    from collections import Counter
    counts = Counter((e.onset_us, e.end_us, e.type, e.location) for e in events)
    top = max(counts.values(), default=1)
    if top > max_stack:
        raise ValueError(f'stack of {top} identical notes exceeds {max_stack}')
    return top


def npy_shape(path):
    """Shape from a .npy header without mapping or reading the data (network volumes: no mmap, one small read)."""
    with open(path, 'rb') as f:
        version = np.lib.format.read_magic(f)
        shape, _, _ = np.lib.format._read_array_header(f, version)
    return shape


def chart_from_canon(doc: dict, shift_us: int = 0) -> Chart:
    evs = []
    for e in doc['events']:
        e = dict(e)
        e.pop('fake', None)
        ev = Event.from_json(e)
        if shift_us:
            ev = replace(ev, onset_us=ev.onset_us + shift_us, end_us=ev.end_us + shift_us)
        evs.append(ev)
    return Chart(doc['mode'], doc['source_format'], str(doc.get('format_version')), tuple(evs), None,
                 Fraction(0), {}, {'source_name': doc.get('native_id', 'chart')}, (), (), doc.get('source_sha256', ''))


class QuantArray:
    """int8 PCA taps on a memmap, dequantised per slice (only the rows a core reads are touched)."""

    def __init__(self, arr, scale):
        self.arr, self.scale = arr, scale  # scale: per-component (raw PCA units) or a scalar (whitened)
        self.raw = arr  # int8 rows for collate's compact transport (dequantised on the device)
        self.shape, self.dtype = arr.shape, np.float32

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, idx):
        return np.asarray(self.arr[idx], np.float32) * self.scale


class Corpus:
    def __init__(self, root=DATA, taps_key=None, layers=(8, 16, 24), apply_codec_constant=True, pca='', whiten=False):
        self.root, self.taps_key, self.layers = root, taps_key, tuple(layers)
        self.scale = {}
        if pca:  # int8 PCA taps: per-component dequantisation scale
            z = np.load(os.path.join(root, 'pca', f'{pca}.npz'))
            self.scale = {l: z[f'scale_{l}'].astype(np.float32) for l in self.layers}
            if whiten:  # scale = 4 sigma / 127, so q * 4/127 = y / sigma: unit variance per component
                self.scale = {l: np.float32(4 / 127) for l in self.layers}
        self.apply_c = apply_codec_constant
        self._rows = {}
        self._charts = {}
        self._salient = {}

    def train_rows(self, rows):
        out = []
        for r in rows:
            self._rows[r['chart_id']] = r
            out.append({'chart_id': r['chart_id'], 'song': r['audio_sha256'], 'mode': r['mode'],
                        'duration_us': int(round(r['audio_sec'] * 1e6))})
        return out

    def chart(self, chart_id) -> Chart:
        hit = self._charts.get(chart_id)
        if hit is not None:
            return hit
        if len(self._charts) > 512:
            self._charts.clear()
        self._charts[chart_id] = c = self._load_chart(chart_id)
        return c

    def _load_chart(self, chart_id) -> Chart:
        r = self._rows[chart_id]
        with open(os.path.join(self.root, 'canon', r['source'], f"{r['native_id']}.json")) as f:
            doc = json.load(f)
        chart = chart_from_canon(doc, model_shift_us(r, self.apply_c))
        if dup_only(r):
            check_stacks(chart.events)
        return chart

    def usable(self, row) -> bool:
        """False for duplicate-only charts whose stacks fail the collapse guard (checked once, cached)."""
        if not dup_only(row):
            return True
        self._rows[row['chart_id']] = row
        try:
            self.chart(row['chart_id'])
            return True
        except ValueError:
            return False

    def salient_us(self, sha, mel=None):
        """Spectral-flux peaks (same envelope as the offset diagnostic) as candidate unselected times."""
        hit = self._salient.get(sha)
        if hit is None:
            from ..ingest.offset import flux_envelope, PEAK_MIN
            mel = np.load(os.path.join(self.root, 'mel10', f'{sha}.npy')) if mel is None else mel
            env = flux_envelope(np.asarray(mel, np.float32))
            pk = np.where((env[1:-1] > env[:-2]) & (env[1:-1] >= env[2:]) & (env[1:-1] > PEAK_MIN))[0] + 1
            if len(self._salient) > 256:
                self._salient.clear()
            hit = self._salient[sha] = (pk * 10_000).tolist()
        return hit

    def mel(self, sha):
        return np.load(os.path.join(self.root, 'mel10', f'{sha}.npy'), mmap_mode='r')

    def song_info(self, sha):
        """Shape-only description for batch.collate_meta (workers never read the tap cache): cell and
        summary counts follow the encoder's window plan from the mel frame count."""
        from ..audio.memory import plan, SUMMARY_CELLS
        from ..model.batch import SongInfo
        frames = npy_shape(os.path.join(self.root, 'mel10', f'{sha}.npy'))[0]
        cells = plan(frames).cells
        return SongInfo(sha, cells, -(-cells // SUMMARY_CELLS))

    @property
    def tap_dim(self):
        """Tap width from one cached song's header (never a directory listing: 8.7k entries on a network
        volume take tens of seconds, and every worker used to do it)."""
        if not hasattr(self, '_tap_dim'):
            sha = next(iter(self._rows.values()))['audio_sha256'] if self._rows else next(iter(os.listdir(os.path.join(self.root, 'taps', self.taps_key))))
            self._tap_dim = npy_shape(os.path.join(self.root, 'taps', self.taps_key, sha, f'L{self.layers[0]}.npy'))[-1]
        return self._tap_dim

    @property
    def quant(self):
        return bool(self.scale)

    def load_numpy(self, sha):
        """Whole-song arrays for a SongPool: raw int8 taps (dequantised on the device), summary, mel."""
        base = os.path.join(self.root, 'taps', self.taps_key, sha)
        paths = [os.path.join(base, f'L{l}.npy') for l in self.layers] + [os.path.join(base, 'S20.npy'),
                                                                          os.path.join(self.root, 'mel10', f'{sha}.npy')]
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(len(paths)) as ex:
            arrs = list(ex.map(np.load, paths))
        return dict(zip(self.layers, arrs[:-2])), arrs[-2], arrs[-1]

    def song(self, sha, in_memory=False) -> SongCache:
        """Memmapped by default: a core reads a few contiguous slices (16 s local audio, one reference
        range, end blocks), far less than whole-song reads. in_memory=True loads whole files with one
        sequential read each (int8 taps stay int8 in memory and are dequantised per slice)."""
        base = os.path.join(self.root, 'taps', self.taps_key, sha)
        mm = None if in_memory else 'r'
        paths = [os.path.join(base, f'L{l}.npy') for l in self.layers] + [os.path.join(base, 'S20.npy'),
                                                                          os.path.join(self.root, 'mel10', f'{sha}.npy')]
        if in_memory:  # one stream per file: a single network-volume read stream is ~25 MB/s
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(len(paths)) as ex:
                arrs = list(ex.map(np.load, paths))
        else:
            arrs = [np.load(q, mmap_mode=mm) for q in paths]
        taps = dict(zip(self.layers, arrs))
        for l, sc in self.scale.items():
            if taps[l].dtype == np.int8:
                taps[l] = QuantArray(taps[l], sc)
        summary, mel = arrs[-2], arrs[-1]
        return SongCache(taps, summary, mel, int(mel.shape[0] * 10_000))
