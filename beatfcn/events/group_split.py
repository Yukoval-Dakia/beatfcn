"""Song-group deduplication and the deterministic 80/10/10 split by group (spec 6.3).

Grouping is a union-find over evidence keys, each recorded for review:
- audio:<sha256>       identical decoded/recording hash supplied by the caller (exact recording);
- fp:<key>             any caller-supplied fingerprint key (e.g. chromaprint / silence-trimmed PCM hash);
- title:<norm>         NFKC + lowercase + whitespace/punctuation stripped title (composition proxy);
- work:<norm>          the same with bracketed qualifiers removed ("(TV size)", "[Speed Up]") so speed
                       variants / cuts / re-uploads merge.
Over-merging is the intended failure mode (it can only cost data, never leak a song across splits).

Split: a group's split is a pure function of (seed, group_id) through SHA-256 -> u in [0,1), thresholds
0.8 / 0.9. It does not depend on row order, and adding unrelated rows never moves an existing group.
Fractions are therefore approximate; the manifest reports the realised counts.

NOT handled: decoding audio or computing fingerprints (callers pass hashes); cover/remix detection beyond
title normalisation; reviewer overrides (pass them as extra keys).
"""
from __future__ import annotations
from collections import Counter, defaultdict
import hashlib
import re
import unicodedata

SPLITS = ('train', 'val', 'test')
_BRACKETS = re.compile(r'[\(\[\{（【「『<＜][^\)\]\}）】」』>＞]*[\)\]\}）】」』>＞]')
_KEEP = re.compile(r'[^\w぀-ヿ㐀-䶿一-鿿가-힯]')


def normalize_title(s) -> str:
    if not s:
        return ''
    s = unicodedata.normalize('NFKC', str(s)).casefold()
    return _KEEP.sub('', re.sub(r'\s+', '', s)).replace('_', '')


def work_key(s) -> str:
    if not s:
        return ''
    return normalize_title(_BRACKETS.sub('', unicodedata.normalize('NFKC', str(s))))


class _UF:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.p[max(a, b)] = min(a, b)


def evidence_keys(row, id_key='id', title_key='name', audio_key='audio_sha256', fp_keys=()):
    keys = []
    if row.get(audio_key):
        keys.append(f'audio:{row[audio_key]}')
    for k in fp_keys:
        if row.get(k):
            keys.append(f'fp:{k}:{row[k]}')
    t, w = normalize_title(row.get(title_key)), work_key(row.get(title_key))
    if t:
        keys.append(f'title:{t}')
    if w:
        keys.append(f'work:{w}')
    if not keys:
        keys.append(f'chart:{row[id_key]}')   # nothing to group on: singleton, flagged in the manifest
    return keys


def group_rows(rows, id_key='id', title_key='name', audio_key='audio_sha256', fp_keys=()):
    uf, row_keys = _UF(), {}
    for r in rows:
        cid = str(r[id_key])
        if cid in row_keys:
            raise ValueError(f'duplicate chart id {cid}')
        keys = evidence_keys(r, id_key, title_key, audio_key, fp_keys)
        row_keys[cid] = keys
        for k in keys[1:]:
            uf.union(keys[0], k)
        uf.find(keys[0])
    members, ev = defaultdict(list), defaultdict(set)
    for cid, keys in row_keys.items():
        root = uf.find(keys[0])
        members[root].append(cid)
        ev[root].update(keys)
    groups, chart_to_group = {}, {}
    for root, ms in members.items():
        gid = 'g' + hashlib.sha256(min(ev[root]).encode()).hexdigest()[:16]
        groups[gid] = {'members': sorted(ms), 'evidence': sorted(ev[root]),
                       'title_only': not any(k.startswith(('audio:', 'fp:')) for k in ev[root])}
        for c in ms:
            chart_to_group[c] = gid
    return groups, chart_to_group


def split_of(group_id: str, seed: int, fractions=(0.8, 0.1, 0.1)) -> str:
    if abs(sum(fractions) - 1) > 1e-9:
        raise ValueError('fractions must sum to 1')
    u = int.from_bytes(hashlib.sha256(f'{seed}:{group_id}'.encode()).digest()[:8], 'big') / 2 ** 64
    acc = 0.0
    for name, f in zip(SPLITS, fractions):
        acc += f
        if u < acc:
            return name
    return SPLITS[-1]


def split_manifest(rows, seed: int, fractions=(0.8, 0.1, 0.1), mode_key=None, **kw) -> dict:
    groups, c2g = group_rows(rows, **kw)
    g2s = {g: split_of(g, seed, fractions) for g in sorted(groups)}
    c2s = {c: g2s[g] for c, g in c2g.items()}
    rows_by_id = {str(r[kw.get('id_key', 'id')]): r for r in rows}
    counts = {s: {'groups': sum(v == s for v in g2s.values()), 'charts': sum(v == s for v in c2s.values())} for s in SPLITS}
    if mode_key:
        for s in SPLITS:
            counts[s]['modes'] = dict(Counter(str(rows_by_id[c].get(mode_key)) for c, v in c2s.items() if v == s))
    return {'method': 'union-find(audio|fp|title|work) + sha256(seed:group_id) thresholds', 'seed': seed,
            'fractions': list(fractions), 'counts': counts, 'n_groups': len(groups), 'n_charts': len(c2s),
            'title_only_groups': sum(g['title_only'] for g in groups.values()),
            'groups': {g: dict(groups[g], split=g2s[g]) for g in sorted(groups)}, 'chart_split': c2s}


def straddling_groups(chart_split: dict, chart_to_group: dict) -> dict:
    """Groups whose member charts sit in more than one split under `chart_split` (= song leakage)."""
    seen = defaultdict(set)
    for c, s in chart_split.items():
        if c in chart_to_group:
            seen[chart_to_group[c]].add(s)
    return {g: sorted(v) for g, v in seen.items() if len(v) > 1}


def old_split_leakage(old_split: dict, rows, **kw) -> dict:
    """How many songs the old chart-id split leaks under the new grouping."""
    _, c2g = group_rows(rows, **kw)
    old = {str(k): v for k, v in old_split.items()}
    bad = straddling_groups(old, c2g)
    members = defaultdict(list)
    for c in old:
        if c in c2g:
            members[c2g[c]].append(c)
    leak_charts = sum(len(members[g]) for g in bad)
    eval_leak = sum(1 for g in bad for c in members[g] if old[c] in ('val', 'test'))
    return {'old_charts': len(old), 'old_charts_without_metadata': sum(c not in c2g for c in old),
            'groups_covered': len(members), 'straddling_groups': len(bad),
            'charts_in_straddling_groups': leak_charts, 'val_or_test_charts_sharing_song_with_other_split': eval_leak,
            'combos': dict(Counter('+'.join(v) for v in bad.values()))}
