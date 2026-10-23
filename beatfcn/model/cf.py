"""Strict counterfactual support-swap pairs and L_cf (design §3.3; FIRST_CANDIDATE: selection mechanism).

Two real charts A, B of one recording and mode. After +-30 ms alignment, A-only and B-only heads are
the real, non-nested disagreements. A support span S and a disjoint 8 s target core T are drawn; S
supplies K A-only and K B-only times; T supplies the unique 40 ms cells of A-only and B-only heads
(cells holding both are dropped). View A marks A-only as selected and B-only as unselected; view B the
reverse. Every attribute is masked identically in both views; the two inputs are asserted equal except
for the selected bit. Target candidates enter only the loss.
"""
from __future__ import annotations
import numpy as np
import torch
from torch.nn import functional as F
from . import features as FE

ALIGN_US = 30_000
MASKED_SELECTED = 6  # type index for a selected head with masked attributes (p1.type_e)
UNSELECTED = -1  # -> type index 4 in p1._events


def only(a, b, tol=ALIGN_US):
    """Heads of a with no head of b within tol (sorted int arrays)."""
    if not len(b):
        return a
    j = np.searchsorted(b, a)
    near = np.minimum(np.abs(a - b[np.clip(j, 0, len(b) - 1)]), np.abs(a - b[np.clip(j - 1, 0, len(b) - 1)]))
    return a[near > tol]


def build_pair(heads_a, heads_b, duration_us, rng, k=8, min_target_cells=4, tries=20):
    """Returns dict(c0, support_times, selected_a [bool], cells_a, cells_b) or None."""
    a_only, b_only = only(heads_a, heads_b), only(heads_b, heads_a)
    if len(a_only) < k + min_target_cells or len(b_only) < k + min_target_cells:
        return None
    for _ in range(tries):
        c0 = int(rng.integers(0, max(duration_us - FE.CORE_US, 1))) // FE.CELL_US * FE.CELL_US
        c1 = c0 + FE.CORE_US
        ta = np.unique((a_only[(a_only >= c0) & (a_only < c1)] - c0) // FE.CELL_US)
        tb = np.unique((b_only[(b_only >= c0) & (b_only < c1)] - c0) // FE.CELL_US)
        shared = np.intersect1d(ta, tb)
        ta, tb = np.setdiff1d(ta, shared), np.setdiff1d(tb, shared)
        if len(ta) < min_target_cells or len(tb) < min_target_cells:
            continue
        # support: a disjoint span (outside the target core and its 4 s halo on both sides)
        far = lambda x: (x < c0 - FE.HALO_US) | (x >= c1 + FE.HALO_US)
        sa, sb = a_only[far(a_only)], b_only[far(b_only)]
        if len(sa) < k or len(sb) < k:
            continue
        pa = np.sort(rng.choice(sa, size=k, replace=False))
        pb = np.sort(rng.choice(sb, size=k, replace=False))
        times = np.concatenate([pa, pb])
        order = np.argsort(times, kind='stable')
        sel_a = np.concatenate([np.ones(k, bool), np.zeros(k, bool)])[order]
        return {'c0': c0, 'support': times[order], 'selected_a': sel_a, 'cells_a': ta, 'cells_b': tb}
    return None


def views(pair):
    """Two strict views [N, 9] that differ only in the selected bit (type 6 vs -1 and flag column 7)."""
    rel = (pair['support'] - pair['c0']) / FE.SEC
    base = np.zeros((len(rel), 9), np.float32)
    base[:, 0] = rel
    base[:, 2:5] = -1  # lane, line, side masked
    out = []
    for sel in (pair['selected_a'], ~pair['selected_a']):
        v = base.copy()
        v[:, 1] = np.where(sel, MASKED_SELECTED, UNSELECTED)
        v[:, 7] = sel.astype(np.float32)
        out.append(v)
    same = [0, 2, 3, 4, 5, 6, 8]
    assert np.array_equal(out[0][:, same], out[1][:, same]), 'CF views differ beyond the selected bit'
    assert np.array_equal(out[0][:, 1] == MASKED_SELECTED, ~(out[1][:, 1] == MASKED_SELECTED))
    return out


def collate_pairs(pairs, songs, taps, local_slice):
    """Batch for P1Model.cf_loss: target local taps, two reference views, shared ref audio, target cells."""
    from .batch import _slice
    B = len(pairs)
    N = max(len(p['support']) for p in pairs)
    Dt = songs[0].taps[taps[0]].shape[-1]
    quant = all(hasattr(s.taps[l], 'raw') for s in songs for l in taps)
    tdt = torch.int8 if quant else torch.float32
    C = FE.CORE_US // FE.CELL_US
    bt = {'audio': torch.zeros(B, len(taps), FE.LOCAL_CELLS, 8, Dt, dtype=tdt), 'refs_a': torch.zeros(B, N, 9), 'refs_b': torch.zeros(B, N, 9),
          'ref_audio': torch.zeros(B, N, len(taps), 8, Dt, dtype=tdt), 'refs_valid': torch.zeros(B, N, dtype=torch.bool),
          'cells_a': torch.zeros(B, C, dtype=torch.bool), 'cells_b': torch.zeros(B, C, dtype=torch.bool)}
    for i, (p, song) in enumerate(zip(pairs, songs)):
        cell0 = (p['c0'] - FE.HALO_US) // FE.CELL_US
        for j, l in enumerate(taps):
            a, _ = _slice(song.taps[l], cell0, FE.LOCAL_CELLS)
            bt['audio'][i, j] = torch.from_numpy(a)
        va, vb = views(p)
        n = len(va)
        bt['refs_a'][i, :n], bt['refs_b'][i, :n] = torch.from_numpy(va), torch.from_numpy(vb)
        bt['refs_valid'][i, :n] = True
        cells = (p['support'] // FE.CELL_US).astype(np.int64)
        for j, l in enumerate(taps):
            arr = song.taps[l]
            for r, cidx in enumerate(cells):
                a, _ = _slice(arr, int(cidx), 1)
                bt['ref_audio'][i, r, j] = torch.from_numpy(a[0])
        bt['cells_a'][i, torch.from_numpy(p['cells_a'])] = True
        bt['cells_b'][i, torch.from_numpy(p['cells_b'])] = True
    if quant:  # same compact transport as collate: dequantised on the device (batch.dequant)
        sc = [np.broadcast_to(np.asarray(songs[0].taps[l].scale, np.float32), (Dt,)) for l in taps]
        bt['tap_scale'] = torch.from_numpy(np.stack(sc).copy())
    return bt


def cf_loss(model, bt, margin=0.5):
    """L_cf = 0.5 [softplus(m - s_A(T_A) + s_A(T_B)) + softplus(m - s_B(T_B) + s_B(T_A))], unique-cell means."""
    from .batch import dequant
    dequant(bt)
    ptr = model.ptr_features(bt['audio'])
    h0 = FE.HALO_US // FE.CELL_US
    core = ptr[:, h0:h0 + FE.CORE_US // FE.CELL_US]
    za = model.intent_only(bt['refs_a'], bt['ref_audio'], bt['refs_valid'])
    zb = model.intent_only(bt['refs_b'], bt['ref_audio'], bt['refs_valid'])
    sa, sb = model.selection_field(core, za), model.selection_field(core, zb)
    ma, mb = bt['cells_a'].float(), bt['cells_b'].float()
    mean = lambda s, m: (s * m).sum(-1) / m.sum(-1).clamp_min(1)
    a_a, a_b, b_b, b_a = mean(sa, ma), mean(sa, mb), mean(sb, mb), mean(sb, ma)
    loss = 0.5 * (F.softplus(margin - a_a + a_b) + F.softplus(margin - b_b + b_a)).mean()
    acc = ((a_a > a_b) & (b_b > b_a)).float().mean()
    return loss, acc
