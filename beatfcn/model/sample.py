"""Constrained generation for one core with a P1 model, and core-level evaluation.

Generation re-encodes the whole core once per group (no KV cache; P5 replaces this). Every
choice is sampled from a legality-masked distribution: new heads only at editable times and
strictly after the previous group, no duplicate or hold-occupied lane, hold ends inside the
editable run and before the next same-lane head. Greedy decoding is available for
reproducibility (design: inference.greedy_available_for_reproducibility).
"""
from __future__ import annotations
import copy
import math
import numpy as np
import torch
from torch.nn import functional as F
from . import features as FE
from .batch import collate, dequant, MAX_NOTES
from .p1 import P1Model, NEG


def _pick(logits, mask, rng, temperature, top_p, greedy):
    lg = logits.float().masked_fill(~mask, NEG)
    if greedy or temperature <= 0:
        return int(lg.argmax())
    p = F.softmax(lg / temperature, -1)
    sp, idx = p.sort(descending=True)
    keep = sp.cumsum(0) - sp < top_p
    sp = sp * keep
    sp = sp / sp.sum()
    return int(idx[int(torch.multinomial(sp, 1, generator=rng))])


@torch.no_grad()
def generate_core(model: P1Model, ex: FE.CoreExample, song, observed_holds, runs, observed_heads, *,
                  temperature=0.8, top_p=0.95, greedy=False, seed=0, device='cpu'):
    """Returns generated groups in CoreExample.groups format (t_rel_us, notes[...])."""
    c = model.c
    rng = torch.Generator(device='cpu').manual_seed(seed)
    work = copy.copy(ex)
    work.groups = []
    holds = list(observed_holds)  # (onset_abs, end_abs, lane) incl. generated context holds
    c0 = ex.c0_us
    t_prev = -1
    for _ in range(FE.MAX_GROUPS):
        # right-lookahead rows for the current teacher-forced prefix
        times = [c0 - 1] + [c0 + g['t_rel_us'] for g in work.groups]
        work.right = np.array([_right(t, observed_heads, runs) for t in times], np.float32)
        bt = dequant({k: (v.to(device) if torch.is_tensor(v) else v) for k, v in collate([work], [song], c.taps, c.top_tap).items()})
        h, pa, fi, _ = model.encode(bt)
        model._sel_core = None
        if model.c.selection:
            c0c = FE.HALO_US // FE.CELL_US
            model._sel_core = model.selection_field(pa[:, c0c:c0c + FE.CORE_US // FE.CELL_US], model._intent_h)
        G = len(work.groups)
        hrow = h[:, G:G + 1]
        logits, mask, ok = model.time_logits(bt, hrow, pa, fi, torch.tensor([[t_prev / 1e6 if t_prev >= 0 else -0.001]], device=device))
        k = _pick(logits[0, 0].cpu(), mask[0, 0].cpu(), rng, temperature, top_p, greedy)
        if k == 0:
            break
        if k <= FE.CORE_BINS:
            b = k - 1
            rl = model.residual_logits(hrow[:, 0], fi, torch.tensor([b], device=device))
            r = _pick(rl[0].cpu(), ok[0, 0, b].cpu(), rng, temperature, top_p, greedy)
            t_rel, copy_idx = b * FE.BIN_US + r * FE.MS, -1
        else:
            copy_idx = k - 1 - FE.CORE_BINS
            t_rel = int(ex.copy_times[copy_idx])
        t_abs = c0 + t_rel
        busy = np.zeros(4, bool)
        for a, e, lane in holds:
            if lane >= 0 and a < t_abs < e:
                busy[lane] = True
        right_new = np.array(_right(t_abs - 1, observed_heads, runs), np.float32)
        notes = _inner(model, bt, hrow[:, 0], pa, fi, t_rel, right_new, busy, song, ex, t_abs, holds, runs, rng,
                       temperature, top_p, greedy, device)
        g = {'t_rel_us': t_rel, 'bin': t_rel // FE.BIN_US, 'residual': t_rel % FE.BIN_US // FE.MS, 'copy': copy_idx,
             'notes': notes, 'busy_lanes': busy, 'right_new': right_new}
        work.groups.append(g)
        for n in notes:
            if 'end_rel_us' in n:
                holds.append((t_abs, t_abs + n['end_rel_us'], n['lane']))
        t_prev = t_rel
    return work.groups


def _right(t, observed_heads, runs):
    nxt = next(((u, lane, line) for u, lane, line in observed_heads if u > t), None)
    run_end = next((b for a, b in runs if a <= t < b), next((b for a, b in runs if a > t), t))
    dt = (nxt[0] - t) / FE.SEC if nxt else 60.0
    return (min(dt, 60.0), nxt[1] if nxt else -1, nxt[2] if nxt else -1, min((run_end - t) / FE.SEC, 60.0))


def _inner(model, bt, h, pa, fi, t_rel, right_new, busy, song, ex, t_abs, holds, runs, rng, temperature, top_p, greedy, device):
    c = model.c
    mode = int(bt['mode'][0])
    hs = model.gru_init(h, pa, fi, torch.tensor([t_rel / FE.SEC], device=device), torch.from_numpy(right_new)[None].to(device),
                        torch.from_numpy(busy)[None].to(device))
    allowed = bt['allowed_types'][0].cpu() & torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]], dtype=torch.bool)[mode]
    used = torch.from_numpy(busy.copy())
    inp = model.start[None]
    notes = []
    for j in range(MAX_NOTES):
        hs = model.gru_cell(inp, hs)
        if j >= 1:
            cont_mask = torch.tensor([True, bool((~used).any()) if mode == 0 else True])
            if _pick(model.cont(hs)[0].cpu(), cont_mask, rng, temperature, top_p, greedy) == 0:
                break
        if mode == 0 and bool(used.all()):
            break
        typ = _pick(model.type_h(hs)[0].cpu(), allowed, rng, temperature, top_p, greedy)
        o1 = hs + model.type_c(torch.tensor([typ], device=device))
        n = {'type': typ, 'lane': -1, 'line': -1, 'side': -1, 'x': 0.0}
        if mode == 0:
            n['lane'] = _pick(model.lane_h(torch.cat([o1, used[None].float().to(device)], -1))[0].cpu(), ~used, rng, temperature, top_p, greedy)
            loc = torch.cat([F.one_hot(torch.tensor([n['lane']]), 4).float(), torch.zeros(1, 60)], -1).to(device)
            used[n['lane']] = True
        elif not c.phira_location:
            loc = torch.zeros(1, 64, device=device)  # placement left to the charter (design §1.3 revision)
        else:
            lines = int(bt['lines'][0])
            lmask = torch.arange(FE.MAX_LINES) < max(lines, 1)
            n['line'] = _pick((model.line_q(o1) @ model.line_tab.weight.T)[0].cpu(), lmask, rng, temperature, top_p, greedy)
            le = model.line_tab(torch.tensor([n['line']], device=device))
            n['side'] = _pick(model.side_h(torch.cat([o1, le], -1))[0].cpu(), torch.ones(2, dtype=torch.bool), rng, temperature, top_p, greedy)
            n['x'] = _sample_x(model.x_h(torch.cat([o1, le], -1))[0].cpu(), rng, greedy)
            loc = le
        o2 = o1 + model.loc_c(torch.cat([loc, model.side_e(torch.tensor([n['side'] + 1], device=device)),
                                         torch.tensor([[n['x']]], device=device)], -1))
        dur = 0.0
        if typ == 1:
            end = _hold_end(model, o2, song, t_abs, n, holds, runs, rng, temperature, top_p, greedy, device)
            if end is None:  # no legal end: fall back to tap if allowed, else stop the group
                if not bool(allowed[0]):
                    break
                n['type'] = typ = 0
            else:
                n['end_rel_us'] = end - t_abs
                n['end_ok_us'] = (1, end - t_abs)
                dur = (end - t_abs) / FE.SEC
        notes.append(n)
        inp = model._note_emb(*(torch.tensor([v], device=device) for v in (typ, n['lane'], n['line'], n['side'])),
                              torch.tensor([n['x']], device=device), torch.tensor([dur], device=device))
    return notes


def _sample_x(params, rng, greedy):
    M = params.shape[-1] // 3
    logit_pi, mu, log_s = params[:M], params[M:2 * M].tanh(), params[2 * M:].clamp(-7, 3)
    k = int(logit_pi.argmax()) if greedy else int(torch.multinomial(F.softmax(logit_pi, -1), 1, generator=rng))
    if greedy:
        return float(mu[k])
    u = torch.rand(1, generator=rng).clamp(1e-5, 1 - 1e-5)
    return float((mu[k] + log_s[k].exp() * (u.log() - (1 - u).log())).clamp(-1, 1))


def _hold_end(model, o, song, onset, n, holds, runs, rng, temperature, top_p, greedy, device):
    run_end = next((b for a, b in runs if a <= onset < b), onset)
    hi = run_end
    if n['lane'] >= 0:
        for a, e, lane in holds:
            if lane == n['lane'] and a > onset:
                hi = min(hi, a)
    lo = onset + 1
    if hi < onset + FE.MS:
        return None
    b0 = onset // FE.BLOCK_US
    bstart = (b0 + torch.arange(FE.MAX_END_BLOCKS)) * FE.BLOCK_US
    bmask = (bstart <= hi) & (bstart + FE.BLOCK_US > lo)
    blocks = torch.zeros(FE.MAX_END_BLOCKS, 8, song.summary.shape[-1])
    a, e = b0, min(b0 + FE.MAX_END_BLOCKS, song.summary.shape[0])
    if e > a:
        blocks[:e - a] = torch.from_numpy(np.asarray(song.summary[a:e], np.float32))
    feats = model.blk_band(model.sum_ln(blocks.to(device))).flatten(-2)
    bl = (model.blk_q(o)[0] @ feats.T) / 16 + model.blk_off.weight[:, 0]
    blk = _pick(bl.cpu(), bmask, rng, temperature, top_p, greedy)
    cells = (b0 + blk) * (FE.BLOCK_US // FE.CELL_US)
    det = np.zeros((FE.BLOCK_US // FE.CELL_US, 8, model.c.tap_dim), np.float32)
    top = song.taps[model.c.top_tap]
    a, e = cells, min(cells + det.shape[0], top.shape[0])
    if e > a:
        det[:e - a] = np.asarray(top[a:e], np.float32)
    det = model.det_proj(model.tap_ln[model.c.taps.index(model.c.top_tap)](torch.from_numpy(det).to(device)) + model.band)
    q = model.det_q(torch.cat([o, feats[blk][None]], -1))[0]
    cell = torch.logsumexp(torch.einsum('p,cnp->cn', q, det) / math.sqrt(q.shape[-1]), -1)
    cl = cell.repeat_interleave(4) + model.det_sub.weight[:, 0].repeat(det.shape[0])
    start = (b0 + blk) * FE.BLOCK_US
    cb = start + torch.arange(FE.BLOCK_US // FE.BIN_US) * FE.BIN_US
    cmask = (cb <= hi) & (cb + FE.BIN_US - FE.MS >= lo)
    if not cmask.any():
        return None
    co = _pick(cl.cpu(), cmask, rng, temperature, top_p, greedy)
    fb = start + co * FE.BIN_US + torch.arange(10) * FE.MS
    fmask = (fb <= hi) & (fb >= lo)
    ctx = det.mean(1)[co // 4]
    fi = _pick(model.end_fine(torch.cat([o, ctx[None]], -1))[0].cpu(), fmask, rng, temperature, top_p, greedy)
    return int(start + co * FE.BIN_US + fi * FE.MS)


def match_f1(pred_us, true_us, tol_us):
    """One-to-one greedy matching of sorted head times within tolerance (mir_eval-style)."""
    pred, true = sorted(pred_us), sorted(true_us)
    i = j = tp = 0
    while i < len(pred) and j < len(true):
        d = pred[i] - true[j]
        if abs(d) <= tol_us:
            tp, i, j = tp + 1, i + 1, j + 1
        elif d < 0:
            i += 1
        else:
            j += 1
    p = tp / len(pred) if pred else (1.0 if not true else 0.0)
    r = tp / len(true) if true else (1.0 if not pred else 0.0)
    return tp, len(pred), len(true), (2 * p * r / (p + r) if p + r else 0.0)


def core_context(sample, ex: FE.CoreExample, line_index):
    """Known holds (observed + earlier generated context), editable runs and observed heads for generation."""
    from ..events.canonical import Event
    observed = [Event.from_json(e) for e in sample.request['observed_events']]
    l0 = ex.c0_us - FE.HALO_US
    generated = [e for e in sample.hidden if e.onset_us < ex.c0_us and max(e.occupancy()[1], e.onset_us + 1) > l0]
    holds = [(e.onset_us, e.end_us, e.location.lane if e.location.kind == 'lane' else -1)
             for e in observed + generated if e.is_hold]
    runs = FE.editable_runs(sample.request['regions'])
    heads = []
    for e in sorted(observed, key=lambda e: (e.onset_us, e.event_id)):
        lane, line, _, _ = FE._loc_fields(e, line_index)
        heads.append((e.onset_us, lane, line))
    return holds, runs, heads
