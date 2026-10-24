"""Constrained generation for one core with a P1 model (v2.1), and core-level evaluation.

Incremental decoding: the prefix rows, the local audio and the song tokens are encoded once per core
(K/V cache); every later group appends one Y row. Every choice is sampled from a legality-masked
distribution: new heads only at editable times and strictly after the previous group (40 ms cell, then
the in-cell ms offset), no duplicate or hold-occupied lane, hold ends inside the editable run and before
the next same-lane head. Greedy decoding is available for reproducibility.
"""
from __future__ import annotations
import copy
import math
import time
import numpy as np
import torch
from torch.nn import functional as F
from . import features as FE
from .batch import collate, collate_meta, dequant, SongInfo, MAX_NOTES, CORE_CELLS, MS_PER_CELL
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
                  temperature=0.8, top_p=0.95, greedy=False, seed=0, device='cpu', diagnostics=None, deadline=None,
                  incremental=True):
    """Returns generated groups in CoreExample.groups format (t_rel_us, notes[...])."""
    c = model.c
    rng = torch.Generator(device='cpu').manual_seed(seed)
    work = copy.copy(ex)
    work.groups = []
    holds = list(observed_holds)  # (onset_abs, end_abs, lane) incl. generated context holds
    c0 = ex.c0_us
    t_prev = -1
    ended = False
    timed_out = False
    cache = {} if incremental else None  # per-core K/V cache: later groups append one Y row instead of re-encoding
    h0c = FE.HALO_US // FE.CELL_US
    for _ in range(FE.MAX_GROUPS):
        if deadline is not None and time.perf_counter() >= deadline:
            timed_out = True
            break
        # right-lookahead rows for the current teacher-forced prefix
        times = [c0 - 1] + [c0 + g['t_rel_us'] for g in work.groups]
        work.right = np.array([_right(t, observed_heads, runs) for t in times], np.float32)
        if cache:  # audio and song tokens are cached: metadata only
            raw = collate_meta([work], [SongInfo.of(song, 0, c.taps[0])], c.taps, c.top_tap, c.tap_dim, song.summary.shape[-1], False)
            raw.pop('_audio')
        else:
            raw = collate([work], [song], c.taps, c.top_tap)
        bt = dequant({k: (v.to(device) if torch.is_tensor(v) else v) for k, v in raw.items()})
        G = len(work.groups)
        if cache is None or not cache:
            h, pa, _ = model.encode(bt, cache)
            hrow = h[:, G:G + 1]
        else:
            hrow, pa, _ = model.encode_step(bt, cache)
        model._sel_core = None
        if model.c.selection:
            model._sel_core = model.selection_field(pa[:, h0c:h0c + CORE_CELLS], model._intent_h)
        logits, mask, ok = model.time_logits(bt, hrow, pa, torch.tensor([[t_prev / 1e6 if t_prev >= 0 else -0.001]], device=device))
        k = _pick(logits[0, 0].cpu(), mask[0, 0].cpu(), rng, temperature, top_p, greedy)
        if k == 0:
            ended = True
            break
        if k <= CORE_CELLS:
            cell = k - 1
            tok = model._cell_tokens(pa, torch.tensor([[h0c + cell]], device=device))[:, 0]  # [1,3,8,P]
            ol = model.offset_logits(model.off_q(hrow[:, 0]), tok, bt['prev_onsets'][:, G],
                                     torch.tensor([cell * FE.CELL_US / FE.SEC], device=device))
            r = _pick(ol[0].cpu(), ok[0, 0, cell].cpu(), rng, temperature, top_p, greedy)
            t_rel, copy_idx = cell * FE.CELL_US + r * FE.MS, -1
        else:
            copy_idx = k - 1 - CORE_CELLS
            t_rel = int(ex.copy_times[copy_idx])
        t_abs = c0 + t_rel
        busy = np.zeros(4, bool)
        for a, e, lane in holds:
            if lane >= 0 and a < t_abs < e:
                busy[lane] = True
        for onset, lane, _ in observed_heads:
            if onset == t_abs and lane >= 0:
                busy[lane] = True
        right_new = np.array(_right(t_abs - 1, observed_heads, runs), np.float32)
        onsets_abs = sorted({c0 + g['t_rel_us'] for g in work.groups} | {u for u, _, _ in observed_heads} | {a for a, _, _ in holds})
        notes = _inner(model, bt, hrow[:, 0], pa, t_rel, right_new, busy, song, ex, t_abs, holds, runs, rng,
                       temperature, top_p, greedy, device, observed_heads, onsets_abs)
        if not notes:
            # No legal note at this time. Advance the cursor, without emitting an empty group.
            t_prev = t_rel
            continue
        g = {'t_rel_us': t_rel, 'bin': t_rel // FE.BIN_US, 'residual': t_rel % FE.BIN_US // FE.MS, 'copy': copy_idx,
             'notes': notes, 'busy_lanes': busy, 'right_new': right_new}
        work.groups.append(g)
        for n in notes:
            if 'end_rel_us' in n:
                holds.append((t_abs, t_abs + n['end_rel_us'], n['lane']))
        t_prev = t_rel
    if diagnostics is not None:
        diagnostics.update(terminated_by_end=ended, group_limit_reached=not ended and not timed_out,
                           deadline_reached=timed_out)
    return work.groups


def _right(t, observed_heads, runs):
    nxt = next(((u, lane, line) for u, lane, line in observed_heads if u > t), None)
    run_end = next((b for a, b in runs if a <= t < b), next((b for a, b in runs if a > t), t))
    dt = (nxt[0] - t) / FE.SEC if nxt else 60.0
    return (min(dt, 60.0), nxt[1] if nxt else -1, nxt[2] if nxt else -1, min((run_end - t) / FE.SEC, 60.0))


def _inner(model, bt, h, pa, t_rel, right_new, busy, song, ex, t_abs, holds, runs, rng, temperature, top_p, greedy, device, observed_heads=(), onsets_abs=()):
    c = model.c
    mode = int(bt['mode'][0])
    hs = model.gru_init(h, pa, torch.tensor([t_rel / FE.SEC], device=device), torch.from_numpy(right_new)[None].to(device),
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
            prev = [t_abs] + [u for u in onsets_abs if u < t_abs][-3:][::-1]  # the hold's onset, then earlier onsets
            end = _hold_end(model, o2, song, t_abs, n, holds, runs, rng, temperature, top_p, greedy, device, observed_heads, prev)
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


def _hold_end(model, o, song, onset, n, holds, runs, rng, temperature, top_p, greedy, device, observed_heads=(), prev_abs_us=None):
    run_end = next((b for a, b in runs if a <= onset < b), onset)
    hi = run_end
    if n['lane'] >= 0:
        for a, e, lane in holds:
            if lane == n['lane'] and a > onset:
                hi = min(hi, a)
        for a, lane, _ in observed_heads:
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
    start = (b0 + blk) * FE.BLOCK_US
    prev = None
    if prev_abs_us is not None:
        from .batch import PREV_ONSETS, NO_ONSET
        vals = [u / FE.SEC for u in prev_abs_us][:PREV_ONSETS]
        prev = torch.tensor([vals + [NO_ONSET] * (PREV_ONSETS - len(vals))], device=device)
    cell, _, dtok = model.hold_cells(o, blocks.to(device)[None], torch.from_numpy(det).to(device)[None], torch.tensor([blk], device=device),
                                     prev, torch.tensor([start / FE.SEC], device=device))
    cb = start + torch.arange(FE.BLOCK_US // FE.CELL_US) * FE.CELL_US
    cmask = (cb <= hi) & (cb + FE.CELL_US - FE.MS >= lo)
    if not cmask.any():
        return None
    co = _pick(cell[0].cpu(), cmask, rng, temperature, top_p, greedy)
    tok = model._cell_tokens(dtok, torch.tensor([[co]], device=device))[:, 0]  # [1,3,8,P]
    ol = model.offset_logits(model.off_hold_q(o), tok, prev, torch.tensor([(start + co * FE.CELL_US) / FE.SEC], device=device))
    fb = start + co * FE.CELL_US + torch.arange(MS_PER_CELL) * FE.MS
    fmask = (fb <= hi) & (fb >= lo)
    off = _pick(ol[0].cpu(), fmask, rng, temperature, top_p, greedy)
    return int(start + co * FE.CELL_US + off * FE.MS)


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
