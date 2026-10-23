"""Throughput-oriented cache encoder: frozen teacher -> int8 taps + S20 summaries, songs streamed in groups.

Design (GPU occupancy first, measured in beatfcn/cloud/encode_app.py):
  * one fixed batch shape; the last partial batch of a group is padded with silence windows that already
    live on the device (a pageable host->device copy may synchronise the stream);
  * the forward *and* the int8 quantisation run as one unit (TapRunner). In 'graph' modes that unit is a
    captured CUDA graph, so the host issues ~6 calls per batch instead of hundreds of kernel launches
    (compiled Inductor code launches each kernel from Python and loses the GIL race to reader/writer
    threads: 68 windows/s on an H100 that computes 207 windows/s in isolation);
  * windows are built and converted to bf16 on a reader thread (pinned), results return by async
    device->host copies that writer threads wait on; the main thread never commits or syncs per group.
"""
from __future__ import annotations
import queue
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from . import memory as M

CENTRE_CELLS = M.HOP // M.FRAMES_PER_CELL  # 128 kept cells per window


def tap_forward(enc, want):
    """Teacher forward returning the centre cells of the wanted taps (same math as TapEncoder)."""
    from .pupu import make_rope
    teacher = enc.teacher
    grid = (M.WINDOW // M.FRAMES_PER_CELL, M.BANDS)
    rope = make_rope(grid, teacher.head_dim, device=teacher.patch_embed.proj.weight.device)
    keep = slice(M.EDGE // M.FRAMES_PER_CELL, (M.EDGE + M.HOP) // M.FRAMES_PER_CELL)
    top = max(want)

    def fwd(w):
        x = teacher.patch_embed(w)
        out = []
        for l, block in enumerate(teacher.teacher.blocks, 1):
            x = block(x, rope)
            if l in want:
                out.append(x.reshape(x.shape[0], *grid, -1)[:, keep])
            if l >= top:
                break
        return out
    return fwd


def quantiser(layers, mean=None, scale=None, comp=None):
    """Per tap: centre (mean), optional projection (comp), int8 with per-dim scale; None -> float16."""
    def q(l, t):
        x = t.float()
        if mean and l in mean:
            x = x - mean[l]
        if comp and l in comp:
            x = x @ comp[l]
        if scale and l in scale:
            return (x / scale[l]).round().clamp(-127, 127).to(torch.int8)
        return x.half()
    return q


class TapRunner:
    """Fixed-batch forward + quantisation. Returns per wanted layer a device tensor [batch, 128, 8, D]:
    int8 (or fp16) for cache layers, the raw tap (bf16) for summary-only layers.

    mode: 'eager' | 'compile' | 'graph' (eager captured as a CUDA graph) | 'compile-graph'.
    In graph modes the returned tensors are the graph's static outputs: consume them (copy) before the
    next call; stream order makes that safe without host syncs."""

    def __init__(self, fwd, want, layers, quant, batch, window_shape, device, mode='eager', dtype=torch.bfloat16):
        self.want, self.layers, self.batch, self.mode = list(want), set(layers), batch, mode
        f = torch.compile(fwd) if 'compile' in mode else fwd

        def unit(w):
            return [quant(l, t) if l in self.layers else t for l, t in zip(self.want, f(w))]
        self.unit = unit
        self.static_in = torch.zeros((batch, *window_shape), dtype=dtype, device=device)
        self.graph = None
        if 'graph' in mode:
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):  # warm-up (and compile) outside capture
                    unit(self.static_in)
            torch.cuda.current_stream().wait_stream(s)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.static_out = unit(self.static_in)

    def __call__(self, w, pad_rows):
        """w: [n<=batch, ...] host (pinned) or device windows in self.static_in's dtype."""
        n = w.shape[0]
        self.static_in[:n].copy_(w, non_blocking=True)
        if n < self.batch:
            self.static_in[n:].copy_(pad_rows[:self.batch - n])
        if self.graph is not None:
            self.graph.replay()
            return self.static_out
        return self.unit(self.static_in)


def summary_half(tap):
    """== memory.summary_means on any device, float16 result."""
    n = -(-tap.shape[0] // M.SUMMARY_CELLS)
    x = tap.new_zeros((n * M.SUMMARY_CELLS, *tap.shape[1:]), dtype=torch.float32)
    x[:tap.shape[0]] = tap.float()
    cnt = torch.full((n,), float(M.SUMMARY_CELLS), device=tap.device)
    cnt[-1] = tap.shape[0] - (n - 1) * M.SUMMARY_CELLS
    return (x.reshape(n, M.SUMMARY_CELLS, *tap.shape[1:]).sum(1) / cnt[:, None, None]).half()


class _NoEvent:
    def record(self):
        pass

    def synchronize(self):
        pass


def run(work, load_mel, save, runner, layers, *, group=4, writers=8, readers=2, device='cuda', summary_layer=20,
        on_first_group=None, on_steady_end=None, guard=None, out_slots=None, in_slots=3, max_cells=15000):
    """Encode songs `work` (ids). load_mel(id) -> float [frames, 128]; save(id, [(name, ndarray)]) writes
    L{l} for each cache layer and S{summary_layer} last. Returns throughput / host-time statistics.
    guard(steady_seconds, steady_windows) -> reason or None, called after each steady group: a reason
    stops encoding early (songs already enqueued are still written completely).

    Host buffers are preallocated rings (pinned on CUDA) so the steady state never calls cudaHostAlloc:
    `in_slots` window buffers for groups of up to group x ceil(max_cells / 128) + 2 windows, `out_slots`
    result buffers for songs up to `max_cells` cells. Larger groups/songs fall back to one-off buffers."""
    cuda = str(device).startswith('cuda')
    dt_in = runner.static_in.dtype
    pad_host = torch.from_numpy(M.silence_row()).to(dt_in)
    pad_rows = pad_host.to(device).view(1, 1, 1, -1).expand(runner.batch, 1, M.WINDOW, pad_host.shape[0])
    groups = [work[i:i + group] for i in range(0, len(work), group)]
    out_slots = out_slots or 3 * group
    max_win = group * (-(-max_cells // CENTRE_CELLS) + 2)
    stats = {'songs': 0, 'windows': 0, 'bytes': 0, 'write_s': 0.0, 'fallback_buffers': 0}
    host_t = {'wait_reader': 0.0, 'launch': 0.0, 'post': 0.0, 'wait_writer': 0.0}
    want = runner.want
    in_free, out_free = queue.Queue(), queue.Queue()
    for _ in range(in_slots):
        in_free.put((torch.empty((max_win, 1, M.WINDOW, pad_host.shape[0]), dtype=dt_in, pin_memory=cuda), _NoEvent()))

    def prep(g):  # reader thread: windows written straight into a ring slot (== window_batch(...).to(dt_in))
        with torch.inference_mode():  # thread-local: slots may be inference tensors when run() is called in that mode
            return _prep(g)

    def _prep(g):
        mels = [torch.from_numpy(np.asarray(load_mel(sha))) for sha in g]
        plans = [M.plan(m.shape[0]) for m in mels]
        total = sum(len(pl.starts) for pl in plans)
        if total <= max_win:
            slot, ev = in_free.get()
            ev.synchronize()  # the previous group's host->device copies from this slot have completed
            buf = slot
        else:
            slot, buf = None, torch.empty((total, 1, M.WINDOW, pad_host.shape[0]), dtype=dt_in, pin_memory=cuda)
        k = 0
        for m, pl in zip(mels, plans):
            mb, T = m.to(dt_in), m.shape[0]
            for st in pl.starts:
                dst = buf[k, 0]
                a, b = max(st, 0), min(st + M.WINDOW, T)
                if a > st or b < st + M.WINDOW:
                    dst.copy_(pad_host.expand(M.WINDOW, -1))
                if b > a:
                    dst[a - st:b - st].copy_(mb[a:b])
                k += 1
        return g, plans, buf[:total], slot

    def write_after(ev, sha, host, slot):
        ev.synchronize()
        t = time.time()
        arrays = [(name, h.numpy()) for name, h in host]
        save(sha, arrays)
        nbytes = sum(a.nbytes for _, a in arrays)
        if slot is not None:
            out_free.put(slot)
        return nbytes, time.time() - t

    reader, writer, pending = ThreadPoolExecutor(readers), ThreadPoolExecutor(writers), []

    def collect(block):
        while pending and (block or pending[0].done()):
            b, ws = pending.pop(0).result()
            stats['bytes'] += b
            stats['write_s'] += ws
    futs = [reader.submit(prep, groups[i]) for i in range(min(readers, len(groups)))]
    t_steady = w_steady = None
    with torch.inference_mode():
        for gi in range(len(groups)):
            tw = time.time()
            g, plans, win, in_slot = futs.pop(0).result()
            host_t['wait_reader'] += time.time() - tw
            if gi + readers < len(groups):
                futs.append(reader.submit(prep, groups[gi + readers]))
            tl = time.time()
            total = win.shape[0]
            acc = {}
            for i in range(0, total, runner.batch):
                w = win[i:i + runner.batch]
                n = w.shape[0]
                outs = runner(w, pad_rows)
                for l, t in zip(want, outs):
                    if l not in acc:
                        acc[l] = t.new_empty((total * CENTRE_CELLS, *t.shape[2:]))
                    acc[l][i * CENTRE_CELLS:(i + n) * CENTRE_CELLS].copy_(t[:n].reshape(-1, *t.shape[2:]))
            if in_slot is not None:  # slot is reusable once the copies just enqueued have run
                ev = torch.cuda.Event() if cuda else _NoEvent()
                ev.record()
                in_free.put((in_slot, ev))
            host_t['launch'] += time.time() - tl
            if gi == 0:  # result ring, shaped by the first group's outputs (one-off, before steady stats)
                shapes = {f'L{l}': (acc[l].shape[1:], acc[l].dtype) for l in layers}
                shapes[f'S{summary_layer}'] = (acc[summary_layer].shape[1:], torch.float16)
                rows = {k: (-(-max_cells // M.SUMMARY_CELLS) if k.startswith('S') else max_cells) for k in shapes}
                for _ in range(out_slots):
                    out_free.put({k: torch.empty((rows[k], *sh), dtype=dt, pin_memory=cuda) for k, (sh, dt) in shapes.items()})
            tp = time.time()
            off = 0
            for sha, pl in zip(g, plans):
                arrays = [(f'L{l}', acc[l][off:off + pl.cells]) for l in layers]
                arrays.append((f'S{summary_layer}', summary_half(acc[summary_layer][off:off + pl.cells])))
                off += len(pl.starts) * CENTRE_CELLS
                if pl.cells <= max_cells:
                    tw = time.time()
                    slot = out_free.get()  # back-pressure: waits only when every result slot is still being written
                    host_t['wait_writer'] += time.time() - tw
                    bufs = slot
                else:
                    slot, bufs = None, {k: torch.empty(t.shape, dtype=t.dtype, pin_memory=cuda) for k, t in arrays}
                    stats['fallback_buffers'] += 1
                host = []
                for name, t in arrays:
                    h = bufs[name][:t.shape[0]]
                    h.copy_(t, non_blocking=cuda)
                    host.append((name, h))
                ev = torch.cuda.Event() if cuda else _NoEvent()
                ev.record()
                pending.append(writer.submit(write_after, ev, sha, host, slot))
                stats['songs'] += 1
                stats['windows'] += len(pl.starts)
            host_t['post'] += time.time() - tp
            if gi == 0:  # first group carries one-off costs (compile/capture warm-up, ring allocation)
                if cuda:
                    torch.cuda.synchronize()
                if on_first_group:
                    on_first_group()
                t_steady, w_steady = time.time(), stats['windows']
                for k in host_t:
                    host_t[k] = 0.0
            collect(False)  # surfaces writer errors early
            if guard and gi > 0:
                reason = guard(time.time() - t_steady, stats['windows'] - w_steady)
                if reason:
                    stats['aborted'] = reason
                    break
    if t_steady is not None and len(groups) > 1:  # steady phase ends with the last enqueue, before the flush
        if cuda:
            torch.cuda.synchronize()
        wall = time.time() - t_steady
        stats['steady_windows_per_s'] = round((stats['windows'] - w_steady) / wall, 1)
        stats['steady_host_s'] = {k: round(v, 1) for k, v in host_t.items()} | {'wall': round(wall, 1)}
    if on_steady_end:
        on_steady_end()
    tf = time.time()
    collect(True)
    stats['flush_s'] = round(time.time() - tf, 1)
    stats['write_MBps_per_thread'] = round(stats['bytes'] / 1e6 / max(stats['write_s'], 1e-9), 1)
    for f in futs:  # prefetched groups not consumed after an early stop: release their slots
        f.cancel()
    reader.shutdown(wait=True, cancel_futures=True)
    writer.shutdown()
    return stats
