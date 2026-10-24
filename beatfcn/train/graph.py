"""One optimiser step as a replayed CUDA graph (forward + backward + clipping + AdamW + zero_grad).

TieredStep: dynamic batches (any shapes) are padded on the device into the static buffers of the smallest
shape tier (batch.TIERS) that holds them; one CUDA graph per tier is captured lazily (3 eager warm-up
steps first), batches beyond the largest tier run eagerly with the same step function. Padding adds only
rows that every loss masks, so results equal the plain loop. The counterfactual loss (every few steps,
variable shapes) runs eagerly just before the replay and its grads accumulate into the same static .grad
tensors (zero_grad(set_to_none=False) runs at the end of the step). `capture=False` keeps everything eager
(tests, CPU smoke).
"""
from __future__ import annotations
import torch
from ..model.batch import TIERS, AXIS, PAD_VALUE, batch_dims, tier_of, tier_shape


class _Tier:
    __slots__ = ('index', 'spec', 'slots', 'graph', 'out', 'seen')

    def __init__(self, index, spec):
        self.index, self.spec, self.slots, self.graph, self.out, self.seen = index, spec, None, None, None, 0


class TieredStep:
    def __init__(self, runs, clip: float, device, capture=True, warmup=3, before_capture=None, tiers=TIERS, max_graphs=2):
        """runs: objects with .model and .opt (AdamW(capturable=True, fused=True) with a tensor lr on CUDA).
        before_capture(): called right before each capture (e.g. SongPool.wait_idle). max_graphs: every captured
        graph keeps its own activation pool (measured 20+ GB per tier at 32 cores), so only the first
        `max_graphs` tiers are captured; larger tiers run eagerly."""
        self.runs, self.clip, self.dev = runs, clip, torch.device(device)
        self.max_graphs = max_graphs
        self.cuda = self.dev.type == 'cuda'
        self.capture = capture and self.cuda
        self.warmup, self.before_capture = warmup, before_capture
        self.tiers = [_Tier(i, t) for i, t in enumerate(tiers)]
        self.stats = {'eager_steps': 0, 'graph_steps': 0, 'captures': 0, 'tier_steps': [0] * len(tiers)}
        # Every backward that is not captured (warm-ups, eager fallback, the counterfactual loss) runs on this one
        # side stream: autograd's AccumulateGrad nodes remember the stream of the last backward, and a later
        # capture fails if that was the default stream (third baseline attempt failed exactly so).
        self.side = torch.cuda.Stream() if self.cuda else None

    # ------------------------------------------------------------ static inputs
    def _compact(self):
        return any(getattr(getattr(r.model, 'c', None), 'compact_inner', False) for r in self.runs)

    def _prepare(self, batches):
        """Drop the compaction indices unless a run uses them (they are dynamic metadata)."""
        if self._compact():
            return batches
        return [{k: v for k, v in b.items() if k not in ('inner_index', 'hold_inner_index')} for b in batches]

    def _alloc(self, tier, batches):
        tier.slots = []
        for bt in batches:
            B = bt['mode'].shape[0]
            buf = {}
            for k, v in bt.items():
                buf[k] = torch.empty(tier_shape(k, v.shape, tier.spec, B), dtype=v.dtype, device=self.dev) if torch.is_tensor(v) else v
            buf['w_select'] = torch.zeros((), device=self.dev)
            tier.slots.append(buf)

    def _fill(self, tier, batches, w_select):
        if tier.slots is None:
            self._alloc(tier, batches)
        if len(batches) != len(tier.slots):
            raise ValueError('accumulation count cannot change after buffers are allocated')
        for bt, buf in zip(batches, tier.slots):
            for k, v in bt.items():
                if not torch.is_tensor(v):
                    buf[k] = v
                    continue
                s = buf[k]
                if s.dtype != v.dtype or (k not in AXIS and s.shape != v.shape):
                    raise ValueError(f'static shape/dtype mismatch for {k}: {tuple(v.shape)} {v.dtype} vs {tuple(s.shape)} {s.dtype}')
                if s.shape == v.shape:
                    s.copy_(v, non_blocking=True)
                else:  # padded axis: reset to collate's padding value, copy the real rows
                    pad = bt['mode'].shape[0] * tier.spec['G'] - 1 if k == 'inner_index' else PAD_VALUE.get(k, 0)
                    s.fill_(pad)  # inner_index: the last group slot, invalid by tier choice
                    dst = s[tuple(slice(0, n) for n in v.shape)]
                    dst.copy_(v, non_blocking=True)  # v is pinned (DataLoader): asynchronous
                    if k == 'inner_index':  # flat (b * G + g) indices: re-encode for the tier's G (review finding 1),
                        # on the device (a host-side re-encode would be a pageable copy = a stream sync per batch)
                        Gd = bt['grp_valid'].shape[1]
                        dst.copy_((dst // Gd) * tier.spec['G'] + dst % Gd)
            buf['w_select'].fill_(w_select)

    # ------------------------------------------------------------ the step
    def _eager(self, slots):
        """_body outside a capture: on the side stream (see __init__), joined back to the current stream."""
        if not self.cuda:
            return self._body(slots)
        self.side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.side):
            out = self._body(slots)
        torch.cuda.current_stream().wait_stream(self.side)
        return out

    def _body(self, slots):
        mets = []
        for r in self.runs:
            acc = {}
            for bt in slots:
                b = dict(bt)  # the model pops/replaces entries (dequant): never on shared buffers
                with torch.autocast(self.dev.type, dtype=torch.bfloat16, enabled=self.cuda, cache_enabled=False):
                    loss, met = r.model(b)
                (loss / len(slots)).backward()
                for k, value in met.items():
                    if k == '_illegal':
                        acc[k] = acc.get(k, 0) + value
                    else:
                        l, n, c = value
                        a = acc.setdefault(k, [0., 0, 0])
                        a[0], a[1], a[2] = a[0] + l * n, a[1] + n, a[2] + c
            mets.append({k: (value if k == '_illegal' else (value[0] / value[1].clamp(min=1), value[1], value[2]))
                         for k, value in acc.items()})  # (mean loss, count, correct), no host sync
        norms = []
        for r in self.runs:
            norms.append(torch.nn.utils.clip_grad_norm_(r.model.parameters(), self.clip).clone())  # own storage: read later
            r.opt.step()
            r.opt.zero_grad(set_to_none=False)  # grads stay allocated: static tensors the graphs write
        return mets, norms

    def step(self, batches, w_select: float):
        """One optimiser step over `batches` (device-resident audio, host metadata; shapes arbitrary).
        Returns ([metrics per run], [grad norms]) as device tensors."""
        batches = self._prepare(batches)
        all_dims = [batch_dims(b) for b in batches]
        dims = {k: max(d[k] for d in all_dims) for k in all_dims[0]}
        ti = tier_of(dims, [t.spec for t in self.tiers])
        if ti is None or not self.capture:  # beyond every tier (rare), or eager mode: same math, no padding
            self.stats['eager_steps'] += 1
            slots = []
            for bt in batches:
                b = {k: (v.to(self.dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in bt.items()}
                b['w_select'] = torch.tensor(float(w_select), device=self.dev)
                slots.append(b)
            return self._eager(slots)
        tier = self.tiers[ti]
        self.stats['tier_steps'][ti] += 1
        if ti >= self.max_graphs:  # tier too large to keep a captured pool for: eager with the batch's own (unpadded)
            # shapes: padding to the tier would double the transient activations (an 8-core tier-2 batch padded to
            # K512/H1024 needed ~15-20 GB on top of the pools and ran the H100 out of memory)
            self.stats['eager_steps'] += 1
            slots = []
            for bt in batches:
                b = {k: (v.to(self.dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in bt.items()}
                b['w_select'] = torch.tensor(float(w_select), device=self.dev)
                slots.append(b)
            return self._eager(slots)
        self._fill(tier, batches, w_select)
        if tier.graph is None:
            if tier.seen < self.warmup:  # eager warm-up on the side stream (allocator / cuBLAS / RNG state)
                out = self._eager(tier.slots)
                tier.seen += 1
                self.stats['eager_steps'] += 1
                return out
            if self.before_capture:
                self.before_capture()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            try:
                # thread_local: the DataLoader pin-memory thread and the SongPool loaders keep making CUDA calls
                # (cudaHostAlloc, copies) in other threads; in the default 'global' mode any of them invalidates
                # the capture (second baseline attempt failed exactly so)
                with torch.cuda.graph(graph, capture_error_mode='thread_local'):
                    out = self._body(tier.slots)
            except Exception as exc:  # a failed capture degrades this run to eager instead of killing it
                torch.cuda.synchronize()
                for r in self.runs:
                    r.opt.zero_grad(set_to_none=False)
                self.max_graphs = ti  # never try to capture this tier (or larger) again
                self.stats['capture_failed'] = repr(exc)[:300]
                self.stats['eager_steps'] += 1
                return self._eager(tier.slots)
            tier.graph, tier.out = graph, out
            self.stats['captures'] += 1
            # the capture itself did not run: replay once for this batch
        tier.graph.replay()
        self.stats['graph_steps'] += 1
        return tier.out


class GraphedStep(TieredStep):
    """Single-shape form (pilots, tests): fill(bt) / fill_many(batches) then step(). The one tier takes the
    shapes of the first batches; later batches must match exactly (no padding)."""

    def __init__(self, runs, clip, device, capture=True, warmup=3, before_capture=None):
        super().__init__(runs, clip, device, capture, warmup, before_capture, tiers=[{k: 10 ** 9 for k in TIERS[0]}])
        self._pending, self._count = None, None

    def fill(self, bt, w_select):
        self.fill_many([bt], w_select)

    def fill_many(self, batches, w_select):
        if self._count is not None and len(batches) != self._count:
            raise ValueError('accumulation count cannot change after buffers are allocated')
        self._count = len(batches)
        self._pending = (batches, w_select)

    def step(self, *a, **k):
        if a or k:
            return super().step(*a, **k)
        batches, w = self._pending
        t = self.tiers[0]
        if t.slots is None:
            dims = [batch_dims(b) for b in self._prepare(batches)]
            t.spec = {name: max(d[name] for d in dims) for name in TIERS[0]}
            if 'N' in dims[0]:  # compaction: the tier must be strictly larger than the batch (sentinel slot)
                t.spec['G'] += 1
        return super().step(batches, w)
