"""One optimiser step as a replayed CUDA graph (forward + backward + clipping + AdamW + zero_grad).

Requirements met elsewhere: fixed batch shapes (batch.STATIC), no host syncs in the model, capturable fused
AdamW with tensor LR. The step reads static input buffers that the loop fills before each replay; the
counterfactual loss (every few steps, variable shapes) runs eagerly just before the replay and its grads
accumulate into the same static .grad tensors (zero_grad(set_to_none=False) runs at the end of the step).
GPU only for capture; `capture=False` runs the same step function eagerly (tests, CPU smoke).
"""
from __future__ import annotations
import torch


class GraphedStep:
    def __init__(self, runs, clip: float, device, capture=True, warmup=3, before_capture=None):
        """runs: objects with .model and .opt (AdamW(capturable=True, fused=True), tensor lr).
        before_capture(): called once right before the capture (e.g. SongPool.wait_idle)."""
        self.runs, self.clip, self.dev = runs, clip, torch.device(device)
        self.before_capture = before_capture
        self.capture = capture and self.dev.type == 'cuda'
        self.warmup, self.static, self.graph, self.out = warmup, None, None, None
        self.seen = 0

    # ------------------------------------------------------------ static inputs
    def fill(self, bt, w_select: float):
        """Copy a (device-resident audio, host metadata) batch into the static buffers."""
        if self.static is None:
            self.static = {k: (torch.empty_like(v, device=self.dev) if torch.is_tensor(v) else v) for k, v in bt.items()}
            self.static['w_select'] = torch.zeros((), device=self.dev)
        for k, v in bt.items():
            if torch.is_tensor(v):
                s = self.static[k]
                if s.shape != v.shape or s.dtype != v.dtype:
                    raise ValueError(f'static shape/dtype mismatch for {k}: {tuple(v.shape)} {v.dtype} vs {tuple(s.shape)} {s.dtype}')
                s.copy_(v, non_blocking=True)
            else:
                self.static[k] = v
        self.static['w_select'].fill_(w_select)

    # ------------------------------------------------------------ the step
    def _body(self):
        cuda = self.dev.type == 'cuda'
        mets = []
        for r in self.runs:
            b = dict(self.static)  # the model pops/replaces entries (dequant): never on the shared dict
            with torch.autocast(self.dev.type, dtype=torch.bfloat16, enabled=cuda, cache_enabled=False):
                loss, met = r.model(b)
            loss.backward()
            mets.append(met)
        norms = []
        for r in self.runs:
            norms.append(torch.nn.utils.clip_grad_norm_(r.model.parameters(), self.clip))
            r.opt.step()
            r.opt.zero_grad(set_to_none=False)  # grads stay allocated: static tensors the graph writes
        return mets, norms

    def step(self):
        """Run one step on the current static inputs; returns ([metrics per run], [grad norms])."""
        if not self.capture:
            return self._body()
        if self.graph is None:
            # parameters that take no gradient (e.g. the disabled Phira placement heads) keep grad=None, exactly
            # as in the plain loop: AdamW skips them; the warm-up step allocates every grad that is used
            if self.seen < self.warmup:  # eager warm-up on a side stream (allocator / cuBLAS / RNG state)
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    out = self._body()
                torch.cuda.current_stream().wait_stream(s)
                self.seen += 1
                return out
            if self.before_capture:
                self.before_capture()
            torch.cuda.synchronize()
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.out = self._body()
            # the capture itself did not run: replay once for this batch
        self.graph.replay()
        return self.out
