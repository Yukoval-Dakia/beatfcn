"""P1 event model (candidate B): FIRST_CANDIDATE §2.2–2.7 without the selection field.

Rows (one sequence per core): R reference evidence | I 8 intent | U controls+mode |
G 2 s song summary | K known events | Y BOS + completed groups. Each of 8 layers runs
ONE softmax over [frozen multi-tap full-band audio K/V (GQA) ; row K/V]. Visibility:
R->R; I->R,I; U,G,K->R,I,U,G,K; Y->prefix + causal Y. Audio is read only by K rows
(t±2 s) and Y rows ([t-1 s, t+3 s)); time enters every timed row and audio key as
1D RoPE (seconds from the core start).

Heads: one pointer softmax over {END, 800 free 10 ms bins, COPY_TIME anchors}, each
bin scored from its own audio cell (query-weighted log-sum-exp over 8 bands) and
10 ms mel features; a 1 ms residual; a GRU per group initialised with audio read
back at the chosen time; type -> location (mania lane / Phira line, side, x as a
discretised logistic mixture) -> hold end (2 s block -> 10 ms -> 1 ms, block detail
revealed after the block step). Losses are teacher-forced, each field normalised
by its own count. Generation lives in sample.py.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F
from . import features as FE
from .batch import MAX_NOTES, END_BLOCK_CELLS

NEG = -1e9


@dataclass(frozen=True)
class P1Config:
    d: int = 512
    layers: int = 8
    heads: int = 8
    audio_kv_heads: int = 2
    ffn: int = 1536
    dropout: float = 0.1
    taps: tuple = (8, 16, 24)
    top_tap: int = 24
    tap_dim: int = 1024
    summary_dim: int = 1024
    ptr: int = 128
    gru: int = 256
    intent: int = 8
    mixture: int = 5
    phira_location: bool = False  # Phira line/side/x are inputs only, never predicted (user decision 2026-10-23)
    k_window: tuple = (-2.0, 2.0)
    y_window: tuple = (-1.0, 3.0)
    # selection mechanism ("A" in the design package): field from target audio x isolated intent
    selection: bool = False
    # pre-registered diagnostics (FIRST_CANDIDATE §3); defaults are the first candidate
    readout: str = 'multitap'  # 'multitap' | 'design_package' (top tap only, 4 learned slots per cell)
    audio_window: bool = True
    time_head: str = 'pointer'  # 'pointer' | 'linear'
    fine_branch: bool = True
    gather_after_time: bool = True
    packed: bool = False  # FlexAttention packed path (same math; measured slower than padded SDPA, and its nonzero() cannot be graph-captured)


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.w, self.eps = nn.Parameter(torch.ones(d)), eps

    def forward(self, x):
        return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).to(x.dtype) * self.w


def rope(x, t, timed):
    """x [B,H,N,D]; t [B,N] seconds; untimed rows keep identity. Periods 20 ms .. 120 s."""
    d = x.shape[-1] // 2
    period = torch.logspace(math.log10(0.02), math.log10(120.0), d, device=x.device)
    ang = (t * timed)[:, None, :, None].float() * (2 * math.pi / period)
    cos, sin = ang.cos().to(x.dtype), ang.sin().to(x.dtype)
    a, b = x[..., :d], x[..., d:]
    return torch.cat([a * cos - b * sin, a * sin + b * cos], -1)


class Layer(nn.Module):
    def __init__(self, c: P1Config):
        super().__init__()
        self.c = c
        hd = c.d // c.heads
        self.n1, self.n2 = RMSNorm(c.d), RMSNorm(c.d)
        self.q, self.k, self.v, self.o = (nn.Linear(c.d, c.d, bias=False) for _ in range(4))
        self.ka = nn.Linear(c.tap_dim, c.audio_kv_heads * hd, bias=False)
        self.va = nn.Linear(c.tap_dim, c.audio_kv_heads * hd, bias=False)
        self.mix = nn.Parameter(torch.zeros(len(c.taps)))
        self.f1, self.f2 = nn.Linear(c.d, 2 * c.ffn, bias=False), nn.Linear(c.ffn, c.d, bias=False)
        self.drop = nn.Dropout(c.dropout)

    def forward(self, x, taps_ln, audio_t, row_t, row_timed, mask):
        c, (B, N, _) = self.c, x.shape
        hd = c.d // c.heads
        z = self.n1(x)
        q, k, v = (f(z).view(B, N, c.heads, hd).transpose(1, 2) for f in (self.q, self.k, self.v))
        q, k = rope(q, row_t, row_timed), rope(k, row_t, row_timed)
        audio = torch.einsum('t,btnd->bnd', self.mix.softmax(0).to(taps_ln.dtype), taps_ln)
        Na = audio.shape[1]
        ka, va = (f(audio).view(B, Na, c.audio_kv_heads, hd).transpose(1, 2) for f in (self.ka, self.va))
        ka = rope(ka, audio_t, torch.ones_like(audio_t))
        rep = c.heads // c.audio_kv_heads
        ka, va = ka.repeat_interleave(rep, 1), va.repeat_interleave(rep, 1)
        att = F.scaled_dot_product_attention(q, torch.cat([ka, k], 2), torch.cat([va, v], 2), attn_mask=mask)
        x = x + self.drop(self.o(att.transpose(1, 2).reshape(B, N, c.d)))
        a, b = self.f1(self.n2(x)).chunk(2, -1)
        return x + self.drop(self.f2(F.silu(a) * b))

    def forward_packed(self, x, taps_ln, audio_t, row_t, row_timed, block_mask):
        """x [1, N, d]: the valid rows of every core, core after core; audio memory stays per core [B, T, Na, Dt]
        and is laid out core after core in front of the rows. Same math as forward() for every valid row."""
        c, (_, N, _) = self.c, x.shape
        hd = c.d // c.heads
        z = self.n1(x)
        q, k, v = (f(z).view(1, N, c.heads, hd).transpose(1, 2) for f in (self.q, self.k, self.v))
        q, k = rope(q, row_t, row_timed), rope(k, row_t, row_timed)
        audio = torch.einsum('t,btnd->bnd', self.mix.softmax(0).to(taps_ln.dtype), taps_ln)
        B, Na = audio.shape[:2]
        ka, va = (f(audio).view(B, Na, c.audio_kv_heads, hd).transpose(1, 2) for f in (self.ka, self.va))
        ka = rope(ka, audio_t, torch.ones_like(audio_t))
        rep = c.heads // c.audio_kv_heads
        ka, va = (t.repeat_interleave(rep, 1).transpose(0, 1).reshape(1, c.heads, B * Na, hd) for t in (ka, va))
        att = flex_attend(q, torch.cat([ka, k.to(ka.dtype)], 2), torch.cat([va, v.to(va.dtype)], 2).to(q.dtype), block_mask)
        x = x + self.drop(self.o(att.transpose(1, 2).reshape(1, N, c.d)))
        a, b = self.f1(self.n2(x)).chunk(2, -1)
        return x + self.drop(self.f2(F.silu(a) * b))


def onehot(x, n):
    """F.one_hot without its host-side range check (a device sync); inputs are clamped / validated upstream."""
    return x[..., None] == torch.arange(n, device=x.device)


def flex_attend(q, k, v, block_mask):
    """FlexAttention, compiled on CUDA (fused; skips the blocks the BlockMask leaves out). Off CUDA (tests,
    local equivalence checks) the same element-exact mask_mod is evaluated densely and fed to SDPA, which
    is the reference semantics of the block mask; FlexAttention has no CPU backward in recent torch."""
    from torch.nn.attention.flex_attention import flex_attention, create_mask
    global _FLEX
    if q.is_cuda:
        if _FLEX is None:
            # static shapes: one kernel per packed-length bucket (PACK_BUCKET keeps that to a handful).
            # dynamic=True failed to lower in torch 2.14.1 (unbacked symint from _local_scalar_dense).
            torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)
            _FLEX = torch.compile(flex_attention, dynamic=False)
        return _FLEX(q, k, v, block_mask=block_mask)
    Q, KV = block_mask.seq_lengths
    dense = create_mask(block_mask.mask_mod, None, None, Q, KV, device=q.device)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=dense)


_FLEX = None


def masked_ce(logits, target, valid, mask=None, name=''):
    """Mean CE over valid rows; logits [..., C]; invalid classes masked.

    Returns device tensors (loss, n, correct, illegal) and never synchronises with the host: rows are
    weighted, not gathered. `illegal` counts valid rows whose target lies outside the legality mask;
    callers accumulate it and the training loop raises at log time (summarize) if it is ever > 0."""
    if mask is not None:
        logits = logits.masked_fill(~mask, NEG)
    C = logits.shape[-1]
    lg = logits.reshape(-1, C).float()
    v = valid.reshape(-1)
    tg = target.reshape(-1).clamp(0, C - 1)  # rows outside `valid` may carry -1 padding
    ce = F.cross_entropy(lg, tg, reduction='none')
    n = v.sum()
    loss = (ce * v).sum() / n.clamp(min=1)
    correct = ((lg.argmax(-1) == tg) & v).sum()
    if mask is not None:
        illegal = (v & ~mask.reshape(-1, C).gather(-1, tg[:, None])[:, 0]).sum()
    else:
        illegal = torch.zeros((), dtype=torch.long, device=lg.device)
    return loss, n, correct, illegal


def logistic_mixture_nll(params, x, bins=256):
    """Discretised logistic mixture on [-1,1] (PixelCNN++ edges). params [N, 3M], x [N]."""
    M = params.shape[-1] // 3
    logit_pi, mu, log_s = params[:, :M], params[:, M:2 * M].tanh(), params[:, 2 * M:].clamp(-7, 3)
    xc = x[:, None]
    inv = torch.exp(-log_s)
    half = 1.0 / (bins - 1)
    cdf_hi = torch.sigmoid(inv * (xc + half - mu))
    cdf_lo = torch.sigmoid(inv * (xc - half - mu))
    p = torch.where(xc < -0.999, cdf_hi, torch.where(xc > 0.999, 1 - cdf_lo, cdf_hi - cdf_lo))
    lp = torch.log(p.clamp_min(1e-12)) + F.log_softmax(logit_pi, -1)
    return -torch.logsumexp(lp, -1)


class P1Model(nn.Module):
    def __init__(self, c: P1Config = P1Config()):
        super().__init__()
        self.c = c
        d, T = c.d, len(c.taps)
        self.tap_ln = nn.ModuleList(nn.LayerNorm(c.tap_dim) for _ in range(T))
        self.band = nn.Parameter(torch.randn(8, c.tap_dim) * 0.02)
        self.layers = nn.ModuleList(Layer(c) for _ in range(c.layers))
        self.norm = RMSNorm(d)
        # row embeddings
        self.type_e = nn.Embedding(7, 64)  # 4 types, 4=unselected, 5=pad, 6=selected with masked attributes (CF view)
        self.lane_e = nn.Embedding(5, 64)
        self.line_e = nn.Embedding(FE.MAX_LINES + 1, 64)
        self.side_e = nn.Embedding(3, 16)
        self.ev_proj = nn.Linear(64 * 3 + 16 + 4, d)  # + x, log dur, flag_a, flag_b
        self.ref_mix = nn.Parameter(torch.zeros(T))
        self.ref_band_q = nn.Parameter(torch.randn(c.tap_dim) * 0.02)
        self.ref_audio = nn.Linear(c.tap_dim, d)
        self.row_kind = nn.Embedding(6, d)
        self.intent = nn.Parameter(torch.randn(c.intent, d) * 0.02)
        self.ctrl_e = nn.Embedding(len(FE.CONTROL_FIELDS) * 32, d)
        self.mode_e = nn.Embedding(len(FE.MODES), d)
        self.allowed_p = nn.Linear(4, d)
        self.sum_ln = nn.LayerNorm(c.summary_dim)
        self.sum_band = nn.Linear(c.summary_dim, d // 8)
        self.sum_stats = nn.Linear(6, d)
        self.bos = nn.Parameter(torch.randn(d) * 0.02)
        self.right_p = nn.Linear(8, d)
        # group / note embeddings (attribute-only feedback, design §4.2)
        self.note_p = nn.Linear(64 * 3 + 16 + 2, c.gru)
        self.group_p = nn.Linear(2 * c.gru, d)
        # pointer
        P = c.ptr
        self.ptr_mix = nn.Parameter(torch.zeros(T))
        self.ptr_proj = nn.Linear(c.tap_dim, P)
        self.fine = nn.Sequential(nn.Conv1d(128, P, 5, padding=2), nn.GELU(), nn.Conv1d(P, P, 3, padding=1))
        self.q_a, self.q_f, self.q_c = nn.Linear(d, P), nn.Linear(d, P), nn.Linear(d, P)
        self.end_head = nn.Linear(d, 1)
        self.rel_bias = nn.Embedding(64, 1)
        self.state_bias = nn.Embedding(3, 1)
        self.copy_key = nn.Sequential(nn.Linear(2 * P, P), nn.GELU(), nn.Linear(P, P))
        self.res_head = nn.Sequential(nn.Linear(P + 3 * P, 256), nn.GELU(), nn.Linear(256, 10))
        self.res_q = nn.Linear(d, P)
        # inner GRU
        H = c.gru
        self.gather_q = nn.Linear(d, P)
        self.right_g = nn.Linear(8, 16)
        self.init = nn.Linear(d + 2 * P + 16 + 4, H)
        self.start = nn.Parameter(torch.randn(H) * 0.02)
        self.gru_cell = nn.GRUCell(H, H)
        self.register_buffer('mode_types', torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]], dtype=torch.bool), persistent=False)
        self.cont = nn.Linear(H, 2)
        self.type_h = nn.Linear(H, 4)
        self.type_c = nn.Embedding(4, H)
        self.lane_h = nn.Linear(H + 4, 4)
        self.line_q = nn.Linear(H, 64)
        self.line_tab = nn.Embedding(FE.MAX_LINES, 64)
        self.side_h = nn.Linear(H + 64, 2)
        self.x_h = nn.Linear(H + 64, 3 * c.mixture)
        self.loc_c = nn.Linear(64 + 16 + 1, H)
        self.blk_band = nn.Linear(c.summary_dim, 32)
        self.blk_q = nn.Linear(H, 256)
        self.blk_off = nn.Embedding(FE.MAX_END_BLOCKS, 1)
        self.det_proj = nn.Linear(c.tap_dim, P)
        self.det_q = nn.Linear(H + 256, P)
        self.det_sub = nn.Embedding(4, 1)
        self.end_fine = nn.Linear(H + P, 10)
        if c.selection:
            self.sel_audio = nn.Linear(P, 64)
            self.sel_z = nn.Linear(d, 64)
            self.sel_v = nn.Linear(d, 1)
            self.sel_lam = nn.Parameter(torch.zeros(()))  # lambda = sigmoid(.) in (0,1), init 0.5
        if c.readout == 'design_package':
            self.slot_q = nn.Parameter(torch.randn(4, c.tap_dim) * 0.02)
        if c.time_head == 'linear':
            self.time_linear = nn.Linear(d, FE.CORE_BINS)

    # ------------------------------------------------------------ embeddings
    def _events(self, ev, selected_flag=False):
        typ = ev[..., 1].long()
        typ = torch.where(typ < 0, torch.full_like(typ, 4), typ)
        lane = ev[..., 2].long().clamp(min=-1) + 1
        lane = torch.where(lane == 0, torch.full_like(lane, 4), lane - 1)
        line = (ev[..., 3].long() + 1).clamp(0, FE.MAX_LINES)
        side = (ev[..., 4].long() + 1).clamp(0, 2)
        scal = torch.stack([ev[..., 5], torch.log1p(ev[..., 6].clamp(min=0)), ev[..., 7], ev[..., 8]], -1)
        return self.ev_proj(torch.cat([self.type_e(typ), self.lane_e(lane), self.line_e(line), self.side_e(side), scal], -1))

    def _note_emb(self, typ, lane, line, side, x, dur):
        lane_i = torch.where(lane < 0, torch.full_like(lane, 4), lane)
        line_i = (line + 1).clamp(0, FE.MAX_LINES)
        side_i = (side + 1).clamp(0, 2)
        return self.note_p(torch.cat([self.type_e(typ.clamp(0, 3)), self.lane_e(lane_i), self.line_e(line_i),
                                      self.side_e(side_i), x[..., None], torch.log1p(dur.clamp(min=0))[..., None]], -1))

    @staticmethod
    def _right_feats(r):
        lane = r[..., 1].long()
        oh = onehot(torch.where(lane < 0, torch.full_like(lane, 4), lane), 5).float()
        return torch.cat([torch.log1p(r[..., 0:1].clamp(min=0)), torch.log1p(r[..., 3:4].clamp(min=0)), oh,
                          (r[..., 2:3].clamp(min=-1) + 1) / FE.MAX_LINES], -1)

    def _rows(self, bt):
        c = self.c
        B, dev = bt['mode'].shape[0], bt['mode'].device
        # R: evidence + its own audio cell (query-pooled over bands, mixed over taps)
        ra = torch.stack([ln(bt['ref_audio'][:, :, j]) for j, ln in enumerate(self.tap_ln)], 2)
        ra = torch.einsum('t,brtnd->brnd', self.ref_mix.softmax(0).to(ra.dtype), ra) + self.band
        w = torch.einsum('brnd,d->brn', ra, self.ref_band_q).softmax(-1)
        R = self._events(bt['refs']) + self.ref_audio(torch.einsum('brn,brnd->brd', w, ra)) + self.row_kind.weight[0]
        I = (self.intent[None] + self.row_kind.weight[1]).expand(B, -1, -1)
        nf = len(FE.CONTROL_FIELDS)
        idx = torch.arange(nf, device=dev) * 32 + (bt['controls'] + 1).clamp(0, 31)
        U = torch.cat([self.ctrl_e(idx), (self.mode_e(bt['mode']) + self.allowed_p(bt['allowed_types'].float()))[:, None]], 1)
        U = U + self.row_kind.weight[2]
        s = self.sum_band(self.sum_ln(bt['summary'])).flatten(-2)
        G = s + self.sum_stats(bt['summary_stats']) + self.row_kind.weight[3]
        K = self._events(bt['known']) + self.row_kind.weight[4]
        rows = [R, I, U, G, K]
        lens = [x.shape[1] for x in rows]
        times = [bt['refs'][..., 0], torch.zeros(B, lens[1], device=dev), torch.zeros(B, lens[2], device=dev),
                 bt['summary_t'], bt['known'][..., 0]]
        timed = [bt['refs_valid'].float(), torch.zeros(B, lens[1], device=dev), torch.zeros(B, lens[2], device=dev),
                 bt['summary_valid'].float(), bt['known_valid'].float()]
        valid = [bt['refs_valid'], torch.ones(B, lens[1], dtype=torch.bool, device=dev),
                 torch.ones(B, lens[2], dtype=torch.bool, device=dev), bt['summary_valid'], bt['known_valid']]
        return rows, lens, times, timed, valid

    def _y_rows(self, bt, group_emb):
        B = bt['mode'].shape[0]
        Y = torch.cat([self.bos.expand(B, 1, -1), group_emb], 1) + self.row_kind.weight[5]
        Y = Y + self.right_p(self._right_feats(bt['right_rows']))
        t = torch.cat([torch.zeros(B, 1, device=Y.device), bt['grp_t']], 1)
        valid = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=Y.device), bt['grp_valid']], 1)
        return Y, t, valid

    def _target_loc(self, bt):
        """Locations of generated groups as the model will see them: Phira placement is unknown
        unless phira_location (inference cannot feed back what it never predicts)."""
        line, side, x = bt['note_line'], bt['note_side'], bt['note_x']
        if not self.c.phira_location:
            ph = (bt['mode'] == 1)[:, None, None]
            line, side, x = line.masked_fill(ph, -1), side.masked_fill(ph, -1), x.masked_fill(ph, 0.0)
        return line, side, x

    def group_embedding(self, bt):
        line, side, x = self._target_loc(bt)
        e = self._note_emb(bt['note_type'], bt['note_lane'], line, side, x, bt['note_dur'])
        nm = (torch.arange(MAX_NOTES, device=e.device)[None, None] < bt['grp_n'][..., None])[..., None]
        s = (e * nm).sum(2)
        m = e.masked_fill(~nm, NEG).max(2).values.masked_fill(~nm.any(2), 0)
        return self.group_p(torch.cat([s, m], -1))

    # ------------------------------------------------------------ masks
    def _mask(self, bt, lens, times, valid, n_y, per_cell=8):
        c = self.c
        dev = bt['mode'].device
        B = bt['mode'].shape[0]
        kinds = torch.cat([torch.full((n,), k, device=dev) for k, n in enumerate(lens + [n_y])])
        q, k = kinds[:, None], kinds[None, :]
        allow = torch.zeros(len(kinds), len(kinds), dtype=torch.bool, device=dev)
        allow |= (q == 0) & (k == 0)
        allow |= (q == 1) & (k <= 1)
        allow |= (q >= 2) & (q <= 4) & (k <= 4)
        allow |= (q == 5) & (k <= 4)
        y0 = sum(lens)
        allow[y0:, y0:] = torch.ones(n_y, n_y, dtype=torch.bool, device=dev).tril()
        rv = torch.cat(valid, 1)
        rows = allow[None] & rv[:, None, :]
        eye = torch.eye(len(kinds), dtype=torch.bool, device=dev)[None]
        rows = torch.where(rv[:, :, None], rows, eye)
        # audio windows
        t_row = torch.cat(times, 1)
        cell_t = (torch.arange(FE.LOCAL_CELLS, device=dev) + 0.5) * FE.CELL_US / FE.SEC - FE.HALO_US / FE.SEC
        dt = cell_t[None, None, :] - t_row[:, :, None]
        kq = (kinds == 4)[None, :, None]
        yq = (kinds == 5)[None, :, None]
        if c.audio_window:
            win = (kq & (dt >= c.k_window[0]) & (dt <= c.k_window[1])) | (yq & (dt >= c.y_window[0]) & (dt < c.y_window[1]))
        else:  # diagnostic: K and Y rows read all local audio
            win = (kq | yq).expand(-1, -1, FE.LOCAL_CELLS)
        win = win & bt['audio_valid'][:, None, :] & rv[:, :, None]
        win = win[..., None].expand(-1, -1, -1, per_cell).reshape(B, len(kinds), FE.LOCAL_CELLS * per_cell)
        return torch.cat([win, rows], -1)[:, None]

    PACK_BUCKET = 2048  # packed length is padded to a multiple of this: few distinct shapes for the compiled kernels

    def _encode_packed(self, bt, xpad, taps, audio_t, row_t, row_timed, lens, valid, per_cell):
        """Run the layers on valid rows only (padding was 62% of rows) and return normed hidden states
        scattered back to the padded layout (padding rows = 0; every loss masks them).

        The packed sequence is padded to a PACK_BUCKET multiple with dummy rows (core = B, kind R) that
        attend only to each other, so the kernels see a handful of shapes instead of one per batch."""
        B, Np, d = xpad.shape
        dev = xpad.device
        rv = torch.cat(valid, 1)  # [B, Np]
        kinds = torch.cat([torch.full((n,), k, device=dev) for k, n in enumerate(lens + [Np - sum(lens)])])
        idx = rv.reshape(-1).nonzero()[:, 0]  # one host sync: the packed length (core-major, so core is sorted)
        N = idx.shape[0]
        Npad = -(-N // self.PACK_BUCKET) * self.PACK_BUCKET
        pad = Npad - N
        core = torch.cat([idx // Np, torch.full((pad,), B, device=dev)])
        pos = idx % Np
        kind = torch.cat([kinds[pos], torch.zeros(pad, dtype=kinds.dtype, device=dev)])
        ypos = torch.cat([pos - sum(lens), torch.zeros(pad, dtype=pos.dtype, device=dev)])
        t_q = torch.cat([row_t.reshape(-1)[idx], torch.zeros(pad, dtype=row_t.dtype, device=dev)])
        x = torch.cat([xpad.reshape(B * Np, d)[idx], xpad.new_zeros(pad, d)])[None]
        rt, rtd = t_q[None], torch.cat([row_timed.reshape(-1)[idx], torch.zeros(pad, dtype=row_timed.dtype, device=dev)])[None]
        bm = self._block_mask(bt, core, kind, ypos, t_q, per_cell)
        for layer in self.layers:
            x = layer.forward_packed(x, taps, audio_t, rt, rtd, bm)
        hn = self.norm(x)[0, :N]
        return hn.new_zeros(B * Np, d).index_copy(0, idx, hn).view(B, Np, d)

    def _block_mask(self, bt, core, kind, ypos, t_q, per_cell, block=128):
        """FlexAttention BlockMask equal to _mask() restricted to valid rows, plus 'same core'. KV layout:
        [audio of core 0 .. core B-1 | packed rows]. The element-exact rule is mask_mod (run by the kernel
        inside partial blocks); the block list is built at block granularity from per-block core ranges
        and time ranges (a superset of the non-empty blocks), never by evaluating the mask densely."""
        from torch.nn.attention.flex_attention import BlockMask
        c = self.c
        dev = core.device
        B = bt['mode'].shape[0]
        Na = FE.LOCAL_CELLS * per_cell
        BNa, N = B * Na, core.shape[0]
        assert BNa % block == 0 and N % block == 0, (BNa, N)
        cell_t = (torch.arange(FE.LOCAL_CELLS, device=dev) + 0.5) * FE.CELL_US / FE.SEC - FE.HALO_US / FE.SEC
        aval = bt['audio_valid']
        (k0, k1), (w0, w1), windowed = c.k_window, c.y_window, c.audio_window

        def mask_mod(b, h, qi, ki):
            kq, cq = kind[qi], core[qi]
            ai = torch.clamp(ki, max=BNa - 1)
            ca, cell = ai // Na, (ai % Na) // per_cell
            dt = cell_t[cell] - t_q[qi]
            if windowed:
                win = ((kq == 4) & (dt >= k0) & (dt <= k1)) | ((kq == 5) & (dt >= w0) & (dt < w1))
            else:  # diagnostic: K and Y rows read all local audio
                win = (kq == 4) | (kq == 5)
            audio_ok = (ca == cq) & win & aval[ca, cell]
            ri = torch.clamp(ki - BNa, min=0)
            kr = kind[ri]
            allow = (((kq == 0) & (kr == 0)) | ((kq == 1) & (kr <= 1)) | ((kq >= 2) & (kq <= 4) & (kr <= 4))
                     | ((kq == 5) & ((kr <= 4) | ((kr == 5) & (ypos[ri] <= ypos[qi])))))
            return torch.where(ki < BNa, audio_ok, (core[ri] == cq) & allow)
        # --- block-level superset ---
        nq = N // block
        cb = core.view(nq, block)
        cmin, cmax = cb.min(1).values, cb.max(1).values  # cores spanned by each q / row block
        inf = float('inf')
        if windowed:
            wlo = torch.where(kind == 4, k0, torch.where(kind == 5, w0, inf))
            whi = torch.where(kind == 4, k1, torch.where(kind == 5, w1, -inf))
        else:
            wlo = torch.where((kind == 4) | (kind == 5), -inf, inf)
            whi = torch.where((kind == 4) | (kind == 5), inf, -inf)
        lo = (t_q + wlo).view(nq, block).min(1).values  # earliest audio time any row of the block may read
        hi = (t_q + whi).view(nq, block).max(1).values
        # audio kv blocks: key range -> cores and cell range (a block straddling two cores takes all cells)
        j = torch.arange(BNa // block, device=dev)
        k_lo, k_hi = j * block, j * block + block - 1
        a_clo, a_chi = k_lo // Na, k_hi // Na
        same = a_clo == a_chi
        c_lo = torch.where(same, (k_lo % Na) // per_cell, torch.zeros_like(k_lo))
        c_hi = torch.where(same, (k_hi % Na) // per_cell, torch.full_like(k_hi, FE.LOCAL_CELLS - 1))
        tmin, tmax = cell_t[c_lo], cell_t[c_hi]
        audio_ok = ((a_clo[None] <= cmax[:, None]) & (a_chi[None] >= cmin[:, None])
                    & (tmin[None] <= hi[:, None]) & (tmax[None] >= lo[:, None]))
        rows_ok = (cmin[None] <= cmax[:, None]) & (cmax[None] >= cmin[:, None])  # row kv blocks = q blocks
        allowed = torch.cat([audio_ok, rows_ok], 1)  # [nq, n_kv_blocks]
        num = allowed.sum(1)
        order = torch.sort((~allowed).to(torch.int8), dim=1, stable=True).indices  # allowed blocks first, ascending
        return BlockMask.from_kv_blocks(num[None, None].to(torch.int32), order[None, None].to(torch.int32).contiguous(),
                                        BLOCK_SIZE=block, mask_mod=mask_mod, seq_lengths=(N, BNa + N))

    # ------------------------------------------------------------ forward
    def encode(self, bt):
        """Transformer over all rows; returns Y hidden states [B, G+1, d] and pointer features."""
        from .batch import dequant
        dequant(bt)  # in place, no-op when to_device already expanded the compact transport format
        c = self.c
        B = bt['mode'].shape[0]
        taps = torch.stack([ln(bt['audio'][:, j]) for j, ln in enumerate(self.tap_ln)], 1)  # [B,T,400,8,Dt]
        taps = taps + self.band
        per_cell = 8
        if c.readout == 'design_package':
            # diagnostic: only the top tap, 8 bands pooled into 4 learned slots per 40 ms cell
            top = taps[:, c.taps.index(c.top_tap)]  # [B,400,8,Dt]
            w = torch.einsum('bcnd,kd->bckn', top, self.slot_q).softmax(-1)
            pooled = torch.einsum('bckn,bcnd->bckd', w, top)  # [B,400,4,Dt]
            taps = pooled[:, None].expand(-1, len(c.taps), -1, -1, -1)
            per_cell = 4
        ptr_src = taps if c.readout != 'design_package' else (torch.stack([ln(bt['audio'][:, j]) for j, ln in enumerate(self.tap_ln)], 1) + self.band)
        taps = taps.flatten(2, 3)  # [B,T,cells*per_cell,Dt]
        if torch.is_autocast_enabled(taps.device.type):  # every layer's einsum would cast it anyway: cast once
            taps = taps.to(torch.get_autocast_dtype(taps.device.type))
        audio_t = ((torch.arange(FE.LOCAL_CELLS, device=taps.device) + 0.5) * FE.CELL_US / FE.SEC - FE.HALO_US / FE.SEC)
        audio_t = audio_t.repeat_interleave(per_cell)[None].expand(B, -1)
        rows, lens, times, timed, valid = self._rows(bt)
        Y, ty, vy = self._y_rows(bt, self.group_embedding(bt))
        x = torch.cat(rows + [Y], 1)
        row_t = torch.cat(times + [ty], 1)
        row_timed = torch.cat(timed + [vy.float()], 1)
        if c.packed:
            hn = self._encode_packed(bt, x, taps, audio_t, row_t, row_timed, lens, valid + [vy], per_cell)
        else:
            mask = self._mask(bt, lens, times + [ty], valid + [vy], Y.shape[1], per_cell)
            for layer in self.layers:
                x = layer(x, taps, audio_t, row_t, row_timed, mask)
            hn = self.norm(x)
        h = hn[:, sum(lens):]
        if c.readout == 'design_package':
            ptr_audio = self.ptr_proj(ptr_src[:, c.taps.index(c.top_tap)])  # top tap only, 8 bands
        else:
            ptr_audio = self.ptr_proj(torch.einsum('t,btcnd->bcnd', self.ptr_mix.softmax(0).to(ptr_src.dtype), ptr_src))
        fine = self.fine(bt['mel'].transpose(1, 2)).transpose(1, 2)  # [B,1600,P]
        if not c.fine_branch:
            fine = torch.zeros_like(fine)
        self._intent_h = hn[:, lens[0]:lens[0] + lens[1]]  # I rows (read R/I only)
        return h, ptr_audio, fine, ty

    def time_logits(self, bt, h, ptr_audio, fine, t_prev):
        """Pointer over [END | 800 bins | A anchors] for each Y row; returns logits and legality mask."""
        B, Ny, _ = h.shape
        dev = h.device
        qa, qf, qc = self.q_a(h), self.q_f(h), self.q_c(h)
        P = qa.shape[-1]
        core_cells = ptr_audio[:, FE.HALO_US // FE.CELL_US:(FE.HALO_US + FE.CORE_US) // FE.CELL_US]  # [B,200,8,P]
        band = torch.einsum('byp,bcnp->bycn', qa, core_cells) / math.sqrt(P)
        cell_score = torch.logsumexp(band, -1).repeat_interleave(4, -1)  # [B,Ny,800]
        f0 = FE.HALO_US // FE.BIN_US
        core_fine = fine[:, f0:f0 + FE.CORE_BINS]
        fine_score = torch.einsum('byp,bkp->byk', qf, core_fine) / math.sqrt(P)
        bins = torch.arange(FE.CORE_BINS, device=dev)
        prev_ms = (t_prev * 1000).round()  # [B,Ny] ms from core start (BOS: -1)
        delta = (bins[None, None] * 10 - prev_ms[..., None]).clamp(min=0)
        bucket = (torch.log2(delta / 10 + 1) * 6).long().clamp(max=63)
        state = bt['cell_state'][:, f0 // 4:f0 // 4 + FE.CORE_BINS // 4].repeat_interleave(4, -1)
        if self.c.time_head == 'linear':  # diagnostic: classify 800 bins from the hidden state alone
            bin_logits = self.time_linear(h) + self.state_bias(state).squeeze(-1)[:, None]
        else:
            bin_logits = cell_score + fine_score + self.rel_bias(bucket).squeeze(-1) + self.state_bias(state).squeeze(-1)[:, None]
        if self.c.selection and getattr(self, '_sel_core', None) is not None:
            bias = torch.sigmoid(self.sel_lam) * 2 * torch.tanh(self._sel_core / 2)  # [B,200]
            bin_logits = bin_logits + bias.repeat_interleave(4, -1)[:, None]
        ms = bins[:, None] * 10 + torch.arange(10, device=dev)[None]  # [800,10]
        ok = bt['bin_ok'][:, None] & (ms[None, None] > prev_ms[..., None, None])  # [B,Ny,800,10]
        bin_mask = ok.any(-1)
        # COPY_TIME anchors
        a_ms = (bt['copy_t'] * 1000).round()
        a_bin = (a_ms / 10).long().clamp(0, FE.CORE_BINS - 1)
        a_cell = (FE.HALO_US // FE.CELL_US + a_bin // 4)
        a_audio = ptr_audio.mean(2).gather(1, a_cell[..., None].expand(-1, -1, P))
        a_fine = fine.gather(1, (f0 + a_bin)[..., None].expand(-1, -1, P))
        a_key = self.copy_key(torch.cat([a_audio, a_fine], -1))
        copy_logits = torch.einsum('byp,bap->bya', qc, a_key) / math.sqrt(P)
        copy_mask = bt['copy_valid'][:, None] & (a_ms[:, None] > prev_ms[..., None])
        logits = torch.cat([self.end_head(h), bin_logits, copy_logits], -1)
        mask = torch.cat([torch.ones(B, Ny, 1, dtype=torch.bool, device=dev), bin_mask, copy_mask], -1)
        return logits, mask, ok

    def ptr_features(self, audio):
        """Pointer/selection audio features [B,400,8,P] from raw local taps [B,T,400,8,Dt]."""
        taps = torch.stack([ln(audio[:, j]) for j, ln in enumerate(self.tap_ln)], 1) + self.band
        if self.c.readout == 'design_package':
            return self.ptr_proj(taps[:, self.c.taps.index(self.c.top_tap)])
        return self.ptr_proj(torch.einsum('t,btcnd->bcnd', self.ptr_mix.softmax(0).to(taps.dtype), taps))

    def selection_field(self, ptr_audio, intent_h):
        """s(cell) from target audio (all 8 bands) querying isolated intent Z; never reads events."""
        q = self.sel_audio(ptr_audio)  # [B,C,8,64]
        k = self.sel_z(intent_h)  # [B,I,64]
        att = torch.einsum('bcnk,bik->bcni', q, k).div(8).softmax(-1)
        v = self.sel_v(intent_h)[..., 0]  # [B,I]
        band = torch.einsum('bcni,bi->bcn', att, v)
        return torch.logsumexp(band, -1) - math.log(band.shape[-1])  # [B,C]

    def intent_only(self, refs, ref_audio, refs_valid):
        """I rows from R alone (the CF path): identical math to the full pass, since I reads only R/I."""
        c = self.c
        B, dev = refs.shape[0], refs.device
        bt = {'refs': refs, 'ref_audio': ref_audio, 'refs_valid': refs_valid}
        ra = torch.stack([ln(ref_audio[:, :, j]) for j, ln in enumerate(self.tap_ln)], 2)
        ra = torch.einsum('t,brtnd->brnd', self.ref_mix.softmax(0).to(ra.dtype), ra) + self.band
        w = torch.einsum('brnd,d->brn', ra, self.ref_band_q).softmax(-1)
        R = self._events(refs) + self.ref_audio(torch.einsum('brn,brnd->brd', w, ra)) + self.row_kind.weight[0]
        I = (self.intent[None] + self.row_kind.weight[1]).expand(B, -1, -1)
        x = torch.cat([R, I], 1)
        nr, ni = R.shape[1], I.shape[1]
        kinds = torch.cat([torch.zeros(nr, device=dev), torch.ones(ni, device=dev)])
        allow = ((kinds[:, None] == 0) & (kinds[None] == 0)) | ((kinds[:, None] == 1) & (kinds[None] <= 1))
        rv = torch.cat([refs_valid, torch.ones(B, ni, dtype=torch.bool, device=dev)], 1)
        rows = allow[None] & rv[:, None, :]
        rows = torch.where(rv[:, :, None], rows, torch.eye(nr + ni, dtype=torch.bool, device=dev)[None])
        t = torch.cat([refs[..., 0], torch.zeros(B, ni, device=dev)], 1)
        timed = torch.cat([refs_valid.float(), torch.zeros(B, ni, device=dev)], 1)
        empty_taps = x.new_zeros(B, len(c.taps), 0, c.tap_dim)
        empty_t = x.new_zeros(B, 0)
        for layer in self.layers:
            x = layer(x, empty_taps, empty_t, t, timed, rows[:, None])
        return self.norm(x)[:, nr:]

    def residual_logits(self, h_sel, fine, b_sel):
        """Row-wise form (sampling): h_sel [N,d], fine [N,F,P] per row, b_sel [N] chosen bin -> [N,10]."""
        return self.residual_logits_bg(h_sel[:, None], fine, b_sel[:, None])[:, 0]

    def residual_logits_bg(self, h, fine, bins):
        """All group slots at once: h [B,G,d], fine [B,F,P] (per core, gathered here), bins [B,G] -> [B,G,10]."""
        B, G, _ = h.shape
        P, f0 = fine.shape[-1], FE.HALO_US // FE.BIN_US
        idx = (f0 + bins[..., None] + torch.arange(-1, 2, device=bins.device)).clamp(0, fine.shape[1] - 1)  # [B,G,3]
        ctx = fine.gather(1, idx.reshape(B, G * 3)[..., None].expand(-1, -1, P)).reshape(B, G, 3 * P)
        return self.res_head(torch.cat([self.res_q(h), ctx], -1))

    def gru_init(self, h_sel, ptr_audio_sel, fine_sel, t_new, right_new, busy):
        """Row-wise form (sampling): one core per row."""
        return self.gru_init_bg(h_sel[:, None], ptr_audio_sel, fine_sel, t_new[:, None], right_new[:, None], busy[:, None])[:, 0]

    def gru_init_bg(self, h, ptr_audio, fine, t_new, right_new, busy):
        """Audio read back at the chosen time (±3 cells × 8 bands) initialises the inner GRU, for all group
        slots: h [B,G,d], ptr_audio [B,400,8,P], fine [B,F,P], t_new [B,G] s, right_new [B,G,4], busy [B,G,4].
        Gathers per core instead of materialising per-row copies of the audio (1.2 GB per step before)."""
        B, G, _ = h.shape
        P = ptr_audio.shape[-1]
        cell = FE.HALO_US // FE.CELL_US + (t_new * FE.SEC / FE.CELL_US).floor().long()
        idx = (cell[..., None] + torch.arange(-3, 4, device=cell.device)).clamp(0, FE.LOCAL_CELLS - 1)  # [B,G,7]
        win = ptr_audio.gather(1, idx.reshape(B, G * 7)[..., None, None].expand(-1, -1, 8, P)).reshape(B, G, 56, P)
        w = torch.einsum('bgp,bgkp->bgk', self.gather_q(h), win).div(math.sqrt(P)).softmax(-1)
        pooled = torch.einsum('bgk,bgkp->bgp', w, win)
        frame = (FE.HALO_US // FE.BIN_US + (t_new * FE.SEC / FE.BIN_US).floor().long()).clamp(0, fine.shape[1] - 1)
        f = fine.gather(1, frame[..., None].expand(-1, -1, P))  # [B,G,P]
        if not self.c.gather_after_time:  # diagnostic: GRU starts from the outer state only
            pooled, f = torch.zeros_like(pooled), torch.zeros_like(f)
        return torch.tanh(self.init(torch.cat([h, pooled, f, self.right_g(self._right_feats(right_new)), busy.float()], -1)))

    def forward(self, bt):
        """Teacher-forced losses for a collated batch. Returns (total_loss, metrics)."""
        h, ptr_audio, fine, ty = self.encode(bt)
        B, Ny, d = h.shape
        G = Ny - 1
        dev = h.device
        m = {}
        self._sel_core = None
        if self.c.selection:
            c0c = FE.HALO_US // FE.CELL_US
            s_core = self.selection_field(ptr_audio[:, c0c:c0c + FE.CORE_US // FE.CELL_US], self._intent_h)
            self._sel_core = s_core
            lab = bt['cell_sel']  # [B,200] 1 head, 0 none, -1 unknown
            pos, neg = lab == 1, lab == 0
            bce = F.binary_cross_entropy_with_logits(s_core.float(), pos.float(), reduction='none')
            n_pos, n_neg = pos.sum(), neg.sum()
            both = (n_pos > 0) & (n_neg > 0)  # balanced BCE only when both classes are present (as before)
            l_sel = 0.5 * ((bce * pos).sum() / n_pos.clamp(min=1) + (bce * neg).sum() / n_neg.clamp(min=1)) * both
            hit = (((s_core > 0) == pos) & (pos | neg)).sum() * both
            m['select'] = (l_sel * bt.get('w_select', 1.0), (n_pos + n_neg) * both, hit, torch.zeros_like(hit))
        t_prev = torch.cat([torch.full((B, 1), -0.001, device=dev), bt['grp_t']], 1)
        logits, mask, ok = self.time_logits(bt, h, ptr_audio, fine, t_prev)
        row_valid = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=dev), bt['grp_valid']], 1)
        # target per Y row i: group i (bin / copy) or END after the last group
        tgt = torch.zeros(B, Ny, dtype=torch.long, device=dev)
        has = torch.cat([bt['grp_valid'], torch.zeros(B, 1, dtype=torch.bool, device=dev)], 1)
        copy = torch.cat([bt['grp_copy'], torch.full((B, 1), -1, device=dev)], 1)
        binx = torch.cat([bt['grp_bin'], torch.zeros(B, 1, dtype=torch.long, device=dev)], 1)
        tgt = torch.where(has & (copy >= 0), 1 + FE.CORE_BINS + copy, torch.where(has, 1 + binx, tgt))
        m['time'] = masked_ce(logits, tgt, row_valid, mask, 'time')
        # 1 ms residual for free-bin targets; every group slot is computed, padding is masked (static shapes)
        hg = h[:, :G]  # Y row i predicts group i
        free = bt['grp_valid'] & (bt['grp_copy'] < 0)
        gbin = bt['grp_bin'].clamp(min=0)
        res = self.residual_logits_bg(hg, fine, gbin)
        res_mask = ok.gather(2, gbin[..., None, None].expand(-1, -1, 1, ok.shape[-1]))[:, :, 0]  # [B,G,10]
        m['residual'] = masked_ce(res, bt['grp_res'], free, res_mask, 'residual')
        # inner GRU over every group slot (teacher forced; invalid slots masked out of every loss)
        self._inner(bt, hg, ptr_audio, fine, m)
        total, metrics, illegal = 0.0, {}, 0
        for k, (l, n, cnum, bad) in m.items():
            total = total + l  # zero when the head had no valid rows
            metrics[k] = (l.detach(), n, cnum)
            illegal = illegal + bad
        metrics['_illegal'] = illegal  # device tensor; checked by loop.summarize
        return total, metrics

    def _inner(self, bt, hg, ptr_audio, fine, m):
        """Inner note decoder over all B x G group slots (fixed MAX_NOTES + 1 GRU steps; masks carry validity)."""
        B, G, _ = hg.shape
        N, dev = B * G, hg.device
        valid = bt['grp_valid'].reshape(N)
        hs = self.gru_init_bg(hg, ptr_audio, fine, bt['grp_t'], bt['right_new'], bt['busy']).reshape(N, -1)
        n_notes = (bt['grp_n'] * bt['grp_valid']).reshape(N)
        tl, ts, tx = self._target_loc(bt)
        flat = lambda t: t.reshape(N, *t.shape[2:])
        typ, lane, line = flat(bt['note_type']), flat(bt['note_lane']), flat(tl)
        side, x, dur = flat(ts), flat(tx), flat(bt['note_dur'])
        true_line, true_side, true_x = flat(bt['note_line']), flat(bt['note_side']), flat(bt['note_x'])
        mode = bt['mode'].repeat_interleave(G)
        # per-mode legal note types (buffer: a host->device constant copy is not allowed during graph capture)
        allowed = (bt['allowed_types'] & self.mode_types[bt['mode']]).repeat_interleave(G, 0)
        n_lines = bt['lines'].repeat_interleave(G)
        used = flat(bt['busy']).clone()
        acc = {k: [] for k in ('cont', 'type', 'lane', 'line', 'side')}
        x_nll, o2_steps = [], []
        inp = self.start.expand(N, -1)
        for j in range(MAX_NOTES + 1):
            hs = self.gru_cell(inp, hs)
            active = (j <= n_notes) & valid  # this step exists (decides continue or emits note j)
            if j >= 1:
                acc['cont'].append((self.cont(hs), (j < n_notes).long(), active, None))
            if j == MAX_NOTES:
                break
            emit = (j < n_notes) & valid
            jt = typ[:, j]
            acc['type'].append((self.type_h(hs), jt, emit, allowed))
            o1 = hs + self.type_c(jt)
            is_m, is_p = emit & (mode == 0), emit & (mode == 1)
            lane_mask = ~used
            acc['lane'].append((self.lane_h(torch.cat([o1, used.float()], -1)), lane[:, j].clamp(min=0), is_m, lane_mask))
            if self.c.phira_location:
                line_logits = self.line_q(o1) @ self.line_tab.weight.T
                line_mask = torch.arange(FE.MAX_LINES, device=dev)[None] < n_lines[:, None]
                acc['line'].append((line_logits, true_line[:, j].clamp(min=0), is_p, line_mask))
                le = self.line_tab(true_line[:, j].clamp(0, FE.MAX_LINES - 1))
                acc['side'].append((self.side_h(torch.cat([o1, le], -1)), true_side[:, j].clamp(min=0), is_p, None))
                x_nll.append((logistic_mixture_nll(self.x_h(torch.cat([o1, le], -1)), true_x[:, j]), is_p))
            else:
                le = torch.zeros(N, 64, device=dev)  # Phira placement unknown to the generator
            loc = torch.where(mode[:, None] == 0, torch.cat([onehot(lane[:, j].clamp(min=0), 4).float(),
                                                             torch.zeros(N, 60, device=dev)], -1), le)
            o2 = o1 + self.loc_c(torch.cat([loc, self.side_e((side[:, j] + 1).clamp(0, 2)), x[:, j, None]], -1))
            o2_steps.append(o2)
            used = used | (is_m[:, None] & onehot(lane[:, j].clamp(min=0), 4))
            inp = self._note_emb(jt, lane[:, j], line[:, j], side[:, j], x[:, j], dur[:, j])
        for k, items in acc.items():
            if not items:
                continue
            lg = torch.cat([a for a, _, _, _ in items])
            tg = torch.cat([b for _, b, _, _ in items])
            vl = torch.cat([v for _, _, v, _ in items])
            mk = None if items[0][3] is None else torch.cat([mm for _, _, _, mm in items])
            m[k] = masked_ce(lg, tg, vl, mk, k)
        if x_nll:
            nll = torch.cat([a for a, _ in x_nll])
            w = torch.cat([b for _, b in x_nll]).float()
            n = w.sum()
            zero = torch.zeros((), dtype=torch.long, device=dev)
            m['x'] = ((nll * w).sum() / n.clamp(min=1), n, zero, zero)
        hi = bt['hold_index']  # [H,3] (b, g, n); collate pads holds and marks them in hold_valid
        if hi.shape[0]:
            o = torch.stack(o2_steps, 1)[hi[:, 0] * G + hi[:, 1], hi[:, 2]]
            self._hold(bt, torch.arange(hi.shape[0], device=dev), o, m, bt.get('hold_valid'))

    def _hold(self, bt, k, o, m, valid=None):
        dev = bt['mode'].device
        onset = bt['hold_onset_us'][k]
        lo, hi = bt['hold_ok_lo'][k], bt['hold_ok_hi'][k]
        b0 = onset // FE.BLOCK_US
        blocks = torch.arange(FE.MAX_END_BLOCKS, device=dev)
        bstart = (b0[:, None] + blocks[None]) * FE.BLOCK_US
        blk_mask = (bstart < hi[:, None] + 1) & (bstart + FE.BLOCK_US > lo[:, None])
        feats = self.blk_band(self.sum_ln(bt['hold_blocks'][k])).flatten(-2)  # [H,16,256]
        blk_logits = torch.einsum('hd,hkd->hk', self.blk_q(o), feats) / 16 + self.blk_off.weight.T
        ones = torch.ones(len(k), dtype=torch.bool, device=dev) if valid is None else valid[k]
        m['end_block'] = masked_ce(blk_logits, bt['hold_block'][k], ones, blk_mask, 'end_block')
        # block detail is revealed only for the (teacher-forced) chosen block
        tb = bt['hold_block'][k]
        det = self.det_proj(self.tap_ln[self.c.taps.index(self.c.top_tap)](bt['hold_detail'][k]) + self.band)  # [H,50,8,P]
        q = self.det_q(torch.cat([o, feats.gather(1, tb[:, None, None].expand(-1, 1, feats.shape[-1]))[:, 0]], -1))
        cell = torch.logsumexp(torch.einsum('hp,hcnp->hcn', q, det) / math.sqrt(q.shape[-1]), -1)
        coarse_logits = cell.repeat_interleave(4, -1) + self.det_sub.weight.T.repeat(1, END_BLOCK_CELLS)
        start = (b0 + tb) * FE.BLOCK_US
        cb = start[:, None] + torch.arange(FE.BLOCK_US // FE.BIN_US, device=dev)[None] * FE.BIN_US
        coarse_mask = (cb <= hi[:, None]) & (cb + FE.BIN_US - FE.MS >= lo[:, None])  # some ms point in [lo, hi]
        m['end_coarse'] = masked_ce(coarse_logits, bt['hold_coarse'][k], ones, coarse_mask, 'end_coarse')
        cbin = start + bt['hold_coarse'][k] * FE.BIN_US
        fb = cbin[:, None] + torch.arange(10, device=dev)[None] * FE.MS
        fine_mask = (fb <= hi[:, None]) & (fb >= lo[:, None])  # exact ms end times; lo = onset + 1 us
        ctx = det.mean(2).gather(1, (bt['hold_coarse'][k] // 4)[:, None, None].expand(-1, 1, det.shape[-1]))[:, 0]
        m['end_fine'] = masked_ce(self.end_fine(torch.cat([o, ctx], -1)), bt['hold_fine'][k], ones, fine_mask, 'end_fine')
