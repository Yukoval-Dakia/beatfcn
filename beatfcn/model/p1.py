"""P1 event model v2.1 (FIRST_CANDIDATE_v2.md §3): whole-song understanding stream + time head v2.

Rows (one sequence per core): R reference evidence | I 8 intent | U controls+mode | K known events |
Y BOS + completed groups. Each of 8 layers runs ONE softmax over [frozen multi-tap full-band local audio
K/V (GQA) ; whole-song tokens S as K/V memory ; row K/V]. Visibility: R->R; I->R,I; U,K->R,I,U,K;
Y->prefix + causal Y. Local audio is read by K rows (t±2 s) and Y rows ([t-1 s, t+3 s)); S is read by K
and Y rows without a window; time enters every timed row, audio key and song token as 1D RoPE (seconds
from the core start, periods 20 ms .. 2400 s).

Song stream (SongStream): every 400 ms block of the song (up to 10.7 min, a window centred on the core
beyond that) becomes one token: a learned block encoder over the block's frozen cells (every other cell
of the last two taps, uniform tap mix -> projection -> in-block attention -> attention pooling) plus a chart token from the block's KNOWN
events (observed + generated prefix: occupancy grid, coverage, counts), plus the pooled controls; a
4-layer bidirectional transformer over the blocks yields S. A masked-block regression objective on the
audio tokens (same transformer, audio only) is added during training (SSL).

Heads: one pointer softmax over {END, 200 free 40 ms cells, COPY_TIME anchors}; the in-cell offset (40 x
1 ms) read from the chosen cell's frozen tokens (±1 cell, 8 bands) by a shared offset head; both biased
by a learned rhythm interval kernel over the most recent onsets (no BPM, no beat tracker). A GRU per
group initialised with audio read back at the chosen time; type -> location -> hold end (2 s block ->
40 ms cell -> the same offset head). Losses are teacher-forced, each field normalised by its own count.
Generation lives in sample.py.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F
from . import features as FE
from .batch import MAX_NOTES, END_BLOCK_CELLS, CORE_CELLS, MS_PER_CELL

NEG = -1e9
ROPE_MIN_S, ROPE_MAX_S = 0.02, 2400.0


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
    gather_after_time: bool = True
    compact_inner: bool = False  # optional CPU-indexed GRU compaction; audio/attention and all targets unchanged
    # whole-song stream (v2.1 §3.2)
    song: bool = True
    song_layers: int = 4
    song_heads: int = 8
    blk_d: int = 128  # block encoder width
    blk_layers: int = 1  # in-block attention layers
    blk_pool: int = 4  # attention-pooling queries per block
    ssl_mask: float = 0.15  # masked-block regression: fraction of audio block tokens masked
    ssl_weight: float = 0.1
    kernel_buckets: int = 128  # rhythm interval kernel: log2-ratio buckets per previous onset


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.w, self.eps = nn.Parameter(torch.ones(d)), eps

    def forward(self, x):
        return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).to(x.dtype) * self.w


def rope(x, t, timed, lo=ROPE_MIN_S, hi=ROPE_MAX_S):
    """x [B,H,N,D]; t [B,N] seconds; untimed rows keep identity. Periods lo .. hi (log-spaced)."""
    d = x.shape[-1] // 2
    period = torch.logspace(math.log10(lo), math.log10(hi), d, device=x.device)
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
        self.ks = nn.Linear(c.d, c.audio_kv_heads * hd, bias=False)  # song tokens as K/V memory
        self.vs = nn.Linear(c.d, c.audio_kv_heads * hd, bias=False)
        self.mix = nn.Parameter(torch.zeros(len(c.taps)))
        self.f1, self.f2 = nn.Linear(c.d, 2 * c.ffn, bias=False), nn.Linear(c.ffn, c.d, bias=False)
        self.drop = nn.Dropout(c.dropout)

    def _expand(self, k, v):
        rep = self.c.heads // self.c.audio_kv_heads
        return k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)

    def audio_kv(self, taps_ln, audio_t):
        """Rotated audio keys / values for this layer, expanded to every query head: [B,H,Na,hd] x 2."""
        c, B = self.c, taps_ln.shape[0]
        hd = c.d // c.heads
        audio = torch.einsum('t,btnd->bnd', self.mix.softmax(0).to(taps_ln.dtype), taps_ln)
        Na = audio.shape[1]
        ka, va = (f(audio).view(B, Na, c.audio_kv_heads, hd).transpose(1, 2) for f in (self.ka, self.va))
        return self._expand(rope(ka, audio_t, torch.ones_like(audio_t)), va)

    def song_kv(self, S, s_t):
        """Rotated song-token keys / values [B,H,NB,hd] x 2 (read-only memory: S rows are never queries)."""
        c, (B, NB, _) = self.c, S.shape
        hd = c.d // c.heads
        ks, vs = (f(S).view(B, NB, c.audio_kv_heads, hd).transpose(1, 2) for f in (self.ks, self.vs))
        return self._expand(rope(ks, s_t, torch.ones_like(s_t)), vs)

    def row_qkv(self, x, row_t, row_timed):
        c, (B, N, _) = self.c, x.shape
        hd = c.d // c.heads
        z = self.n1(x)
        q, k, v = (f(z).view(B, N, c.heads, hd).transpose(1, 2) for f in (self.q, self.k, self.v))
        return rope(q, row_t, row_timed), rope(k, row_t, row_timed), v

    def finish(self, x, att):
        c, (B, N, _) = self.c, x.shape
        x = x + self.drop(self.o(att.transpose(1, 2).reshape(B, N, c.d)))
        a, b = self.f1(self.n2(x)).chunk(2, -1)
        return x + self.drop(self.f2(F.silu(a) * b))

    def forward(self, x, taps_ln, audio_t, row_t, row_timed, mask, cache=None, S=None, s_t=None):
        """mask columns: [local audio | song tokens (if S) | rows]. cache (dict, inference): receives this
        layer's rotated memory K/V (audio + song) and row K/V for incremental decoding."""
        q, k, v = self.row_qkv(x, row_t, row_timed)
        ka, va = self.audio_kv(taps_ln, audio_t)
        if S is not None:
            ks, vs = self.song_kv(S, s_t)
            ka, va = torch.cat([ka, ks], 2), torch.cat([va, vs], 2)
        if cache is not None:
            cache.update(ka=ka, va=va, k=k, v=v)
        att = F.scaled_dot_product_attention(q, torch.cat([ka, k], 2), torch.cat([va, v], 2), attn_mask=mask)
        return self.finish(x, att)

    def forward_step(self, x_new, row_t, row_timed, mask_row, cache):
        """One appended row [B,1,d] attending to the cached memory and rows (plus itself); extends the cache."""
        q, k, v = self.row_qkv(x_new, row_t, row_timed)
        cache['k'], cache['v'] = torch.cat([cache['k'], k], 2), torch.cat([cache['v'], v], 2)
        att = F.scaled_dot_product_attention(q, torch.cat([cache['ka'], cache['k']], 2), torch.cat([cache['va'], cache['v']], 2),
                                             attn_mask=mask_row)
        return self.finish(x_new, att)


def _ms_floor(t_s):
    """Seconds (float32, exact microseconds) -> whole milliseconds, floored, as float: round to us first so
    7.999999 s is 7999 ms, not 7998 (float32 7.9999990463) nor 8000 (round)."""
    return torch.div((t_s.double() * 1e6).round(), 1000, rounding_mode='floor').float()


def onehot(x, n):
    """F.one_hot without its host-side range check (a device sync); inputs are clamped / validated upstream."""
    return x[..., None] == torch.arange(n, device=x.device)


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


# ---------------------------------------------------------------- whole-song stream
def _mix_cells(c, dt):
    """[T, C, 8, Dt] (int8 or float) -> [C*8, Dt] uniform tap mean in dtype dt, one tap at a time."""
    acc = c[0].to(dt)
    for t in range(1, c.shape[0]):
        acc = acc + c[t].to(dt)
    scale = c.shape[0] * (32.0 if c.dtype == torch.int8 else 1.0)  # int8 quantised units ~N(0, 32): standardise
    return (acc / scale).reshape(-1, c.shape[-1])


class Int8MixProj(torch.autograd.Function):
    """out[b] = mean_t(cells[b, t]).reshape(C*8, Dt) @ W, one core at a time: the int8 song cells stay
    resident (nothing of size B x T x C x 8 x Dt is ever materialised or saved in floating point); the
    backward recomputes the cast per core for grad W. Graph-capturable (static loops, no host syncs)."""

    @staticmethod
    def forward(ctx, cells, W):
        dt = W.dtype
        if cells.device.type == 'cuda' and torch.is_autocast_enabled('cuda'):
            dt = torch.get_autocast_dtype('cuda')
        w = W.to(dt)
        out = torch.stack([_mix_cells(cells[b], dt) @ w for b in range(cells.shape[0])])
        ctx.save_for_backward(cells, W)
        ctx.dt = dt
        return out

    @staticmethod
    def backward(ctx, g):
        cells, W = ctx.saved_tensors
        gw = torch.zeros_like(W)
        for b in range(cells.shape[0]):
            x = _mix_cells(cells[b], ctx.dt)
            gw = gw + (x.transpose(0, 1) @ g[b].to(ctx.dt)).to(W.dtype)
        return None, gw


class _Attn(nn.Module):
    """Pre-norm self-attention + gated FFN block (song stream); optional RoPE over times and key mask."""

    def __init__(self, d, heads, ffn, dropout):
        super().__init__()
        self.heads = heads
        self.n1, self.n2 = RMSNorm(d), RMSNorm(d)
        self.qkv, self.o = nn.Linear(d, 3 * d, bias=False), nn.Linear(d, d, bias=False)
        self.f1, self.f2 = nn.Linear(d, 2 * ffn, bias=False), nn.Linear(ffn, d, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, t=None, mask=None, lo=1.0):
        N, L, d = x.shape
        hd = d // self.heads
        q, k, v = self.qkv(self.n1(x)).view(N, L, 3, self.heads, hd).permute(2, 0, 3, 1, 4)
        if t is not None:
            ones = torch.ones_like(t)
            q, k = rope(q, t, ones, lo), rope(k, t, ones, lo)
        att = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        x = x + self.drop(self.o(att.transpose(1, 2).reshape(N, L, d)))
        a, b = self.f1(self.n2(x)).chunk(2, -1)
        return x + self.drop(self.f2(F.silu(a) * b))


class SongStream(nn.Module):
    """Whole-song tokens S [B, NB, d] from the song cells (block encoder), the known-chart block features and
    the pooled controls; plus the masked-block regression loss on the audio tokens (training)."""

    def __init__(self, c: P1Config):
        super().__init__()
        self.c = c
        d, bd, L = c.d, c.blk_d, FE.BLOCK_CELLS // FE.STREAM_STRIDE * 8
        self.proj = nn.Parameter(torch.randn(c.tap_dim, bd) * (1.0 / math.sqrt(c.tap_dim)))
        self.pos = nn.Parameter(torch.randn(L, bd) * 0.02)  # cell x band position inside a block
        self.blk_layers = nn.ModuleList(_Attn(bd, 4, 3 * bd, c.dropout) for _ in range(c.blk_layers))
        self.pool_q = nn.Parameter(torch.randn(c.blk_pool, bd) * 0.02)
        self.pool_k, self.pool_v = nn.Linear(bd, bd, bias=False), nn.Linear(bd, bd, bias=False)
        self.audio_out = nn.Linear(c.blk_pool * bd, d)
        self.chart_in = nn.Sequential(nn.Linear(FE.BLOCK_CELLS * 4 * 2 + 3 + 4, d), nn.GELU(), nn.Linear(d, d))
        self.time_in = nn.Linear(2, d)
        self.mask_tok = nn.Parameter(torch.randn(d) * 0.02)
        self.layers = nn.ModuleList(_Attn(d, c.song_heads, c.ffn, c.dropout) for _ in range(c.song_layers))
        self.norm = RMSNorm(d)
        self.ssl_head = nn.Linear(d, c.tap_dim)

    def audio_tokens(self, cells):
        """[B, T, C, 8, Dt] song cells -> [B, NB, d] audio block tokens."""
        B, T, C, _, Dt = cells.shape  # C: stored cells (song cells / STREAM_STRIDE)
        per_blk = FE.BLOCK_CELLS // FE.STREAM_STRIDE
        NB, L, bd = C // per_blk, per_blk * 8, self.c.blk_d
        x = Int8MixProj.apply(cells, self.proj).view(B * NB, L, bd) + self.pos.to(cells.device)
        for layer in self.blk_layers:
            x = layer(x)
        k, v = self.pool_k(x), self.pool_v(x)
        att = torch.einsum('pd,nld->npl', self.pool_q.to(k.dtype), k).div(math.sqrt(bd)).softmax(-1)
        pooled = torch.einsum('npl,nld->npd', att, v).reshape(B, NB, -1)
        return self.audio_out(pooled)

    @staticmethod
    def block_means(cells):
        """SSL regression target: per block, the mean frozen feature (uniform tap mix) [B, NB, Dt]; int8 cells
        are kept in quantised units / 32 (a per-dimension standardised scale)."""
        B, T, C, _, Dt = cells.shape
        NB = C // (FE.BLOCK_CELLS // FE.STREAM_STRIDE)
        out = []
        for b in range(B):
            acc = None
            for t in range(T):
                x = cells[b, t].float().view(NB, -1, Dt).mean(1)
                acc = x if acc is None else acc + x
            out.append(acc / T)
        out = torch.stack(out)
        return out / 32.0 if cells.dtype == torch.int8 else out

    def run(self, x, t, valid):
        mask = valid[:, None, None, :]
        for layer in self.layers:
            x = layer(x, t, mask)
        return self.norm(x)

    def forward(self, bt, cells, u_pool, ssl: bool):
        a_tok = self.audio_tokens(cells)
        t, valid = bt['blk_t'], bt['blk_valid']
        tfeat = self.time_in(torch.stack([torch.tanh(t / 60.0), bt['blk_cov'][..., 0]], -1))
        chart = self.chart_in(torch.cat([bt['blk_grid'].flatten(2).float(), bt['blk_cov'], bt['blk_counts']], -1))
        S = self.run(a_tok + chart + u_pool[:, None] + tfeat, t, valid)
        loss, n = None, None
        if ssl and self.c.ssl_weight > 0:
            with torch.no_grad():
                target = self.block_means(cells)
            # deterministic pseudo-random mask from the block times and counts (varies with the core position and
            # the chart; equal for equal batches, so eager / graph / compact paths stay step-equivalent in tests)
            u = torch.frac(torch.sin(t * 12.9898 + bt['blk_counts'].sum(-1) * 78.233 + bt['mode'][:, None].float()) * 43758.5453)
            m = (u < self.c.ssl_mask) & valid
            xa = torch.where(m[..., None], self.mask_tok.to(a_tok.dtype), a_tok) + tfeat
            pred = self.ssl_head(self.run(xa, t, valid)).float()
            err = ((pred - target) ** 2).mean(-1)
            n = m.sum()
            loss = (err * m).sum() / n.clamp(min=1)
        return S, loss, n


# ---------------------------------------------------------------- the event model
class P1Model(nn.Module):
    def __init__(self, c: P1Config = P1Config()):
        super().__init__()
        self.c = c
        d, T, P, H = c.d, len(c.taps), c.ptr, c.gru
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
        self.row_kind = nn.Embedding(5, d)  # R I U K Y
        self.intent = nn.Parameter(torch.randn(c.intent, d) * 0.02)
        self.ctrl_e = nn.Embedding(len(FE.CONTROL_FIELDS) * 32, d)
        self.mode_e = nn.Embedding(len(FE.MODES), d)
        self.allowed_p = nn.Linear(4, d)
        self.bos = nn.Parameter(torch.randn(d) * 0.02)
        self.right_p = nn.Linear(8, d)
        self.sum_ln = nn.LayerNorm(c.summary_dim)  # hold end-block candidates (2 s summary rows)
        # group / note embeddings (attribute-only feedback, design §4.2)
        self.note_p = nn.Linear(64 * 3 + 16 + 2, H)
        self.group_p = nn.Linear(2 * H, d)
        # song stream
        self.song = SongStream(c) if c.song else None
        # pointer over 40 ms cells
        self.ptr_mix = nn.Parameter(torch.zeros(T))
        self.ptr_proj = nn.Linear(c.tap_dim, P)
        self.q_a, self.q_c = nn.Linear(d, P), nn.Linear(d, P)
        self.end_head = nn.Linear(d, 1)
        self.rel_bias = nn.Embedding(64, 1)
        self.state_bias = nn.Embedding(3, 1)
        self.copy_key = nn.Sequential(nn.Linear(P, P), nn.GELU(), nn.Linear(P, P))
        # rhythm interval kernel (v2.1 §3.3 item 3): log2-ratio buckets per previous onset, zero at init
        self.kern = nn.Embedding(3 * c.kernel_buckets, 1)
        nn.init.zeros_(self.kern.weight)
        self.no_ctx = nn.Parameter(torch.zeros(()))
        # shared in-cell offset head (onsets and hold ends): query x the chosen cell's tokens (±1 cell, 8 bands)
        self.off_q = nn.Linear(d, P)
        self.off_hold_q = nn.Linear(H, P)
        self.off_pos = nn.Parameter(torch.randn(3 * 8, P) * 0.02)
        self.off_mlp = nn.Sequential(nn.Linear(2 * P, 256), nn.GELU(), nn.Linear(256, MS_PER_CELL))
        # inner GRU
        self.gather_q = nn.Linear(d, P)
        self.right_g = nn.Linear(8, 16)
        self.init = nn.Linear(d + P + 16 + 4, H)
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
        # hold end: 2 s block (summary rows) -> 40 ms cell (top-tap detail) -> shared offset head
        self.blk_band = nn.Linear(c.summary_dim, 32)
        self.blk_q = nn.Linear(H, 256)
        self.blk_off = nn.Embedding(FE.MAX_END_BLOCKS, 1)
        self.det_proj = nn.Linear(c.tap_dim, P)
        self.det_q = nn.Linear(H + 256, P)
        if c.selection:
            self.sel_audio = nn.Linear(P, 64)
            self.sel_z = nn.Linear(d, 64)
            self.sel_v = nn.Linear(d, 1)
            self.sel_lam = nn.Parameter(torch.zeros(()))  # lambda = sigmoid(.) in (0,1), init 0.5
        if c.readout == 'design_package':
            self.slot_q = nn.Parameter(torch.randn(4, c.tap_dim) * 0.02)
        if c.time_head == 'linear':
            self.time_linear = nn.Linear(d, CORE_CELLS)
        self._intent_h = None
        self._sel_core = None
        self._ssl = None

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
        K = self._events(bt['known']) + self.row_kind.weight[3]
        rows = [R, I, U, K]
        lens = [x.shape[1] for x in rows]
        zeros = lambda n: torch.zeros(B, n, device=dev)
        times = [bt['refs'][..., 0], zeros(lens[1]), zeros(lens[2]), bt['known'][..., 0]]
        timed = [bt['refs_valid'].float(), zeros(lens[1]), zeros(lens[2]), bt['known_valid'].float()]
        ones = lambda n: torch.ones(B, n, dtype=torch.bool, device=dev)
        valid = [bt['refs_valid'], ones(lens[1]), ones(lens[2]), bt['known_valid']]
        return rows, lens, times, timed, valid

    def _y_rows(self, bt, group_emb):
        B = bt['mode'].shape[0]
        Y = torch.cat([self.bos.expand(B, 1, -1), group_emb], 1) + self.row_kind.weight[4]
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
    def _mask(self, bt, lens, times, valid, n_y, per_cell=8, s_valid=None):
        """[B,1,N, Na + NB + N] boolean attention mask: local audio window | song tokens | rows."""
        c = self.c
        dev = bt['mode'].device
        B = bt['mode'].shape[0]
        kinds = torch.cat([torch.full((n,), k, device=dev) for k, n in enumerate(lens + [n_y])])  # R0 I1 U2 K3 Y4
        q, k = kinds[:, None], kinds[None, :]
        allow = torch.zeros(len(kinds), len(kinds), dtype=torch.bool, device=dev)
        allow |= (q == 0) & (k == 0)
        allow |= (q == 1) & (k <= 1)
        allow |= (q >= 2) & (q <= 3) & (k <= 3)
        allow |= (q == 4) & (k <= 3)
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
        kq = (kinds == 3)[None, :, None]
        yq = (kinds == 4)[None, :, None]
        if c.audio_window:
            win = (kq & (dt >= c.k_window[0]) & (dt <= c.k_window[1])) | (yq & (dt >= c.y_window[0]) & (dt < c.y_window[1]))
        else:  # diagnostic: K and Y rows read all local audio
            win = (kq | yq).expand(-1, -1, FE.LOCAL_CELLS)
        win = win & bt['audio_valid'][:, None, :] & rv[:, :, None]
        win = win[..., None].expand(-1, -1, -1, per_cell).reshape(B, len(kinds), FE.LOCAL_CELLS * per_cell)
        parts = [win]
        if s_valid is not None:  # song tokens: K and Y rows, every valid block
            parts.append(s_valid[:, None, :] & (kq | yq) & rv[:, :, None])
        parts.append(rows)
        return torch.cat(parts, -1)[:, None]

    # ------------------------------------------------------------ forward
    def encode(self, bt, cache=None):
        """Transformer over all rows; returns Y hidden states [B, G+1, d], pointer features and Y times.
        cache (dict): filled for incremental decoding (encode_step appends Y rows without re-encoding)."""
        from .batch import dequant
        dequant(bt)  # in place, no-op when to_device already expanded the compact transport format
        c = self.c
        B = bt['mode'].shape[0]
        if cache is not None:
            if c.readout == 'design_package':
                raise ValueError('incremental decoding supports the multitap read-out only')
            cache['layers'] = [{} for _ in self.layers]
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
        S, s_t, s_valid = None, None, None
        self._ssl = None
        if self.song is not None and 'song_cells' in bt:
            S, ssl_loss, ssl_n = self.song(bt, bt['song_cells'], rows[2].mean(1), self.training)
            s_t, s_valid = bt['blk_t'], bt['blk_valid']
            if ssl_loss is not None:
                self._ssl = (ssl_loss, ssl_n)
        x = torch.cat(rows + [Y], 1)
        row_t = torch.cat(times + [ty], 1)
        row_timed = torch.cat(timed + [vy.float()], 1)
        mask = self._mask(bt, lens, times + [ty], valid + [vy], Y.shape[1], per_cell, s_valid)
        for i, layer in enumerate(self.layers):
            x = layer(x, taps, audio_t, row_t, row_timed, mask, None if cache is None else cache['layers'][i], S, s_t)
        hn = self.norm(x)
        h = hn[:, sum(lens):]
        if c.readout == 'design_package':
            ptr_audio = self.ptr_proj(ptr_src[:, c.taps.index(c.top_tap)])  # top tap only, 8 bands
        else:
            ptr_audio = self.ptr_proj(torch.einsum('t,btcnd->bcnd', self.ptr_mix.softmax(0).to(ptr_src.dtype), ptr_src))
        self._intent_h = hn[:, lens[0]:lens[0] + lens[1]]  # I rows (read R/I only)
        if cache is not None:  # rows in the cache: prefix + the (padded) Y block; appended rows follow
            cache.update(lens=lens, rows_valid=torch.cat(valid + [vy], 1), n_valid_y=int(vy.sum()), appended=0,
                         audio_valid=bt['audio_valid'], per_cell=per_cell, ptr_audio=ptr_audio, ty=ty,
                         intent_h=self._intent_h, blk_valid=s_valid)
        return h, ptr_audio, ty

    def encode_step(self, bt, cache):
        """Incremental decoding: `bt` holds G generated groups (collated as during generation) and the cache
        holds Y rows 0..G-1 (BOS + G-1 groups) with every prefix row, the audio and the song tokens. Computes
        Y row G (the one that predicts group G) through all layers against the cache, appends it, and returns
        its hidden state [B,1,d] with the per-core pointer features. Equals encode(bt)[0][:, G:G+1]."""
        c = self.c
        B = bt['mode'].shape[0]
        dev = bt['mode'].device
        if B != 1:
            raise ValueError('incremental decoding is per core (batch of one)')
        Y, ty, vy = self._y_rows(bt, self.group_embedding(bt))
        G = int(bt['grp_valid'].sum())  # valid groups (the Y block itself is padded to a bucket)
        if cache['n_valid_y'] == G + 1:  # row G was already computed for a time that produced no note: recompute it
            for lc in cache['layers']:
                lc['k'], lc['v'] = lc['k'][:, :, :-1], lc['v'][:, :, :-1]
            cache['n_valid_y'] -= 1
            cache['appended'] -= 1
        if cache['n_valid_y'] != G:
            raise ValueError(f'cache holds {cache["n_valid_y"]} Y rows, batch has {G} groups: rows are appended one at a time')
        x = Y[:, G:G + 1]  # the row that predicts group G: BOS for G == 0, else the embedding of group G - 1
        t_row, timed = ty[:, G:G + 1], torch.ones(B, 1, device=dev)
        # mask for this Y row: audio window (kind Y); song tokens; valid cached rows; itself
        prefix_ok = torch.cat([cache['rows_valid'], torch.ones(B, cache['appended'], dtype=torch.bool, device=dev)], 1)
        own = torch.ones(B, 1, dtype=torch.bool, device=dev)
        cell_t = (torch.arange(FE.LOCAL_CELLS, device=dev) + 0.5) * FE.CELL_US / FE.SEC - FE.HALO_US / FE.SEC
        dt = cell_t[None] - t_row  # [B,400]
        (w0, w1) = c.y_window
        win = ((dt >= w0) & (dt < w1)) if c.audio_window else torch.ones_like(dt, dtype=torch.bool)
        win = win & cache['audio_valid']
        win = win[..., None].expand(-1, -1, cache['per_cell']).reshape(B, -1)
        parts = [win] + ([cache['blk_valid']] if cache.get('blk_valid') is not None else []) + [prefix_ok, own]
        mask = torch.cat(parts, 1)[:, None, None]
        for i, layer in enumerate(self.layers):
            x = layer.forward_step(x, t_row, timed, mask, cache['layers'][i])
        cache['n_valid_y'] = G + 1
        cache['appended'] += 1
        self._intent_h = cache['intent_h']
        return self.norm(x), cache['ptr_audio'], ty

    # ------------------------------------------------------------ time head v2
    def _kernel(self, prev, tau):
        """Rhythm interval bias at candidate times tau [..., M] from the most recent onsets prev [..., 4]
        (seconds, most recent first, NO_ONSET padding): learned buckets of log2((tau - t_j) / (t_1 - t_2))
        for j = 1..3, i.e. integer multiples and fractions of the last inter-onset interval, plus a learned
        'no context' offset. Never reads a BPM or a beat tracker."""
        KB = self.c.kernel_buckets
        t1, t2 = prev[..., 0:1], prev[..., 1:2]
        have1, have2 = t1 > -500.0, t2 > -500.0
        ref = (t1 - t2).clamp(min=0.005)
        out = torch.where(have1, torch.zeros_like(tau), self.no_ctx.to(tau.dtype))
        for j in range(3):
            tj = prev[..., j:j + 1]
            dj = tau - tj
            ratio = (dj / ref).clamp(min=1e-3)
            b = (torch.log2(ratio) * 16 + 64).round().long().clamp(0, KB - 1) + j * KB
            e = self.kern(b)[..., 0]
            out = out + torch.where(have1 & have2 & (tj > -500.0) & (dj > 0), e, torch.zeros_like(e))
        return out

    def time_logits(self, bt, h, ptr_audio, t_prev):
        """Pointer over [END | 200 cells | A anchors] for each Y row; returns logits, legality mask and the
        per-cell ms legality ok [B,Ny,200,40] (for the offset head)."""
        B, Ny, _ = h.shape
        dev = h.device
        qa, qc = self.q_a(h), self.q_c(h)
        P = qa.shape[-1]
        h0c = FE.HALO_US // FE.CELL_US
        core_cells = ptr_audio[:, h0c:h0c + CORE_CELLS]  # [B,200,8,P]
        band = torch.einsum('byp,bcnp->bycn', qa, core_cells) / math.sqrt(P)
        cell_score = torch.logsumexp(band.float(), -1)  # [B,Ny,200]
        prev_ms = _ms_floor(t_prev)  # [B,Ny] ms from core start (BOS: -1); floor like the collated targets
        cell_ms = torch.arange(CORE_CELLS, device=dev) * MS_PER_CELL + MS_PER_CELL // 2
        delta = (cell_ms[None, None] - prev_ms[..., None]).clamp(min=0)
        bucket = (torch.log2(delta / 10 + 1) * 6).long().clamp(max=63)
        state = bt['cell_state'][:, h0c:h0c + CORE_CELLS]
        if self.c.time_head == 'linear':  # diagnostic: classify the cells from the hidden state alone
            cell_logits = self.time_linear(h).float() + self.state_bias(state).squeeze(-1)[:, None]
        else:
            tau = (cell_ms.float() / 1000.0)[None, None].expand(B, Ny, -1)
            cell_logits = (cell_score + self.rel_bias(bucket).squeeze(-1) + self.state_bias(state).squeeze(-1)[:, None]
                           + self._kernel(bt['prev_onsets'][:, :Ny], tau))
        if self.c.selection and self._sel_core is not None:
            cell_logits = cell_logits + (torch.sigmoid(self.sel_lam) * 2 * torch.tanh(self._sel_core / 2))[:, None]
        ms = torch.arange(CORE_CELLS, device=dev)[:, None] * MS_PER_CELL + torch.arange(MS_PER_CELL, device=dev)[None]  # [200,40]
        ok = bt['bin_ok'].reshape(B, CORE_CELLS, MS_PER_CELL)[:, None] & (ms[None, None] > prev_ms[..., None, None])  # [B,Ny,200,40]
        cell_mask = ok.any(-1)
        # COPY_TIME anchors
        a_ms = _ms_floor(bt['copy_t'])
        a_cell = (h0c + (a_ms / MS_PER_CELL).long().clamp(0, CORE_CELLS - 1))
        a_audio = ptr_audio.mean(2).gather(1, a_cell[..., None].expand(-1, -1, P))
        copy_logits = (torch.einsum('byp,bap->bya', qc, self.copy_key(a_audio)) / math.sqrt(P)).float()
        copy_mask = bt['copy_valid'][:, None] & (a_ms[:, None] > prev_ms[..., None])
        logits = torch.cat([self.end_head(h).float(), cell_logits, copy_logits], -1)
        mask = torch.cat([torch.ones(B, Ny, 1, dtype=torch.bool, device=dev), cell_mask, copy_mask], -1)
        return logits, mask, ok

    @staticmethod
    def _cell_tokens(tok, cell):
        """tok [B,C,8,P] cell tokens, cell [B,G] indices -> the chosen cell and its neighbours [B,G,3,8,P]."""
        B, G = cell.shape
        P = tok.shape[-1]
        idx = (cell[..., None] + torch.arange(-1, 2, device=cell.device)).clamp(0, tok.shape[1] - 1)
        return tok.gather(1, idx.reshape(B, G * 3)[..., None, None].expand(-1, -1, 8, P)).reshape(B, G, 3, 8, P)

    def offset_logits(self, q, tok, prev=None, base_s=None):
        """In-cell offset logits [N, 40]: q [N,P] (off_q(h) for onsets, off_hold_q(o) for hold ends) attends over
        the chosen cell's tokens tok [N,3,8,P]; prev [N,4] / base_s [N] (cell start, s) add the rhythm kernel."""
        N, P = q.shape
        tok = tok.reshape(N, 3 * 8, P) + self.off_pos.to(tok.dtype)
        w = torch.einsum('np,nkp->nk', q, tok).div(math.sqrt(P)).softmax(-1)
        pooled = torch.einsum('nk,nkp->np', w, tok)
        z = self.off_mlp(torch.cat([q, pooled], -1)).float()
        if prev is not None:
            tau = base_s[:, None] + (torch.arange(MS_PER_CELL, device=q.device).float() + 0.5) / 1000.0  # [N,40]
            z = z + self._kernel(prev[:, None, :], tau[:, None, :])[:, 0]
        return z

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

    def gru_init(self, h_sel, ptr_audio_sel, t_new, right_new, busy):
        """Row-wise form (sampling): one core per row."""
        return self.gru_init_bg(h_sel[:, None], ptr_audio_sel, t_new[:, None], right_new[:, None], busy[:, None])[:, 0]

    def gru_init_bg(self, h, ptr_audio, t_new, right_new, busy):
        """Audio read back at the chosen time (±3 cells × 8 bands) initialises the inner GRU, for all group
        slots: h [B,G,d], ptr_audio [B,400,8,P], t_new [B,G] s, right_new [B,G,4], busy [B,G,4]."""
        B, G, _ = h.shape
        P = ptr_audio.shape[-1]
        cell = FE.HALO_US // FE.CELL_US + (t_new * FE.SEC / FE.CELL_US).floor().long()
        idx = (cell[..., None] + torch.arange(-3, 4, device=cell.device)).clamp(0, FE.LOCAL_CELLS - 1)  # [B,G,7]
        win = ptr_audio.gather(1, idx.reshape(B, G * 7)[..., None, None].expand(-1, -1, 8, P)).reshape(B, G, 56, P)
        w = torch.einsum('bgp,bgkp->bgk', self.gather_q(h), win).div(math.sqrt(P)).softmax(-1)
        pooled = torch.einsum('bgk,bgkp->bgp', w, win)
        if not self.c.gather_after_time:  # diagnostic: GRU starts from the outer state only
            pooled = torch.zeros_like(pooled)
        return torch.tanh(self.init(torch.cat([h, pooled, self.right_g(self._right_feats(right_new)), busy.float()], -1)))

    def forward(self, bt):
        """Teacher-forced losses for a collated batch. Returns (total_loss, metrics)."""
        h, ptr_audio, ty = self.encode(bt)
        B, Ny, d = h.shape
        G = Ny - 1
        dev = h.device
        h0c = FE.HALO_US // FE.CELL_US
        m = {}
        self._sel_core = None
        if self.c.selection:
            s_core = self.selection_field(ptr_audio[:, h0c:h0c + CORE_CELLS], self._intent_h)
            self._sel_core = s_core
            lab = bt['cell_sel']  # [B,200] 1 head, 0 none, -1 unknown
            pos, neg = lab == 1, lab == 0
            bce = F.binary_cross_entropy_with_logits(s_core.float(), pos.float(), reduction='none')
            n_pos, n_neg = pos.sum(), neg.sum()
            both = (n_pos > 0) & (n_neg > 0)  # balanced BCE only when both classes are present (as before)
            l_sel = 0.5 * ((bce * pos).sum() / n_pos.clamp(min=1) + (bce * neg).sum() / n_neg.clamp(min=1)) * both
            hit = (((s_core > 0) == pos) & (pos | neg)).sum() * both
            m['select'] = (l_sel * bt.get('w_select', 1.0), (n_pos + n_neg) * both, hit, torch.zeros_like(hit))
        if self.training:
            self._intent_h = None  # no autograd graph kept alive across steps (stale-stream capture hazard)
        t_prev = torch.cat([torch.full((B, 1), -0.001, device=dev), bt['grp_t']], 1)
        logits, mask, ok = self.time_logits(bt, h, ptr_audio, t_prev)
        row_valid = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=dev), bt['grp_valid']], 1)
        # target per Y row i: group i (cell / copy) or END after the last group
        tgt = torch.zeros(B, Ny, dtype=torch.long, device=dev)
        has = torch.cat([bt['grp_valid'], torch.zeros(B, 1, dtype=torch.bool, device=dev)], 1)
        copy = torch.cat([bt['grp_copy'], torch.full((B, 1), -1, device=dev)], 1)
        cellx = torch.cat([bt['grp_cell'], torch.zeros(B, 1, dtype=torch.long, device=dev)], 1)
        tgt = torch.where(has & (copy >= 0), 1 + CORE_CELLS + copy, torch.where(has, 1 + cellx, tgt))
        m['time'] = masked_ce(logits, tgt, row_valid, mask, 'time')
        # in-cell offset for free-cell targets; every group slot is computed, padding is masked (static shapes)
        hg = h[:, :G]  # Y row i predicts group i
        free = bt['grp_valid'] & (bt['grp_copy'] < 0)
        gcell = bt['grp_cell'].clamp(0, CORE_CELLS - 1)
        tok = self._cell_tokens(ptr_audio, h0c + gcell)
        off = self.offset_logits(self.off_q(hg).reshape(B * G, -1), tok.reshape(B * G, 3, 8, -1),
                                 bt['prev_onsets'][:, :G].reshape(B * G, -1),
                                 (gcell.float() * (FE.CELL_US / FE.SEC)).reshape(-1)).view(B, G, MS_PER_CELL)
        off_mask = ok[:, :G].gather(2, gcell[..., None, None].expand(-1, -1, 1, MS_PER_CELL))[:, :, 0]  # [B,G,40]
        m['offset'] = masked_ce(off, bt['grp_off'], free, off_mask, 'offset')
        # inner GRU over every group slot (teacher forced; invalid slots masked out of every loss)
        self._inner(bt, hg, ptr_audio, m)
        if self._ssl is not None:
            loss, n = self._ssl
            zero = torch.zeros((), dtype=torch.long, device=dev)
            m['ssl'] = (self.c.ssl_weight * loss, n, zero, zero)
        total, metrics, illegal = 0.0, {}, 0
        for k, (l, n, cnum, bad) in m.items():
            total = total + l  # zero when the head had no valid rows
            metrics[k] = (l.detach(), n, cnum)
            illegal = illegal + bad
        metrics['_illegal'] = illegal  # device tensor; checked by loop.summarize
        return total, metrics

    def _inner(self, bt, hg, ptr_audio, m):
        """Inner note decoder over all B x G group slots (fixed MAX_NOTES + 1 GRU steps; masks carry validity)."""
        B, G, _ = hg.shape
        slots, dev = B * G, hg.device
        index = bt['inner_index'] if self.c.compact_inner else None
        select = lambda t: t if index is None else t.index_select(0, index)
        flat = lambda t: select(t.reshape(slots, *t.shape[2:]))
        valid = flat(bt['grp_valid'])
        hs = select(self.gru_init_bg(hg, ptr_audio, bt['grp_t'], bt['right_new'], bt['busy']).reshape(slots, -1))
        N = hs.shape[0]
        n_notes = flat(bt['grp_n'] * bt['grp_valid'])
        tl, ts, tx = self._target_loc(bt)
        typ, lane, line = flat(bt['note_type']), flat(bt['note_lane']), flat(tl)
        side, x, dur = flat(ts), flat(tx), flat(bt['note_dur'])
        true_line, true_side, true_x = flat(bt['note_line']), flat(bt['note_side']), flat(bt['note_x'])
        mode = select(bt['mode'].repeat_interleave(G))
        # per-mode legal note types (buffer: a host->device constant copy is not allowed during graph capture)
        allowed = select((bt['allowed_types'] & self.mode_types[bt['mode']]).repeat_interleave(G, 0))
        n_lines = select(bt['lines'].repeat_interleave(G))
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
            inner_rows = bt['hold_inner_index'] if self.c.compact_inner else hi[:, 0] * G + hi[:, 1]
            o = torch.stack(o2_steps, 1)[inner_rows, hi[:, 2]]
            self._hold(bt, torch.arange(hi.shape[0], device=dev), o, m, bt.get('hold_valid'))

    def hold_cells(self, o, blocks, detail, tb, prev=None, start_s=None):
        """Hold end, cell step: o [H,gru] note state, blocks [H,16,8,D] summary rows, detail [H,50,8,Dt] top-tap
        cells of the chosen block tb [H] -> (cell logits [H,50], block features [H,16,256], cell tokens [H,50,8,P]).
        prev [H,4] (absolute s: the hold's onset, then earlier onsets) and start_s [H] (block start, absolute s)
        add the rhythm interval kernel: hold lengths sit on the beat grid like onsets do."""
        feats = self.blk_band(self.sum_ln(blocks)).flatten(-2)  # [H,16,256]
        det = self.det_proj(self.tap_ln[self.c.taps.index(self.c.top_tap)](detail) + self.band)  # [H,50,8,P]
        q = self.det_q(torch.cat([o, feats.gather(1, tb[:, None, None].expand(-1, 1, feats.shape[-1]))[:, 0]], -1))
        cell = torch.logsumexp((torch.einsum('hp,hcnp->hcn', q, det) / math.sqrt(q.shape[-1])).float(), -1)
        if prev is not None:
            tau = start_s[:, None] + (torch.arange(END_BLOCK_CELLS, device=cell.device).float() + 0.5) * (FE.CELL_US / FE.SEC)
            cell = cell + self._kernel(prev[:, None, :], tau[:, None, :])[:, 0]
        return cell, feats, det

    def _hold(self, bt, k, o, m, valid=None):
        dev = bt['mode'].device
        onset = bt['hold_onset_us'][k]
        lo, hi = bt['hold_ok_lo'][k], bt['hold_ok_hi'][k]
        b0 = onset // FE.BLOCK_US
        blocks = torch.arange(FE.MAX_END_BLOCKS, device=dev)
        bstart = (b0[:, None] + blocks[None]) * FE.BLOCK_US
        blk_mask = (bstart < hi[:, None] + 1) & (bstart + FE.BLOCK_US > lo[:, None])
        tb = bt['hold_block'][k]
        start = (b0 + tb) * FE.BLOCK_US
        prev = bt['hold_prev'][k] if 'hold_prev' in bt else None
        cell, feats, det = self.hold_cells(o, bt['hold_blocks'][k], bt['hold_detail'][k], tb, prev, start.float() / FE.SEC)
        blk_logits = torch.einsum('hd,hkd->hk', self.blk_q(o), feats) / 16 + self.blk_off.weight.T
        ones = torch.ones(len(k), dtype=torch.bool, device=dev) if valid is None else valid[k]
        m['end_block'] = masked_ce(blk_logits, tb, ones, blk_mask, 'end_block')
        # block detail is revealed only for the (teacher-forced) chosen block: 40 ms cell, then the offset
        cb = start[:, None] + torch.arange(END_BLOCK_CELLS, device=dev)[None] * FE.CELL_US
        cell_mask = (cb <= hi[:, None]) & (cb + FE.CELL_US - FE.MS >= lo[:, None])  # some ms point in [lo, hi]
        tc = bt['hold_cell'][k]
        m['end_cell'] = masked_ce(cell, tc, ones, cell_mask, 'end_cell')
        tok = self._cell_tokens(det, tc[:, None])[:, 0]  # [H,3,8,P]
        off = self.offset_logits(self.off_hold_q(o), tok, prev, (start + tc * FE.CELL_US).float() / FE.SEC)
        fb = (start + tc * FE.CELL_US)[:, None] + torch.arange(MS_PER_CELL, device=dev)[None] * FE.MS
        off_mask = (fb <= hi[:, None]) & (fb >= lo[:, None])  # exact ms end times; lo = onset + 1 us
        m['end_off'] = masked_ce(off, bt['hold_off'][k], ones, off_mask, 'end_off')
