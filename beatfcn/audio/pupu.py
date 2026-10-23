"""Frozen PupuM2D Large EMA encoder, without predictor or masking.

Reconstructed from sizigi/PupuM2D and timm v1.0.22's audited EVA/RoPE
implementation. The checkpoint has NO q/k normalization parameters:
historical EvaBlock swallowed qk_norm in **kwargs without forwarding it.
Do not add QK normalization based only on args.json's qk_norm=true.

Input is the official normalized magnitude-log-mel tensor [B,1,T,128].
Output [B,T//4,1024] averages eight frequency patches. This frequency
aggregation is an explicitly chosen distillation target, not an official
released downstream recipe. return_tokens=True preserves [B,T//4,8,1024].
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F


def make_rope(grid: tuple[int, int], head_dim: int = 64, *, device=None):
    """Exact default pixel-mode axial RotaryEmbeddingCat from timm1.0.22."""
    bands = 2 ** torch.linspace(
        0, math.log(224., 2) - 1, head_dim // 4, dtype=torch.float32,
        device=device,
    )
    bands = bands * torch.pi
    axes = [torch.linspace(-1., 1., s, dtype=torch.float32, device=device) for s in grid]
    coord = torch.stack(torch.meshgrid(axes, indexing="ij"), dim=-1).unsqueeze(-1)
    phase = coord * bands
    sin = phase.sin().flatten(-2).flatten(0, 1).repeat_interleave(2, -1)
    cos = phase.cos().flatten(-2).flatten(0, 1).repeat_interleave(2, -1)
    return torch.cat((sin, cos), dim=-1)[None, None]


def apply_rope(x: torch.Tensor, rope: torch.Tensor):
    sin, cos = rope.chunk(2, -1)
    rot = torch.stack((-x[..., 1::2], x[..., ::2]), dim=-1).reshape(x.shape)
    return (x * cos + rot * sin).to(x.dtype)


class PatchEmbed(nn.Module):
    def __init__(self, dim: int = 1024):
        super().__init__()
        self.proj = nn.Conv2d(1, dim, kernel_size=(4, 16), stride=(4, 16))
        self.norm_layer = nn.LayerNorm(dim)

    def forward(self, mel):
        x = self.proj(mel).permute(0, 2, 3, 1)
        return self.norm_layer(x.reshape(x.shape[0], -1, x.shape[-1]))


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, rope):
        b, n, c = x.shape
        q, k, v = self.qkv(x).reshape(b, n, 3, self.heads, -1).permute(2, 0, 3, 1, 4).unbind(0)
        q, k = apply_rope(q, rope), apply_rope(k, rope)
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=0.)
        return self.proj(x.transpose(1, 2).reshape(b, n, c))


class SwiGLU(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, 8 * dim)
        self.fc2 = nn.Linear(4 * dim, dim)

    def forward(self, x):
        gate, value = self.fc1(x).chunk(2, -1)
        return self.fc2(F.silu(gate) * value)


class Block(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = SwiGLU(dim)

    def forward(self, x, rope):
        x = x + self.attn(self.norm1(x), rope)
        return x + self.mlp(self.norm2(x))


class Encoder(nn.Module):
    def __init__(self, dim: int = 1024, depth: int = 24, heads: int = 16):
        super().__init__()
        self.blocks = nn.ModuleList(Block(dim, heads) for _ in range(depth))
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, rope):
        for block in self.blocks:
            x = block(x, rope)
        return self.norm(x)


class FrozenLargeTeacher(nn.Module):
    def __init__(self, dim: int = 1024, depth: int = 24, heads: int = 16):
        super().__init__()
        self.patch_embed = PatchEmbed(dim)
        self.teacher = Encoder(dim, depth, heads)
        self.dim = dim
        self.head_dim = dim // heads

    @torch.inference_mode()
    def forward(self, mel: torch.Tensor, *, return_tokens: bool = False):
        if mel.ndim != 4 or mel.shape[1] != 1 or mel.shape[3] != 128 or mel.shape[2] < 4:
            raise ValueError(f"Expected [B,1,T>=4,128] normalized mel, got {tuple(mel.shape)}")
        weight = self.patch_embed.proj.weight
        mel = mel.to(device=weight.device, dtype=weight.dtype)
        grid = (mel.shape[2] // 4, 8)
        rope = make_rope(grid, self.head_dim, device=mel.device)
        x = self.teacher(self.patch_embed(mel), rope)
        x = x.reshape(mel.shape[0], grid[0], 8, self.dim)
        return x if return_tokens else x.float().mean(dim=2)


def expected_checkpoint_shapes() -> dict[str, tuple[int, ...]]:
    with torch.device("meta"):
        model = FrozenLargeTeacher()
    return {key: tuple(value.shape) for key, value in model.state_dict().items()}


def load_teacher(path: str | Path, *, dtype=torch.bfloat16, device="cpu") -> FrozenLargeTeacher:
    """Strictly load all294 published patch/EMA tensors, no random fallback.

    Both the original full safetensors and a correctly assembled subset work.
    Only patch_embed.* and teacher.* are read; student/predictor are ignored.
    F32 tensors are converted one at a time to avoid an F32 model allocation.
    The safetensors source remains memory-mapped rather than fully copied.
    """
    from safetensors import safe_open

    with torch.device("meta"):
        model = FrozenLargeTeacher()
    expected = expected_checkpoint_shapes()
    tensors = {}
    with safe_open(str(path), framework="pt", device="cpu") as source:
        keys = {key for key in source.keys() if key.startswith(("patch_embed.", "teacher."))}
        if keys != set(expected):
            raise ValueError(f"Teacher keys mismatch: missing={sorted(set(expected)-keys)}, extra={sorted(keys-set(expected))}")
        for key in sorted(expected):
            shape = tuple(source.get_slice(key).get_shape())
            if shape != expected[key]:
                raise ValueError(f"Shape mismatch for {key}: {shape} != {expected[key]}")
            tensors[key] = source.get_tensor(key).to(dtype=dtype, device=device)
    model.load_state_dict(tensors, strict=True, assign=True)
    model.requires_grad_(False)
    model.eval()
    actual = sum(p.numel() for p in model.parameters())
    if actual != 403_141_632:
        raise RuntimeError(f"Unexpected parameter count {actual}")
    return model
