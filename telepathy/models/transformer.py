"""Tiny pre-LN decoder-only transformer pieces."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, dm: int, mult: int = 4):
        super().__init__()
        self.fc = nn.Linear(dm, mult * dm)
        self.out = nn.Linear(mult * dm, dm)

    def forward(self, x):
        return self.out(F.gelu(self.fc(x)))


class SelfBlock(nn.Module):
    """Causal self-attention block."""

    def __init__(self, dm: int, nh: int):
        super().__init__()
        self.nh = nh
        self.ln1 = nn.LayerNorm(dm)
        self.qkv = nn.Linear(dm, 3 * dm)
        self.proj = nn.Linear(dm, dm)
        self.ln2 = nn.LayerNorm(dm)
        self.mlp = MLP(dm)

    def forward(self, x):
        B, L, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(D, -1)
        q, k, v = (t.view(B, L, self.nh, D // self.nh).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(y.transpose(1, 2).reshape(B, L, D))
        return x + self.mlp(self.ln2(x))


class CrossAttn(nn.Module):
    """Queries from x attend over memory m. mask: [B, M] True = visible."""

    def __init__(self, dm: int, nh: int, dm_mem: int | None = None):
        super().__init__()
        self.nh = nh
        dm_mem = dm_mem or dm
        self.ln_q = nn.LayerNorm(dm)
        self.ln_m = nn.LayerNorm(dm_mem)
        self.q = nn.Linear(dm, dm)
        self.kv = nn.Linear(dm_mem, 2 * dm)
        self.proj = nn.Linear(dm, dm)

    def forward(self, x, m, mask=None, bias=None):
        """bias: optional additive attention bias [L, M] (float)."""
        B, L, D = x.shape
        M = m.shape[1]
        q = self.q(self.ln_q(x)).view(B, L, self.nh, D // self.nh).transpose(1, 2)
        k, v = self.kv(self.ln_m(m)).split(D, -1)
        k = k.view(B, M, self.nh, D // self.nh).transpose(1, 2)
        v = v.view(B, M, self.nh, D // self.nh).transpose(1, 2)
        am = None if mask is None else mask[:, None, None, :]
        if bias is not None:
            neg = torch.zeros(B, 1, 1, M) if mask is None else (~mask[:, None, None, :]).float() * -1e9
            am = bias[None, None] + neg
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=am)
        return self.proj(y.transpose(1, 2).reshape(B, L, D))
