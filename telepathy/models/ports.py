"""Communication ports: SendPort compresses hidden states into N latent slots;
ReceivePort reads slots through a gated cross-attention pathway conditioned on trust."""
from __future__ import annotations

import torch
import torch.nn as nn

from ..config import ModelConfig
from .transformer import MLP, CrossAttn


class SendPort(nn.Module):
    """N learned slot queries (plus a purpose embedding) attend over the sender's final states."""

    def __init__(self, mc: ModelConfig, n_purpose: int = 5, depth: int = 2):
        super().__init__()
        dm = mc.d_model
        self.slot_q = nn.Parameter(torch.randn(mc.N, dm) * 0.02)
        self.purpose = nn.Embedding(n_purpose, dm)
        self.xa = nn.ModuleList([CrossAttn(dm, mc.n_heads) for _ in range(depth)])
        self.ln = nn.ModuleList([nn.LayerNorm(dm) for _ in range(depth)])
        self.mlp = nn.ModuleList([MLP(dm) for _ in range(depth)])
        self.ln_out = nn.LayerNorm(dm)
        self.out = nn.Linear(dm, mc.d)
        # Learnable positional attention bias, initialised so that slot i starts out
        # looking at table positions i+1, i+1+N, ... This only breaks the symmetry of
        # the cold start; training is free to move it.
        bias = torch.zeros(mc.N, mc.max_len)
        for i in range(mc.N):
            bias[i, 1 + i:: mc.N] = mc.pos_bias_init
        self.pos_bias = nn.Parameter(bias)

    def forward(self, h, visible, purpose):
        z = self.slot_q[None] + self.purpose(purpose)[:, None]
        L = h.shape[1] // 2  # memory = [input-level states ; final states]
        b = self.pos_bias[:, :L]
        bias = torch.cat([b, torch.zeros_like(b)], 1)
        for xa, ln, mlp in zip(self.xa, self.ln, self.mlp):
            z = z + xa(z, h, visible, bias)
            z = z + mlp(ln(z))
        return self.out(self.ln_out(z))  # [B, N, d]


class ReceivePort(nn.Module):
    """Gated cross-attention from the receiver's token stream into the slots.

    Provenance separation: slots are memory for this layer only; they never enter
    self-attention as tokens. A learned null slot is always visible, so a message with
    no slots is well defined. The gate is a function of the token state and the
    receiver-assigned trust t in [0, 1]."""

    def __init__(self, mc: ModelConfig):
        super().__init__()
        dm = mc.d_model
        self.N = mc.N
        self.inp = nn.Linear(mc.d, dm)
        self.pos = nn.Embedding(mc.N + 1, dm)
        self.null = nn.Parameter(torch.zeros(dm))
        self.xa = CrossAttn(dm, mc.n_heads)
        self.trust = nn.Sequential(nn.Linear(1, 16), nn.GELU(), nn.Linear(16, 16))
        self.ln = nn.LayerNorm(dm)
        self.gate = nn.Linear(dm + 16, 1)

    def forward(self, x, slots, mask, trust):
        B, L, _ = x.shape
        n = slots.shape[1]
        m = self.inp(slots) + self.pos.weight[1: n + 1][None]
        null = (self.null + self.pos.weight[0]).expand(B, 1, -1)
        m = torch.cat([null, m], 1)
        vis = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=x.device), mask.bool()], 1)
        tf = self.trust(trust.float().view(B, 1))
        g = torch.sigmoid(self.gate(torch.cat([self.ln(x), tf[:, None].expand(B, L, -1)], -1)))
        x = x + g * self.xa(x, m, vis)
        return x, g[:, -1, 0]
