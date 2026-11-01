"""Telepathy-1 ports on frozen language models (brief section 6).

SendPort: N learned slot queries cross-attend over the sender's hidden states at layer L_s and
emit N slots of dimension d, fake-quantized to b bits with a float16 per-slot scale (QAT).
No intent token or embedding touches the slot path.

ReceivePort: a gated cross-attention block applied to the receiver's residual stream after layer
L_r. Slots are memory for this block only and never enter self-attention as tokens. The attention
mask handles missing slots (a learned null slot is always visible), and the gate is conditioned on
the session's effective trust t, assigned by the receiver."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..models.quant import quantize


class XAttn(nn.Module):
    def __init__(self, dq: int, dkv: int, heads: int):
        super().__init__()
        self.h = heads
        self.q = nn.Linear(dq, dq, bias=False)
        self.k = nn.Linear(dkv, dq, bias=False)
        self.v = nn.Linear(dkv, dq, bias=False)
        self.o = nn.Linear(dq, dq, bias=False)

    def forward(self, x, mem, vis, bias=None):
        """vis [B, S] bool; bias: optional additive attention bias [B, L, S]."""
        B, L, D = x.shape
        S = mem.shape[1]
        q = self.q(x).view(B, L, self.h, -1).transpose(1, 2)
        k = self.k(mem).view(B, S, self.h, -1).transpose(1, 2)
        v = self.v(mem).view(B, S, self.h, -1).transpose(1, 2)
        if bias is None:
            am = vis[:, None, None, :]
        else:
            am = bias[:, None].masked_fill(~vis[:, None, None, :], float("-inf"))
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=am)
        return self.o(y.transpose(1, 2).reshape(B, L, D))


class FF(nn.Module):
    def __init__(self, d: int, mult: int = 4):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, mult * d), nn.GELU(), nn.Linear(mult * d, d))

    def forward(self, x):
        return self.net(x)


class SendPort(nn.Module):
    """Position 0 of the sender's sequence is its attention sink (a BOS token, or one that
    FrozenLM.states prepends); its residual norm is ~60x that of other tokens, so it is masked out.

    Cold start: with uniform attention over ~400 tokens every slot is nearly the same average for
    every passage, and the receiver learns to ignore the slots. A learned bias over POS_BINS
    relative-position bins, initialised so that slot i reads its own stretch of the passage, breaks
    that symmetry (Telepathy-0 needed the same, SendPort.pos_bias); training is free to move it. The
    first layer has no residual from the learned queries, so the slots are the attention results."""
    POS_BINS = 16
    POS_INIT = 3.0

    def __init__(self, d_sender: int, N: int = 8, d: int = 64, width: int = 512, depth: int = 2, heads: int = 8):
        super().__init__()
        self.N, self.d = N, d
        bias = torch.zeros(N, self.POS_BINS)
        for i in range(N):
            lo, hi = i * self.POS_BINS // N, max(i * self.POS_BINS // N + 1, (i + 1) * self.POS_BINS // N)
            bias[i, lo:hi] = self.POS_INIT
        self.pos_bias = nn.Parameter(bias)
        self.inp = nn.Sequential(nn.LayerNorm(d_sender), nn.Linear(d_sender, width))
        self.slot_q = nn.Parameter(torch.randn(N, width) * 0.02)
        self.ln_q = nn.ModuleList([nn.LayerNorm(width) for _ in range(depth)])
        self.xa = nn.ModuleList([XAttn(width, width, heads) for _ in range(depth)])
        self.ff = nn.ModuleList([FF(width) for _ in range(depth)])
        self.out = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, d))

    def forward(self, h, mask, bits=16):
        """h [B, L, D_s] float32, mask [B, L] bool -> slots [B, N, d] (fake-quantized)."""
        mask = mask.clone()
        mask[:, 0] = False
        mem = self.inp(h)
        B, L = mask.shape
        n = mask.sum(1, keepdim=True).clamp(min=1)
        pos = (torch.cumsum(mask.long(), 1) - 1).clamp(min=0)
        bins = (pos * self.POS_BINS // n).clamp(max=self.POS_BINS - 1)          # [B, L]
        bias = self.pos_bias[:, bins].permute(1, 0, 2)                           # [B, N, L]
        z = self.slot_q[None].expand(B, -1, -1)
        for j, (ln, xa, ff) in enumerate(zip(self.ln_q, self.xa, self.ff)):
            a = xa(ln(z), mem, mask, bias if j == 0 else None)
            z = a if j == 0 else z + a
            z = z + ff(z)
        if torch.is_tensor(bits) and bits.dim() == 1:
            bits = bits[:, None].expand(-1, self.N)
        return quantize(self.out(z), bits)


class ReceivePort(nn.Module):
    def __init__(self, d_recv: int, N: int = 8, d: int = 64, heads: int = 8):
        super().__init__()
        self.N = N
        self.inp = nn.Sequential(nn.Linear(d, d_recv), nn.GELU(), nn.Linear(d_recv, d_recv))
        self.pos = nn.Parameter(torch.randn(N + 1, d_recv) * 0.02)
        self.null = nn.Parameter(torch.zeros(d_recv))
        self.ln = nn.LayerNorm(d_recv)
        self.xa = XAttn(d_recv, d_recv, heads)
        self.ff = FF(d_recv, mult=2)
        self.trust = nn.Sequential(nn.Linear(1, 32), nn.GELU(), nn.Linear(32, 32))
        self.gate = nn.Linear(d_recv + 32, 1)
        # The output projections start at zero, so the port is the identity at initialisation and
        # the frozen receiver is unchanged; tanh(alpha) starts open so that gradients reach them.
        self.alpha = nn.Parameter(torch.ones(2))
        nn.init.zeros_(self.xa.o.weight)
        nn.init.zeros_(self.ff.net[-1].weight)
        nn.init.zeros_(self.ff.net[-1].bias)
        self.last_gate = None

    def memory(self, slots, mask):
        B, n, _ = slots.shape
        m = self.inp(slots) + self.pos[1: n + 1][None]
        m = torch.cat([(self.null + self.pos[0]).expand(B, 1, -1), m], 1)
        vis = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=slots.device), mask.bool()], 1)
        return m, vis

    def forward(self, x, mem, vis, trust):
        """x [B, L, D_r] in the receiver's dtype; mem/vis from memory(); trust [B]."""
        dt = x.dtype
        xf = x.float()
        B, L, _ = xf.shape
        hx = self.ln(xf)
        tf = self.trust(trust.float().view(B, 1, 1)).expand(B, L, -1)
        g = torch.sigmoid(self.gate(torch.cat([hx, tf], -1)))
        self.last_gate = g.detach()
        # scale by the token's residual RMS so that the port can move the stream on any model
        # (Gemma's residual stream is an order of magnitude larger than Qwen's)
        r = xf.pow(2).mean(-1, keepdim=True).sqrt().detach()
        a = self.xa(hx, mem, vis)
        y = xf + torch.tanh(self.alpha[0]) * g * r * a
        y = y + torch.tanh(self.alpha[1]) * g * r * self.ff(a)
        return y.to(dt)
