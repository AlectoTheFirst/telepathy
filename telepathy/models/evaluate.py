"""Evaluation helpers shared by the experiments."""
from __future__ import annotations

import torch

from ..data import Batch, TaskData
from .quant import quantize


@torch.no_grad()
def send(sender, batch: Batch, bits: int = 16, n_prefix: int | None = None, intent=None):
    """Encode, quantize and prefix-mask. Returns (slots [B,N,d], mask [B,N])."""
    it = batch.intent if intent is None else intent
    slots = sender.encode(batch.a_tokens, it)
    slots = quantize(slots, bits)
    N = slots.shape[1]
    p = N if n_prefix is None else n_prefix
    mask = (torch.arange(N)[None] < p).expand(len(batch), N).clone()
    return slots * mask[..., None], mask


@torch.no_grad()
def receive(receiver, batch: Batch, slots, mask, trust=1.0):
    B = len(batch)
    t = trust if torch.is_tensor(trust) else torch.full((B,), float(trust))
    logits, gate = receiver.answer(batch.b_tokens, slots, mask, t)
    pred = logits.argmax(-1)
    return {"acc": (pred == batch.answer).float().mean().item(), "pred": pred,
            "gate": None if gate is None else gate.mean().item(), "logits": logits}


@torch.no_grad()
def latent_acc(sender, receiver, batch: Batch, bits=16, n_prefix=None, trust=1.0, noise=0.0, chunk=1000):
    accs, gates, n = 0.0, 0.0, 0
    for i in range(0, len(batch), chunk):
        b = batch.index(slice(i, i + chunk))
        slots, mask = send(sender, b, bits, n_prefix)
        if noise:
            slots = slots + noise * torch.randn_like(slots) * mask[..., None]
        r = receive(receiver, b, slots, mask, trust)
        accs += r["acc"] * len(b)
        gates += (r["gate"] or 0.0) * len(b)
        n += len(b)
    return accs / n, gates / n


@torch.no_grad()
def baseline_acc(model, td: TaskData, batch: Batch, kind: str, budget=None, chunk=1000):
    from .train import baseline_tokens
    acc, n = 0.0, 0
    for i in range(0, len(batch), chunk):
        b = batch.index(slice(i, i + chunk))
        logits, _ = model.answer(baseline_tokens(td, b, kind, budget))
        acc += (logits.argmax(-1) == b.answer).float().sum().item()
        n += len(b)
    return acc / n
