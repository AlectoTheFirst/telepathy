"""Agent = tiny decoder-only transformer + send port + receive port + answer head."""
from __future__ import annotations

import os
from dataclasses import asdict

import torch
import torch.nn as nn

from ..config import CKPT_DIR, ModelConfig, TaskConfig
from ..data import TaskData
from .ports import ReceivePort, SendPort
from .transformer import SelfBlock


# Gradient from the send port into the shared trunk is scaled down: early in joint
# training the receiver is untrained, so this gradient is mostly noise, and Adam turns
# noise into full-size steps that destroy the key-value binding in the trunk.
SEND_TRUNK_GRAD = 0.1
SEND_MEMORY = os.environ.get("TELEPATHY_SEND_MEMORY", "both")


def grad_scale(x, s: float):
    return x * s + x.detach() * (1 - s)


class Agent(nn.Module):
    def __init__(self, td: TaskData, mc: ModelConfig, ports: bool = True):
        super().__init__()
        self.td, self.mc = td, mc
        dm = mc.d_model
        self.emb = nn.Embedding(td.vocab.size, dm)
        self.pos = nn.Embedding(mc.max_len, dm)
        self.blocks = nn.ModuleList([SelfBlock(dm, mc.n_heads) for _ in range(mc.n_layers)])
        self.ln_f = nn.LayerNorm(dm)
        self.head = nn.Linear(dm, td.ans.size)
        self.ports = ports
        if ports:
            self.send = SendPort(mc)
            self.recv = ReceivePort(mc)

    def hidden(self, tokens, slots=None, mask=None, trust=None, return_input=False):
        L = tokens.shape[1]
        e = self.emb(tokens)
        if tokens.dim() == 3:  # multi-channel positions (sender table entries)
            e = e.sum(2)
        x = e + self.pos.weight[:L][None]
        x0 = x
        gate = None
        for i, blk in enumerate(self.blocks):
            if i == self.mc.recv_layer and slots is not None:
                x, gate = self.recv(x, slots, mask, trust)
            x = blk(x)
        if self.mc.recv_layer >= len(self.blocks) and slots is not None:
            x, gate = self.recv(x, slots, mask, trust)
        if return_input:
            return self.ln_f(x), gate, x0
        return self.ln_f(x), gate

    def encode(self, a_tokens, intent):
        """Send port: table (+ purpose) -> [B, N, d] latent message (unquantized).

        The port reads two levels of A's internal state: the input-level states and
        the final hidden states. Self-attention in A mixes entries; the input level
        keeps each key bound to its value, which makes the joint training converge
        within the time budget (see RESULTS.md, training notes)."""
        h, _, x0 = self.hidden(a_tokens, return_input=True)
        visible = a_tokens[..., 0] != self.td.vocab.PAD
        if SEND_MEMORY == "input":
            return self.send(torch.cat([x0, x0], 1), torch.cat([visible, visible], 1), intent)
        h = grad_scale(h, SEND_TRUNK_GRAD)
        return self.send(torch.cat([x0, h], 1), torch.cat([visible, visible], 1), intent)

    def answer(self, b_tokens, slots=None, mask=None, trust=None):
        """Receive port + answer head. Returns (logits, gate at the answer position)."""
        if slots is not None and trust is None:
            trust = torch.ones(b_tokens.shape[0])
        h, gate = self.hidden(b_tokens, slots, mask, trust)
        return self.head(h[:, -1]), gate


# ------------------------------------------------------------------ checkpoints
def save_ckpt(path, models: dict, meta: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"state": {k: m.state_dict() for k, m in models.items()}, "meta": meta}, path)


def load_ckpt(name_or_path: str):
    path = name_or_path if os.path.sep in name_or_path else os.path.join(CKPT_DIR, name_or_path + ".pt")
    blob = torch.load(path, map_location="cpu", weights_only=False)
    meta = blob["meta"]
    td = TaskData(TaskConfig(**meta["task"]))
    mc = ModelConfig(**meta["model"])
    models = {}
    for k, sd in blob["state"].items():
        m = Agent(td, mc, ports=meta.get("ports", True))
        m.load_state_dict(sd)
        m.eval()
        models[k] = m
    return models, meta, td


def ckpt_meta(td: TaskData, mc: ModelConfig, seed: int, kind: str, dialect: str, ports: bool, **extra):
    return {"task": asdict(td.task), "model": asdict(mc), "seed": seed, "kind": kind,
            "dialect_id": dialect, "ports": ports, **extra}
