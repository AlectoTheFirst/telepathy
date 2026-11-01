"""Anchor codec: a model's own send port encodes the public anchor tables; its own receive port
reads (possibly translated) peer anchor latents and is scored against the known answers."""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch

from ..config import INTENT_SHARE_VALUE
from ..models.agent import load_ckpt
from ..tlp.session import Capabilities


class ModelCodec:
    def __init__(self, sender, receiver, td, N: int, d: int):
        self.sender, self.receiver, self.td = sender, receiver, td
        self.N, self.d = N, d
        self._enc = {}

    def batch(self, which: str, n: int):
        return self.td.anchors(which, n)

    @torch.no_grad()
    def encode(self, which: str, n: int) -> np.ndarray:
        key = (which, n)
        if key not in self._enc:
            b = self.batch(which, n)
            slots = self.sender.encode(b.a_tokens, b.intent)
            self._enc[key] = slots.half().float().numpy()  # float16 on the wire
        return self._enc[key]

    @torch.no_grad()
    def read(self, latents: np.ndarray, which: str, trust: float = 1.0) -> np.ndarray:
        n = latents.shape[0]
        b = self.batch(which, n)
        slots = torch.from_numpy(np.asarray(latents, np.float32))
        mask = torch.ones(n, slots.shape[1], dtype=torch.bool)
        logits, _ = self.receiver.answer(b.b_tokens, slots, mask, torch.full((n,), trust))
        return logits.argmax(-1).numpy()

    def score(self, latents: np.ndarray, which: str) -> float:
        b = self.batch(which, latents.shape[0])
        return float((self.read(latents, which) == b.answer.numpy()).mean())


def model_capabilities(ckpt: str, side: str = "A", bits: int = 16, N: int | None = None) -> Capabilities:
    """Capabilities + codec for one model of a trained pair. Each model uses its own send port
    and its own receive port (symmetric training gives both)."""
    models, meta, td = _load(ckpt)
    m = models[side]
    mc = meta["model"]
    codec = ModelCodec(m, m, td, N or mc["N"], mc["d"])
    return Capabilities(N=N or mc["N"], d=mc["d"], bits=bits, model_id=meta["model_ids"][side],
                        dialect_id=meta["dialect_id"], codec=codec)


@lru_cache(maxsize=16)
def _load(ckpt: str):
    return load_ckpt(ckpt)
