"""Negotiation between port dialects (brief sections 5, 6 and 9 E2).

A sender's dialect is its send port. The receiver negotiates by encoding the public anchor passages
with its own send port, receiving the peer's encodings of the same passages, and fitting a
translator from peer slots to its own. The gate score is the receiver's task accuracy on the
held-out anchors (every question of the 32 held-out anchor passages) read through the translator,
as in Telepathy-0 section 10. The gate is 0.90 (a policy parameter)."""
from __future__ import annotations

import numpy as np
import torch

from . import data as D
from .train import Pair, evaluate, prompt

GATE = 0.90


@torch.no_grad()
def encode(pair: Pair, passages, bits: int = 16, batch: int = 16) -> np.ndarray:
    """The pair's send port on a list of passage dicts -> [n, N, d] float32 (float16 on the wire)."""
    out = []
    for i in range(0, len(passages), batch):
        s = pair.encode([p["passage"] for p in passages[i: i + batch]], bits)
        out.append(s.half().float().cpu().numpy())
    return np.concatenate(out)


def all_question_items(passages):
    return [{"pid": p["id"], "passage": p["passage"], "pidx": i, **q}
            for i, p in enumerate(passages) for q in p["questions"]]


@torch.no_grad()
def read(recv: Pair, slots: np.ndarray, items, pidx, batch: int = 32, trust: float = 1.0):
    """The receiver reads given slots for each item. pidx[i]: the row of `slots` for item i.
    Returns EM accuracy and predictions."""
    dev = recv.r_lm.dev
    ok, preds = [], []
    for i in range(0, len(items), batch):
        chunk = items[i: i + batch]
        s = torch.from_numpy(slots[pidx[i: i + batch]]).to(dev)
        mask = torch.ones(s.shape[:2], dtype=torch.bool, device=dev)
        out = recv.answer(s, mask, [prompt(c["question"]) for c in chunk], torch.full((len(chunk),), trust, device=dev))
        for c, o in zip(chunk, out):
            preds.append(o)
            ok.append(D.exact_match(o, c["answer"]))
    return float(np.mean(ok)), preds


def heldout_score(recv: Pair, translated_held: np.ndarray, held_passages) -> float:
    items = all_question_items(held_passages)
    return read(recv, translated_held, items, [it["pidx"] for it in items])[0]


def test_accuracy(recv: Pair, translated_test: np.ndarray, test_items) -> float:
    return read(recv, translated_test, test_items, list(range(len(test_items))))[0]


__all__ = ["GATE", "encode", "read", "heldout_score", "test_accuracy", "all_question_items", "evaluate"]
