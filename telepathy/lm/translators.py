"""Translators for Telepathy-1 negotiation (brief section 6), in the order they are tried:

1. relative   relative representations: each side expresses slot i of a message as its cosine
              similarities to its own encodings (slot i) of the K anchors. The receiver reads the
              N x K similarity vector through a decoder that maps relative coordinates back into
              its own slot space. The decoder is fitted by the receiver alone, on its own messages
              (it needs no paired data), so the only thing two peers share is the anchor set.
              This is also the hub-mode coordinate system (SPEC.md, hub mode).
2. procrustes orthogonal Procrustes with scale, shared across slots (Telepathy-0 shared_procrustes).
3. shared_mlp linear map plus residual MLP shared across slots (Telepathy-0 shared_mlp).

Per-slot families are not used: they failed at every width in Telepathy-0 (RESULTS.md 10.3).
All arrays are [n, N, d] float32."""
from __future__ import annotations

import numpy as np

from ..negotiate.translators import SharedMLP, SharedProcrustes, Translator, _ridge, pinned_threads

FAMILIES = ("relative", "procrustes", "shared_mlp")


def _unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def rel_coords(x: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    """x [n, N, d], anchors [K, N, d] (the same side's encodings) -> [n, N*K] cosine similarities,
    slot i against anchor slot i."""
    xu, au = _unit(x), _unit(anchors)
    return np.einsum("nid,kid->nik", xu, au).reshape(len(x), -1).astype(np.float32)


class RelativeDecoder:
    """Receiver-local map from relative coordinates (N*K) to own slots (N*d): ridge regression
    fitted on the receiver's own messages, ridge strength picked on a held-out part of them."""

    def __init__(self, own_anchors: np.ndarray, own_msgs: np.ndarray, ridges=(1e-3, 1e-2, 1e-1, 1.0)):
        self.anchors = np.asarray(own_anchors, np.float32)
        n, N, d = own_msgs.shape
        self.N, self.d = N, d
        X = rel_coords(own_msgs, self.anchors)
        Y = own_msgs.reshape(n, -1)
        cut = int(0.8 * n)
        best = None
        for r in ridges:
            W = _ridge(X[:cut], Y[:cut], r)
            err = float(((self._apply(W, X[cut:]) - Y[cut:]) ** 2).mean())
            if best is None or err < best[0]:
                best = (err, r)
        self.ridge = best[1]
        self.W = _ridge(X, Y, self.ridge)

    @staticmethod
    def _apply(W, X):
        return np.concatenate([X, np.ones((len(X), 1), np.float32)], 1) @ W

    def decode(self, rel: np.ndarray) -> np.ndarray:
        return self._apply(self.W, rel).reshape(len(rel), self.N, self.d).astype(np.float32)


class Relative(Translator):
    """peer slots -> peer relative coordinates (peer's own anchors) -> receiver decoder."""
    family = "relative"

    def __init__(self, peer_anchors, decoder: RelativeDecoder):
        self.peer_anchors = np.asarray(peer_anchors, np.float32)
        self.dec = decoder

    def apply_batch(self, x):
        return self.dec.decode(rel_coords(np.asarray(x, np.float32), self.peer_anchors))

    def apply_slot(self, i, x):
        raise NotImplementedError("relative translation needs the complete message")


def fit(family: str, peer_anchors, own_anchors, seed: int = 0, own_msgs=None) -> Translator:
    peer_anchors = np.asarray(peer_anchors, np.float32)
    own_anchors = np.asarray(own_anchors, np.float32)
    if family == "relative":
        return Relative(peer_anchors, RelativeDecoder(own_anchors, own_msgs))
    if family == "procrustes":
        return SharedProcrustes(peer_anchors, own_anchors)
    if family == "shared_mlp":
        return SharedMLP(peer_anchors, own_anchors, seed=seed)
    raise ValueError(family)


def heldout_cosine(tr: Translator, peer_held, own_held) -> float:
    """Diagnostic only: mean cosine between translated peer slots and the receiver's own slots on
    the held-out anchors. The gate score is task accuracy on the held-out anchors (negotiate.py)."""
    a = _unit(tr.apply_batch(np.asarray(peer_held, np.float32)))
    b = _unit(np.asarray(own_held, np.float32))
    return float((a * b).sum(-1).mean())
