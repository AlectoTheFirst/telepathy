"""Per-slot translators from a peer's dialect into the local dialect, fitted on public anchors.

Families: linear least squares with bias (ridge), orthogonal Procrustes, small MLP, each fitted
per slot; the same three shared across slots (one map fitted on the pooled slot pairs, N times the
data); and the full-message linear diagnostic (N*d -> N*d) used when every per-slot family fails.
apply_batch is the vectorised form used by the experiments.
"""
from __future__ import annotations

import contextlib
import threading

import numpy as np
import torch

STOCHASTIC = ("mlp", "shared_mlp")   # families whose fit depends on the seed


_FIT_LOCK = threading.Lock()


@contextlib.contextmanager
def pinned_threads(n: int = 1):
    """A fixed torch thread count during a fit, and one fit at a time in the process: the seed and
    the thread count are process-global in torch, and both sides of an in-process handshake fit at
    the same time. With the lock, a fit depends only on its seed and its data."""
    with _FIT_LOCK:
        old = torch.get_num_threads()
        torch.set_num_threads(n)
        try:
            yield
        finally:
            torch.set_num_threads(old)


def _ridge(X, Y, ridge):
    """Least squares with bias; ridge relative to the data scale, bias not penalised."""
    n = X.shape[0]
    Xb = np.concatenate([X, np.ones((n, 1), np.float32)], 1).astype(np.float64)
    G = Xb.T @ Xb
    lam = ridge * max(np.trace(G[:-1, :-1]) / max(1, X.shape[1]), 1e-9)
    pen = lam * np.eye(G.shape[0])
    pen[-1, -1] = 0.0
    return np.linalg.solve(G + pen, Xb.T @ np.asarray(Y, np.float64)).astype(np.float32)


class Translator:
    family = "identity"

    def apply_slot(self, i: int, x: np.ndarray) -> np.ndarray:
        return x

    def apply_message(self, m: np.ndarray) -> np.ndarray:
        return np.stack([self.apply_slot(i, m[i]) for i in range(m.shape[0])])

    def apply_batch(self, x: np.ndarray) -> np.ndarray:
        """x [n, N, d] -> [n, N, d]"""
        return np.stack([self.apply_message(m) for m in x])


def _pool(a):
    n, N, d = a.shape
    return a.reshape(n * N, 1, d)


class LinearSlot(Translator):
    family = "linear"

    def __init__(self, peer, own, ridge: float = 1e-4):
        n, N, d = peer.shape
        self.W = []
        for i in range(N):
            self.W.append(_ridge(peer[:, i], own[:, i], ridge))

    def apply_slot(self, i, x):
        return np.append(x, 1.0).astype(np.float32) @ self.W[i]

    def apply_batch(self, x):
        xb = np.concatenate([x, np.ones(x.shape[:2] + (1,), np.float32)], -1)
        return np.stack([xb[:, i] @ self.W[i] for i in range(x.shape[1])], 1).astype(np.float32)


class ProcrustesSlot(Translator):
    family = "procrustes"

    def __init__(self, peer, own):
        n, N, d = peer.shape
        self.mu_p, self.mu_o, self.R, self.s = [], [], [], []
        for i in range(N):
            P = peer[:, i] - peer[:, i].mean(0)
            O = own[:, i] - own[:, i].mean(0)
            U, S, Vt = np.linalg.svd(P.T @ O)
            R = U @ Vt
            scale = S.sum() / max((P ** 2).sum(), 1e-9)
            self.mu_p.append(peer[:, i].mean(0))
            self.mu_o.append(own[:, i].mean(0))
            self.R.append(R.astype(np.float32))
            self.s.append(np.float32(scale))

    def apply_slot(self, i, x):
        return (self.s[i] * (x - self.mu_p[i]) @ self.R[i] + self.mu_o[i]).astype(np.float32)

    def apply_batch(self, x):
        return np.stack([self.apply_slot(i, x[:, i]) for i in range(x.shape[1])], 1)


class MLPSlot(Translator):
    family = "mlp"

    def __init__(self, peer, own, hidden: int = 64, steps: int = 300, seed: int = 0, lr: float = 3e-3):
        with pinned_threads(1):
            self._fit(peer, own, hidden, steps, seed, lr)

    def _fit(self, peer, own, hidden, steps, seed, lr):
        n, N, d = peer.shape
        torch.manual_seed(seed)
        self.nets = []
        X = torch.from_numpy(peer).float()
        Y = torch.from_numpy(own).float()
        for i in range(N):
            net = torch.nn.Sequential(torch.nn.Linear(d, hidden), torch.nn.GELU(), torch.nn.Linear(hidden, d))
            lin = LinearSlot(peer[:, i: i + 1], own[:, i: i + 1])  # residual around the linear fit
            W = torch.from_numpy(lin.W[0])
            opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
            xi, yi = X[:, i], Y[:, i]
            base = torch.cat([xi, torch.ones(n, 1)], 1) @ W
            for _ in range(steps):
                loss = ((base + net(xi) - yi) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
            net.eval()
            self.nets.append((W, net))

    def apply_slot(self, i, x):
        W, net = self.nets[i]
        with torch.no_grad():
            xt = torch.from_numpy(np.asarray(x, np.float32))
            return (torch.cat([xt, torch.ones(1)]) @ W + net(xt[None])[0]).numpy()

    def apply_batch(self, x):
        out = []
        with torch.no_grad():
            for i, (W, net) in enumerate(self.nets):
                xt = torch.from_numpy(np.ascontiguousarray(x[:, i], np.float32))
                out.append((torch.cat([xt, torch.ones(len(xt), 1)], 1) @ W + net(xt)).numpy())
        return np.stack(out, 1)


class _Shared(Translator):
    """One map for every slot, fitted on the pooled (peer slot i, own slot i) pairs."""
    base = None

    def __init__(self, peer, own, **kw):
        self.inner = self.base(_pool(peer), _pool(own), **kw)

    def apply_slot(self, i, x):
        return self.inner.apply_slot(0, x)

    def apply_batch(self, x):
        n, N, d = x.shape
        return self.inner.apply_batch(x.reshape(n * N, 1, d)).reshape(n, N, d)


class SharedLinear(_Shared):
    family = "shared_linear"
    base = LinearSlot


class SharedProcrustes(_Shared):
    family = "shared_procrustes"
    base = ProcrustesSlot


class SharedMLP(_Shared):
    """The pooled fit has N times the data of a per-slot fit, so it gets more capacity: at 256
    anchors, 64 hidden units underfit, and the per-slot MLP with 256 units overfits (n3_diagnostics)."""
    family = "shared_mlp"
    base = MLPSlot

    def __init__(self, peer, own, hidden: int = 256, steps: int = 1500, **kw):
        super().__init__(peer, own, hidden=hidden, steps=steps, **kw)


class FullMessageLinear(Translator):
    """Diagnostic: one linear map on the concatenated complete message (N*d -> N*d)."""
    family = "full_linear"

    def __init__(self, peer, own, ridge: float = 1e-3):
        n, N, d = peer.shape
        self.N, self.d = N, d
        self.W = _ridge(peer.reshape(n, -1), own.reshape(n, -1), ridge)

    def apply_message(self, m):
        return (np.append(m.reshape(-1), 1.0).astype(np.float32) @ self.W).reshape(self.N, self.d)

    def apply_batch(self, x):
        n = x.shape[0]
        xb = np.concatenate([x.reshape(n, -1), np.ones((n, 1), np.float32)], 1)
        return (xb @ self.W).reshape(n, self.N, self.d).astype(np.float32)

    def apply_slot(self, i, x):
        raise NotImplementedError("full-message translator needs the complete message")


FAMILIES = {"linear": LinearSlot, "procrustes": ProcrustesSlot, "mlp": MLPSlot,
            "shared_linear": SharedLinear, "shared_procrustes": SharedProcrustes, "shared_mlp": SharedMLP,
            "full_linear": FullMessageLinear}


def fit_translator(family: str, peer: np.ndarray, own: np.ndarray, seed: int | None = None, **kw) -> Translator:
    """seed applies to the stochastic families (STOCHASTIC); the others are deterministic."""
    if seed is not None and family in STOCHASTIC:
        kw["seed"] = seed
    return FAMILIES[family](np.asarray(peer, np.float32), np.asarray(own, np.float32), **kw)
