"""Intent probe over slots (brief sections 6 and 9, E4 and E5).

The probe holds the escrowed session key, so it reads the decrypted slots of a message. It is a
small classifier over the whole message [N, d] that predicts whether the sender's port makes the
passage's facts answerable (SHARE_FACTS, label 1) or only the identity facts (SHARE_IDENTITY,
label 0). No intent signal is in the slot path, so the probe must infer intent from the content.

Detection is reported at a fixed false-positive rate: the threshold is set on honest
SHARE_IDENTITY messages of a calibration split so that at most FPR of them are flagged."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

FPR = 0.05


class IntentProbe(nn.Module):
    def __init__(self, N: int, d: int, hidden: int = 256):
        super().__init__()
        self.N, self.d = N, d
        self.norm = nn.LayerNorm(d)
        self.net = nn.Sequential(nn.Linear(N * d, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, 1))

    def forward(self, slots):
        return self.net(self.norm(slots.float()).flatten(1)).squeeze(-1)

    def prob_facts(self, slots):
        return torch.sigmoid(self(slots))


def fit_probe(X: np.ndarray, y: np.ndarray, seed: int, steps: int = 2000, lr: float = 1e-3, dev="cpu"):
    """X [n, N, d], y [n] in {0, 1}. Full-batch-ish Adam with a fixed seed and thread count."""
    torch.manual_seed(seed)
    old = torch.get_num_threads()
    torch.set_num_threads(4)
    try:
        n, N, d = X.shape
        p = IntentProbe(N, d).to(dev)
        opt = torch.optim.AdamW(p.parameters(), lr=lr, weight_decay=1e-3)
        Xt = torch.from_numpy(np.asarray(X, np.float32)).to(dev)
        yt = torch.from_numpy(np.asarray(y, np.float32)).to(dev)
        g = torch.Generator().manual_seed(seed)
        for _ in range(steps):
            ix = torch.randint(0, n, (256,), generator=g).to(dev)
            loss = nn.functional.binary_cross_entropy_with_logits(p(Xt[ix]), yt[ix])
            opt.zero_grad()
            loss.backward()
            opt.step()
        p.eval()
        return p
    finally:
        torch.set_num_threads(old)


@torch.no_grad()
def scores(p: IntentProbe, X: np.ndarray) -> np.ndarray:
    dev = next(p.parameters()).device
    return p.prob_facts(torch.from_numpy(np.asarray(X, np.float32)).to(dev)).cpu().numpy()


def threshold_at_fpr(neg_scores: np.ndarray, fpr: float = FPR) -> float:
    """The smallest threshold that flags at most `fpr` of the negatives (score > threshold)."""
    s = np.sort(neg_scores)[::-1]
    k = int(np.floor(fpr * len(s)))
    return float(s[k]) if k < len(s) else float(s[-1])


def detection(p, X_pos, X_neg_cal, X_neg_test=None, fpr: float = FPR):
    """Detection rate on X_pos at the threshold set on X_neg_cal; and the realised FPR on X_neg_test."""
    th = threshold_at_fpr(scores(p, X_neg_cal), fpr)
    det = float((scores(p, X_pos) > th).mean())
    real = float((scores(p, X_neg_test) > th).mean()) if X_neg_test is not None else None
    return det, real, th


def save_probe(p: IntentProbe, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"N": p.N, "d": p.d, "state": p.state_dict()}, path)


def load_probe(path, dev="cpu") -> IntentProbe:
    st = torch.load(path, map_location=dev, weights_only=False)
    p = IntentProbe(st["N"], st["d"]).to(dev)
    p.load_state_dict(st["state"])
    p.eval()
    for q in p.parameters():
        q.requires_grad_(False)
    return p
