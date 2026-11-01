"""6.6 Monitor tap: a probe (small MLP) trained to decode the table from captured DATA payloads.

Keyed DATA is AES-256-GCM per packet (crypto/aead.py). Two conditions:
  (a) with the key: the tap holds the session key (as the escrow probe does) and decrypts;
  (b) without the key: the tap sees ciphertext, one fresh session key per message, and reads the
      ciphertext bytes as if they were slot values.
"""
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..crypto.aead import TAG_LEN, PacketAEAD
from ..models.quant import decode_slot, encode_slot, quantize
from ..tlp.header import Flag, MsgType, Packet
from .common import CAT, DEFAULT, figure, models, save, savefig


class TableProbe(nn.Module):
    """Decodes the table entry carried by each slot: slot j (plus its index) -> key and value of
    entry j. Shared across slots; the slot index lets it learn per-slot codes."""

    def __init__(self, N, d, keys, values, hidden=256):
        super().__init__()
        self.N = N
        self.net = nn.Sequential(nn.Linear(d + N, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU())
        self.k = nn.Linear(hidden, keys)
        self.v = nn.Linear(hidden, values)

    def forward(self, x):
        n, N, d = x.shape
        h = self.net(torch.cat([x, torch.eye(N).expand(n, N, N)], -1))
        return self.k(h), self.v(h)


def targets(batch, N):
    """Per-slot targets: key and value of table entry j (A's input order), for j < N."""
    return batch.keys[:, :N].clone(), batch.vals[:, :N].clone()


def value_acc(logits, t):
    """Fraction of table entries recovered exactly (key and value)."""
    (lk, lv), (tk, tv) = logits, t
    present = tk >= 0
    ok = (lk.argmax(-1) == tk) & (lv.argmax(-1) == tv) & present
    return float(ok.sum() / present.sum())


def train_probe(X, T, Xt, Tt, keys, values, steps=3000, seed=0):
    torch.manual_seed(seed)
    n, N, d = X.shape
    p = TableProbe(N, d, keys, values)
    opt = torch.optim.Adam(p.parameters(), lr=2e-3)
    g = torch.Generator().manual_seed(seed)
    tk, tv = T
    for step in range(steps):
        idx = torch.randint(0, n, (256,), generator=g)
        lk, lv = p(X[idx])
        loss = F.cross_entropy(lk.flatten(0, 1), tk[idx].flatten(), ignore_index=-1) + \
            F.cross_entropy(lv.flatten(0, 1), tv[idx].flatten(), ignore_index=-1)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        return value_acc(p(Xt), Tt), p


def seal_messages(X: torch.Tensor, seed: int):
    """Encrypt every slot of every message as a keyed DATA packet; one fresh key per message.
    Returns (ciphertexts read as slots [n,N,d], decrypted slots [n,N,d])."""
    rng = np.random.default_rng(seed)
    n, N, d = X.shape
    seen = np.zeros((n, N, d), np.float32)
    back = np.zeros((n, N, d), np.float32)
    x = X.numpy()
    for m in range(n):
        a = PacketAEAD(rng.bytes(32), rng.bytes(4))
        for i in range(N):
            p = Packet(MsgType.DATA, 7, 107, m, i + 1, Flag.KEYED | (Flag.PROTECTED if i < N // 2 else 0), 1, 1, i,
                       N, i, d, encode_slot(x[m, i], 16))
            p.payload = a.seal(p, b"R")
            body = p.payload[: len(p.payload) - TAG_LEN]
            seen[m, i] = np.nan_to_num(decode_slot(body, d, 16), nan=0.0, posinf=0.0, neginf=0.0).clip(-1e4, 1e4)
            back[m, i] = decode_slot(a.open(p, b"R"), d, 16)
    return torch.from_numpy(seen), torch.from_numpy(back)


def main(n_train=20000, n_test=2000):
    m, meta, td = models(DEFAULT)
    A = m["A"]
    N, d = meta["model"]["N"], meta["model"]["d"]
    K, V = td.task.key_range, td.task.value_range

    def capture(n, seed):
        b = td.fixed_set(seed, n)
        with torch.no_grad():
            x = quantize(A.encode(b.a_tokens, b.intent), 16)
        return b, x

    btr, Xtr = capture(n_train, 501)
    bte, Xte = capture(n_test, 502)
    Ttr, Tte = targets(btr, N), targets(bte, N)
    Ctr, Ptr = seal_messages(Xtr, 11)
    Cte, Pte = seal_messages(Xte, 12)
    # the ciphertext as the tap reads it: scale per feature so the MLP sees comparable ranges
    mu, sd = Ctr.mean((0, 1)), Ctr.std((0, 1)) + 1e-6
    res = {}
    res["with_key"], _ = train_probe(Ptr, Ttr, Pte, Tte, K, V)
    res["without_key_ciphertext"], _ = train_probe((Ctr - mu) / sd, Ttr, (Cte - mu) / sd, Tte, K, V)
    res["decrypt_matches_plaintext"] = bool(torch.allclose(Pte, Xte.half().float(), atol=1e-3))
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items()})
    chance = 1 / (V * K)
    save("e6_monitor_tap", {"probe_entry_acc": res, "chance": chance, "n_train": n_train, "n_test": n_test,
                            "metric": "fraction of table entries whose key and value the tap probe recovers",
                            "cipher": "AES-256-GCM per DATA packet, fresh key per message (session)"})
    fig, ax = figure(5.2, 3.6)
    names = ["(a) with the key\n(escrowed probe)", "(b) without the key\n(ciphertext)"]
    vals = [res["with_key"], res["without_key_ciphertext"]]
    ax.bar(range(2), vals, color=[CAT[1], CAT[0]], width=0.5)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.4f}", ha="center", fontsize=8)
    ax.axhline(chance, color="#8a8a85", ls=":", lw=1)
    ax.set_xticks(range(2), names, fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("tap probe: table entries recovered")
    ax.set_title(f"Monitor tap on DATA payloads (d={d}, N={N}); dotted: chance", fontsize=9)
    savefig(fig, "e6_monitor_tap")


if __name__ == "__main__":
    main()
