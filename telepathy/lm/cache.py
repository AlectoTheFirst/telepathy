"""Disk cache of a frozen sender's hidden states at one layer, for port training.

The sender is frozen, so its states for a passage never change. Computing them is half of a
training step, so they are computed once per (model, layer) and read back from a float16 memmap.
Layout in cache/t1/<model>_L<layer>/: states.f16 (all passages' [L_i, D] rows, concatenated),
index.json (passage id -> [offset, length]).

Usage: python -m telepathy.lm.cache <repo> <layer> [n_train_passages]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

from . import data as D

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "cache" / "t1"


def cache_dir(short: str, layer: int) -> Path:
    return CACHE / f"{short}_L{layer}"


class StateCache:
    def __init__(self, short: str, layer: int, dev):
        p = cache_dir(short, layer)
        meta = json.loads((p / "index.json").read_text())
        self.D, self.index, self.dev = meta["hidden"], meta["index"], dev
        self.mm = np.memmap(p / "states.f16", dtype=np.float16, mode="r").reshape(-1, self.D)

    def __contains__(self, pid):
        return pid in self.index

    def get(self, pids):
        rows = [self.index[p] for p in pids]
        L = max(n for _, n in rows)
        h = np.zeros((len(pids), L, self.D), np.float16)
        m = np.zeros((len(pids), L), bool)
        for i, (o, n) in enumerate(rows):
            h[i, :n] = self.mm[o: o + n]
            m[i, :n] = True
        return torch.from_numpy(h).to(self.dev).float(), torch.from_numpy(m).to(self.dev)


def build(repo: str, layer: int, passages, batch: int = 8):
    from .models import FrozenLM
    lm = FrozenLM(repo)
    p = cache_dir(lm.short, layer)
    p.mkdir(parents=True, exist_ok=True)
    extra = 0 if lm._wants_bos() else 1   # the sink token that states() prepends
    lens = [len(lm.tok(x["passage"]).input_ids) + extra for x in passages]
    total = sum(lens)
    mm = np.memmap(p / "states.f16", dtype=np.float16, mode="w+", shape=(total, lm.hidden))
    order = sorted(range(len(passages)), key=lambda i: lens[i])   # little padding per batch
    offs, o = {}, 0
    for i, n in enumerate(lens):
        offs[passages[i]["id"]] = [o, n]
        o += n
    for b in range(0, len(order), batch):
        ix = order[b: b + batch]
        h, m = lm.states([passages[i]["passage"] for i in ix], layer)
        h = h.half().cpu().numpy()
        for j, i in enumerate(ix):
            off, n = offs[passages[i]["id"]]
            assert int(m[j].sum()) == n
            mm[off: off + n] = h[j, :n]
        if b % 800 == 0:
            print(f"{lm.short} L{layer}: {b}/{len(order)}", flush=True)
    mm.flush()
    (p / "index.json").write_text(json.dumps({"repo": repo, "layer": layer, "hidden": lm.hidden, "index": offs}))
    print(f"wrote {p} ({total} rows, {total * lm.hidden * 2 / 1e9:.1f} GB)")


if __name__ == "__main__":
    repo, layer = sys.argv[1], int(sys.argv[2])
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 12000
    build(repo, layer, D.load("train")[:n])
