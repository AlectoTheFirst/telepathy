#!/usr/bin/env python
"""Telepathy-1 port training (brief section 6).

  python scripts/train_t1.py sweep        (optional, not run) layer sweep on Qwen -> Qwen: L_s in {2/3, last} x L_r in {1/3, 1/2},
                                          20 min each on a 2000-passage state cache; writes results/t1_layer_sweep.json
  python scripts/train_t1.py cache <repo> build the 12000-passage state cache at the chosen sender layer
  python scripts/train_t1.py pairs        Qwen seed 1, Qwen seed 2 (control), Gemma -> Gemma; stage 1, <= 170 min each
  python scripts/train_t1.py one <name>   train or resume one named run of PAIRS

Every run resumes from its checkpoint in checkpoints/t1/<name>/.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from telepathy.lm.models import GEMMA, QWEN  # noqa: E402
from telepathy.lm.train import CKPT, TrainConfig, train  # noqa: E402

RESULTS = ROOT / "results"
SWEEP = [(19, 9), (19, 14), (28, 9), (28, 14)]   # Qwen3-0.6B: 28 layers


def chosen_layers():
    """Relative depths (sender fraction, receiver fraction). The layer sweep is not run: in 20-minute
    runs no port carried passage content yet (true-slot and swapped-slot losses equal), so a short
    sweep cannot rank the layers. The defaults are two-thirds depth for the sender and one-half depth
    for the receiver; a sweep result, if one exists, overrides them."""
    f = RESULTS / "t1_layer_sweep.json"
    if f.exists():
        r = json.loads(f.read_text())
        return r["chosen"]["s_frac"], r["chosen"]["r_frac"]
    return 2 / 3, 1 / 2


def layers_for(n_send, n_recv):
    sf, rf = chosen_layers()
    return round(sf * n_send), round(rf * n_recv)


def pairs():
    q_s, q_r = layers_for(28, 28)
    g_s, g_r = layers_for(26, 26)
    return {
        "qwen_s1": TrainConfig("qwen_s1", QWEN, QWEN, q_s, q_r, seed=1),
        "qwen_s2": TrainConfig("qwen_s2", QWEN, QWEN, q_s, q_r, seed=2),
        "gemma_s1": TrainConfig("gemma_s1", GEMMA, GEMMA, g_s, g_r, seed=1),
    }


def log(s):
    print(time.strftime("%H:%M:%S"), s, flush=True)


def cmd_sweep():
    out, lms = [], {}
    for ls, lr in SWEEP:
        name = f"sweep_Ls{ls}_Lr{lr}"
        cfg = TrainConfig(name, QWEN, QWEN, ls, lr, max_minutes=20, eval_every=10 ** 9, seed=1)
        _, s = train(cfg, lms, log)
        out.append({"L_s": ls, "L_r": lr, "val_acc_200": s["val_acc_200"], "steps": s["steps"],
                    "minutes": s["train_minutes"]})
    best = max(out, key=lambda r: r["val_acc_200"])
    res = {"model": QWEN, "budget_min": 20, "passages_cached": 2000, "runs": out,
           "chosen": {"L_s": best["L_s"], "L_r": best["L_r"], "s_frac": best["L_s"] / 28, "r_frac": best["L_r"] / 28},
           "note": "20-minute runs on a 2000-passage state cache; the chosen relative depths apply to every model."}
    (RESULTS / "t1_layer_sweep.json").write_text(json.dumps(res, indent=2))
    log(json.dumps(res, indent=2))


def cmd_cache(repo):
    from telepathy.lm import data as D
    from telepathy.lm.cache import build
    n = {QWEN: 28, GEMMA: 26}[repo]
    ls, _ = layers_for(n, n)
    build(repo, ls, D.load("train")[:12000])


def cmd_one(name):
    cfg = pairs()[name]
    _, s = train(cfg, None, log)
    return s


def cmd_pairs():
    for name in pairs():
        if (CKPT / name / "summary.json").exists():
            log(f"{name}: done, skipped")
            continue
        cmd_one(name)


if __name__ == "__main__":
    cmd = sys.argv[1]
    {"sweep": cmd_sweep, "pairs": cmd_pairs}.get(cmd, lambda: None)() if cmd in ("sweep", "pairs") else None
    if cmd == "cache":
        cmd_cache(sys.argv[2])
    elif cmd == "one":
        cmd_one(sys.argv[2])
