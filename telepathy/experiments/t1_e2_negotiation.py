"""Telepathy-1 E2: negotiation between strangers (G1).

Gemma 3 1B sends through its own send port (gemma_s1); Qwen3-0.6B answers through the receive
port of qwen_s1. Translators from section 6, in order: relative, procrustes, shared_mlp, fitted on
K in {16, 32, 64, 128, 256} anchors. Five fits per cell: fit f uses a seeded random K-subset of
the 256 fitting anchors (all 256 at K = 256) and seed f for the stochastic families; the relative
decoder is fitted on a seeded 3000-passage sample of the receiver's own messages.

Ceiling: qwen_s1 -> qwen_s1. Floor: gemma_s1 slots forced into qwen_s1 without a translator.
Control: qwen_s2 -> qwen_s1 (same architecture, different port seed).
Hub mode: relative coordinates over the whole public anchor set (256) with no fitting at connect
time; the receiver's decoder is fixed in advance.

Reported per cell: held-out score (task accuracy on every question of the 32 held-out anchors),
test accuracy on the fixed 1000-item test set (uniform questions) and translation loss (ceiling
minus test accuracy), mean +- SE over 5 fits.

Usage: python -m telepathy.experiments.t1_e2_negotiation [sender=gemma_s1] [receiver=qwen_s1]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from ..lm import data as D
from ..lm import translators as T
from ..lm.negotiate import GATE, encode, heldout_score, test_accuracy
from ..lm.train import load_pair

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
KS = (16, 32, 64, 128, 256)
FITS = 5
N_OWN = 3000


def mse(x):
    x = np.asarray(x, float)
    return float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0


def main(sender="gemma_s1", receiver="qwen_s1", families=T.FAMILIES, ks=KS, out_name=None):
    t_start = time.time()
    lms = {}
    recv = load_pair(receiver, lms)
    send = load_pair(sender, lms) if sender != receiver else recv
    fit_a, held_a = D.anchors()
    test = D.load("test")
    test_items = D.eval_items(test, "uniform")
    own_pool = D.load("train")[12000:12000 + N_OWN]   # outside the cached training passages

    t = time.time()
    enc = {
        "own_fit": encode(recv, fit_a), "own_held": encode(recv, held_a), "own_test": encode(recv, test),
        "own_msgs": encode(recv, own_pool),
        "peer_fit": encode(send, fit_a), "peer_held": encode(send, held_a), "peer_test": encode(send, test),
    }
    enc_s = time.time() - t
    print(f"encoded anchors, test and own messages in {enc_s:.0f} s", flush=True)

    ceiling = test_accuracy(recv, enc["own_test"], test_items)
    ceiling_held = heldout_score(recv, enc["own_held"], held_a)
    floor = test_accuracy(recv, enc["peer_test"], test_items)
    floor_held = heldout_score(recv, enc["peer_held"], held_a)
    print(f"ceiling {ceiling:.3f} (held-out {ceiling_held:.3f}); floor {floor:.3f} (held-out {floor_held:.3f})", flush=True)

    rows = []
    for fam in families:
        for K in ks:
            held, acc, cos, secs = [], [], [], []
            for f in range(FITS):
                rng = np.random.default_rng(1000 * K + f)
                sub = np.arange(len(fit_a)) if K == len(fit_a) else rng.choice(len(fit_a), K, replace=False)
                own_sub = rng.choice(N_OWN, int(0.9 * N_OWN), replace=False)
                t = time.time()
                tr = T.fit(fam, enc["peer_fit"][sub], enc["own_fit"][sub], seed=f, own_msgs=enc["own_msgs"][own_sub])
                secs.append(time.time() - t)
                held.append(heldout_score(recv, tr.apply_batch(enc["peer_held"]), held_a))
                acc.append(test_accuracy(recv, tr.apply_batch(enc["peer_test"]), test_items))
                cos.append(T.heldout_cosine(tr, enc["peer_held"], enc["own_held"]))
            h, hse = mse(held)
            a, ase = mse(acc)
            gap = [ceiling - x for x in acc]
            g, gse = mse(gap)
            row = {"family": fam, "K": K, "heldout": h, "heldout_se": hse, "acc": a, "acc_se": ase,
                   "translation_loss": g, "translation_loss_se": gse, "cosine": mse(cos)[0],
                   "pass_gate": bool(h >= GATE), "fit_s": float(np.mean(secs)),
                   "fits": {"heldout": held, "acc": acc}}
            rows.append(row)
            print(f"{fam:<11} K={K:<3} held-out {h:.3f}+-{hse:.3f} acc {a:.3f}+-{ase:.3f} loss {g:.3f} "
                  f"{'PASS' if row['pass_gate'] else 'fail'}", flush=True)

    # hub mode: the whole public anchor set, decoder fixed in advance (fitted on own messages only)
    hub = []
    for f in range(FITS):
        rng = np.random.default_rng(9000 + f)
        own_sub = rng.choice(N_OWN, int(0.9 * N_OWN), replace=False)
        dec = T.RelativeDecoder(enc["own_fit"], enc["own_msgs"][own_sub])
        tr = T.Relative(enc["peer_fit"], dec)
        hub.append((heldout_score(recv, tr.apply_batch(enc["peer_held"]), held_a),
                    test_accuracy(recv, tr.apply_batch(enc["peer_test"]), test_items)))
    hh, hhse = mse([x[0] for x in hub])
    ha, hase = mse([x[1] for x in hub])
    hub_row = {"heldout": hh, "heldout_se": hhse, "acc": ha, "acc_se": hase,
               "translation_loss": ceiling - ha, "pass_gate": bool(hh >= GATE), "K": len(fit_a)}
    print(f"hub mode: held-out {hh:.3f}+-{hhse:.3f} acc {ha:.3f}+-{hase:.3f}", flush=True)

    res = {"sender": sender, "receiver": receiver, "sender_repo": send.cfg.sender, "receiver_repo": recv.cfg.receiver,
           "gate": GATE, "fits": FITS, "test_items": len(test_items), "heldout_questions": sum(len(p["questions"]) for p in held_a),
           "ceiling": ceiling, "ceiling_heldout": ceiling_held, "floor": floor, "floor_heldout": floor_held,
           "rows": rows, "hub": hub_row, "encode_s": enc_s, "wall_s": time.time() - t_start,
           "note": "held-out score = receiver task accuracy (EM) on every question of the 32 held-out anchor passages; "
                   "acc = EM on the 1000-item test set, uniform questions; translation_loss = ceiling - acc. "
                   "The relative decoder is fitted by the receiver on its own messages only."}
    out = out_name or f"t1_e2_{sender}_to_{receiver}"
    (RESULTS / f"{out}.json").write_text(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    a = sys.argv[1:]
    main(*(a[:2] if a else []))
