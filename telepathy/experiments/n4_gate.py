"""10.4 How much of the negotiated link works: real A' -> B handshakes over the default transport,
5 translator seeds each, then 200 value queries through each established session.

d = 16 runs with the verification gate lowered to 0.80, and again at gate 0 so that every fit gets
through and its task accuracy can be measured. These are measurements, not a policy change: the
protocol default stays at 0.90. d = 32 runs at the default gate as the reference.
Scores: B's held-out score of A' (A' -> B), A''s held-out score of B (B -> A'), and B's task accuracy
on the translated messages (trust 1, no loss).

Translation loss (the headline metric of section 10): for each width, the jointly trained pair
(A -> B, seed 0) answers the same 200 queries with no translator and no network. The loss of a fit
is that accuracy minus the fit's task accuracy (a paired difference on the same queries).
`python -m telepathy.experiments.n4_gate --reference-only` adds the reference and the translation
loss to an existing results/n4_gate.json without new sessions (the fits are deterministic).
"""
import sys

import numpy as np
import torch

from ..models.evaluate import send
from ..negotiate.anchors import model_capabilities
from ..sock.socket import HandshakeError
from ..tlp.session import TLPOptions
from .common import models, save, session_pair

SEEDS = 5


def se(v):
    v = np.asarray(v, float)
    return float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0


def link(d: int, gate: float, seed: int, n_msgs: int = 200, ep: int = 0) -> dict:
    own, peer = f"latent_d{d}_s0", f"latent_d{d}_s1"
    mo, _, td = models(own)
    mp, _, _ = models(peer)
    opts = TLPOptions(anchor_threshold=gate, translator_seed=seed, n_fit=256)
    try:
        s, c, sc, cc = session_pair(model_capabilities(peer, "A"), model_capabilities(own, "B"), opts=opts,
                                    server_ep=200 + ep, client_ep=300 + ep)
    except HandshakeError as e:
        return {"d": d, "gate": gate, "seed": seed, "established": False, "error": str(e)}
    ev = td.eval_set(n_msgs)
    slots, _ = send(mp["A"], ev)
    correct = []
    for i in range(n_msgs):
        sc.send_latent(slots[i], protected=4)
        final = None
        for x, mask, fin in cc.recv_latent(deadline=0.5, timeout=2.0):
            if fin:
                final = (x, mask)
        if final is None:
            correct.append(0.0)
            continue
        with torch.no_grad():
            lg, _ = mo["B"].answer(ev.b_tokens[i: i + 1], final[0][None], final[1][None], torch.ones(1))
        correct.append(float(int(lg.argmax(-1)) == int(ev.answer[i])))
    out = {"d": d, "gate": gate, "seed": seed, "established": True,
           "heldout_a_prime_to_b": cc.session.anchor_score, "heldout_b_to_a_prime": sc.session.anchor_score,
           "task_acc": float(np.mean(correct)), "n_msgs": n_msgs}
    s.close(), c.close()
    return out


def reference(d: int, n_msgs: int = 200) -> dict:
    """Accuracy on the n4 queries without negotiation: the jointly trained pair (A -> B, seed 0),
    and A' forced through to B without a translator."""
    mo, _, td = models(f"latent_d{d}_s0")
    mp, _, _ = models(f"latent_d{d}_s1")
    ev = td.eval_set(n_msgs)
    out = {"d": d, "n_msgs": n_msgs}
    for name, sender in (("joint", mo["A"]), ("forced_through", mp["A"])):
        slots, _ = send(sender, ev)
        mask = torch.ones(slots.shape[:2], dtype=torch.bool)
        with torch.no_grad():
            lg, _ = mo["B"].answer(ev.b_tokens, slots, mask, torch.ones(n_msgs))
        out[name] = float((lg.argmax(-1) == ev.answer).float().mean())
    return out


def translation_loss(res: dict, default_gate: float = 0.90) -> list:
    """Per width: negotiated task accuracy, joint accuracy, translation loss (mean +- SE over the
    fits of the gate-0 run, or of the default-gate run where every fit passed), and pass/fail at
    the default gate from the held-out scores in both directions."""
    rows = []
    for d in sorted({s["d"] for s in res["settings"]}):
        ref = res["reference"][str(d)]
        sets = [s for s in res["settings"] if s["d"] == d and s["established"] == f"{res['seeds']}/{res['seeds']}"]
        runs = sets[0]["runs"]
        acc = [r["task_acc"] for r in runs]
        loss = [ref["joint"] - a for a in acc]
        passed = sum(r["heldout_a_prime_to_b"] >= default_gate and r["heldout_b_to_a_prime"] >= default_gate
                     for r in runs)
        rows.append({"d": d, "negotiated_acc": {"mean": float(np.mean(acc)), "se": se(acc)},
                     "joint_acc": ref["joint"], "forced_through_acc": ref["forced_through"],
                     "translation_loss": {"mean": float(np.mean(loss)), "se": se(loss)},
                     "gate": default_gate, "pass_at_gate": f"{passed}/{len(runs)}", "n_msgs": ref["n_msgs"]})
    return rows


def add_reference(res: dict) -> dict:
    res["reference"] = {str(d): reference(d) for d in sorted({s["d"] for s in res["settings"]})}
    res["translation_loss"] = translation_loss(res)
    for r in res["translation_loss"]:
        print(r, flush=True)
    return res


def main():
    if "--reference-only" in sys.argv:
        import json
        import os
        from ..config import RESULTS_DIR
        with open(os.path.join(RESULTS_DIR, "n4_gate.json")) as f:
            save("n4_gate", add_reference(json.load(f)))
        return
    res = {"seeds": SEEDS, "settings": []}
    k = 0
    # gate 0 at d = 16: every fit gets through, so task accuracy is measured for all 5 seeds
    for d, gate in ((16, 0.80), (16, 0.0), (32, 0.90)):
        runs = []
        for seed in range(SEEDS):
            r = link(d, gate, seed, ep=k)
            k += 1
            runs.append(r)
            print(r, flush=True)
        ok = [r for r in runs if r["established"]]
        summ = {"d": d, "gate": gate, "established": f"{len(ok)}/{len(runs)}", "runs": runs}
        for key in ("heldout_a_prime_to_b", "heldout_b_to_a_prime", "task_acc"):
            v = [r[key] for r in ok]
            summ[key] = {"mean": float(np.mean(v)) if v else None, "se": se(v) if v else None}
        res["settings"].append(summ)
        print({k_: v for k_, v in summ.items() if k_ != "runs"}, flush=True)
    save("n4_gate", add_reference(res))


if __name__ == "__main__":
    main()
