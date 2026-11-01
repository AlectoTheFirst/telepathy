"""10.4 How much of the negotiated link works: real A' -> B handshakes over the default transport,
5 translator seeds each, then 200 value queries through each established session.

d = 16 runs with the verification gate lowered to 0.80, and again at gate 0 so that every fit gets
through and its task accuracy can be measured. These are measurements, not a policy change: the
protocol default stays at 0.9. d = 32 runs at the default gate as the reference.
Scores: B's held-out score of A' (A' -> B), A''s held-out score of B (B -> A'), and B's task accuracy
on the translated messages (trust 1, no loss).
"""
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


def main():
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
    save("n4_gate", res)


if __name__ == "__main__":
    main()
