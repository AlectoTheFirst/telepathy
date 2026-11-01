"""10.3 Negotiation diagnostics: why A' -> B negotiation fails.

1. Decode matrix: each model's receive port reading each model's send port (self-decode included),
   for both seeds at d=32 and d=16. A bad self-decode would make B's own latents a bad target.
2. Slot roles: which table entry each slot carries, per model (sensitivity of slot i to a change of
   entry j's value). Different slot orders across dialects would make slot i <-> slot i the wrong unit.
3. Translators: per-slot, shared across slots (N times the pairs), and the full-message linear map,
   with accuracy on the fitting anchors next to the held-out anchors and the 1000-query eval set,
   and R^2 on the latents. Anchor counts above 256 exceed the brief's limit; they are a diagnostic
   for "too few anchors" against "no map of this family exists".
"""
import time

import numpy as np
import torch

from ..models.evaluate import latent_acc, receive, send
from ..negotiate.anchors import model_capabilities
from ..negotiate.translators import STOCHASTIC, fit_translator
from .common import CAT, NEUTRAL, models, save, savefig

ANCHORS = (16, 32, 64, 128, 256, 1024, 4096)
BRIEF_MAX = 256
SEEDS = 5   # stochastic fits (the MLP families): every score is mean +- SE over 5 seeded fits
FAMILIES = ("linear", "procrustes", "mlp", "shared_linear", "shared_procrustes", "shared_mlp", "full_linear")
PLOT = {"linear": (CAT[0], "o", "-"), "mlp": (CAT[1], "^", "-"), "shared_linear": (CAT[2], "o", "--"),
        "shared_mlp": (CAT[3], "^", "--"), "full_linear": (NEUTRAL, "D", ":")}
# Capacity control at 256 anchors: is the shared fit better because of sharing or because of size?
CONTROL = (("mlp", {"hidden": 64, "steps": 300}), ("mlp", {"hidden": 256, "steps": 1500}),
           ("shared_mlp", {"hidden": 64, "steps": 300}), ("shared_mlp", {"hidden": 256, "steps": 1500}))


def decode_matrix(name, n=1000):
    m, _, td = models(name)
    ev = td.eval_set(n)
    return {f"{s}->{r}": latent_acc(m[s], m[r], ev)[0] for s in "AB" for r in "AB"}


@torch.no_grad()
def slot_roles(sender, td, n=512):
    """[N, K] share of each slot's change caused by a change of entry j's value."""
    ev = td.eval_set(n)
    g = torch.Generator().manual_seed(5)
    base = sender.encode(ev.a_tokens, ev.intent)
    K = td.task.K
    D = torch.zeros(base.shape[1], K)
    for j in range(K):
        a = ev.a_tokens.clone()
        a[:, 1 + j, 1] = (ev.vals[:, j] + torch.randint(1, td.task.value_range, (n,), generator=g)) % td.task.value_range + td.vocab.VAL0
        D[:, j] = (sender.encode(a, ev.intent) - base).norm(dim=-1).mean(0)
    return D / D.sum(1, keepdim=True)


def r2(pred, target):
    t = target.reshape(len(target), -1)
    p = pred.reshape(len(pred), -1)
    return float(1 - ((p - t) ** 2).sum() / ((t - t.mean(0)) ** 2).sum())


def translators(d, direction="A'->B", n_eval=1000, anchors=ANCHORS, families=FAMILIES, control=True):
    """direction "A'->B": A' (seed 1, side A) sends, B (seed 0) translates and reads, as in the demo.
    direction "B->A'": the other half of the same handshake (each side scores the peer's anchors)."""
    s0, s1 = f"latent_d{d}_s0", f"latent_d{d}_s1"
    if direction == "A'->B":
        (own_name, own_side), (peer_name, peer_side) = (s0, "B"), (s1, "A")
    else:
        (own_name, own_side), (peer_name, peer_side) = (s1, "A"), (s0, "B")
    mo, _, td = models(own_name)
    mp, _, _ = models(peer_name)
    receiver, own_sender = mo[own_side], mo["B" if own_side == "A" else "A"]
    own = model_capabilities(own_name, own_side).codec
    peer = model_capabilities(peer_name, peer_side).codec
    ev = td.eval_set(n_eval)
    peer_slots, mask = send(mp[peer_side], ev)
    ps = peer_slots.numpy()
    ceiling = receive(receiver, ev, *send(own_sender, ev), 1.0)["acc"]
    floor = receive(receiver, ev, peer_slots, mask, 1.0)["acc"]
    ph, oh = peer.encode("heldout", 32), own.encode("heldout", 32)
    rows = []
    jobs = [(fam, n, {}) for fam in families for n in anchors]
    if control:
        jobs += [(fam, BRIEF_MAX, kw) for fam, kw in CONTROL]
    keys = ("fit_acc", "heldout_acc", "eval_acc", "r2_fit", "r2_heldout")
    for fam, n, kw in jobs:
        pf, of = peer.encode("fit", n), own.encode("fit", n)
        per = []
        t0 = time.time()
        for seed in (range(SEEDS) if fam in STOCHASTIC else (0,)):
            tr = fit_translator(fam, pf, of, seed=seed, **kw)
            tf, th = tr.apply_batch(pf), tr.apply_batch(ph)
            per.append({"fit_acc": own.score(tf, "fit"), "heldout_acc": own.score(th, "heldout"),
                        "eval_acc": receive(receiver, ev, torch.from_numpy(tr.apply_batch(ps)), mask, 1.0)["acc"],
                        "r2_fit": r2(tf, of), "r2_heldout": r2(th, oh)})
        fit_s = (time.time() - t0) / len(per)
        row = {"family": fam, "anchors": n, "within_brief": n <= BRIEF_MAX, "fit_seconds": fit_s, "control": kw or None,
               "n_fits": len(per), "per_seed": per}
        for k in keys:
            v = np.array([x[k] for x in per])
            row[k] = float(v.mean())
            row[k + "_se"] = float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0
        rows.append(row)
        print(f"{direction} d={d} {fam:<17} {n:5d} {kw or ''}: fit {row['fit_acc']:.3f} held-out {row['heldout_acc']:.3f}"
              f"+-{row['heldout_acc_se']:.3f} "
              f"eval {row['eval_acc']:.3f}  R2 fit {row['r2_fit']:.2f} held-out {row['r2_heldout']:.2f}  {fit_s:.1f}s",
              flush=True)
    return {"d": d, "direction": direction, "ceiling": ceiling, "floor": floor, "rows": rows}


def main():
    res = {"decode_matrix": {}, "slot_roles": {}, "translators": []}
    for name in ("latent_d32_s0", "latent_d32_s1", "latent_d16_s0", "latent_d16_s1"):
        res["decode_matrix"][name] = dm = decode_matrix(name)
        print(name, {k: round(v, 3) for k, v in dm.items()})
        m, _, td = models(name)
        for side in "AB":
            D = slot_roles(m[side], td)
            res["slot_roles"][f"{name}:{side}"] = {"entry_of_slot": D.argmax(1).tolist(),
                                                   "share": [round(x, 3) for x in D.max(1).values.tolist()],
                                                   "matrix": D.numpy().round(3)}
            print(f"  {side} slot -> entry {D.argmax(1).tolist()}")
    for d in (32, 16):
        res["translators"].append(translators(d))
    # the other direction of the same handshake: the full sweep is symmetric in cost, so run the
    # families and anchor counts that matter for the threshold
    for d in (32, 16):
        res["translators"].append(translators(d, "B->A'", anchors=(64, 128, 256, 1024),
                                              families=("linear", "mlp", "shared_mlp", "full_linear"), control=False))
    save("n3_diagnostics", res)

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.4), dpi=130, sharex=True, sharey=True)
    for col, st in enumerate(res["translators"][:2]):
        for row, (key, label) in enumerate((("fit_acc", "fitting anchors"), ("eval_acc", "held-out eval set (1000)"))):
            ax = axes[row, col]
            for fam, (c, mk, ls) in PLOT.items():
                r = [x for x in st["rows"] if x["family"] == fam and not x["control"]]
                ax.plot([x["anchors"] for x in r], [x[key] for x in r], color=c, marker=mk, ls=ls, lw=2, ms=5,
                        label=fam.replace("_", " "))
            ax.axhline(st["ceiling"], color="#333", lw=1, ls="--")
            ax.axvline(BRIEF_MAX, color=NEUTRAL, lw=1, ls=":")
            ax.set_xscale("log", base=2)
            ax.set_title(f"A' -> B, d={st['d']}: accuracy on {label}", fontsize=9)
            ax.grid(True, alpha=0.25, lw=0.6)
            ax.spines[["top", "right"]].set_visible(False)
            if row == 1:
                ax.set_xlabel("fitting anchors (dotted line: brief limit 256)")
            if col == 0:
                ax.set_ylabel("accuracy (value query)")
    axes[0, 0].text(17, res["translators"][0]["ceiling"] - 0.08, "ceiling (joint pair)", fontsize=7)
    axes[0, 0].set_ylim(-0.02, 1.05)
    axes[0, 0].legend(fontsize=7, frameon=False, loc="center left")
    savefig(fig, "n3_diagnostics")


if __name__ == "__main__":
    main()
