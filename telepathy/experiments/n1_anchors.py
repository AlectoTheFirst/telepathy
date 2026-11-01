"""10.1 Negotiation: accuracy against number of fitting anchors per translator family, at the default
d (32) and a smaller d (16). Ceiling: the jointly trained pair. Floor: the mismatched pair forced
through. Also the full-message linear diagnostic and the intent probe's accuracy on A' messages."""
import numpy as np
import torch

from ..config import DATA_INTENTS
from ..models.evaluate import receive, send
from ..negotiate.anchors import model_capabilities
from ..negotiate.translators import STOCHASTIC, fit_translator
from ..otel.probe import load_probe
from .common import COLORS, DEFAULT, MARKERS, models, save, savefig

ANCHORS = (16, 32, 64, 128, 256)
FAMILIES = ("linear", "procrustes", "mlp", "shared_mlp")


def translate(tr, slots):
    return torch.from_numpy(tr.apply_batch(slots.numpy()))


def setting(d, n_eval=1000):
    own_name, peer_name = f"latent_d{d}_s0", f"latent_d{d}_s1"
    mo, _, td = models(own_name)
    mp, _, _ = models(peer_name)
    own = model_capabilities(own_name, "B").codec
    peer = model_capabilities(peer_name, "A").codec
    ev = td.eval_set(n_eval)
    peer_slots, mask = send(mp["A"], ev)
    ceiling = receive(mo["B"], ev, *send(mo["A"], ev), 1.0)["acc"]
    floor = receive(mo["B"], ev, peer_slots, mask, 1.0)["acc"]
    rows = []
    for fam in FAMILIES:
        for n in ANCHORS:
            accs, helds = [], []
            for seed in (range(5) if fam in STOCHASTIC else (0,)):   # mean +- SE over 5 seeded fits
                tr = fit_translator(fam, peer.encode("fit", n), own.encode("fit", n), seed=seed)
                accs.append(receive(mo["B"], ev, translate(tr, peer_slots), mask, 1.0)["acc"])
                helds.append(own.score(tr.apply_batch(peer.encode("heldout", 32)), "heldout"))
            se = lambda v: float(np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0
            acc, held = float(np.mean(accs)), float(np.mean(helds))
            rows.append({"family": fam, "anchors": n, "acc": acc, "acc_se": se(accs), "heldout_score": held,
                         "heldout_se": se(helds), "n_fits": len(accs)})
            print(f"d={d} {fam:<10} anchors {n:3d}: eval acc {acc:.3f} held-out {held:.3f}")
    tr = fit_translator("full_linear", peer.encode("fit", 256), own.encode("fit", 256))
    diag = receive(mo["B"], ev, translate(tr, peer_slots), mask, 1.0)["acc"]
    print(f"d={d} full-message linear (256 anchors): {diag:.3f}; ceiling {ceiling:.3f}; floor {floor:.3f}")
    return {"d": d, "rows": rows, "ceiling": ceiling, "floor": floor, "full_message_linear_256": diag}


def probe_on_prime(n=1500):
    """Intent probe (trained on dialect A) reading A' messages through a linear translator fitted on
    the anchors it witnessed in the clear (A' fit anchors -> A's own fit latents)."""
    clf, _ = load_probe(DEFAULT)
    mp, meta, td = models("latent_d32_s1")
    ev = td.eval_set(n, intents=DATA_INTENTS)
    slots, mask = send(mp["A"], ev)
    ref = model_capabilities(DEFAULT, "A").codec
    peer = model_capabilities("latent_d32_s1", "A").codec
    out = {"untranslated": float((clf.classify(slots, mask)[0] == ev.intent).float().mean())}
    for k in ANCHORS:
        for fam in ("linear", "shared_mlp"):
            tr = fit_translator(fam, peer.encode("fit", k), ref.encode("fit", k))
            key = f"translated_{k}" if fam == "linear" else f"translated_{fam}_{k}"
            out[key] = float((clf.classify(translate(tr, slots), mask)[0] == ev.intent).float().mean())
    print("probe intent accuracy on A' messages:", {k: round(v, 3) for k, v in out.items()})
    return out


def main():
    res = {"settings": [setting(32), setting(16)], "probe_on_prime": probe_on_prime()}
    save("n1_anchors", res)
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), dpi=130, sharey=True)
    for ax, st in zip(axes, res["settings"]):
        for fam in FAMILIES:
            r = [x for x in st["rows"] if x["family"] == fam]
            ax.plot([x["anchors"] for x in r], [x["acc"] for x in r], marker=MARKERS[fam], color=COLORS[fam], lw=2, label=fam)
        ax.axhline(st["ceiling"], color="#333", lw=1, ls="--")
        ax.text(16, st["ceiling"] - 0.07, f"ceiling (joint pair) {st['ceiling']:.2f}", fontsize=7)
        ax.axhline(st["floor"], color="#8a8a85", lw=1, ls=":")
        ax.text(16, st["floor"] + 0.09, f"floor (forced through) {st['floor']:.2f}", fontsize=7)
        ax.scatter([256], [st["full_message_linear_256"]], marker=MARKERS["full_linear"], color=COLORS["full_linear"],
                   s=40, zorder=3, label="full-message linear (diag.)")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("fitting anchors")
        ax.set_title(f"A' -> B negotiated, d={st['d']}", fontsize=9)
        ax.grid(True, alpha=0.25, lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("accuracy (value query)")
    axes[0].set_ylim(-0.02, 1.05)
    axes[0].legend(fontsize=7, frameon=False, loc="center right")
    savefig(fig, "n1_anchors")


if __name__ == "__main__":
    main()
