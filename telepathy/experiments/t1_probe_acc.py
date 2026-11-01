"""8.1 Intent probe accuracy per intent class, and against slot prefix length."""
import torch

from ..config import DATA_INTENTS, INTENT_NAMES
from ..models.evaluate import send
from ..otel.probe import load_probe
from .common import CAT, DEFAULT, figure, models, save, savefig


def main(n=3000):
    m, meta, td = models(DEFAULT)
    clf, _ = load_probe(DEFAULT)
    ev = td.eval_set(n, intents=DATA_INTENTS)
    N = meta["model"]["N"]
    per_class, per_prefix = {}, []
    for p in range(1, N + 1):
        slots, mask = send(m["A"], ev, n_prefix=p)
        obs, conf, _ = clf.classify(slots, mask)
        ok = obs == ev.intent
        per_prefix.append({"slots": p, "acc": float(ok.float().mean()),
                           **{INTENT_NAMES[i]: float(ok[ev.intent == i].float().mean()) for i in DATA_INTENTS}})
        if p == N:
            per_class = {INTENT_NAMES[i]: float(ok[ev.intent == i].float().mean()) for i in DATA_INTENTS}
    print("per class (full message):", {k: round(v, 3) for k, v in per_class.items()})
    print("vs prefix:", [round(r["acc"], 3) for r in per_prefix])
    save("t1_probe_acc", {"per_class": per_class, "per_prefix": per_prefix, "n": n})
    fig, ax = figure(6.0, 3.6)
    for k, i in enumerate(DATA_INTENTS):
        name = INTENT_NAMES[i]
        ax.plot([r["slots"] for r in per_prefix], [r[name] for r in per_prefix], marker="os^"[k], color=CAT[k], lw=2, label=name)
    ax.set_xlabel("slots seen by the probe (prefix)")
    ax.set_ylabel("intent accuracy")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7, frameon=False)
    ax.set_title("Intent probe accuracy by class and prefix", fontsize=9)
    savefig(fig, "t1_probe_acc")


if __name__ == "__main__":
    main()
