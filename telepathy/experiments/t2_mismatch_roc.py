"""8.2 Mismatch detection: messages generated under one intent and declared as another; detection
rate and false-positive rate across the probe's confidence threshold (ROC)."""
import numpy as np
import torch

from ..config import DATA_INTENTS
from ..models.quant import quantize
from ..otel.probe import load_probe
from .common import CAT, DEFAULT, NEUTRAL, figure, models, save, savefig


def main(n=3000):
    m, meta, td = models(DEFAULT)
    clf, _ = load_probe(DEFAULT)
    g = torch.Generator().manual_seed(21)
    ev = td.eval_set(n, intents=DATA_INTENTS)           # ev.intent = declared intent
    lie = torch.rand(n, generator=g) < 0.5
    shift = torch.randint(1, 3, (n,), generator=g)
    encoded = torch.where(lie, (ev.intent - 1 + shift) % 3 + 1, ev.intent)
    with torch.no_grad():
        k2, v2, _ = td.make_tables(g, encoded)
        slots = quantize(m["A"].encode(td.a_tokens(ev.keys, ev.vals, encoded), encoded), 16)
    mask = torch.ones(n, meta["model"]["N"], dtype=torch.bool)
    obs, conf, probs = clf.classify(slots, mask)
    score = 1 - probs.gather(1, (ev.intent - 1)[:, None])[:, 0]     # 1 - P(declared)
    y = lie.numpy()
    sc = score.numpy()
    thr = np.unique(np.concatenate([[0, 1.01], sc]))
    roc = [{"thr": float(t), "tpr": float((sc[y] >= t).mean()), "fpr": float((sc[~y] >= t).mean())} for t in thr[::-1]]
    fpr = np.array([r["fpr"] for r in roc]); tpr = np.array([r["tpr"] for r in roc])
    auc = float(np.trapezoid(tpr, fpr))
    # operating point used by the service: flag when observed != declared; RESET when confidence >= 0.9
    flag = (obs != ev.intent).numpy()
    reset = flag & (conf.numpy() >= 0.9)
    op = {"detect_rate": float(flag[y].mean()), "false_positive_rate": float(flag[~y].mean()),
          "reset_rate_on_lies": float(reset[y].mean()), "reset_rate_on_honest": float(reset[~y].mean())}
    print(f"AUC {auc:.3f}; operating point {op}")
    save("t2_mismatch_roc", {"auc": auc, "operating_point": op, "roc": roc[:: max(1, len(roc) // 200)], "n": n})
    fig, ax = figure(4.6, 4.2)
    ax.plot(fpr, tpr, color=CAT[0], lw=2, label=f"probe (AUC {auc:.3f})")
    ax.plot([0, 1], [0, 1], color=NEUTRAL, ls=":", lw=1)
    ax.scatter([op["false_positive_rate"]], [op["detect_rate"]], color=CAT[1], s=40, zorder=3, label="argmax != declared")
    ax.set_xlabel("false-positive rate (honest messages flagged)")
    ax.set_ylabel("detection rate (mislabelled messages flagged)")
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    ax.set_title("Mismatch detection ROC", fontsize=9)
    savefig(fig, "t2_mismatch_roc")


if __name__ == "__main__":
    main()
