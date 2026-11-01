"""6.4 Trust: accuracy against receiver-assigned trust t at hint reliability s, with the correct peer
and with a mismatched peer, plus the gate activation. Curves should cross near t ~ s."""
import torch

from ..models.evaluate import receive, send
from .common import CAT, DEFAULT, figure, models, save, savefig

T_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
S_GRID = (0.3, 0.6, 0.9)


def main(n: int = 2000):
    m, _, td = models(DEFAULT)
    A, B = m["A"], m["B"]
    g = torch.Generator().manual_seed(11)
    base = td.eval_set(n)
    other = base.index(torch.roll(torch.arange(n), 1))   # a different table for each example
    good, mask = send(A, base)
    bad, _ = send(A, other)
    rows = []
    for s in S_GRID:
        b = td.with_hint(base, g, s)
        for t in T_GRID:
            rc = receive(B, b, good, mask, t)
            rm = receive(B, b, bad, mask, t)
            rows.append({"s": s, "t": t, "acc_correct_peer": rc["acc"], "acc_mismatched_peer": rm["acc"],
                         "gate_correct": rc["gate"], "gate_mismatched": rm["gate"]})
            print(f"s={s} t={t:.2f} correct {rc['acc']:.3f} mismatched {rm['acc']:.3f} gate {rc['gate']:.2f}")
    save("e4_trust", {"rows": rows})
    fig, axes = __import__("matplotlib.pyplot", fromlist=["subplots"]).subplots(1, 3, figsize=(10, 3.4), dpi=130, sharey=True)
    for ax, s in zip(axes, S_GRID):
        r = [x for x in rows if x["s"] == s]
        ax.plot(T_GRID, [x["acc_correct_peer"] for x in r], marker="o", color=CAT[0], lw=2, label="correct peer")
        ax.plot(T_GRID, [x["acc_mismatched_peer"] for x in r], marker="s", color=CAT[1], lw=2, label="mismatched peer")
        ax.plot(T_GRID, [x["gate_correct"] for x in r], marker="^", color="#8a8a85", lw=1.2, ls="--", label="gate (correct peer)")
        ax.axhline(s, color="#555", lw=0.8, ls=":")
        ax.axvline(s, color="#555", lw=0.8, ls=":")
        ax.set_title(f"hint reliability s={s}", fontsize=9)
        ax.set_xlabel("receiver-assigned trust t")
        ax.grid(True, alpha=0.25, lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("accuracy / gate")
    axes[0].legend(fontsize=7, frameon=False)
    savefig(fig, "e4_trust")


if __name__ == "__main__":
    main()
