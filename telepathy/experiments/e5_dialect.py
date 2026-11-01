"""6.5 Dialect mismatch: the A'-B handshake fails at ANCHOR; forced through, accuracy is near chance."""
from ..models.evaluate import latent_acc
from ..negotiate.anchors import model_capabilities
from ..tlp.session import TLPOptions
from ..tlp.state import State
from .common import CAT, DEFAULT, NEUTRAL, PRIME, figure, models, save, savefig, session_pair


def main():
    m0, meta0, td = models(DEFAULT)
    m1, meta1, _ = models(PRIME)
    ev = td.eval_set(2000)
    hs = {}
    try:
        s, c, sc, cc = session_pair(model_capabilities(PRIME, "A"), model_capabilities(DEFAULT, "B"),
                                    opts=TLPOptions(), negotiate=False)
        hs = {"failed": False, "state": cc.fsm.state.value}
        s.close(), c.close()
    except Exception as e:
        conn = None
        hs = {"failed": True, "error": str(e)}
    same = {}
    s, c, sc, cc = session_pair(model_capabilities(DEFAULT, "A"), model_capabilities(DEFAULT, "B"), opts=TLPOptions())
    same = {"anchor_score": cc.session.anchor_score, "peer_anchor_score": cc.session.peer_anchor_score,
            "history": [x.value for x in cc.fsm.history]}
    s.close(), c.close()
    rows = {
        "A -> B (joint pair)": latent_acc(m0["A"], m0["B"], ev)[0],
        "A' -> B' (joint pair, seed 1)": latent_acc(m1["A"], m1["B"], ev)[0],
        "A' -> B (forced through)": latent_acc(m1["A"], m0["B"], ev)[0],
        "A -> B' (forced through)": latent_acc(m0["A"], m1["B"], ev)[0],
    }
    for k, v in rows.items():
        print(f"{k:<32} {v:.3f}")
    print("cross-seed handshake:", hs)
    save("e5_dialect", {"dialects": [meta0["dialect_id"], meta1["dialect_id"]], "handshake_cross": hs,
                        "handshake_same": same, "accuracy": rows, "chance": 1 / td.task.value_range})
    fig, ax = figure(6.4, 3.6)
    names = list(rows)
    cols = [CAT[0] if "joint" in n else CAT[1] for n in names]
    ax.barh(range(len(names)), [rows[n] for n in names], color=cols, height=0.55)
    for i, n in enumerate(names):
        ax.text(rows[n] + 0.01, i, f"{rows[n]:.3f}", va="center", fontsize=8)
    ax.axvline(1 / td.task.value_range, color=NEUTRAL, ls=":", lw=1)
    ax.set_yticks(range(len(names)), names, fontsize=8)
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("accuracy (value query, K=8)")
    ax.set_title("Dialect mismatch: cross-seed handshake " + ("fails at ANCHOR" if hs.get("failed") else "passed"), fontsize=9)
    savefig(fig, "e5_dialect")


if __name__ == "__main__":
    main()
