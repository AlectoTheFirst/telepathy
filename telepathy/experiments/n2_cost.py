"""10.2 Handshake cost: bytes and time spent on negotiation against anchor count (real in-process
handshakes, A' -> B; the held-out threshold is set to 0 so every handshake completes and is measured)."""
import time

from ..negotiate.anchors import model_capabilities
from ..tlp.session import TLPOptions
from .common import CAT, DEFAULT, PRIME, models, save, savefig, session_pair
from .n1_anchors import ANCHORS


def main():
    rows = []
    s, c, sc, cc = session_pair(model_capabilities(DEFAULT, "A"), model_capabilities(DEFAULT, "B"), opts=TLPOptions())
    same = {"handshake_bytes": cc.session.stats.handshake_bytes + sc.session.stats.handshake_bytes,
            "handshake_ms": 1e3 * cc.session.stats.handshake_seconds}
    s.close(), c.close()
    for n in ANCHORS:
        t0 = time.monotonic()
        o = TLPOptions(n_fit=n, anchor_threshold=0.0)
        s, c, sc, cc = session_pair(model_capabilities(PRIME, "A"), model_capabilities(DEFAULT, "B"), opts=o)
        st_c, st_s = cc.session.stats, sc.session.stats
        rows.append({"anchors": n, "handshake_bytes": st_c.handshake_bytes + st_s.handshake_bytes,
                     "negotiate_bytes": st_c.negotiate_bytes + st_s.negotiate_bytes,
                     "negotiate_ms": 1e3 * st_c.negotiate_seconds, "handshake_ms": 1e3 * st_c.handshake_seconds,
                     "heldout_score": cc.session.anchor_score})
        print(rows[-1])
        s.close(), c.close()
    save("n2_cost", {"same_dialect": same, "rows": rows})
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.3), dpi=130)
    x = [r["anchors"] for r in rows]
    axes[0].plot(x, [r["handshake_bytes"] / 1024 for r in rows], marker="o", color=CAT[0], lw=2, label="negotiated handshake")
    axes[0].axhline(same["handshake_bytes"] / 1024, color="#8a8a85", ls=":", lw=1, label="same-dialect handshake")
    axes[0].set_ylabel("KiB on the wire (both directions)")
    axes[1].plot(x, [r["handshake_ms"] for r in rows], marker="o", color=CAT[0], lw=2, label="handshake total")
    axes[1].plot(x, [r["negotiate_ms"] for r in rows], marker="s", color=CAT[1], lw=2, label="negotiation (fit + verify)")
    axes[1].set_ylabel("ms (in-process)")
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xlabel("fitting anchors")
        ax.grid(True, alpha=0.25, lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=7, frameon=False)
    savefig(fig, "n2_cost")


if __name__ == "__main__":
    main()
