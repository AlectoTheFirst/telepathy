"""8.5 Trust loop: VERDICT on and off. A peer injects mismatched messages (encoded for key existence,
declared SHARE_VALUE). B (hint reliability s) should stop relying on the flagged peer when the loop is on."""
import threading
import time

import numpy as np
import torch

from ..config import INTENT_SHARE_KEY_EXISTENCE, INTENT_SHARE_VALUE
from ..models.quant import quantize
from ..negotiate.anchors import model_capabilities
from ..otel.probe import ProbeService
from ..tlp.session import TLPOptions
from ..tlp.transport import InprocBus
from .common import CAT, DEFAULT, figure, models, save, savefig, session_pair

S_HINT = 0.6


def session(verdicts: bool, ev, honest_until: int, n_msgs: int, rep: int):
    m, meta, td = models(DEFAULT)
    bus = InprocBus()
    probe = ProbeService(bus=bus, ckpt=DEFAULT, send_verdicts=verdicts, reset_policy=False)
    s, c, sc, cc = session_pair(model_capabilities(DEFAULT, "A"), model_capabilities(DEFAULT, "B"), bus=bus,
                                opts=TLPOptions(), audit=True, auditor=probe.auditor, server_ep=7 + rep, client_ep=107 + rep)
    rows = []
    for i in range(n_msgs):
        it = INTENT_SHARE_VALUE if i < honest_until else INTENT_SHARE_KEY_EXISTENCE
        b = ev.index(slice(i, i + 1))
        with torch.no_grad():
            x = quantize(m["A"].encode(td.a_tokens(b.keys, b.vals, torch.tensor([it])), torch.tensor([it])), 16)[0]
        sc.send_latent(x, intent=INTENT_SHARE_VALUE)
        ans, trust = -1, None
        for slots, mask, fin in cc.recv_latent(deadline=0.15):
            if fin:
                time.sleep(0.005)   # the probe classifies on LAST_SLOT; give its VERDICT a moment
                mid = cc.last_message().msg_id
                trust = cc.message_trust(mid)
                with torch.no_grad():
                    lg, _ = m["B"].answer(b.b_tokens, slots[None], mask[None], torch.tensor([trust]))
                ans = int(lg.argmax(-1))
        rows.append({"i": i, "injected": i >= honest_until, "trust": trust, "correct": float(ans == int(b.answer[0]))})
    s.close(), c.close(), probe.close()
    return rows


def main(n_msgs=30, honest_until=10, reps=12):
    _, _, td = models(DEFAULT)
    out = {}
    for verdicts in (True, False):
        allr = []
        for rep in range(reps):
            ev = td.with_hint(td.fixed_set(700 + rep, n_msgs), torch.Generator().manual_seed(rep), S_HINT)
            allr.append(session(verdicts, ev, honest_until, n_msgs, rep))
        per = [{"i": i, "trust": float(np.mean([r[i]["trust"] for r in allr])),
                "acc": float(np.mean([r[i]["correct"] for r in allr]))} for i in range(n_msgs)]
        inj = [x for x in per if x["i"] >= honest_until]
        out["on" if verdicts else "off"] = {"per_message": per,
                                            "acc_on_injected": float(np.mean([x["acc"] for x in inj])),
                                            "acc_on_honest": float(np.mean([x["acc"] for x in per if x["i"] < honest_until])),
                                            "trust_on_injected": float(np.mean([x["trust"] for x in inj]))}
        print("loop", "on " if verdicts else "off", {k: round(v, 3) for k, v in out["on" if verdicts else "off"].items() if k != "per_message"})
    out.update({"hint_reliability": S_HINT, "honest_until": honest_until, "reps": reps})
    save("t5_trust_loop", out)
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), dpi=130)
    for key, c, mk in (("on", CAT[0], "o"), ("off", CAT[1], "s")):
        per = out[key]["per_message"]
        axes[0].plot([x["i"] for x in per], [x["trust"] for x in per], color=c, marker=mk, ms=4, lw=1.8, label=f"VERDICT loop {key}")
        axes[1].plot([x["i"] for x in per], [x["acc"] for x in per], color=c, marker=mk, ms=4, lw=1.8, label=f"VERDICT loop {key}")
    for ax, t in zip(axes, ("B's effective trust in the peer", "B's accuracy (hint s=0.6)")):
        ax.axvspan(honest_until - 0.5, n_msgs - 0.5, color="#8a8a85", alpha=0.12, lw=0)
        ax.set_xlabel("message index (shaded: injected mismatched messages)", fontsize=8)
        ax.set_title(t, fontsize=9)
        ax.grid(True, alpha=0.25, lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(-0.03, 1.05)
    axes[0].legend(fontsize=7, frameon=False)
    savefig(fig, "t5_trust_loop")


if __name__ == "__main__":
    main()
