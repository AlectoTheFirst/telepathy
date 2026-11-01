"""6.7 Loss handling: a mechanism check under identical simulated loss and 20 ms one-way delay
(the in-process injector; tc netem is Linux-only). All three disciplines send the same TLP packets
over the same transport (TLP over UDP by default):
  "datagrams"  = plain datagrams, no recovery (the "UDP, no TLP" baseline);
  "tcp-like"   = every slot reliable, in-order delivery (head-of-line blocking); not a kernel TCP;
  "TLP"        = TLP over UDP, priority-aware (protected prefix retransmitted, best-effort never).
The order datagrams < TLP < tcp-like in accuracy is by construction; the experiment measures the
cost of the choice (time, bytes) and the size of the gaps.

Part 2 (results/e7_transport_zipf.*) repeats the sweep with the pair trained on Zipf(1) value
queries, on Zipf queries: there the protected prefix can hold the popular entries."""
import threading
import time

import numpy as np
import torch

from ..negotiate.anchors import model_capabilities
from ..models.evaluate import send
from ..tlp.session import TLPOptions
from ..tlp.transport import Injector
from .common import COLORS, DEFAULT, MARKERS, TRANSPORT, ZIPF, models, save, savefig, session_pair

LOSS = (0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30)
LABELS = {"udp": "datagrams, no TLP recovery", "tcp": "TCP-like (reliable, in order)", "tlp": "TLP over UDP"}
DELAY = 0.020


def run_cell(rel: str, loss: float, n_msgs: int, ev, slots_all, B, seed: int, ckpt: str = DEFAULT,
             keep_rows: bool = False):
    # anchor_threshold 0: this experiment measures the transport, not verification. The Zipf pair
    # (0.81 accuracy) fails its own same-dialect anchor check at 0.9 (0.84 / 0.81, RESULTS.md 6.3).
    opts = TLPOptions(rto=0.10, ack_delay=0.03, grace=0.06, reliability=rel, tcp_max_retx=50, anchor_threshold=0.0)
    s, c, sc, cc = session_pair(model_capabilities(ckpt, "A"), model_capabilities(ckpt, "B"), opts=opts)
    s.transport.injector = Injector(loss=loss, delay=DELAY, seed=seed)
    c.transport.injector = Injector(loss=loss, delay=DELAY, seed=seed + 1)
    deadline = 3.0 if rel == "tcp" else 0.12
    sent_at, done = {}, threading.Event()
    rows = []

    def sender():
        for i in range(n_msgs):
            done.clear()
            sent_at[i] = time.monotonic()
            sc.send_latent(slots_all[i], protected=4)
            done.wait(5)

    th = threading.Thread(target=sender)
    th.start()
    for i in range(n_msgs):
        b0 = sc.session.stats.bytes_sent + cc.session.stats.bytes_sent
        r0 = sc.session.stats.retransmits
        first = final = None
        ans = -1
        try:
            for slots, mask, fin in cc.recv_latent(deadline=deadline, timeout=2.0):
                with torch.no_grad():
                    lg, _ = B.answer(ev.b_tokens[i: i + 1], slots[None], mask[None], torch.ones(1))
                ans = int(lg.argmax(-1))
                now = time.monotonic()
                if first is None:
                    first = now - sent_at[i]
                if fin:
                    final = now - sent_at[i]
        except TimeoutError:
            pass
        time.sleep(0.01)   # let late ACK/retransmits settle into this message's byte count
        rows.append({"acc": float(ans == int(ev.answer[i])), "t_first": first, "t_final": final,
                     "bytes": sc.session.stats.bytes_sent + cc.session.stats.bytes_sent - b0,
                     "retx": sc.session.stats.retransmits - r0})
        done.set()
    th.join(10)
    s.close()
    c.close()

    def mean(k):
        v = [r[k] for r in rows if r[k] is not None]
        return float(np.mean(v)) if v else None

    return {"transport": rel, "loss": loss, "acc": mean("acc"), "t_first_ms": 1e3 * mean("t_first"),
            "t_final_ms": 1e3 * mean("t_final"), "t_final_p95_ms": 1e3 * float(np.percentile(
                [r["t_final"] for r in rows if r["t_final"] is not None], 95)),
            "bytes": mean("bytes"), "retransmits": mean("retx"), "n": len(rows),
            **({"per_message": rows} if keep_rows else {})}


def main(n_msgs: int = 100):
    sweep(DEFAULT, "e7_transport", n_msgs)
    zipf_sweeps(n_msgs)
    zipf_cell(queries="zipf")
    zipf_cell(queries="uniform")


def zipf_sweeps(n_msgs: int = 100):
    """Zipf value queries: the Zipf-trained pair, and the uniform-trained pair as the control."""
    _, _, tdz = models(ZIPF)
    sweep(ZIPF, "e7_transport_zipf", n_msgs)
    sweep(DEFAULT, "e7_transport_zipf_control", n_msgs, eval_td=tdz)


def zipf_cell(n_msgs: int = 1000, loss: float = 0.30, n_boot: int = 2000, queries: str = "zipf"):
    """The 30% loss cell with more messages: TLP and TCP-like for the Zipf pair and for the
    uniform-trained control. queries = "zipf" evaluates on Zipf(1) value queries
    (results/e7_zipf_cell.json); "uniform" on uniform value queries (results/e7_zipf_cell_uniform.json).
    Reports accuracy +- SE, the share of TCP-like accuracy that TLP keeps, and the difference of the
    shares between the two pairs (bootstrap SE over messages)."""
    if queries == "zipf":
        _, _, td_q = models(ZIPF)
    else:
        _, _, td_q = models(DEFAULT)
    ev = td_q.eval_set(n_msgs)
    out = {"loss": loss, "n_msgs": n_msgs, "queries": queries, "query_zipf": td_q.task.query_zipf, "cells": {}}
    raw = {}
    for name, ckpt in (("zipf_pair", ZIPF), ("control", DEFAULT)):
        m, _, _ = models(ckpt)
        slots_all, _ = send(m["A"], ev)
        for k, rel in enumerate(("tlp", "tcp")):
            r = run_cell(rel, loss, n_msgs, ev, slots_all, m["B"], seed=7000 + 10 * k + (0 if name == "zipf_pair" else 1),
                         ckpt=ckpt, keep_rows=True)
            acc = np.array([x["acc"] for x in r["per_message"]])
            raw[(name, rel)] = acc
            out["cells"][f"{name}/{rel}"] = {"acc": float(acc.mean()), "se": float(acc.std(ddof=1) / np.sqrt(len(acc))),
                                             "t_final_ms": r["t_final_ms"], "bytes": r["bytes"]}
            print(f"{name:<9} {rel}: acc {acc.mean():.3f} +- {acc.std(ddof=1) / np.sqrt(len(acc)):.3f}", flush=True)
    rng = np.random.default_rng(0)
    n = n_msgs

    def share(name, idx_t, idx_c):
        return raw[(name, "tlp")][idx_t].mean() / raw[(name, "tcp")][idx_c].mean()
    boots = {"zipf_pair": [], "control": [], "diff": []}
    for _ in range(n_boot):
        ii = [rng.integers(0, n, n) for _ in range(4)]
        sz, sc = share("zipf_pair", ii[0], ii[1]), share("control", ii[2], ii[3])
        boots["zipf_pair"].append(sz)
        boots["control"].append(sc)
        boots["diff"].append(sz - sc)
    full = np.arange(n)
    est = {"zipf_pair": share("zipf_pair", full, full), "control": share("control", full, full)}
    est["diff"] = est["zipf_pair"] - est["control"]
    out["tlp_share_of_tcp"] = {k: {"value": float(est[k]), "se": float(np.std(boots[k], ddof=1))} for k in est}
    d = out["tlp_share_of_tcp"]["diff"]
    out["diff_over_se"] = d["value"] / d["se"] if d["se"] else None
    print(f"share zipf {est['zipf_pair']:.3f}, control {est['control']:.3f}, diff {d['value']:.3f} +- {d['se']:.3f}")
    save("e7_zipf_cell" if queries == "zipf" else "e7_zipf_cell_uniform", out)
    return out


def sweep(ckpt: str, out_name: str, n_msgs: int = 100, eval_td=None):
    m, _, td = models(ckpt)
    td = eval_td or td
    ev = td.eval_set(n_msgs)
    slots_all, _ = send(m["A"], ev)
    out = []
    for loss in LOSS:
        for k, rel in enumerate(("udp", "tcp", "tlp")):
            r = run_cell(rel, loss, n_msgs, ev, slots_all, m["B"], seed=int(loss * 1000) + 10 * k, ckpt=ckpt)
            out.append(r)
            print(f"loss {loss:4.2f} {rel}: acc {r['acc']:.3f} first {r['t_first_ms']:.0f} ms final "
                  f"{r['t_final_ms']:.0f} ms (p95 {r['t_final_p95_ms']:.0f}) bytes {r['bytes']:.0f} retx {r['retransmits']:.2f}")
    res = {"checkpoint": ckpt, "transport": TRANSPORT, "labels": LABELS, "query_zipf": td.task.query_zipf, "delay_ms": DELAY * 1e3, "loss": LOSS, "rows": out,
                          "note": "udp = plain datagrams without TLP recovery, tcp = reliable in-order discipline; "
                                  "both are delivery disciplines over the same TLP packets, not kernel stacks"}
    save(out_name, res)
    plot(res, out_name, td.task.query_zipf, eval_td is not None)


def plot(res, out_name, query_zipf=0.0, control=False):
    out = res["rows"]
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), dpi=130)
    for rel in ("udp", "tcp", "tlp"):
        r = [x for x in out if x["transport"] == rel]
        L = [100 * x["loss"] for x in r]
        kw = dict(color=COLORS[rel], marker=MARKERS[rel], lw=2, ms=6, label=LABELS[rel])
        axes[0].plot(L, [x["acc"] for x in r], **kw)
        axes[1].plot(L, [x["t_final_ms"] for x in r], **kw)
        axes[2].plot(L, [x["bytes"] for x in r], **kw)
    for ax, t in zip(axes, ("accuracy", "time to final answer (ms)", "bytes per message (both directions)")):
        ax.set_xlabel("packet loss (%)")
        ax.set_title(t, fontsize=9)
        ax.grid(True, alpha=0.25, lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(fontsize=8, frameon=False)
    axes[0].set_ylim(0, 1.05)
    if query_zipf:
        who = "control: pair trained on uniform queries" if control else "pair trained on them"
        fig.suptitle(f"Zipf({query_zipf:g}) value queries, {who}", fontsize=9)
    savefig(fig, out_name)


if __name__ == "__main__":
    main()
