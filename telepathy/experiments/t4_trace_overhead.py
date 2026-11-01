"""8.4 Tracing overhead: bytes and latency per message with EXT_TRACE on and off."""
import time

import numpy as np
import torch

from ..negotiate.anchors import model_capabilities
from ..models.evaluate import send
from ..otel.setup import init_tracing
from ..otel.spans import TLPTracer, ext_from_span
from ..tlp.session import TLPOptions
from .common import CAT, DEFAULT, figure, latent_wire_bytes, models, save, savefig, session_pair


def run(trace: bool, slots_all, n):
    tr_s = tr_r = None
    if trace:
        tr_s = TLPTracer(init_tracing("A-overhead"), "A")
        tr_r = TLPTracer(init_tracing("B-overhead"), "B")
    s, c, sc, cc = session_pair(model_capabilities(DEFAULT, "A"), model_capabilities(DEFAULT, "B"),
                                opts=TLPOptions(), tracer_s=tr_s, tracer_r=tr_r)
    byts, lat = [], []
    for i in range(n):
        b0 = sc.session.stats.bytes_sent
        t0 = time.monotonic()
        if trace:
            with tr_s.span("latent.send", tlp__session_id=sc.session.session_id) as sp:
                ext = ext_from_span(sp)
                sc.trace_ext = lambda: ext
                sc.send_latent(slots_all[i])
        else:
            sc.send_latent(slots_all[i])
        for _ in cc.recv_latent(deadline=0.2):
            pass
        lat.append(time.monotonic() - t0)
        byts.append(sc.session.stats.bytes_sent - b0)
    s.close(), c.close()
    return float(np.mean(byts)), 1e3 * float(np.mean(lat)), 1e3 * float(np.percentile(lat, 95))


def main(n=200):
    m, meta, td = models(DEFAULT)
    ev = td.eval_set(n)
    slots_all, _ = send(m["A"], ev)
    off = run(False, slots_all, n)
    on = run(True, slots_all, n)
    N = meta["model"]["N"]
    small = {"d=4 slot payload (B)": 8, "EXT_TRACE (B)": 28, "d=4 message off (B)": latent_wire_bytes(N, 4, 16),
             "d=4 message on (B)": latent_wire_bytes(N, 4, 16, trace=True)}
    res = {"off": {"bytes": off[0], "lat_ms": off[1], "p95_ms": off[2]},
           "on": {"bytes": on[0], "lat_ms": on[1], "p95_ms": on[2]},
           "bytes_overhead_pct": 100 * (on[0] - off[0]) / off[0], "small_slot_example": small}
    print(res)
    save("t4_trace_overhead", res)
    fig, axes = __import__("matplotlib.pyplot", fromlist=["x"]).subplots(1, 2, figsize=(7.5, 3.2), dpi=130)
    for ax, key, lab in ((axes[0], "bytes", "bytes per message (d=32, N=8)"), (axes[1], "lat_ms", "send -> final (ms, in-process)")):
        v = [res["off"][key], res["on"][key]]
        ax.bar([0, 1], v, color=[CAT[0], CAT[1]], width=0.5)
        for i, x in enumerate(v):
            ax.text(i, x * 1.02, f"{x:.1f}", ha="center", fontsize=8)
        ax.set_xticks([0, 1], ["EXT_TRACE off", "EXT_TRACE on"], fontsize=8)
        ax.set_title(lab, fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    savefig(fig, "t4_trace_overhead")


if __name__ == "__main__":
    main()
