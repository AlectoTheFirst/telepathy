"""Telepathy-0 end-to-end demo.

    .venv/bin/python scripts/demo.py                       # TLP over UDP, separate processes (default)
    sudo .venv/bin/python scripts/demo.py --transport raw  # raw IP protocol 253, separate processes
    .venv/bin/python scripts/demo.py --transport inproc    # same scenario, one process

Parties: resolver, probe (auditor), A1 (k0-k7), A2 (k8-k15), A' (other dialect), B (asker).
Runs: path 1 (resolve by knowledge) and path 2 (ROUTE) for an A1 query and an A2 query; a loss
run; an injected mismatched-intent message; the A'-B dialect negotiation. Then the trace tree.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from telepathy.config import (INTENT_SHARE_ENTRY_COUNT, INTENT_SHARE_VALUE, RESULTS_DIR)  # noqa: E402

IP = "127.0.0.1"
RES = (IP, 1)
CKPT, CKPT_PRIME = "latent_d32_s0", "latent_d32_s1"
HOLDERS = {
    # name: (ckpt, side, keys, endpoint, values seed, lie policy)
    "A1": (CKPT, "A", list(range(0, 8)), 11, 101, {INTENT_SHARE_ENTRY_COUNT: INTENT_SHARE_VALUE}),
    "A2": (CKPT, "A", list(range(8, 16)), 12, 202, {}),
    "A-prime": (CKPT_PRIME, "A", [0, 1, 2, 3, 8, 9, 10, 11], 13, 303, {}),
}


def _opts():
    from telepathy.tlp.session import TLPOptions
    return TLPOptions(rto=0.08, ack_delay=0.05, grace=0.08, n_fit=256, translator="shared_mlp")


def run_party(role: str, transport: str, t0: float, trace_path: str, bus=None, ready=None):
    from telepathy.agents import Holder, make_logger
    from telepathy.otel.probe import ProbeService, auditor_for
    from telepathy.otel.setup import init_tracing
    from telepathy.otel.spans import ResolverTracer, TLPTracer
    from telepathy.resolver.service import Resolver
    log = make_logger(role, t0=t0)
    tracer = init_tracing(role, trace_path)
    auditor = auditor_for(IP)
    if role == "resolver":
        obj = Resolver(IP, transport, bus=bus, auditor=auditor, log=log, tracer=ResolverTracer(tracer))
    elif role == "probe":
        obj = ProbeService(IP, transport, bus=bus, ckpt=CKPT, tracer=TLPTracer(tracer, "probe"), log=log)
    else:
        ckpt, side, keys, ep, seed, lie = HOLDERS[role]
        obj = Holder(role, ckpt, side, keys, ep, seed, transport, bus, IP, auditor, tracer, log, _opts(), lie_for=lie)
        for _ in range(50):
            try:
                obj.register(RES)
                break
            except TimeoutError:
                time.sleep(0.2)
        obj.serve_forever()
    if ready is not None:
        ready.set()
    if bus is None:  # separate process: stay alive until terminated
        while True:
            time.sleep(1)
    return obj


def banner(text):
    print(f"\n{'=' * 100}\n== {text}\n{'=' * 100}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transport", default="udp", choices=["udp", "raw", "inproc"])
    ap.add_argument("--trace", default=os.path.join(RESULTS_DIR, "demo_trace.jsonl"))
    a = ap.parse_args()

    from telepathy.agents import Asker, holder_table, make_logger
    from telepathy.otel.probe import auditor_for, load_probe
    from telepathy.otel.setup import init_tracing, shutdown_tracing
    from telepathy.otel.spans import print_trace_tree
    from telepathy.tlp.transport import RxLoss, raw_available

    if a.transport == "raw" and not raw_available():
        sys.exit("raw sockets need root on macOS (root or CAP_NET_RAW on Linux): "
                 "sudo .venv/bin/python scripts/demo.py --transport raw   (or the default --transport udp)")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if os.path.exists(a.trace):
        os.remove(a.trace)
    load_probe(CKPT)          # train the intent probe once if it is missing
    auditor_for(IP)           # create the probe's static key before the parties start
    t0 = time.time()
    roles = ["resolver", "probe", "A1", "A2", "A-prime"]
    procs, objs = [], []
    if a.transport in ("raw", "udp"):
        ctx = mp.get_context("spawn")
        for r in roles:
            p = ctx.Process(target=run_party, args=(r, a.transport, t0, a.trace), daemon=True)
            p.start()
            procs.append(p)
    else:
        import threading
        from telepathy.tlp.transport import DEFAULT_BUS
        for r in roles:
            ev = threading.Event()
            objs.append(run_party(r, "inproc", t0, a.trace, bus=DEFAULT_BUS, ready=ev))

    log = make_logger("B", t0=t0)
    tracer = init_tracing("B", a.trace)
    B = Asker(CKPT, "B", a.transport, None, IP, 30, auditor_for(IP), tracer, log, _opts())
    # wait until every holder is registered
    from telepathy.sock.socket import TLPSocket
    probe_sock = TLPSocket(a.transport, name="B-wait")
    for name in HOLDERS:
        for _ in range(200):
            try:
                probe_sock.resolve(RES, name=name)
                break
            except (TimeoutError, LookupError):
                time.sleep(0.25)
    probe_sock.close()

    def expect(name, key):
        keys, vals = holder_table(HOLDERS[name][2], HOLDERS[name][4])
        return f"{vals[keys.index(key)]:02d}" if key in keys else "?"

    summary = []

    def run(title, key, holder, **kw):
        banner(title)
        r = B.ask(key, **kw)
        want = kw.pop("_want", None) or (expect(holder, key) if key is not None else None)
        ok = r.get("ok", False)
        line = (f"RESULT {title}: holder={r.get('holder')} answer={r.get('answer_text')} expected={want} "
                f"trust={r.get('trust', 0):.2f}" if ok else f"RESULT {title}: no session ({r.get('error')})")
        print(line, flush=True)
        summary.append(line)
        return r

    run("Path 1: resolve by knowledge, query k5 (held by A1)", 5, "A1", path="resolve")
    run("Path 1: resolve by knowledge, query k12 (held by A2)", 12, "A2", path="resolve")
    run("Path 2: knowledge-addressed ROUTE via the resolver, query k3 (A1)", 3, "A1", path="route")
    run("Path 2: knowledge-addressed ROUTE via the resolver, query k14 (A2)", 14, "A2", path="route")
    run("Loss run: 10% random loss on DATA, plus forced loss of protected slot 2 and best-effort slot 6",
        9, "A2", path="resolve", rx_loss=RxLoss(0.10, seed=7, force_slots={2, 6}), deadline=0.5)
    banner("Intent mismatch: B asks A1 'count?' via ROUTE; A1 (injected) encodes the whole table")
    r = B.ask(None, intent=INTENT_SHARE_ENTRY_COUNT, path="route", route_keys=list(range(0, 8)))
    msgs = r.get("messages") or [{}]
    prog = msgs[-1].get("progress", []) if msgs else []
    trust_path = [round(p["trust"], 2) for p in prog]
    line = f"RESULT intent mismatch: answer={r.get('answer_text')} trust per yield={trust_path} session trust={r.get('trust', 0):.2f}"
    print(line, flush=True)
    summary.append(line)
    r = run("Dialect negotiation: B reaches A' (different dialect) by name, query k2", 2, "A-prime", path="name",
            name="A-prime", await_verdict=20.0)
    v = r.get("verdicts") or []
    line = (f"RESULT A' audit: probe VERDICT match={v[-1].get('match')} (read through its own translator)" if v
            else "RESULT A' audit: no VERDICT within 20 s")
    print(line, flush=True)
    summary.append(line)

    time.sleep(1.0)
    shutdown_tracing()
    for p in procs:
        p.terminate()
    time.sleep(0.5)
    banner("Summary")
    for line in summary:
        print(line)
    banner(f"Trace tree from {os.path.relpath(a.trace)}")
    print_trace_tree(a.trace)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
