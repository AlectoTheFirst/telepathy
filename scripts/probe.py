"""Run the OpenTelepathy intent probe (auditor) as its own process.

    sudo .venv/bin/python scripts/probe.py            # raw transport, taps protocol 253 on loopback
    .venv/bin/python scripts/probe.py --train         # (re)train the intent classifier only
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="latent_d32_s0")
    ap.add_argument("--ip", default="127.0.0.1")
    ap.add_argument("--transport", default="udp", choices=["udp", "raw", "inproc"])
    ap.add_argument("--trace", default=None, help="JSONL span file")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--no-reset", action="store_true")
    a = ap.parse_args()
    from telepathy.otel.probe import ProbeService, train_probe
    if a.train:
        train_probe(a.ckpt)
        return
    from telepathy.agents import make_logger
    from telepathy.otel.setup import init_tracing
    from telepathy.otel.spans import TLPTracer
    tracer = TLPTracer(init_tracing("probe", a.trace), "probe")
    svc = ProbeService(a.ip, a.transport, ckpt=a.ckpt, tracer=tracer, log=make_logger("probe"),
                       reset_policy=not a.no_reset)
    print(f"probe listening on {a.ip} endpoint 2, dialect {svc.dialect}; public key {svc.auditor.pub.hex()}", flush=True)
    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()
