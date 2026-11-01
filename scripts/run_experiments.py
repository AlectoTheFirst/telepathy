"""Run every experiment script (sections 6, 8 and 10); each writes results/<name>.json and .png."""
import argparse
import importlib
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

EXPERIMENTS = ["e1_link", "e2_bandwidth", "e3_progressive", "e4_trust", "e5_dialect", "e6_monitor_tap",
               "e7_transport", "t1_probe_acc", "t2_mismatch_roc", "t3_probe_key", "t4_trace_overhead",
               "t5_trust_loop", "n1_anchors", "n2_cost", "n3_diagnostics", "n4_gate"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="comma list of experiment names")
    a = ap.parse_args()
    names = a.only.split(",") if a.only else EXPERIMENTS
    from telepathy.otel.probe import load_probe
    load_probe("latent_d32_s0")   # train the intent probe once if missing
    times = {}
    t_all = time.time()
    for n in names:
        t0 = time.time()
        print(f"\n===== {n}", flush=True)
        importlib.import_module(f"telepathy.experiments.{n}").main()
        times[n] = time.time() - t0
        print(f"===== {n} done in {times[n]:.0f}s", flush=True)
    print("\n" + "\n".join(f"{k:<20} {v:6.0f}s" for k, v in times.items()))
    print(f"total {time.time() - t_all:.0f}s")


if __name__ == "__main__":
    main()
