"""Train every model for Telepathy-0 in parallel processes (one CPU thread each).

Runs: latent pairs seed 0 at d in {2,4,8,16,32} (N=8); pair seed 1 (A', B') at d=32 and
d=16 for the dialect experiments; text-link B; no-link B. --zipf trains the Zipf-query runs. Checkpoints go to checkpoints/.
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _run(job):
    import torch
    torch.set_num_threads(1)
    from telepathy.models import train as T
    kind, d, seed, steps, *rest = job
    zipf = rest[0] if rest else 0.0
    pos_bias = rest[1] if len(rest) > 1 else None
    suffix = rest[2] if len(rest) > 2 else ""
    log = lambda m: print(m, flush=True)
    if kind == "latent":
        # the suffix keeps runs of one configuration at different step counts apart
        return T.train_latent(d, seed, steps=steps, log=log, zipf=zipf, pos_bias=pos_bias, name_suffix=suffix)
    return T.train_baseline(kind, seed, steps=steps, log=log)


def _name(job):
    kind, d, seed, _, *rest = job
    if kind != "latent":
        return kind
    zipf = rest[0] if rest else 0.0
    pb = rest[1] if len(rest) > 1 else None
    return (f"latent_d{d}_s{seed}" + (f"_zipf{zipf:g}" if zipf else "") + (f"_pb{pb:g}" if pb is not None else "")
            + (rest[2] if len(rest) > 2 else ""))


# Zipf(1) value queries over key popularity (RESULTS.md 6.3, 6.7): (kind, d, seed, steps, zipf, pos_bias_init,
# name suffix). The 12000-step runs take about 10 minutes, so they are not in the default 5-minute set.
ZIPF_JOBS = [("latent", 32, 0, 4000, 1.0),
             ("latent", 32, 0, 4000, 1.0, 1.0, "_4k"), ("latent", 32, 0, 4000, 1.0, 0.0, "_4k"),
             ("latent", 32, 0, 12000, 1.0, 1.0), ("latent", 32, 0, 12000, 1.0, 0.0)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4000, help="latent pair steps")
    ap.add_argument("--baseline-steps", type=int, default=6000)
    ap.add_argument("--workers", type=int, default=min(9, os.cpu_count() or 1))
    ap.add_argument("--only", default=None, help="comma list of run names to train")
    ap.add_argument("--zipf", action="store_true", help="train only the Zipf-query runs (about 10 min)")
    a = ap.parse_args()
    jobs = [("latent", d, 0, a.steps) for d in (32, 16, 8, 4, 2)]
    # A' (seed 1): default d for the dialect experiments, and d=16 as the small-d
    # negotiation case (d <= 8 does not train symmetrically within the budget)
    jobs += [("latent", 32, 1, a.steps), ("latent", 16, 1, a.steps)]
    jobs += [("text", 0, 0, a.baseline_steps), ("nolink", 0, 0, a.baseline_steps)]
    if a.zipf:
        jobs = list(ZIPF_JOBS)
    if a.only:
        jobs = jobs + [j for j in ZIPF_JOBS if j not in jobs]
        keep = set(a.only.split(","))
        jobs = [j for j in jobs if _name(j) in keep]
    t0 = time.time()
    with ProcessPoolExecutor(a.workers, mp_context=get_context("spawn")) as ex:
        results = list(ex.map(_run, jobs))
    wall = time.time() - t0
    from telepathy.config import CKPT_DIR
    os.makedirs(CKPT_DIR, exist_ok=True)
    path = os.path.join(CKPT_DIR, "train_summary.json")
    summary = {"wall_seconds": wall, "runs": results}
    if (a.only or a.zipf) and os.path.exists(path):  # partial run: update the existing summary in place
        with open(path) as f:
            summary = json.load(f)
        new = {r["name"]: r for r in results}
        summary["runs"] = [new.pop(r["name"], r) for r in summary["runs"]] + list(new.values())
        summary.setdefault("partial_runs", []).append({"only": a.only or "--zipf", "wall_seconds": wall})
    with open(path, "w") as f:
        json.dump(summary, f, indent=1)
    print(f"trained {len(results)} runs in {wall:.0f}s wall")


if __name__ == "__main__":
    main()
