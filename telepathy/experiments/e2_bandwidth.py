"""6.2 Bandwidth sweep N x d x b against payload bits, the 67-bit floor, the text link and the
binary codec; plus a noise sweep to estimate the effective bits per dimension the models use.

Budget rule (both sides): payload bytes on the wire x 8, headers excluded. Latent payload includes
the per-slot float16 scale for b < 16."""
import math

import numpy as np
import torch

from ..models.evaluate import baseline_acc, latent_acc
from .common import (B_SWEEP, CODEC_ENTRY_BITS, COLORS, D_SWEEP, DEFAULT, N_SWEEP, NEUTRAL, codec_acc, figure,
                     latent_payload_bits, latent_payload_bytes, models, save, savefig)


def main(n_eval: int = 1000):
    _, _, td = models(DEFAULT)
    ev = td.eval_set(n_eval)
    floor = td.task.floor_bits()
    grid = []
    for d in D_SWEEP:
        m, meta, _ = models(f"latent_d{d}_s0")
        for N in N_SWEEP:
            for b in B_SWEEP:
                acc, _ = latent_acc(m["A"], m["B"], ev, bits=b, n_prefix=N)
                grid.append({"N": N, "d": d, "b": b, "bits": latent_payload_bits(N, d, b), "data_bits": N * d * b,
                             "payload_bytes": latent_payload_bytes(N, d, b), "acc": acc,
                             "schedule": meta.get("schedule")})
        print(f"d={d}: " + " ".join(f"{g['acc']:.2f}" for g in grid if g["d"] == d))
    text, _, _ = models("text")
    text_curve = []
    for nbytes in range(0, td.text_max + 1):
        text_curve.append({"bytes": nbytes, "bits": 8 * nbytes, "acc": baseline_acc(text["B"], td, ev, "text", budget=nbytes)})

    codec_curve = [{"bits": k * CODEC_ENTRY_BITS, "acc": codec_acc(td, ev, k * CODEC_ENTRY_BITS)}
                   for k in range(td.task.K + 1)]

    # envelope: best latent configuration with <= B payload bits, against text and codec at B bits
    def text_at(bits):
        nb = min(int(bits // 8), td.text_max)
        return text_curve[nb]["acc"]

    budgets = sorted({g["bits"] for g in grid})
    env, best = [], 0.0
    for B in budgets:
        best = max([g["acc"] for g in grid if g["bits"] <= B] + [0.0])
        env.append({"bits": B, "latent_best": best, "text": text_at(B), "codec": codec_acc(td, ev, B)})
    wins = [e["bits"] for e in env if e["latent_best"] > e["text"] + 0.01]
    crossover = max(wins) if wins else None
    # bits each link needs for >= 0.99 accuracy
    need = {"latent": min([g["bits"] for g in grid if g["acc"] >= 0.99], default=None),
            "text": min([t["bits"] for t in text_curve if t["acc"] >= 0.99], default=None),
            "codec": min([c["bits"] for c in codec_curve if c["acc"] >= 0.99], default=None)}

    # noise sweep (d=32, N=8, b=16): additive Gaussian noise relative to the slot RMS
    m, _, _ = models(DEFAULT)
    with torch.no_grad():
        slots = m["A"].encode(ev.a_tokens, ev.intent)
    rms = float(slots.pow(2).mean().sqrt())
    noise = []
    for rel in (0.0, 0.03, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5):
        acc, _ = latent_acc(m["A"], m["B"], ev, noise=rel * rms)
        noise.append({"noise_rel": rel, "acc": acc})
    clean = noise[0]["acc"]
    half = [x for x in noise if x["acc"] <= clean / 2]
    sigma_half = half[0]["noise_rel"] if half else None
    eff_bits = 0.5 * math.log2(1 + 1 / sigma_half ** 2) if sigma_half else None

    out = {"floor_bits": floor, "grid": grid, "text": text_curve, "codec": codec_curve, "envelope": env,
           "latent_beats_text_up_to_bits": crossover, "bits_for_99": need,
           "budget_rule": "payload bytes x 8 on both sides, headers excluded; latent includes the per-slot fp16 scale",
           "noise": noise, "slot_rms": rms, "noise_half_rel": sigma_half, "effective_bits_per_dim": eff_bits}
    save("e2_bandwidth", out)
    print(f"floor {floor:.1f} bits; latent beats text up to {crossover} bits; bits for 0.99: {need}; "
          f"noise for half accuracy {sigma_half} x RMS -> ~{eff_bits} bits/dim")

    fig, ax = figure(7.6, 4.6)
    for b in B_SWEEP:
        pts = sorted([g for g in grid if g["b"] == b], key=lambda g: g["bits"])
        ax.scatter([g["bits"] for g in pts], [g["acc"] for g in pts], s=22, color=COLORS[f"b{b}"], label=f"latent b={b}",
                   edgecolors="white", linewidths=0.5, zorder=3)
    ax.plot([e["bits"] for e in env], [e["latent_best"] for e in env], color=COLORS["latent"], lw=2, label="latent best (envelope)")
    tc = [t for t in text_curve if t["bits"] > 0]
    ax.plot([t["bits"] for t in tc], [t["acc"] for t in tc], color=COLORS["text"], lw=2, label="text link (truncated)")
    cc = [c for c in codec_curve if c["bits"] > 0]
    ax.plot([c["bits"] for c in cc], [c["acc"] for c in cc], color=NEUTRAL, lw=2, marker="D", ms=4,
            label="binary codec (4+7 bits per entry)")
    ax.axvline(floor, color="#555", lw=1, ls="--")
    ax.text(floor * 1.05, 0.95, f"floor {floor:.0f} bits", fontsize=8, color="#333")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("payload bits per message (8 x payload bytes, headers excluded)")
    ax.set_ylabel("accuracy (value query, K=8)")
    ax.set_ylim(-0.02, 1.05)
    ax.legend(fontsize=7, loc="lower right", frameon=False)
    ax.set_title("Bandwidth sweep: N in {1,2,4,8}, d in {2..32}, b in {16,8,4,2}", fontsize=9)
    savefig(fig, "e2_bandwidth")

    fig, ax = figure(5.6, 3.6)
    ax.plot([x["noise_rel"] for x in noise], [x["acc"] for x in noise], marker="o", color=COLORS["latent"], lw=2, ms=6)
    ax.set_xlabel("additive Gaussian noise (std / slot RMS)")
    ax.set_ylabel("accuracy")
    ax.set_title(f"Noise sweep (N=8, d=32): ~{eff_bits:.1f} effective bits/dim" if eff_bits else "Noise sweep", fontsize=9)
    savefig(fig, "e2_noise")


if __name__ == "__main__":
    main()
