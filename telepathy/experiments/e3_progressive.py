"""6.3 Progressive delivery: accuracy against slots received, with text truncated to the same bytes.

Part 2 repeats the sweep with Zipf(1) value queries over key popularity. With uniform queries every
entry is worth the same, so no slot order can help a prefix; with skewed queries a sender can put the
popular entries first. Both the uniform-trained pair and the Zipf-trained pair read the same Zipf set."""
from ..models.evaluate import baseline_acc, latent_acc
from ..models.quant import slot_payload_len
from .common import CAT, COLORS, DEFAULT, NEUTRAL, ZIPF, codec_acc, figure, models, save, savefig


def main():
    m, meta, td = models(DEFAULT)
    N, d = meta["model"]["N"], meta["model"]["d"]
    ev = td.eval_set(2000)
    text, _, _ = models("text")
    rows = []
    for bits in (16, 4):
        per = slot_payload_len(d, bits)
        for n in range(1, N + 1):
            acc, _ = latent_acc(m["A"], m["B"], ev, bits=bits, n_prefix=n)
            tacc = baseline_acc(text["B"], td, ev, "text", budget=n * per)
            rows.append({"bits": bits, "slots": n, "bytes": n * per, "latent": acc, "text_same_bytes": tacc,
                         "codec_same_bytes": codec_acc(td, ev, 8 * n * per)})
            print(f"b={bits} slots={n} bytes={n * per:4d} latent {acc:.3f} text {tacc:.3f}")
    # text at the same bytes is generous here (one 64 B float16 slot holds ~9 text entries), so also
    # compare against text truncated to the same *fraction* of its full length
    full = td.text_max
    frac = [{"slots": n, "text_same_fraction": baseline_acc(text["B"], td, ev, "text", budget=round(full * n / N))}
            for n in range(1, N + 1)]
    mono = all(rows[i]["latent"] <= rows[i + 1]["latent"] + 0.01 for i in range(N - 1))
    zipf = zipf_part()
    save("e3_progressive", {"rows": rows, "text_same_fraction": frac, "monotone_b16": mono, "zipf": zipf})
    fig, ax = figure(6.4, 4.0)
    for bits, c in ((16, COLORS["b16"]), (4, COLORS["b4"])):
        r = [x for x in rows if x["bits"] == bits]
        ax.plot([x["slots"] for x in r], [x["latent"] for x in r], marker="o", color=c, lw=2, label=f"latent b={bits}")
    ax.plot([x["slots"] for x in frac], [x["text_same_fraction"] for x in frac], marker="s", color=COLORS["text"], lw=2,
            ls="--", label="text, same fraction of its bytes")
    r = [x for x in rows if x["bits"] == 4]
    ax.plot([x["slots"] for x in r], [x["text_same_bytes"] for x in r], marker="s", color=COLORS["text"], lw=1.5,
            label="text, same bytes as latent b=4")
    ax.set_xlabel("slots received (priority order)")
    ax.set_ylabel("accuracy (value query, K=8)")
    ax.set_ylim(-0.02, 1.05)
    ax.legend(fontsize=7, frameon=False)
    ax.set_title("Progressive delivery (nested-dropout slot order)", fontsize=9)
    savefig(fig, "e3_progressive")
    plot_zipf(zipf)



ZIPF_RUNS = (  # (checkpoint, label); the uniform-query pair is the control
    ("latent_d32_s0", "uniform queries, prior 3.0, 4000 steps (control)"),
    ("latent_d32_s0_zipf1", "Zipf, prior 3.0, 4000 steps"),
    ("latent_d32_s0_zipf1_pb1_4k", "Zipf, prior 1.0, 4000 steps"),
    ("latent_d32_s0_zipf1_pb0_4k", "Zipf, prior 0, 4000 steps"),
    ("latent_d32_s0_zipf1_pb1", "Zipf, prior 1.0, 12000 steps"),
    ("latent_d32_s0_zipf1_pb0", "Zipf, prior 0, 12000 steps"),
)


def zipf_part(n_eval=2000):
    """Every pair on Zipf(1) value queries (the skewed case) and on uniform value queries (what a
    skewed sender gives up): accuracy against slots received, per-slot query mass, and accuracy
    against the popularity rank of the queried key with all slots and with the protected half."""
    import os
    import torch
    from ..config import CKPT_DIR
    from ..data import TaskData
    _, _, tdz = models(ZIPF)
    tdu = TaskData()
    evz, evu = tdz.eval_set(n_eval), tdu.eval_set(n_eval)
    text, _, _ = models("text")
    N = 8
    full = tdz.text_max
    out = {"query_zipf": tdz.task.query_zipf, "runs": [],
           "text_same_fraction": {"zipf": [baseline_acc(text["B"], tdz, evz, "text", budget=round(full * n / N)) for n in range(1, N + 1)],
                                  "uniform": [baseline_acc(text["B"], tdu, evu, "text", budget=round(full * n / N)) for n in range(1, N + 1)]}}
    rank = tdz.key_rank[evz.qkey]                       # popularity rank of each queried key (0 = most popular)
    for name, label in ZIPF_RUNS:
        if not os.path.exists(os.path.join(CKPT_DIR, name + ".pt")):
            print(f"skip {name}: no checkpoint")
            continue
        m, meta, _ = models(name)
        row = {"checkpoint": name, "label": label, "steps": meta.get("steps"),
               "pos_bias_init": meta["model"].get("pos_bias_init", 3.0), "query_zipf": meta["task"].get("query_zipf", 0.0),
               "zipf_by_prefix": [latent_acc(m["A"], m["B"], evz, n_prefix=n)[0] for n in range(1, N + 1)],
               "uniform_by_prefix": [latent_acc(m["A"], m["B"], evu, n_prefix=n)[0] for n in range(1, N + 1)],
               "query_mass_by_slot": query_mass_by_slot(m, tdz, n_eval)}
        for k, key in ((N, "acc_by_rank_all_slots"), (N // 2, "acc_by_rank_protected_half")):
            from ..models.evaluate import receive, send
            slots, mask = send(m["A"], evz, n_prefix=k)
            ok = (receive(m["B"], evz, slots, mask)["pred"] == evz.answer).float()
            row[key] = [float(ok[rank == r].mean()) if (rank == r).any() else None for r in range(tdz.task.key_range)]
        row["query_share_by_rank"] = [float((rank == r).float().mean()) for r in range(tdz.task.key_range)]
        out["runs"].append(row)
        print(f"{label}: zipf {[round(x, 2) for x in row['zipf_by_prefix']]} uniform {[round(x, 2) for x in row['uniform_by_prefix']]}")
        print(f"   slot mass {[round(x, 2) for x in row['query_mass_by_slot']]}")
        print(f"   acc by rank (8 slots) {[round(x, 2) for x in row['acc_by_rank_all_slots']]}")
        print(f"   acc by rank (4 slots) {[round(x, 2) for x in row['acc_by_rank_protected_half']]}")
    return out


def query_mass_by_slot(m, td, n=2000):
    """For each slot i: the accuracy gain on Zipf queries when only slot i is present, against no slot."""
    import torch
    from ..models.evaluate import receive, send
    ev = td.eval_set(n)
    slots, _ = send(m["A"], ev)
    base = receive(m["B"], ev, slots * 0, torch.zeros(n, slots.shape[1], dtype=torch.bool))["acc"]
    out = []
    for i in range(slots.shape[1]):
        mask = torch.zeros(n, slots.shape[1], dtype=torch.bool)
        mask[:, i] = True
        out.append(receive(m["B"], ev, slots * mask[..., None], mask)["acc"] - base)
    return out


def plot_zipf(z):
    """Left: accuracy against slots on Zipf queries. Middle: the same pairs on uniform queries.
    Right: accuracy against popularity rank with the protected half (4 slots)."""
    import matplotlib.pyplot as plt
    pick = (DEFAULT, "latent_d32_s0_zipf1", "latent_d32_s0_zipf1_pb1", "latent_d32_s0_zipf1_pb0")
    show = [r for n in pick for r in z["runs"] if r["checkpoint"] == n]
    colors = [NEUTRAL, CAT[0], CAT[2], CAT[3]]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), dpi=130)
    x = list(range(1, 9))
    for r, c in zip(show, colors):
        # the control is drawn on top, dashed: the prior-3.0 Zipf curve lies exactly on it
        kw = dict(ls="--", zorder=3) if r["checkpoint"] == DEFAULT else {}
        axes[0].plot(x, r["zipf_by_prefix"], marker="o", color=c, lw=2, label=r["label"], **kw)
        axes[1].plot(x, r["uniform_by_prefix"], marker="o", color=c, lw=2, label=r["label"], **kw)
        axes[2].plot(range(16), r["acc_by_rank_protected_half"], marker="o", color=c, lw=2, label=r["label"], **kw)
    axes[2].set_xticks(range(0, 16, 2))
    axes[0].plot(x, z["text_same_fraction"]["zipf"], marker="s", color=COLORS["text"], lw=1.5, ls="--", label="text, same fraction of bytes")
    axes[1].plot(x, z["text_same_fraction"]["uniform"], marker="s", color=COLORS["text"], lw=1.5, ls="--")
    for ax, t, xl in ((axes[0], "Zipf(1) value queries", "slots received (priority order)"),
                      (axes[1], "uniform value queries", "slots received (priority order)"),
                      (axes[2], "Zipf queries, 4 protected slots only", "popularity rank of the queried key (0 = most popular)")):
        ax.set_title(t, fontsize=9)
        ax.set_xlabel(xl, fontsize=8)
        ax.set_ylim(-0.02, 1.05)
        ax.grid(True, alpha=0.25, lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("accuracy (value query)")
    axes[0].legend(fontsize=6, frameon=False, loc="upper left")
    savefig(fig, "e3_progressive_zipf")


if __name__ == "__main__":
    main()
