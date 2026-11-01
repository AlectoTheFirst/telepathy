"""6.1 Link comparison at K=8: no link, text link at equal bytes, latent link at b=16 and b=4."""
from ..models.evaluate import baseline_acc, latent_acc
from .common import (CODEC_ENTRY_BITS, COLORS, DEFAULT, codec_acc, figure, latent_payload_bytes,
                     latent_wire_bytes, models, save, savefig, text_wire_bytes)


def overhead(packets: int) -> dict:
    """Packetisation overhead, kept apart from the payload comparison: the 28-byte TLP header and
    the 16-byte AEAD tag are paid once per packet. The latent link sends one slot per packet (N
    packets); text and codec send one packet per message."""
    from ..crypto.aead import TAG_LEN
    from ..tlp.header import HDR
    return {"packets": packets, "header_bytes": packets * HDR.size, "tag_bytes": packets * TAG_LEN}


def main():
    (m, meta, td) = models(DEFAULT)
    N, d = meta["model"]["N"], meta["model"]["d"]
    ev = td.eval_set(2000)
    nolink, _, _ = models("nolink")
    text, _, _ = models("text")
    full_text = int(td.text_bytes(ev).float().mean())
    rows = [{"link": "no link", "acc": baseline_acc(nolink["B"], td, ev, "nolink"), "payload_bytes": 0, "wire_bytes": 0,
             **overhead(0)}]
    for bits in (16, 4):
        pb = latent_payload_bytes(N, d, bits)
        acc, _ = latent_acc(m["A"], m["B"], ev, bits=bits)
        rows.append({"link": f"latent b={bits}", "acc": acc, "payload_bytes": pb, "wire_bytes": latent_wire_bytes(N, d, bits),
                     **overhead(N)})
        tb = min(pb, td.text_max)
        rows.append({"link": f"text @ {pb} B", "acc": baseline_acc(text["B"], td, ev, "text", budget=pb),
                     "payload_bytes": tb, "wire_bytes": text_wire_bytes(tb), **overhead(1)})
    # the binary codec needs only 11 B for the whole table; also show it at the floor-sized 8 B budget
    cb = (td.task.K * CODEC_ENTRY_BITS + 7) // 8
    for nbytes in (8, cb):
        rows.append({"link": f"codec @ {nbytes} B", "acc": codec_acc(td, ev, 8 * nbytes), "payload_bytes": nbytes,
                     "wire_bytes": text_wire_bytes(nbytes), **overhead(1)})
    rows.append({"link": "text unbounded", "acc": baseline_acc(text["B"], td, ev, "text"), "payload_bytes": full_text,
                 "wire_bytes": text_wire_bytes(full_text), **overhead(1)})
    for r in rows:
        print(f"{r['link']:<16} acc {r['acc']:.3f}  payload {r['payload_bytes']:>4} B  packets {r['packets']}  headers "
              f"{r['header_bytes']:>3} B  tags {r['tag_bytes']:>3} B  wire {r['wire_bytes']:>4} B")
        assert r["wire_bytes"] == r["payload_bytes"] + r["header_bytes"] + r["tag_bytes"]
    save("e1_link", {"N": N, "d": d, "K": td.task.K, "rows": rows, "mean_text_bytes": full_text,
                     "note": "text at equal bytes = serialised table truncated to the latent payload bytes; "
                             "the full table is ~6.9 B/entry, so at these budgets text carries the whole table"})
    fig, ax = figure(7.2, 3.8)
    names = [r["link"] for r in rows]
    cols = [COLORS["nolink"] if "no link" in n else COLORS["latent"] if "latent" in n else
            "#555555" if "codec" in n else COLORS["text"] for n in names]
    bars = ax.bar(range(len(rows)), [r["acc"] for r in rows], color=cols, width=0.6)
    for i, r in enumerate(rows):
        ax.text(i, r["acc"] + 0.02, f"{r['acc']:.2f}\n{r['payload_bytes']} B", ha="center", va="bottom", fontsize=7, color="#333")
    ax.set_xticks(range(len(rows)), names, rotation=20, ha="right", fontsize=8)
    ax.set_ylim(0, 1.2)
    ax.set_ylabel("answer accuracy (value query, K=8)")
    ax.set_title(f"Link comparison (N={N}, d={d}); labels: accuracy, payload bytes (headers excluded)", fontsize=9)
    savefig(fig, "e1_link")


if __name__ == "__main__":
    main()
