"""6.8 Bits under loss: the latent exchange of 6.7 against text sent as one zstd-compressed message.

Latent: N = 8, d = 32, b = 16, TLP over UDP with P = 4 protected slots (the same exchange as e7).
Text:   the serialised table ("k3=19 k7=42 ...", the text link of 6.1), zstd level 19, sent as one
        keyed DATA packet (send_bytes, slot_dim = 0) with the TCP-like discipline: retransmitted
        until acknowledged. B is the text-baseline model of 6.1; it reads the decompressed text.

Loss in {0, 5, 10, 20, 30}%, 20 ms one-way delay, identical seeded loss in both directions through
the injector, 1000 messages per cell. Per message: answer accuracy, bytes on the wire in both
directions (headers, AEAD tags, ACKs and retransmissions), time to the first answer, time to the
final answer, and the bytes on the wire up to the first correct answer."""
import threading
import time

import numpy as np
import torch

try:
    from compression import zstd as _zstd          # Python 3.14+

    def zcompress(b: bytes) -> bytes:
        return _zstd.compress(b, level=ZSTD_LEVEL)

    def zdecompress(b: bytes) -> bytes:
        return _zstd.decompress(b)
except ImportError:   # pragma: no cover - older Pythons need the zstandard package
    import zstandard as _zstd

    def zcompress(b: bytes) -> bytes:
        return _zstd.ZstdCompressor(level=ZSTD_LEVEL).compress(b)

    def zdecompress(b: bytes) -> bytes:
        return _zstd.ZstdDecompressor().decompress(b)

from ..models.evaluate import send
from ..negotiate.anchors import model_capabilities
from ..tlp.session import TLPOptions
from ..tlp.transport import Injector
from .common import COLORS, DEFAULT, MARKERS, TRANSPORT, figure, models, save, savefig, session_pair

LOSS = (0.0, 0.05, 0.10, 0.20, 0.30)
DELAY = 0.020
ZSTD_LEVEL = 19
N_MSGS = 1000
LABELS = {"latent": "latent, TLP over UDP (P = 4)", "text": "text, zstd, one TCP-like message"}


def text_tokens_from(td, text: str, b_tokens_row: torch.Tensor) -> torch.Tensor:
    """Tokens for the text-baseline B from received text: each complete "kX=VV" field is one
    (key, value) token, as in TaskData.text_tokens, then SEP, then B's own query tokens."""
    v, K = td.vocab, td.task.K
    toks = []
    for f in text.split():
        k, _, x = f.partition("=")
        if k.startswith("k") and k[1:].isdigit() and x.isdigit():
            toks.append((v.KEY0 + int(k[1:]), v.VAL0 + int(x)))
    txt = torch.full((K, 2), v.PAD, dtype=torch.long)
    toks = toks[:K]
    if toks:
        txt[K - len(toks):] = torch.tensor(toks)
    sep = torch.tensor([[v.SEP, v.PAD]])
    bt = torch.stack([b_tokens_row, torch.full_like(b_tokens_row, v.PAD)], -1)
    return torch.cat([txt, sep, bt], 0)[None]


def run_cell(link: str, loss: float, n_msgs: int, ev, payloads, B, td, seed: int):
    rel = "tlp" if link == "latent" else "tcp"
    opts = TLPOptions(rto=0.10, ack_delay=0.03, grace=0.06, reliability=rel, tcp_max_retx=50, anchor_threshold=0.0)
    caps_a, caps_b = model_capabilities(DEFAULT, "A"), model_capabilities(DEFAULT, "B")
    s, c, sc, cc = session_pair(caps_a, caps_b, opts=opts)
    s.transport.injector = Injector(loss=loss, delay=DELAY, seed=seed)
    c.transport.injector = Injector(loss=loss, delay=DELAY, seed=seed + 1)
    sent_at, done = {}, threading.Event()
    rows = []

    def wire():
        return sc.session.stats.bytes_sent + cc.session.stats.bytes_sent

    def sender():
        for i in range(n_msgs):
            done.clear()
            sent_at[i] = time.monotonic()
            if link == "latent":
                sc.send_latent(payloads[i], protected=4)
            else:
                sc.send_bytes(payloads[i])
            done.wait(15)

    th = threading.Thread(target=sender)
    b0 = wire()
    th.start()
    for i in range(n_msgs):
        r0 = sc.session.stats.retransmits
        truth = int(ev.answer[i])
        first = final = first_ok_bytes = None
        ans = -1
        try:
            if link == "latent":
                for slots, mask, fin in cc.recv_latent(deadline=0.12, timeout=5.0):
                    with torch.no_grad():
                        lg, _ = B.answer(ev.b_tokens[i: i + 1], slots[None], mask[None], torch.ones(1))
                    ans = int(lg.argmax(-1))
                    now = time.monotonic()
                    first = first if first is not None else now - sent_at[i]
                    if ans == truth and first_ok_bytes is None:
                        first_ok_bytes = wire() - b0
                    if fin:
                        final = now - sent_at[i]
            else:
                blob = cc.recv_bytes(timeout=15.0)
                text = zdecompress(blob).decode() if blob is not None else ""
                with torch.no_grad():
                    lg, _ = B.answer(text_tokens_from(td, text, ev.b_tokens[i]))
                ans = int(lg.argmax(-1))
                first = final = time.monotonic() - sent_at[i]
                if ans == truth:
                    first_ok_bytes = wire() - b0
        except TimeoutError:
            pass
        time.sleep(0.01)   # let late ACKs and retransmits settle into this message's byte count
        b1 = wire()
        rows.append({"acc": float(ans == truth), "t_first": first, "t_final": final, "bytes": b1 - b0,
                     "bytes_to_first_correct": first_ok_bytes, "retx": sc.session.stats.retransmits - r0})
        b0 = b1
        done.set()
    th.join(20)
    s.close()
    c.close()
    return summarise(link, loss, rows)


def summarise(link, loss, rows):
    def stat(k):
        v = np.array([r[k] for r in rows if r[k] is not None], dtype=float)
        if not len(v):
            return None, None
        return float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0

    acc, acc_se = stat("acc")
    tf, tf_se = stat("t_first")
    tl, tl_se = stat("t_final")
    by, by_se = stat("bytes")
    bfc, bfc_se = stat("bytes_to_first_correct")
    finals = [r["t_final"] for r in rows if r["t_final"] is not None]
    n_ok = sum(r["acc"] for r in rows)
    return {"link": link, "loss": loss, "n": len(rows), "acc": acc, "acc_se": acc_se,
            "t_first_ms": 1e3 * tf, "t_first_se_ms": 1e3 * tf_se, "t_final_ms": 1e3 * tl, "t_final_se_ms": 1e3 * tl_se,
            "t_final_p95_ms": 1e3 * float(np.percentile(finals, 95)),
            "bytes": by, "bytes_se": by_se, "retransmits": stat("retx")[0],
            "bytes_to_first_correct": bfc, "bytes_to_first_correct_se": bfc_se,
            "never_correct": sum(r["bytes_to_first_correct"] is None for r in rows),
            "bytes_per_correct_answer": sum(r["bytes"] for r in rows) / n_ok if n_ok else None}


def main(n_msgs: int = N_MSGS):
    m, _, td = models(DEFAULT)
    text_m, _, _ = models("text")
    ev = td.eval_set(n_msgs)
    slots_all, _ = send(m["A"], ev)
    raw = [td.serialise(k.tolist(), v.tolist()).encode() for k, v in zip(ev.keys, ev.vals)]
    blobs = [zcompress(r) for r in raw]
    assert all(zdecompress(b) == r for b, r in zip(blobs, raw))
    out = []
    for loss in LOSS:
        for k, link in enumerate(("latent", "text")):
            t0 = time.time()
            payloads = slots_all if link == "latent" else blobs
            B = m["B"] if link == "latent" else text_m["B"]
            r = run_cell(link, loss, n_msgs, ev, payloads, B, td, seed=8000 + int(loss * 1000) + 10 * k)
            r["wall_s"] = time.time() - t0
            out.append(r)
            print(f"loss {loss:4.2f} {link:<6}: acc {r['acc']:.3f} first {r['t_first_ms']:.0f} ms final "
                  f"{r['t_final_ms']:.0f} ms bytes {r['bytes']:.0f} to-first-correct {r['bytes_to_first_correct']:.0f} "
                  f"retx {r['retransmits']:.2f} ({r['wall_s']:.0f} s)", flush=True)
    res = {"checkpoint": DEFAULT, "text_checkpoint": "text", "transport": TRANSPORT, "delay_ms": DELAY * 1e3,
           "loss": LOSS, "n_msgs": n_msgs, "zstd_level": ZSTD_LEVEL,
           "text_bytes_raw_mean": float(np.mean([len(r) for r in raw])),
           "text_bytes_zstd_mean": float(np.mean([len(b) for b in blobs])),
           "labels": LABELS, "rows": out,
           "note": "bytes = both directions on the wire per message: TLP headers, AEAD tags, payload, ACKs and "
                   "retransmissions. bytes_to_first_correct = wire bytes from the start of the message to the first "
                   "answer that is correct (messages never answered correctly are excluded and counted in "
                   "never_correct). text = one keyed DATA packet carrying the zstd frame, TCP-like reliability."}
    save("e8_bits_under_loss", res)
    plot(res)
    return res


def plot(res):
    fig, ax = figure(6.4, 4.0)
    for link in ("latent", "text"):
        r = [x for x in res["rows"] if x["link"] == link]
        L = [100 * x["loss"] for x in r]
        y = [x["bytes_to_first_correct"] for x in r]
        e = [x["bytes_to_first_correct_se"] for x in r]
        col = COLORS["tlp" if link == "latent" else "text"]
        ax.errorbar(L, y, yerr=e, color=col, marker=MARKERS["tlp" if link == "latent" else "text"], lw=2, ms=6,
                    capsize=3, label=LABELS[link])
    ax.set_xlabel("packet loss (%)")
    ax.set_ylabel("bytes on the wire to the first correct answer")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8, frameon=False)
    ax.set_title(f"Bits under loss ({res['n_msgs']} messages per cell, {res['delay_ms']:.0f} ms delay)", fontsize=9)
    savefig(fig, "e8_bits_under_loss")


if __name__ == "__main__":
    import sys
    main(int(sys.argv[1]) if len(sys.argv) > 1 else N_MSGS)
