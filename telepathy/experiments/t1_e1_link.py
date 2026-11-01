"""Telepathy-1 E1: link comparison (G2), Qwen -> Qwen.

Baselines on the fixed 1000-item test set (uniform questions), same frozen receiver:
  none, oracle sentence, passage cut, passage zstd, at budgets of {64, ..., 2048} real bits;
  latent over the pair's ports at every trained (N, d) and b in {16, 8, 4}.

Per point: accuracy (EM), real payload bits (payload bytes x 8, including the float16 per-slot
scales for b < 16), packets, header bytes and AEAD tag bytes as separate columns (TLP over UDP, slot
batching M = 4 for latent; text as one keyed DATA packet per 1342 payload bytes), tokens generated
by the sender, tokens read by the receiver, and compute latency: sender encode and receiver answer
on MPS. Transfer time over TLP over UDP is measured in t1_e6 (loss 0 row) and added there.

Usage: python -m telepathy.experiments.t1_e1_link [pair names...]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from ..lm import data as D
from ..lm import textlink as TL
from ..lm.models import FrozenLM, QWEN
from ..lm.train import load_pair, prompt
from ..models.quant import slot_payload_len

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
BUDGETS = (64, 128, 256, 512, 1024, 2048)
BITS = (16, 8, 4)
HEADER_BYTES = 28      # TLP header (SPEC.md section 3)
TAG_BYTES = 16         # AES-GCM tag per keyed packet
MAX_PAYLOAD = 1342     # 1400-byte packets minus header, tag and UDP framing (SPEC.md section 13)
BATCH_M = 4


def sync(dev):
    if dev.type == "mps":
        torch.mps.synchronize()


def mean_se(x):
    x = np.asarray(x, float)
    return float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x)))


@torch.no_grad()
def text_points(lm: FrozenLM, items, batch=16):
    rows = []
    kinds = [("none", None)] + [(k, b) for k in ("oracle", "cut", "zstd") for b in BUDGETS]
    for kind, budget in kinds:
        ok, wire, toks, t_send, t_recv = [], [], [], 0.0, 0.0
        for i in range(0, len(items), batch):
            chunk = items[i: i + batch]
            t0 = time.perf_counter()
            pl = [TL.payload(kind, it, budget or 0) for it in chunk]
            t_send += time.perf_counter() - t0
            prompts = [TL.text_prompt(ctx, it["question"]) for (_, ctx), it in zip(pl, chunk)]
            sync(lm.dev)
            t0 = time.perf_counter()
            out = lm.generate(prompts)
            sync(lm.dev)
            t_recv += time.perf_counter() - t0
            for (w, ctx), it, o in zip(pl, chunk, out):
                ok.append(D.exact_match(o, it["answer"]))
                wire.append(len(w))
                toks.append(len(lm.tok(ctx, add_special_tokens=False).input_ids))
        a, se = mean_se(ok)
        nb = float(np.mean(wire))
        pk = int(np.ceil(nb / MAX_PAYLOAD)) if nb else 0
        rows.append({"link": kind, "budget_bits": budget, "acc": a, "acc_se": se, "payload_bits": 8 * nb,
                     "packets": pk, "header_bytes": pk * HEADER_BYTES, "tag_bytes": pk * TAG_BYTES,
                     "sender_tokens_generated": 0, "receiver_context_tokens": float(np.mean(toks)),
                     "sender_ms": 1e3 * t_send / len(items), "receiver_ms": 1e3 * t_recv / len(items) * batch})
        print(f"{kind:<7} {str(budget):>5} bits: acc {a:.3f} payload {8 * nb:.0f} bits, ctx {np.mean(toks):.0f} tokens", flush=True)
    return rows


@torch.no_grad()
def latent_points(pair, items, batch=16):
    rows = []
    N, d = pair.cfg.N, pair.cfg.d
    dev = pair.r_lm.dev
    for bits in BITS:
        ok, t_send, t_recv = [], 0.0, 0.0
        for i in range(0, len(items), batch):
            chunk = items[i: i + batch]
            sync(dev)
            t0 = time.perf_counter()
            slots = pair.encode([c["passage"] for c in chunk], bits)
            sync(dev)
            t_send += time.perf_counter() - t0
            mask = torch.ones(slots.shape[:2], dtype=torch.bool, device=dev)
            t0 = time.perf_counter()
            out = pair.answer(slots, mask, [prompt(c["question"]) for c in chunk])
            sync(dev)
            t_recv += time.perf_counter() - t0
            ok += [D.exact_match(o, c["answer"]) for o, c in zip(out, chunk)]
        a, se = mean_se(ok)
        nbytes = N * slot_payload_len(d, bits)
        pk = int(np.ceil(N / BATCH_M))
        rows.append({"link": "latent", "pair": pair.cfg.name, "N": N, "d": d, "b": bits, "acc": a, "acc_se": se,
                     "payload_bits": 8 * nbytes, "nominal_bits": N * d * bits, "packets": pk,
                     "header_bytes": pk * HEADER_BYTES, "tag_bytes": pk * TAG_BYTES,
                     "sender_tokens_generated": 0, "receiver_context_tokens": 0, "receiver_slots": N,
                     "sender_ms": 1e3 * t_send / len(items) * batch, "receiver_ms": 1e3 * t_recv / len(items) * batch})
        print(f"latent {pair.cfg.name} N={N} d={d} b={bits}: acc {a:.3f} payload {8 * nbytes} bits", flush=True)
    return rows


def main(pairs=("qwen_s1",), text=True):
    items = D.eval_items(D.load("test"), "uniform")
    res_path = RESULTS / "t1_e1_link.json"
    res = json.loads(res_path.read_text()) if res_path.exists() else {"rows": []}
    lms = {}
    if text and not any(r["link"] != "latent" for r in res["rows"]):
        lms[QWEN] = FrozenLM(QWEN)
        res["rows"] += text_points(lms[QWEN], items)
    for name in pairs:
        res["rows"] = [r for r in res["rows"] if r.get("pair") != name]
        res["rows"] += latent_points(load_pair(name, lms), items)
        res_path.write_text(json.dumps(res, indent=2))
    res.update({"test_items": len(items), "budgets_bits": BUDGETS, "receiver": QWEN,
                "note": "latency columns are per item at batch 16 on MPS (receiver_ms is per batch-of-16 call "
                        "divided per item x 16, i.e. the time of one call). payload_bits are real: slot bytes "
                        "including float16 scales, or text bytes on the wire."})
    res_path.write_text(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    main(tuple(sys.argv[1:]) or ("qwen_s1",))
