"""Joint end-to-end training of a latent pair, plus the text-link and no-link baselines."""
from __future__ import annotations

import math
import os
import time

import torch
import torch.nn.functional as F

from ..config import BITS_CHOICES, CKPT_DIR, DATA_INTENTS, INTENT_SHARE_VALUE, ModelConfig, TaskConfig, TrainConfig, dialect_id, seed_everything
from ..data import TaskData
from .agent import Agent, ckpt_meta, save_ckpt
from .quant import quantize

PAIRS = ((0, 1), (1, 0), (0, 0), (1, 1))  # (sender, receiver); 0 = A, 1 = B


def sample_prefix_mask(g: torch.Generator, B: int, N: int, p_holes: float) -> torch.Tensor:
    """Nested dropout: prefix length p in 1..N biased toward short prefixes, plus
    occasional holes after slot 0 so the receiver also tolerates lost best-effort slots."""
    uni = torch.randint(1, N + 1, (B,), generator=g)
    geo = (1 + torch.floor(torch.log(torch.rand(B, generator=g).clamp(min=1e-9)) / math.log(0.5))).long().clamp(max=N)
    p = torch.where(torch.rand(B, generator=g) < 0.5, uni, geo)
    mask = torch.arange(N)[None] < p[:, None]
    holes = (torch.rand(B, 1, generator=g) < p_holes) & (torch.rand(B, N, generator=g) < 0.25)
    holes[:, 0] = False
    return mask & ~holes


# Training schedule inside the single joint run (fractions of the total steps):
#   [0, pair_only)        only the (A -> B) pair; afterwards all four pairs
#   [0, value_only)       value tables only, warm settings, size curriculum
#   [value_only, warm)    all intents, warm settings
#   [warm, 1]             all intents; trust, hint, prefix dropout, holes, b sampled
# Training all intents from step 0 lets the easy count objective take over the slot
# code before the key-value binding forms (measured; see RESULTS.md training notes).
SCHEDULES = {
    # used for d >= 16: all four pairs from step 0; symmetric dialect, trust gate trained
    "symmetric": {"pair_only": 0.0, "value_only": 0.35, "warm": 0.5},
    # used for d <= 8: symmetric training does not converge within the time budget at small
    # d, so the A -> B dialect forms first (the bandwidth sweep uses only A -> B)
    "ab_first": {"pair_only": 0.35, "value_only": 0.45, "warm": 0.6},
}


def schedule_for(d: int) -> str:
    return "symmetric" if d >= 16 else "ab_first"


def curriculum(step: int, steps: int, K: int):
    """Value/existence table size range: grows from 1-2 entries to K over the first 40%
    of training, then spans 1..K. Breaks the slot-assignment symmetry early on."""
    frac = step / max(1, steps)
    if frac < 0.2:
        hi = 2 + int((K - 2) * frac / 0.2)
        return (1, max(1, min(K, hi)))
    return (max(1, K // 2), K)


def sample_trust(g, B):
    t = torch.rand(B, generator=g)
    return torch.where(torch.rand(B, generator=g) < 0.4, torch.ones(B), t)


def train_latent(d: int, seed: int, steps: int | None = None, log=print, mc: ModelConfig | None = None,
                 tc: TrainConfig | None = None, schedule: str | None = None, zipf: float = 0.0,
                 pos_bias: float | None = None, name_suffix: str = "") -> dict:
    schedule = schedule or schedule_for(d)
    sch = SCHEDULES[schedule]
    tc = tc or TrainConfig(seed=seed)
    if zipf:
        tc.task = TaskConfig(query_zipf=zipf)
    if steps:
        tc.steps = steps
    mc = mc or (ModelConfig(d=d) if pos_bias is None else ModelConfig(d=d, pos_bias_init=pos_bias))
    seed_everything(seed)
    td = TaskData(tc.task)
    agents = [Agent(td, mc), Agent(td, mc)]
    params = [p for a in agents for p in a.parameters()]
    opt = torch.optim.AdamW(params, lr=tc.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=tc.lr, total_steps=tc.steps, pct_start=0.1)
    g = torch.Generator().manual_seed(seed)
    bits_t = torch.tensor(BITS_CHOICES)
    t0 = time.time()
    B = tc.batch
    for step in range(tc.steps):
        td.size_range = curriculum(step, tc.steps, td.task.K)
        frac = step / tc.steps
        warm = frac < sch["warm"]
        intents = DATA_INTENTS if frac >= sch["value_only"] else (INTENT_SHARE_VALUE,)
        batch = td.make_batch(g, B, intents=intents, s=None if warm else "mix",
                              p_hint_absent=tc.p_hint_absent)
        k2, v2, _ = td.make_tables(g, batch.intent)
        a_bad = td.a_tokens(k2, v2, batch.intent)
        # A->B only while the dialect forms, then all four (sender, receiver) pairs
        pair = torch.randint(0, 4 if frac >= sch["pair_only"] else 1, (B,), generator=g)
        snd = torch.tensor([PAIRS[i][0] for i in range(4)])[pair]
        rcv = torch.tensor([PAIRS[i][1] for i in range(4)])[pair]

        good = torch.zeros(B, mc.N, mc.d)
        bad = torch.zeros(B, mc.N, mc.d)
        for who in (0, 1):
            idx = (snd == who).nonzero(as_tuple=True)[0]
            if len(idx) == 0:
                continue
            if warm:  # trust is 1 in the warm phase: mismatched slots are never used
                good = good.index_put((idx,), agents[who].encode(batch.a_tokens[idx], batch.intent[idx]))
                continue
            toks = torch.cat([batch.a_tokens[idx], a_bad[idx]])
            out = agents[who].encode(toks, torch.cat([batch.intent[idx]] * 2))
            good = good.index_put((idx,), out[: len(idx)])
            bad = bad.index_put((idx,), out[len(idx):])

        if warm:  # warm phase: learn the dialect before trust, hint, prefix and quantization noise
            trust = torch.ones(B)
            bits = torch.full((B,), 16)
            mask = torch.ones(B, mc.N, dtype=torch.bool)
        else:
            trust = sample_trust(g, B)
            bits = bits_t[torch.randint(0, len(BITS_CHOICES), (B,), generator=g)]
            mask = sample_prefix_mask(g, B, mc.N, tc.p_holes)
        use_bad = torch.rand(B, generator=g) > trust
        slots = torch.where(use_bad[:, None, None], bad, good)
        slots = quantize(slots, bits[:, None])
        slots = slots * mask[..., None]

        logits = torch.zeros(B, td.ans.size)
        for who in (0, 1):
            idx = (rcv == who).nonzero(as_tuple=True)[0]
            if len(idx) == 0:
                continue
            lg, _ = agents[who].answer(batch.b_tokens[idx], slots[idx], mask[idx], trust[idx])
            logits = logits.index_put((idx,), lg)
        loss = F.cross_entropy(logits, batch.answer)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        if step % 500 == 0 or step == tc.steps - 1:
            acc = (logits.argmax(-1) == batch.answer).float().mean().item()
            log(f"[latent d={d} seed={seed}] step {step} loss {loss.item():.3f} acc {acc:.3f} {time.time() - t0:.0f}s")

    name = f"latent_d{d}_s{seed}" + (f"_zipf{zipf:g}" if zipf else "") + (f"_pb{pos_bias:g}" if pos_bias is not None else "") + name_suffix
    dia = dialect_id(td.task, mc, seed)
    meta = ckpt_meta(td, mc, seed, "latent", dia, True, model_ids={"A": f"A-{dia}", "B": f"B-{dia}"},
                     train_seconds=time.time() - t0, steps=tc.steps, schedule=schedule)
    save_ckpt(os.path.join(CKPT_DIR, name + ".pt"), {"A": agents[0], "B": agents[1]}, meta)
    return {"name": name, "seconds": time.time() - t0}


def baseline_tokens(td: TaskData, batch, kind: str, budget=None):
    if kind == "nolink":
        return batch.b_tokens
    txt = td.text_tokens(batch, td.text_max if budget is None else budget)
    B, P = len(batch), td.vocab.PAD
    sep = torch.tensor([td.vocab.SEP, P]).expand(B, 1, 2)
    bt = torch.stack([batch.b_tokens, torch.full_like(batch.b_tokens, P)], -1)
    return torch.cat([txt, sep, bt], 1)


def train_baseline(kind: str, seed: int = 0, steps: int | None = None, log=print) -> dict:
    """kind: 'text' (B reads the truncated serialised table as prefix) or 'nolink'."""
    tc = TrainConfig(seed=seed)
    if steps:
        tc.steps = steps
    mc = ModelConfig()
    seed_everything(seed)
    td = TaskData(tc.task)
    model = Agent(td, mc, ports=False)
    opt = torch.optim.AdamW(model.parameters(), lr=tc.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=tc.lr, total_steps=tc.steps, pct_start=0.1)
    g = torch.Generator().manual_seed(seed + 77)
    t0 = time.time()
    for step in range(tc.steps):
        # text link baseline: value tables (the comparison in 6.1-6.3); no-link: all intents
        intents = (INTENT_SHARE_VALUE,) if kind == "text" else DATA_INTENTS
        warm = step < 0.4 * tc.steps  # no hint at first, so the link is learned before the hint
        batch = td.make_batch(g, tc.batch, intents=intents, s=None if warm and kind == "text" else "mix",
                              p_hint_absent=tc.p_hint_absent)
        budget = None
        if kind == "text":
            full = torch.full((tc.batch,), td.text_max)
            budget = torch.where(torch.rand(tc.batch, generator=g) < 0.3, full,
                                 torch.randint(0, td.text_max + 1, (tc.batch,), generator=g))
        logits, _ = model.answer(baseline_tokens(td, batch, kind, budget))
        loss = F.cross_entropy(logits, batch.answer)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 500 == 0 or step == tc.steps - 1:
            acc = (logits.argmax(-1) == batch.answer).float().mean().item()
            log(f"[{kind}] step {step} loss {loss.item():.3f} acc {acc:.3f} {time.time() - t0:.0f}s")
    meta = ckpt_meta(td, mc, seed, kind, f"{kind}-{seed}", False, train_seconds=time.time() - t0, steps=tc.steps)
    save_ckpt(os.path.join(CKPT_DIR, f"{kind}.pt"), {"B": model}, meta)
    return {"name": kind, "seconds": time.time() - t0}
