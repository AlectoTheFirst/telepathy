"""Port training for Telepathy-1 (brief section 6).

Stage 1 (capacity): send port on the frozen sender, receive port on the frozen receiver, trained on
answer-token likelihood. Quantization-aware: b is drawn per passage from BITS. Missing slots: each
slot is dropped independently with probability SLOT_DROP so that the receive port's mask path is
trained (this is not nested dropout; there is no order). Trust: t ~ U(0, 1) per question, and with
probability 1 - t the slots are replaced by those of a different passage; a hint (a noisy copy of
the answer, correct with probability s) is present for HINT_P of the questions.

Stage 2 (ordering, G3): fine-tunes the send port only, with nested dropout (a prefix of k slots,
k uniform in 1..N) under the Zipf question distribution.

Intent: SHARE_FACTS trains on all of a passage's questions; SHARE_IDENTITY only on IDENTITY_KEYS.
No intent signal enters the slot path.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F

from . import data as D
from .cache import StateCache, cache_dir
from .models import FrozenLM
from .ports import ReceivePort, SendPort

ROOT = Path(__file__).resolve().parents[2]
CKPT = ROOT / "checkpoints" / "t1"
BITS = (16, 8, 4)
SLOT_DROP = 0.1
HINT_P = 0.5
AUX_TEMP = 0.1


class AuxHeads(torch.nn.Module):
    """Training-only contrastive heads (not part of any port, never on the wire). The send port's
    message must pick out its own passage's questions among the batch: questions are embedded as the
    mean of the receiver's input embeddings of their tokens. Without this, the receive port learns to
    use the slots as a fixed memory before the send port carries passage content, and the true-slot and
    swapped-slot losses stay equal (diagnostic of 2026-11-01, RESULTS.md training notes)."""

    def __init__(self, d_recv: int, N: int, d: int, dim: int = 256):
        super().__init__()
        self.q = torch.nn.Linear(d_recv, dim)
        self.s = torch.nn.Linear(N * d, dim)

    def forward(self, qv, slots, target):
        a = F.normalize(self.q(qv), dim=-1)
        b = F.normalize(self.s(slots.flatten(1)), dim=-1)
        return F.cross_entropy(a @ b.T / AUX_TEMP, target)


@dataclass
class TrainConfig:
    name: str
    sender: str = "Qwen/Qwen3-0.6B-Base"
    receiver: str = "Qwen/Qwen3-0.6B-Base"
    L_s: int = 0            # 0: two-thirds depth of the sender
    L_r: int = 0            # 0: one-half depth of the receiver
    N: int = 8
    d: int = 64
    width: int = 512
    intent: str = "facts"   # facts | identity
    dist: str = "uniform"
    stage: int = 1
    init_from: str = ""     # stage 2: the stage 1 checkpoint name
    seed: int = 1
    passages: int = 8       # passages per step
    questions: int = 4      # questions per passage per step
    lr: float = 5e-4        # receive port
    lr_send: float = 1e-3   # send port and the auxiliary heads
    aux: float = 0.5        # weight of the contrastive term (stage 1 only)
    warmup: int = 200
    max_steps: int = 20000
    max_minutes: float = 170.0
    eval_every: int = 1000
    ckpt_every_min: float = 15.0
    adversary: dict = field(default_factory=dict)   # E4: {"probe": ckpt path, "lam": float}
    cache: int = 1          # read sender states from the disk cache (cache.py) when it exists


def prompt(q: str, hint: str | None = None) -> str:
    return (f"Hint: {hint}\n" if hint is not None else "") + f"Question: {q}\nAnswer:"


def resolve_layers(cfg: TrainConfig, s_lm, r_lm):
    if not cfg.L_s:
        cfg.L_s = round(2 * s_lm.n_layers / 3)
    if not cfg.L_r:
        cfg.L_r = r_lm.n_layers // 2
    return cfg


class Pair:
    """A sender LM + send port and a receiver LM + receive port."""

    def __init__(self, cfg: TrainConfig, s_lm: FrozenLM, r_lm: FrozenLM):
        self.cfg = resolve_layers(cfg, s_lm, r_lm)
        self.s_lm, self.r_lm = s_lm, r_lm
        dev = r_lm.dev
        self.send = SendPort(s_lm.hidden, cfg.N, cfg.d, cfg.width).to(dev)
        self.recv = ReceivePort(r_lm.hidden, cfg.N, cfg.d).to(dev)

    def encode(self, passages, bits=16, send=None):
        h, m = self.s_lm.states(passages, self.cfg.L_s)
        return (send or self.send)(h, m, bits)

    def nll(self, slots, mask, trust, prompts, answers, reduction="mean"):
        """Answer-token NLL with the receive port inserted. slots [B, N, d] (one row per prompt)."""
        ids, am, pos, tgt = self.r_lm.build(prompts, answers)
        mem, vis = self.recv.memory(slots, mask)
        with self.r_lm.inserted(self.cfg.L_r, lambda x: self.recv(x, mem, vis, trust)):
            logits = self.r_lm.answer_logits(ids, am, pos)
        loss = F.cross_entropy(logits, tgt, reduction="none")
        if reduction == "mean":
            return loss.mean()
        out = torch.zeros(len(prompts), device=loss.device).index_add_(0, pos[0], loss)
        return out

    @torch.no_grad()
    def answer(self, slots, mask, prompts, trust=None, max_new_tokens=8):
        B = len(prompts)
        trust = torch.ones(B, device=self.r_lm.dev) if trust is None else trust
        if slots is None:
            return self.r_lm.generate(prompts, max_new_tokens)
        mem, vis = self.recv.memory(slots, mask)
        with self.r_lm.inserted(self.cfg.L_r, lambda x: self.recv(x, mem, vis, trust)):
            return self.r_lm.generate(prompts, max_new_tokens)

    def state(self):
        return {"cfg": asdict(self.cfg), "send": self.send.state_dict(), "recv": self.recv.state_dict()}

    def save(self, path: Path, extra=None):
        path.mkdir(parents=True, exist_ok=True)
        torch.save({**self.state(), **(extra or {})}, path / "ports.pt")

    def load(self, path: Path):
        st = torch.load(path / "ports.pt", map_location=self.r_lm.dev, weights_only=False)
        self.send.load_state_dict(st["send"])
        self.recv.load_state_dict(st["recv"])
        return st


def load_pair(name: str, lms: dict | None = None) -> Pair:
    """Load a trained pair by checkpoint name. lms: repo -> FrozenLM cache (shared between pairs)."""
    st = torch.load(CKPT / name / "ports.pt", map_location="cpu", weights_only=False)
    cfg = TrainConfig(**st["cfg"])
    lms = {} if lms is None else lms
    for r in (cfg.sender, cfg.receiver):
        if r not in lms:
            lms[r] = FrozenLM(r)
    p = Pair(cfg, lms[cfg.sender], lms[cfg.receiver])
    p.load(CKPT / name)
    p.send.eval()
    p.recv.eval()
    return p


# ----------------------------------------------------------------------------- evaluation

@torch.no_grad()
def evaluate(pair: Pair, items, batch=16, bits=16, mask_fn=None, slots_fn=None, trust=1.0, hints=None):
    """EM on eval items. slots_fn(passages) -> slots overrides the pair's own sender (translators).
    mask_fn(B, N) -> mask overrides the full mask. Returns (accuracy, list of predictions)."""
    preds, ok = [], []
    for i in range(0, len(items), batch):
        chunk = items[i: i + batch]
        ps = [c["passage"] for c in chunk]
        slots = slots_fn(ps) if slots_fn else pair.encode(ps, bits)
        B, N = slots.shape[:2]
        mask = mask_fn(B, N) if mask_fn else torch.ones(B, N, dtype=torch.bool, device=slots.device)
        hs = hints[i: i + batch] if hints else [None] * len(chunk)
        out = pair.answer(slots, mask, [prompt(c["question"], h) for c, h in zip(chunk, hs)],
                          torch.full((B,), float(trust), device=slots.device))
        for c, o in zip(chunk, out):
            preds.append(o)
            ok.append(D.exact_match(o, c["answer"]))
    return sum(ok) / len(ok), preds


# ----------------------------------------------------------------------------- training

def _draw_questions(rng, p, n, cfg):
    qs = p["questions"]
    if cfg.intent == "identity":
        qs = [q for q in qs if q["key"] in D.IDENTITY_KEYS]
    return rng.choices(qs, weights=D.question_weights(qs, cfg.dist), k=n)


def train(cfg: TrainConfig, lms: dict | None = None, log=print):
    torch.manual_seed(cfg.seed)
    torch.set_num_threads(4)
    rng = random.Random(cfg.seed)
    lms = {} if lms is None else lms
    for r in (cfg.sender, cfg.receiver):
        if r not in lms:
            lms[r] = FrozenLM(r)
    pair = Pair(cfg, lms[cfg.sender], lms[cfg.receiver])
    out = CKPT / cfg.name
    dev = pair.r_lm.dev
    train_set, val = D.load("train"), D.load("val")
    states = None
    if cfg.cache and cache_dir(pair.s_lm.short, cfg.L_s).joinpath("index.json").exists():
        states = StateCache(pair.s_lm.short, cfg.L_s, dev)
        train_set = [p for p in train_set if p["id"] in states]
        log(f"sender states from cache: {len(train_set)} passages")
    val_items = D.eval_items(val[:200], cfg.dist)
    if cfg.intent == "identity":
        val_items = [dict(it, **q) for it, p in zip(val_items, val[:200])
                     for q in p["questions"] if q["key"] == "name"]
    pool = {}
    for p in train_set:
        for q in p["questions"]:
            pool.setdefault(q["key"], []).append(q["answer"])

    aux = AuxHeads(pair.r_lm.hidden, cfg.N, cfg.d).to(dev)
    emb = pair.r_lm.model.get_input_embeddings().weight

    def qvec(questions):
        out = []
        for q in questions:
            ids = torch.tensor(pair.r_lm.tok(q, add_special_tokens=False).input_ids, device=dev)
            out.append(emb[ids].float().mean(0))
        return torch.stack(out)

    if cfg.stage == 2:
        pair.load(CKPT / cfg.init_from)
        for p_ in pair.recv.parameters():
            p_.requires_grad_(False)
        groups = [{"params": list(pair.send.parameters()), "scale": cfg.lr_send / cfg.lr}]
    else:
        groups = [{"params": list(pair.send.parameters()) + list(aux.parameters()), "scale": cfg.lr_send / cfg.lr},
                  {"params": list(pair.recv.parameters()), "scale": 1.0}]
    params = [q for g in groups for q in g["params"]]
    opt = torch.optim.AdamW(groups, lr=cfg.lr, weight_decay=0.01)
    # fixed diagnostic: 64 val questions, NLL with the true and with swapped slots
    dval = val[200:216]
    dq = [(i, q) for i, p in enumerate(dval) for q in p["questions"][:4]]
    didx = torch.tensor([i for i, _ in dq], device=dev)
    dh, dm = pair.s_lm.states([p["passage"] for p in dval], cfg.L_s)

    @torch.no_grad()
    def diag():
        sl = pair.send(dh, dm, 16)
        f = torch.ones(len(dq), cfg.N, dtype=torch.bool, device=dev)
        o = torch.ones(len(dq), device=dev)
        pr, an = [prompt(q["question"]) for _, q in dq], [q["answer"] for _, q in dq]
        return (pair.nll(sl[didx], f, o, pr, an).item(), pair.nll(sl[(didx + 1) % len(dval)], f, o, pr, an).item())
    step, elapsed, hist = 0, 0.0, []
    if (out / "ports.pt").exists():
        st = pair.load(out)
        if "opt" in st:
            opt.load_state_dict(st["opt"])
        if "aux" in st:
            aux.load_state_dict(st["aux"])
        step, elapsed, hist = st.get("step", 0), st.get("elapsed", 0.0), st.get("hist", [])
        rng.setstate(st["rng"]) if "rng" in st else None
        log(f"resumed {cfg.name} at step {step} ({elapsed / 60:.1f} min)")
    adv = None
    if cfg.adversary:
        from .probe import load_probe
        adv = load_probe(cfg.adversary["probe"], dev)

    def lr_at(s):
        if s < cfg.warmup:
            return cfg.lr * (s + 1) / cfg.warmup
        return cfg.lr * 0.5 * (1 + math.cos(math.pi * min(1.0, s / cfg.max_steps)))

    t0, last_ck, run_loss = time.time() - elapsed, time.time(), []
    pair.send.train()
    pair.recv.train(cfg.stage == 1)
    while step < cfg.max_steps and (time.time() - t0) / 60 < cfg.max_minutes:
        ps = rng.sample(train_set, cfg.passages)
        bits = torch.tensor([rng.choice(BITS) for _ in ps], device=dev)
        if states is not None:
            h, m = states.get([p["id"] for p in ps])
            slots = pair.send(h, m, bits)
        else:
            slots = pair.encode([p["passage"] for p in ps], bits)
        P, N = slots.shape[:2]
        Q = cfg.questions
        idx = torch.arange(P, device=dev).repeat_interleave(Q)
        trust = torch.rand(P * Q, device=dev)
        swap = torch.rand(P * Q, device=dev) > trust
        other = (idx + torch.randint(1, P, (P * Q,), device=dev)) % P
        src = torch.where(swap, other, idx)
        row_slots = slots[src]
        if cfg.stage == 1:
            mask = torch.rand(P * Q, N, device=dev) > SLOT_DROP
        else:
            k = torch.randint(1, N + 1, (P * Q, 1), device=dev)
            mask = torch.arange(N, device=dev)[None] < k
        prompts, answers, questions = [], [], []
        for p in ps:
            for q in _draw_questions(rng, p, Q, cfg):
                hint = None
                if cfg.stage == 1 and rng.random() < HINT_P:
                    s = rng.random()
                    hint = q["answer"] if rng.random() < s else rng.choice(pool[q["key"]])
                prompts.append(prompt(q["question"], hint))
                answers.append(q["answer"])
                questions.append(q["question"])
        loss = pair.nll(row_slots, mask, trust, prompts, answers)
        if cfg.stage == 1 and cfg.aux:
            loss = loss + cfg.aux * aux(qvec(questions), slots, idx)
        if adv is not None:
            # E4: push the frozen probe's confidence in SHARE_FACTS down on the sender's own slots
            p_facts = adv.prob_facts(slots)
            loss = loss + cfg.adversary["lam"] * p_facts.mean()
        for g in opt.param_groups:
            g["lr"] = lr_at(step) * g["scale"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if dev.type == "mps":
            torch.mps.empty_cache()   # variable sequence lengths otherwise grow the MPS cache into swap
        step += 1
        run_loss.append(float(loss.detach()))
        if step % 50 == 0:
            el = time.time() - t0
            log(f"{cfg.name} step {step} loss {sum(run_loss) / len(run_loss):.3f} "
                f"alpha {pair.recv.alpha.detach().tanh().tolist()} {el / 60:.1f} min ({el / step:.2f} s/step)")
            hist.append({"step": step, "loss": sum(run_loss) / len(run_loss), "min": el / 60})
            run_loss = []
        if step % 250 == 0:
            pair.send.eval()
            lt, ls = diag()
            pair.send.train()
            hist.append({"step": step, "diag_true": lt, "diag_swapped": ls})
            log(f"{cfg.name} step {step} diag nll true {lt:.3f} swapped {ls:.3f} gap {ls - lt:.3f}")
        if step % cfg.eval_every == 0:
            pair.send.eval(); pair.recv.eval()
            acc, _ = evaluate(pair, val_items[:200])
            pair.send.train(); pair.recv.train(cfg.stage == 1)
            hist.append({"step": step, "val_acc": acc, "min": (time.time() - t0) / 60})
            log(f"{cfg.name} step {step} val acc {acc:.3f}")
        if (time.time() - last_ck) / 60 > cfg.ckpt_every_min:
            pair.save(out, {"aux": aux.state_dict(), "opt": opt.state_dict(), "step": step, "elapsed": time.time() - t0,
                            "hist": hist, "rng": rng.getstate()})
            last_ck = time.time()
    elapsed = time.time() - t0
    pair.send.eval(); pair.recv.eval()
    acc, _ = evaluate(pair, val_items[:200])
    hist.append({"step": step, "val_acc": acc, "min": elapsed / 60})
    pair.save(out, {"aux": aux.state_dict(), "opt": opt.state_dict(), "step": step, "elapsed": elapsed, "hist": hist, "rng": rng.getstate()})
    summary = {"name": cfg.name, "cfg": asdict(cfg), "steps": step, "train_minutes": elapsed / 60,
               "val_acc_200": acc, "device": str(dev)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    log(f"{cfg.name} done: {step} steps, {elapsed / 60:.1f} min, val acc {acc:.3f}")
    return pair, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    for f_, v in TrainConfig.__dataclass_fields__.items():
        if f_ in ("name", "adversary"):
            continue
        ap.add_argument(f"--{f_.replace('_', '-')}", type=type(v.default), default=v.default)
    a = vars(ap.parse_args())
    cfg = TrainConfig(**{k: v for k, v in a.items()})
    train(cfg, log=lambda s: print(s, flush=True))


if __name__ == "__main__":
    main()
