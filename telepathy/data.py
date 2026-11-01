"""Split-knowledge task: tables for A, queries and hints for B, anchors, eval sets."""
from __future__ import annotations

from dataclasses import dataclass, fields

import torch

from .config import (INTENT_SHARE_ENTRY_COUNT, INTENT_SHARE_KEY_EXISTENCE, INTENT_SHARE_VALUE,
                     Answers, TaskConfig, Vocab)

ANCHOR_FIT_SEED = 1001
ANCHOR_HELDOUT_SEED = 1002
EVAL_SEED = 999


@dataclass
class Batch:
    intent: torch.Tensor      # [B] declared/true purpose given to the sender
    keys: torch.Tensor        # [B, T] key ids, -1 = no entry
    vals: torch.Tensor        # [B, T]
    size: torch.Tensor        # [B]
    a_tokens: torch.Tensor    # [B, La] sender input (purpose + table)
    qkey: torch.Tensor        # [B] queried key, -1 for count queries
    answer: torch.Tensor      # [B] answer id
    hint: torch.Tensor        # [B] hinted value, -1 = absent
    b_tokens: torch.Tensor    # [B, Lb] receiver input (hint + query), left padded

    def __len__(self):
        return self.intent.shape[0]

    def index(self, idx) -> "Batch":
        return Batch(**{f.name: getattr(self, f.name)[idx] for f in fields(self)})

    @staticmethod
    def cat(batches: list["Batch"]) -> "Batch":
        return Batch(**{f.name: torch.cat([getattr(b, f.name) for b in batches]) for f in fields(Batch)})


class TaskData:
    def __init__(self, task: TaskConfig | None = None):
        self.task = task or TaskConfig()
        self.vocab = Vocab(self.task)
        self.ans = Answers(self.task)
        t = self.task
        self.T = t.table_max
        self.nd = t.n_digits
        self.La = 1 + self.T      # positions; each sender position has 2 channels (key, value)
        self.Lb = 2 + 3
        # text link: "kXX=VV" per entry plus separators
        self.size_range = None  # training curriculum: (lo, hi) table size for value/existence
        # Text link is used on value tables (K entries): "kXX=VV" + separator per entry.
        self.text_max = t.K * (len(f"k{t.key_range - 1}") + 1 + self.nd + 1) - 1
        # key popularity for Zipf queries: rank r(k) from a fixed permutation, weight 1/(r+1)^s
        rank = torch.randperm(t.key_range, generator=torch.Generator().manual_seed(t.popularity_seed))
        self.key_rank = rank
        self.key_weight = (1.0 / (rank.float() + 1) ** t.query_zipf) if t.query_zipf else torch.ones(t.key_range)

    # ------------------------------------------------------------------ tables
    def make_tables(self, g: torch.Generator, intent: torch.Tensor):
        t, B = self.task, intent.shape[0]
        perm = torch.rand(B, t.key_range, generator=g).argsort(1)[:, : self.T]
        vals = torch.randint(0, t.value_range, (B, self.T), generator=g)
        size = torch.full((B,), t.K, dtype=torch.long)
        if self.size_range is not None:
            lo, hi = self.size_range
            size = torch.randint(lo, hi + 1, (B,), generator=g)
        cnt = torch.randint(1, t.count_max + 1, (B,), generator=g)
        size = torch.where(intent == INTENT_SHARE_ENTRY_COUNT, cnt, size)
        present = torch.arange(self.T)[None, :] < size[:, None]
        keys = torch.where(present, perm, torch.full_like(perm, -1))
        vals = torch.where(present, vals, torch.full_like(vals, -1))
        return keys, vals, size

    def a_tokens(self, keys, vals, intent) -> torch.Tensor:
        """[B, 1 + T, 2]: position 0 = (purpose, pad), then one position per entry
        (key token, value token). The model sums the channel embeddings."""
        v = self.vocab
        present = keys >= 0
        ktok = torch.where(present, keys + v.KEY0, torch.full_like(keys, v.PAD))
        vtok = torch.where(present, vals + v.VAL0, torch.full_like(vals, v.PAD))
        ent = torch.stack([ktok, vtok], -1)
        lut = torch.zeros(5, dtype=torch.long)
        for code, tok in v.PURPOSE.items():
            lut[code] = tok
        head = torch.stack([lut[intent], torch.full_like(intent, v.PAD)], -1)[:, None]
        return torch.cat([head, ent], 1)

    # ----------------------------------------------------------------- queries
    def make_batch(self, g: torch.Generator, B: int, intents=None, s=None,
                   p_hint_absent: float = 0.0, intent: torch.Tensor | None = None) -> Batch:
        """intents: tuple of intent codes to sample from (default: value only).
        s: hint reliability. None = absent; float = fixed; "mix" = U[0,1] per example
        with probability p_hint_absent of absent."""
        t, v, a = self.task, self.vocab, self.ans
        if intent is None:
            intents = intents or (INTENT_SHARE_VALUE,)
            choice = torch.randint(0, len(intents), (B,), generator=g)
            intent = torch.tensor(intents, dtype=torch.long)[choice]
        keys, vals, size = self.make_tables(g, intent)

        # value query: a key that is in the table (uniform, or Zipf over key popularity)
        if t.query_zipf:
            present = torch.arange(self.T)[None, :] < size[:, None]
            w = self.key_weight[keys.clamp(min=0)] * present
            idx = torch.multinomial(w, 1, generator=g)[:, 0]
        else:
            idx = (torch.rand(B, generator=g) * size).long()
        vk = keys.gather(1, idx[:, None])[:, 0]
        vv = vals.gather(1, idx[:, None])[:, 0]
        # existence query: any key in the full range
        ek = torch.randint(0, t.key_range, (B,), generator=g)
        present = (keys == ek[:, None]).any(1)
        is_val = intent == INTENT_SHARE_VALUE
        is_ex = intent == INTENT_SHARE_KEY_EXISTENCE
        qkey = torch.where(is_val, vk, torch.where(is_ex, ek, torch.full_like(ek, -1)))
        answer = torch.where(is_val, vv,
                             torch.where(is_ex, torch.where(present, a.YES, a.NO),
                                         a.COUNT0 + size - 1))

        # hint: a noisy copy of the queried value, value queries only
        if s is None:
            rel = torch.full((B,), -1.0)
        elif s == "mix":
            rel = torch.rand(B, generator=g)
            rel = torch.where(torch.rand(B, generator=g) < p_hint_absent, torch.full_like(rel, -1.0), rel)
        else:
            rel = torch.full((B,), float(s))
        correct = torch.rand(B, generator=g) < rel
        rnd = torch.randint(0, t.value_range, (B,), generator=g)
        hint = torch.where(correct, vv, rnd)
        hint = torch.where((rel < 0) | ~is_val, torch.full_like(hint, -1), hint)

        b_tokens = self.b_tokens(intent, qkey, hint)
        return Batch(intent, keys, vals, size, self.a_tokens(keys, vals, intent), qkey, answer, hint, b_tokens)

    def b_tokens(self, intent, qkey, hint) -> torch.Tensor:
        """[B, 5], left padded: hint, hinted value or <absent>, then the query
        "k7=?" (value), "k7?" (existence) or "count?"."""
        v = self.vocab
        B = intent.shape[0]
        pad = torch.full((B,), v.PAD)
        ht = torch.full((B,), v.HINT)
        hv = torch.where(hint >= 0, hint + v.VAL0, torch.full_like(hint, v.ABSENT))
        kt = qkey.clamp(min=0) + v.KEY0
        is_val = intent == INTENT_SHARE_VALUE
        is_ex = intent == INTENT_SHARE_KEY_EXISTENCE
        val_row = torch.stack([ht, hv, kt, torch.full((B,), v.EQ), torch.full((B,), v.Q)], 1)
        ex_row = torch.stack([pad, ht, hv, kt, torch.full((B,), v.Q)], 1)
        cnt_row = torch.stack([pad, ht, hv, torch.full((B,), v.COUNT), torch.full((B,), v.Q)], 1)
        return torch.where(is_val[:, None], val_row, torch.where(is_ex[:, None], ex_row, cnt_row))

    def with_hint(self, batch: Batch, g: torch.Generator, s) -> Batch:
        """Return a copy of batch with a new hint at reliability s (None = absent)."""
        t = self.task
        B = len(batch)
        is_val = batch.intent == INTENT_SHARE_VALUE
        if s is None:
            hint = torch.full((B,), -1, dtype=torch.long)
        else:
            correct = torch.rand(B, generator=g) < float(s)
            rnd = torch.randint(0, t.value_range, (B,), generator=g)
            hint = torch.where(correct, batch.answer.clamp(max=t.value_range - 1), rnd)
            hint = torch.where(is_val, hint, torch.full_like(hint, -1))
        nb = batch.index(slice(None))
        nb.hint = hint
        nb.b_tokens = self.b_tokens(batch.intent, batch.qkey, hint)
        return nb

    # -------------------------------------------------------------- text link
    def serialise(self, keys_row, vals_row) -> str:
        ents = sorted((int(k), int(v)) for k, v in zip(keys_row, vals_row) if k >= 0)
        return " ".join(f"k{k}={v:0{self.nd}d}" for k, v in ents)

    def text_bytes(self, batch: Batch) -> torch.Tensor:
        """Length in bytes of each serialised table (the unbounded text link)."""
        return torch.tensor([len(self.serialise(k.tolist(), v.tolist())) for k, v in zip(batch.keys, batch.vals)])

    def text_tokens(self, batch: Batch, budget_bytes) -> torch.Tensor:
        """Serialised table ("k3=19 k7=42 ...", sorted by key) truncated to budget bytes,
        then tokenized with the same tokenizer A uses for its table: each complete
        "kX=VV" field is one 2-channel token (key, value). A field cut by the budget
        is dropped. Returns [B, K, 2], left padded."""
        v = self.vocab
        B = len(batch)
        if not torch.is_tensor(budget_bytes):
            budget_bytes = torch.full((B,), int(budget_bytes))
        out = torch.full((B, self.task.K, 2), v.PAD, dtype=torch.long)
        for i in range(B):
            ents = sorted((int(k), int(x)) for k, x in zip(batch.keys[i].tolist(), batch.vals[i].tolist()) if k >= 0)
            used, toks = 0, []
            for k, x in ents[: self.task.K]:
                n = len(f"k{k}={x:0{self.nd}d}") + (1 if toks else 0)
                if used + n > int(budget_bytes[i]):
                    break
                used += n
                toks.append((v.KEY0 + k, v.VAL0 + x))
            if toks:
                out[i, self.task.K - len(toks):] = torch.tensor(toks)
        return out

    # ------------------------------------------------------------ fixed sets
    def fixed_set(self, seed: int, n: int, intents=(INTENT_SHARE_VALUE,), s=None) -> Batch:
        g = torch.Generator().manual_seed(seed)
        return self.make_batch(g, n, intents=intents, s=s)

    def anchors(self, which: str, n: int) -> Batch:
        """Public anchor probes: value tables with a known query and answer."""
        seed = ANCHOR_FIT_SEED if which == "fit" else ANCHOR_HELDOUT_SEED
        return self.fixed_set(seed, n)

    def eval_set(self, n: int = 2000, intents=(INTENT_SHARE_VALUE,)) -> Batch:
        return self.fixed_set(EVAL_SEED + 7 * len(intents), n, intents=intents)
