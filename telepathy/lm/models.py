"""Frozen language models for Telepathy-1: loading, sender hidden states, and a receiver forward pass
with the receive port inserted after one decoder layer by a forward hook.

The models are never updated: every parameter has requires_grad = False. Gradients reach the
receive port through the receiver's upper layers, and the LM head is computed only at the answer
positions (Gemma's 262k vocabulary makes the full head the largest tensor otherwise)."""
from __future__ import annotations

import contextlib
import os

import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

QWEN = "Qwen/Qwen3-0.6B-Base"
GEMMA = "google/gemma-3-1b-pt"
SMOL = "HuggingFaceTB/SmolLM2-360M"
QWEN_17 = "Qwen/Qwen3-1.7B-Base"
SHORT = {QWEN: "qwen", GEMMA: "gemma", SMOL: "smol", QWEN_17: "qwen17"}


def device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class _Stop(Exception):
    pass


class FrozenLM:
    def __init__(self, repo: str, dtype=torch.bfloat16, dev=None):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.repo, self.dev, self.dtype = repo, dev or device(), dtype
        self.tok = AutoTokenizer.from_pretrained(repo)
        self.tok.padding_side = "right"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(repo, dtype=dtype).to(self.dev).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        cfg = self.model.config
        self.cfg = getattr(cfg, "text_config", cfg)
        self.hidden = self.cfg.hidden_size
        self.n_layers = self.cfg.num_hidden_layers
        self.softcap = getattr(self.cfg, "final_logit_softcapping", None)
        self.base = self.model.model
        self.layers = self.base.layers
        self.head = self.model.lm_head

    @property
    def short(self):
        return SHORT.get(self.repo, self.repo.split("/")[-1])

    def encode(self, texts, add_bos=True, max_len=None):
        enc = self.tok(list(texts), return_tensors="pt", padding=True, add_special_tokens=add_bos,
                       truncation=max_len is not None, max_length=max_len)
        return enc.input_ids.to(self.dev), enc.attention_mask.to(self.dev)

    # ------------------------------------------------------------------ sender side

    @torch.no_grad()
    def states(self, texts, layer: int):
        """Residual stream after decoder layer `layer` (1-based; n_layers = the last layer, before the
        final norm). Position 0 is always a sink token (BOS, or a prepended EOS for tokenizers without
        BOS); the send port masks it out. Stops the forward pass there. Returns (h [B, L, D] float32, mask [B, L] bool)."""
        ids, am = self.encode(texts)
        if not self._wants_bos():
            # no BOS: prepend one sink token, so that position 0 is never passage content
            sink = torch.full((ids.shape[0], 1), self.tok.eos_token_id, dtype=ids.dtype, device=ids.device)
            ids, am = torch.cat([sink, ids], 1), torch.cat([torch.ones_like(am[:, :1]), am], 1)
        out = {}

        def hook(_m, _i, o):
            out["h"] = o[0] if isinstance(o, tuple) else o
            raise _Stop

        hd = self.layers[layer - 1].register_forward_hook(hook)
        try:
            self.base(input_ids=ids, attention_mask=am, use_cache=False)
        except _Stop:
            pass
        finally:
            hd.remove()
        return out["h"].float(), am.bool()

    # ------------------------------------------------------------------ receiver side

    @contextlib.contextmanager
    def inserted(self, layer: int, fn):
        """Insert fn(h) -> h after decoder layer `layer` (1-based) for the duration of the block."""
        def hook(_m, _i, o):
            if isinstance(o, tuple):
                return (fn(o[0]),) + tuple(o[1:])
            return fn(o)
        hd = self.layers[layer - 1].register_forward_hook(hook)
        try:
            yield
        finally:
            hd.remove()

    def answer_logits(self, ids, am, positions):
        """Final hidden states -> logits at `positions` only. positions: (batch_idx, seq_idx)."""
        h = self.base(input_ids=ids, attention_mask=am, use_cache=False).last_hidden_state
        logits = self.head(h[positions]).float()
        if self.softcap:
            logits = torch.tanh(logits / self.softcap) * self.softcap
        return logits

    def build(self, prompts, answers):
        """Token ids for prompt + ' ' + answer + eos, with the target positions of the answer tokens.
        Returns ids, am, pred_pos (positions whose logits predict a target), targets."""
        bos = [self.tok.bos_token_id] if self.tok.bos_token_id is not None and self._wants_bos() else []
        rows, tgt_rows = [], []
        for p, a in zip(prompts, answers):
            pi = bos + self.tok(p, add_special_tokens=False).input_ids
            ai = self.tok(" " + a, add_special_tokens=False).input_ids + [self.tok.eos_token_id]
            rows.append(pi + ai)
            tgt_rows.append((len(pi), ai))
        L = max(map(len, rows))
        ids = torch.full((len(rows), L), self.tok.pad_token_id, dtype=torch.long)
        am = torch.zeros((len(rows), L), dtype=torch.long)
        bi, si, tg = [], [], []
        for b, (r, (start, ai)) in enumerate(zip(rows, tgt_rows)):
            ids[b, : len(r)] = torch.tensor(r)
            am[b, : len(r)] = 1
            for j, t in enumerate(ai):
                bi.append(b)
                si.append(start + j - 1)
                tg.append(t)
        pos = (torch.tensor(bi, device=self.dev), torch.tensor(si, device=self.dev))
        return ids.to(self.dev), am.to(self.dev), pos, torch.tensor(tg, device=self.dev)

    def _wants_bos(self):
        return len(self.tok("a").input_ids) > len(self.tok("a", add_special_tokens=False).input_ids)

    @torch.no_grad()
    def generate(self, prompts, max_new_tokens=8):
        """Greedy decoding. Returns the decoded continuation per prompt."""
        self.tok.padding_side = "left"
        try:
            ids, am = self.encode(prompts)
        finally:
            self.tok.padding_side = "right"
        out = self.model.generate(input_ids=ids, attention_mask=am, max_new_tokens=max_new_tokens,
                                  do_sample=False, pad_token_id=self.tok.pad_token_id)
        return self.tok.batch_decode(out[:, ids.shape[1]:], skip_special_tokens=True)
