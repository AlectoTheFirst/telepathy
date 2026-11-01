"""Shared configuration, vocabulary and intent codes for Telepathy-0."""
from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import asdict, dataclass, field

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT_DIR = os.path.join(ROOT, "checkpoints")
RESULTS_DIR = os.path.join(ROOT, "results")

# Intent codes (section 8). Only DATA and ANCHOR messages carry a non-zero intent.
INTENT_UNSPECIFIED = 0
INTENT_SHARE_VALUE = 1
INTENT_SHARE_KEY_EXISTENCE = 2
INTENT_SHARE_ENTRY_COUNT = 3
INTENT_ANCHOR_VERIFY = 4
INTENT_NAMES = {
    0: "UNSPECIFIED",
    1: "SHARE_VALUE",
    2: "SHARE_KEY_EXISTENCE",
    3: "SHARE_ENTRY_COUNT",
    4: "ANCHOR_VERIFY",
}
DATA_INTENTS = (INTENT_SHARE_VALUE, INTENT_SHARE_KEY_EXISTENCE, INTENT_SHARE_ENTRY_COUNT)

# Payload bits per dimension negotiated at HELLO.
BITS_CHOICES = (16, 8, 4, 2)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass(frozen=True)
class TaskConfig:
    K: int = 8                 # entries per table for value / existence intents
    key_range: int = 16        # keys k0 .. k{key_range-1}
    value_range: int = 100     # values 0 .. value_range-1
    # Value-query distribution. 0 = uniform over the table entries (the default task).
    # s > 0 = Zipf(s) over key popularity: a fixed random order of the keys (seed below),
    # meant to differ from the key-id order of the sorted text serialisation. With seed 4242 the
    # four most popular keys are k3, k1, k4, k2, so by chance sorted text sends them first.
    query_zipf: float = 0.0
    popularity_seed: int = 4242

    @property
    def count_max(self) -> int:
        # SHARE_ENTRY_COUNT tables vary in size so that the count carries information.
        return min(self.key_range, 2 * self.K)

    @property
    def table_max(self) -> int:
        return max(self.K, self.count_max)

    @property
    def n_digits(self) -> int:
        return len(str(self.value_range - 1))

    def floor_bits(self) -> float:
        """Information floor of one value table: log2 C(key_range, K) + K log2(value_range)."""
        from math import comb, log2
        return log2(comb(self.key_range, self.K)) + self.K * log2(self.value_range)


class Vocab:
    """Token ids. Structured tokens for the models, plus characters for the text link."""

    def __init__(self, task: TaskConfig):
        self.task = task
        names = ["<pad>", "<sep>", "=", "?", "<count>", "<hint>", "<absent>",
                 "<p:value>", "<p:keys>", "<p:count>"]
        names += [f"k{i}" for i in range(task.key_range)]
        names += [f"v{i}" for i in range(task.value_range)]
        # character tokens for the serialised text link ("k3=19 k7=42")
        names += ["c:k", "c:=", "c: "] + [f"c:{i}" for i in range(10)]
        self.names = names
        self.id = {n: i for i, n in enumerate(names)}
        self.size = len(names)
        self.PAD = self.id["<pad>"]
        self.SEP = self.id["<sep>"]
        self.EQ = self.id["="]
        self.Q = self.id["?"]
        self.COUNT = self.id["<count>"]
        self.HINT = self.id["<hint>"]
        self.ABSENT = self.id["<absent>"]
        self.PURPOSE = {INTENT_SHARE_VALUE: self.id["<p:value>"],
                        INTENT_SHARE_KEY_EXISTENCE: self.id["<p:keys>"],
                        INTENT_SHARE_ENTRY_COUNT: self.id["<p:count>"]}
        self.KEY0 = self.id["k0"]
        self.VAL0 = self.id["v0"]
        self.CHAR = {"k": self.id["c:k"], "=": self.id["c:="], " ": self.id["c: "]}
        for i in range(10):
            self.CHAR[str(i)] = self.id[f"c:{i}"]

    def char_ids(self, text: str) -> list[int]:
        return [self.CHAR[c] for c in text]


class Answers:
    """One answer vocabulary: values, no, yes, counts 1..count_max."""

    def __init__(self, task: TaskConfig):
        self.task = task
        self.NO = task.value_range
        self.YES = task.value_range + 1
        self.COUNT0 = task.value_range + 2  # answer id for count c is COUNT0 + c - 1
        self.size = task.value_range + 2 + task.count_max

    def decode(self, a: int) -> str:
        if a < self.task.value_range:
            return f"{a:0{self.task.n_digits}d}"
        if a == self.NO:
            return "no"
        if a == self.YES:
            return "yes"
        return str(a - self.COUNT0 + 1)


@dataclass(frozen=True)
class ModelConfig:
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 4
    N: int = 8                 # max latent slots
    d: int = 32                # slot dimension
    recv_layer: int = 1        # receive port sits after this many blocks
    max_len: int = 128
    pos_bias_init: float = 3.0  # send port: initial attention bias from slot i to table position i


@dataclass
class TrainConfig:
    seed: int = 0
    steps: int = 2500
    batch: int = 128
    lr: float = 2e-3
    p_hint_absent: float = 0.35
    p_holes: float = 0.15      # probability of random missing best-effort slots
    task: TaskConfig = field(default_factory=TaskConfig)
    model: ModelConfig = field(default_factory=ModelConfig)


def dialect_id(task: TaskConfig, model: ModelConfig, seed: int) -> str:
    t = asdict(task)
    if not t.get("query_zipf"):  # uniform task: keep the dialect ids of the original runs
        t.pop("query_zipf", None)
        t.pop("popularity_seed", None)
    mdl = asdict(model)
    if mdl.get("pos_bias_init") == 3.0:  # default prior: keep the dialect ids of the original runs
        mdl.pop("pos_bias_init")
    blob = json.dumps({"task": t, "model": mdl, "seed": seed}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:8]
