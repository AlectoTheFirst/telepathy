"""Resolver records and the shared knowledge embedder."""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field

import numpy as np

EMB_DIM = 64


def key_vector(key: int, dim: int = EMB_DIM) -> np.ndarray:
    """Fixed pseudo-random +-1 vector for key k (hash seeded), unit norm."""
    h = hashlib.sha256(f"k{key}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(h[:8], "big"))
    v = rng.choice([-1.0, 1.0], size=dim).astype(np.float32)
    return v / np.linalg.norm(v)


def embed_keys(keys, dim: int = EMB_DIM) -> np.ndarray:
    """Shared hashed bag-of-keys embedder. A holder's K record is embed_keys(its key set);
    a query embedding is embed_keys([queried key]). Telepathy-0 uses one fixed embedder for
    all parties; the real system replaces it with a learned head."""
    v = np.zeros(dim, np.float32)
    for k in keys:
        v += key_vector(int(k), dim)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


@dataclass
class NameRecord:          # like an A record
    name: str
    ip: str
    endpoint: int
    model_id: str
    dialect_id: str
    ttl: float = 300.0
    created: float = field(default_factory=time.time)

    def alive(self) -> bool:
        return time.time() < self.created + self.ttl


@dataclass
class KRecord:             # knowledge embedding + tags -> name
    name: str
    embedding: np.ndarray
    tags: list
    ttl: float = 300.0
    created: float = field(default_factory=time.time)

    def alive(self) -> bool:
        return time.time() < self.created + self.ttl


class RecordStore:
    def __init__(self):
        self.names: dict[str, NameRecord] = {}
        self.krecs: dict[str, KRecord] = {}
        self.trecs = TRecordStore()

    def register(self, name, ip, endpoint, model_id, dialect_id, embedding=None, tags=(), ttl=300.0):
        self.names[name] = NameRecord(name, ip, endpoint, model_id, dialect_id, ttl)
        if embedding is not None:
            e = np.asarray(embedding, np.float32)
            self.krecs[name] = KRecord(name, e / max(np.linalg.norm(e), 1e-9), list(tags), ttl)

    def by_name(self, name) -> NameRecord | None:
        r = self.names.get(name)
        return r if r and r.alive() else None

    def by_knowledge(self, query) -> tuple[NameRecord, float] | None:
        q = np.asarray(query, np.float32)
        q = q / max(np.linalg.norm(q), 1e-9)
        best, score = None, -2.0
        for name, k in self.krecs.items():
            if not k.alive() or self.by_name(name) is None:
                continue
            s = float(k.embedding @ q)
            if s > score:
                best, score = name, s
        return (self.names[best], score) if best else None


@dataclass
class TRecord:             # Telepathy-1: a fitted translator, cached (SPEC.md 14.1)
    dialect_from: str
    dialect_to: str
    family: str
    blob: bytes            # opaque: serialised by the publisher, never interpreted by the resolver
    score: float           # the publisher's held-out score when it fitted the translator
    ttl: float = 3600.0
    created: float = field(default_factory=time.time)

    def alive(self) -> bool:
        return time.time() < self.created + self.ttl

    def ttl_left(self) -> float:
        return max(0.0, self.created + self.ttl - time.time())


class TRecordStore:
    """Keyed by (dialect id from, dialect id to, family). A dialect id changes whenever weights
    change, so a translator for old weights is simply never found."""

    def __init__(self):
        self.recs: dict[tuple[str, str, str], TRecord] = {}

    def publish(self, dialect_from, dialect_to, family, blob: bytes, score: float, ttl: float = 3600.0) -> TRecord:
        r = TRecord(dialect_from, dialect_to, family, bytes(blob), float(score), float(ttl))
        self.recs[(dialect_from, dialect_to, family)] = r
        return r

    def lookup(self, dialect_from, dialect_to, family) -> TRecord | None:
        key = (dialect_from, dialect_to, family)
        r = self.recs.get(key)
        if r is not None and not r.alive():
            del self.recs[key]
            return None
        return r
