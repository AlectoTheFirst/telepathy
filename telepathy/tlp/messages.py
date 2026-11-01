"""Payload codecs for TLP message types. Control payloads are compact JSON; ACK is binary."""
from __future__ import annotations

import base64
import json

import numpy as np


def jdump(obj) -> bytes:
    return json.dumps(obj, separators=(",", ":"), sort_keys=True).encode()


def jload(b: bytes) -> dict:
    return json.loads(b.decode()) if b else {}


def b64(b: bytes) -> str:
    return base64.b64encode(b).decode()


def unb64(s: str) -> bytes:
    return base64.b64decode(s.encode())


def emb_encode(v) -> str:
    return b64(np.asarray(v, dtype="<f2").tobytes())


def emb_decode(s: str) -> np.ndarray:
    return np.frombuffer(unb64(s), dtype="<f2").astype(np.float32)


def ack_payload(kind: int, count: int, received) -> bytes:
    """ACK: u8 acknowledged msg_type, then a bitmap of received slot indices (LSB first)."""
    bm = bytearray((count + 7) // 8)
    for i in received:
        bm[i // 8] |= 1 << (i % 8)
    return bytes([kind]) + bytes(bm)


def ack_parse(payload: bytes, count: int) -> tuple[int, set[int]]:
    kind = payload[0]
    got = set()
    for i in range(count):
        if payload[1 + i // 8] >> (i % 8) & 1:
            got.add(i)
    return kind, got


# ------------------------------------------------------------------ fragmented messages (T records)
# T_PUBLISH and T_REPLY carry a JSON header and an opaque blob that can be far larger than one
# packet. The message u32 json_len | json | blob is cut into chunks; fragment k is one packet with
# msg_id = k and payload u16 total | chunk. A fragment fits one packet with the trace extension.
FRAG_CHUNK = 1400 - 28 - 28 - 2
MAX_FRAGMENTS = 0xFFFF


def frag_encode(meta: dict, blob: bytes = b"") -> list[bytes]:
    j = jdump(meta)
    body = len(j).to_bytes(4, "big") + j + bytes(blob)
    chunks = [body[i: i + FRAG_CHUNK] for i in range(0, len(body), FRAG_CHUNK)] or [b""]
    if len(chunks) > MAX_FRAGMENTS:
        raise ValueError(f"message of {len(body)} bytes needs more than {MAX_FRAGMENTS} fragments")
    n = len(chunks).to_bytes(2, "big")
    return [n + c for c in chunks]


class Reassembler:
    """Collects the fragments of one message (duplicates ignored). done() -> (meta, blob) or None."""

    def __init__(self):
        self.total = None
        self.parts: dict[int, bytes] = {}

    def add(self, index: int, payload: bytes):
        if len(payload) < 2:
            return
        self.total = int.from_bytes(payload[:2], "big")
        if index < self.total:
            self.parts.setdefault(index, payload[2:])

    def done(self):
        if self.total is None or len(self.parts) < self.total:
            return None
        body = b"".join(self.parts[k] for k in range(self.total))
        n = int.from_bytes(body[:4], "big")
        return jload(body[4: 4 + n]), body[4 + n:]
