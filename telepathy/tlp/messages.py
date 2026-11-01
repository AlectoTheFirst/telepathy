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
