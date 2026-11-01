"""b-bit slot quantization with a float16 per-slot scale (straight-through in training).

Wire format of one slot (little-endian):
  b = 16: slot_dim float16 values.
  b < 16: float16 scale, then slot_dim signed b-bit integers packed LSB-first.
"""
from __future__ import annotations

import numpy as np
import torch


def qmax(bits: int) -> int:
    return 2 ** (bits - 1) - 1


def quantize(x: torch.Tensor, bits) -> torch.Tensor:
    """Fake-quantize x [..., d]. bits: int or tensor broadcastable to x[..., 0].
    Returns dequantized values; gradients pass straight through."""
    if not torch.is_tensor(bits):
        bits = torch.full(x.shape[:-1], int(bits), dtype=torch.long, device=x.device)
    bits = bits.expand(x.shape[:-1])
    qm = (2.0 ** (bits.float() - 1) - 1).clamp(min=1)[..., None]
    scale = x.detach().abs().amax(-1, keepdim=True).clamp(min=1e-6) / qm
    scale = scale.half().float()
    q = (x / scale).round().clamp(-qm, qm) * scale
    q = torch.where(bits[..., None] >= 16, x.half().float(), q)
    return x + (q - x).detach()


def slot_payload_len(d: int, bits: int) -> int:
    return 2 * d if bits >= 16 else 2 + (d * bits + 7) // 8


def encode_slot(x: np.ndarray, bits: int) -> bytes:
    x = np.asarray(x, dtype=np.float32)
    if bits >= 16:
        return x.astype("<f2").tobytes()
    qm = qmax(bits)
    scale = np.float16(max(float(np.abs(x).max()), 1e-6) / qm)
    q = np.clip(np.round(x / np.float32(scale)), -qm, qm).astype(np.int64)
    u = (q & ((1 << bits) - 1)).astype(np.uint64)  # two's complement in b bits
    acc = 0
    for i, val in enumerate(u.tolist()):
        acc |= int(val) << (i * bits)
    nbytes = (len(x) * bits + 7) // 8
    return np.array([scale], dtype="<f2").tobytes() + acc.to_bytes(nbytes, "little")


def decode_slot(buf: bytes, d: int, bits: int) -> np.ndarray:
    if bits >= 16:
        return np.frombuffer(buf[: 2 * d], dtype="<f2").astype(np.float32)
    scale = np.frombuffer(buf[:2], dtype="<f2").astype(np.float32)[0]
    acc = int.from_bytes(buf[2: 2 + (d * bits + 7) // 8], "little")
    mask = (1 << bits) - 1
    out = np.empty(d, dtype=np.float32)
    for i in range(d):
        u = (acc >> (i * bits)) & mask
        if u >= 1 << (bits - 1):
            u -= 1 << bits
        out[i] = u * scale
    return out
