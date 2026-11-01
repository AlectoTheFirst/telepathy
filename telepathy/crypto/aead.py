"""Per-packet AEAD for keyed DATA: AES-256-GCM, one packet at a time, so a lost slot does not
affect the others.

Key and nonce salt come from HKDF over the X25519 secret (kx.SessionKeys).

nonce (12 B) = nonce_salt (4) | direction (1: b"I" initiator sends, b"R" responder sends) | 0 (3) | seq (4)
AAD          = the 28-byte TLP header with the fields a relay rewrites set to zero: source and
               destination endpoints, session id, checksum, and the ROUTED and EXT_TRACE flags.

The brief asks for nonce = session id | seq and AAD = the TLP header. A ROUTE relay rewrites the
session id and the endpoints on each leg (SPEC.md section 5), so the two ends see different values:
the session-stable nonce salt takes the place of the session id, and the rewritten fields are
zeroed in the AAD. The direction byte separates the two senders, which both count seq from 1
under one key. Every send, retransmits included, takes a new seq, so no nonce repeats.
"""
from __future__ import annotations

import struct

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from ..tlp.header import HDR, TLP_VERSION, Flag, Packet

TAG_LEN = 16
_MUTABLE_FLAGS = Flag.ROUTED | Flag.EXT_TRACE


class AuthError(ValueError):
    pass


def nonce(salt: bytes, direction: bytes, seq: int) -> bytes:
    return salt + direction + b"\0\0\0" + struct.pack("!I", seq & 0xFFFFFFFF)


def aad(pkt: Packet, payload_len: int) -> bytes:
    return HDR.pack(TLP_VERSION, int(pkt.msg_type), int(pkt.flags) & ~int(_MUTABLE_FLAGS) & 0xFF, pkt.intent,
                    0, 0, 0, pkt.seq & 0xFFFFFFFF, pkt.msg_id & 0xFFFF, pkt.slot_index, pkt.slot_count,
                    pkt.slot_priority, 0, pkt.slot_dim, payload_len, 0)


class PacketAEAD:
    def __init__(self, key: bytes, salt: bytes):
        assert len(key) == 32 and len(salt) == 4
        self.aes = AESGCM(key)
        self.salt = salt

    def seal(self, pkt: Packet, direction: bytes) -> bytes:
        """Ciphertext + tag for pkt.payload. pkt.seq and pkt.flags must be final."""
        n = len(pkt.payload) + TAG_LEN
        return self.aes.encrypt(nonce(self.salt, direction, pkt.seq), pkt.payload, aad(pkt, n))

    def open(self, pkt: Packet, direction: bytes) -> bytes:
        try:
            return self.aes.decrypt(nonce(self.salt, direction, pkt.seq), pkt.payload, aad(pkt, len(pkt.payload)))
        except Exception as e:  # cryptography raises InvalidTag
            raise AuthError("AEAD authentication failed") from e


def sender_direction(role: str) -> bytes:
    """Direction byte of packets sent by the side with this role."""
    return b"I" if role == "initiator" else b"R"
