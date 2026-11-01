"""X25519 key exchange, HKDF key schedule, KEY_CONFIRM HMAC and escrow wrapping.

Not production cryptography: the exchange is unauthenticated (an active attacker can sit in
the middle). Keyed DATA payloads use per-packet AES-GCM (aead.py). See SPEC.md.
"""
from __future__ import annotations

import hashlib
import hmac
import os
from dataclasses import dataclass

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

RAW = serialization.Encoding.Raw
RAWPUB = serialization.PublicFormat.Raw
RAWPRIV = serialization.PrivateFormat.Raw


def pub_bytes(priv: X25519PrivateKey) -> bytes:
    return priv.public_key().public_bytes(RAW, RAWPUB)


def priv_bytes(priv: X25519PrivateKey) -> bytes:
    return priv.private_bytes(RAW, RAWPRIV, serialization.NoEncryption())


def hkdf(secret: bytes, salt: bytes, info: bytes, length: int) -> bytes:
    return HKDF(algorithm=hashes.SHA256(), length=length, salt=salt, info=info).derive(secret)


def transcript_hash(*parts: bytes) -> bytes:
    h = hashlib.sha256()
    for p in parts:
        h.update(len(p).to_bytes(4, "big"))
        h.update(p)
    return h.digest()


@dataclass
class SessionKeys:
    data_key: bytes         # 32 bytes, AES-256-GCM key for keyed DATA (aead.py)
    confirm_key: bytes      # 32 bytes, KEY_CONFIRM HMAC key
    nonce_salt: bytes       # 4 bytes, session-stable nonce prefix (aead.py)

    def escrow_blob(self) -> bytes:
        """What the receiver escrows to the probe: the AEAD key and the nonce salt."""
        return self.data_key + self.nonce_salt

    @staticmethod
    def from_escrow(blob: bytes) -> tuple[bytes, bytes]:
        return blob[:32], blob[32:36]


class KeyExchange:
    """One side of the ephemeral X25519 exchange. The transcript covers the HELLO and
    HELLO_ACK payloads and both public keys, not endpoints or session ids, so a relay that
    rewrites the header does not break the confirmation."""

    def __init__(self):
        self.priv = X25519PrivateKey.generate()
        self.pub = pub_bytes(self.priv)

    def derive(self, peer_pub: bytes, hello: bytes, hello_ack: bytes, initiator_pub: bytes,
               responder_pub: bytes) -> tuple[SessionKeys, bytes]:
        shared = self.priv.exchange(X25519PublicKey.from_public_bytes(peer_pub))
        th = transcript_hash(hello, hello_ack, initiator_pub, responder_pub)
        okm = hkdf(shared, th, b"telepathy-0 key schedule", 68)
        return SessionKeys(okm[:32], okm[32:64], okm[64:68]), th


def confirm_tag(keys: SessionKeys, transcript: bytes, role: bytes) -> bytes:
    return hmac.new(keys.confirm_key, role + transcript, hashlib.sha256).digest()


def confirm_ok(keys: SessionKeys, transcript: bytes, role: bytes, tag: bytes) -> bool:
    return hmac.compare_digest(confirm_tag(keys, transcript, role), tag)


# -------------------------------------------------------------------- escrow
def wrap_for(static_pub: bytes, secret: bytes, aad: bytes) -> bytes:
    """Wrap a session secret to the probe's static X25519 public key (ephemeral-static DH,
    HKDF, AES-GCM). Output: eph_pub(32) | nonce(12) | ciphertext+tag."""
    eph = X25519PrivateKey.generate()
    shared = eph.exchange(X25519PublicKey.from_public_bytes(static_pub))
    k = hkdf(shared, pub_bytes(eph) + static_pub, b"telepathy-0 escrow", 32)
    nonce = os.urandom(12)
    return pub_bytes(eph) + nonce + AESGCM(k).encrypt(nonce, secret, aad)


def unwrap_with(static_priv: X25519PrivateKey, blob: bytes, aad: bytes) -> bytes:
    eph_pub, nonce, ct = blob[:32], blob[32:44], blob[44:]
    shared = static_priv.exchange(X25519PublicKey.from_public_bytes(eph_pub))
    k = hkdf(shared, eph_pub + pub_bytes(static_priv), b"telepathy-0 escrow", 32)
    return AESGCM(k).decrypt(nonce, ct, aad)


def load_or_create_static(path: str) -> X25519PrivateKey:
    if os.path.exists(path):
        with open(path, "rb") as f:
            return X25519PrivateKey.from_private_bytes(f.read())
    os.makedirs(os.path.dirname(path), exist_ok=True)
    k = X25519PrivateKey.generate()
    with open(path, "wb") as f:
        f.write(priv_bytes(k))
    os.chmod(path, 0o600)
    return k
