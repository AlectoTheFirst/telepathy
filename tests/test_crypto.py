import numpy as np
import pytest
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey

from telepathy.crypto.aead import TAG_LEN, AuthError, PacketAEAD, sender_direction
from telepathy.crypto.kx import KeyExchange, SessionKeys, confirm_ok, confirm_tag, pub_bytes, unwrap_with, wrap_for
from telepathy.models.quant import decode_slot, encode_slot
from telepathy.tlp.header import Flag, MsgType, Packet


def _pkt(seq, payload, **kw):
    base = dict(src_endpoint=7, dst_endpoint=107, session_id=0x1234, seq=seq, flags=Flag.KEYED | Flag.PROTECTED,
                intent=1, msg_id=3, slot_index=seq % 8, slot_count=8, slot_priority=seq % 8, slot_dim=32)
    base.update(kw)
    return Packet(MsgType.DATA, payload=payload, **base)


@pytest.fixture
def aead():
    return PacketAEAD(b"k" * 32, b"salt")


def test_aead_round_trip_per_packet_with_lost_packets(aead):
    rng = np.random.default_rng(0)
    slots = rng.standard_normal((8, 32)).astype(np.float32)
    sealed = {}
    for i in range(8):
        p = _pkt(i + 1, encode_slot(slots[i], 16))
        p.payload = aead.seal(p, b"R")
        assert len(p.payload) == 64 + TAG_LEN
        sealed[i] = Packet.decode(p.encode())   # through the wire codec
    del sealed[2], sealed[5]                    # lost packets do not affect the others
    for i, p in sealed.items():
        x = decode_slot(aead.open(p, b"R"), 32, 16)
        assert np.allclose(x, slots[i], atol=2e-3)


def test_aead_rejects_tampering(aead):
    p = _pkt(9, b"\x01" * 64)
    p.payload = aead.seal(p, b"R")
    bad = bytearray(p.payload)
    bad[0] ^= 1
    with pytest.raises(AuthError):
        aead.open(p.copy(payload=bytes(bad)), b"R")
    for change in (dict(slot_index=4), dict(intent=2), dict(msg_id=4), dict(seq=10),
                   dict(flags=Flag.KEYED)):             # header fields are authenticated (AAD)
        with pytest.raises(AuthError):
            aead.open(p.copy(**change), b"R")
    with pytest.raises(AuthError):                      # the other direction uses other nonces
        aead.open(p, b"I")
    with pytest.raises(AuthError):
        PacketAEAD(b"x" * 32, b"salt").open(p, b"R")


def test_aead_survives_relay_rewrites(aead):
    """A ROUTE relay rewrites endpoints and the session id and sets ROUTED; the trace extension
    is added or forwarded. None of these fields are in the AAD."""
    p = _pkt(5, b"\x02" * 64)
    p.payload = aead.seal(p, b"R")
    q = p.copy(src_endpoint=40001, dst_endpoint=31, session_id=0xBEEF, flags=p.flags | Flag.ROUTED)
    assert aead.open(Packet.decode(q.encode()), b"R") == b"\x02" * 64


def test_direction_bytes():
    assert sender_direction("initiator") == b"I" and sender_direction("responder") == b"R"


def test_key_exchange_and_confirm():
    a, b = KeyExchange(), KeyExchange()
    ka, th_a = a.derive(b.pub, b"hello", b"ack", a.pub, b.pub)
    kb, th_b = b.derive(a.pub, b"hello", b"ack", a.pub, b.pub)
    assert ka.data_key == kb.data_key and ka.nonce_salt == kb.nonce_salt and th_a == th_b
    assert len(ka.data_key) == 32 and len(ka.nonce_salt) == 4
    tag = confirm_tag(ka, th_a, b"I")
    assert confirm_ok(kb, th_b, b"I", tag)
    assert not confirm_ok(kb, th_b, b"R", tag)
    # a different transcript gives different keys
    kc, _ = b.derive(a.pub, b"hello2", b"ack", a.pub, b.pub)
    assert kc.data_key != kb.data_key


def test_escrow_wraps_the_aead_key():
    probe = X25519PrivateKey.generate()
    keys = SessionKeys(b"d" * 32, b"c" * 32, b"salt")
    blob = wrap_for(pub_bytes(probe), keys.escrow_blob(), b"sid")
    assert SessionKeys.from_escrow(unwrap_with(probe, blob, b"sid")) == (b"d" * 32, b"salt")
    with pytest.raises(Exception):
        unwrap_with(probe, blob, b"other")


def test_relayed_packet_with_modified_seq_fails(aead):
    """The relay rewrites endpoints, session id and ROUTED, never seq (resolver/service.py _relay).
    seq is in both the nonce and the AAD, so a relayed packet whose seq changed must fail."""
    p = _pkt(5, b"\x03" * 64)
    p.payload = aead.seal(p, b"R")
    relayed = p.copy(src_endpoint=40001, dst_endpoint=31, session_id=0xBEEF, flags=p.flags | Flag.ROUTED)
    assert aead.open(relayed, b"R") == b"\x03" * 64
    with pytest.raises(AuthError):
        aead.open(relayed.copy(seq=6), b"R")


def test_direction_byte_is_in_the_nonce():
    """Both sides count seq from 1 under one key: the same seq must give different nonces."""
    from telepathy.crypto.aead import nonce
    salt = b"salt"
    assert nonce(salt, b"I", 5) != nonce(salt, b"R", 5)
    assert nonce(salt, b"I", 5)[4:5] == b"I" and len(nonce(salt, b"I", 5)) == 12
    # and the ciphertexts of one payload at one seq differ by direction
    a = PacketAEAD(b"k" * 32, salt)
    p = _pkt(5, b"\x04" * 64)
    assert a.seal(p, b"I") != a.seal(p, b"R")
