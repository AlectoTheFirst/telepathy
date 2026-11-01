import numpy as np
import pytest

from telepathy.models.quant import decode_slot, encode_slot, slot_payload_len
from telepathy.tlp.header import EXT, HDR, MAX_PACKET, Flag, MsgType, Packet, PacketError, TraceExt


def test_header_is_28_bytes_and_aligned():
    assert HDR.size == 28
    assert EXT.size == 28
    # offsets of the u32 and u16 fields are aligned
    raw = Packet(MsgType.DATA, 1, 2, 0x11223344, 0x55667788, slot_dim=0x0102).encode()
    assert raw[8:12] == bytes.fromhex("11223344")
    assert raw[12:16] == bytes.fromhex("55667788")
    assert raw[22:24] == bytes.fromhex("0102")
    assert raw[21] == 0  # reserved


def test_round_trip_all_fields():
    p = Packet(MsgType.DATA, src_endpoint=513, dst_endpoint=7, session_id=0xDEADBEEF, seq=42,
               flags=Flag.PROTECTED | Flag.KEYED, intent=3, msg_id=0xBEEF, slot_index=5, slot_count=8,
               slot_priority=5, slot_dim=32, payload=bytes(range(64)))
    q = Packet.decode(p.encode())
    for f in ("msg_type", "src_endpoint", "dst_endpoint", "session_id", "seq", "intent", "msg_id",
              "slot_index", "slot_count", "slot_priority", "slot_dim", "payload"):
        assert getattr(q, f) == getattr(p, f), f
    assert q.flags == p.flags
    assert q.ext is None


def test_trace_extension_round_trip():
    ext = TraceExt(bytes(range(16)), bytes(range(100, 108)), 1)
    p = Packet(MsgType.ANCHOR, 1, 2, 3, 4, payload=b"xyz", ext=ext)
    raw = p.encode()
    assert len(raw) == 28 + 28 + 3
    q = Packet.decode(raw)
    assert q.flags & Flag.EXT_TRACE
    assert q.ext == ext
    assert q.payload == b"xyz"


def test_checksum_detects_corruption():
    raw = bytearray(Packet(MsgType.DATA, 1, 2, 3, 4, payload=b"hello world!").encode())
    raw[30] ^= 0x01
    with pytest.raises(PacketError):
        Packet.decode(bytes(raw))


def test_packet_size_limit():
    Packet(MsgType.DATA, payload=b"\0" * (MAX_PACKET - 28)).encode()
    with pytest.raises(PacketError):
        Packet(MsgType.DATA, payload=b"\0" * (MAX_PACKET - 27)).encode()


@pytest.mark.parametrize("bits", [16, 8, 4, 2])
def test_slot_codec(bits):
    rng = np.random.default_rng(0)
    x = rng.standard_normal(32).astype(np.float32)
    buf = encode_slot(x, bits)
    assert len(buf) == slot_payload_len(32, bits)
    y = decode_slot(buf, 32, bits)
    scale = np.abs(x).max() / max(1, 2 ** (bits - 1) - 1)
    tol = 1e-2 if bits == 16 else scale * 0.51 + 1e-3
    assert np.abs(x - y).max() <= tol


def test_max_slot_fits_mtu():
    # one slot per DATA packet: header + extension + slot_dim x 2 bytes stays under 1400
    assert 28 + 28 + slot_payload_len(512, 16) <= MAX_PACKET
