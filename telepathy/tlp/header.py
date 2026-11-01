"""TLP header codec (28 bytes, network byte order) and the EXT_TRACE extension header."""
from __future__ import annotations

import struct
from dataclasses import dataclass, field, replace
from enum import IntEnum, IntFlag

TLP_VERSION = 0
IPPROTO_TLP = 253
MAX_PACKET = 1400

#            ver typ flg int src dst sid seq mid sidx scnt prio rsv sdim plen csum
HDR = struct.Struct("!BBBBHHIIHBBBBHHH")
EXT = struct.Struct("!16s8sB3x")  # W3C trace context: trace_id, parent_span_id, trace_flags, pad
assert HDR.size == 28 and EXT.size == 28

RESOLVER_EP = 1   # well-known endpoint of the resolver
PROBE_EP = 2      # well-known endpoint of the intent probe (auditor)
TAP_EP = 3        # UDP transport: the port mirror that copies every datagram to the probe


class MsgType(IntEnum):
    HELLO = 1
    HELLO_ACK = 2
    ANCHOR = 3
    ANCHOR_RESULT = 4
    KEY = 5
    KEY_CONFIRM = 6
    DATA = 7
    ACK = 8
    FIN = 9
    FIN_ACK = 10
    RESET = 11
    REGISTER = 12
    REGISTER_ACK = 13
    RESOLVE = 14
    RESOLVE_REPLY = 15
    ROUTE = 16
    KEY_ESCROW = 17
    ESCROW_ACK = 18
    VERDICT = 19
    NEGOTIATE = 20


class Flag(IntFlag):
    PROTECTED = 0x01
    LAST_SLOT = 0x02
    KEYED = 0x04
    ROUTED = 0x08
    ACK_REQ = 0x10
    EXT_TRACE = 0x20


class PacketError(ValueError):
    pass


@dataclass
class TraceExt:
    trace_id: bytes          # 16 bytes
    span_id: bytes           # 8 bytes (parent span id for the receiver)
    trace_flags: int = 1

    def pack(self) -> bytes:
        return EXT.pack(self.trace_id, self.span_id, self.trace_flags)

    @classmethod
    def unpack(cls, buf: bytes) -> "TraceExt":
        t, s, f = EXT.unpack(buf[: EXT.size])
        return cls(t, s, f)


@dataclass
class Packet:
    msg_type: int
    src_endpoint: int = 0
    dst_endpoint: int = 0
    session_id: int = 0
    seq: int = 0
    flags: int = 0
    intent: int = 0
    msg_id: int = 0
    slot_index: int = 0
    slot_count: int = 0
    slot_priority: int = 0
    slot_dim: int = 0
    payload: bytes = b""
    ext: TraceExt | None = None
    version: int = TLP_VERSION
    # filled in by the transport on receive (not on the wire in the TLP header)
    src_ip: str = field(default="", compare=False)
    dst_ip: str = field(default="", compare=False)

    def copy(self, **kw) -> "Packet":
        return replace(self, **kw)

    def encode(self) -> bytes:
        flags = self.flags | (Flag.EXT_TRACE if self.ext is not None else 0)
        if self.ext is None:
            flags &= ~Flag.EXT_TRACE
        ext = self.ext.pack() if self.ext is not None else b""
        fields = (self.version, int(self.msg_type), int(flags), self.intent, self.src_endpoint, self.dst_endpoint,
                  self.session_id, self.seq & 0xFFFFFFFF, self.msg_id & 0xFFFF, self.slot_index, self.slot_count,
                  self.slot_priority, 0, self.slot_dim, len(self.payload), 0)
        raw = HDR.pack(*fields) + ext + self.payload
        cs = checksum(raw)
        raw = raw[:26] + struct.pack("!H", cs) + raw[28:]
        if len(raw) > MAX_PACKET:
            raise PacketError(f"packet of {len(raw)} bytes exceeds {MAX_PACKET}")
        return raw

    @classmethod
    def decode(cls, buf: bytes) -> "Packet":
        if len(buf) < HDR.size:
            raise PacketError("short packet")
        (ver, typ, flags, intent, src, dst, sid, seq, mid, sidx, scnt, prio, rsv, sdim, plen, cs) = HDR.unpack(buf[: HDR.size])
        if ver != TLP_VERSION:
            raise PacketError(f"version {ver}")
        off = HDR.size
        ext = None
        if flags & Flag.EXT_TRACE:
            if len(buf) < off + EXT.size:
                raise PacketError("short extension")
            ext = TraceExt.unpack(buf[off:])
            off += EXT.size
        if len(buf) < off + plen:
            raise PacketError("short payload")
        total = buf[: off + plen]
        if checksum(total[:26] + b"\0\0" + total[28:]) != cs:
            raise PacketError("bad checksum")
        try:
            mt = MsgType(typ)
        except ValueError as e:
            raise PacketError(f"unknown msg_type {typ}") from e
        return cls(mt, src, dst, sid, seq, flags, intent, mid, sidx, scnt, prio, sdim,
                   bytes(buf[off: off + plen]), ext, ver)

    def wire_len(self) -> int:
        return HDR.size + (EXT.size if self.ext is not None else 0) + len(self.payload)


def checksum(data: bytes) -> int:
    """RFC 1071 ones-complement sum over the header (checksum zeroed), extension and payload."""
    if len(data) % 2:
        data += b"\0"
    s = sum(struct.unpack(f"!{len(data) // 2}H", data))
    while s >> 16:
        s = (s & 0xFFFF) + (s >> 16)
    return (~s) & 0xFFFF
