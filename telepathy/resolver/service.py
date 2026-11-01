"""Resolver service on the well-known endpoint, with knowledge-addressed relaying (ROUTE).

The relay rewrites endpoints and session ids and forwards packets both ways. The handshake
(HELLO / ANCHOR / KEY) runs end to end between the peers, so the relay never holds the session
key and cannot read keyed latents. It forwards EXT_TRACE unchanged, and it drops DATA toward the
receiver until it has seen the probe's ESCROW_ACK for the session.
"""
from __future__ import annotations

import itertools
import random
import threading
from dataclasses import dataclass

from ..tlp.header import RESOLVER_EP, Flag, MsgType, Packet
from ..tlp.messages import emb_decode, jdump, jload
from ..sock.socket import TLPSocket
from .records import RecordStore

M = MsgType


@dataclass
class RelayEntry:
    endpoint: int
    initiator: tuple[str, int]
    holder: tuple[str, int]
    holder_name: str
    sid_i: int | None = None        # session id on the initiator leg (chosen by the initiator)
    sid_h: int = 0                  # session id on the holder leg (chosen by the relay)
    audited: bool = False
    forwarded: int = 0
    dropped: int = 0


class Resolver:
    def __init__(self, ip="127.0.0.1", transport="udp", bus=None, auditor=None, log=None, tracer=None,
                 endpoint=RESOLVER_EP, require_audit: bool = True):
        self.store = RecordStore()
        self.sock = TLPSocket(transport, bus=bus, audit=False, auditor=auditor, log=log, name="resolver")
        self.sock.bind((ip, endpoint))
        self.sock.packet_handler = self._handle
        self.ip = ip
        self.auditor = auditor
        self.relays: dict[int, RelayEntry] = {}
        self._ep = itertools.count(40000)
        self.lock = threading.Lock()
        self.require_audit = require_audit
        self.tracer = tracer        # optional OpenTelepathy hook: tracer.relay(entry, pkt, direction)

    def close(self):
        self.sock.close()

    # ----------------------------------------------------------- control
    def _reply(self, req: Packet, msg_type, obj: dict, ext=None):
        p = Packet(msg_type, req.dst_endpoint, req.src_endpoint, req.session_id, 1, payload=jdump(obj), ext=ext)
        self.sock.send_packet(req.src_ip, p)
        self.sock.log("send", type=M(msg_type).name, sid=req.session_id,
                      info=obj.get("name") if isinstance(obj, dict) else None)

    def _handle(self, pkt: Packet):
        if pkt.dst_endpoint != RESOLVER_EP and pkt.dst_endpoint in self.relays:
            self._relay(pkt)
            return
        t = pkt.msg_type
        if t == M.REGISTER:
            r = jload(pkt.payload)
            emb = emb_decode(r["emb"]) if r.get("emb") else None
            self.store.register(r["name"], r.get("ip") or pkt.src_ip, r["endpoint"], r["model"], r["dialect"],
                                emb, r.get("tags", []), r.get("ttl", 300.0))
            self.sock.log("recv", type="REGISTER", sid=pkt.session_id,
                          info=f"{r['name']} -> {r.get('ip') or pkt.src_ip}:{r['endpoint']} tags={r.get('tags', [])}")
            self._reply(pkt, M.REGISTER_ACK, {"ok": True, "name": r["name"]}, ext=pkt.ext)
        elif t == M.RESOLVE:
            q = jload(pkt.payload)
            if q.get("name"):
                rec, score = self.store.by_name(q["name"]), 1.0
            else:
                hit = self.store.by_knowledge(emb_decode(q["emb"]))
                rec, score = hit if hit else (None, 0.0)
            if self.tracer:
                self.tracer.resolver_span("resolve", pkt, rec, score)
            if rec is None:
                self._reply(pkt, M.RESOLVE_REPLY, {"ok": False, "error": "not found"}, ext=pkt.ext)
                return
            self.sock.log("recv", type="RESOLVE", sid=pkt.session_id, info=f"best {rec.name} score {score:.3f}")
            self._reply(pkt, M.RESOLVE_REPLY, {"ok": True, "name": rec.name, "ip": rec.ip, "endpoint": rec.endpoint,
                                               "model": rec.model_id, "dialect": rec.dialect_id, "score": score},
                        ext=pkt.ext)
        elif t == M.ROUTE:
            q = jload(pkt.payload)
            hit = self.store.by_knowledge(emb_decode(q["emb"]))
            if self.tracer:
                self.tracer.resolver_span("route", pkt, hit[0] if hit else None, hit[1] if hit else 0.0)
            if not hit:
                self._reply(pkt, M.RESOLVE_REPLY, {"ok": False, "error": "no holder"}, ext=pkt.ext)
                return
            rec, score = hit
            with self.lock:
                ep = next(self._ep)
                e = RelayEntry(ep, (pkt.src_ip, pkt.src_endpoint), (rec.ip, rec.endpoint), rec.name,
                               sid_h=random.getrandbits(32) or 1)
                self.relays[ep] = e
                self.sock.add_endpoint(ep)
            self.sock.log("recv", type="ROUTE", sid=pkt.session_id,
                          info=f"best {rec.name} score {score:.3f}; relay endpoint {ep} (holder address withheld)")
            # The reply names the holder and its attested model id, but not its address.
            self._reply(pkt, M.RESOLVE_REPLY, {"ok": True, "routed": True, "name": rec.name, "ip": self.ip,
                                               "endpoint": ep, "model": rec.model_id, "dialect": rec.dialect_id,
                                               "score": score}, ext=pkt.ext)

    # ------------------------------------------------------------- relay
    def _relay(self, pkt: Packet):
        e = self.relays[pkt.dst_endpoint]
        src = (pkt.src_ip, pkt.src_endpoint)
        aud = self.auditor
        from_probe = aud is not None and src == (aud.ip, aud.ep)
        if from_probe:
            if pkt.msg_type == M.ESCROW_ACK:
                e.audited = True
            out = pkt.copy(src_endpoint=e.endpoint, dst_endpoint=e.holder[1], session_id=e.sid_h,
                           flags=pkt.flags | Flag.ROUTED)
            self._fwd(e, e.holder[0], out, pkt, "probe->holder")
            return
        if src == e.initiator:
            if e.sid_i is None:
                e.sid_i = pkt.session_id
            if pkt.session_id != e.sid_i:
                return
            out = pkt.copy(src_endpoint=e.endpoint, dst_endpoint=e.holder[1], session_id=e.sid_h,
                           flags=pkt.flags | Flag.ROUTED)
            self._fwd(e, e.holder[0], out, pkt, "initiator->holder")
        elif src == e.holder and pkt.session_id == e.sid_h and e.sid_i is not None:
            if pkt.msg_type == M.DATA and self.require_audit and not e.audited:
                e.dropped += 1
                self.sock.log("drop", type="DATA", sid=e.sid_i, slot=pkt.slot_index, info="relay: no ESCROW_ACK yet")
                return
            out = pkt.copy(src_endpoint=e.endpoint, dst_endpoint=e.initiator[1], session_id=e.sid_i,
                           flags=pkt.flags | Flag.ROUTED)
            self._fwd(e, e.initiator[0], out, pkt, "holder->initiator")

    def _fwd(self, e: RelayEntry, ip: str, out: Packet, orig: Packet, direction: str):
        e.forwarded += 1
        self.sock.send_packet(ip, out)  # EXT_TRACE (out.ext) is forwarded unchanged
        if self.tracer:
            self.tracer.relay(e, orig, direction)
        if orig.msg_type in (M.DATA, M.HELLO, M.KEY, M.ESCROW_ACK, M.RESET):
            self.sock.log("relay", type=M(orig.msg_type).name, sid=out.session_id, slot=orig.slot_index
                          if orig.msg_type == M.DATA else None, info=direction)


def register(sock: TLPSocket, resolver: tuple[str, int], name: str, endpoint: int, model_id: str,
             dialect_id: str, embedding=None, tags=(), ip: str | None = None, ttl: float = 300.0) -> dict:
    from ..tlp.messages import emb_encode
    obj = {"name": name, "ip": ip or sock.addr[0], "endpoint": endpoint, "model": model_id, "dialect": dialect_id,
           "tags": list(tags), "ttl": ttl}
    if embedding is not None:
        obj["emb"] = emb_encode(embedding)
    _, rep = sock.request(resolver, M.REGISTER, obj, {M.REGISTER_ACK})
    return rep
