"""TLPSocket: a BSD-style socket API over the Telepathy Link Protocol.

    s = TLPSocket(transport="raw"); s.bind(("127.0.0.1", 7)); s.listen()
    conn, peer = s.accept()                          # full handshake
    c = TLPSocket(); c.connect(("127.0.0.1", 7), capabilities=caps)
    msg_id = conn.send_latent(slots, protected=4)
    for slots, mask, final in c.recv_latent(deadline=0.2): ...

The loss injector sits between this API and the transport. Protocol logic (ACK handling,
retransmission, finalisation, demultiplexing) runs in the receive thread; the handshake runs
in the thread that calls connect() or accept().
"""
from __future__ import annotations

import contextlib
import heapq
import itertools
import os
import queue
import random
import threading
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from ..config import INTENT_ANCHOR_VERIFY, INTENT_SHARE_VALUE
from ..crypto.kx import KeyExchange, confirm_ok, confirm_tag, wrap_for
from ..crypto.aead import AuthError, PacketAEAD, sender_direction
from ..models.quant import decode_slot, encode_slot
from ..tlp.header import RESOLVER_EP, Flag, MsgType, Packet, PacketError
from ..tlp.messages import ack_parse, ack_payload, b64, emb_encode, jdump, jload
from ..tlp.session import Auditor, Capabilities, Session, TLPOptions
from ..tlp.state import State, StateMachine
from ..tlp.transport import Injector, InprocBus, make_transport

M = MsgType
HEARTBEAT = 0.05
ANCHOR_WINDOW = 32   # anchor messages in flight during VERIFYING and NEGOTIATING
FIT_BASE = 1000  # msg_id offset of fitting anchors


class HandshakeError(RuntimeError):
    pass


class ConnectionReset(RuntimeError):
    pass


# ------------------------------------------------------------------------ timers
class Timers:
    def __init__(self):
        self._heap: list = []
        self._cv = threading.Condition()
        self._ctr = itertools.count()
        self._stop = False
        threading.Thread(target=self._run, daemon=True, name="tlp-timers").start()

    def call_later(self, delay: float, fn) -> list:
        h = [True]
        with self._cv:
            heapq.heappush(self._heap, (time.monotonic() + delay, next(self._ctr), fn, h))
            self._cv.notify()
        return h

    @staticmethod
    def cancel(h):
        if h:
            h[0] = False

    def _run(self):
        while True:
            with self._cv:
                while not self._heap and not self._stop:
                    self._cv.wait()
                if self._stop:
                    return
                t, _, fn, h = self._heap[0]
                now = time.monotonic()
                if t > now:
                    self._cv.wait(t - now)
                    continue
                heapq.heappop(self._heap)
            if h[0]:
                try:
                    fn()
                except Exception as e:  # never kill the timer thread
                    print("timer error:", repr(e))

    def stop(self):
        with self._cv:
            self._stop = True
            self._cv.notify()


# ------------------------------------------------------------- message records
@dataclass
class _Out:
    msg_type: int
    msg_id: int
    count: int
    P: int
    packets: list
    acked: set = field(default_factory=set)
    retx: int = 0
    retransmits: int = 0
    timer: list | None = None
    done: threading.Event = field(default_factory=threading.Event)
    max_retx: int = 3


@dataclass
class _In:
    msg_type: int
    msg_id: int
    count: int
    intent: int = 0
    vecs: dict = field(default_factory=dict)      # slot index -> np vector (decoded, inverted)
    order: list = field(default_factory=list)     # arrival order of slot indices
    prot_max: int = -1
    be_min: int = 1 << 30
    last_seen: bool = False
    ack_sent_at: float | None = None
    first_at: float = 0.0
    final: bool = False
    final_reason: str = ""
    duplicates: int = 0
    bytes: int = 0
    ack_timer: list | None = None
    trace: object = None


# ------------------------------------------------------------------- connection
class Connection:
    """One TLP session. Returned by accept(); also used inside a connected TLPSocket."""

    def __init__(self, sock: "TLPSocket", sid: int, peer: tuple[str, int], role: str, caps: Capabilities | None):
        self.sock = sock
        self.opts: TLPOptions = sock.opts
        self.caps = caps
        self.session = Session(sid, sock.addr, peer, role)
        self.fsm = StateMachine(on_change=self._on_state)
        self.inbox: queue.Queue = queue.Queue()
        self.stash: list[Packet] = []
        self.out: dict = {}
        self.inc: dict = {}
        self.data_q: queue.Queue = queue.Queue()
        self.lock = threading.RLock()
        self.cv = threading.Condition(self.lock)
        self.seq = itertools.count(1)
        self.msg_ids = itertools.count(1)
        self.reset_reason: str | None = None
        self.verdicts: list[dict] = []
        self.trace_ext = None     # callable -> TraceExt | None, set by the tracing layer
        self.on_verdict = None    # optional callback(dict)
        self.hello = b""
        self.hello_ack = b""
        self.held: list[Packet] = []
        self.hello_ext = None

    # ---------------------------------------------------------------- logging
    def log(self, ev: str, **kw):
        s = self.session
        self.sock.log(ev, sid=s.session_id, trust=s.trust, state=self.fsm.state.value, **kw)

    def _on_state(self, old, new, reason):
        self.log("state", info=f"{old.value} -> {new.value}" + (f" ({reason})" if reason else ""))
        if new == State.ESTABLISHED:
            with self.lock:
                held, self.held = self.held, []
            for p in held:
                self.on_packet(p)
        elif new == State.CLOSED:
            with self.lock:
                self.held = []

    # ------------------------------------------------------------------- send
    def _send(self, msg_type, payload=b"", flags=0, intent=0, msg_id=0, slot_index=0, slot_count=0,
              prio=0, slot_dim=0, dst: tuple[str, int] | None = None, quiet=False):
        s = self.session
        ip, ep = dst or s.peer
        if s.routed:
            flags |= Flag.ROUTED
        pkt = Packet(msg_type, s.local[1], ep, s.session_id, next(self.seq), flags, intent, msg_id,
                     slot_index, slot_count, prio, slot_dim, payload)
        if flags & Flag.KEYED:  # per-packet AEAD; seq and flags are final here
            pkt.payload = s.aead.seal(pkt, sender_direction(s.role))
        if self.trace_ext is not None:
            pkt.ext = self.trace_ext()
        data = pkt.encode()
        s.stats.packets_sent += 1
        s.stats.bytes_sent += len(data)
        if not self.fsm.established:
            s.stats.handshake_bytes += len(data)
            if self.fsm.state == State.NEGOTIATING:
                s.stats.negotiate_bytes += len(data)
        self.sock._tx(ip, data)
        if not quiet:
            self.log("send", type=M(msg_type).name, slot=slot_index if msg_type in (M.DATA, M.ANCHOR) else None,
                     prio=prio if msg_type in (M.DATA, M.ANCHOR) else None, msg_id=msg_id,
                     flags=_flagstr(flags), nbytes=len(data))
        return pkt

    def send_slots(self, msg_type, slots: np.ndarray, protected: int, intent: int, bits: int,
                   keyed: bool, msg_id: int | None = None, max_retx: int | None = None, quiet=False) -> int:
        """Send one latent message: one slot per packet, in priority order (index order)."""
        n, d = slots.shape
        mid = next(self.msg_ids) & 0xFFFF if msg_id is None else msg_id
        P = max(0, min(protected, n))
        keyed = keyed and self.session.aead is not None
        pkts = []
        for i in range(n):
            x = slots[i]
            flags = (Flag.PROTECTED if i < P else 0) | (Flag.LAST_SLOT if i == n - 1 else 0) | (Flag.KEYED if keyed else 0)
            pkts.append(dict(msg_type=msg_type, payload=encode_slot(x, bits), flags=flags, intent=intent,
                             msg_id=mid, slot_index=i, slot_count=n, prio=i, slot_dim=d))
        o = _Out(msg_type, mid, n, P, pkts, max_retx=self.opts.max_retx if max_retx is None else max_retx)
        with self.lock:
            self.out[(int(msg_type), mid)] = o
        for p in pkts:
            self._send(**p, quiet=quiet)
        if P:
            o.timer = self.sock.timers.call_later(self.opts.rto, lambda: self._rto(o))
        else:
            o.done.set()
        return mid

    def _retransmit(self, o: _Out, missing):
        for i in missing:
            p = dict(o.packets[i])
            p["flags"] = p["flags"] | Flag.ACK_REQ
            o.retransmits += 1
            self.session.stats.retransmits += 1
            self._send(**p)
            self.log("retransmit", type=M(o.msg_type).name, slot=i, prio=i, msg_id=o.msg_id)

    def _rto(self, o: _Out):
        if o.done.is_set() or self.fsm.state == State.CLOSED:
            return
        missing = [i for i in range(o.P) if i not in o.acked]
        if not missing:
            o.done.set()
            return
        if o.retx >= o.max_retx:
            self.log("give_up", type=M(o.msg_type).name, msg_id=o.msg_id, info=f"missing {missing}")
            o.done.set()
            return
        o.retx += 1
        self._retransmit(o, missing)
        o.timer = self.sock.timers.call_later(self.opts.rto * (1.5 ** o.retx), lambda: self._rto(o))

    # ---------------------------------------------------------------- receive
    def on_packet(self, pkt: Packet):
        s = self.session
        s.stats.packets_recv += 1
        s.stats.bytes_recv += pkt.wire_len()
        t = pkt.msg_type
        if t == M.RESET:
            self.reset_reason = jload(pkt.payload).get("reason", "reset") if pkt.payload else "reset"
            self.log("recv", type="RESET", info=self.reset_reason)
            self.fsm.reset(f"RESET: {self.reset_reason}")
            self.inbox.put(pkt)
            with self.cv:
                self.cv.notify_all()
            return
        if t in (M.DATA, M.ANCHOR):
            if t == M.DATA and not self.fsm.accepts(t):
                if self.fsm.state in (State.KEYING, State.ESCROW):
                    # The peer reached ESTABLISHED first. Hold, deliver only once ESTABLISHED.
                    with self.lock:
                        self.held.append(pkt)
                    return
                self.log("drop", type="DATA", slot=pkt.slot_index, info=f"not ESTABLISHED ({self.fsm.state.value})")
                return
            self._on_slot(pkt)
            return
        if t == M.ACK:
            self._on_ack(pkt)
            return
        if t == M.VERDICT:
            self._on_verdict(pkt)
            return
        if t == M.FIN:
            self.log("recv", type="FIN")
            self._send(M.FIN_ACK)
            self.fsm.reset("FIN")
            with self.cv:
                self.cv.notify_all()
            return
        self.log("recv", type=M(t).name, msg_id=pkt.msg_id)
        self.inbox.put(pkt)

    def _on_slot(self, pkt: Packet):
        s = self.session
        key = (int(pkt.msg_type), pkt.msg_id)
        is_data = pkt.msg_type == M.DATA
        plain = pkt.payload
        if pkt.flags & Flag.KEYED:
            peer_role = "responder" if s.role == "initiator" else "initiator"
            try:
                if s.aead is None:
                    raise AuthError("keyed packet before KEY_CONFIRM")
                plain = s.aead.open(pkt, sender_direction(peer_role))
            except AuthError as e:
                s.stats.auth_failures += 1
                self.log("drop", type=M(pkt.msg_type).name, slot=pkt.slot_index, info=str(e))
                return
        elif is_data and s.aead is not None:
            s.stats.auth_failures += 1
            self.log("drop", type="DATA", slot=pkt.slot_index, info="unkeyed DATA on a keyed session")
            return
        with self.lock:
            m = self.inc.get(key)
            new = m is None
            if new:
                m = _In(int(pkt.msg_type), pkt.msg_id, pkt.slot_count, pkt.intent, first_at=time.monotonic())
                self.inc[key] = m
            i = pkt.slot_index
            m.bytes += pkt.wire_len()
            if i in m.vecs or m.final:
                m.duplicates += 1
            else:
                bits = s.bits if is_data else 16
                x = decode_slot(plain, pkt.slot_dim, bits)
                if is_data and s.translator is not None:
                    x = s.translator.apply_slot(i, x)
                m.vecs[i] = x
                m.order.append(i)
                if pkt.flags & Flag.PROTECTED:
                    m.prot_max = max(m.prot_max, i)
                else:
                    m.be_min = min(m.be_min, i)
            if pkt.flags & Flag.LAST_SLOT:
                m.last_seen = True
        if is_data:
            self.log("recv", type="DATA", slot=pkt.slot_index, prio=pkt.slot_priority, msg_id=pkt.msg_id,
                     flags=_flagstr(pkt.flags))
        if new:
            if is_data:
                self.data_q.put(key)
            m.ack_timer = self.sock.timers.call_later(self.opts.ack_delay, lambda: self._ack_timer(m))
        if pkt.flags & (Flag.LAST_SLOT | Flag.ACK_REQ):
            self._send_ack(m, quiet=not is_data)
        self._check_final(m)

    def _send_ack(self, m: _In, quiet=False):
        with self.lock:
            got = list(m.vecs.keys())
            m.ack_sent_at = time.monotonic()
        self._send(M.ACK, ack_payload(m.msg_type, m.count, got), msg_id=m.msg_id, quiet=quiet)
        self.sock.timers.call_later(self.opts.grace, lambda: self._check_final(m))

    def _ack_timer(self, m: _In):
        if not m.final and not m.last_seen:
            self._send_ack(m, quiet=m.msg_type != M.DATA)
        self._check_final(m)

    def _check_final(self, m: _In):
        with self.cv:
            if m.final:
                return
            have = m.vecs
            reason = None
            if len(have) == m.count:
                reason = "complete"
            elif m.ack_sent_at is not None:
                lo = m.prot_max + 1
                hi = min(m.be_min, m.count)
                prot_ok = all(i in have for i in range(lo))
                if prot_ok and (all(i in have for i in range(hi)) or time.monotonic() >= m.ack_sent_at + self.opts.grace - 1e-3):
                    reason = "protected"
            if reason:
                m.final, m.final_reason = True, reason
                if m.ack_timer:
                    Timers.cancel(m.ack_timer)
            self.cv.notify_all()

    def _on_ack(self, pkt: Packet):
        with self.lock:
            kind = pkt.payload[0]
            o = self.out.get((kind, pkt.msg_id))
        if o is None:
            return
        _, got = ack_parse(pkt.payload, o.count)
        o.acked |= got
        if kind == M.DATA:
            self.log("recv", type="ACK", msg_id=pkt.msg_id, info=f"bitmap {sorted(got)}")
        missing = [i for i in range(o.P) if i not in o.acked]
        if not missing:
            o.done.set()
            Timers.cancel(o.timer)
            return
        if o.retx >= o.max_retx:
            o.done.set()
            return
        Timers.cancel(o.timer)
        o.retx += 1
        self._retransmit(o, missing)
        o.timer = self.sock.timers.call_later(self.opts.rto * (1.5 ** o.retx), lambda: self._rto(o))

    def _on_verdict(self, pkt: Packet):
        v = jload(pkt.payload)
        self.verdicts.append(v)
        s = self.session
        if not v.get("match", True):
            s.flagged = True
            s.flagged_msgs.add(v.get("msg_id"))
            s.trust = self.base_trust() * self.opts.verdict_factor
        self.log("recv", type="VERDICT", msg_id=v.get("msg_id"),
                 info=f"declared={v.get('declared')} observed={v.get('observed')} conf={v.get('confidence', 0):.2f}")
        if self.on_verdict:
            self.on_verdict(v)
        with self.cv:
            self.cv.notify_all()

    # ---------------------------------------------------------------- trust
    def base_trust(self) -> float:
        s, o = self.session, self.opts
        path = 1.0 if not s.routed else (o.routed_factor if s.attested else o.unattested_factor)
        return float(max(0.0, min(1.0, s.anchor_score)) * path)

    def message_trust(self, msg_id: int | None = None) -> float:
        s = self.session
        t = self.base_trust()
        if s.flagged or (msg_id is not None and msg_id in s.flagged_msgs):
            t *= self.opts.verdict_factor
        return t

    # -------------------------------------------------------------- public API
    def send_latent(self, slots, protected: int | None = None, intent: int = INTENT_SHARE_VALUE) -> int:
        if not self.fsm.established:
            raise ConnectionReset(f"not established ({self.fsm.state.value})")
        x = slots.detach().cpu().numpy() if torch.is_tensor(slots) else np.asarray(slots)
        x = x[: self.session.N, : self.session.d].astype(np.float32)
        n = x.shape[0]
        rel = self.opts.reliability
        if rel == "udp":
            P, retx = 0, 0
        elif rel == "tcp":
            P, retx = n, self.opts.tcp_max_retx
        else:
            P, retx = (n // 2 if protected is None else protected), None
        return self.send_slots(M.DATA, x, P, intent, self.session.bits, keyed=True, max_retx=retx)

    def recv_latent(self, deadline: float = 0.2, timeout: float = 10.0):
        """Generator over one latent message: yields (slots [N,d], mask [N], final) after every
        slot arrival, then once more with final=True. deadline bounds the wait for stragglers,
        counted from the first slot's arrival."""
        s = self.session
        try:
            key = self.data_q.get(timeout=timeout)
        except queue.Empty:
            if self.fsm.state == State.CLOSED:
                raise ConnectionReset(self.reset_reason or "closed")
            raise TimeoutError("no latent message")
        m: _In = self.inc[key]
        end = m.first_at + deadline
        n_seen = 0
        tcp = self.opts.reliability == "tcp"
        while True:
            with self.cv:
                while (len(m.order) == n_seen and not m.final and time.monotonic() < end
                       and self.fsm.state != State.CLOSED):
                    self.cv.wait(max(0.0, min(HEARTBEAT, end - time.monotonic())))
                    self._check_final_nolock(m)
                order = list(m.order)
                vecs = dict(m.vecs)
                final = m.final or time.monotonic() >= end or self.fsm.state == State.CLOSED
                if final and not m.final:
                    m.final, m.final_reason = True, "deadline" if self.fsm.state != State.CLOSED else "closed"
            new = order[n_seen:]
            n_seen = len(order)
            for j, _ in enumerate(new):
                last = j == len(new) - 1
                if last and final:
                    break
                yield self._assemble(vecs, order[: n_seen - len(new) + j + 1], tcp) + (False,)
            if final:
                yield self._assemble(vecs, order, tcp) + (True,)
                return

    def _check_final_nolock(self, m):
        # called with the condition held; _check_final re-acquires the RLock safely
        self._check_final(m)

    def _assemble(self, vecs, present, tcp: bool):
        N, d = self.session.N, self.session.d
        slots = torch.zeros(N, d)
        mask = torch.zeros(N, dtype=torch.bool)
        pres = set(present)
        if tcp:  # in-order delivery: only the contiguous prefix is visible (head-of-line blocking)
            k = 0
            while k in pres:
                k += 1
            pres = set(range(k))
        for i in pres:
            if i < N:
                slots[i] = torch.from_numpy(vecs[i])
                mask[i] = True
        return slots, mask

    def last_message(self, kind=M.DATA) -> _In | None:
        with self.lock:
            ms = [m for (k, _), m in self.inc.items() if k == kind]
        return ms[-1] if ms else None

    def close(self):
        if self.fsm.state == State.ESTABLISHED:
            self.fsm.to(State.CLOSING)
            self._send(M.FIN)
            try:
                self._expect({M.FIN_ACK}, timeout=1.0)
            except (HandshakeError, TimeoutError):
                pass
            self.fsm.reset("closed")
        self.sock._drop_conn(self)

    # -------------------------------------------------------------- handshake
    def _expect(self, types, timeout: float) -> Packet:
        for k, p in enumerate(self.stash):
            if p.msg_type in types:
                return self.stash.pop(k)
        end = time.monotonic() + timeout
        while True:
            rem = end - time.monotonic()
            if rem <= 0:
                raise TimeoutError(f"waiting for {[M(t).name for t in types]}")
            try:
                p = self.inbox.get(timeout=rem)
            except queue.Empty:
                continue
            if p.msg_type == M.RESET:
                raise HandshakeError(f"peer RESET: {self.reset_reason}")
            if p.msg_type in types:
                return p
            self.stash.append(p)

    def _reset(self, reason: str):
        self._send(M.RESET, jdump({"reason": reason}))
        self.fsm.reset(reason)
        self.reset_reason = reason
        raise HandshakeError(reason)

    def _phase(self, name: str, **attrs):
        tr = self.sock.tracer
        if tr is None:
            return contextlib.nullcontext(None)
        return tr.span(name, tlp__session_id=self.session.session_id, **attrs)

    def handshake(self, negotiate: bool = True):
        tr = self.sock.tracer
        if tr is None:
            return self._handshake(negotiate)
        from ..otel.spans import ext_from_current, ext_from_span
        if self.session.role == "initiator":
            root = ext_from_current()          # the task span: B's packets carry it
            if root is not None:
                self.trace_ext = lambda: root
            with self._phase("handshake", tlp__role="initiator") as sp:
                self._handshake(negotiate, sp)
        else:
            with tr.attach(self.hello_ext):    # join the initiator's trace
                with self._phase("handshake", tlp__role="responder") as sp:
                    ext = ext_from_span(sp)
                    self.trace_ext = lambda: ext
                    self._handshake(negotiate, sp)

    def _handshake(self, negotiate: bool = True, span=None):
        t0 = time.monotonic()
        try:
            with self._phase("hello"):
                if self.session.role == "initiator":
                    self._hello_initiator(negotiate)
                else:
                    self._hello_responder()
            self._verify()
            with self._phase("key"):
                self._keying()
            with self._phase("escrow", tlp__audit=self.sock.audit):
                self._escrow()
        except HandshakeError as e:
            if span is not None:
                from ..otel.spans import TLPTracer
                TLPTracer.error(span, str(e))
            raise
        except TimeoutError as e:
            if self.fsm.state != State.CLOSED:
                try:
                    self._reset(f"timeout: {e}")
                except HandshakeError:
                    pass
            raise HandshakeError(f"timeout: {e}") from e
        finally:
            self.session.stats.handshake_seconds = time.monotonic() - t0
        s = self.session
        s.trust = self.base_trust()
        if span is not None:
            span.set_attribute("tlp.trust", s.trust)
            span.set_attribute("tlp.anchor_score", s.anchor_score)
            span.set_attribute("tlp.dialect.local", s.local_dialect)
            span.set_attribute("tlp.dialect.peer", s.peer_dialect)
        self.log("established", info=f"N={s.N} d={s.d} bits={s.bits} anchor={s.anchor_score:.3f} "
                                     f"negotiated={s.negotiated} routed={s.routed}")

    def _caps_payload(self, extra: dict) -> bytes:
        c = self.caps
        d = c.wire()
        d.update(extra)
        return jdump(d)

    def _hello_initiator(self, negotiate: bool):
        o = self.opts
        self.fsm.to(State.HELLO_SENT)
        self.hello = self._caps_payload({"nonce": os.urandom(8).hex(), "heldout": o.n_heldout,
                                         "fit": o.n_fit if (negotiate and o.negotiate) else 0,
                                         "translator": o.translator})
        self._send(M.HELLO, self.hello)
        p = self._expect({M.HELLO_ACK}, o.handshake_timeout)
        self.hello_ack = p.payload
        self._apply_peer_caps(jload(self.hello), jload(p.payload))

    def _hello_responder(self):
        p = self._expect({M.HELLO}, self.opts.handshake_timeout)
        self.fsm.to(State.HELLO_RCVD)
        self.hello = p.payload
        if p.flags & Flag.ROUTED:
            self.session.routed = True
        h = jload(p.payload)
        fit = h.get("fit", 0) if self.opts.negotiate else 0
        self.hello_ack = self._caps_payload({"nonce": os.urandom(8).hex(), "heldout": h.get("heldout", 32),
                                             "fit": fit, "translator": h.get("translator", "linear")})
        self._send(M.HELLO_ACK, self.hello_ack)
        self._apply_peer_caps(h, jload(self.hello_ack))

    def _apply_peer_caps(self, hello: dict, ack: dict):
        s = self.session
        mine, peer = (hello, ack) if s.role == "initiator" else (ack, hello)
        s.N = min(mine["N"], peer["N"])
        s.d = min(mine["d"], peer["d"])
        s.bits = min(mine["bits"], peer["bits"])
        s.local_dialect, s.peer_dialect, s.peer_model = mine["dialect"], peer["dialect"], peer["model"]
        self.n_heldout = ack["heldout"]
        self.n_fit = ack["fit"]
        self.family = ack.get("translator", "linear")
        self.fsm.to(State.VERIFYING)
        self.log("hello", info=f"peer model={peer['model']} dialect={peer['dialect']} local={mine['dialect']} "
                               f"N={s.N} d={s.d} bits={s.bits}" + (" MISMATCH" if s.local_dialect != s.peer_dialect else ""))
        if mine["d"] != peer["d"] and mine["dialect"] != peer["dialect"]:
            self._reset("dimension mismatch between dialects (out of scope for Telepathy-0)")

    def _exchange_anchors(self, which_ids: list[tuple[str, int, int]]) -> dict:
        """Send own anchor latents (in the clear, all PROTECTED) and collect the peer's.
        which_ids: list of (set, count, msg_id base). Returns {set: [n, N, d] np array}."""
        codec, s = self.caps.codec, self.session
        # Send window: at most ANCHOR_WINDOW anchor messages wait for their ACK at one time. Without
        # it, a 288-message burst on raw sockets (every raw socket on the host receives every packet)
        # puts the receivers behind the RTO, and the spurious retransmits make the backlog grow.
        sent = []
        for which, n, base in which_ids:
            lat = codec.encode(which, n)[:, : s.N, : s.d]
            for j in range(n):
                if len(sent) >= ANCHOR_WINDOW:
                    o = self.out.get((int(M.ANCHOR), sent[-ANCHOR_WINDOW]))
                    if o is not None:
                        o.done.wait(self.opts.handshake_timeout)
                    if self.fsm.state == State.CLOSED:
                        raise HandshakeError(f"peer RESET: {self.reset_reason}")
                self.send_slots(M.ANCHOR, lat[j], protected=s.N, intent=INTENT_ANCHOR_VERIFY, bits=16,
                                keyed=False, msg_id=base + j, quiet=True)
                sent.append(base + j)
        self.log("send", type="ANCHOR", info=f"{sum(n for _, n, _ in which_ids)} anchor messages x {s.N} slots (clear, PROTECTED)")
        want = {(int(M.ANCHOR), base + j) for _, n, base in which_ids for j in range(n)}
        end = time.monotonic() + self.opts.handshake_timeout
        with self.cv:
            while True:
                done = [k for k in want if k in self.inc and self.inc[k].final]
                if len(done) == len(want):
                    break
                if self.fsm.state == State.CLOSED:
                    raise HandshakeError(f"peer RESET: {self.reset_reason}")
                if time.monotonic() > end:
                    raise TimeoutError(f"anchors: {len(done)}/{len(want)}")
                self.cv.wait(HEARTBEAT)
        out = {}
        for which, n, base in which_ids:
            arr = np.zeros((n, s.N, s.d), np.float32)
            for j in range(n):
                for i, v in self.inc[(int(M.ANCHOR), base + j)].vecs.items():
                    arr[j, i] = v
            out[which] = arr
        return out

    def _verify(self):
        s, o, codec = self.session, self.opts, self.caps.codec
        same = s.local_dialect == s.peer_dialect
        if same or not self.n_fit:
            with self._phase("anchor", tlp__anchors=self.n_heldout) as sp:
                self._verify_direct(sp)
            return
        with self._phase("negotiate", tlp__dialect__local=s.local_dialect, tlp__dialect__peer=s.peer_dialect,
                         tlp__translator=self.family, tlp__anchors__fit=self.n_fit) as sp:
            self._verify_negotiate(sp)

    def _verify_direct(self, sp=None):
        s, o, codec = self.session, self.opts, self.caps.codec
        got = self._exchange_anchors([("heldout", self.n_heldout, 0)])
        score = codec.score(got["heldout"], "heldout")
        s.anchor_score = score
        self._send(M.ANCHOR_RESULT, jdump({"score": score, "n": self.n_heldout}))
        peer = jload(self._expect({M.ANCHOR_RESULT}, o.handshake_timeout).payload)
        s.peer_anchor_score = peer["score"]
        self.log("anchor", info=f"local score {score:.3f}, peer score {peer['score']:.3f}, threshold {o.anchor_threshold}")
        if sp is not None:
            sp.set_attribute("tlp.anchor.score", score)
            sp.set_attribute("tlp.anchor.peer_score", peer["score"])
        if score < o.anchor_threshold or peer["score"] < o.anchor_threshold:
            self._reset(f"anchor check failed (local {score:.3f}, peer {peer['score']:.3f})")
        self.fsm.to(State.KEYING)

    def _verify_negotiate(self, sp=None):
        """Different dialects: negotiate a translator from public anchors."""
        s, o, codec = self.session, self.opts, self.caps.codec
        from ..negotiate.translators import fit_translator
        t0 = time.monotonic()
        self.fsm.to(State.NEGOTIATING)
        got = self._exchange_anchors([("fit", self.n_fit, FIT_BASE), ("heldout", self.n_heldout, 0)])
        own_fit = codec.encode("fit", self.n_fit)[:, : s.N, : s.d]
        tr = fit_translator(self.family, got["fit"], own_fit, seed=self.opts.translator_seed)
        heldout = np.stack([tr.apply_message(m) for m in got["heldout"]])
        score = codec.score(heldout, "heldout")
        s.anchor_score = score
        s.stats.negotiate_seconds = time.monotonic() - t0
        self._send(M.NEGOTIATE, jdump({"family": self.family, "score": score, "fit": self.n_fit}))
        peer = jload(self._expect({M.NEGOTIATE}, o.handshake_timeout).payload)
        s.peer_anchor_score = peer["score"]
        self.log("negotiate", info=f"family={self.family} anchors={self.n_fit} held-out score local {score:.3f} "
                                   f"peer {peer['score']:.3f}")
        if sp is not None:
            sp.set_attribute("tlp.negotiate.heldout_score", score)
            sp.set_attribute("tlp.negotiate.peer_heldout_score", peer["score"])
            sp.set_attribute("tlp.negotiate.seconds", s.stats.negotiate_seconds)
        if score < o.anchor_threshold or peer["score"] < o.anchor_threshold:
            self._reset(f"negotiation failed (local {score:.3f}, peer {peer['score']:.3f})")
        s.translator, s.negotiated = tr, self.family
        self.fsm.to(State.VERIFYING, "negotiated")
        self.fsm.to(State.KEYING)

    def _keying(self):
        s, o = self.session, self.opts
        kx = KeyExchange()
        self._send(M.KEY, kx.pub)
        peer_pub = self._expect({M.KEY}, o.handshake_timeout).payload
        ipub, rpub = (kx.pub, peer_pub) if s.role == "initiator" else (peer_pub, kx.pub)
        keys, th = kx.derive(peer_pub, self.hello, self.hello_ack, ipub, rpub)
        me, them = (b"I", b"R") if s.role == "initiator" else (b"R", b"I")
        self._send(M.KEY_CONFIRM, confirm_tag(keys, th, me))
        tag = self._expect({M.KEY_CONFIRM}, o.handshake_timeout).payload
        if not confirm_ok(keys, th, them, tag):
            self._reset("key confirmation failed")
        s.keys = keys
        s.aead = PacketAEAD(keys.data_key, keys.nonce_salt)
        self.log("key", info="X25519 + HKDF ok, KEY_CONFIRM verified")

    def _escrow(self):
        s, aud = self.session, self.sock.auditor
        if not self.sock.audit:
            self.fsm.to(State.ESTABLISHED, "audit off")
            return
        if aud is None:
            self._reset("no auditor configured: no auditor, no session")
        self.fsm.to(State.ESCROW)
        if s.role == "initiator":  # the receiver escrows (Telepathy-0: the initiator receives latents)
            sid = s.session_id.to_bytes(4, "big")
            wrapped = wrap_for(aud.pub, s.keys.escrow_blob(), sid)
            body = {"sid": s.session_id, "peer": list(s.peer), "self": list(s.local), "role": s.role, "N": s.N, "d": s.d,
                    "bits": s.bits, "local_dialect": s.local_dialect, "peer_dialect": s.peer_dialect,
                    "routed": s.routed, "translator": s.negotiated, "wrapped": b64(wrapped)}
            self._send(M.KEY_ESCROW, jdump(body), dst=(aud.ip, aud.ep))
        try:
            self._expect({M.ESCROW_ACK}, self.opts.escrow_timeout)
        except TimeoutError:
            self._reset("no ESCROW_ACK: no auditor, no session")
        s.audited = True
        self.fsm.to(State.ESTABLISHED)


def _flagstr(f: int) -> str:
    return "|".join(n for n, v in (("P", Flag.PROTECTED), ("L", Flag.LAST_SLOT), ("K", Flag.KEYED),
                                   ("R", Flag.ROUTED), ("A", Flag.ACK_REQ), ("T", Flag.EXT_TRACE)) if f & v)


# --------------------------------------------------------------------- socket
class TLPSocket:
    def __init__(self, transport: str = "udp", injector: Injector | None = None, bus: InprocBus | None = None,
                 audit: bool = True, auditor: Auditor | None = None, opts: TLPOptions | None = None,
                 log=None, name: str = "", tracer=None):
        self.kind = transport
        self.tracer = tracer          # OpenTelepathy TLPTracer or None (tracing off)
        self.injector = injector
        self.bus = bus
        self.audit = audit
        self.auditor = auditor
        self.opts = opts or TLPOptions()
        self.name = name
        self._log = log
        self.addr: tuple[str, int] | None = None
        self.transport = None
        self.timers = Timers()
        self.conns: dict[int, Connection] = {}
        self.listening = False
        self.accept_q: queue.Queue = queue.Queue()
        self.packet_handler = None   # service mode: handler(pkt) for packets to this endpoint
        self.tap_handler = None      # promiscuous: handler(pkt) for every TLP packet seen
        self.extra_endpoints: set[int] = set()
        self._waiters: dict[int, queue.Queue] = {}
        self._closed = False
        self.conn: Connection | None = None
        self.rx_dropped = 0
        self.rx_injector = None      # receive-side loss injection (demo loss run)

    # ---------------------------------------------------------------- basics
    def log(self, ev, **kw):
        if self._log:
            self._log(self.name or f"ep{self.addr[1] if self.addr else '?'}", ev, kw)

    def bind(self, addr: tuple[str, int]):
        self.addr = (addr[0], int(addr[1]))
        self.transport = make_transport(self.kind, self.addr[0], self.injector, self.bus)
        if hasattr(self.transport, "add_endpoint"):
            self.transport.add_endpoint(self.addr[1])
        threading.Thread(target=self._rx_loop, daemon=True, name=f"tlp-rx-{self.addr[1]}").start()
        return self

    def add_endpoint(self, endpoint: int):
        """Also receive packets for this endpoint (a resolver's relay endpoints)."""
        self.extra_endpoints.add(endpoint)
        if hasattr(self.transport, "add_endpoint"):
            self.transport.add_endpoint(endpoint)

    def enable_tap(self, handler):
        """Promiscuous mode: handler(pkt) for every TLP packet seen. Raw and in-process transports
        see every packet already; over UDP the socket also listens on the tap port."""
        self.tap_handler = handler
        if hasattr(self.transport, "add_endpoint"):
            from ..tlp.header import TAP_EP
            self.transport.add_endpoint(TAP_EP, tapped=True)

    def _autobind(self, ip="127.0.0.1"):
        if self.addr is None:
            self.bind((ip, random.randint(20000, 60000)))

    def listen(self, backlog: int = 16):
        self.listening = True

    def _tx(self, ip: str, data: bytes):
        self.transport.send(ip, data)

    def send_packet(self, ip: str, pkt: Packet):
        self._tx(ip, pkt.encode())

    def _rx_loop(self):
        while not self._closed:
            r = self.transport.recv(0.1)
            if r is None:
                continue
            src_ip, dst_ip, data = r[:3]
            tapped = len(r) > 3 and r[3]
            try:
                pkt = Packet.decode(data)
            except PacketError:
                self.rx_dropped += 1
                continue
            pkt.src_ip, pkt.dst_ip = src_ip, dst_ip
            try:
                if tapped:   # UDP tap copy: only for the tap handler; own packets also arrive directly
                    own = pkt.dst_endpoint == self.addr[1] or pkt.dst_endpoint in self.extra_endpoints
                    if self.tap_handler and not own:
                        self.tap_handler(pkt)
                    continue
                self._dispatch(pkt)
            except Exception as e:
                print(f"[{self.name}] dispatch error: {e!r}")

    def _dispatch(self, pkt: Packet):
        if self.tap_handler:
            self.tap_handler(pkt)
        if pkt.dst_endpoint != self.addr[1] and pkt.dst_endpoint not in self.extra_endpoints:
            return
        if self.rx_injector is not None and self.rx_injector.should_drop(pkt):
            self.log("lost", type=M(pkt.msg_type).name, sid=pkt.session_id,
                     slot=pkt.slot_index if pkt.msg_type == M.DATA else None,
                     prio=pkt.slot_priority if pkt.msg_type == M.DATA else None, info="injected loss")
            return
        if self.addr[0] not in ("0.0.0.0", pkt.dst_ip) and self.kind == "raw":
            return
        if pkt.session_id in self._waiters:
            self._waiters[pkt.session_id].put(pkt)
            return
        if self.packet_handler:
            self.packet_handler(pkt)
            return
        conn = self.conns.get(pkt.session_id)
        if conn is not None:
            src = (pkt.src_ip, pkt.src_endpoint)
            aud = self.auditor
            from_auditor = aud is not None and src == (aud.ip, aud.ep)
            if src == conn.session.peer or (from_auditor and pkt.msg_type in (M.ESCROW_ACK, M.VERDICT, M.RESET)):
                conn.on_packet(pkt)
            else:
                self.rx_dropped += 1
            return
        if pkt.msg_type == M.HELLO and self.listening:
            c = Connection(self, pkt.session_id, (pkt.src_ip, pkt.src_endpoint), "responder", self._listen_caps)
            c.hello_ext = pkt.ext
            if pkt.flags & Flag.ROUTED:
                c.session.routed = True
            self.conns[pkt.session_id] = c
            c.on_packet(pkt)
            self.accept_q.put(c)
            return
        self.rx_dropped += 1

    def _drop_conn(self, c: Connection):
        self.conns.pop(c.session.session_id, None)

    # ---------------------------------------------------------- client side
    def set_capabilities(self, caps: Capabilities):
        self._listen_caps = caps

    def accept(self, timeout: float | None = None, capabilities: Capabilities | None = None):
        if capabilities is not None:
            self._listen_caps = capabilities
        c: Connection = self.accept_q.get(timeout=timeout)
        if c.caps is None:
            c.caps = self._listen_caps
        try:
            c.handshake()
        except HandshakeError:
            self._drop_conn(c)
            raise
        return c, c.session.peer

    def connect(self, addr: tuple[str, int], capabilities: Capabilities, negotiate: bool = True,
                routed: bool = False, attested: bool = False) -> Connection:
        self._autobind(addr[0])
        sid = random.getrandbits(32) or 1
        c = Connection(self, sid, (addr[0], int(addr[1])), "initiator", capabilities)
        c.session.routed, c.session.attested = routed, attested
        self.conns[sid] = c
        self.conn = c
        try:
            c.handshake(negotiate=negotiate)
        except HandshakeError:
            self._drop_conn(c)
            raise
        return c

    def request(self, dst: tuple[str, int], msg_type, obj: dict, reply_types, timeout=1.0, retries=3) -> tuple[Packet, dict]:
        self._autobind(dst[0])
        sid = random.getrandbits(32) or 1
        q: queue.Queue = queue.Queue()
        self._waiters[sid] = q
        try:
            ext = None
            if self.tracer is not None:
                from ..otel.spans import ext_from_current
                ext = ext_from_current()
            for _ in range(retries):
                self._tx(dst[0], Packet(msg_type, self.addr[1], dst[1], sid, 1, payload=jdump(obj), ext=ext).encode())
                self.log("send", type=M(msg_type).name, sid=sid)
                try:
                    p = q.get(timeout=timeout)
                except queue.Empty:
                    continue
                if p.msg_type in reply_types:
                    self.log("recv", type=M(p.msg_type).name, sid=sid)
                    return p, jload(p.payload)
            raise TimeoutError(f"no reply to {M(msg_type).name}")
        finally:
            self._waiters.pop(sid, None)

    def resolve(self, resolver: tuple[str, int], name: str | None = None, embedding=None) -> dict:
        obj = {"name": name} if name else {"emb": emb_encode(embedding)}
        _, rec = self.request(resolver, M.RESOLVE, obj, {M.RESOLVE_REPLY})
        if not rec.get("ok", False):
            raise LookupError(rec.get("error", "not found"))
        return rec

    def connect_by_knowledge(self, resolver, query_embedding, capabilities: Capabilities, negotiate=True):
        resolver = _res(resolver)
        rec = self.resolve(resolver, embedding=query_embedding)
        self.log("resolved", info=f"{rec['name']} at {rec['ip']}:{rec['endpoint']} score {rec['score']:.3f}")
        c = self.connect((rec["ip"], rec["endpoint"]), capabilities, negotiate=negotiate)
        c.resolved = rec
        return c

    def route(self, resolver, query_embedding, capabilities: Capabilities, negotiate=True):
        """Knowledge-addressed, relayed session: the resolver picks the holder and relays;
        this side never learns the holder's address."""
        resolver = _res(resolver)
        _, rep = self.request(resolver, M.ROUTE, {"emb": emb_encode(query_embedding)}, {M.RESOLVE_REPLY})
        if not rep.get("ok", False):
            raise LookupError(rep.get("error", "no route"))
        self.log("routed", info=f"relay {rep['ip']}:{rep['endpoint']} -> {rep['name']} (score {rep['score']:.3f})")
        c = self.connect((rep["ip"], rep["endpoint"]), capabilities, negotiate=negotiate, routed=True,
                         attested=bool(rep.get("model")))
        c.resolved = rep
        if c.session.peer_model != rep.get("model"):
            c.session.attested = False
            c.session.trust = c.base_trust()
        return c

    # ------------------------------------------------ connected-socket façade
    def send_latent(self, *a, **kw):
        return self.conn.send_latent(*a, **kw)

    def recv_latent(self, *a, **kw):
        return self.conn.recv_latent(*a, **kw)

    @property
    def session(self) -> Session:
        return self.conn.session

    def close(self):
        if self.conn is not None:
            self.conn.close()
        for c in list(self.conns.values()):
            try:
                c.close()
            except Exception:
                pass
        self._closed = True
        self.timers.stop()
        if self.transport:
            self.transport.close()


def _res(resolver):
    if resolver is None:
        return ("127.0.0.1", RESOLVER_EP)
    if isinstance(resolver, str):
        return (resolver, RESOLVER_EP)
    return tuple(resolver)
