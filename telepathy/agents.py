"""Knowledge holders (A) and the asker (B) on top of TLPSocket. Used by the demo and experiments."""
from __future__ import annotations

import dataclasses
import sys
import threading
import time

import numpy as np
import torch

from .config import INTENT_NAMES, INTENT_SHARE_ENTRY_COUNT, INTENT_SHARE_KEY_EXISTENCE, INTENT_SHARE_VALUE
from .models.agent import load_ckpt
from .negotiate.anchors import model_capabilities
from .otel.spans import TLPTracer, ext_from_span, intent_name
from .resolver.records import embed_keys
from .resolver.service import register
from .sock.socket import HandshakeError, TLPSocket
from .tlp.messages import jload

T0 = time.time()


def make_logger(proc: str, out=None, t0: float | None = None, quiet_types=("ACK",)):
    """Interleaved protocol log: timestamp, process, event, message type, session, slot, priority, trust."""
    out = out or sys.stdout
    base = T0 if t0 is None else t0

    def log(name, ev, kw):
        if kw.get("type") in quiet_types and ev in ("send", "recv"):
            return
        parts = [f"{time.time() - base:8.3f}", f"{proc:<9}", f"{ev:<12}"]
        if kw.get("type"):
            parts.append(f"{kw['type']:<13}")
        if kw.get("sid") is not None:
            parts.append(f"sid={kw['sid']:08x}")
        if kw.get("msg_id") is not None and kw.get("type") in ("DATA", "ACK", "VERDICT") or ev == "intent.check":
            parts.append(f"msg={kw.get('msg_id')}")
        if kw.get("slot") is not None:
            parts.append(f"slot={kw['slot']}")
        if kw.get("prio") is not None:
            parts.append(f"prio={kw['prio']}")
        if kw.get("flags"):
            parts.append(f"[{kw['flags']}]")
        if kw.get("trust") is not None and ev not in ("send",) or ev == "established":
            parts.append(f"trust={kw.get('trust', 0):.2f}")
        if kw.get("info"):
            parts.append(str(kw["info"]))
        out.write(" ".join(parts) + "\n")
        out.flush()
    return log


def holder_table(keys, seed: int, value_range: int = 100):
    rng = np.random.default_rng(seed)
    keys = sorted(int(k) for k in keys)
    return keys, [int(v) for v in rng.integers(0, value_range, len(keys))]


class Holder:
    """A knowledge holder: registers NAME + K records, accepts sessions, pushes its table as a
    latent message encoded for the purpose the asker requested (or a different one if lying)."""

    def __init__(self, name, ckpt, side, keys, ep, values_seed, transport="udp", bus=None, ip="127.0.0.1",
                 auditor=None, tracer=None, log=None, opts=None, protected=None, lie_intent: int | None = None,
                 audit=True, bits=16, lie_for: dict | None = None):
        self.name, self.ep, self.ip = name, ep, ip
        self.lie_for = lie_for or {}       # requested purpose -> purpose actually encoded
        self.caps = model_capabilities(ckpt, side, bits=bits)
        models, meta, td = load_ckpt(ckpt)
        self.model, self.td, self.meta = models[side], td, meta
        self.keys, self.vals = holder_table(keys, values_seed, td.task.value_range)
        self.tr = TLPTracer(tracer, self.caps.model_id) if tracer else None
        self.sock = TLPSocket(transport, bus=bus, audit=audit, auditor=auditor, opts=opts, log=log, name=name,
                              tracer=self.tr)
        self.sock.bind((ip, ep))
        self.sock.listen()
        self.sock.set_capabilities(self.caps)
        self.protected = protected
        self.lie_intent = lie_intent       # encode under this intent while declaring the requested one
        self.sent = []

    def register(self, resolver=("127.0.0.1", 1)):
        return register(self.sock, resolver, self.name, self.ep, self.caps.model_id, self.caps.dialect_id,
                        embed_keys(self.keys), tags=["kv", f"k{self.keys[0]}-k{self.keys[-1]}"], ip=self.ip)

    @torch.no_grad()
    def encode(self, intent: int):
        it = torch.tensor([intent])
        keys = torch.tensor([self.keys + [-1] * (self.td.T - len(self.keys))])
        vals = torch.tensor([self.vals + [-1] * (self.td.T - len(self.vals))])
        return self.model.encode(self.td.a_tokens(keys, vals, it), it)[0]

    def serve_forever(self):
        def loop():
            while True:
                try:
                    conn, _ = self.sock.accept()
                except HandshakeError:
                    continue
                except Exception:
                    return
                threading.Thread(target=self.serve_conn, args=(conn,), daemon=True).start()
        th = threading.Thread(target=loop, daemon=True)
        th.start()
        return th

    def serve_conn(self, conn, n_messages: int = 1, gap: float = 0.0):
        purpose = jload(conn.hello).get("purpose", INTENT_SHARE_VALUE)
        for k in range(n_messages):
            used = self.lie_for.get(purpose, purpose)
            if self.lie_intent is not None:
                used = self.lie_intent
            slots = self.encode(used)
            if self.tr:
                with self.tr.span("latent.send", parent_ext=conn.hello_ext, tlp__session_id=conn.session.session_id,
                                  tlp__intent__declared=intent_name(purpose), tlp__slots__sent=conn.session.N) as sp:
                    ext = ext_from_span(sp)
                    conn.trace_ext = lambda: ext
                    before = conn.session.stats.bytes_sent
                    mid = conn.send_latent(slots, protected=self.protected, intent=purpose)
                    sp.set_attribute("tlp.msg_id", mid)
                    sp.set_attribute("tlp.bytes", conn.session.stats.bytes_sent - before)
            else:
                mid = conn.send_latent(slots, protected=self.protected, intent=purpose)
            self.sent.append({"msg_id": mid, "declared": purpose, "encoded": used})
            if used != purpose:
                conn.log("note", info=f"(injected) encoded under {intent_name(used)}, declared {intent_name(purpose)}")
            if gap:
                time.sleep(gap)

    def close(self):
        self.sock.close()


class Asker:
    """The receiver B: gets a query, reaches a holder, reads the latent message progressively."""

    def __init__(self, ckpt, side="B", transport="udp", bus=None, ip="127.0.0.1", ep=30, auditor=None,
                 tracer=None, log=None, opts=None, audit=True, bits=16):
        self.caps = model_capabilities(ckpt, side, bits=bits)
        models, meta, td = load_ckpt(ckpt)
        self.model, self.td = models[side], td
        self.tr = TLPTracer(tracer, self.caps.model_id) if tracer else None
        self.kw = dict(transport=transport, bus=bus, audit=audit, auditor=auditor, opts=opts, log=log, tracer=self.tr)
        self.ip, self.ep = ip, ep
        self.log = log
        self.n = 0

    def _span(self, name, **attrs):
        import contextlib
        return self.tr.span(name, **attrs) if self.tr else contextlib.nullcontext(None)

    @torch.no_grad()
    def _answer(self, b_tokens, slots, mask, trust):
        logits, gate = self.model.answer(b_tokens, slots[None], mask[None], torch.tensor([trust]))
        return int(logits.argmax(-1)), float(gate)

    def ask(self, query_key: int | None, intent: int = INTENT_SHARE_VALUE, path: str = "resolve", name=None,
            resolver=("127.0.0.1", 1), deadline: float = 0.3, negotiate: bool = True, n_messages: int = 1,
            rx_loss=None, route_keys=None, await_verdict: float = 0.0):
        """await_verdict: keep the session open up to this many seconds for the probe's VERDICT
        (the probe audits a negotiated session only after it fits its own translator)."""
        caps = dataclasses.replace(self.caps, purpose=intent)
        sock = TLPSocket(name="B", **self.kw)
        sock.bind((self.ip, self.ep + self.n))
        sock.rx_injector = rx_loss
        self.n += 1
        it = torch.tensor([intent])
        qk = torch.tensor([query_key if query_key is not None else -1])
        b_tokens = self.td.b_tokens(it, qk, torch.tensor([-1]))
        qtext = {INTENT_SHARE_VALUE: f"k{query_key}=?", INTENT_SHARE_KEY_EXISTENCE: f"k{query_key}?",
                 INTENT_SHARE_ENTRY_COUNT: "count?"}[intent]
        result = {"query": qtext, "path": path, "progress": [], "messages": []}
        emb = embed_keys(route_keys if route_keys is not None else ([query_key] if query_key is not None else []))
        with self._span("task", query=qtext, path=path) as task:
            try:
                if path == "route":
                    with self._span("route"):
                        conn = sock.route(resolver, emb, caps, negotiate=negotiate)
                elif path == "name":
                    with self._span("resolve", by="name"):
                        rec = sock.resolve(resolver, name=name)
                    conn = sock.connect((rec["ip"], rec["endpoint"]), caps, negotiate=negotiate)
                    conn.resolved = rec
                else:
                    with self._span("resolve", by="knowledge"):
                        rec = sock.resolve(resolver, embedding=emb)
                    sock.log("resolved", info=f"{rec['name']} at {rec['ip']}:{rec['endpoint']} score {rec['score']:.3f}")
                    conn = sock.connect((rec["ip"], rec["endpoint"]), caps, negotiate=negotiate)
                    conn.resolved = rec
            except HandshakeError as e:
                result.update(ok=False, error=str(e))
                sock.close()
                return result
            result["holder"] = conn.resolved["name"]
            result["session"] = conn.session
            for k in range(n_messages):
                msg = self._recv_one(conn, b_tokens, deadline, sock)
                result["messages"].append(msg)
            last = result["messages"][-1] if result["messages"] else {}
            result.update(ok=bool(last), answer=last.get("answer"), answer_text=last.get("answer_text"),
                          trust=conn.session.trust)
            if task is not None and last:
                task.set_attribute("answer", last.get("answer_text", ""))
            time.sleep(0.05)
            end = time.monotonic() + await_verdict
            while await_verdict and not conn.verdicts and time.monotonic() < end:
                time.sleep(0.05)
            result["verdicts"] = list(conn.verdicts)
            conn.close()
        sock.close()
        return result

    def _recv_one(self, conn, b_tokens, deadline, sock):
        t0 = time.monotonic()
        prog = []
        with self._span("latent.recv", tlp__session_id=conn.session.session_id) as sp:
            try:
                for slots, mask, final in conn.recv_latent(deadline=deadline):
                    m = conn.last_message()
                    trust = conn.message_trust(m.msg_id if m else None)
                    a, gate = self._answer(b_tokens, slots, mask, trust)
                    txt = self.td.ans.decode(a)
                    prog.append({"slots": int(mask.sum()), "answer": a, "text": txt, "final": final, "trust": trust,
                                 "gate": gate, "t": time.monotonic() - t0})
                    sock.log("answer", sid=conn.session.session_id, trust=trust,
                             info=f"{'FINAL ' if final else ''}after {int(mask.sum())}/{conn.session.N} slots: "
                                  f"{txt} (gate {gate:.2f})")
            except Exception as e:
                sock.log("error", info=repr(e))
                return {}
            m = conn.last_message()
            if sp is not None and m is not None:
                sp.set_attribute("tlp.msg_id", m.msg_id)
                sp.set_attribute("tlp.slots.received", len(m.vecs))
                sp.set_attribute("tlp.bytes", m.bytes)
                sp.set_attribute("tlp.trust", prog[-1]["trust"] if prog else 0.0)
                sp.set_attribute("tlp.intent.declared", intent_name(m.intent))
                if conn.verdicts:
                    v = [x for x in conn.verdicts if x.get("msg_id") == m.msg_id]
                    if v:
                        sp.set_attribute("tlp.intent.observed", intent_name(v[-1]["observed"]))
                        sp.set_attribute("tlp.intent.match", bool(v[-1]["match"]))
        if not prog:
            return {}
        with self._span("answer", answer=prog[-1]["text"], tlp__trust=prog[-1]["trust"]):
            pass
        return {"answer": prog[-1]["answer"], "answer_text": prog[-1]["text"], "progress": prog,
                "msg_id": m.msg_id if m else None, "slots_received": len(m.vecs) if m else 0,
                "bytes": m.bytes if m else 0, "time_first": prog[0]["t"], "time_final": prog[-1]["t"]}
