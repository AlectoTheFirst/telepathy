"""EXT_TRACE round trip and trace context propagation across a TLP session."""
import json
import threading

import torch
from opentelemetry import trace

from telepathy.otel.setup import init_tracing
from telepathy.otel.spans import TLPTracer, context_from_ext, ext_from_span
from telepathy.sock.socket import TLPSocket
from telepathy.tlp.header import Flag, MsgType, Packet, TraceExt
from telepathy.tlp.transport import InprocBus

from conftest import caps, opts, TRANSPORT


def test_ext_from_span_round_trip(tmp_path):
    tr = init_tracing("t", str(tmp_path / "s.jsonl"))
    with tr.start_as_current_span("root") as sp:
        ext = ext_from_span(sp)
        pkt = Packet.decode(Packet(MsgType.DATA, 1, 2, 3, 4, payload=b"x", ext=ext).encode())
        assert pkt.flags & Flag.EXT_TRACE and pkt.ext == ext
        ctx = context_from_ext(pkt.ext)
        sc = trace.get_current_span(ctx).get_span_context()
        assert sc.trace_id == sp.get_span_context().trace_id and sc.span_id == sp.get_span_context().span_id
        assert sc.is_remote


def test_context_propagates_through_session(tmp_path):
    path = str(tmp_path / "spans.jsonl")
    bus = InprocBus()
    ta, tb = TLPTracer(init_tracing("A", path), "A"), TLPTracer(init_tracing("B", path), "B")
    s = TLPSocket(TRANSPORT, bus=bus, audit=False, opts=opts(), tracer=ta)
    s.bind(("127.0.0.1", 7))
    s.listen()
    s.set_capabilities(caps(0))
    out = {}

    def acc():
        conn, _ = s.accept(timeout=5)
        with ta.span("latent.send", parent_ext=conn.hello_ext) as sp:
            ext = ext_from_span(sp)
            conn.trace_ext = lambda: ext
            conn.send_latent(torch.randn(4, 8))
        out["conn"] = conn

    th = threading.Thread(target=acc)
    th.start()
    c = TLPSocket(TRANSPORT, bus=bus, audit=False, opts=opts(), tracer=tb)
    with tb.span("task") as task:
        conn = c.connect(("127.0.0.1", 7), caps(0))
        data_exts = []
        for _ in conn.recv_latent(deadline=1.0):
            pass
        th.join(5)
    s.close(), c.close()
    spans = [json.loads(x) for x in open(path)]
    tid = f"{task.get_span_context().trace_id:032x}"
    names = {(x["service"], x["name"]) for x in spans if x["trace_id"] == tid}
    # one trace across both parties: B's task/handshake and A's handshake + latent.send
    assert ("B", "task") in names and ("B", "handshake") in names and ("B", "anchor") in names
    assert ("A", "handshake") in names and ("A", "latent.send") in names
    send = next(x for x in spans if x["name"] == "latent.send")
    task_id = f"{task.get_span_context().span_id:016x}"
    assert send["parent_id"] == task_id     # A's span is a child of B's task (context from HELLO)
