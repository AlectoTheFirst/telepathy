"""Span helpers: W3C trace context <-> TLP EXT_TRACE extension header, and the TLPTracer that
the socket layer uses for handshake spans."""
from __future__ import annotations

import contextlib

from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.trace import NonRecordingSpan, SpanContext, Status, StatusCode, TraceFlags

from ..config import INTENT_NAMES
from ..tlp.header import TraceExt


def ext_from_span(span) -> TraceExt | None:
    ctx = span.get_span_context() if span is not None else None
    if ctx is None or not ctx.is_valid:
        return None
    return TraceExt(ctx.trace_id.to_bytes(16, "big"), ctx.span_id.to_bytes(8, "big"), int(ctx.trace_flags) & 0xFF)


def ext_from_current() -> TraceExt | None:
    return ext_from_span(trace.get_current_span())


def context_from_ext(ext: TraceExt | None):
    """An OpenTelemetry Context whose parent is the remote span named in the extension."""
    if ext is None:
        return None
    sc = SpanContext(trace_id=int.from_bytes(ext.trace_id, "big"), span_id=int.from_bytes(ext.span_id, "big"),
                     is_remote=True, trace_flags=TraceFlags(ext.trace_flags))
    if not sc.is_valid:
        return None
    return trace.set_span_in_context(NonRecordingSpan(sc))


def intent_name(code: int) -> str:
    return INTENT_NAMES.get(int(code), str(code))


class TLPTracer:
    """Thin wrapper used by the socket layer and the agents."""

    def __init__(self, tracer, model_id: str = ""):
        self.tracer = tracer
        self.model_id = model_id

    @contextlib.contextmanager
    def span(self, name: str, parent_ext: TraceExt | None = None, **attrs):
        ctx = context_from_ext(parent_ext) if parent_ext is not None else None
        clean = {k.replace("__", "."): v for k, v in attrs.items() if v is not None}
        if self.model_id:
            clean.setdefault("model.id", self.model_id)
        with self.tracer.start_as_current_span(name, context=ctx, attributes=clean) as sp:
            yield sp

    @contextlib.contextmanager
    def attach(self, ext: TraceExt | None):
        """Make a remote context current (for a responder joining the initiator's trace)."""
        ctx = context_from_ext(ext)
        if ctx is None:
            yield
            return
        tok = otel_context.attach(ctx)
        try:
            yield
        finally:
            otel_context.detach(tok)

    @staticmethod
    def error(span, msg: str):
        span.set_status(Status(StatusCode.ERROR, msg))


class ResolverTracer(TLPTracer):
    """Resolver spans: resolve / route (children of the asker's span via EXT_TRACE) and one
    relay span per relayed session."""

    def __init__(self, tracer):
        super().__init__(tracer, "resolver")
        self._relayed = set()

    def resolver_span(self, kind, pkt, rec, score):
        with self.span(kind, parent_ext=pkt.ext, tlp__session_id=pkt.session_id,
                       resolver__match=rec.name if rec else "", resolver__score=float(score)):
            pass

    def relay(self, entry, pkt, direction):
        from ..tlp.header import MsgType
        if pkt.msg_type != MsgType.HELLO or entry.endpoint in self._relayed:
            return
        self._relayed.add(entry.endpoint)
        with self.span("relay", parent_ext=pkt.ext, tlp__session_id=pkt.session_id,
                       relay__endpoint=entry.endpoint, relay__holder=entry.holder_name,
                       relay__holder_session_id=entry.sid_h):
            pass


def print_trace_tree(path: str, out=None, only_trace: str | None = None, max_traces: int | None = None):
    """Print every trace in a JSONL span file as an indented tree."""
    import json
    import sys
    out = out or sys.stdout
    spans = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                spans.append(json.loads(line))
    by_trace = {}
    for s in spans:
        by_trace.setdefault(s["trace_id"], []).append(s)
    shown = 0
    for tid, ss in sorted(by_trace.items(), key=lambda kv: min(x["start"] for x in kv[1])):
        if only_trace and tid != only_trace:
            continue
        ids = {s["span_id"] for s in ss}
        kids = {}
        for s in ss:
            kids.setdefault(s["parent_id"] if s["parent_id"] in ids else None, []).append(s)
        roots = kids.get(None, [])
        services = sorted({s["service"] for s in ss})
        errors = sum(1 for s in ss if s["status"] == "ERROR")
        out.write(f"trace {tid}  spans={len(ss)}  services={','.join(services)}  errors={errors}\n")

        def show(s, depth):
            a = s["attributes"]
            keys = ("query", "answer", "tlp.session_id", "tlp.msg_id", "tlp.intent.declared", "tlp.intent.observed",
                    "tlp.intent.match", "tlp.trust", "tlp.slots.sent", "tlp.slots.received", "tlp.bytes",
                    "resolver.match", "relay.holder", "tlp.dialect.local", "tlp.dialect.peer", "tlp.translator",
                    "tlp.negotiate.heldout_score", "tlp.anchor.score", "tlp.dialect.read_in")
            info = " ".join(f"{k}={a[k]:.3f}" if isinstance(a[k], float) else f"{k}={a[k]}" for k in keys if k in a)
            mark = "  <ERROR>" if s["status"] == "ERROR" else ""
            dur = (s["end"] - s["start"]) / 1e6
            out.write(f"{'  ' * (depth + 1)}{s['name']} [{s['service']}] {dur:.1f}ms {info}{mark}\n")
            for e in s.get("events", []):
                out.write(f"{'  ' * (depth + 2)}! event {e['name']} {e['attributes']}\n")
            for k in sorted(kids.get(s["span_id"], []), key=lambda x: x["start"]):
                show(k, depth + 1)

        for r in sorted(roots, key=lambda x: x["start"]):
            show(r, 0)
        shown += 1
        if max_traces and shown >= max_traces:
            break
    return by_trace
