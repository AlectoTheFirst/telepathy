"""OpenTelepathy tracer setup: OpenTelemetry SDK with an OTLP exporter (optional) and a JSONL
file exporter (the SDK ships a console exporter, not a file one)."""
from __future__ import annotations

import json
import os
import threading

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor, SpanExporter, SpanExportResult


class JsonlSpanExporter(SpanExporter):
    """One JSON object per finished span, appended to a file. Lines are written with a single
    O_APPEND write, so several processes can share one file."""

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.lock = threading.Lock()

    def export(self, spans) -> SpanExportResult:
        lines = []
        for s in spans:
            ctx = s.get_span_context()
            lines.append(json.dumps({
                "name": s.name,
                "service": s.resource.attributes.get("service.name", ""),
                "trace_id": f"{ctx.trace_id:032x}",
                "span_id": f"{ctx.span_id:016x}",
                "parent_id": f"{s.parent.span_id:016x}" if s.parent else None,
                "start": s.start_time, "end": s.end_time,
                "status": s.status.status_code.name,
                "attributes": {k: (list(v) if isinstance(v, tuple) else v) for k, v in s.attributes.items()},
                "events": [{"name": e.name, "attributes": dict(e.attributes)} for e in s.events],
            }, default=str) + "\n")
        data = "".join(lines).encode()
        with self.lock:
            fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
            try:
                os.write(fd, data)
            finally:
                os.close(fd)
        return SpanExportResult.SUCCESS

    def shutdown(self):
        pass


_providers: list[TracerProvider] = []


def init_tracing(service: str, jsonl_path: str | None = None, otlp_endpoint: str | None = None):
    """Return a tracer for one party (resolver, A1, A2, B, probe). Uses its own provider so
    several parties in one process keep distinct service names."""
    provider = TracerProvider(resource=Resource.create({"service.name": service}))
    if jsonl_path:
        provider.add_span_processor(SimpleSpanProcessor(JsonlSpanExporter(jsonl_path)))
    ep = otlp_endpoint or os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    if ep:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=ep.rstrip("/") + "/v1/traces")))
    _providers.append(provider)
    return provider.get_tracer("telepathy")


def shutdown_tracing():
    for p in _providers:
        p.shutdown()
    _providers.clear()
