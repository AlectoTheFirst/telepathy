import threading

import numpy as np
import torch

from telepathy.resolver.records import embed_keys
from telepathy.resolver.service import Resolver, register
from telepathy.sock.socket import TLPSocket
from telepathy.tlp.transport import InprocBus

from conftest import caps, opts, TRANSPORT

RES = ("127.0.0.1", 1)


def _holder(bus, name, ep, keys, dialect=0):
    s = TLPSocket(TRANSPORT, bus=bus, audit=False, opts=opts(), name=name)
    s.bind(("127.0.0.1", ep))
    s.listen()
    s.set_capabilities(caps(dialect))
    register(s, RES, name, ep, f"m{dialect}", f"dia{dialect}", embed_keys(keys), tags=["kv"])
    return s


def _serve_once(sock, payload, out):
    def run():
        conn, peer = sock.accept(timeout=10)
        out["peer"] = peer
        out["conn"] = conn
        conn.send_latent(payload, protected=2)
    t = threading.Thread(target=run)
    t.start()
    return t


def setup():
    bus = InprocBus()
    res = Resolver(transport=TRANSPORT, bus=bus, require_audit=False)
    a1 = _holder(bus, "A1", 11, range(0, 8))
    a2 = _holder(bus, "A2", 12, range(8, 16))
    return bus, res, a1, a2


def test_records_and_lookup():
    bus, res, a1, a2 = setup()
    b = TLPSocket(TRANSPORT, bus=bus, audit=False, opts=opts(), name="B")
    assert b.resolve(RES, name="A2")["endpoint"] == 12
    for k, want in ((3, "A1"), (7, "A1"), (8, "A2"), (15, "A2")):
        rec = b.resolve(RES, embedding=embed_keys([k]))
        assert rec["name"] == want and rec["score"] > 0.2
    for s in (a1, a2, b, res):
        s.close()


def test_connect_by_knowledge():
    bus, res, a1, a2 = setup()
    x = torch.randn(4, 8)
    out = {}
    t = _serve_once(a2, x, out)
    b = TLPSocket(TRANSPORT, bus=bus, audit=False, opts=opts(), name="B")
    conn = b.connect_by_knowledge(RES, embed_keys([12]), caps(0))
    *_, (slots, mask, final) = list(b.recv_latent(deadline=1.0))
    t.join(5)
    assert conn.resolved["name"] == "A2" and conn.session.peer == ("127.0.0.1", 12)
    assert torch.allclose(slots, x.half().float(), atol=2e-3)
    assert not conn.session.routed
    for s in (a1, a2, b, res):
        s.close()


def test_route_relays_end_to_end_and_hides_holder():
    bus, res, a1, a2 = setup()
    x = torch.randn(4, 8)
    out = {}
    t = _serve_once(a1, x, out)
    b = TLPSocket(TRANSPORT, bus=bus, audit=False, opts=opts(), name="B")
    conn = b.route(RES, embed_keys([5]), caps(0))
    *_, (slots, mask, final) = list(b.recv_latent(deadline=1.0))
    t.join(5)
    assert conn.resolved["name"] == "A1"
    # B only knows the relay; A1 sees the relay as its peer
    assert conn.session.peer[1] >= 40000 and conn.session.peer[1] != 11
    assert out["peer"][1] == conn.session.peer[1]
    assert conn.session.routed and out["conn"].session.routed
    assert torch.allclose(slots, x.half().float(), atol=2e-3)
    # keys were agreed end to end; the relay never held them
    assert conn.session.keys.data_key == out["conn"].session.keys.data_key
    entry = next(iter(res.relays.values()))
    assert entry.forwarded > 0 and entry.sid_i == conn.session.session_id and entry.sid_h != entry.sid_i
    # routed trust uses the resolver's record as attestation
    assert abs(conn.session.trust - 0.9) < 1e-6
    for s in (a1, a2, b, res):
        s.close()
