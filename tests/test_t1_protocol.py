"""Telepathy-1 protocol additions (SPEC.md 14): T records, hub mode, slot batching.
No language model is loaded: the dialects are the synthetic FakeCodec of conftest."""
import io
import time

import numpy as np
import pytest
import torch

from telepathy.negotiate.translators import LinearSlot
from telepathy.resolver.records import TRecordStore
from telepathy.resolver.service import Resolver, TRecordClient, t_lookup, t_publish
from telepathy.sock.socket import FIT_BASE, TLPSocket, max_batch
from telepathy.tlp.header import MAX_PACKET, MsgType, Packet
from telepathy.tlp.messages import Reassembler, frag_encode
from telepathy.tlp.state import State
from telepathy.tlp.transport import Injector, InprocBus

from conftest import TRANSPORT, FakeCodec, caps, make_pair, opts


# --------------------------------------------------------------------------- T records

def test_t_record_store_publish_lookup_miss_ttl_stale():
    st = TRecordStore()
    st.publish("diaA", "diaB", "shared_mlp", b"blob", 0.97, ttl=0.2)
    r = st.lookup("diaA", "diaB", "shared_mlp")
    assert r is not None and r.blob == b"blob" and r.score == pytest.approx(0.97)
    assert st.lookup("diaB", "diaA", "shared_mlp") is None          # direction is part of the key
    assert st.lookup("diaA", "diaB", "procrustes") is None          # family is part of the key
    assert st.lookup("diaA2", "diaB", "shared_mlp") is None         # new weights, new dialect id: miss
    time.sleep(0.25)
    assert st.lookup("diaA", "diaB", "shared_mlp") is None          # expired


def test_fragments_round_trip():
    blob = bytes(range(256)) * 400                                  # 102 400 bytes, many fragments
    frags = frag_encode({"k": 1}, blob)
    assert len(frags) > 50
    ra = Reassembler()
    for k in reversed(range(len(frags))):                           # any order, with duplicates
        ra.add(k, frags[k])
        ra.add(k, frags[k])
    meta, got = ra.done()
    assert meta == {"k": 1} and got == blob
    for k, f in enumerate(frags):
        Packet(MsgType.T_REPLY, 1, 2, 3, k + 1, msg_id=k, payload=f).encode()   # each fits one packet


def _resolver(ep=1):
    bus = InprocBus()
    return Resolver(transport=TRANSPORT, bus=bus, endpoint=ep), bus


def test_t_record_over_the_wire():
    res, bus = _resolver(ep=611)
    try:
        cli = TLPSocket(TRANSPORT, bus=bus, audit=False, name="cli")
        cli.bind(("127.0.0.1", 612))
        dst = ("127.0.0.1", 611)
        blob = np.random.default_rng(0).standard_normal(20000).astype("<f2").tobytes()   # 40 kB
        ok, nbytes = t_publish(cli, dst, "diaA", "diaB", "shared_mlp", blob, 0.95, ttl=0.5)
        assert ok and nbytes > len(blob)
        got, _ = t_lookup(cli, dst, "diaA", "diaB", "shared_mlp")
        assert got is not None and got[0] == blob and got[1] == pytest.approx(0.95)
        assert t_lookup(cli, dst, "diaA", "diaC", "shared_mlp")[0] is None        # stale or unknown dialect
        time.sleep(0.6)
        assert t_lookup(cli, dst, "diaA", "diaB", "shared_mlp")[0] is None        # TTL expired
    finally:
        res.close()


def _dump_linear(tr) -> bytes:
    buf = io.BytesIO()
    np.savez(buf, W=np.stack(tr.W))
    return buf.getvalue()


def _load_linear(blob: bytes):
    tr = LinearSlot.__new__(LinearSlot)
    tr.W = list(np.load(io.BytesIO(blob), allow_pickle=False)["W"])
    return tr


def _anchor_ids(conn):
    return sorted(mid for (k, mid) in conn.inc if k == int(MsgType.ANCHOR))


def test_t_record_hit_skips_fit_and_fitting_anchors():
    res, bus = _resolver(ep=621)
    try:
        dst = ("127.0.0.1", 621)

        def session(ep):
            stores = []

            def client_opts():
                o = opts(translator="linear")
                return o
            so, co = client_opts(), client_opts()
            server = TLPSocket(TRANSPORT, bus=bus, audit=False, opts=so, name="server")
            server.bind(("127.0.0.1", ep))
            server.listen()
            server.set_capabilities(caps(0))
            client = TLPSocket(TRANSPORT, bus=bus, audit=False, opts=co, name="client")
            client.bind(("127.0.0.1", ep + 100))
            so.t_records = TRecordClient(server, dst, _dump_linear, _load_linear)
            co.t_records = TRecordClient(client, dst, _dump_linear, _load_linear)
            import threading
            out = {}
            t = threading.Thread(target=lambda: out.update(conn=server.accept(timeout=5)[0]))
            t.start()
            c = client.connect(("127.0.0.1", ep), capabilities=caps(1))
            t.join(10)
            return server, client, out["conn"], c, so.t_records, co.t_records

        s1, c1, sc1, cc1, st1, ct1 = session(631)
        assert cc1.session.t_record == "miss" and sc1.session.t_record == "miss"
        assert any(m >= FIT_BASE for m in _anchor_ids(cc1))             # the fitting anchors were sent
        assert st1.publishes == 1 and ct1.publishes == 1
        x = torch.from_numpy(FakeCodec(0, 4, 8).encode("heldout", 1)[0])
        sc1.send_latent(x, protected=4)
        *_, (slots1, mask1, _) = list(c1.recv_latent(deadline=1.0))
        s1.close(), c1.close()

        s2, c2, sc2, cc2, st2, ct2 = session(641)
        assert cc2.session.t_record == "hit" and sc2.session.t_record == "hit"
        assert all(m < FIT_BASE for m in _anchor_ids(cc2))              # held-out anchors only
        assert len(_anchor_ids(cc2)) == opts().n_heldout
        assert cc2.session.anchor_score >= 0.9                          # the held-out check still ran
        sc2.send_latent(x, protected=4)
        *_, (slots2, mask2, _) = list(c2.recv_latent(deadline=1.0))
        assert mask2.all() and torch.allclose(slots1, slots2, atol=1e-3)
        s2.close(), c2.close()
    finally:
        res.close()


# --------------------------------------------------------------------------- hub mode

class HubCodec(FakeCodec):
    """Hub coordinates = the canonical anchor space of FakeCodec: each dialect maps into it with its
    own per-slot rotation, without any knowledge of the peer."""

    def to_hub(self, x):
        return np.stack([x[:, i] @ self.D[i].T for i in range(self.N)], 1)

    def from_hub(self, h):
        return np.stack([h[:, i] @ self.D[i] for i in range(self.N)], 1)


def hub_caps(seed, anchors="public-v1", N=4, d=8):
    c = caps(seed, N=N, d=d, hub=True, anchor_set_id=anchors, hub_dim=d)
    c.codec = HubCodec(seed, N, d)
    return c


def test_hub_mode_skips_negotiate():
    s, c, conn, e1, e2, _ = make_pair(hub_caps(0), hub_caps(1))
    assert e1 is None and e2 is None, (e1, e2)
    for sess, fsm in ((c.session, c.conn.fsm), (conn.session, conn.fsm)):
        assert sess.hub == "public-v1" and sess.negotiated == "hub"
        assert State.NEGOTIATING not in fsm.history
        assert sess.stats.negotiate_bytes == 0
    assert all(m < FIT_BASE for m in _anchor_ids(c.conn))               # no fitting anchors
    z = FakeCodec.canon("heldout", 1, 4, 8)
    x = HubCodec(0).from_hub(z)[0]                                      # the server's own encoding
    conn.send_latent(torch.from_numpy(x), protected=4)
    *_, (slots, mask, final) = list(c.recv_latent(deadline=1.0))
    want = HubCodec(1).from_hub(z)[0]                                   # the client's own encoding
    assert final and mask.all()
    assert np.abs(slots.numpy() - want).max() < 1e-2
    c.close(), s.close()


def test_hub_mode_falls_back_to_negotiate_on_other_anchor_set():
    o = opts(translator="linear")
    s, c, conn, e1, e2, _ = make_pair(hub_caps(0, "public-v1"), hub_caps(1, "public-v2"),
                                      server_opts=o, client_opts=opts(translator="linear"))
    assert e1 is None and e2 is None, (e1, e2)
    assert c.session.hub is None and c.session.negotiated == "linear"
    assert State.NEGOTIATING in c.conn.fsm.history
    c.close(), s.close()


# --------------------------------------------------------------------------- slot batching

class DropFirstSlot:
    """Drop the first copy of each DATA packet whose first slot index is in `starts`."""

    def __init__(self, starts):
        self.pending = set(starts)

    def __call__(self, data):
        p = Packet.decode(data)
        if p.msg_type == MsgType.DATA and p.slot_index in self.pending:
            self.pending.discard(p.slot_index)
            return True
        return False


def _groups(n, P, M):
    return [(a, min(a + M, hi)) for lo, hi in ((0, P), (P, n)) for a in range(lo, hi, M)]


@pytest.mark.parametrize("M", [1, 4, 8])
def test_batching_round_trip_and_overhead(M):
    N, P = 10, 5                       # N is not a multiple of 4 or 8
    s, c, conn, e1, e2, _ = make_pair(caps(0, N=N, batch=M), caps(0, N=N, batch=M))
    assert e1 is None and e2 is None
    assert conn.session.batch == M and c.session.batch == M
    x = torch.randn(N, 8)
    st = conn.session.stats
    p0, k0 = st.data_packets, st.keyed_packets
    conn.send_latent(x, protected=P)
    *_, (slots, mask, final) = list(c.recv_latent(deadline=1.0))
    assert final and mask.all()
    assert torch.allclose(slots, x.half().float())
    n_pkts = len(_groups(N, P, M))
    assert st.data_packets - p0 == n_pkts                # one header and one tag per packet
    assert st.keyed_packets - k0 == n_pkts
    assert n_pkts == {1: 10, 4: 4, 8: 2}[M]
    c.close(), s.close()


@pytest.mark.parametrize("M", [1, 4, 8])
def test_batching_loss_treats_packet_as_unit(M):
    N, P = 10, 5
    s, c, conn, e1, e2, _ = make_pair(caps(0, N=N, batch=M), caps(0, N=N, batch=M))
    assert e1 is None and e2 is None
    s.transport.injector = Injector(drop_fn=DropFirstSlot({0, P}))   # a protected and a best-effort packet
    conn.send_latent(torch.randn(N, 8), protected=P)
    *_, (slots, mask, final) = list(c.recv_latent(deadline=1.0))
    lost = set(range(P, min(P + M, N)))                              # the best-effort packet's slots
    assert final
    assert mask.tolist() == [i not in lost for i in range(N)]
    assert conn.session.stats.retransmits == 1                        # the protected packet, once
    c.close(), s.close()


def test_batch_capped_by_packet_size():
    assert max_batch(128, 16, 8) == 5                    # 256-byte slots: 5 fit in 1400 bytes
    assert max_batch(32, 16, 4) == 4                     # negotiated M is the cap
    s, c, conn, e1, e2, _ = make_pair(caps(0, N=8, d=128, batch=8), caps(0, N=8, d=128, batch=8))
    assert e1 is None and e2 is None
    conn.send_latent(torch.randn(8, 128), protected=8)
    *_, (slots, mask, final) = list(c.recv_latent(deadline=1.0))
    assert mask.all()
    assert conn.session.stats.data_packets == 2          # 5 + 3 slots
    c.close(), s.close()


def test_batch_negotiated_as_min():
    s, c, conn, e1, e2, _ = make_pair(caps(0, batch=8), caps(0, batch=2))
    assert e1 is None and e2 is None
    assert conn.session.batch == 2 and c.session.batch == 2
    c.close(), s.close()
