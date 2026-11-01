"""Semantic loss handling: protected slots are retransmitted, best-effort slots are not."""
import torch

from telepathy.tlp.header import MsgType, Packet
from telepathy.tlp.transport import Injector

from conftest import make_pair


class DropOnce:
    """Drop the first copy of DATA packets with the given slot indices (receiver-bound)."""

    def __init__(self, slots, msg_type=MsgType.DATA):
        self.pending = set(slots)
        self.msg_type = msg_type

    def __call__(self, data):
        p = Packet.decode(data)
        if p.msg_type == self.msg_type and p.slot_index in self.pending:
            self.pending.discard(p.slot_index)
            return True
        return False


def _pair_with_server_drops(drop):
    # the server sends DATA: give the *server* an injector by swapping its transport's injector
    s, c, conn, e1, e2, bus = make_pair()
    assert e1 is None and e2 is None
    s.transport.injector = Injector(drop_fn=drop)
    return s, c, conn


def test_protected_slot_retransmitted_best_effort_dropped():
    drop = DropOnce({1, 3})   # N=4, P=2: slot 1 protected, slot 3 best-effort (and LAST_SLOT)
    s, c, conn = _pair_with_server_drops(drop)
    x = torch.randn(4, 8)
    mid = conn.send_latent(x, protected=2)
    *_, (slots, mask, final) = list(c.recv_latent(deadline=1.0))
    assert final
    assert mask.tolist() == [True, True, True, False]
    out = conn.out[(int(MsgType.DATA), mid)]
    assert out.retransmits == 1
    assert conn.session.stats.retransmits == 1
    m = c.conn.inc[(int(MsgType.DATA), mid)]
    assert m.final_reason == "protected"
    c.close(), s.close()


def test_last_slot_lost_ack_timer():
    drop = DropOnce({3})      # LAST_SLOT lost: receiver ACKs on its timer, nothing to retransmit
    s, c, conn = _pair_with_server_drops(drop)
    conn.send_latent(torch.randn(4, 8), protected=2)
    *_, (slots, mask, final) = list(c.recv_latent(deadline=1.0))
    assert final and mask.tolist() == [True, True, True, False]
    assert conn.session.stats.retransmits == 0
    c.close(), s.close()


def test_progressive_delivery_yields_each_slot():
    s, c, conn, *_ = make_pair()
    conn.send_latent(torch.randn(4, 8), protected=2)
    outs = list(c.recv_latent(deadline=1.0))
    counts = [int(m.sum()) for _, m, _ in outs]
    assert counts == sorted(counts) and counts[-1] == 4
    assert outs[-1][2] and not any(f for _, _, f in outs[:-1])
    c.close(), s.close()


def test_deadline_bounds_wait():
    drop = DropOnce({0, 1, 2, 3})
    s, c, conn = _pair_with_server_drops(drop)
    s.opts.max_retx = 0
    conn.opts.max_retx = 0
    conn.send_latent(torch.randn(4, 8), protected=4)
    # all first copies lost and no retransmits: nothing arrives
    import pytest
    with pytest.raises(TimeoutError):
        next(c.recv_latent(deadline=0.1, timeout=0.3))
    c.close(), s.close()


def test_tcp_and_udp_disciplines():
    for rel, expect_retx in (("udp", 0), ("tcp", 1)):
        drop = DropOnce({2})
        s, c, conn = _pair_with_server_drops(drop)
        conn.opts.reliability = rel
        c.opts.reliability = rel
        conn.send_latent(torch.randn(4, 8))
        *_, (slots, mask, final) = list(c.recv_latent(deadline=1.0))
        if rel == "udp":
            assert mask.tolist() == [True, True, False, True]
        else:
            assert mask.all()
        assert conn.session.stats.retransmits == expect_retx
        c.close(), s.close()


def test_byte_message_keyed_and_retransmitted():
    """send_bytes: one keyed DATA packet with slot_dim = 0. TCP-like retransmits it when lost,
    and the receiver gets the bytes back unchanged."""
    payload = bytes(range(256)) * 2
    drop = DropOnce({0})
    s, c, conn = _pair_with_server_drops(drop)
    conn.opts.reliability = "tcp"
    c.opts.reliability = "tcp"
    conn.send_bytes(payload)
    assert c.recv_bytes(timeout=5.0) == payload
    assert conn.session.stats.retransmits == 1
    assert conn.session.stats.auth_failures == 0 and c.session.stats.auth_failures == 0
    c.close(), s.close()
