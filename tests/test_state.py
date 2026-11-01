import pytest
import torch

from telepathy.tlp.header import Flag, MsgType, Packet
from telepathy.tlp.state import State, StateError, StateMachine

from conftest import caps, make_pair, opts


def test_legal_path():
    sm = StateMachine()
    for s in (State.HELLO_SENT, State.VERIFYING, State.KEYING, State.ESCROW, State.ESTABLISHED,
              State.CLOSING, State.CLOSED):
        sm.to(s)
    assert sm.state == State.CLOSED


def test_negotiating_branch_and_illegal_transitions():
    sm = StateMachine()
    sm.to(State.HELLO_RCVD)
    sm.to(State.VERIFYING)
    sm.to(State.NEGOTIATING)
    sm.to(State.VERIFYING)
    with pytest.raises(StateError):
        sm.to(State.ESTABLISHED)  # must key (and escrow) first
    sm.reset("anchor fail")
    assert sm.state == State.CLOSED


def test_data_only_when_established():
    sm = StateMachine()
    assert not sm.accepts(MsgType.DATA)
    sm.to(State.HELLO_SENT)
    sm.to(State.VERIFYING)
    assert not sm.accepts(MsgType.DATA)


def test_direct_branch_on_dialect_match():
    s, c, conn, e1, e2, _ = make_pair(caps(0), caps(0))
    assert e1 is None and e2 is None
    hist = c.conn.fsm.history
    assert State.NEGOTIATING not in hist
    assert hist[-1] == State.ESTABLISHED
    assert c.session.anchor_score == 1.0
    assert c.session.negotiated is None
    c.close(), s.close()


def test_negotiating_branch_on_dialect_mismatch():
    # the fake dialects differ by a separate linear map per slot, so the per-slot linear family
    # fits them; the default shared family assumes one map for every slot and fails here
    o = opts(translator="linear")
    s, c, conn, e1, e2, _ = make_pair(caps(0), caps(1), server_opts=o, client_opts=o)
    assert e1 is None and e2 is None, (e1, e2)
    hist = c.conn.fsm.history
    assert State.NEGOTIATING in hist
    i = hist.index(State.NEGOTIATING)
    assert hist[i + 1] == State.VERIFYING and State.KEYING in hist[i:]
    assert c.session.negotiated == "linear"
    assert c.session.anchor_score >= 0.9
    c.close(), s.close()


def test_mismatch_without_negotiation_fails_at_anchor():
    s, c, conn, e1, e2, _ = make_pair(caps(0), caps(1), negotiate=False)
    assert e1 is not None and "anchor check failed" in str(e1)
    assert c.conn.fsm.state == State.CLOSED
    assert State.KEYING not in c.conn.fsm.history
    c.close(), s.close()


def test_data_before_established_is_dropped(pair):
    s, c, conn = pair
    # inject a DATA packet for an unknown session: dropped by demux
    before = s.rx_dropped
    pkt = Packet(MsgType.DATA, 99, 7, 12345, 1, flags=Flag.PROTECTED, slot_count=1, slot_dim=8, payload=b"\0" * 16)
    c.send_packet("127.0.0.1", pkt)
    import time
    time.sleep(0.1)
    assert s.rx_dropped > before


def test_audit_required_without_auditor_resets():
    s, c, conn, e1, e2, _ = make_pair(audit=True)
    assert e1 is not None and "no auditor" in str(e1)


def test_latent_round_trip_keyed(pair):
    s, c, conn = pair
    x = torch.randn(4, 8)
    conn.send_latent(x, protected=2)
    outs = list(c.recv_latent(deadline=1.0))
    slots, mask, final = outs[-1]
    assert final and mask.all()
    assert torch.allclose(slots, x.half().float(), atol=2e-3)
    # progressive: one yield per arrival, then final
    assert len(outs) >= 4
    assert c.session.keys.data_key == conn.session.keys.data_key
