"""Raw IP (protocol 253) transport tests. Skipped without raw-socket privileges
(root on macOS; root or CAP_NET_RAW on Linux). Run: sudo .venv/bin/pytest tests/test_raw.py"""
import threading
import time

import pytest
import torch

from telepathy.tlp.header import MsgType, Packet
from telepathy.tlp.transport import RawTransport, raw_available
from telepathy.sock.socket import TLPSocket

from conftest import caps, opts

pytestmark = pytest.mark.skipif(not raw_available(), reason="needs raw-socket privileges (root / CAP_NET_RAW)")


def test_raw_loopback_sees_own_and_peer_packets():
    a = RawTransport("127.0.0.1")
    b = RawTransport("127.0.0.1")
    try:
        pkt = Packet(MsgType.DATA, 4001, 4002, 0xABCD, 1, payload=b"latent!")
        a.send("127.0.0.1", pkt.encode())
        seen = {"a": False, "b": False}
        end = time.time() + 2
        while time.time() < end and not all(seen.values()):
            for name, t in (("a", a), ("b", b)):
                r = t.recv(0.05)
                if r:
                    src, dst, data = r
                    try:
                        q = Packet.decode(data)
                    except Exception:
                        continue
                    if q.session_id == 0xABCD and q.payload == b"latent!":
                        assert src == "127.0.0.1" and dst == "127.0.0.1"
                        seen[name] = True
        assert seen == {"a": True, "b": True}, "every raw socket must see every protocol-253 packet"
    finally:
        a.close()
        b.close()


def test_raw_handshake_and_transfer():
    server = TLPSocket("raw", audit=False, opts=opts(), name="server")
    server.bind(("127.0.0.1", 4107))
    server.listen()
    server.set_capabilities(caps(0))
    x = torch.randn(4, 8)
    out = {}

    def run():
        conn, _ = server.accept(timeout=10)
        conn.send_latent(x, protected=2)
        out["conn"] = conn

    t = threading.Thread(target=run)
    t.start()
    client = TLPSocket("raw", audit=False, opts=opts(), name="client")
    client.bind(("127.0.0.1", 4207))
    try:
        client.connect(("127.0.0.1", 4107), capabilities=caps(0))
        *_, (slots, mask, final) = list(client.recv_latent(deadline=1.0))
        t.join(5)
        assert final and mask.all()
        assert torch.allclose(slots, x.half().float(), atol=2e-3)
    finally:
        client.close()
        server.close()
