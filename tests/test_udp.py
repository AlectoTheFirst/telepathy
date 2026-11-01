"""TLP over UDP: the datagram is the unchanged TLP packet; endpoints map to ports; the tap port
gets a copy of every datagram for the probe."""
import socket
import time

from telepathy.sock.socket import TLPSocket
from telepathy.tlp.header import TAP_EP, MsgType, Packet
from telepathy.tlp.transport import UDP_BASE, udp_port


def test_port_mapping():
    assert udp_port(1) == UDP_BASE + 1 and udp_port(TAP_EP) == UDP_BASE + TAP_EP
    assert udp_port(40001) == 40001      # relay endpoints keep their number (base + 40001 > 65535)


def test_datagram_is_the_tlp_packet_and_tap_gets_a_copy():
    got, tapped = [], []
    rx = TLPSocket("udp", audit=False, name="rx")
    rx.bind(("127.0.0.1", 901))
    rx.packet_handler = got.append
    tap = TLPSocket("udp", audit=False, name="tap")
    tap.bind(("127.0.0.1", 902))
    tap.enable_tap(tapped.append)
    pkt = Packet(MsgType.FIN, 905, 901, 77, 5, payload=b"hello")
    raw = pkt.encode()
    # a plain UDP sender: no TLP socket needed, the datagram is the packet
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.sendto(raw, ("127.0.0.1", udp_port(901)))
    s.close()
    tx = TLPSocket("udp", audit=False, name="tx")
    tx.bind(("127.0.0.1", 905))
    tx.send_packet("127.0.0.1", pkt)            # through the transport: direct + tap copy
    end = time.time() + 2
    while (len(got) < 2 or not tapped) and time.time() < end:
        time.sleep(0.01)
    assert len(got) == 2 and all(p.encode() == raw for p in got)
    assert len(tapped) == 1 and tapped[0].session_id == 77 and tapped[0].dst_endpoint == 901
    for x in (rx, tap, tx):
        x.close()
