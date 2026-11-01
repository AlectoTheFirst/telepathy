"""Transports behind the TLP socket API: TLP over UDP (the default), raw IP sockets
(protocol 253) and an in-process bus, plus the loss/delay injector that sits between the API
and the transport.

UDP carries the identical TLP packet (header + payload) as one datagram; see UdpTransport.

The raw and in-process transports deliver every TLP packet to every attached socket, including
the sender's own, exactly like raw sockets do. UDP delivers a packet to its destination port only,
plus a copy to the tap port. Demultiplexing happens in the TLP header (dst_endpoint,
session_id), never in the transport.
"""
from __future__ import annotations

import heapq
import itertools
import os
import queue
import random
import socket
import struct
import sys
import threading
import time

from .header import IPPROTO_TLP, TAP_EP


# --------------------------------------------------------------------- injector
class Injector:
    """Seeded loss and one-way delay. drop_fn(data) -> bool forces specific drops (tests, demo)."""

    def __init__(self, loss: float = 0.0, delay: float = 0.0, seed: int = 0, drop_fn=None, jitter: float = 0.0):
        self.loss, self.delay, self.jitter = loss, delay, jitter
        self.rng = random.Random(seed)
        self.drop_fn = drop_fn
        self.dropped = 0
        self.passed = 0
        self._heap: list = []
        self._cv = threading.Condition()
        self._ctr = itertools.count()
        self._thread = None
        self._stop = False

    def submit(self, deliver, data: bytes):
        if (self.drop_fn and self.drop_fn(data)) or (self.loss and self.rng.random() < self.loss):
            self.dropped += 1
            return
        self.passed += 1
        d = self.delay + (self.rng.uniform(0, self.jitter) if self.jitter else 0.0)
        if d <= 0:
            deliver(data)
            return
        with self._cv:
            heapq.heappush(self._heap, (time.monotonic() + d, next(self._ctr), deliver, data))
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, daemon=True, name="injector")
                self._thread.start()
            self._cv.notify()

    def _run(self):
        while True:
            with self._cv:
                while not self._heap and not self._stop:
                    self._cv.wait()
                if self._stop:
                    return
                t, _, deliver, data = self._heap[0]
                now = time.monotonic()
                if t > now:
                    self._cv.wait(t - now)
                    continue
                heapq.heappop(self._heap)
            deliver(data)

    def close(self):
        with self._cv:
            self._stop = True
            self._cv.notify()


class RxLoss:
    """Receive-side loss for one socket: random loss on DATA plus optional forced first-copy drops."""

    def __init__(self, loss: float, seed: int = 0, force_slots=(), types=None):
        self.loss = loss
        self.rng = random.Random(seed)
        self.force = set(force_slots)
        self.types = types
        self.dropped = []

    def should_drop(self, pkt) -> bool:
        from .header import MsgType
        types = self.types or (MsgType.DATA,)
        if pkt.msg_type not in types:
            return False
        if pkt.slot_index in self.force:
            self.force.discard(pkt.slot_index)
            self.dropped.append(pkt.slot_index)
            return True
        if self.rng.random() < self.loss:
            self.dropped.append(pkt.slot_index)
            return True
        return False


# ------------------------------------------------------------------- transports
class Transport:
    ip: str

    def send(self, dst_ip: str, data: bytes) -> None: ...
    def recv(self, timeout: float | None) -> tuple[str, str, bytes] | None: ...
    def close(self) -> None: ...


class InprocBus:
    def __init__(self):
        self.lock = threading.Lock()
        self.members: list["InprocTransport"] = []
        self.packets = 0
        self.bytes = 0

    def attach(self, t):
        with self.lock:
            self.members.append(t)

    def detach(self, t):
        with self.lock:
            if t in self.members:
                self.members.remove(t)

    def broadcast(self, src_ip, dst_ip, data):
        with self.lock:
            members = list(self.members)
            self.packets += 1
            self.bytes += len(data)
        for m in members:
            m.q.put((src_ip, dst_ip, data))


DEFAULT_BUS = InprocBus()


class InprocTransport(Transport):
    def __init__(self, ip: str = "127.0.0.1", bus: InprocBus | None = None, injector: Injector | None = None):
        self.ip = ip
        self.bus = bus or DEFAULT_BUS
        self.injector = injector
        self.q: queue.Queue = queue.Queue()
        self.bus.attach(self)
        self.closed = False

    def send(self, dst_ip, data):
        deliver = lambda d: self.bus.broadcast(self.ip, dst_ip, d)
        if self.injector:
            self.injector.submit(deliver, data)
        else:
            deliver(data)

    def recv(self, timeout):
        try:
            return self.q.get(timeout=timeout)
        except queue.Empty:
            return None

    def close(self):
        self.closed = True
        self.bus.detach(self)


IS_BSD = sys.platform == "darwin" or "bsd" in sys.platform


def raw_available() -> bool:
    """True if this process may open a raw IP socket (root on macOS; root or CAP_NET_RAW on Linux)."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_RAW, IPPROTO_TLP)
        s.close()
        return True
    except (PermissionError, OSError):
        return False


def build_ip_header(src: str, dst: str, payload_len: int, ident: int = 0, ttl: int = 64) -> bytes:
    """IPv4 header for IP_HDRINCL sends. BSD-derived stacks (macOS) expect ip_len and ip_off
    in host byte order; Linux expects network byte order. The kernel fills the checksum."""
    total = 20 + payload_len
    if IS_BSD:
        len_off = struct.pack("=HH", total, 0)
    else:
        len_off = struct.pack("!HH", total, 0)
    return (struct.pack("!BB", 0x45, 0) + len_off[:2] + struct.pack("!H", ident) + len_off[2:]
            + struct.pack("!BBH4s4s", ttl, IPPROTO_TLP, 0, socket.inet_aton(src), socket.inet_aton(dst)))


class RawTransport(Transport):
    """TLP directly over IP (protocol 253). Every raw socket for protocol 253 on the host sees
    every TLP packet, including its own sends."""

    def __init__(self, ip: str = "127.0.0.1", injector: Injector | None = None, hdrincl: bool = False):
        self.ip = ip
        self.injector = injector
        self.hdrincl = hdrincl
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_RAW, IPPROTO_TLP)
        if hdrincl:
            self.sock.setsockopt(socket.IPPROTO_IP, socket.IP_HDRINCL, 1)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 << 20)
        self._ident = itertools.count(1)
        self.closed = False

    def send(self, dst_ip, data):
        def deliver(d):
            if self.closed:
                return
            if self.hdrincl:
                d = build_ip_header(self.ip, dst_ip, len(d), next(self._ident) & 0xFFFF) + d
            self.sock.sendto(d, (dst_ip, 0))
        if self.injector:
            self.injector.submit(deliver, data)
        else:
            deliver(data)

    def recv(self, timeout):
        self.sock.settimeout(timeout)
        try:
            data, _ = self.sock.recvfrom(65535)
        except (socket.timeout, BlockingIOError):
            return None
        except OSError:
            if self.closed:
                return None
            raise
        # Both Linux and BSD deliver the IP header on raw receive. Strip it by IHL; do not
        # trust ip_len (BSD may report it in host order and without the header length).
        if len(data) < 20:
            return None
        ihl = (data[0] & 0x0F) * 4
        if data[9] != IPPROTO_TLP:
            return None
        src = socket.inet_ntoa(data[12:16])
        dst = socket.inet_ntoa(data[16:20])
        return src, dst, data[ihl:]

    def close(self):
        self.closed = True
        try:
            self.sock.close()
        except OSError:
            pass


# ------------------------------------------------------------------ TLP over UDP
UDP_BASE = int(os.environ.get("TLP_UDP_BASE", "47000"))


def udp_port(endpoint: int, base: int | None = None) -> int:
    """UDP port of a TLP endpoint. Endpoints below 1024 (the well-known resolver 1, probe 2 and
    tap 3, and small test endpoints) map to base + endpoint, off the privileged ports; the other
    endpoints use the same number as their port. Both are 16-bit, so base + endpoint would
    overflow for relay endpoints (40000 and up)."""
    base = UDP_BASE if base is None else base
    return base + endpoint if endpoint < 1024 else endpoint


class UdpTransport(Transport):
    """TLP over UDP: each TLP packet is one UDP datagram, unchanged. The destination port comes
    from the dst_endpoint field of the TLP header, so the socket API does not change. One UDP
    socket per bound TLP endpoint (a resolver that relays binds more).

    The tap: every datagram is also copied to the tap port (TAP_EP), as a network port mirror
    would do. The intent probe listens there. recv() marks tap copies with tapped=True, so the
    socket gives them only to its tap handler."""

    def __init__(self, ip: str = "127.0.0.1", injector: Injector | None = None, base: int | None = None,
                 tap: bool = True):
        import selectors
        self.ip = ip
        self.injector = injector
        self.base = UDP_BASE if base is None else base
        self.tap = tap
        self.sel = selectors.DefaultSelector()
        self.socks: dict[int, socket.socket] = {}
        self.out = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.closed = False
        self._lock = threading.Lock()

    def add_endpoint(self, endpoint: int, tapped: bool = False):
        port = udp_port(endpoint, self.base)
        with self._lock:
            if port in self.socks:
                return
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 << 20)
            s.bind((self.ip if self.ip != "0.0.0.0" else "", port))
            s.setblocking(False)
            self.socks[port] = s
            self.sel.register(s, 1, tapped)

    def send(self, dst_ip, data):
        dst_ep = struct.unpack_from("!H", data, 6)[0]
        port = udp_port(dst_ep, self.base)

        def deliver(d):
            if self.closed:
                return
            try:
                self.out.sendto(d, (dst_ip, port))
                if self.tap and dst_ep != TAP_EP:
                    self.out.sendto(d, (dst_ip, udp_port(TAP_EP, self.base)))
            except OSError:
                pass   # no listener: UDP is fire and forget
        if self.injector:
            self.injector.submit(deliver, data)
        else:
            deliver(data)

    def recv(self, timeout):
        if not self.socks:
            time.sleep(timeout or 0.01)
            return None
        for key, _ in self.sel.select(timeout):
            try:
                data, (src, _) = key.fileobj.recvfrom(65535)
            except (BlockingIOError, OSError):
                continue
            return src, self.ip, data, bool(key.data)
        return None

    def close(self):
        self.closed = True
        with self._lock:
            for s in self.socks.values():
                try:
                    self.sel.unregister(s)
                except Exception:
                    pass
                s.close()
            self.socks.clear()
        self.out.close()


def make_transport(kind: str, ip: str, injector: Injector | None = None, bus: InprocBus | None = None) -> Transport:
    if kind == "udp":
        return UdpTransport(ip, injector)
    if kind == "inproc":
        return InprocTransport(ip, bus, injector)
    if kind == "raw":
        return RawTransport(ip, injector, hdrincl=os.environ.get("TLP_HDRINCL") == "1")
    raise ValueError(kind)
