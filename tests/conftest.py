"""Shared fixtures: a fake anchor codec (no checkpoints needed) and socket pairs on TLP_TRANSPORT."""
import os
import threading

import numpy as np
import pytest

from telepathy.sock.socket import TLPSocket
from telepathy.tlp.session import Capabilities, TLPOptions
from telepathy.tlp.transport import InprocBus

# The suite runs on any transport: TLP_TRANSPORT=udp (default) or inproc. Raw needs root (test_raw.py).
TRANSPORT = os.environ.get("TLP_TRANSPORT", "udp")


@pytest.fixture(autouse=True)
def close_sockets(monkeypatch):
    """Close every TLP socket a test binds. Over UDP an open socket keeps its port."""
    opened = []
    real_bind = TLPSocket.bind

    def bind(self, addr):
        opened.append(self)
        return real_bind(self, addr)
    monkeypatch.setattr(TLPSocket, "bind", bind)
    yield
    for s in opened:
        try:
            s.close()
        except Exception:
            pass


class FakeCodec:
    """A synthetic dialect: anchor j has canonical latent Z[j]; dialect k encodes it as
    Z[j] @ D_k (per-slot orthogonal map). The codec reads latents by mapping back with its
    own D and matching the nearest canonical anchor."""

    def __init__(self, dialect_seed: int, N=4, d=8):
        self.N, self.d = N, d
        rng = np.random.default_rng(dialect_seed)
        self.D = [np.linalg.qr(rng.standard_normal((d, d)))[0].astype(np.float32) for _ in range(N)]

    @staticmethod
    def canon(which, n, N, d):
        rng = np.random.default_rng(1 if which == "fit" else 2)
        return rng.standard_normal((n, N, d)).astype(np.float32)

    def encode(self, which, n):
        Z = self.canon(which, n, self.N, self.d)
        return np.stack([Z[:, i] @ self.D[i] for i in range(self.N)], 1)

    def score(self, latents, which):
        n = latents.shape[0]
        Z = self.canon(which, n, self.N, self.d)
        back = np.stack([latents[:, i] @ self.D[i].T for i in range(self.N)], 1)
        dist = ((back[:, None] - Z[None]) ** 2).sum((2, 3))
        return float((dist.argmin(1) == np.arange(n)).mean())


def caps(dialect_seed=0, N=4, d=8, bits=16):
    return Capabilities(N=N, d=d, bits=bits, model_id=f"m{dialect_seed}", dialect_id=f"dia{dialect_seed}",
                        codec=FakeCodec(dialect_seed, N, d))


def opts(**kw):
    base = dict(n_heldout=8, n_fit=64, rto=0.05, ack_delay=0.03, grace=0.05, handshake_timeout=5.0)
    base.update(kw)
    return TLPOptions(**base)


def make_pair(server_caps=None, client_caps=None, injector=None, negotiate=True, audit=False,
              server_opts=None, client_opts=None, ep=7):
    bus = InprocBus()
    server = TLPSocket(TRANSPORT, bus=bus, audit=audit, opts=server_opts or opts(), name="server")
    server.bind(("127.0.0.1", ep))
    server.listen()
    server.set_capabilities(server_caps or caps(0))
    client = TLPSocket(TRANSPORT, bus=bus, audit=audit, opts=client_opts or opts(), injector=injector, name="client")
    client.bind(("127.0.0.1", ep + 100))
    out = {}

    def acc():
        try:
            out["conn"], _ = server.accept(timeout=5)
        except Exception as e:
            out["err"] = e

    t = threading.Thread(target=acc)
    t.start()
    err = None
    try:
        client.connect(("127.0.0.1", ep), capabilities=client_caps or caps(0), negotiate=negotiate)
    except Exception as e:
        err = e
    t.join(10)
    return server, client, out.get("conn"), err, out.get("err"), bus


@pytest.fixture
def pair():
    s, c, conn, e1, e2, bus = make_pair()
    assert e1 is None and e2 is None, (e1, e2)
    yield s, c, conn
    c.close()
    s.close()
