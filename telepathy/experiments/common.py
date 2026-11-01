"""Shared helpers for the experiment scripts: models, eval sets, byte accounting, plots, results."""
from __future__ import annotations

import json
import os
import threading
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from ..config import RESULTS_DIR  # noqa: E402
from ..models.agent import load_ckpt  # noqa: E402
from ..models.quant import slot_payload_len  # noqa: E402
from ..crypto.aead import TAG_LEN  # noqa: E402
from ..tlp.header import EXT, HDR  # noqa: E402

# Transport for every protocol experiment: TLP over UDP by default (TLP_TRANSPORT=inproc also works).
TRANSPORT = os.environ.get("TLP_TRANSPORT", "udp")
DEFAULT = "latent_d32_s0"
PRIME = "latent_d32_s1"
# d=32 pair trained with Zipf(1) value queries and a weaker positional prior (1.0), 12000 steps:
# the only Zipf run that both orders entries by popularity and learns most of the table (6.3, 6.7)
ZIPF = "latent_d32_s0_zipf1_pb1"
D_SWEEP = (2, 4, 8, 16, 32)
N_SWEEP = (1, 2, 4, 8)
B_SWEEP = (16, 8, 4, 2)
# Reference categorical order (blue, orange, aqua, yellow), validated for CVD separation; the
# contrast WARN on aqua/yellow is covered by legends, markers and the JSON tables in results/.
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
NEUTRAL = "#8a8a85"
COLORS = {"latent": CAT[0], "text": CAT[1], "nolink": NEUTRAL, "tlp": CAT[0], "udp": CAT[1], "tcp": CAT[2],
          "linear": CAT[0], "procrustes": CAT[1], "mlp": CAT[2], "shared_mlp": CAT[3], "full_linear": NEUTRAL,
          # bits per dimension: ordinal one-hue ramp, darker = more bits
          "b16": "#104281", "b8": "#256abf", "b4": "#5598e7", "b2": "#9ec5f4"}
MARKERS = {"latent": "o", "text": "s", "nolink": "x", "tlp": "o", "udp": "s", "tcp": "^",
           "linear": "o", "procrustes": "s", "mlp": "^", "shared_mlp": "v", "full_linear": "D"}

_cache = {}


def models(name: str):
    if name not in _cache:
        _cache[name] = load_ckpt(name)
    return _cache[name]


def latent_wire_bytes(N: int, d: int, bits: int, trace: bool = False, keyed: bool = True) -> int:
    """Bytes on the wire for one latent message: one TLP packet per slot, header included, plus the
    16-byte AEAD tag per packet when keyed (DATA always is)."""
    return N * (HDR.size + (EXT.size if trace else 0) + slot_payload_len(d, bits) + (TAG_LEN if keyed else 0))


def latent_payload_bytes(N: int, d: int, bits: int) -> int:
    return N * slot_payload_len(d, bits)


def latent_payload_bits(N: int, d: int, bits: int) -> int:
    """Budget rule for every comparison: payload bytes on the wire x 8, including the per-slot
    float16 scale (b < 16); headers excluded. The AEAD tag is excluded on both sides, because an
    encrypted text or codec link would carry the same tag."""
    return 8 * latent_payload_bytes(N, d, bits)


# Fixed lossless binary codec baseline: 4-bit key index + 7-bit value per entry (88 bits for K = 8),
# entries in key order (as the text link), truncated by whole entries.
CODEC_KEY_BITS, CODEC_VALUE_BITS = 4, 7
CODEC_ENTRY_BITS = CODEC_KEY_BITS + CODEC_VALUE_BITS


def codec_acc(td, batch, budget_bits) -> float:
    """Accuracy of an exact decoder of the codec at a bit budget on value queries: correct if the
    queried entry fits in the budget, else a uniform guess over the values."""
    k = int(budget_bits // CODEC_ENTRY_BITS)
    keys = batch.keys[:, : td.task.K].clone()
    keys[keys < 0] = 10 ** 6
    order = keys.sort(1).values[:, :k]
    hit = (order == batch.qkey[:, None]).any(1).float() if k else torch.zeros(len(batch))
    return float((hit + (1 - hit) / td.task.value_range).mean())


def text_wire_bytes(text_bytes: int, trace: bool = False, keyed: bool = True) -> int:
    """The text (or codec) link sent as one keyed TLP packet: header, payload and AEAD tag."""
    return HDR.size + (EXT.size if trace else 0) + text_bytes + (TAG_LEN if keyed else 0)


def save(name: str, obj: dict):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, name + ".json")
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=_np)
    print(f"wrote {os.path.relpath(path)}")
    return path


def _np(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if torch.is_tensor(o):
        return o.tolist()
    return str(o)


def figure(w=6.4, h=4.0):
    fig, ax = plt.subplots(figsize=(w, h), dpi=130)
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    return fig, ax


def savefig(fig, name: str):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, name + ".png")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {os.path.relpath(path)}")


class Timer:
    def __enter__(self):
        self.t = time.time()
        return self

    def __exit__(self, *a):
        self.s = time.time() - self.t


# ------------------------------------------------------------ network harness
def session_pair(sender_caps, receiver_caps, bus=None, opts=None, audit=False, auditor=None, tracer_s=None,
                 tracer_r=None, server_ep=7, client_ep=107, negotiate=True):
    """Connect a receiver (initiator) to a sender (responder) over the in-process transport.
    Returns (server_sock, client_sock, server_conn, client_conn)."""
    from ..sock.socket import TLPSocket
    from ..tlp.transport import InprocBus
    bus = bus or InprocBus()
    s = TLPSocket(TRANSPORT, bus=bus, audit=audit, auditor=auditor, opts=opts, name="A", tracer=tracer_s)
    s.bind(("127.0.0.1", server_ep))
    s.listen()
    s.set_capabilities(sender_caps)
    c = TLPSocket(TRANSPORT, bus=bus, audit=audit, auditor=auditor, opts=opts, name="B", tracer=tracer_r)
    c.bind(("127.0.0.1", client_ep))
    out = {}

    def acc():
        try:
            out["conn"], _ = s.accept(timeout=60)
        except Exception as e:
            out["err"] = e

    th = threading.Thread(target=acc)
    th.start()
    try:
        cc = c.connect(("127.0.0.1", server_ep), receiver_caps, negotiate=negotiate)
    except Exception as e:
        th.join(10)
        s.close()
        c.close()
        raise e
    th.join(10)
    return s, c, out.get("conn"), cc
