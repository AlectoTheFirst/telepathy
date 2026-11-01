"""OpenTelepathy intent probe: an independent classifier of the intent behind latent messages,
and the auditor service that taps the wire, holds escrowed session keys, and closes the loop
with VERDICT (and RESET) messages.

The probe classifies intent; it never reconstructs content. It is a trusted third party (like a
TLS-inspecting middlebox): compromising it exposes every escrowed session. A sender trained
against it can evade it; it catches mislabelled and naive misuse, not an adaptive adversary.
"""
from __future__ import annotations

import os
import threading
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import BITS_CHOICES, CKPT_DIR, DATA_INTENTS, seed_everything
from ..crypto.kx import load_or_create_static, pub_bytes, unwrap_with
from ..crypto.aead import AuthError, PacketAEAD, sender_direction
from ..crypto.kx import SessionKeys
from ..models.agent import load_ckpt
from ..models.quant import decode_slot, quantize
from ..tlp.header import PROBE_EP, Flag, MsgType, Packet
from ..tlp.messages import jdump, jload, unb64
from ..tlp.session import Auditor
from ..sock.socket import FIT_BASE, TLPSocket
from .spans import TLPTracer, context_from_ext, ext_from_span, intent_name

M = MsgType
KEY_PATH = os.path.join(CKPT_DIR, "probe_x25519.key")


# ------------------------------------------------------------------ classifier
class IntentMLP(nn.Module):
    def __init__(self, N: int, d: int, hidden: int = 128, n_cls: int = 3):
        super().__init__()
        self.N, self.d = N, d
        self.net = nn.Sequential(nn.Linear(N * d + N, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, n_cls))

    def forward(self, slots, mask):
        x = torch.cat([(slots * mask[..., None]).flatten(1), mask.float()], 1)
        return self.net(x)

    @torch.no_grad()
    def classify(self, slots, mask):
        """slots [B,N,d], mask [B,N] -> (observed intent codes [B], confidence [B], probs [B,3])"""
        p = F.softmax(self(slots, mask), -1)
        conf, idx = p.max(-1)
        return idx + 1, conf, p


def probe_path(dialect: str) -> str:
    return os.path.join(CKPT_DIR, f"probe_{dialect}.pt")


def train_probe(ckpt: str = "latent_d32_s0", side: str = "A", steps: int = 1500, seed: int = 0, log=print) -> IntentMLP:
    """Train on the sender's unkeyed messages with intent labels, over random slot prefixes and b."""
    seed_everything(seed)
    models, meta, td = load_ckpt(ckpt)
    sender = models[side]
    N, d = meta["model"]["N"], meta["model"]["d"]
    clf = IntentMLP(N, d)
    opt = torch.optim.Adam(clf.parameters(), lr=2e-3)
    g = torch.Generator().manual_seed(seed + 5)
    bits_t = torch.tensor(BITS_CHOICES)
    t0 = time.time()
    for step in range(steps):
        b = td.make_batch(g, 256, intents=DATA_INTENTS)
        with torch.no_grad():
            slots = sender.encode(b.a_tokens, b.intent)
            slots = quantize(slots, bits_t[torch.randint(0, 4, (256,), generator=g)][:, None])
        p = torch.randint(1, N + 1, (256,), generator=g)
        mask = torch.arange(N)[None] < p[:, None]
        loss = F.cross_entropy(clf(slots, mask), b.intent - 1)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 500 == 0 or step == steps - 1:
            log(f"[probe {meta['dialect_id']}] step {step} loss {loss.item():.3f} {time.time() - t0:.0f}s")
    clf.eval()
    torch.save({"state": clf.state_dict(), "N": N, "d": d, "dialect": meta["dialect_id"], "ckpt": ckpt}, probe_path(meta["dialect_id"]))
    return clf


def load_probe(ckpt: str = "latent_d32_s0", side: str = "A", train_if_missing: bool = True) -> tuple[IntentMLP, dict]:
    _, meta, _ = load_ckpt(ckpt)
    path = probe_path(meta["dialect_id"])
    if not os.path.exists(path):
        if not train_if_missing:
            raise FileNotFoundError(path)
        train_probe(ckpt, side)
    blob = torch.load(path, map_location="cpu", weights_only=False)
    clf = IntentMLP(blob["N"], blob["d"])
    clf.load_state_dict(blob["state"])
    clf.eval()
    return clf, blob


def auditor_for(ip: str = "127.0.0.1", key_path: str = KEY_PATH) -> Auditor:
    return Auditor(ip, pub_bytes(load_or_create_static(key_path)))


# --------------------------------------------------------------------- service
class ProbeService:
    def __init__(self, ip="127.0.0.1", transport="udp", bus=None, ckpt="latent_d32_s0", tracer=None, log=None,
                 reset_policy: bool = True, reset_threshold: float = 0.9, send_verdicts: bool = True,
                 finalize_delay: float = 0.03, key_path: str = KEY_PATH, use_keys: bool = True):
        self.priv = load_or_create_static(key_path)
        self.auditor = Auditor(ip, pub_bytes(self.priv))
        self.clf, meta = load_probe(ckpt)
        self.dialect = meta["dialect"]
        from ..negotiate.anchors import model_capabilities
        self.ref = model_capabilities(ckpt, "A").codec     # reference latents of the probe's dialect
        self.N, self.d = self.clf.N, self.clf.d
        self.tr: TLPTracer | None = tracer
        self.reset_policy, self.reset_threshold = reset_policy, reset_threshold
        self.send_verdicts = send_verdicts
        self.finalize_delay = finalize_delay
        self.use_keys = use_keys
        self.sock = TLPSocket(transport, bus=bus, audit=False, log=log, name="probe")
        self.sock.bind((ip, PROBE_EP))
        self.sock.enable_tap(self._tap)
        self.sock.packet_handler = lambda pkt: None   # addressed packets are handled in the tap
        self.lock = threading.RLock()
        self.sessions: dict[int, dict] = {}
        self.anchor_buf: dict = defaultdict(dict)      # (sid, src) -> {msg_id: {slot: vec}}
        self.msgs: dict = {}
        self.unaudited: set = set()
        self.relay_addrs: set = set()
        self.results: list[dict] = []
        self.alerts: list[dict] = []

    def close(self):
        self.sock.close()

    # ---------------------------------------------------------------- tap
    def _tap(self, pkt: Packet):
        t = pkt.msg_type
        if t == M.KEY_ESCROW and pkt.dst_endpoint == PROBE_EP:
            self._escrow(pkt)
        elif t == M.ANCHOR:
            if len(self.anchor_buf) < 4096:
                buf = self.anchor_buf[(pkt.session_id, (pkt.src_ip, pkt.src_endpoint))]
                buf.setdefault(pkt.msg_id, {})[pkt.slot_index] = decode_slot(pkt.payload, pkt.slot_dim, 16)
        elif t == M.DATA:
            self._data(pkt)

    def _escrow(self, pkt: Packet):
        b = jload(pkt.payload)
        sid = b["sid"]
        try:
            key, salt = SessionKeys.from_escrow(unwrap_with(self.priv, unb64(b["wrapped"]), sid.to_bytes(4, "big")))
        except Exception as e:
            self.sock.log("escrow", sid=sid, info=f"unwrap failed: {e!r}")
            return
        receiver, peer = tuple(b["self"]), tuple(b["peer"])
        sess = {"sid": sid, "receiver": receiver, "peer": peer, "N": b["N"], "d": b["d"], "bits": b["bits"],
                "aead": PacketAEAD(key, salt) if self.use_keys else None,
                # DATA comes from the receiver's peer; the receiver's role is in the escrow body
                "peer_dir": sender_direction("responder" if b.get("role", "initiator") == "initiator" else "initiator"),
                "peer_dialect": b["peer_dialect"], "local_dialect": b["local_dialect"], "translator": None,
                "read_in": b["peer_dialect"], "routed": b.get("routed", False)}
        if b.get("routed"):
            self.relay_addrs.add(peer)
        ctx_ext = pkt.ext
        ack = jdump({"sid": sid, "ok": True})
        for ip, ep in (receiver, peer):
            self.sock.send_packet(ip, Packet(M.ESCROW_ACK, PROBE_EP, ep, sid, 1, payload=ack, ext=ctx_ext))
        # Fit the translator after ESCROW_ACK: a shared MLP fit takes seconds, longer than the
        # escrow timeout. This handler runs on the probe's only receive thread, so the session's
        # DATA waits in the receive queue until the fit is done and is still read translated.
        if sess["peer_dialect"] != self.dialect:
            self._fit_translator(sess)
        with self.lock:
            self.sessions[sid] = sess
        if self.tr:
            with self.tr.span("escrow", parent_ext=ctx_ext, tlp__session_id=sid,
                              tlp__dialect__peer=sess["peer_dialect"], tlp__dialect__read_in=sess["read_in"]):
                pass
        self.sock.log("escrow", sid=sid, info=f"key escrowed by {receiver}; ESCROW_ACK -> receiver and {peer}; "
                                              f"reading in dialect {sess['read_in']}")

    def _fit_translator(self, sess):
        """Negotiated session in a dialect the probe was not trained on: fit a translator into
        the probe's dialect from the anchors it witnessed in the clear."""
        from ..negotiate.translators import fit_translator
        buf = self.anchor_buf.get((sess["sid"], sess["peer"]), {})
        fit_ids = sorted(m for m in buf if m >= FIT_BASE)
        if len(fit_ids) < 8:
            sess["read_in"] = f"{sess['peer_dialect']} (untranslated: no fit anchors seen)"
            return
        n, N, d = len(fit_ids), sess["N"], sess["d"]
        peer = np.zeros((n, N, d), np.float32)
        for j, mid in enumerate(fit_ids):
            for i, v in buf[mid].items():
                peer[j, i] = v
        idx = [m - FIT_BASE for m in fit_ids]
        own = self.ref.encode("fit", max(idx) + 1)[idx][:, :N, :d]
        sess["translator"] = fit_translator("shared_mlp", peer, own)
        sess["read_in"] = f"{self.dialect} (translated from {sess['peer_dialect']}, {n} anchors)"

    def _data(self, pkt: Packet):
        sid = pkt.session_id
        src, dst = (pkt.src_ip, pkt.src_endpoint), (pkt.dst_ip, pkt.dst_endpoint)
        with self.lock:
            sess = self.sessions.get(sid)
        if sess is not None:
            if dst != sess["receiver"]:
                return
            if pkt.slot_dim == 0:   # opaque byte message (send_bytes), not a latent: nothing to classify
                return
            key = (sid, pkt.msg_id)
            with self.lock:
                m = self.msgs.get(key)
                if m is None:
                    m = {"vecs": {}, "intent": pkt.intent, "ext": pkt.ext, "done": False, "count": pkt.slot_count,
                         "first": time.monotonic()}
                    self.msgs[key] = m
                    self.sock.timers.call_later(self.finalize_delay, lambda: self._classify(sess, key))
                if m["done"] or pkt.slot_index in m["vecs"]:
                    return
                plain = pkt.payload
                if pkt.flags & Flag.KEYED:
                    if sess["aead"] is None:   # use_keys=False: the probe reads ciphertext as if it were slots
                        plain = pkt.payload[: len(pkt.payload) - 16]
                    else:
                        try:
                            plain = sess["aead"].open(pkt, sess["peer_dir"])
                        except AuthError:
                            self.sock.log("drop", sid=sid, slot=pkt.slot_index, info="probe: AEAD check failed")
                            return
                x = decode_slot(plain, pkt.slot_dim, sess["bits"])
                if sess["translator"] is not None:
                    x = sess["translator"].apply_slot(pkt.slot_index, x)
                m["vecs"][pkt.slot_index] = x
                full = len(m["vecs"]) == m["count"]
            if full or pkt.flags & Flag.LAST_SLOT:
                self._classify(sess, key)
            return
        if src in self.relay_addrs or dst in self.relay_addrs:
            return   # holder leg of a relayed session whose receiver leg is escrowed
        if sid in self.unaudited:
            return
        self.unaudited.add(sid)
        alert = {"sid": sid, "src": src, "dst": dst, "kind": "unaudited"}
        self.alerts.append(alert)
        self.sock.log("alert", sid=sid, info=f"UNAUDITED session: DATA without KEY_ESCROW {src} -> {dst}; RESET")
        if self.tr:
            with self.tr.span("session.unaudited", parent_ext=pkt.ext, tlp__session_id=sid) as sp:
                sp.add_event("session.unaudited", {"src": str(src), "dst": str(dst)})
                TLPTracer.error(sp, "DATA without KEY_ESCROW")
        if self.reset_policy:
            for ip, ep in (src, dst):
                self.sock.send_packet(ip, Packet(M.RESET, PROBE_EP, ep, sid, 1,
                                                 payload=jdump({"reason": "unaudited session (probe)"})))

    def _classify(self, sess, key):
        with self.lock:
            m = self.msgs.get(key)
            if m is None or m["done"] or not m["vecs"]:
                return
            m["done"] = True
            vecs = dict(m["vecs"])
        N, d = self.N, self.d
        slots = torch.zeros(1, N, d)
        mask = torch.zeros(1, N, dtype=torch.bool)
        for i, v in vecs.items():
            if i < N:
                slots[0, i, : len(v)] = torch.from_numpy(np.asarray(v[:d], np.float32))
                mask[0, i] = True
        obs, conf, _ = self.clf.classify(slots, mask)
        observed, confidence = int(obs[0]), float(conf[0])
        declared = int(m["intent"])
        match = observed == declared
        sid, msg_id = key
        res = {"sid": sid, "msg_id": msg_id, "declared": declared, "observed": observed, "confidence": confidence,
               "match": match, "slots": len(vecs), "read_in": sess["read_in"]}
        self.results.append(res)
        self.sock.log("intent.check", sid=sid, msg_id=msg_id,
                      info=f"declared={intent_name(declared)} observed={intent_name(observed)} conf={confidence:.2f} "
                           f"{'match' if match else 'MISMATCH'} (read in {sess['read_in']})")
        if self.tr:
            with self.tr.span("intent.check", parent_ext=m["ext"], tlp__session_id=sid, tlp__msg_id=msg_id,
                              tlp__intent__declared=intent_name(declared), tlp__intent__observed=intent_name(observed),
                              tlp__intent__match=match, tlp__intent__confidence=confidence,
                              tlp__slots__received=len(vecs), tlp__dialect__read_in=sess["read_in"]) as sp:
                if not match:
                    sp.add_event("intent.mismatch", {"declared": declared, "observed": observed,
                                                     "confidence": confidence})
                    TLPTracer.error(sp, "declared intent does not match observed intent")
        rip, rep = sess["receiver"]
        if self.send_verdicts:
            self.sock.send_packet(rip, Packet(M.VERDICT, PROBE_EP, rep, sid, 1, msg_id=msg_id, payload=jdump(res),
                                              ext=m["ext"]))
        if not match and confidence >= self.reset_threshold and self.reset_policy:
            self.alerts.append({**res, "kind": "mismatch_reset"})
            self.sock.log("alert", sid=sid, msg_id=msg_id, info=f"high-confidence mismatch ({confidence:.2f}): RESET")
            for ip, ep in (sess["receiver"], sess["peer"]):
                self.sock.send_packet(ip, Packet(M.RESET, PROBE_EP, ep, sid, 1,
                                                 payload=jdump({"reason": "intent mismatch (probe)"})))
