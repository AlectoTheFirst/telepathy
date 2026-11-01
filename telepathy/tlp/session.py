"""Session state: the negotiated dialect, not just sequence numbers."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .header import PROBE_EP


@dataclass
class Capabilities:
    N: int
    d: int
    bits: int = 16
    model_id: str = ""
    dialect_id: str = ""
    version: int = 0
    codec: Any = None  # local only: anchor codec (send + receive port of this model)
    purpose: int = 0   # task-level instruction for the sender (an intent code); never the query

    def wire(self) -> dict:
        w = {"v": self.version, "N": self.N, "d": self.d, "bits": self.bits,
             "model": self.model_id, "dialect": self.dialect_id}
        if self.purpose:
            w["purpose"] = self.purpose
        return w


@dataclass
class Auditor:
    """Well-known intent probe: address and static X25519 public key for escrow."""
    ip: str
    pub: bytes
    ep: int = PROBE_EP


@dataclass
class TLPOptions:
    rto: float = 0.08            # sender retransmit timeout (s)
    max_retx: int = 3            # R: retransmissions of a missing protected slot
    ack_delay: float = 0.04      # receiver ACK timer if LAST_SLOT is lost (s)
    grace: float = 0.08          # wait after an ACK for retransmits of ambiguous slots (s)
    anchor_threshold: float = 0.9
    n_heldout: int = 32
    n_fit: int = 256
    negotiate: bool = True
    translator_seed: int = 0         # seed of a stochastic translator fit (the MLP families)
    translator: str = "shared_mlp"   # the family that passes the held-out check (RESULTS.md 10.3)
    handshake_timeout: float = 30.0
    escrow_timeout: float = 3.0
    reliability: str = "tlp"     # tlp | udp | tcp  (udp/tcp: simulated baselines, experiment 6.7)
    tcp_max_retx: int = 50
    routed_factor: float = 0.9   # trust factor for a relayed session with an attested record
    unattested_factor: float = 0.5
    verdict_factor: float = 0.1  # trust multiplier after a probe VERDICT mismatch


@dataclass
class SessionStats:
    packets_sent: int = 0
    packets_recv: int = 0
    bytes_sent: int = 0
    bytes_recv: int = 0
    retransmits: int = 0
    auth_failures: int = 0      # keyed packets that failed AEAD, and unkeyed DATA on a keyed session
    handshake_bytes: int = 0
    handshake_seconds: float = 0.0
    negotiate_bytes: int = 0
    negotiate_seconds: float = 0.0


@dataclass
class Session:
    session_id: int
    local: tuple[str, int]
    peer: tuple[str, int]
    role: str                          # "initiator" | "responder"
    N: int = 0
    d: int = 0
    bits: int = 16
    local_dialect: str = ""
    peer_dialect: str = ""
    peer_model: str = ""
    anchor_score: float = 0.0
    peer_anchor_score: float = 0.0
    negotiated: str | None = None      # translator family when dialects differ
    trust: float = 0.0                 # effective trust assigned by this receiver
    routed: bool = False
    attested: bool = False
    audited: bool = False
    flagged_msgs: set = field(default_factory=set)
    flagged: bool = False
    keys: Any = None
    aead: Any = None          # crypto.aead.PacketAEAD after KEY_CONFIRM
    translator: Any = None
    stats: SessionStats = field(default_factory=SessionStats)
