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
    batch: int = 4     # M: most slots per DATA packet this side accepts (Telepathy-1, SPEC.md 14.3)
    hub: bool = False  # HUB capability: this side can send and read hub coordinates (SPEC.md 14.2)
    anchor_set_id: str = ""   # the public anchor set that defines the hub coordinates
    hub_dim: int = 0   # values per slot in hub coordinates

    def wire(self) -> dict:
        w = {"v": self.version, "N": self.N, "d": self.d, "bits": self.bits,
             "model": self.model_id, "dialect": self.dialect_id}
        if self.purpose:
            w["purpose"] = self.purpose
        if self.batch != 1:          # absent = 1, so a Telepathy-0 HELLO is unchanged
            w["batch"] = self.batch
        if self.hub:
            w["hub"] = {"anchors": self.anchor_set_id, "dim": self.hub_dim}
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
    t_records: Any = None        # T-record client: lookup(from, to, family) and publish(...) (SPEC.md 14.1)
    t_ttl: float = 3600.0        # TTL of a published T record (s)


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
    keyed_packets: int = 0      # packets sent with an AEAD tag
    data_packets: int = 0       # DATA packets sent, retransmits included


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
    batch: int = 1                     # M: slots per DATA packet (min of both HELLOs)
    hub: str | None = None             # anchor-set id when the session runs in hub mode
    wire_d: int = 0                    # values per slot on the wire (d, or hub_dim in hub mode)
    t_record: str | None = None        # "hit", "miss" or None (no lookup)
    stats: SessionStats = field(default_factory=SessionStats)
