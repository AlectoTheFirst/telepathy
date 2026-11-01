# Telepathy Link Protocol (TLP), version 0

TLP carries latent messages between models that negotiated a session. The packet format is the same on every transport. The default transport is UDP: each TLP packet is one UDP datagram (section 13). TLP can also run directly on IP as IP protocol 253, which RFC 3692 reserves for experimentation. The raw IP mode needs root, and it is verified only when the raw tests run as root.

TLP knows the structure of its payload:
- A latent message is N slots of dimension d.
- The slots are ordered by importance.
- A partial message is still useful.
- The bytes have meaning only for the two models that negotiated the session.

This document is the reference for `telepathy/tlp/`, `telepathy/sock/`, `telepathy/resolver/`, `telepathy/crypto/`, `telepathy/negotiate/` and `telepathy/otel/`.

## 1. Addressing

The TLP header carries a source endpoint id and a destination endpoint id (u16). A bind address is `(ip, endpoint)`.

- **Over UDP** (default), each bound endpoint has its own UDP port (section 13). The kernel delivers a datagram to its destination port only.
- **Over raw IP**, every raw socket for protocol 253 on a host receives every protocol-253 packet, including its own sends. The kernel does not demultiplex.

On both transports, a TLP socket keeps a packet only if the destination endpoint is its own. It then dispatches the packet by session id. It drops all other packets.

| Well-known endpoint | Service |
| --- | --- |
| 1 | Resolver |
| 2 | Intent probe (auditor) |
| 3 | Tap (UDP only): the port mirror that copies every datagram to the probe |
| 40000 and up | Relay endpoints that the resolver allocates for ROUTE |

## 2. Header

The header is 28 bytes, in network byte order, with the fields in this order. Every u16 and u32 field is at an aligned offset.

| Offset | Size | Field | Meaning |
| --- | --- | --- | --- |
| 0 | u8 | version | 0 for Telepathy-0 |
| 1 | u8 | msg_type | See section 4 |
| 2 | u8 | flags | See the flags table |
| 3 | u8 | intent | The declared intent code (section 9). 0 = unspecified. |
| 4 | u16 | src_endpoint | Sender endpoint id |
| 6 | u16 | dst_endpoint | Receiver endpoint id |
| 8 | u32 | session_id | Chosen by the initiator in HELLO |
| 12 | u32 | seq | Per-session packet sequence number |
| 16 | u16 | msg_id | One latent message = N slots under one msg_id |
| 18 | u8 | slot_index | Position of this slot in the message |
| 19 | u8 | slot_count | Total slots in the message |
| 20 | u8 | slot_priority | 0 = highest. Slots go out in priority order. |
| 21 | u8 | reserved | Zero. It pads the header to 28 bytes. |
| 22 | u16 | slot_dim | Values per slot |
| 24 | u16 | payload_len | Payload bytes |
| 26 | u16 | checksum | See below |

The checksum is the RFC 1071 ones-complement sum over the header (with the checksum field set to zero), the extension header if present, and the payload. Raw IP has no payload checksum, so TLP must have its own.

There is no trust field. The receiver assigns trust (section 8). A trust claim from a sender is an attack surface, not a signal.

| Flag | Bit | Meaning |
| --- | --- | --- |
| PROTECTED | 0x01 | Slot is in the protected prefix. The sender retransmits it if it is lost. |
| LAST_SLOT | 0x02 | Last slot of the message. The receiver sends an ACK. |
| KEYED | 0x04 | Payload is AES-256-GCM ciphertext + 16-byte tag under the session key (section 12). |
| ROUTED | 0x08 | Session goes through a relay. |
| ACK_REQ | 0x10 | Retransmission. The receiver sends an ACK at once. |
| EXT_TRACE | 0x20 | The 28-byte trace extension header follows the base header. |

**Size limit.** Every TLP packet is at most 1400 bytes, so that it also fits an Ethernet MTU. A DATA packet carries one slot. At d = 32 and float16, a DATA packet is 28 + 64 = 92 bytes, or 120 bytes with EXT_TRACE.

### 2.1 Trace extension header (EXT_TRACE)

The extension is present when the EXT_TRACE flag is set. It follows W3C Trace Context sizes:

| Offset | Size | Field |
| --- | --- | --- |
| 28 | 16 bytes | trace_id |
| 44 | 8 bytes | parent_span_id |
| 52 | u8 | trace_flags |
| 53 | 3 bytes | padding (zero) |

Every message in a traced exchange carries the extension, including the resolver messages. The relay forwards it unchanged. With small slots the extension is larger than the payload. At d = 4 and float16, the payload is 8 bytes and the extension is 28 bytes. Thus the bandwidth and transport sweeps run with tracing off, and experiment 8.4 measures the overhead.

## 3. Slot payload

One slot per DATA or ANCHOR packet: slot_dim values, little-endian.

| b (bits per dimension) | Encoding | Bytes |
| --- | --- | --- |
| 16 | float16 values | 2 × d |
| 8, 4, 2 | float16 per-slot scale, then d signed b-bit integers packed LSB-first in two's complement. Value = q × scale. Scale = max\|x\| / (2^(b−1) − 1). | 2 + ceil(d × b / 8) |

The session negotiates b in HELLO. The session uses the smaller of the two proposals. ANCHOR payloads are always float16 and are never keyed. A keyed DATA payload is the AES-GCM ciphertext of this encoding plus a 16-byte tag (section 12).

Bit budgets in RESULTS.md count the payload bytes in this table × 8, including the per-slot scale. Headers and the AEAD tag are excluded, as for the text and codec baselines.

## 4. Message types and payloads

Control payloads are compact JSON (UTF-8, sorted keys). Latent and key payloads are binary.

| Code | Type | Payload |
| --- | --- | --- |
| 1 | HELLO | JSON capabilities: `v`, `N`, `d`, `bits`, `model`, `dialect` (fingerprint of the training run), `nonce`, `heldout` (number of verification anchors), `fit` (number of fitting anchors, 0 = no negotiation), `translator` (family), and optionally `purpose` (a task-level intent code for the sender, never the query) |
| 2 | HELLO_ACK | The same fields from the responder. `heldout`, `fit` and `translator` echo the values that the responder accepts. |
| 3 | ANCHOR | One anchor slot (float16, in the clear, always PROTECTED, intent 4). msg_id = anchor index. Fitting anchors use msg_id 1000 + j. |
| 4 | ANCHOR_RESULT | JSON `{score, n}`: the fraction of the peer's held-out anchors that this side read correctly |
| 5 | KEY | 32-byte X25519 ephemeral public key |
| 6 | KEY_CONFIRM | 32-byte HMAC-SHA256(confirm_key, role ‖ transcript). Role is `I` or `R`. |
| 7 | DATA | One slot of a latent message (section 3). KEYED. |
| 8 | ACK | u8 acknowledged msg_type, then a bitmap of received slot indices (LSB first, ceil(slot_count / 8) bytes). msg_id is in the header. |
| 9 | FIN | Optional JSON reason |
| 10 | FIN_ACK | Empty |
| 11 | RESET | JSON `{reason}` |
| 12 | REGISTER | JSON `{name, ip, endpoint, model, dialect, tags, emb, ttl}`. `emb` is base64 float16 (64 values). |
| 13 | REGISTER_ACK | JSON `{ok, name}` |
| 14 | RESOLVE | JSON `{name}` or `{emb}` |
| 15 | RESOLVE_REPLY | JSON `{ok, name, ip, endpoint, model, dialect, score}`. For ROUTE, it also has `routed: true`, and ip and endpoint are the relay's. |
| 16 | ROUTE | JSON `{emb}`: a query embedding in place of a destination |
| 17 | KEY_ESCROW | JSON `{sid, self, peer, N, d, bits, local_dialect, peer_dialect, routed, translator, wrapped}`. `wrapped` = base64(eph_pub ‖ nonce ‖ AES-GCM(seed)), with associated data = sid. |
| 18 | ESCROW_ACK | JSON `{sid, ok}`. The probe sends it to both peers. |
| 19 | VERDICT | JSON `{sid, msg_id, declared, observed, confidence, match, slots, read_in}` |
| 20 | NEGOTIATE | JSON `{family, score, fit}`: the translator family and this side's held-out score |

## 5. Session state

A session is a negotiated dialect, not only a set of sequence numbers. The state of a session contains:
- The peer address, the negotiated N, d and b, and both dialect ids.
- The anchor or negotiation score, and the effective trust.
- The key material: the AES-256-GCM data key, the 4-byte nonce salt and the confirm key.
- The translator, if one was negotiated.
- The path (direct or routed).
- The audit status.
- The per-message state: msg_id, the received bitmap, the protected-range bounds, the ACK timers and the retransmit timers.

### 5.1 State machine

```
CLOSED --HELLO sent--> HELLO_SENT --HELLO_ACK--> VERIFYING
CLOSED --HELLO rcvd--> HELLO_RCVD --HELLO_ACK sent--> VERIFYING
VERIFYING --same dialect, anchors pass--> KEYING
VERIFYING --different dialect, negotiation on--> NEGOTIATING --held-out scores pass--> VERIFYING --> KEYING
KEYING --KEY_CONFIRM ok--> ESCROW --ESCROW_ACK--> ESTABLISHED
KEYING --KEY_CONFIRM ok, audit off (unit tests only)--> ESTABLISHED
ESTABLISHED --FIN--> CLOSING --FIN_ACK--> CLOSED
any state --RESET / anchor fail / negotiation fail / key confirm fail / no ESCROW_ACK--> CLOSED
```

The session does not accept DATA before ESTABLISHED. One exception applies: DATA that arrives in KEYING or ESCROW is held, because the peer reached ESTABLISHED first. The session delivers held DATA to the model only when it reaches ESTABLISHED, and discards it if the session closes. DATA in any earlier state is dropped. No auditor, no session: with audit on, neither side reaches ESTABLISHED without ESCROW_ACK.

## 6. Handshake

The peers negotiate the channel in this order: shared mathematics in the open first, then the private channel.

1. **HELLO / HELLO_ACK.** The peers exchange capabilities. N = min, d = min, b = min. If d differs and the dialects differ, the session sends RESET, because a translator that changes dimension is out of scope.
2. **ANCHOR / ANCHOR_RESULT (same dialect id).** Both sides hold the same public anchor set: tables with a known query and answer (`telepathy/data.py`, held-out seed 1002, 32 tables).
   - Each side encodes the held-out anchors with its own send port and sends them in the clear, all PROTECTED.
   - Each side reads the peer's anchors through its own receive port (trust 1) and scores the answers.
   - Each side sends ANCHOR_RESULT. Data flows only if both scores reach the threshold (default 0.9, that is 29 of 32). The score is an input to the effective trust.
3. **KEY / KEY_CONFIRM.** The peers do an X25519 ephemeral key exchange.
   - transcript = SHA-256(HELLO payload ‖ HELLO_ACK payload ‖ initiator pub ‖ responder pub), with length prefixes.
   - HKDF-SHA256(shared secret, salt = transcript, info = "telepathy-0 key schedule") gives 68 bytes: the data key (32), the confirm key (32) and the nonce salt (4).
   - KEY_CONFIRM carries HMAC(confirm_key, role ‖ transcript).
   - The transcript does not contain endpoints or session ids, so a relay can rewrite the header without effect on the confirmation.
4. **ESCROW.** The receiver of latents (the initiator in Telepathy-0) wraps the data key and the nonce salt to the probe's static X25519 public key and sends KEY_ESCROW to the probe. The probe sends ESCROW_ACK to both peers. The receiver escrows, not the sender, so a sender that lies cannot opt out.

### 6.1 Negotiation between strangers (different dialect ids)

1. Both sides encode the fitting anchors (up to 256, seed 1001) and the held-out anchors (32) with their own send port. They exchange the latents in ANCHOR messages. All are PROTECTED, because a lost fitting anchor silently degrades the translator. A send window keeps at most 32 anchor messages unacknowledged at one time. Without the window, a burst of 288 messages on raw sockets put the receivers behind the RTO, and the spurious retransmits made the backlog grow.
2. Each side fits a translator from the peer's dialect into its own: from the peer's fitting-anchor latents to its own latents for the same tables, slot i to slot i. The families are in `telepathy/negotiate/translators.py`:
   - Per slot: linear least squares with bias (ridge), orthogonal Procrustes, and a small MLP (64 hidden units).
   - Shared across slots: the same three families, fitted on the pooled pairs of all N slots. The shared MLP has 256 hidden units and 1500 full-batch steps.
   - The default family is `shared_mlp`. It is the only family that passes the held-out check (RESULTS.md 10.3). It assumes that one map serves every slot, so it cannot fit two dialects that differ by a separate map per slot.
3. Each side translates the peer's held-out latents, reads them through its own receive port, and scores them.
4. Each side sends NEGOTIATE with its family and held-out score. Data flows only if both scores reach the threshold. Otherwise either side sends RESET. The held-out score is an input to the effective trust.

For each slot, the receiver does these steps in this order: decrypt and authenticate the packet (AEAD), dequantize, apply the translator, then pass the slot to the receive port. A packet that fails authentication is dropped and counted.

A slot-to-slot translator is a modelling assumption. If every family fails, the full-message linear map (N × d to N × d) is a diagnostic. If it works where the per-slot maps fail, the slot roles differ between dialects. In Telepathy-0, the slot roles are the same in every trained model (slot i carries table entry i), and no linear map exists between two dialects, per slot or for the full message. The shared MLP works at d = 32 (RESULTS.md 10.3).

## 7. Loss handling and progressive delivery

- **Send order.** Slots go out in priority order (slot index = priority). The first P slots are PROTECTED. P is configurable, and the default is N/2. The last slot carries LAST_SLOT.
- **ACK.** The receiver sends an ACK with a received-slot bitmap when it sees LAST_SLOT, when a packet has ACK_REQ, or when its ACK timer expires (default 40 ms after the first slot). The timer covers a lost LAST_SLOT: the receiver knows slot_count from any packet.
- **Retransmission.** On an ACK, or on its RTO timer (default 80 ms, backoff × 1.5), the sender retransmits the missing PROTECTED slots with ACK_REQ, up to R times (default 3). It never retransmits best-effort slots.
- **Finalisation.** The receiver finalises a message when one of these conditions is true:
  - All slots are present.
  - Its deadline expires. The deadline is counted from the first arrival of the message.
  - All protected slots are present after an ACK. If the protected range is ambiguous, the receiver waits a grace period (default 80 ms) for retransmissions.

  The protected range comes from the flags: every slot below the highest PROTECTED index seen is protected, and the lowest best-effort index seen is an upper bound. After finalisation, the receiver continues with the slots it has.
- **Progressive delivery.** `recv_latent(deadline)` yields `(slots, mask, final)` after every arrival, so the model can answer early and refine.

For experiment 6.7, the same packets also run under two simulated disciplines:
- UDP: P = 0 and no retransmission.
- TCP: every slot PROTECTED, up to 50 retransmissions, and the receiver sees only the contiguous prefix (head-of-line blocking).

## 8. Trust and provenance

The receiver assigns trust from local policy. Trust is never on the wire. The receiver writes the effective trust in the trace (`tlp.trust`).

```
trust = anchor_or_heldout_score × path × verdict
path    = 1.0 direct; 0.9 routed with the resolver's record as attestation (the HELLO_ACK model id
          matches the record); 0.5 routed without attestation
verdict = 1.0; 0.1 after a probe VERDICT with match = false (for the flagged message, if the verdict
          arrives before finalisation, and for every later message in the session)
```

The model's receive port has a gate that takes this trust as an input. The receive port reads the slots through its own cross-attention pathway. The slots never enter self-attention as tokens.

## 9. Intent codes

| Code | Name | Payload produced by the sender's purpose input |
| --- | --- | --- |
| 0 | UNSPECIFIED | Control messages |
| 1 | SHARE_VALUE | The full table (for "k7=?") |
| 2 | SHARE_KEY_EXISTENCE | The set of present keys (for "k7?") |
| 3 | SHARE_ENTRY_COUNT | The number of entries (for "count?") |
| 4 | ANCHOR_VERIFY | Anchor latents |

Only DATA and ANCHOR carry a non-zero intent. B does not read the declared intent. B's task follows its query form.

## 10. Resolver records and routing

| Record | Contents | Lookup |
| --- | --- | --- |
| NAME (like an A record) | name → ip, endpoint, model id, dialect id, TTL | RESOLVE `{name}` |
| K (new) | knowledge embedding + tags → name, TTL | RESOLVE `{emb}`: the best match by cosine similarity, with the score |

**Embedding space.** All parties use one shared, fixed, hashed bag-of-keys embedder (`telepathy/resolver/records.py`). Each key maps to a pseudo-random ±1 vector of 64 values. A holder's K record is the normalised sum over its key set. A query embedding is the vector of the queried key. The real system replaces this embedder with a learned head. Telepathy-0 needs only correct routing. A1 holds k0 to k7 and A2 holds k8 to k15. A query for a key that a holder has scores about 0.35, and the other holder scores about 0.

**Knowledge-addressed routing (ROUTE).**
1. B sends ROUTE with a query embedding.
2. The resolver selects the best K record and allocates a relay endpoint E.
3. The resolver replies with RESOLVE_REPLY. The reply contains the holder's name and attested model id, and the address `(resolver ip, E)`. It does not contain the holder's address.
4. The relay rewrites endpoints and session ids in both directions:
   - Initiator leg: the session id that B chose.
   - Holder leg: a session id that the relay chose.
5. The relay sets ROUTED, recomputes the checksum, and forwards EXT_TRACE unchanged.

HELLO, ANCHOR and KEY run end to end, so the relay never has the key and cannot read keyed latents. The AEAD leaves the fields that the relay rewrites out of its associated data (section 12), so routed DATA still authenticates. The relay drops DATA toward the receiver until it has seen the probe's ESCROW_ACK for the session. Thus two peers that collude would have to fake the probe toward the network, not only toward each other.

## 11. OpenTelepathy: intent tracing

- **Spans.** Every process (resolver, holders, B, probe) uses the OpenTelemetry SDK. Spans go to a JSONL file through `JsonlSpanExporter`, and to OTLP when `OTEL_EXPORTER_OTLP_ENDPOINT` is set.
- **One trace per task.** Each trace starts at B's `task` span and has these children:
  - `resolve` or `route`. The resolver adds its own `resolve`, `route` and `relay` spans.
  - `handshake`, with the children `hello`, then `anchor` or `negotiate`, then `key` and `escrow`. The responder has its own `handshake` span.
  - `latent.send` (holder). The probe adds `escrow` and `intent.check` spans.
  - `latent.recv` and `answer` (B).
- **Context propagation.**
  - The initiator's packets carry the task span's context.
  - The responder joins the trace from the HELLO extension.
  - The holder's DATA carries its `latent.send` context. The probe's `intent.check` is a child of that context.
- **Probe.**
  - The probe is a classifier: a small MLP over the masked slot prefix, trained on A's unkeyed messages with intent labels. It runs as its own service. It taps the wire. Over raw IP on loopback, a raw socket for protocol 253 already sees every packet. Over UDP, every transport copies each datagram to the tap port (endpoint 3), as a network port mirror would, and the probe listens there.
  - It unwraps the escrowed key, decrypts and authenticates each DATA packet, classifies each message, and emits `intent.check` with the declared intent, the observed intent and a match flag.
  - On a mismatch, it sets the span status to error, adds an `intent.mismatch` event, and sends VERDICT to the receiver.
  - At confidence ≥ 0.9, and by policy, it sends RESET to both peers. For a routed session, it sends the RESET through the relay.
  - It flags any session that reaches DATA without KEY_ESCROW as unaudited, with the same alert and RESET.
  - For a negotiated session in another dialect, it fits its own `shared_mlp` translator from the fitting anchors that it saw in the clear, and it reports the dialect in which it read each message (`tlp.dialect.read_in`). It sends ESCROW_ACK before the fit, because the fit (about 3 s) is longer than the escrow timeout. The probe has one receive thread, so the session's DATA waits in its receive queue until the fit is done. Thus the VERDICT for the first message of a negotiated session can come after B has finalised it.

## 12. Security properties (stated honestly)

- **Per-packet AEAD.** Keyed DATA is AES-256-GCM, one packet at a time, so a lost slot does not affect the others (`telepathy/crypto/aead.py`).
  - Key: the 32-byte data key from the HKDF key schedule (section 6).
  - Nonce (12 bytes): nonce salt (4) ‖ direction (1 byte: `I` if the initiator sends, `R` if the responder sends) ‖ 3 zero bytes ‖ seq (4). Each send, retransmits included, takes a new seq, so a nonce is never used twice under one key. Both sides count seq from 1, and the direction byte keeps their nonces apart.
  - Associated data: the 28-byte TLP header, with the fields that a ROUTE relay rewrites set to zero: the source and destination endpoints, the session id, the checksum, and the ROUTED and EXT_TRACE flags. All other header fields are authenticated: type, intent, seq, msg_id, slot index, count and priority, slot dim, payload length and the remaining flags.
  - Deviation from the brief, which asks for nonce = session id ‖ seq and AAD = the TLP header. The relay gives each leg its own session id and endpoints, so the two ends would compute different nonces and AAD, and no routed packet would authenticate. The session-stable nonce salt takes the place of the session id.
  - ANCHOR messages stay in the clear before keying. Control messages (ACK, FIN, RESET, VERDICT and the handshake) are not encrypted.
  - Experiment 6.6: with the key, a tap probe recovers 100% of the table entries. Without the key, it recovers 0.06%, which is chance (1/1600).
- **Active attacks.** X25519 here is unauthenticated, so an active attacker can sit in the middle of the key exchange and hold both keys. Telepathy-0 shows a negotiated, encrypted channel for a latent dialect. It is not production cryptography.
- **The probe is a trusted third party**, like a TLS-inspecting middlebox. If an attacker compromises the probe, every escrowed session is exposed.
- **The observed intent is a second declared intent.** The send port adds the purpose embedding to every slot query, so the declared purpose is written into every slot and is easy to read from them. The probe therefore reads what the sender's own port declared, not a hidden intent. Its detection rate (RESULTS.md section 8) is evidence against a sender whose label and purpose input disagree, not against a sender that hides its intent.
- **Probe evasion.** A sender trained against the probe can evade it. OpenTelepathy catches mislabelled and naive misuse, not an adaptive adversary. The adaptive-adversary experiment (a sender trained to fool the probe while it still serves B) is deferred to Telepathy-1.
- **The trust loop protects integrity only.** It reduces the influence of a lying peer on B. It does nothing for confidentiality: in an exfiltration case the data has already reached B, and B can be the beneficiary. Thus the response to a high-confidence mismatch is an alert and, by policy, a RESET.
- **RESET is unauthenticated.** A probe-issued RESET is also a denial-of-service surface for anyone who can see the session.
- **The tap is a demo convenience.** Over raw IP on loopback, every raw socket sees every packet. Over UDP, the transports copy every datagram to the tap port, which a sender's own stack could turn off. In a real network, the tap must be a network function (a port mirror on the path), not a function of the endpoints.

## 13. Encapsulation: TLP over UDP

TLP over UDP is the default transport (`telepathy/tlp/transport.py`, `UdpTransport`).

- **Reason.** NAT devices and many middleboxes drop IP protocols that they do not know, so a new IP protocol does not cross most networks. QUIC is the precedent: a transport protocol of its own, carried in UDP datagrams. Raw IP protocol 253 stays as an option (root only).
- **Packet.** One TLP packet (28-byte header, optional 28-byte trace extension, payload) is one UDP datagram, without change. The TLP checksum stays, although UDP has its own.
- **Ports.** The destination port comes from the `dst_endpoint` field in the TLP header, so the socket API does not change. Endpoints below 1024 use port `TLP_UDP_BASE + endpoint` (default base 47000), which keeps the well-known endpoints off the privileged ports. Other endpoints use the same number as their port, because base + endpoint would overflow 16 bits for relay endpoints.
- **Unchanged on UDP:** the flags, the session state machine, the handshake, ACK bitmaps, retransmission of PROTECTED slots only, the finalisation rules and progressive delivery. They are all in the TLP header and the TLP state, not in the transport. UDP adds no reliability, so TLP's loss handling sees the same losses as on raw IP.
- **Size.** Every TLP packet is at most 1400 bytes, so a datagram stays below a 1500-byte path MTU with IP and UDP headers.
- **Tap.** For the probe, every UDP transport also sends a copy of each datagram to the tap port (endpoint 3). A socket gives a tap copy only to its tap handler, and it ignores a tap copy of a packet that is addressed to itself, because that packet also arrives directly.

**Consequence: the UDP tap is cooperative.** In Telepathy-0, the endpoints themselves send the copies to the tap port. Two peers that collude can turn off their own mirroring and talk directly, without escrow. The probe then sees nothing, and its unaudited-session alert (section 11) never fires. The alert catches a direct session without escrow only when the mirror sits in the network, on the path between the peers (a switch port mirror or a middlebox), where the endpoints cannot turn it off. Routed sessions are different: the relay drops DATA until it sees ESCROW_ACK (section 10), so collusion through the resolver still needs a forged probe. Over raw IP on loopback, the kernel gives the probe every packet, so the tap there does not depend on the endpoints.
