# Telepathy-0

Telepathy-0 is a research prototype for model-to-model communication in latent slots, and for the network layer under it. Two models exchange latent slots in place of text over TLP, a transport of its own (TLP over UDP by default, or raw IP protocol 253). Before data flows, they verify a shared latent dialect on public anchors, or negotiate a translator, and key the channel with per-packet AES-GCM. An intent probe audits each message through OpenTelemetry.

The agents are not language models. Each agent is a 2-layer transformer with d_model = 64, trained jointly with its partner from random weights on structured tokens (keys, values and query tokens of a key-value table task), not on language. The claims are about the protocol, not about the models.

- **SPEC.md**: the protocol, the handshake, routing, tracing and the security properties.
- **RESULTS.md**: every number and plot, including the negative results.

## Setup (macOS or Linux, CPU only)

```
python3 -m venv .venv                 # Python 3.11+; tested with 3.15 and torch 2.15
.venv/bin/pip install -e '.[dev]'
```

Use `.venv/bin/python` and `.venv/bin/pytest` for every command.

## Commands

| Command | What it does | Time on an M1 Pro |
| --- | --- | --- |
| `.venv/bin/pytest -q` | All protocol tests over TLP over UDP. `TLP_TRANSPORT=inproc` runs them on the in-process bus. Raw-socket tests skip without privileges. | ~2 s |
| `sudo .venv/bin/pytest tests/test_raw.py` | Raw IP protocol 253 on loopback | ~2 s |
| `.venv/bin/python scripts/train.py` | 9 training runs in parallel processes, written to `checkpoints/` | ~4 min |
| `.venv/bin/python scripts/train.py --zipf` | 5 more runs with Zipf value queries (RESULTS.md 6.3b and 6.7b) | ~10 min |
| `.venv/bin/python scripts/run_experiments.py` | 16 experiments, written to `results/*.json` and `results/*.png` | ~47 min |
| `.venv/bin/python scripts/demo.py` | Resolver, probe, A1, A2, A′ and B as separate processes over TLP over UDP, with an interleaved log and the trace tree | ~15 s |
| `sudo .venv/bin/python scripts/demo.py --transport raw` | The same over raw IP protocol 253 | ~15 s |
| `.venv/bin/python scripts/demo.py --transport inproc` | The same scenario in one process | ~10 s |
| `.venv/bin/python scripts/probe.py` | The intent probe as a standalone process (`--transport raw` needs root) | |

To export spans to Jaeger (or another OTLP collector) as well as to the JSONL file, set `OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318`.

## Transports

`telepathy/tlp/transport.py` holds the three transports. The packet format is the same on all of them.

- **TLP over UDP (default).** Each TLP packet is one UDP datagram, without change. The UDP port comes from the destination endpoint in the TLP header (SPEC.md section 13). The base port for endpoints below 1024 is `TLP_UDP_BASE` (default 47000). Each transport also copies every datagram to the tap port for the intent probe.
- **In-process bus.** For tests and single-process runs.
- **Raw IP protocol 253.** Needs root. The raw transport is not verified until `sudo .venv/bin/pytest tests/test_raw.py` passes on the machine. On this Mac, those tests passed once as root, before the AEAD and UDP changes. Run them again to verify the current code.

### Raw sockets on macOS and Linux

The raw transport has these platform differences:

- **Privileges.** On macOS, a raw socket needs root. On Linux, it needs root or `CAP_NET_RAW`. `raw_available()` tries to open a raw socket, and the raw tests skip if it fails.
- **Receive.** Both kernels deliver the IP header on receive. The transport strips it by IHL. The packet length comes from TLP's `payload_len`, because BSD-derived stacks can report `ip_len` in host byte order and without the header length.
- **Send.** The transport sends without `IP_HDRINCL` by default. With `TLP_HDRINCL=1`, it builds the IP header itself: `ip_len` and `ip_off` are in host byte order on macOS and BSD, and in network byte order on Linux.
- **Demultiplexing.** Every raw socket for protocol 253 receives every TLP packet, including its own sends. Demultiplexing uses the TLP header (destination endpoint, then session id).

## Repository layout

```
telepathy/
  config.py data.py agents.py      task, vocabulary, holders (A) and asker (B)
  models/      transformer, send/receive ports, quantization, training, evaluation
  tlp/         header codec, messages, state machine, session, transports and injector
  sock/        TLPSocket API (handshake, loss handling, progressive delivery)
  resolver/    NAME/K records, embedder, resolver service and relay
  crypto/      X25519/HKDF/HMAC, escrow wrapping, per-packet AES-GCM (aead.py)
  negotiate/   anchor codec and translator families (per slot, shared across slots, full message)
  otel/        tracer setup, JSONL exporter, span helpers, intent probe
  experiments/ one script for each experiment in sections 6, 8 and 10, plus n3_diagnostics (10.3) and n4_gate (10.4)
docs/          telepathy1_gate_test.md (a specification for later, not run)
scripts/       train.py demo.py probe.py run_experiments.py
tests/
```

These are differences from the brief's layout:
- The transports (UDP, raw IP, in-process) and the injector are in one file (`tlp/transport.py`).
- The session state is in `tlp/session.py`.

## Deviations from the brief

| Brief | Telepathy-0 | Reason |
| --- | --- | --- |
| Linux, `tc netem` | macOS and Linux. Experiment 6.7 uses the in-process loss injector on the UDP transport. "Datagrams, no TLP recovery" and "TCP-like" are delivery disciplines on the same TLP packets, not kernel stacks. | The user's machine is macOS. |
| TLP directly on IP (protocol 253) | TLP over UDP is the default. Raw IP protocol 253 is an option. | NAT and middleboxes drop unknown IP protocols (SPEC.md section 13). |
| Keyed per-slot transform | Per-packet AES-256-GCM. The nonce uses a session-stable salt in place of the session id, and the AAD zeroes the header fields that a relay rewrites. | A rotation is not a cipher. A relay gives each leg its own session id, so the literal nonce and AAD would fail on routed sessions (SPEC.md section 12). |
| "A' at default and a small d" | The small-d negotiation case uses d = 16. | Runs with d ≤ 8 do not train symmetrically within the time limit (RESULTS.md). |
| Text baseline tokens | B reads the serialised table (`k3=19 k7=42 …`, truncated by bytes) with the same tokenizer that A uses on its table: one token for each complete field. | A character-level reader did not learn to 100% within the budget. Bytes are counted on the real text. |
| Send port reads "final hidden states" | The send port reads the input-level states and the final states. The gradient from the send port into the shared trunk is scaled by 0.1. | Without this change, joint training did not converge (RESULTS.md, training notes). |
| Per-slot translator families (linear, Procrustes, MLP) | The same three families shared across slots were added. The protocol default is `shared_mlp`. | Every per-slot family failed the held-out check. The shared MLP passes at d = 32 (RESULTS.md 10.3). |
| Up to 256 fitting anchors | The protocol uses at most 256. Experiment n3 also fits with 1024 and 4096 anchors, as a diagnostic only. | To separate "too few anchors" from "no map of this family exists". |

## Related work and what Telepathy-0 does differently

- **Emergent communication.** Lazaridou, Peysakhovich and Baroni (ICLR 2017, arXiv:1612.07182) and Foerster et al. (2016, arXiv:1605.06676) train agents to communicate through a learned channel. Telepathy-0 trains its pairs the same way, from random weights. Its contribution is the network layer for such a channel: transport, routing, sessions and audit.
- **Ordered representations.** Nested dropout (Rippel, Gelbart and Adams, 2014, arXiv:1402.0915) and Matryoshka representations (Kusupati et al., 2022, arXiv:2205.13147) give representations whose prefixes are useful. Telepathy-0 uses nested dropout over slots so that a transport can protect a prefix and drop the rest.
- **Interlat** (Du et al., arXiv:2511.09149) sends an LLM's last hidden states between agents as their "thought", with a learned compression. The abstract does not describe a transport. Telepathy-0 puts such messages on a network: a packet per slot, loss handling by priority, and a session that verifies the dialect.
- **LatentMAS** (Zou et al., arXiv:2511.20639) is training-free collaboration in latent space through a shared latent working memory. Telepathy-0 has no shared memory: the two models are separate processes, with a key exchange and an audit between them.
- **K-V cache alignment** (Dery et al., arXiv:2601.06123) learns adapters into a shared representation space for the k-v caches of several models, with the pretrained weights frozen. Telepathy-0 negotiates a translator at connection time from public anchors, and the session fails closed when the held-out check fails.
- **Latent channels against text** (Wenzel, arXiv:2607.14103) compares sparse-latent, dense-latent and text channels between language model agents, and finds that the latent channel matches but never exceeds text on its concept tasks. It also aligns Llama and Mistral with Procrustes. Telepathy-0 measures its link against text and a binary codec under a byte budget, and studies what such a channel needs from the network.
