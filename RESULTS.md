# Telepathy-0 results

All numbers come from `results/*.json`, and each plot is next to its JSON file. The machine was an Apple M1 Pro (CPU only), with Python 3.15 and torch 2.15. All seeds are fixed.

The eval set is 2000 value queries at K = 8, with the hint absent, unless a section says otherwise. Chance is 1%.

| Step | Wall time |
| --- | --- |
| `scripts/train.py` (9 runs in parallel, one thread each) | 247 s (latent runs about 238 s each) |
| `scripts/train.py --zipf` (5 runs; the two 12000-step runs are the longest) | about 9.5 min |
| `scripts/run_experiments.py` (17 experiments, TLP over UDP, AEAD) | about 70 min (sum of the measured parts below) |

The suite was not timed again as one run after the last changes. The measured parts: e7 509 s for the three 100-message sweeps, plus 641 s for the 1000-message 30% cell on Zipf queries and about 670 s for the same cell on uniform queries (from file times), e8 732 s, n3 1371 s (5 seeded fits for each MLP row), n1 122 s, n4 86 s, e6 38 s, and less than 15 s for each of the others. The protocol experiments (e5, e7, e8, t5, n2, n4) run over TLP over UDP on loopback. Keyed DATA is AES-256-GCM.

## Summary

1. **The latent link works, but it lands near verbose text and far from the information floor. A fixed binary codec reaches the floor.**
   - With the link, B answers 100% of value queries. Without it, B answers 1.3%.
   - The budget rule is the same on both sides: payload bytes on the wire × 8, including the latent per-slot scale, with headers excluded.
   - Bits for 0.99 accuracy: binary codec 88, text 416, latent 640. The information floor is 66.8 bits.
   - Below 512 bits, latent and text take turns: latent is ahead at 192 bits (0.52 against 0.47) and 384 bits (0.98 against 0.86), and text is ahead at 256 bits (0.59 against 0.52) and 320 bits (0.74 against 0.52). There is no single crossover.
   - The codec (4-bit key + 7-bit value per entry) beats both at every budget from 24 bits up.
2. **Loss handling works as designed (a mechanism check).** On TLP over UDP at 10% loss, TLP gets 0.97 accuracy with a 61 ms final answer. TCP-like gets 1.00 at 88 ms, and plain datagrams get 0.86 at 47 ms. The order is by construction. The measured part is the cost and the gap size. With Zipf queries and a pair that orders its slots by popularity, TLP keeps a larger share of TCP-like accuracy at 30% loss on Zipf queries (0.92 against 0.81, difference 0.105 ± 0.026, 1000 messages per cell). But that pair reaches only 0.80 without loss, so control TLP is still better in absolute accuracy (0.811 against 0.726 on Zipf queries, and 0.793 against 0.602 on uniform queries) (6.3b, 6.7b). Priority ordering is learnable and the transport exploits it. At this scale, the ordering costs more accuracy than loss handling recovers.
   - Against text compressed with zstd and sent as one reliable packet, the latent link loses at every loss level from 0% to 30% (6.8). To its first correct answer, latent needs 894 to 1006 bytes on the wire, and text needs 122 to 217. Text stays at 1.000 accuracy, and latent falls to 0.808 at 30% loss. Only the first partial answer is faster with latent (26 to 31 ms, against 27 to 85 ms). That is the progressive-delivery property, and no more.
3. **Discovery, verification and keying work.**
   - Routing by knowledge finds the right holder.
   - The anchor check passes for the same dialect (score 1.00) and fails for a different dialect (0.03).
   - DATA is AES-256-GCM per packet. A tap without the key recovers 0.06% of the table entries (chance). With the escrowed key, it recovers 100%.
4. **OpenTelepathy catches mislabelled messages, in a weak sense.** It detects 97% of them with 0% false positives (AUC 0.996), and its verdicts reduce the influence of a lying peer. But the sender's purpose input is written into every slot, so the observed intent is in effect a second declared-intent field. The result is not evidence against a sender that hides its intent (section 8.1).
5. **A shared nonlinear translator negotiates a working dialect at d = 32 with 256 anchors. At d = 16 it does not. Per-slot translators fail at every width.** The headline metric is the translation loss: the jointly trained pair's accuracy minus the negotiated link's accuracy, on the same 200 queries, mean ± SE over 5 fits (section 10.0).
   - At d = 32, the link answers 0.995 ± 0.002 against 1.000, a loss of 0.005 ± 0.002. 5/5 fits pass the gate (0.90).
   - At d = 16, the link answers 0.750 ± 0.010 against 0.935, a loss of 0.185 ± 0.010. 0/5 fits pass the gate.
   - No linear map exists between two dialects, per slot or for the full message, even with 4096 anchors. A per-slot MLP has too few samples (held-out 0.23 ± 0.03 at d = 32). Sections 10.3 and 10.4 give the reasons and the gate measurements.

## Training notes (what it took to train in under 5 minutes)

Each item below is a measurement, not a design wish.

- **Send-port memory.** When the send port read only A's final hidden states, joint training did not learn K = 8 retrieval in 4000 steps. With frozen trunk blocks, it learned (96% at step 1200). Thus Adam steps on the shared trunk, which the early gradient from an untrained receiver drives, damage the key-value binding. The fix: the send port reads the input-level states and the final states, and the gradient from the send port into the trunk is scaled by 0.1.
- **Cold start.** A learnable positional attention bias starts slot i on table position i + 1. This breaks the cold-start symmetry of the slots.
- **Intent schedule.** When all three intents trained from step 0, the easy count objective took over the slot code, and value retrieval stayed at chance. Value tables come first: 35% of the steps for d ≥ 16, and 45% for d ≤ 8.
- **Warm phase.** Trust, the hint, prefix dropout and quantization start after a warm phase: 50% of the steps, or 60% for d ≤ 8.
- **Symmetric training** over the four (sender, receiver) pairs converges for d = 32 (A to B 1.00, B to A 1.00) and d = 16 (0.92 and 0.94).
  - For d ≤ 8, it does not converge within the time limit. The d ≤ 8 runs first train A to B only, so they serve only the A to B bandwidth sweep. At d = 8, A to B is 0.99 and B to A is 0.02.
  - At d = 4 and d = 2, even A to B stays near the small-table level (0.12 and 0.06).
- **Existence queries** learn only at d = 32 (0.98 for seed 0, 0.61 for seed 1).
- **Zipf queries and the positional prior** (6.3b). With the default prior (3.0), a pair trained on Zipf queries keeps the positional slot order and ignores popularity. With a prior of 1.0 or 0, the sender routes popular entries into the first slots, but in 4000 steps it learns only part of the table (Zipf accuracy 0.57 and 0.61). With 12000 steps (about 9.5 min, two runs in parallel), the prior-1.0 pair gets 0.81 and the prior-0 pair stays at 0.57. These runs are outside the 5-minute budget: `scripts/train.py --zipf`.

## 6.1 Link comparison (N = 8, d = 32, K = 8) — `e1_link.png`

The message comparison is the payload column. The packetisation overhead is in separate columns, because it depends on how a link is cut into packets, not on its encoding: the 28-byte TLP header and the 16-byte AEAD tag are paid once per packet. The latent link sends one slot per packet (8 packets), and text and codec send one packet per message.

| Link | Accuracy | Payload bytes | Packets | Header bytes | AEAD tag bytes | Wire total |
| --- | --- | --- | --- | --- | --- | --- |
| No link | 0.013 | 0 | 0 | 0 | 0 | 0 |
| Latent, b = 16 | 1.000 | 512 | 8 | 224 | 128 | 864 |
| Text at 512 B (the whole table fits) | 1.000 | 55 | 1 | 28 | 16 | 99 |
| Latent, b = 4 | 1.000 | 144 | 8 | 224 | 128 | 496 |
| Text at 144 B (the whole table fits) | 1.000 | 55 | 1 | 28 | 16 | 99 |
| Binary codec at 8 B (64 bits, 5 entries) | 0.638 | 8 | 1 | 28 | 16 | 52 |
| Binary codec, whole table (88 bits) | 1.000 | 11 | 1 | 28 | 16 | 55 |
| Text, unbounded | 1.000 | 50 (mean) | 1 | 28 | 16 | 94 |

At these budgets, the text link carries the whole table in 55 bytes, against 144 to 512 bytes of latent payload, and the binary codec needs 11 bytes. A key-value table is easy to encode exactly. Per-packet overhead makes the gap larger: the latent link pays 8 headers and 8 tags (352 B) for one message, and text pays one of each (44 B). That overhead buys the per-slot loss handling of section 6.7. It is not part of the encoding comparison.

## 6.2 Bandwidth sweep — `e2_bandwidth.png`, `e2_noise.png`

The information floor of one table is log2 C(16, 8) + 8 × log2 100 = 13.7 + 53.1 = **66.8 bits**.

**Budget rule, the same on every side.** Latent bits = payload bytes on the wire × 8: N × 2d bytes at b = 16, and N × (2 + ceil(d × b / 8)) bytes at b < 16, so the per-slot float16 scale is included. Text bits = the encoded bytes × 8. Codec bits = 11 per entry. Headers and the AEAD tag are excluded on every side. (An earlier version of this file counted latent bits as N × d × b without the scale. That favoured the latent link at b < 16.)

Accuracy for each (d, N, b). One model is trained at N = 8 for each d, and it is evaluated on prefixes and quantization levels.

| d | N=1 (b 16/8/4/2) | N=2 | N=4 | N=8 |
| --- | --- | --- | --- | --- |
| 2 | 0.05 0.05 0.05 0.04 | 0.06 0.06 0.06 0.04 | 0.06 0.06 0.06 0.04 | 0.06 0.06 0.05 0.03 |
| 4 | 0.11 0.12 0.12 0.07 | 0.12 0.12 0.12 0.07 | 0.12 0.12 0.12 0.08 | 0.12 0.12 0.11 0.07 |
| 8 | 0.14 0.14 0.14 0.07 | 0.26 0.26 0.26 0.12 | 0.52 0.52 0.52 0.22 | 0.99 0.99 0.98 0.34 |
| 16 | 0.14 0.14 0.14 0.09 | 0.26 0.26 0.26 0.17 | 0.52 0.52 0.52 0.34 | 0.92 0.92 0.92 0.51 |
| 32 | 0.14 0.14 0.14 0.12 | 0.26 0.26 0.26 0.22 | 0.53 0.53 0.53 0.46 | 1.00 1.00 1.00 0.82 |

Best latent configuration at or below each budget, against the text link and the binary codec at the same number of payload bits:

| Payload bits | 32 | 64 | 96 | 128 | 192 | 256 | 320 | 384 | 512 | 640 | 1024 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Latent (best config) | 0.12 | 0.14 | 0.26 | 0.26 | 0.52 | 0.52 | 0.52 | 0.98 | 0.98 | 0.99 | 0.99 |
| Config | N1 d4 b4 | N1 d8 b4 | N2 d8 b4 | N2 d8 b4 | N4 d8 b4 | N4 d8 b4 | N4 d8 b8 | N8 d8 b4 | N8 d8 b4 | N8 d8 b8 | N8 d8 b16 |
| Text (truncated) | 0.01 | 0.13 | 0.23 | 0.23 | 0.47 | 0.59 | 0.74 | 0.86 | 1.00 | 1.00 | 1.00 |
| Binary codec | 0.23 | 0.59 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

**The latent link lands near verbose text and far from the floor.**
- To reach 0.99, the codec needs 88 bits (1.3 × the floor), text needs 416 bits, and the latent link needs 640 bits (N = 8, d = 8, b = 8), which is 9.6 × the floor.
- Below 512 bits, latent and text take turns. Neither one dominates, so this file states no crossover. The earlier "crossover" came from the budget rule that left out the latent scale.
- The codec beats both at every budget, because it is exact and its entries are 11 bits.
- The text baseline is the verbose format that agents use today, about 6.9 bytes per entry (`k12=07 `). The latent link is comparable to that format, not to an efficient encoding.

**Shape.** Accuracy grows about linearly with the number of slots, because each slot carries about one table entry. B can answer only the queries whose entry has arrived.
- The models do not compress below one entry per slot. At d = 32, b = 2, N = 8 (640 payload bits), accuracy is 0.82.
- Below about 128 bits, accuracy stays near the 1/K "entry in the first slot" level, not at the information floor.
- At d ≤ 4, the models do not learn the code at all.

**Noise sweep** (d = 32, N = 8, additive Gaussian noise relative to the slot RMS):

| Noise / RMS | 0 | 0.1 | 0.2 | 0.3 | 0.5 | 0.75 | 1.0 | 1.5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Accuracy | 0.999 | 0.999 | 0.999 | 0.993 | 0.913 | 0.628 | 0.392 | 0.156 |

Accuracy stays at 0.99 or more up to noise of 0.3 × RMS, so the models need about 0.5 × log2(1 + 1/0.09) ≈ 1.8 bits of resolution per dimension. That agrees with the quantization result: b = 4 matches b = 16, and b = 2 (ternary, 1.58 bits) loses accuracy. Half accuracy is at noise 1.0 × RMS (≈ 0.5 bits per dimension). The effective capacity is far below the 16 nominal bits.

## 6.3 Progressive delivery — `e3_progressive.png`

| Slots received | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Latent b = 16 (bytes) | 0.134 (64) | 0.264 (128) | 0.407 (192) | 0.509 (256) | 0.635 (320) | 0.765 (384) | 0.885 (448) | 1.000 (512) |
| Latent b = 4 (bytes) | 0.135 (18) | 0.264 (36) | 0.407 (54) | 0.509 (72) | 0.635 (90) | 0.765 (108) | 0.885 (126) | 1.000 (144) |
| Text, same bytes as b = 4 | 0.389 | 0.715 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Text, same fraction of its length | 0.140 | 0.262 | 0.389 | 0.519 | 0.639 | 0.763 | 0.891 | 1.000 |

Nested dropout gives a monotone curve. The latent link degrades exactly like text truncated to the same fraction: each slot is about one entry. At the same bytes, text wins, because one 18-byte latent slot carries one entry and 18 bytes of text carry about 2.6 entries. The binary codec at the same bytes is 1.00 at every slot count, because one b = 4 slot (18 bytes, 144 bits) already holds the whole 88-bit table.

### 6.3b Skewed queries — `e3_progressive_zipf.png`

With uniform queries, every table entry has the same value, so no slot order can help a prefix. This part repeats the sweep with Zipf(1) value queries: the key of popularity rank r is queried with weight 1/(r + 1). The ranks 0 to 3 get 56% of the queries, 4 to 7 get 20%, and 8 to 15 get 24%. Each pair is evaluated on Zipf queries (the skewed case) and on uniform queries (what a skewed sender gives up). "Prior" is the initial positional attention bias of the send port (section "Training notes").

| Pair (d = 32) | Zipf: 1 / 2 / 4 / 8 slots | Uniform: 1 / 4 / 8 slots | Zipf, 4 slots, by rank 0–3 / 4–7 / 8–15 |
| --- | --- | --- | --- |
| Uniform queries, prior 3.0, 4000 steps (control) | 0.15 / 0.27 / 0.51 / 1.00 | 0.13 / 0.51 / 1.00 | 0.53 / 0.49 / 0.49 |
| Zipf, prior 3.0, 4000 steps | 0.14 / 0.27 / 0.51 / 1.00 | 0.14 / 0.51 / 1.00 | 0.53 / 0.49 / 0.50 |
| Zipf, prior 1.0, 4000 steps | 0.38 / 0.46 / 0.54 / 0.57 | 0.14 / 0.33 / 0.34 | 0.66 / 0.54 / 0.07 |
| Zipf, prior 0, 4000 steps | 0.36 / 0.41 / 0.52 / 0.61 | 0.12 / 0.34 / 0.45 | 0.53 / 0.56 / 0.13 |
| **Zipf, prior 1.0, 12000 steps** | **0.38 / 0.47 / 0.63 / 0.81** | 0.14 / 0.47 / 0.69 | **0.73** / 0.43 / 0.35 |
| Zipf, prior 0, 12000 steps | 0.38 / 0.45 / 0.54 / 0.57 | 0.14 / 0.37 / 0.38 | 0.57 / 0.58 / 0.16 |
| Text, same fraction of its bytes | 0.17 / 0.44 / 0.70 / 1.00 | 0.14 / 0.52 / 1.00 | — |

**The trade-off is real: a sender can learn a priority order, but in this budget it gives up the rare entries.**
- With the default prior (3.0), the Zipf-trained sender does not reorder anything. Slot i still carries table position i, and the curves are the same as the control. The strong prior fixes the slot order before the query skew can act.
- With a weak prior, the sender routes the popular entries into slot 0 by content. One slot then answers 38% of Zipf queries, against 15% for the control. The top-ranked key is answered at 0.95 to 0.99 from the protected half alone.
- The cost is the rare entries. The best Zipf pair (prior 1.0, 12000 steps) reaches only 0.81 on Zipf queries and 0.69 on uniform queries with all 8 slots, against 1.00 for the control. The ranks 8 to 15 get 0.48 with all slots. Several slots carry copies of the popular entries (the query mass of slots 0 and 5 is 0.37 each).
- The best Zipf pair fails its own same-dialect anchor check (0.84 and 0.81, against the gate 0.90). The anchors use the same Zipf queries.
- Text caveat: the popularity order is a fixed random permutation (seed 4242). By chance, the four most popular keys are k1 to k4, so the key-sorted text sends them first. The text row is therefore in effect "text sorted by popularity".

The progressive gain is real only in the prefix. At 1 to 5 slots, the best Zipf pair beats the control on Zipf queries (0.38 against 0.15 at 1 slot, 0.63 against 0.51 at 4, 0.71 against 0.64 at 5). From 6 slots on, the control is better.

## 6.4 Trust — `e4_trust.png`

The table shows accuracy with the correct peer, accuracy with a mismatched peer, and the gate activation (correct peer). The hint is correct with probability s.

| s \ t | 0 | 0.25 | 0.5 | 0.75 | 1 |
| --- | --- | --- | --- | --- | --- |
| 0.3 | 0.29 / 0.29 / 0.05 | 0.30 / 0.27 / 0.12 | 0.48 / 0.19 / 0.25 | 0.85 / 0.06 / 0.45 | 0.98 / 0.02 / 0.66 |
| 0.6 | 0.59 / 0.57 / 0.05 | 0.60 / 0.54 / 0.12 | 0.68 / 0.36 / 0.25 | 0.90 / 0.11 / 0.45 | 0.98 / 0.04 / 0.65 |
| 0.9 | 0.91 / 0.88 / 0.05 | 0.91 / 0.83 / 0.12 | 0.93 / 0.55 / 0.25 | 0.98 / 0.17 / 0.45 | 1.00 / 0.06 / 0.66 |

The trust conditioning is real:
- At low t, B ignores the peer and uses its hint. Accuracy is about s with either peer.
- At high t, B follows the slots. With the correct peer it is near 1.0, and with the mismatched peer it is near 0.

The correct and mismatched curves start at the same point at t = 0 and diverge as t grows. They move apart at about t = 0.5 for every s. The brief expected that the switch point would move with s. It does not: the gate depends mostly on t and does not compare t with s. The gate activation does not depend on the peer (0.05 at t = 0, 0.66 at t = 1).

## 6.5 Dialect mismatch — `e5_dialect.png`

- **Cross-seed handshake (A′ to B).** It fails at ANCHOR. The local score is 0.031 and the peer score is 0.000, against the gate 0.90. The session closes with RESET.
- **Same-dialect handshake.** The anchor scores are 1.00 and 0.97.

| Pair | Accuracy |
| --- | --- |
| A to B (joint) | 1.000 |
| A′ to B′ (joint, seed 1) | 0.999 |
| A′ to B, forced through | 0.011 |
| A to B′, forced through | 0.011 |

## 6.6 Monitor tap — `e6_monitor_tap.png`

Keyed DATA is AES-256-GCM per packet (SPEC.md section 12). A small MLP probe decodes the table entry that each slot carries. It is trained on 20,000 captured messages and tested on 2000, with one fresh session key per message.

| Tap setting | Table entries recovered (key and value exact) |
| --- | --- |
| (a) With the key (as the escrow probe has it): decrypt, then read the slots | 1.000 |
| (b) Without the key: the ciphertext bytes, read as slot values | 0.0006 (chance 1/1600 = 0.000625) |

Without the key, the tap learns nothing. With the key, it reads everything, because the slots themselves are not obfuscated. Confidentiality comes from the cipher and the key, and the escrowed probe can read every session.

## 6.7 Loss handling: a mechanism check — `e7_transport.png`

This experiment is a mechanism check, not a finding. The accuracy order "datagrams < TLP < TCP-like" is true by construction: TLP retransmits the first N/2 slots and never the others. What the experiment measures is the cost of that choice (time and bytes) and the size of the gaps.

The setup:
- Messages: N = 8, d = 32, b = 16, with P = 4 protected slots for TLP. DATA is keyed (AES-GCM).
- Transport: TLP over UDP on loopback, with 20 ms one-way delay and identical seeded loss in both directions, through the loss injector.
- The three disciplines send the same TLP packets. "Datagrams" sends each slot once, with no TLP recovery (the "UDP, no TLP" baseline). "TCP-like" makes every slot reliable, with in-order delivery and head-of-line blocking. It is a discipline, not a kernel TCP. "TLP" is TLP over UDP.
- 100 messages per cell, so the standard error of an accuracy is about 0.04. Bytes count both directions, with headers and AEAD tags.

| Loss | Datagrams: acc / final ms / bytes | TCP-like: acc / final ms (p95) / bytes / retx | TLP over UDP: acc / final ms (p95) / bytes / retx |
| --- | --- | --- | --- |
| 0% | 1.00 / 37 / 885 | 1.00 / 38 (47) / 893 / 0.00 | 1.00 / 39 (49) / 885 / 0.00 |
| 1% | 0.99 / 41 / 893 | 1.00 / 44 (79) / 915 / 0.16 | 1.00 / 42 (64) / 897 / 0.03 |
| 2% | 0.97 / 42 / 893 | 1.00 / 48 (81) / 940 / 0.34 | 1.00 / 41 (77) / 895 / 0.07 |
| 5% | 0.97 / 42 / 893 | 1.00 / 55 (85) / 959 / 0.54 | 0.99 / 54 (100) / 970 / 0.56 |
| 10% | 0.86 / 47 / 893 | 1.00 / 88 (229) / 1244 / 2.66 | 0.97 / 61 (145) / 1017 / 0.98 |
| 20% | 0.76 / 51 / 892 | 1.00 / 107 (225) / 1689 / 6.09 | 0.93 / 89 (150) / 1177 / 2.13 |
| 30% | 0.63 / 60 / 893 | 1.00 / 160 (312) / 1874 / 7.69 | 0.83 / 109 (155) / 1304 / 3.23 |

The time to the first answer is 26 to 31 ms for all three disciplines at every loss level.

What the mechanism costs and gives:
- At 10% loss, TLP keeps 0.97 accuracy and gives the final answer at 61 ms, against 88 ms for TCP-like. It sends 1017 B, against 1244 B.
- At 30% loss, TLP keeps 0.83, against 1.00 for TCP-like and 0.63 for datagrams. Its p95 final time stays near 155 ms, against 312 ms for TCP-like.
- The accuracy gap between TLP and TCP-like is small up to 10% loss, because each slot carries about one entry and every entry has the same value. A lost best-effort slot costs one entry. The Zipf comparison (6.7b) is the part of this experiment where a gap can move.

### 6.7b Skewed queries — `e7_transport_zipf.png`, `e7_transport_zipf_control.png`

The same sweep on Zipf(1) value queries, with two pairs: the best Zipf pair (prior 1.0, 12000 steps, section 6.3b), and the uniform-trained pair as the control on the same queries. The anchor threshold is 0 in all e7 cells, because the Zipf pair fails its own anchor check (6.3b).

| Loss | Zipf pair: datagrams / TCP-like / TLP acc | TLP final ms (TCP-like) | Control: datagrams / TCP-like / TLP acc | TLP final ms (TCP-like) |
| --- | --- | --- | --- | --- |
| 0% | 0.80 / 0.80 / 0.80 | 37 (34) | 1.00 / 1.00 / 1.00 | 42 (42) |
| 5% | 0.80 / 0.80 / 0.81 | 60 (59) | 0.95 / 1.00 / 0.97 | 52 (50) |
| 10% | 0.75 / 0.80 / 0.78 | 77 (86) | 0.88 / 1.00 / 0.91 | 66 (80) |
| 20% | 0.67 / 0.79 / 0.74 | 100 (136) | 0.79 / 1.00 / 0.90 | 90 (118) |
| 30% | 0.58 / 0.80 / 0.70 | 112 (193) | 0.68 / 1.00 / 0.77 | 107 (164) |

**The relative gap opens a little, but the absolute result is worse.**
- With the Zipf pair, TLP keeps 87% of TCP-like accuracy at 30% loss and 94% at 20%. The control keeps 77% and 90%. Datagrams keep 72% with the Zipf pair and 68% with the control. The protected half holds the popular entries, so a lost best-effort slot costs less.
- The 100-message cells above cannot settle these differences. The 30% loss cell was therefore run again with 1000 messages per cell, under both query distributions. The Zipf rows are `results/e7_zipf_cell.json`, and the uniform rows are `results/e7_zipf_cell_uniform.json`. The SE of a share comes from a bootstrap over messages.

| 30% loss, 1000 messages | Queries | TLP acc ± SE | TCP-like acc ± SE | TLP share of TCP-like ± SE |
| --- | --- | --- | --- | --- |
| Zipf pair | Zipf(1) | 0.726 ± 0.014 | 0.790 ± 0.013 | 0.919 ± 0.023 |
| Uniform-trained control | Zipf(1) | 0.811 ± 0.012 | 0.996 ± 0.002 | 0.814 ± 0.013 |
| Zipf pair | Uniform | 0.602 ± 0.015 | 0.698 ± 0.015 | 0.862 ± 0.029 |
| Uniform-trained control | Uniform | 0.793 ± 0.013 | 0.996 ± 0.002 | 0.796 ± 0.013 |

  The 0.726 and 0.811 figures in the earlier version of this section, and in the summary, are under Zipf(1) queries.
  - **Zipf queries.** The share that TLP keeps is higher with the Zipf pair by **0.105 ± 0.026** (4.0 SE). The SE is below a third of the difference, so the relative effect is settled. The absolute effect goes the other way: control TLP is better than Zipf TLP by 0.085.
  - **Uniform queries.** The share is higher with the Zipf pair by 0.066 ± 0.032 (2.1 SE), which does not settle the relative effect. The absolute gap is larger: control TLP is better than Zipf TLP by 0.191, because the Zipf pair gives up the rare entries (6.3b).
- The control is better in absolute accuracy at every loss level, because the Zipf pair has a ceiling of 0.80 (6.3b).

**Priority ordering is learnable and the transport exploits it. At this scale, the ordering costs more accuracy than loss handling recovers.**
- The Zipf pair is worse in absolute accuracy under both query distributions, including the Zipf distribution that it was trained on. It keeps a larger share of a lower ceiling.
- The share effect is real, and on Zipf queries it is settled (4.0 SE). The training trade-off is larger than that effect, so the brief's prediction is not supported as written.
- A two-stage training could change this result: train on the full table first, then fine-tune the order. That is a question for Telepathy-1.

## 6.8 Bits under loss — `e8_bits_under_loss.png`, `results/e8_bits_under_loss.json`

**Compressed text beats the latent link at every loss level, on bytes, on accuracy and on the final answer time. The latent link gives a first answer sooner.**

The setup:
- Latent: the exchange of 6.7. N = 8, d = 32, b = 16, TLP over UDP, P = 4 protected slots, keyed DATA.
- Text: the serialised table of 6.1 (`k3=19 k7=42 …`), compressed with zstd level 19, sent as one keyed DATA packet with the TCP-like discipline. The packet is retransmitted until it is acknowledged (at most 50 times). B is the text-baseline model of 6.1, and it reads the decompressed text. The packet is a byte message (SPEC.md section 3).
- Both: TLP over UDP on loopback, 20 ms one-way delay, identical seeded loss in both directions, 1000 value queries per cell, uniform queries, the same session handshake.
- Bytes are counted in both directions on the wire for each message: TLP headers, AEAD tags, payloads, ACKs and retransmissions. The handshake is excluded.
- Bytes to the first correct answer are the wire bytes from the start of the message to the first answer of B that is correct. A byte counts when it is sent, so the latent figure includes all 8 slots, because the sender sends them together. Messages that are never answered correctly are excluded from this mean, and the table gives their count.
- The text table is 50.0 bytes on average, and 47.9 bytes after zstd. At level 3, zstd makes it larger (59.0 bytes), because a zstd frame has a fixed header.

| Loss | Link | Accuracy ± SE | Bytes per message ± SE | Bytes to the first correct answer ± SE | Never correct | First answer ms ± SE | Final answer ms ± SE (p95) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0% | Latent | 0.999 ± 0.001 | 894 ± 0 | 894 ± 0 | 1 | 26.1 ± 0.1 | 33.6 ± 0.1 (37) |
| 0% | Text, zstd | 1.000 ± 0.000 | 122 ± 0 | 122 ± 0 | 0 | 27.1 ± 0.1 | 27.1 ± 0.1 (31) |
| 5% | Latent | 0.975 ± 0.005 | 955 ± 4 | 918 ± 3 | 24 | 26.3 ± 0.1 | 46.9 ± 0.7 (90) |
| 5% | Text, zstd | 1.000 ± 0.000 | 134 ± 1 | 132 ± 1 | 0 | 33.2 ± 0.7 | 33.2 ± 0.7 (38) |
| 10% | Latent | 0.947 ± 0.007 | 1026 ± 6 | 932 ± 4 | 53 | 30.5 ± 0.1 | 67.8 ± 1.1 (148) |
| 10% | Text, zstd | 1.000 ± 0.000 | 145 ± 2 | 143 ± 2 | 0 | 45.3 ± 1.5 | 45.3 ± 1.5 (142) |
| 20% | Latent | 0.883 ± 0.010 | 1173 ± 9 | 981 ± 6 | 116 | 28.1 ± 0.2 | 89.3 ± 1.3 (152) |
| 20% | Text, zstd | 1.000 ± 0.000 | 181 ± 3 | 173 ± 3 | 0 | 60.7 ± 3.4 | 60.7 ± 3.4 (147) |
| 30% | Latent | 0.808 ± 0.012 | 1316 ± 11 | 1006 ± 8 | 192 | 28.4 ± 0.2 | 110.1 ± 1.3 (155) |
| 30% | Text, zstd | 1.000 ± 0.000 | 225 ± 5 | 217 ± 5 | 0 | 85.1 ± 4.9 | 85.1 ± 4.9 (289) |

- **Bytes.** The latent link needs 894 to 1006 bytes on the wire to the first correct answer, and text needs 122 to 217 bytes: 4.6 to 7.3 times less. Total bytes divided by the number of correct answers: latent 895 at 0% loss and 1628 at 30%, text 122 and 225.
- **Accuracy.** Text keeps 1.000 at every loss level, because its single packet is retransmitted until it arrives. The latent link falls to 0.808 at 30% loss, because TLP never retransmits the best-effort slots.
- **Time.** The latent link gives its first answer at 26 to 31 ms at every loss level, because the first slot that arrives gives a partial answer. Text gives its only answer at 27 ms without loss and at 85 ms at 30% loss. The mean final answer is earlier for text at every loss level. At 30% loss, the p95 of text (289 ms) is later than the p95 of latent (155 ms), because a packet that is lost more than once waits for longer retransmit timeouts.
- **Why.** The whole compressed text fits in one datagram, so there is no loss cliff for the latent link to avoid. A loss costs text a retransmit of one 92-byte packet, so text pays mostly in time and little in bytes. The latent link pays 8 headers and 8 tags (352 bytes) before its payload, and a loss costs it an entry or a slot retransmit.

**The loss argument fails on this task.** The one property of the latent link that survives is progressive delivery: B has a partial answer after the first slot arrives. This experiment shows no more than that. A latent advantage under loss can exist only for a message that spans many packets, such as content of passage size. That is a question for Telepathy-1.

## 8. OpenTelepathy

### 8.1 Probe accuracy — `t1_probe_acc.png`

The probe accuracy is 1.000 for SHARE_VALUE, SHARE_KEY_EXISTENCE and SHARE_ENTRY_COUNT, and 1.000 at every prefix length from 1 to 8 slots (3000 messages).

**What the probe reads.** The send port adds the purpose embedding to every slot query, so the declared purpose is written into every slot, and a small classifier reads it from any one slot. The observed intent is therefore, in effect, a second declared-intent field: it shows the purpose that the sender's own port was given, not what the sender hides. The detection rates in 8.1 and 8.2 measure agreement between the header label and the purpose input. They are not evidence against a sender that hides its intent. An adaptive sender (deferred to Telepathy-1) is the real test.

### 8.2 Mismatch detection — `t2_mismatch_roc.png`

The test used 3000 messages: half honest, half generated under one intent and declared as another.

| Metric | Value |
| --- | --- |
| ROC AUC (score = 1 − P(declared)) | 0.996 |
| Detection rate at the service operating point (observed ≠ declared) | 97.1% |
| False-positive rate | 0.0% |
| RESET rate at confidence ≥ 0.9 | 96.9% of lies, 0% of honest messages |

### 8.3 With and without the session key — `t3_probe_key.png`

| Probe input | Intent accuracy |
| --- | --- |
| With the escrowed key (decrypted) | 1.000 |
| Without the key (AES-GCM ciphertext, read as slots) | 0.338 (chance 0.333) |

### 8.4 Tracing overhead — `t4_trace_overhead.png`

- **Bytes.** At N = 8 and d = 32, a keyed message is 864 B without the extension and 1088 B with it (+26%: 28 B per packet). Each packet has the 28-byte header, 64 bytes of slot payload and the 16-byte AEAD tag.
- **Small slots.** At d = 4, the 28-byte extension is larger than the 8-byte slot payload. A message grows from 416 B to 640 B (+54%). That is why the section 6 sweeps run with tracing off.
- **Latency.** Over UDP on loopback, the latency (about 0.8 ms for each message) does not change measurably.

### 8.5 Trust loop — `t5_trust_loop.png`

The setup:
- A peer sends 30 messages in a session. Messages 10 to 29 are encoded for key existence but declared SHARE_VALUE.
- B asks value queries with a local hint (s = 0.6).
- There are 12 sessions for each setting.

| | Accuracy on honest messages | Accuracy on injected messages | Trust on injected messages |
| --- | --- | --- | --- |
| VERDICT loop on | 0.992 | 0.617 | 0.10 |
| VERDICT loop off | 0.992 | 0.321 | 1.00 |

With the loop on, the first injected message is still read at trust 0.9 to 1.0 in most sessions, because the verdict arrives after B finalises. From the next message on, the trust is 0.1, and B falls back to its hint: accuracy is about s. With the loop off, B keeps following the bad slots.

The loop protects integrity only. The exfiltration in the demo (the whole table under a "count" label) reaches B before any verdict.

## 10. Negotiation between strangers

Every score of a stochastic translator fit (the MLP families) is the mean ± SE over 5 seeded fits. A fit runs on one torch thread with a fixed seed, and one fit at a time in a process, so it is reproducible (`negotiate/translators.py`). The deterministic families (linear, Procrustes) need one fit. The held-out set has 32 anchors, so one anchor is 0.03 of a held-out score.

The gate is 0.90: both held-out scores must be 0.90 or more (29 of 32 anchors) before data flows. The gate is a policy parameter (`TLPOptions.anchor_threshold`), not a measured threshold. No experiment derives 0.90 from data.

### 10.0 Translation loss (the headline metric) — `results/n4_gate.json`

**A shared nonlinear translator negotiates a working dialect at d = 32 with 256 anchors. At d = 16 it does not. Per-slot translators fail at every width.**

Translation loss is the accuracy of the jointly trained pair minus the accuracy of the negotiated link, on the same queries. The negotiated link is a real A′ to B session over TLP over UDP, with the shared MLP translator from 256 fitting anchors, trust 1 and no loss. Each width has 5 translator fits, and each established session carries the same 200 value queries. The jointly trained pair (A to B, seed 0) answers those 200 queries with no translator and no network. The loss of each fit is a paired difference, so its SE is the SE of the negotiated accuracy over the 5 fits.

| d | Negotiated link (mean ± SE, 5 fits) | Jointly trained pair | Translation loss (mean ± SE) | Pass at the gate 0.90 |
| --- | --- | --- | --- | --- |
| 32 | 0.995 ± 0.002 | 1.000 | 0.005 ± 0.002 | 5/5 |
| 16 | 0.750 ± 0.010 | 0.935 | 0.185 ± 0.010 | 0/5 |

- At d = 32, the negotiated link loses half a percentage point against its jointly trained ceiling, and every fit passes the gate.
- At d = 16, the link answers 0.750 against 0.935, and no fit passes the gate. The highest held-out score of any d = 16 fit is 0.844. On the 1000-query set of 10.1, the d = 16 joint pair answers 0.924.
- Without a translator, A′ forced through to B answers 0.020 of the same queries at both widths.
- The d = 16 numbers come from the gate-0 run of 10.4, in which every fit gets through. The fits are deterministic, so pass or fail at 0.90 follows from their held-out scores.

The per-slot families never come near a working link. This table uses the offline evaluation of 10.1: 256 fitting anchors, 1000 value queries, no network. The loss is against the joint pair on the same 1000 queries (0.999 at d = 32, 0.924 at d = 16). The held-out score is A′ to B.

| Translator, 256 anchors | d = 32: accuracy / loss / held-out | d = 16: accuracy / loss / held-out | Pass at 0.90 |
| --- | --- | --- | --- |
| Linear, per slot | 0.042 / 0.957 / 0.09 | 0.022 / 0.902 / 0.00 | No, at both widths |
| Procrustes, per slot | 0.014 / 0.985 / 0.00 | 0.007 / 0.917 / 0.03 | No, at both widths |
| MLP, per slot | 0.307 ± 0.005 / 0.692 ± 0.005 / 0.23 ± 0.03 | 0.132 ± 0.002 / 0.792 ± 0.002 / 0.07 ± 0.01 | No, at both widths |
| Shared MLP (one map for all slots) | 0.989 ± 0.001 / 0.010 ± 0.001 / 0.99 ± 0.01 | 0.755 ± 0.002 / 0.169 ± 0.002 / 0.74 ± 0.03 | Yes at d = 32, no at d = 16 |

Sections 10.1 to 10.4 give the anchor sweep, the cost, the diagnosis and the gate measurements behind these tables.

### 10.1 Accuracy against fitting anchors — `n1_anchors.png`

The test translates A′ messages into B's dialect and evaluates them on 1000 value queries. Each cell is eval accuracy / held-out score (gate 0.90).

| d = 32 | 16 | 32 | 64 | 128 | 256 |
| --- | --- | --- | --- | --- | --- |
| Linear | 0.017 / 0.00 | 0.036 / 0.03 | 0.049 / 0.06 | 0.059 / 0.00 | 0.042 / 0.09 |
| Procrustes | 0.014 / 0.00 | 0.025 / 0.00 | 0.020 / 0.00 | 0.016 / 0.00 | 0.014 / 0.00 |
| MLP, per slot | 0.018 / 0.00 ± 0.00 | 0.039 / 0.03 ± 0.01 | 0.066 / 0.09 ± 0.02 | 0.140 / 0.08 ± 0.03 | 0.307 / 0.23 ± 0.03 |
| Shared MLP (one map for all slots) | 0.142 / 0.04 ± 0.01 | 0.362 / 0.26 ± 0.02 | 0.709 / 0.68 ± 0.02 | 0.958 / 0.91 ± 0.02 | 0.989 / 0.99 ± 0.01 |

At d = 32, the ceiling (joint pair) is 0.999, the floor (forced through) is 0.010, and the full-message linear diagnostic at 256 anchors is 0.017.

| d = 16 | 16 | 32 | 64 | 128 | 256 |
| --- | --- | --- | --- | --- | --- |
| Linear | 0.027 / 0.03 | 0.019 / 0.06 | 0.017 / 0.00 | 0.023 / 0.00 | 0.022 / 0.00 |
| Procrustes | 0.012 / 0.03 | 0.011 / 0.00 | 0.011 / 0.03 | 0.014 / 0.00 | 0.007 / 0.03 |
| MLP, per slot | 0.028 / 0.03 ± 0.00 | 0.036 / 0.03 ± 0.01 | 0.053 / 0.01 ± 0.01 | 0.086 / 0.06 ± 0.02 | 0.132 / 0.07 ± 0.01 |
| Shared MLP (one map for all slots) | 0.098 / 0.15 ± 0.02 | 0.177 / 0.17 ± 0.02 | 0.341 / 0.28 ± 0.03 | 0.608 / 0.61 ± 0.03 | 0.755 / 0.74 ± 0.03 |

At d = 16, the ceiling is 0.924, the floor is 0.011, and the full-message linear diagnostic is 0.014.

**Negotiation works at d = 32 with the shared MLP, and fails at d = 16.**
- At d = 32, the shared MLP passes the held-out check at 256 anchors in both directions (A′ to B 0.99 ± 0.01, B to A′ 0.97 ± 0.00). It is the protocol default.
- **Change from the earlier result.** The first version of this file reported the best result as the per-slot MLP at 256 anchors: held-out 0.16 (one fit). Over 5 fits, that family gets 0.23 ± 0.03. The shared MLP gets 0.99 ± 0.01, with eval accuracy 0.989 ± 0.001 against a ceiling of 0.999.
- At d = 16, no family reaches the gate 0.90 within 256 anchors. The best is the shared MLP: 0.74 ± 0.03 held-out, against a ceiling of 0.92 for the jointly trained pair. Section 10.4 measures how much of that link works.
- Every per-slot family and the full-message linear map fail at both values of d. Section 10.3 gives the reasons.

**Probe on A′ messages.** The probe reads 49.9% of the intents of A′ messages untranslated. With a linear translator fitted on the witnessed anchors, it reads 33.9% (chance). With a shared MLP from 256 witnessed anchors, it reads 74.9%. The anchors are value tables only, so the translator never sees a count or existence message. This probably limits the probe on those intents. The probe records the dialect in which it read each message (`tlp.dialect.read_in`).

### 10.2 Handshake cost — `n2_cost.png`

The test used real handshakes over TLP over UDP, A′ to B, with the default `shared_mlp` translator (seed 0). The threshold was set to 0 so that every handshake completes and can be measured.

| Fitting anchors | Handshake bytes (both directions) | Negotiation ms | Handshake ms | Held-out score |
| --- | --- | --- | --- | --- |
| Same dialect (no negotiation) | 49,716 | — | 41 | 1.00 |
| 16 | 74,278 | 1,014 | 1,015 | 0.03 |
| 32 | 98,786 | 770 | 1,464 | 0.25 |
| 64 | 147,808 | 2,436 | 2,437 | 0.62 |
| 128 | 245,866 | 2,255 | 4,328 | 0.84 |
| 256 | 441,958 | 8,314 | 8,315 | 1.00 |

The byte cost grows linearly: about 1.5 KiB per fitting anchor (8 slots × 92 B, in both directions, plus ACKs). The time is mostly the `shared_mlp` fit (256 hidden units, 1500 full-batch steps). Both sides run in one process here, and the fits run one at a time for reproducibility, so the negotiation time is about two fits: 8.3 s at 256 anchors. In separate processes (the demo), the two fits run at the same time (about 3 to 4 s). The anchor exchange sends at most 32 anchor messages before it waits for ACKs (SPEC.md 6.1).

### 10.3 Why negotiation failed, and why the shared MLP works — `n3_diagnostics.png`

The test measured five things. Anchor counts above 256 exceed the brief's limit. They are a diagnostic only: they separate "too few anchors" from "no map of this family exists".

**1. Self-decode works, so the translator target is good.** Each model's receive port reads each model's send port, on 1000 value queries:

| Pair | A to A | A to B | B to A | B to B |
| --- | --- | --- | --- | --- |
| Seed 0, d = 32 (B) | 0.997 | 0.999 | 0.998 | 0.999 |
| Seed 1, d = 32 (A′) | 0.999 | 1.000 | 0.997 | 1.000 |
| Seed 0, d = 16 | 0.932 | 0.924 | 0.937 | 0.918 |
| Seed 1, d = 16 | 0.922 | 0.912 | 0.930 | 0.915 |

Symmetric training took at d = 32 and d = 16. (The "A to B only" note in the training notes applies to d ≤ 8, which no negotiation test uses.)

**2. The slot roles are the same in every dialect.** For each sender, the test changed the value of one table entry and measured the change of each slot. In all 8 senders (2 seeds × 2 sides × 2 values of d), slot i responds most to table entry i, with 71% to 88% of its change. The positional attention bias that starts training (slot i to position i) fixes the slot order in both dialects. Thus slot i to slot i is the correct unit, and a different slot order does not explain the failure.

**3. Fitting-set accuracy against held-out accuracy (A′ to B, d = 32):**

| Family | Anchors | Fitting set | Held-out anchors | Eval set (1000) | R² fit / held-out |
| --- | --- | --- | --- | --- | --- |
| Linear, per slot | 256 | 0.152 | 0.094 | 0.042 | 0.42 / 0.27 |
| Linear, per slot | 4096 | 0.043 | 0.031 | 0.047 | 0.35 / 0.34 |
| Full-message linear | 256 | 0.641 | 0.062 | 0.017 | 0.74 / −0.57 |
| Full-message linear | 4096 | 0.050 | 0.031 | 0.032 | 0.31 / 0.24 |
| MLP, per slot | 256 | 0.900 ± 0.007 | 0.225 ± 0.027 | 0.307 ± 0.005 | 0.92 / 0.57 |
| MLP, per slot | 4096 | 0.649 ± 0.012 | 0.613 ± 0.025 | 0.628 ± 0.015 | 0.84 / 0.83 |
| Linear, shared | 4096 | 0.032 | 0.000 | 0.044 | 0.31 / 0.30 |
| MLP, shared | 256 | 0.998 ± 0.001 | 0.994 ± 0.006 | 0.989 ± 0.001 | 0.97 / 0.94 |

- **Low fitting accuracy: no linear map exists.** With 4096 anchors, the linear families cannot fit even their own fitting set (0.03 to 0.05). This is true per slot, shared, and for the full message. The full-message map fits 256 anchors only because it has more parameters than data points (256 × 256 + 256), and it does not generalise. Procrustes is lower still (R² 0.12 at 4096), so the two codes are not rotations of each other.
- **High fitting accuracy, low held-out accuracy: too few anchors.** The per-slot MLP at 256 anchors fits 0.90 and generalises to 0.23. Each slot's map sees only 256 samples of a code for 1600 (key, value) pairs.
- **The shared MLP pools the 8 slots: 2048 samples for one map.** It fits 1.00 and generalises to 0.99. This works because all slots use nearly the same code for an entry.

**4. Sharing or capacity?** The shared MLP has more capacity than the per-slot MLP (256 hidden units and 1500 steps, against 64 and 300). A control at 256 anchors, d = 32, separates the two effects:

| Translator | Fitting set | Held-out | Eval |
| --- | --- | --- | --- |
| MLP per slot, 64 hidden, 300 steps | 0.900 ± 0.007 | 0.225 ± 0.027 | 0.307 ± 0.005 |
| MLP per slot, 256 hidden, 1500 steps | 1.000 ± 0.000 | 0.481 ± 0.035 | 0.465 ± 0.005 |
| MLP shared, 64 hidden, 300 steps | 0.605 ± 0.035 | 0.400 ± 0.040 | 0.503 ± 0.031 |
| MLP shared, 256 hidden, 1500 steps | 0.998 ± 0.001 | 0.994 ± 0.006 | 0.989 ± 0.001 |

With equal capacity, the per-slot MLP overfits and the shared MLP generalises. The small shared MLP underfits. Both sharing and capacity are necessary.

**5. Per-direction scores.** Each side of a handshake scores the peer's held-out anchors in its own dialect. At 256 anchors (held-out / eval):

| Direction | d = 32 | d = 16 |
| --- | --- | --- |
| A′ to B (B translates and reads) | 0.99 ± 0.01 / 0.989 | 0.74 ± 0.03 / 0.755 |
| B to A′ (A′ translates and reads) | 0.97 ± 0.00 / 0.977 | 0.76 ± 0.03 / 0.691 |

At d = 16, even 1024 anchors do not pass in either direction: B to A′ gets 0.89 ± 0.02, and A′ to B gets 0.77 ± 0.02 (and 0.78 ± 0.02 at 4096).

**Comparison with representation alignment at scale.** Orthogonal alignment fails here at every anchor count. The two dialects started from random weights and trained on a small discrete task, with no shared pretraining. Each slot encodes one of 1600 discrete (key, value) pairs in a code that the training run chose. Nothing makes two such codes isometric. A possible reason why orthogonal maps can work between large pretrained models is that their representations come from similar data. This experiment does not test that.

### 10.4 How much of the d = 16 link works — `results/n4_gate.json`

Real A′ to B handshakes over TLP over UDP, shared MLP, 256 fitting anchors, 5 translator seeds. Each established session then carries 200 value queries (trust 1, no loss). Both sides must reach the gate. The gate 0.80 run is a measurement, not a policy change: the protocol default stays at 0.90 (a policy parameter, section 10).

| d = 16, seed | Held-out, A′ to B | Held-out, B to A′ | Task accuracy (200 queries) |
| --- | --- | --- | --- |
| 0 | 0.812 | 0.812 | 0.780 |
| 1 | 0.750 | 0.781 | 0.745 |
| 2 | 0.688 | 0.844 | 0.745 |
| 3 | 0.656 | 0.719 | 0.720 |
| 4 | 0.812 | 0.656 | 0.760 |
| **Mean ± SE** | **0.744 ± 0.032** | **0.762 ± 0.034** | **0.750 ± 0.010** |

(This table is the gate-0 run, in which every fit gets through. The scores are the same as in the gate-0.80 run, because the fits are deterministic.)

- **With the gate at 0.80, 1/5 handshakes pass.** Only seed 0 has both scores at or above 0.80 (0.81 and 0.81). Its task accuracy is 0.78.
- **Over all 5 fits, the link delivers 0.750 ± 0.010 task accuracy**, against 0.92 for the jointly trained d = 16 pair and 0.01 forced through without a translator. The held-out score tracks the task accuracy: about 0.75 for both.
- **Reference, d = 32 at the default gate 0.90:** 5/5 handshakes pass, with held-out 0.994 ± 0.006 (A′ to B) and 0.969 ± 0.000 (B to A′), and task accuracy 0.995 ± 0.002.

Thus at d = 16 about three quarters of the link works. A gate of 0.80 lets one fit in five through. A gate low enough to pass every fit would also pass a link that is wrong on a quarter of its answers.

## Demo

`.venv/bin/python scripts/demo.py` (TLP over UDP, the resolver, the probe, A1, A2 and A′ as separate processes, B in the main process) gave these results:

| Run | Holder | Answer (expected) | Trust |
| --- | --- | --- | --- |
| Path 1, resolve by knowledge, k5 | A1 | 78 (78) | 1.00 |
| Path 1, resolve by knowledge, k12 | A2 | 25 (25) | 1.00 |
| Path 2, ROUTE, k3 | A1 | 35 (35) | 0.90 |
| Path 2, ROUTE, k14 | A2 | 23 (23) | 0.90 |
| Loss run, k9 | A2 | 34 (34) | 1.00 |
| A′ by name, k2 (negotiated, `shared_mlp`, 256 anchors) | A′ | 87 (87) | 1.00 |

More detail on the special runs:
- **Keys and audit.** Every DATA packet carries the KEYED flag (AES-GCM). The probe received the key by escrow for all 7 sessions, decrypted the DATA from the tap port, and sent an `intent.check` for each. No packet failed authentication. The two routed sessions authenticate through the relay's header rewrites.
- **Loss run.** Protected slot 2 was retransmitted twice, and best-effort slots 4 and 6 stayed lost.
- **Intent mismatch.** B asked "count?" through ROUTE, and A1 sent the whole table under a count label. The probe flagged it: declared SHARE_ENTRY_COUNT, observed SHARE_VALUE, confidence 1.00. B's trust fell from 0.90 to 0.09 after the first slot, and the probe sent RESET. The trace spans B, the resolver (route and relay), A1 and the probe, with the `intent.mismatch` event.
- **A′.** The log shows the dialect mismatch at HELLO, then NEGOTIATING, with 0 ANCHOR retransmits (the send window). The held-out scores are 1.000 (B) and 0.969 (A′), so the session goes on to KEYING, ESCROW and ESTABLISHED. B answers correctly. The probe fits its own translator after ESCROW_ACK, reads the message in B's dialect, and sends `VERDICT match=True` about 3.4 s after B's answer. B keeps the session open for the verdict (`await_verdict`, demo only).
- **In-process transport.** `--transport inproc` gives the same results in one process.
- **Raw IP protocol 253: verified on 2026-11-01 for the current code.** As root on this Mac, `sudo .venv/bin/pytest tests/test_raw.py` gave `2 passed in 0.04s`. `sudo .venv/bin/python scripts/demo.py --transport raw` completed all 7 sessions: paths 1 and 2, the loss run, the mismatch run and the negotiated A′ session. The trace of that run is `results/demo_trace.jsonl` (132 spans, local only, because `.gitignore` excludes `results/*.jsonl`). It has 7 `answer` spans, 7 `intent.check` spans and 2 `negotiate` spans. B's answers are the same as in the table above (78, 25, 35, 23, 34 and 87). The only span with status ERROR is the expected `intent.check` of the mismatch run (declared SHARE_ENTRY_COUNT, observed SHARE_VALUE), after which B's trust is 0.09. The A′ negotiation used `shared_mlp` with 256 anchors, with held-out scores of 1.000 (B) and 0.969 (A′).
- **The earlier raw run, for reference.** Before the AEAD and UDP changes, the raw demo ended A′ in RESET (linear translator), and the anchor exchange sent about 5,300 spurious retransmits in each direction. The send window, the shared MLP and AEAD came after that run.
