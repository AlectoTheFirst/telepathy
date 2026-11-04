# Telepathy-1: status

Last update: 2026-11-01. Branch: `telepathy-1`. `main` stays at v0.1.1.

The work follows the Telepathy-1 build brief of 2026-11-01, which has four goals:

| Goal | Short name |
| --- | --- |
| G1 | Negotiation between strangers: Gemma 3 1B sends, Qwen3-0.6B answers |
| G2 | Latent against text at passage scale: receiver context tokens and prefill latency against the passage in context, and bits against a terse text fact list |
| G3 | Ordering after capacity: two-stage training |
| G4 | The probe against an adaptive sender |

The work stopped at a clean break, before the first port training. No port pair is trained, and no experiment of E1 to E7 has run. There are no results yet.

## What Telepathy-1 has to show, and what it cannot claim

Telepathy-0 was negative on the premise that a latent message beats text (RESULTS.md on `main`, summary items 1 and 2). Telepathy-1 is the test of that premise at passage scale. The brief's G2 wording, "latent beats fair text on latency, tokens and bits", is replaced above by the quantities a latent message can win and the baseline it must also face:

- **Receiver context tokens and prefill latency**, against the passage in context. This is the win that the compression literature (gist tokens, ICAE, xRAG) also reports, so the comparison must state the compression ratio at matched accuracy, not only a win against a truncated passage.
- **Bits**, against a terse text fact list written by the sender, not only against passage prefixes. A passage of 8 to 12 facts as a fact list is about 200 to 300 bytes and exact. That baseline is the analogue of the binary codec that beat the latent link in Telepathy-0. E1 does not have it yet.
- **Sender latency** only against a sender that generates a summary. Against a forwarded passage the sender does no model work, so latent cannot win there.

The G1 gate (0.90 exact match on the held-out anchors) is absolute. The Qwen-to-Qwen ceiling is not 1.0 here, so the gate must also be reported relative to the pair's own ceiling, or no negotiation can pass by construction.

## What is done

| Part | Files | State |
| --- | --- | --- |
| Shared dataset | `telepathy/lm/data.py`, `data/t1/` | Done. See "Dataset" below. |
| Frozen model wrapper | `telepathy/lm/models.py` | Done. The receive port goes in with a forward hook after one decoder layer. The LM head is computed only at the answer positions. |
| Ports | `telepathy/lm/ports.py` | Done, with changes after the first diagnostics. See "Training findings". |
| Training loop | `telepathy/lm/train.py` | Done. Stage 1 and stage 2, trust, hint, intents, QAT, resume from checkpoints. Not yet run to the end. |
| Sender state cache | `telepathy/lm/cache.py` | Done. |
| Translators | `telepathy/lm/translators.py` | Done: relative, procrustes, shared_mlp. |
| Negotiation | `telepathy/lm/negotiate.py` | Done. The gate score is the task accuracy on the held-out anchors, as in Telepathy-0. The gate is 0.90. |
| Text baselines | `telepathy/lm/textlink.py` | Done: none, oracle sentence, passage cut, passage zstd. |
| Intent probe | `telepathy/lm/probe.py` | Done. Not yet used. |
| Protocol additions (brief section 7) | `telepathy/resolver/`, `telepathy/tlp/`, `telepathy/sock/socket.py`, SPEC.md section 14 | Done: T record (message types 21 to 23), hub mode, slot batching (header byte 21). |
| Training script | `scripts/train_t1.py` | Done. |
| Experiment scripts | `telepathy/experiments/t1_e1_link.py`, `t1_e2_negotiation.py` | Written. Not run. |
| Tests | `tests/test_lm_ports.py`, `tests/test_t1_protocol.py` | 62 passed and 2 skipped, on UDP and on inproc. The 2 skipped tests are the raw-socket tests, which need root. No test loads a language model. |

## Dataset

- **Generator:** `python -m telepathy.lm.data`, seed 20261101. The generator is deterministic.
- **Passages:** synthetic biographies of 308 to 480 Qwen3 tokens, with 8 to 12 facts each and invented names, places and numbers.
- **Splits:** the splits have disjoint entities.

| Split | Passages | Questions | In git |
| --- | --- | --- | --- |
| train | 24000 | 239998 | No. The file is 99 MB. Run the generator to make it again. |
| val | 500 | 4982 | Yes |
| test | 1000 | 10017 | Yes |
| anchors | 288 (256 for fit, 32 held out) | 2910 | Yes |

- **SHARE_IDENTITY questions:** the full name, asked through the name of the firm, and the field of study.

## Training findings so far

These findings go into the RESULTS.md training notes.

1. **Memory.** The MPS allocator keeps blocks of each sequence length, and the machine went into swap. Training became 10 times slower. The fix is `torch.mps.empty_cache()` after each step. Other programs also used about 9 GB of the 16 GB. Close them before a training run.
2. **The slots carried no passage content.** For 400 steps, the validation loss with swapped slots was equal to the loss with the true slots. These tests found the cause:
   - **Receive side:** with slots made from the answer tokens, the receive port learned to read them in 50 steps (loss 3.87 against 4.84 with swapped slots). The receive side works.
   - **Send side:** this side learns slowly. A contrastive match of a passage to its own questions takes 300 steps to leave chance level. The receive port meanwhile learns to use the slots as a fixed memory.
3. **Changes made to the send port and the training:**
   - Position 0 is an attention sink. Its residual norm is about 60 times that of other tokens. The send port masks it out. For Qwen, which has no BOS token, `FrozenLM.states` puts an EOS token in front of the passage as the sink.
   - A learned bias over 16 relative-position bins. At the start, it makes slot i read its own part of the passage.
   - No residual from the learned queries in the first send port layer.
   - An auxiliary contrastive loss during stage 1, with weight 0.5. The slots of a passage must match that passage's questions in the batch. The heads are used for training only and are never on the wire.
   - Learning rate 1e-3 for the send port and 5e-4 for the receive port.
   - **These changes are not yet confirmed.** The first full run must show a gap between the swapped-slot and true-slot validation losses. The training log prints this gap every 250 steps as `diag nll true ... swapped ... gap ...`.
4. **The layer sweep is not run.** In 20-minute runs no port carried content yet, so a short sweep cannot rank the layers. The default layers are two-thirds depth for the sender and one-half depth for the receiver:
   - Qwen: sender layer 19, receiver layer 14.
   - Gemma: sender layer 17, receiver layer 13.

## Deviations from the brief so far

| Brief | Now | Reason |
| --- | --- | --- |
| Layer sweep for the sender and the receiver | Not run. Fixed at two-thirds and one-half depth. | Finding 4 |
| Training on answer-token likelihood | Answer-token likelihood plus an auxiliary contrastive loss in stage 1 | Finding 2 |
| No nested dropout in stage 1 | No nested dropout. Each slot is dropped independently with probability 0.1 so that the mask path of the receive port is trained. There is no order. | Loss handling needs the mask path. |
| Relative representations: "the receiver reads the similarity vector" | The receiver maps the similarity vector to its own slots with a ridge decoder. It fits the decoder on its own messages only, so the peers share only the anchor set. | The receive port reads slots, not similarity vectors. |
| E1 sweep of 9 (N, d) port pairs | Plan: 1 hour of training for each, not 3 hours. Not decided. | Time: 27 hours at 3 hours each |

## Local files that are not in git

| Path | Content | How to make it again |
| --- | --- | --- |
| `data/t1/train.jsonl` | Train split | `python -m telepathy.lm.data` |
| `cache/t1/qwen_L19/` | Qwen hidden states at layer 19 for the first 12000 train passages, 9.7 GB, complete | `python scripts/train_t1.py cache Qwen/Qwen3-0.6B-Base` |
| `checkpoints/t1/` | Empty | Training writes it. |

## Next steps, in order

1. Close other programs to free memory.
2. Train the first pair: `python scripts/train_t1.py one qwen_s1`. It takes up to 170 minutes and resumes from its checkpoints. Watch the `diag ... gap` lines. If the gap stays near 0 after about 2000 steps, stop the run and change the port design before you train more pairs.
3. Build the Gemma cache: `python scripts/train_t1.py cache google/gemma-3-1b-pt` (about 10.5 GB).
4. Train `qwen_s2`, the control, and `gemma_s1`.
5. E2, negotiation: `python -m telepathy.experiments.t1_e2_negotiation gemma_s1 qwen_s1`, then the control `qwen_s2 qwen_s1`. If G1 misses, do the control first and then one repeat with Qwen3-1.7B-Base as the receiver. Those weights must be downloaded first.
6. E1, then E3, E5, E6, E7 and E4. The scripts for E3 to E7 and the demo `scripts/demo_t1.py` are not written yet.

The estimate for the remaining compute is about 37 hours on the M1 Pro, if the port training works.
