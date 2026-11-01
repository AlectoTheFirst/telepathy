# Telepathy-1 model gate test

Date: 2026-11-01. Spec: docs/telepathy1_gate_test.md. Data: 200 items, seed 1234, passages of 300-500 tokens for each model's tokenizer (see Notes). Greedy, max 8 new tokens, stop at newline.

Hardware: Apple M1 Pro, 16 GB unified memory, macOS 27.2. Python 3.15.0, torch 2.15.0, transformers 5.22.1, huggingface_hub 2.4.0. PYTORCH_ENABLE_MPS_FALLBACK=1.

Pass rule (on EM): full >= 0.80, none <= 0.10, sentence >= full - 0.05.

| Model | Device / dtype | Full EM | Full contains | None EM | None contains | Sentence EM | Sentence contains | Pass | Gen tok/s | Peak mem (GB) | Wall (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen/Qwen3-0.6B-Base | mps / bfloat16 | 0.945 | 0.950 | 0.010 | 0.010 | 0.970 | 0.970 | pass | 13.7 | 3.56 | 255.5 |
| google/gemma-3-1b-pt | mps / bfloat16 | 0.895 | 0.895 | 0.000 | 0.000 | 0.985 | 0.985 | pass | 9.01 | 3.98 | 328.5 |
| google/gemma-3-270m | mps / bfloat16 | 0.665 | 0.665 | 0.005 | 0.005 | 0.980 | 0.980 | fail | 23.29 | 2.24 | 142.9 |
| HuggingFaceTB/SmolLM2-360M | mps / bfloat16 | 0.880 | 0.880 | 0.010 | 0.010 | 0.945 | 0.945 | pass | 20.59 | 1.22 | 160.3 |

Notes:
- Peak memory is the maximum of torch.mps.driver_allocated_memory() sampled after each item (torch has no MPS peak counter). Process max RSS: 1.21 GB.
- Gen tok/s = generated tokens / total per-item generation time (prefill included).
- Passage token lengths per tokenizer: Qwen/Qwen3-0.6B-Base 312-448; google/gemma-3-1b-pt 314-445; google/gemma-3-270m 314-445; HuggingFaceTB/SmolLM2-360M 312-455.
- Qwen/Qwen3-0.6B-Base: 9 passages of the shared set (data/gate/, sized in characters) were outside 300-500 tokens, so this model ran on data/gate/Qwen__Qwen3-0.6B-Base/ (same seed, sized in its own tokens).
- google/gemma-3-1b-pt: 17 passages of the shared set (data/gate/, sized in characters) were outside 300-500 tokens, so this model ran on data/gate/google__gemma-3-1b-pt/ (same seed, sized in its own tokens).
- google/gemma-3-270m: 17 passages of the shared set (data/gate/, sized in characters) were outside 300-500 tokens, so this model ran on data/gate/google__gemma-3-270m/ (same seed, sized in its own tokens).
- HuggingFaceTB/SmolLM2-360M: 8 passages of the shared set (data/gate/, sized in characters) were outside 300-500 tokens, so this model ran on data/gate/HuggingFaceTB__SmolLM2-360M/ (same seed, sized in its own tokens).
