# Telepathy-1 — model gate test

Status: specification only, not run yet. Saved for later testing.

Standalone. Do not touch the telepathy/ package, the ports or the experiments.
No training, inference only. Apple Silicon: use MPS, bf16 where it works, fall back
per op if it does not.

## Data
- Generate 200 synthetic biographies, 300–500 tokens each, with invented names,
  places, dates and numbers so nothing is answerable from pretraining.
- One short-answer question per passage with a single short answer (a name, city,
  year or number). Fixed seed; save the set to data/gate/.

## Conditions, same prompt for every model
- A fixed 3-shot prompt, identical across models and conditions; greedy decoding,
  max 8 new tokens, stop at newline.
- (1) full passage in context; (2) no passage; (3) only the sentence that holds the
  answer.
- Score exact match after normalisation (lowercase, strip punctuation and articles).
  Also score "contains answer" separately, so a format failure (right answer, kept
  generating) is distinguishable from a content failure.

## Candidates, base checkpoints only
- Qwen/Qwen3-0.6B-Base
- google/gemma-3-1b-pt
- google/gemma-3-270m
- HuggingFaceTB/SmolLM2-360M
Verify each repo exists on Hugging Face before loading; skip missing ones and say so.

## Pass rule
Full ≥ 0.80, none ≤ 0.10, sentence ≥ full − 0.05. Report pass/fail per model.

## Output
results/gate_test.json and results/gate_test.md: per model, EM and contains-answer
for all three conditions, pass/fail, tokens/s, peak memory, wall time.

## Done when
The table has a row for every candidate that exists on Hugging Face.
