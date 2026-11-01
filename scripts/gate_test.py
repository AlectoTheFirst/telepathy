#!/usr/bin/env python
"""Telepathy-1 model gate test (spec: docs/telepathy1_gate_test.md).

Standalone: does not import the telepathy package. Inference only.

Usage:
  python scripts/gate_test.py data     # generate data/gate/ (offline, fixed seed)
  python scripts/gate_test.py check    # verify repos exist and weights are downloadable
  python scripts/gate_test.py run      # check, then evaluate (aborts if any weights blocked)
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import random
import re
import resource
import string
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "gate"
RESULTS_DIR = ROOT / "results"
SEED = 1234
N_ITEMS = 200
MAX_NEW_TOKENS = 8
RUN_DATE = "2026-11-01"

CANDIDATES = [
    "Qwen/Qwen3-0.6B-Base",
    "google/gemma-3-1b-pt",
    "google/gemma-3-270m",
    "HuggingFaceTB/SmolLM2-360M",
]
CONDITIONS = ["full", "none", "sentence"]
TOK_LO, TOK_HI = 300, 500   # spec: passages of 300-500 tokens, checked per tokenizer

# ----------------------------------------------------------------------------- data

_ONSETS = ["b", "br", "d", "dr", "f", "g", "gr", "h", "k", "kl", "l", "m", "n", "p",
           "qu", "r", "s", "st", "t", "th", "tr", "v", "z", "zh", "y", "w", "c", "j"]
_VOWELS = ["a", "e", "i", "o", "u", "ae", "ei", "ou", "ia", "y"]
_CODAS = ["", "", "n", "r", "l", "s", "th", "m", "x", "nd", "rk", "v"]
_CITY_SUFFIX = ["haven", "mere", "ford", "vale", "port", "holm", "stad", "wick", "burg",
                "dor", "grad", "moor", "fell", "by", "ton"]


def _syll(rng):
    return rng.choice(_ONSETS) + rng.choice(_VOWELS) + rng.choice(_CODAS)


def _word(rng, n_min=2, n_max=3):
    return "".join(_syll(rng) for _ in range(rng.randint(n_min, n_max))).capitalize()


class _Pool:
    """Draws values guaranteed unique within one passage (no substring collisions)."""

    def __init__(self, rng):
        self.rng = rng
        self.used: list[str] = []

    def _ok(self, v):
        lv = v.lower()
        return all(lv not in u.lower() and u.lower() not in lv for u in self.used)

    def take(self, fn):
        for _ in range(1000):
            v = fn()
            if self._ok(v):
                self.used.append(v)
                return v
        raise RuntimeError("pool exhausted")

    def person(self):
        return self.take(lambda: f"{_word(self.rng, 1, 2)} {_word(self.rng, 2, 3)}")

    def first(self):
        return self.take(lambda: _word(self.rng, 1, 2))

    def city(self):
        return self.take(lambda: _word(self.rng, 1, 2) + self.rng.choice(_CITY_SUFFIX))

    def year(self, lo=1890, hi=1990):
        return self.take(lambda: str(self.rng.randint(lo, hi)))

    def number(self, lo, hi):
        return self.take(lambda: str(self.rng.randint(lo, hi)))


_FIELDS = ["hydrology", "glass chemistry", "cartography", "textile engineering",
           "orchard genetics", "acoustics", "bridge design", "seismology", "ceramics",
           "clockmaking", "marine biology", "printing", "botany", "metallurgy"]

_FILLER = [
    "Colleagues described the work as patient, careful and often ahead of its time.",
    "The early years were marked by long hours in borrowed rooms and constant shortages of equipment.",
    "Letters from that period show a steady interest in practical problems rather than theory.",
    "Local newspapers covered several of the projects, though rarely on the front page.",
    "Students remembered the lectures for their clear diagrams and dry humour.",
    "Much of the archive was later donated to a regional library, where it remains catalogued.",
    "Friends noted a lifelong habit of keeping detailed notebooks about every experiment.",
    "Funding was uncertain for many years and depended on small private grants.",
    "Several of the original instruments are still on display in a modest museum.",
    "The family rarely spoke publicly about this period, and few photographs survive.",
    "Critics at the time were divided, but later reviews were considerably kinder.",
    "A small circle of apprentices carried the methods forward after the workshop closed.",
]


def make_bio(rng, count=None):
    """count: None sizes the passage in characters (the shared set); a function str -> int sizes it
    in the tokens of one tokenizer, within TOK_LO..TOK_HI."""
    p = _Pool(rng)
    name = p.person()
    first, last = name.split(" ")
    f = {
        "birth_city": p.city(), "birth_year": p.year(1900, 1960), "mentor": p.person(),
        "university_town": p.city(), "workshop_city": p.city(), "company": _word(rng, 2, 3),
        "patents": p.number(11, 98), "award": _word(rng, 2, 2) + " Medal",
        "award_year": p.year(1965, 2010), "dog": p.first(), "books": p.number(3, 29),
        "spouse": p.person(), "retire_city": p.city(), "children": p.number(2, 9),
        "rival": p.person(), "sibling": p.first(), "visit_city": p.city(),
        "grad_year": p.year(1920, 1985), "staff": p.number(120, 990),
        "river": _word(rng, 2, 2), "field": rng.choice(_FIELDS),
    }
    p.used += [f["company"], f["award"], f["river"]]
    S = {
        "birth_city": f"{name} was born in the town of {f['birth_city']}, on the banks of the {f['river']} river.",
        "birth_year": f"The birth registry of the district records the year as {f['birth_year']}.",
        "sibling": f"{first} grew up alongside a younger sibling named {f['sibling']}, who later became a schoolteacher.",
        "grad": f"After moving to {f['university_town']} for university, {first} completed a degree in {f['field']} in {f['grad_year']}.",
        "mentor": f"During those years {first} studied under the guidance of {f['mentor']}, a demanding but generous teacher.",
        "workshop_city": f"{first} opened a first independent workshop in {f['workshop_city']}, renting two rooms above a bakery.",
        "company": f"That workshop eventually grew into a firm called {f['company']}, which at its peak employed {f['staff']} people.",
        "patents": f"Over the course of a long career, {last} filed {f['patents']} patents, mostly for small but useful improvements.",
        "rival": f"A long professional rivalry with {f['rival']} produced several heated public debates.",
        "visit": f"A lecture tour that included a winter in {f['visit_city']} brought the work to a wider audience.",
        "award_year": f"In {f['award_year']}, {last} received the {f['award']} for contributions to {f['field']}.",
        "books": f"{last} also published {f['books']} books, ranging from technical manuals to a short memoir.",
        "spouse": f"{first} married {f['spouse']}, whom {first} had met at a conference on {f['field']}.",
        "children": f"The couple raised {f['children']} children in a narrow house near the old market.",
        "dog": f"Visitors often mentioned a loyal dog called {f['dog']} that slept beside the workbench.",
        "retire_city": f"In later life {last} retired to {f['retire_city']}, where a quiet garden replaced the workshop.",
    }
    Q = {
        "birth_city": (f"In which town was {name} born?", f["birth_city"], "birth_city"),
        "birth_year": (f"In what year was {name} born?", f["birth_year"], "birth_year"),
        "mentor": (f"Who was {name}'s teacher at university?", f["mentor"], "mentor"),
        "workshop_city": (f"In which city did {name} open a first workshop?", f["workshop_city"], "workshop_city"),
        "patents": (f"How many patents did {name} file?", f["patents"], "patents"),
        "award_year": (f"In what year did {name} receive the {f['award']}?", f["award_year"], "award_year"),
        "books": (f"How many books did {name} publish?", f["books"], "books"),
        "spouse": (f"Who did {name} marry?", f["spouse"], "spouse"),
        "children": (f"How many children did {name} raise?", f["children"], "children"),
        "dog": (f"What was the name of {name}'s dog?", f["dog"], "dog"),
        "retire_city": (f"To which town did {name} retire?", f["retire_city"], "retire_city"),
        "sibling": (f"What was the name of {name}'s younger sibling?", f["sibling"], "sibling"),
    }
    qkey = rng.choice(list(Q.keys()))
    question, answer, skey = Q[qkey]
    # keep a subset of fact sentences (always the answer's), in narrative order
    keep = set(rng.sample([k for k in S if k != skey], rng.randint(9, 11))) | {skey}
    out = [S[k] for k in S if k in keep]
    fillers = rng.sample(_FILLER, len(_FILLER))
    if count is None:
        # pad with filler to a character target (~350-480 tokens for typical BPE tokenizers)
        target_chars = rng.randint(1400, 1750)
        for fs in fillers:
            if len(" ".join(out)) + len(fs) + 1 > target_chars:
                break
            out.insert(rng.randrange(1, len(out) + 1), fs)
    else:
        # pad with filler to a token target of this tokenizer; skip a filler that would overshoot
        target = rng.randint(TOK_LO + 20, TOK_HI - 20)
        for fs in fillers:
            if count(" ".join(out + [fs])) > target:
                continue
            out.insert(rng.randrange(1, len(out) + 1), fs)
        n = count(" ".join(out))
        if not TOK_LO <= n <= TOK_HI:
            raise ValueError(f"passage of {n} tokens outside {TOK_LO}-{TOK_HI}")
    answer_sentence = S[skey]
    passage = " ".join(out)
    if not (passage.count(answer) == 1 and answer in answer_sentence):
        raise ValueError("answer is not unique in the passage")
    return {"name": name, "passage": passage, "question": question, "answer": answer,
            "answer_sentence": answer_sentence, "qtype": qkey, "n_words": len(passage.split()), "n_chars": len(passage)}


SHOTS = [
    {"passage": "Orlena Vasquith was born in Dremholt in 1921. She trained as a surveyor and later moved to Kelvaran, where she ran a small mapping office.",
     "question": "In which city did Orlena Vasquith run a mapping office?", "answer": "Kelvaran"},
    {"passage": "The engineer Tavik Morrow designed 34 footbridges along the Ulsa river. His closest collaborator was Hesper Quill, who handled the stonework.",
     "question": "How many footbridges did Tavik Morrow design?", "answer": "34"},
    {"passage": "Bryn Altamire composed music for village festivals. In 1958 she won the Corvane Prize, and she later taught harmony to Ilse Varnet.",
     "question": "Who did Bryn Altamire teach harmony to?", "answer": "Ilse Varnet"},
]
HEADER = "Answer each question with a short answer taken from the passage.\n\n"


def build_prompt(item, condition):
    shots = "".join(f"Passage: {s['passage']}\nQuestion: {s['question']}\nAnswer: {s['answer']}\n\n"
                    for s in SHOTS)
    ctx = {"full": item["passage"], "none": "", "sentence": item["answer_sentence"]}[condition]
    return HEADER + shots + f"Passage: {ctx}\nQuestion: {item['question']}\nAnswer:"


def make_items(count=None):
    rng = random.Random(SEED)
    items = []
    for i in range(N_ITEMS):
        while True:   # draw again (same rng stream) if a passage fails its checks
            try:
                it = make_bio(rng, count)
                break
            except ValueError:
                continue
        it["id"] = i
        items.append(it)
    return items


def write_items(items, path: Path, extra=None):
    path.mkdir(parents=True, exist_ok=True)
    with open(path / "items.jsonl", "w") as fh:
        for it in items:
            fh.write(json.dumps(it) + "\n")
    meta = {"seed": SEED, "n": N_ITEMS, "shots": SHOTS, "header": HEADER, **(extra or {}),
            "example_prompt_full": build_prompt(items[0], "full")}
    (path / "meta.json").write_text(json.dumps(meta, indent=2))


def cmd_data():
    items = make_items()
    write_items(items, DATA_DIR, {"sized_in": "characters"})
    w = [it["n_words"] for it in items]; c = [it["n_chars"] for it in items]
    print(f"wrote {len(items)} items to {DATA_DIR}; words {min(w)}-{max(w)}, chars {min(c)}-{max(c)}")


def load_items(path: Path = DATA_DIR):
    with open(path / "items.jsonl") as fh:
        return [json.loads(l) for l in fh]


def items_for(repo, tok):
    """The shared set if every passage is TOK_LO..TOK_HI tokens for this tokenizer. Otherwise a set
    sized in this tokenizer's tokens, same seed, in data/gate/<model>/ (made once, then reused)."""
    def count(text):
        return len(tok(text, add_special_tokens=False).input_ids)
    shared = load_items()
    n = [count(it["passage"]) for it in shared]
    outside = sum(not TOK_LO <= x <= TOK_HI for x in n)
    if not outside:
        return shared, str(DATA_DIR.relative_to(ROOT)), 0
    path = DATA_DIR / repo.replace("/", "__")
    meta = path / "meta.json"
    if not (meta.exists() and json.loads(meta.read_text()).get("sized_in") == f"tokens of {repo}"):
        write_items(make_items(count), path, {"sized_in": f"tokens of {repo}", "token_range": [TOK_LO, TOK_HI]})
    return load_items(path), str(path.relative_to(ROOT)), outside

# ----------------------------------------------------------------------------- scoring

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = str.maketrans("", "", string.punctuation)


def normalise(s):
    s = s.lower().translate(_PUNCT)
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def score(raw, answer):
    first_line = raw.split("\n")[0]
    na, pred = normalise(answer), normalise(first_line)
    em = pred == na
    contains = bool(na) and re.search(rf"(^| ){re.escape(na)}( |$)", normalise(raw)) is not None
    return em, contains

# ----------------------------------------------------------------------------- gate check


def cmd_check():
    """Returns (available, missing, blocked). blocked = [(repo, error)]."""
    from huggingface_hub import HfApi, get_hf_file_metadata, hf_hub_url
    try:
        from huggingface_hub.errors import RepositoryNotFoundError, GatedRepoError
    except ImportError:  # older layout
        from huggingface_hub.utils import RepositoryNotFoundError, GatedRepoError
    api = HfApi()
    available, missing, blocked = [], [], []
    for repo in CANDIDATES:
        try:
            info = api.model_info(repo)
        except GatedRepoError as e:  # exists but gated
            blocked.append((repo, f"{type(e).__name__}: {e}"))
            continue
        except RepositoryNotFoundError as e:
            # Unauthenticated access to a gated/private repo can also surface as 401 here.
            missing.append((repo, f"{type(e).__name__}: {e}"))
            continue
        files = [s.rfilename for s in (info.siblings or [])]
        weights = [f for f in files if f.endswith(".safetensors")] or \
                  [f for f in files if f.endswith(".bin") and "pytorch_model" in f]
        if not weights:
            blocked.append((repo, f"no weight files listed: {files}"))
            continue
        try:
            for f in ["config.json"] + weights:
                get_hf_file_metadata(hf_hub_url(repo, f))  # HEAD; raises on 401/403
            available.append(repo)
        except Exception as e:
            blocked.append((repo, f"{type(e).__name__}: {e}"))
    for r in available:
        print(f"OK       {r}")
    for r, e in missing:
        print(f"MISSING  {r}: {e}")
    for r, e in blocked:
        print(f"BLOCKED  {r}: {e}")
    return available, missing, blocked

# ----------------------------------------------------------------------------- eval


def _load(repo, torch):
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from packaging.version import Version
    kw = "dtype" if Version(transformers.__version__) >= Version("4.56") else "torch_dtype"
    tok = AutoTokenizer.from_pretrained(repo)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    notes = []
    for dt in (torch.bfloat16, torch.float32):
        try:
            model = AutoModelForCausalLM.from_pretrained(repo, **{kw: dt}).to(device).eval()
            ids = tok("Passage: test\nQuestion: ok?\nAnswer:", return_tensors="pt").input_ids.to(device)
            with torch.no_grad():
                logits = model(input_ids=ids).logits
            if not torch.isfinite(logits.float()).all():
                raise RuntimeError("non-finite logits")
            return tok, model, device, str(next(model.parameters()).dtype), notes
        except Exception as e:
            notes.append(f"{dt} failed on {device}: {type(e).__name__}: {e}")
    raise RuntimeError("; ".join(notes))


def _mem(torch, device):
    return torch.mps.driver_allocated_memory() if device == "mps" else 0


def _greedy(model, tok, prompt, device, torch, peak):
    ids = tok(prompt, return_tensors="pt").input_ids.to(device)
    n_prompt = ids.shape[1]
    out_ids, text = [], ""
    with torch.no_grad():
        out = model(input_ids=ids, use_cache=True)
        for _ in range(MAX_NEW_TOKENS):
            nxt = out.logits[:, -1, :].argmax(-1)
            t = int(nxt.item())
            if tok.eos_token_id is not None and t == tok.eos_token_id:
                break
            out_ids.append(t)
            text = tok.decode(out_ids, skip_special_tokens=True)
            if "\n" in text:
                break
            out = model(input_ids=nxt[:, None], past_key_values=out.past_key_values, use_cache=True)
    peak[0] = max(peak[0], _mem(torch, device))
    return text, n_prompt, len(out_ids)


def eval_model(repo, torch):
    t0 = time.time()
    tok, model, device, dtype, notes = _load(repo, torch)
    load_s = time.time() - t0
    items, data_path, shared_outside = items_for(repo, tok)
    peak = [_mem(torch, device)]
    res = {"repo": repo, "data": data_path, "shared_set_passages_outside_range": shared_outside,
           "n_items": len(items), "device": device, "dtype": dtype, "dtype_notes": notes,
           "load_s": round(load_s, 1), "conditions": {}, "samples": {}}
    gen_tok = prompt_tok = 0
    gen_time = 0.0
    passage_tokens = [len(tok(it["passage"], add_special_tokens=False).input_ids) for it in items]
    res["passage_tokens"] = {"min": min(passage_tokens), "max": max(passage_tokens),
                             "mean": round(sum(passage_tokens) / len(passage_tokens), 1)}
    for cond in CONDITIONS:
        em = ct = 0
        samples = []
        for it in items:
            ts = time.time()
            text, n_p, n_g = _greedy(model, tok, build_prompt(it, cond), device, torch, peak)
            gen_time += time.time() - ts
            prompt_tok += n_p; gen_tok += n_g
            e, c = score(text, it["answer"])
            em += e; ct += c
            if len(samples) < 5:
                samples.append({"id": it["id"], "answer": it["answer"], "output": text})
        res["conditions"][cond] = {"em": em / len(items), "contains": ct / len(items)}
        res["samples"][cond] = samples
        print(f"  {repo} {cond}: EM={em/len(items):.3f} contains={ct/len(items):.3f}", flush=True)
    c = res["conditions"]
    full, none, sent = c["full"]["em"], c["none"]["em"], c["sentence"]["em"]
    res["pass_checks"] = {"full>=0.80": full >= 0.80, "none<=0.10": none <= 0.10,
                          "sentence>=full-0.05": sent >= full - 0.05}
    res["pass"] = all(res["pass_checks"].values())
    res["gen_tokens_per_s"] = round(gen_tok / gen_time, 2)
    res["total_tokens_per_s"] = round((gen_tok + prompt_tok) / gen_time, 1)
    res["peak_mps_driver_mem_gb"] = round(peak[0] / 1e9, 2)
    res["wall_s"] = round(time.time() - t0, 1)
    del model
    if device == "mps":
        torch.mps.empty_cache()
    return res


def _env():
    import torch, transformers, huggingface_hub
    try:
        chip = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        mem = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)) / 2**30
    except Exception:
        chip, mem = platform.processor(), 0
    return {"date": RUN_DATE, "hardware": f"{chip}, {mem:.0f} GB unified memory",
            "os": f"macOS {platform.mac_ver()[0]}", "python": sys.version.split()[0],
            "torch": torch.__version__, "transformers": transformers.__version__,
            "huggingface_hub": huggingface_hub.__version__,
            "PYTORCH_ENABLE_MPS_FALLBACK": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK")}


def write_md(out):
    L = [f"# Telepathy-1 model gate test", "",
         f"Date: {out['env']['date']}. Spec: docs/telepathy1_gate_test.md. Data: {N_ITEMS} items, seed {SEED}, "
         f"passages of {TOK_LO}-{TOK_HI} tokens for each model's tokenizer (see Notes). "
         f"Greedy, max {MAX_NEW_TOKENS} new tokens, stop at newline.", "",
         f"Hardware: {out['env']['hardware']}, {out['env']['os']}. Python {out['env']['python']}, "
         f"torch {out['env']['torch']}, transformers {out['env']['transformers']}, "
         f"huggingface_hub {out['env']['huggingface_hub']}. "
         f"PYTORCH_ENABLE_MPS_FALLBACK={out['env']['PYTORCH_ENABLE_MPS_FALLBACK']}.", "",
         "Pass rule (on EM): full >= 0.80, none <= 0.10, sentence >= full - 0.05.", "",
         "| Model | Device / dtype | Full EM | Full contains | None EM | None contains | "
         "Sentence EM | Sentence contains | Pass | Gen tok/s | Peak mem (GB) | Wall (s) |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in out["models"]:
        c = r["conditions"]
        L.append(f"| {r['repo']} | {r['device']} / {r['dtype'].replace('torch.', '')} | "
                 f"{c['full']['em']:.3f} | {c['full']['contains']:.3f} | {c['none']['em']:.3f} | "
                 f"{c['none']['contains']:.3f} | {c['sentence']['em']:.3f} | {c['sentence']['contains']:.3f} | "
                 f"{'pass' if r['pass'] else 'fail'} | {r['gen_tokens_per_s']} | {r['peak_mps_driver_mem_gb']} | {r['wall_s']} |")
    L += ["", "Notes:",
          "- Peak memory is the maximum of torch.mps.driver_allocated_memory() sampled after each item "
          "(torch has no MPS peak counter). Process max RSS: "
          f"{out['process_max_rss_gb']} GB.",
          "- Gen tok/s = generated tokens / total per-item generation time (prefill included).",
          "- Passage token lengths per tokenizer: " + "; ".join(
              f"{r['repo']} {r['passage_tokens']['min']}-{r['passage_tokens']['max']}" for r in out["models"]) + "."]
    for r in out["models"]:
        if r["shared_set_passages_outside_range"]:
            L.append(f"- {r['repo']}: {r['shared_set_passages_outside_range']} passages of the shared set "
                     f"(data/gate/, sized in characters) were outside {TOK_LO}-{TOK_HI} tokens, so this model "
                     f"ran on {r['data']}/ (same seed, sized in its own tokens).")
    for r in out["models"]:
        if r["dtype_notes"]:
            L.append(f"- {r['repo']}: dtype fallback: {'; '.join(r['dtype_notes'])}")
    for repo, err in out["missing"]:
        L.append(f"- {repo}: skipped, repo not found on Hugging Face ({err.splitlines()[0]}).")
    (RESULTS_DIR / "gate_test.md").write_text("\n".join(L) + "\n")


def cmd_run():
    available, missing, blocked = cmd_check()
    if blocked:
        print("ABORT: weights not downloadable for: " + ", ".join(r for r, _ in blocked))
        sys.exit(2)
    import torch
    torch.manual_seed(SEED)
    out = {"env": _env(), "seed": SEED, "n_items": N_ITEMS, "missing": missing, "models": []}
    for repo in available:
        print(f"== {repo}", flush=True)
        out["models"].append(eval_model(repo, torch))
        out["process_max_rss_gb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, 2)
        (RESULTS_DIR / "gate_test.json").write_text(json.dumps(out, indent=2))
    write_md(out)
    print("wrote results/gate_test.json and results/gate_test.md")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["data", "check", "run"])
    a = ap.parse_args()
    {"data": cmd_data, "check": lambda: sys.exit(2 if cmd_check()[2] else 0), "run": cmd_run}[a.cmd]()
