"""Telepathy-1 shared dataset: synthetic biographies sized in Qwen3 tokens.

One set for every model, generated once with a fixed seed (brief section 5):
- passages of 300-500 tokens of the receiver's tokenizer (Qwen/Qwen3-0.6B-Base);
- 8-12 facts per passage, every fact answerable by a short span that occurs once in the passage;
- invented names, places, dates and numbers, so nothing is answerable from pretraining;
- splits train / val / test / anchors with disjoint entities (no subject name, and no person or
  place value, occurs in two splits).

Every passage carries all of its questions. An eval item is (passage, one question). The question
distribution is uniform over the passage's facts by default; `zipf` weights fact slot r (the rank of
the fact key in FACT_KEYS) by 1 / r**ZIPF_S, used only for G3.

Usage: python -m telepathy.lm.data   (writes data/t1/*.jsonl and data/t1/meta.json)
"""
from __future__ import annotations

import json
import os
import random
import re
import string
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "t1"
SEED = 20261101
TOKENIZER = "Qwen/Qwen3-0.6B-Base"
TOK_LO, TOK_HI = 300, 500
SPLITS = {"train": 24000, "val": 500, "test": 1000, "anchors": 288}
N_ANCHOR_FIT, N_ANCHOR_HELD = 256, 32
ZIPF_S = 1.1

_ONSETS = ["b", "br", "d", "dr", "f", "g", "gr", "h", "k", "kl", "l", "m", "n", "p",
           "qu", "r", "s", "st", "t", "th", "tr", "v", "z", "zh", "y", "w", "c", "j"]
_VOWELS = ["a", "e", "i", "o", "u", "ae", "ei", "ou", "ia", "y"]
_CODAS = ["", "", "n", "r", "l", "s", "th", "m", "x", "nd", "rk", "v"]
_CITY_SUFFIX = ["haven", "mere", "ford", "vale", "port", "holm", "stad", "wick", "burg",
                "dor", "grad", "moor", "fell", "by", "ton"]
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
    "Winters were spent indoors revising drafts and answering a large volume of correspondence.",
    "The workshop kept irregular hours, and lights were often seen in the windows after midnight.",
    "Few of the early prototypes survived, because most were taken apart to reuse their parts.",
    "Visitors were struck by how quietly the most important decisions were made.",
    "Some of the methods were criticised as too slow, although they rarely produced errors.",
    "In interviews the emphasis was always on the team rather than on any single person.",
    "A habit of walking long distances each morning was mentioned in almost every account.",
    "The surviving accounts disagree on several minor details of these years.",
    "Many of the tools used in the workshop were made by hand from salvaged materials.",
    "Public lectures drew small but loyal audiences, many of whom returned every season.",
    "There was little interest in fame, and several honorary titles were politely declined.",
    "Records from the period are incomplete, as a flood destroyed part of the town archive.",
]

# Fact keys in slot-rank order (rank 1 first; used by the Zipf question distribution).
FACT_KEYS = ["birth_city", "birth_year", "mentor", "workshop_city", "patents", "award_year",
             "books", "spouse", "children", "dog", "retire_city", "sibling", "rival", "visit_city",
             "staff", "grad_year", "name", "field"]
# SHARE_IDENTITY trains the send port on these questions only (brief section 6: the entity's name
# and occupation). The name question is asked through the firm; the occupation is the field.
IDENTITY_KEYS = ["name", "field"]
ALWAYS = ["name", "field", "staff", "grad_year"]   # staff names the firm; grad_year states the field
NON_UNIQUE = {"name", "field"}   # these answers occur more than once in a passage by construction


def _syll(rng):
    return rng.choice(_ONSETS) + rng.choice(_VOWELS) + rng.choice(_CODAS)


def _word(rng, n_min=2, n_max=3):
    return "".join(_syll(rng) for _ in range(rng.randint(n_min, n_max))).capitalize()


class _Pool:
    """Draws values unique within one passage (no substring collisions), and, with a global set,
    unique across splits for persons and places."""

    def __init__(self, rng, global_used: set):
        self.rng, self.g = rng, global_used
        self.used: list[str] = []

    def _ok(self, v, glob):
        lv = v.lower()
        if glob and lv in self.g:
            return False
        return all(lv not in u.lower() and u.lower() not in lv for u in self.used)

    def take(self, fn, glob=False):
        for _ in range(1000):
            v = fn()
            if self._ok(v, glob):
                self.used.append(v)
                return v
        raise ValueError("pool exhausted")

    def person(self):
        return self.take(lambda: f"{_word(self.rng, 1, 2)} {_word(self.rng, 2, 3)}", glob=True)

    def first(self):
        return self.take(lambda: _word(self.rng, 1, 2), glob=True)

    def city(self):
        return self.take(lambda: _word(self.rng, 1, 2) + self.rng.choice(_CITY_SUFFIX), glob=True)

    def year(self, lo, hi):
        return self.take(lambda: str(self.rng.randint(lo, hi)))

    def number(self, lo, hi):
        return self.take(lambda: str(self.rng.randint(lo, hi)))


def make_bio(rng, count, global_used: set):
    """One passage. count: str -> number of tokens of the receiver's tokenizer."""
    p = _Pool(rng, global_used)
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
        "grad_year": f"After moving to {f['university_town']} for university, {first} completed a degree in {f['field']} in {f['grad_year']}.",
        "mentor": f"During those years {first} studied under the guidance of {f['mentor']}, a demanding but generous teacher.",
        "workshop_city": f"{first} opened a first independent workshop in {f['workshop_city']}, renting two rooms above a bakery.",
        "staff": f"The workshop of {name} eventually grew into a firm called {f['company']}, which at its peak employed {f['staff']} people.",
        "patents": f"Over the course of a long career, {last} filed {f['patents']} patents, mostly for small but useful improvements.",
        "rival": f"A long professional rivalry with {f['rival']} produced several heated public debates.",
        "visit_city": f"A lecture tour that included a winter in {f['visit_city']} brought the work to a wider audience.",
        "award_year": f"In {f['award_year']}, {last} received the {f['award']} for contributions to {f['field']}.",
        "books": f"{last} also published {f['books']} books, ranging from technical manuals to a short memoir.",
        "spouse": f"{first} married {f['spouse']}, whom {first} had met at a conference on {f['field']}.",
        "children": f"The couple raised {f['children']} children in a narrow house near the old market.",
        "dog": f"Visitors often mentioned a loyal dog called {f['dog']} that slept beside the workbench.",
        "retire_city": f"In later life {last} retired to {f['retire_city']}, where a quiet garden replaced the workshop.",
    }
    Q = {
        "birth_city": f"In which town was {name} born?",
        "birth_year": f"In what year was {name} born?",
        "mentor": f"Who was {name}'s teacher at university?",
        "workshop_city": f"In which city did {name} open a first workshop?",
        "patents": f"How many patents did {name} file?",
        "award_year": f"In what year did {name} receive the {f['award']}?",
        "books": f"How many books did {name} publish?",
        "spouse": f"Who did {name} marry?",
        "children": f"How many children did {name} raise?",
        "dog": f"What was the name of {name}'s dog?",
        "retire_city": f"To which town did {name} retire?",
        "sibling": f"What was the name of {name}'s younger sibling?",
        "rival": f"Who was {name}'s long professional rival?",
        "visit_city": f"Where did {name} spend a winter on a lecture tour?",
        "staff": f"How many people did {name}'s firm employ at its peak?",
        "grad_year": f"In what year did {name} complete a university degree?",
        "name": f"What is the full name of the person who founded the firm {f['company']}?",
        "field": f"In which field did {name} complete a degree?",
    }
    S["name"], S["field"] = S["staff"], S["grad_year"]
    f["name"] = name
    # 8-12 facts, counted as questions; the identity facts and their sentences are always present
    n_facts = rng.randint(8, 12)
    others = [k for k in FACT_KEYS if k not in ALWAYS]
    keep = set(ALWAYS) | set(rng.sample(others, n_facts - len(ALWAYS)))
    order = [k for k in S if k not in NON_UNIQUE]   # narrative order
    out = [S[k] for k in order if k in keep]
    target = rng.randint(TOK_LO + 20, TOK_HI - 20)
    for fs in rng.sample(_FILLER, len(_FILLER)):
        if count(" ".join(out + [fs])) > target:
            continue
        out.insert(rng.randrange(1, len(out) + 1), fs)
    passage = " ".join(out)
    n_tok = count(passage)
    if not TOK_LO <= n_tok <= TOK_HI:
        raise ValueError(f"passage of {n_tok} tokens outside {TOK_LO}-{TOK_HI}")
    qs = []
    for k in FACT_KEYS:
        if k not in keep:
            continue
        a = f[k]
        if a not in S[k] or (k not in NON_UNIQUE and passage.count(a) != 1):
            raise ValueError(f"answer {a!r} for {k} is not unique in the passage")
        qs.append({"key": k, "rank": FACT_KEYS.index(k) + 1, "question": Q[k], "answer": a, "sentence": S[k]})
    for v in p.used:
        global_used.add(v.lower())
    return {"name": name, "passage": passage, "n_tokens": n_tok, "questions": qs}


def generate(count, sizes=SPLITS, seed=SEED):
    rng = random.Random(seed)
    used: set = set()
    out = {}
    for split, n in sizes.items():
        items = []
        while len(items) < n:
            try:
                it = make_bio(rng, count, used)
            except ValueError:
                continue
            it["id"] = f"{split}-{len(items)}"
            items.append(it)
        out[split] = items
    return out


# ----------------------------------------------------------------------------- question distributions

def question_weights(qs, dist: str):
    if dist == "uniform":
        return [1.0] * len(qs)
    if dist == "zipf":
        return [1.0 / q["rank"] ** ZIPF_S for q in qs]
    raise ValueError(dist)


def eval_items(passages, dist: str = "uniform", seed: int = 7):
    """The fixed eval set: one question per passage, drawn under `dist` with a fixed seed."""
    rng = random.Random(f"{seed}-{dist}")
    items = []
    for p in passages:
        qs = p["questions"]
        q = rng.choices(qs, weights=question_weights(qs, dist))[0]
        items.append({"pid": p["id"], "passage": p["passage"], **q})
    return items


# ----------------------------------------------------------------------------- scoring (as the gate test)

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = str.maketrans("", "", string.punctuation)


def normalise(s: str) -> str:
    s = s.lower().translate(_PUNCT)
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def exact_match(raw: str, answer: str) -> bool:
    return normalise(raw.split("\n")[0]) == normalise(answer)


# ----------------------------------------------------------------------------- io

def load(split: str, path: Path = DATA_DIR):
    with open(path / f"{split}.jsonl") as fh:
        return [json.loads(l) for l in fh]


def anchors(path: Path = DATA_DIR):
    a = load("anchors", path)
    return a[:N_ANCHOR_FIT], a[N_ANCHOR_FIT:N_ANCHOR_FIT + N_ANCHOR_HELD]


def main():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER)

    def count(t):
        return len(tok(t, add_special_tokens=False).input_ids)

    data = generate(count)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    stats = {}
    for split, items in data.items():
        with open(DATA_DIR / f"{split}.jsonl", "w") as fh:
            for it in items:
                fh.write(json.dumps(it) + "\n")
        nt = [it["n_tokens"] for it in items]
        nf = [len(it["questions"]) for it in items]
        stats[split] = {"passages": len(items), "questions": sum(nf), "tokens": [min(nt), max(nt)],
                        "facts": [min(nf), max(nf)]}
    meta = {"seed": SEED, "tokenizer": TOKENIZER, "token_range": [TOK_LO, TOK_HI], "splits": stats,
            "fact_keys": FACT_KEYS, "identity_keys": IDENTITY_KEYS, "zipf_s": ZIPF_S,
            "anchors": {"fit": N_ANCHOR_FIT, "held_out": N_ANCHOR_HELD}}
    (DATA_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
