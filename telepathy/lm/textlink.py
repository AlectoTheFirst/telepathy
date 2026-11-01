"""Text baselines for Telepathy-1 (brief section 8), read by the frozen receiver with the fixed
3-shot prompt of the gate test. No adaptation on the text side: the gate showed no format failures
(EM equals contains-answer on every model).

Budgets are real bits on the wire (payload bytes x 8):
- oracle:   the one sentence that holds the answer, cut to the budget (question-aware: the text
            ceiling for a budget, not the comparison);
- cut:      the passage cut to the budget (UTF-8 bytes, at a character boundary);
- zstd:     the longest passage prefix whose zstd frame (level 19) fits the budget, decompressed
            by the receiver. Question-unaware, as the latent push.
"""
from __future__ import annotations

try:
    from compression import zstd as _zstd

    def zc(b: bytes) -> bytes:
        return _zstd.compress(b, level=19)

    def zd(b: bytes) -> bytes:
        return _zstd.decompress(b)
except ImportError:   # pragma: no cover
    import zstandard as _zstd

    def zc(b: bytes) -> bytes:
        return _zstd.ZstdCompressor(level=19).compress(b)

    def zd(b: bytes) -> bytes:
        return _zstd.ZstdDecompressor().decompress(b)

SHOTS = [
    {"passage": "Orlena Vasquith was born in Dremholt in 1921. She trained as a surveyor and later moved to Kelvaran, where she ran a small mapping office.",
     "question": "In which city did Orlena Vasquith run a mapping office?", "answer": "Kelvaran"},
    {"passage": "The engineer Tavik Morrow designed 34 footbridges along the Ulsa river. His closest collaborator was Hesper Quill, who handled the stonework.",
     "question": "How many footbridges did Tavik Morrow design?", "answer": "34"},
    {"passage": "Bryn Altamire composed music for village festivals. In 1958 she won the Corvane Prize, and she later taught harmony to Ilse Varnet.",
     "question": "Who did Bryn Altamire teach harmony to?", "answer": "Ilse Varnet"},
]
HEADER = "Answer each question with a short answer taken from the passage.\n\n"


def text_prompt(context: str, question: str) -> str:
    shots = "".join(f"Passage: {s['passage']}\nQuestion: {s['question']}\nAnswer: {s['answer']}\n\n" for s in SHOTS)
    return HEADER + shots + f"Passage: {context}\nQuestion: {question}\nAnswer:"


def cut_bytes(text: str, nbytes: int) -> str:
    return text.encode()[:max(0, nbytes)].decode("utf-8", errors="ignore")


def zstd_prefix(text: str, nbytes: int):
    """Longest character prefix whose zstd frame is <= nbytes. Returns (frame, decoded text)."""
    lo, hi, best = 0, len(text), b""
    if len(zc(b"")) > nbytes:
        return b"", ""
    while lo <= hi:
        mid = (lo + hi) // 2
        f = zc(text[:mid].encode())
        if len(f) <= nbytes:
            best, lo = f, mid + 1
        else:
            hi = mid - 1
    return best, zd(best).decode() if best else ""


def payload(kind: str, item, budget_bits: int):
    """(wire bytes, receiver context text) for one item."""
    nb = budget_bits // 8
    if kind == "oracle":
        t = cut_bytes(item["sentence"], nb)
        return t.encode(), t
    if kind == "cut":
        t = cut_bytes(item["passage"], nb)
        return t.encode(), t
    if kind == "zstd":
        return zstd_prefix(item["passage"], nb)
    if kind == "none":
        return b"", ""
    raise ValueError(kind)
