"""Corpus preprocessing into train/dev/test files, plus corpus statistics."""

import io
import json
import random
import zipfile
from collections import Counter
from pathlib import Path

from .normalize import CHAR_MAP, preprocess

SPLITS = ("train", "dev", "test")


_TERMINATORS = ("።", "፡፡", "?", "!", "፧", "::", "\n")


def _chunks_at_sentence_ends(fh, chunk_chars):
    """Read a text stream in pieces of about chunk_chars characters, each cut just after the
    last sentence terminator so no sentence is split between pieces. GPAC is a single 1 GB
    line, so reading line by line would load the whole corpus as one string."""
    carry = ""
    while True:
        block = fh.read(chunk_chars)
        if not block:
            if carry.strip():
                yield carry
            return
        text = carry + block
        cut = max((i + len(t) for t in _TERMINATORS if (i := text.rfind(t)) >= 0), default=-1)
        if cut <= 0:
            cut = text.rfind(" ") + 1
        if cut <= 0:  # no break point at all; keep accumulating
            carry = text
            continue
        yield text[:cut]
        carry = text[cut:]


def iter_raw_text(path, chunk_chars=1 << 20):
    """Text of a file, of every *.txt file under a directory, or of every *.txt member of a
    .zip archive (sorted), as pieces ending at sentence boundaries."""
    path = Path(path)
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as zf:
            for name in sorted(n for n in zf.namelist() if n.endswith(".txt")):
                with zf.open(name) as fh:
                    yield from _chunks_at_sentence_ends(
                        io.TextIOWrapper(fh, encoding="utf-8", errors="replace"), chunk_chars)
        return
    files = sorted(path.rglob("*.txt")) if path.is_dir() else [path]
    for f in files:
        with open(f, encoding="utf-8", errors="replace") as fh:
            yield from _chunks_at_sentence_ends(fh, chunk_chars)


def read_sentences(path, limit=None):
    """Token lists from a preprocessed file (one space-separated sentence per line)."""
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            toks = line.split()
            if toks:
                yield toks


def prepare_corpus(raw_path, out_dir, dev_frac=0.05, test_frac=0.05, seed=13, max_chars=None):
    """Normalize, tokenize and split a raw corpus; write {train,dev,test}.txt and stats.json.

    max_chars stops after about that many raw characters (whole sentences only), for a
    quick run on part of the corpus. Statistics are collected on both the raw-spelling tokens and the normalized tokens so
    that the effect of homophone normalization on vocabulary size can be reported.
    Tokenization does not depend on normalization (the character map only rewrites
    Ethiopic letters into Ethiopic letters), so normalized types are derived from raw types.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    files = {s: open(out_dir / f"{s}.txt", "w", encoding="utf-8") for s in SPLITS}
    raw_types = Counter()
    stats = {"sentences": 0, "tokens": 0, "characters": 0, "dropped_tokens": 0,
             "split_sentences": dict.fromkeys(SPLITS, 0), "split_tokens": dict.fromkeys(SPLITS, 0)}
    try:
        for text in iter_raw_text(raw_path):
            if max_chars is not None and stats["characters"] >= max_chars:
                break
            stats["characters"] += len(text)
            for toks in preprocess(text, normalize=False, stats=stats):
                raw_types.update(toks)
                r = rng.random()
                split = "dev" if r < dev_frac else "test" if r < dev_frac + test_frac else "train"
                files[split].write(" ".join(toks).translate(CHAR_MAP) + "\n")
                stats["sentences"] += 1
                stats["tokens"] += len(toks)
                stats["split_sentences"][split] += 1
                stats["split_tokens"][split] += len(toks)
    finally:
        for f in files.values():
            f.close()

    norm_types = Counter()
    for w, c in raw_types.items():
        norm_types[w.translate(CHAR_MAP)] += c
    stats.update(_type_stats("raw", raw_types))
    stats.update(_type_stats("normalized", norm_types))
    stats["avg_sentence_length"] = stats["tokens"] / max(stats["sentences"], 1)
    stats["top_words"] = norm_types.most_common(50)
    with open(out_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=1)
    return stats


def _type_stats(prefix, counter):
    tokens = sum(counter.values())
    lengths = sum(len(w) * c for w, c in counter.items())
    return {
        f"{prefix}_types": len(counter),
        f"{prefix}_hapax": sum(1 for c in counter.values() if c == 1),
        f"{prefix}_type_token_ratio": len(counter) / max(tokens, 1),
        f"{prefix}_avg_word_length": lengths / max(tokens, 1),
    }
