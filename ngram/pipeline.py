"""End-to-end build of a GPAC model directory, and loading it back.

A model directory holds:
  train.txt dev.txt test.txt   preprocessed sentences (one per line, space-separated)
  stats.json                   corpus statistics from preprocessing
  vocab.txt                    vocabulary, one word per line, line number = id
  {split}.ids.npy              encoded splits (int32)
  counts/                      NGramCounts over train (one .npy per table + meta.json)

Every stage streams or memory-maps its data so that the full GPAC corpus (about 66 million
tokens) builds in roughly 1.5 GB of RAM.
"""

import json
import tempfile
from pathlib import Path

import numpy as np

from .corpus import SPLITS, prepare_corpus, read_sentences
from .counts import NGramCounts
from .vocab import BOS, EOS, Vocab, count_words

DEFAULT_MIN_COUNTS = (1, 1, 2, 2)  # keep every unigram and bigram; drop 3-/4-grams seen once


def build(raw_path, out_dir, order=4, min_count=2, min_counts=DEFAULT_MIN_COUNTS, max_chars=None,
          force=False, mem_budget_mb=512, log=print):
    """Run every missing stage (or all with force) and return load(out_dir, order).

    min_count is the vocabulary threshold; min_counts[k-1] the pruning threshold of order k.
    """
    out = Path(out_dir)
    if force or not (out / "train.txt").exists():
        log("preprocessing corpus ...")
        prepare_corpus(raw_path, out, max_chars=max_chars)
    with open(out / "stats.json", encoding="utf-8") as f:
        stats = json.load(f)
    if force or not (out / "vocab.txt").exists():
        log("building vocabulary ...")
        vocab = Vocab.from_counter(count_words(read_sentences(out / "train.txt")), min_count)
        vocab.save(out / "vocab.txt")
    vocab = Vocab.load(out / "vocab.txt")
    for split in SPLITS:
        path = out / f"{split}.ids.npy"
        if force or not path.exists():
            log(f"encoding {split} ...")
            n = stats["split_tokens"][split] + 2 * stats["split_sentences"][split]
            encode_to_npy(vocab, read_sentences(out / f"{split}.txt"), path, n)
    if force or not (out / "counts" / "meta.json").exists():
        log(f"counting 1..{order}-grams (min counts {tuple(min_counts[:order])}) ...")
        train = np.load(out / "train.ids.npy", mmap_mode="r")
        with tempfile.TemporaryDirectory(dir=out) as work:
            counts = NGramCounts.build(train, len(vocab), order, list(min_counts[:order]),
                                       workdir=work, mem_budget_mb=mem_budget_mb, log=log)
        counts.save(out / "counts")
        del counts
    log("done")
    return load(out, order)


def encode_to_npy(vocab, sentences, path, n_ids, flush_every=1 << 20):
    """Write `<s> w1 .. wm </s>` ids for every sentence into an int32 .npy of length n_ids
    without holding the encoded corpus in memory."""
    arr = np.lib.format.open_memmap(path, mode="w+", dtype=np.int32, shape=(n_ids,))
    get, unk = vocab.index.get, vocab.id("<unk>")
    buf, pos = [], 0
    for toks in sentences:
        buf.append(BOS)
        buf.extend(get(w, unk) for w in toks)
        buf.append(EOS)
        if len(buf) >= flush_every:
            arr[pos:pos + len(buf)] = buf
            pos += len(buf)
            buf = []
    arr[pos:pos + len(buf)] = buf
    pos += len(buf)
    assert pos == n_ids, f"expected {n_ids} ids, wrote {pos}"
    arr.flush()


def filter_ids(ids, drop, path, chunk=8_000_000):
    """Write `ids` without the tokens in `drop` to an int32 .npy at `path`, chunk by chunk,
    and return it memory-mapped. Words on either side of a removed token become adjacent."""
    drop = np.asarray(sorted(drop), dtype=np.int32)
    n = sum(int(np.count_nonzero(~np.isin(ids[s:s + chunk], drop))) for s in range(0, len(ids), chunk))
    out = np.lib.format.open_memmap(path, mode="w+", dtype=np.int32, shape=(n,))
    pos = 0
    for s in range(0, len(ids), chunk):
        part = np.asarray(ids[s:s + chunk])
        part = part[~np.isin(part, drop)]
        out[pos:pos + len(part)] = part
        pos += len(part)
    out.flush()
    del out
    return np.load(path, mmap_mode="r")


def load(out_dir, order=4):
    """(vocab, counts, {split: ids}, stats) from a model directory; arrays are memory-mapped."""
    out = Path(out_dir)
    vocab, counts = load_counts(out, order)
    ids = {s: np.load(out / f"{s}.ids.npy", mmap_mode="r") for s in SPLITS}
    with open(out / "stats.json", encoding="utf-8") as f:
        stats = json.load(f)
    return vocab, counts, ids, stats


def load_counts(model_dir, order):
    """(vocab, counts) only, for consumers that need no corpus splits (the app)."""
    d = Path(model_dir)
    return Vocab.load(d / "vocab.txt"), NGramCounts.load(d / "counts", order)
