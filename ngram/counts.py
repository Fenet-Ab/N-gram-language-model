"""N-gram count tables stored as sorted integer arrays, built in bounded memory.

Order k holds the distinct k-grams kept after pruning. The k-gram (h, w), with h its
(k-1)-word history and w its last word id, is stored as the int64 key

    key = row_{k-1}(h) * V + w

where row_{k-1}(h) is the index of h in the order-(k-1) table and V the vocabulary size.
Order 1 keys are the word ids themselves. Keys are sorted, so lookups are binary searches
and all continuations of h occupy the contiguous key range [row(h) * V, (row(h) + 1) * V).

Pruning drops k-grams seen fewer than min_counts[k-1] times. A k-gram occurs at most as
often as its prefix and its suffix, so with non-decreasing min_counts every kept k-gram's
prefix and suffix are kept too, and the key encoding above stays valid.
"""

import json
import os
from pathlib import Path

import numpy as np

from .vocab import BOS, EOS

# Per-position row codes used while building, besides real rows (>= 0):
_INVALID = -1  # no k-gram starts here inside one sentence
_PRUNED = -2   # a k-gram starts here inside one sentence but was pruned


def sentence_ids(ids):
    """Index of the sentence each position belongs to; sentences start at BOS."""
    return np.cumsum(ids == BOS, dtype=np.int64) - 1


class NGramCounts:
    """Counts for orders 1..order over an encoded corpus.

    Per order k (1-based, index k-1 in the lists):
      keys[k-1]    sorted int64 keys, one per kept k-gram
      counts[k-1]  int32 occurrence counts c(g)
      cont[k-1]    (k < order) int32 continuation counts N1+(• g): the number of distinct
                   words seen immediately before g, counted over all occurrences before
                   pruning. Used by Kneser-Ney for lower orders.
    meta holds, per order and computed before pruning: "total" (k-gram occurrences),
    "distinct" (distinct k-grams), "n1"/"n2" (k-grams seen exactly once / twice), plus
    "min_counts".
    """

    def __init__(self, order, V, keys, counts, cont, meta):
        self.order, self.V = order, V
        self.keys, self.counts, self.cont, self.meta = keys, counts, cont, meta

    # ---- building ---------------------------------------------------------------------

    @classmethod
    def build(cls, ids, V, order, min_counts=None, workdir=None, mem_budget_mb=512,
              chunk=4_000_000, log=None):
        """Count all k-grams (k = 1..order) of `ids`, a 1-D array of `<s> ... </s>` sentences.

        n-grams never cross a sentence boundary. `ids` may be a memory-mapped array.

        Memory: each counting job runs in P key-range partitions, one pass over the corpus per
        partition, with P chosen so that one partition's distinct keys fit `mem_budget_mb`.
        Peak memory is that budget plus the kept tables. The row of the k-gram at every
        position (needed to form (k+1)-gram keys) is kept in a `workdir` file when given,
        otherwise in memory.

        Pruning thresholds must be 1 or 2. A k-gram whose (k-1)-word prefix was pruned at
        threshold 2 then occurs exactly once, so it is counted in meta without being
        enumerated, and the statistics stay exact.
        """
        min_counts = list(min_counts or [1] * order)
        assert len(min_counts) == order and min_counts == sorted(min_counts), \
            "min_counts must be non-decreasing so that prefixes and suffixes of kept n-grams are kept"
        assert set(min_counts) <= {1, 2}, "pruning statistics are exact only for thresholds 1 and 2"
        T = len(ids)
        meta = {"total": [], "distinct": [], "n1": [], "n2": [], "min_counts": min_counts}
        keys, counts, cont = [], [], []
        prev_rows = None
        for k in range(1, order + 1):
            m = max(T - k + 1, 0)

            def gram_keys(s, e, k=k, prev_rows=prev_rows):
                return _position_keys(ids, prev_rows, k, V, s, e)

            # Pass 1: count kept-prefix k-grams, partition by partition, and prune.
            kept_k, kept_c = [], []
            total = distinct = n1 = n2 = 0
            n_hist = len(keys[-1]) if k > 1 else 1
            for u, c in _partitioned_unique(gram_keys, m, V, n_hist, chunk, mem_budget_mb, log, f"{k}-grams"):
                total += int(c.sum())
                distinct += len(u)
                n1 += int(np.count_nonzero(c == 1))
                n2 += int(np.count_nonzero(c == 2))
                keep = c >= min_counts[k - 1]
                kept_k.append(u[keep])
                kept_c.append(c[keep].astype(np.int32))
                del u, c, keep
            # Partitions cover ascending key ranges, so their concatenation is sorted.
            keys.append(np.concatenate(kept_k))
            counts.append(np.concatenate(kept_c))
            del kept_k, kept_c

            # Pass 2: row of the kept k-gram at each position. Positions whose prefix was
            # pruned hold singleton k-grams (see docstring); add them to the statistics.
            rows = _alloc(workdir, f"rows{k}.tmp", m)
            pruned_prefix = 0
            for s in range(0, m, chunk):
                e = min(s + chunk, m)
                key = gram_keys(s, e)
                r = _lookup(keys[-1], key)
                rows[s:e] = np.where(key == _INVALID, _INVALID, np.where(r >= 0, r, _PRUNED))
                pruned_prefix += int(np.count_nonzero(key == _PRUNED))
            meta["total"].append(total + pruned_prefix)
            meta["distinct"].append(distinct + pruned_prefix)
            meta["n1"].append(n1 + pruned_prefix)
            meta["n2"].append(n2)

            if k < order:
                cont.append(_continuation_counts(ids, rows, V, len(keys[-1]), chunk, mem_budget_mb, log))
            _free(prev_rows)
            prev_rows = rows
        _free(prev_rows)
        return cls(order, V, keys, counts, cont, meta)

    # ---- lookup ---------------------------------------------------------------------

    def find(self, k, keys):
        """Rows of `keys` in the order-k table, -1 where absent. Vectorized."""
        return _lookup(self.keys[k - 1], keys)

    def extend(self, k, hist_rows, words):
        """Rows of (h, w) at order k given rows of h at order k-1; -1 propagates."""
        hist_rows = np.asarray(hist_rows, dtype=np.int64)
        keys = np.where(hist_rows >= 0, hist_rows * self.V + np.asarray(words, np.int64), -1)
        return self.find(k, keys)

    def row(self, gram):
        """Row of a single n-gram (sequence of word ids), or -1."""
        r = int(self.find(1, [gram[0]])[0])
        for k, w in enumerate(gram[1:], start=2):
            if r < 0:
                return -1
            r = int(self.extend(k, [r], [w])[0])
        return r

    def count(self, gram):
        r = self.row(gram)
        return int(self.counts[len(gram) - 1][r]) if r >= 0 else 0

    def positional_rows(self, ids, max_k=None):
        """For each k <= max_k, the row of the k-gram starting at each position of `ids`.

        rows[k-1][i] is -1 when the k-gram is not in the table or crosses a sentence boundary.
        """
        ids = np.asarray(ids, dtype=np.int64)
        sid = sentence_ids(ids)
        max_k = max_k or self.order
        rows = []
        prev = None
        for k in range(1, max_k + 1):
            m = max(len(ids) - k + 1, 0)
            if k == 1:
                r = self.find(1, ids)
            else:
                r = self.extend(k, prev[:m], ids[k - 1:k - 1 + m])
                r[sid[:m] != sid[k - 1:k - 1 + m]] = -1
            rows.append(r)
            prev = r
        return rows

    def followers(self, k, hist_row):
        """(word ids, counts) of every kept w after the order-(k-1) history row `hist_row`."""
        table = self.keys[k - 1]
        lo, hi = np.searchsorted(table, [hist_row * self.V, (hist_row + 1) * self.V])
        return np.asarray(table[lo:hi]) % self.V, np.asarray(self.counts[k - 1][lo:hi])

    def first_words(self, k):
        """First word id of every order-k row."""
        f = np.asarray(self.keys[0])
        for j in range(2, k + 1):
            f = f[self.keys[j - 1] // self.V]
        return f

    def last_words(self, k):
        return np.asarray(self.keys[k - 1]) % self.V

    def history_totals(self, k):
        """c(h •) for every order-(k-1) row h: the number of times h is followed by a token.

        Every occurrence of h is followed by a token of the same sentence unless h ends in
        </s>, so c(h •) = c(h) for those rows and 0 otherwise. This stays exact when
        order-k n-grams have been pruned.
        """
        c = np.asarray(self.counts[k - 2], dtype=np.int64)
        return np.where(self.last_words(k - 1) == EOS, 0, c)

    def decode(self, k, rows):
        """Word-id tuples (as a (len(rows), k) array) for rows of the order-k table."""
        rows = np.asarray(rows, dtype=np.int64)
        out = np.empty((len(rows), k), np.int64)
        for j in range(k, 0, -1):
            key = np.asarray(self.keys[j - 1][rows])
            if j == 1:
                out[:, 0] = key
            else:
                out[:, j - 1] = key % self.V
                rows = key // self.V
        return out

    def top(self, k, n=10, exclude=()):
        """Rows of the n most frequent k-grams, skipping k-grams containing an id in `exclude`."""
        c = self.counts[k - 1]
        bad = np.asarray(list(exclude), np.int64)
        want = 4 * n
        while True:
            want = min(want, len(c))
            cand = np.argpartition(-c, want - 1)[:want] if want < len(c) else np.arange(len(c))
            cand = cand[np.argsort(-c[cand], kind="stable")]
            ok = ~np.isin(self.decode(k, cand), bad).any(axis=1) if len(bad) else np.ones(len(cand), bool)
            if ok.sum() >= n or want == len(c):
                return cand[ok][:n]
            want *= 4

    def num_grams(self, k):
        return len(self.keys[k - 1])

    # ---- persistence ----------------------------------------------------------------

    def save(self, directory):
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        for k in range(1, self.order + 1):
            np.save(d / f"keys{k}.npy", self.keys[k - 1])
            np.save(d / f"counts{k}.npy", self.counts[k - 1])
            if k < self.order:
                np.save(d / f"cont{k}.npy", self.cont[k - 1])
        with open(d / "meta.json", "w") as f:
            json.dump({"order": self.order, "V": self.V, **self.meta}, f)

    @classmethod
    def load(cls, directory, order=None, mmap=True):
        """Load orders 1..order (default: all). With mmap the tables stay on disk and the
        operating system pages in only the parts that lookups touch."""
        d = Path(directory)
        with open(d / "meta.json") as f:
            meta = json.load(f)
        order = order or meta["order"]
        mode = "r" if mmap else None
        keys = [np.load(d / f"keys{k}.npy", mmap_mode=mode) for k in range(1, order + 1)]
        counts = [np.load(d / f"counts{k}.npy", mmap_mode=mode) for k in range(1, order + 1)]
        cont = [np.load(d / f"cont{k}.npy", mmap_mode=mode) for k in range(1, order)]
        per_order = {name: meta[name][:order] for name in ("total", "distinct", "n1", "n2", "min_counts")}
        return cls(order, meta["V"], keys, counts, cont, per_order)


# ---- helpers ------------------------------------------------------------------------


def _position_keys(ids, prev_rows, k, V, s, e):
    """Keys of the k-grams starting at positions s..e-1.

    For k >= 2 the key is _INVALID when the k-gram would cross a sentence boundary (the
    (k-1)-gram at i is invalid or the next token starts a new sentence) and _PRUNED when it
    lies inside one sentence but its (k-1)-word prefix was pruned.
    """
    last = np.asarray(ids[s + k - 1:e + k - 1], dtype=np.int64)
    if k == 1:
        return last
    prev = np.asarray(prev_rows[s:e], dtype=np.int64)
    in_sentence = (prev != _INVALID) & (last != BOS)
    return np.where(in_sentence, np.where(prev >= 0, prev * V + last, _PRUNED), _INVALID)


def _partitioned_unique(make_keys, m, V, n_hist, chunk, mem_budget_mb, log=None, what="keys"):
    """Yield (sorted unique keys, counts) for the non-negative keys produced by
    make_keys(s, e) over positions 0..m-1, one key range at a time.

    Keys have the form row * V + word with row in [0, n_hist). Partition p holds the rows
    [p * n_hist / P, (p + 1) * n_hist / P), so partitions come out in ascending key order and
    their concatenation is globally sorted. P is sized so that one partition's distinct keys,
    with the temporaries of merging them (about 96 bytes per key), fit `mem_budget_mb`.
    """
    budget = mem_budget_mb * 2**20
    P = max(1, min(n_hist, int(np.ceil(m * 64 / budget))))
    if log:
        log(f"counting {what}: {m:,} positions, {P} partition(s)")
    for p in range(P):
        lo, hi = (p * n_hist // P) * V, ((p + 1) * n_hist // P) * V
        parts_k, parts_c = [], []
        for s in range(0, m, chunk):
            key = make_keys(s, min(s + chunk, m))
            key = key[(key >= lo) & (key < hi)] if P > 1 else key[key >= 0]
            u, c = np.unique(key, return_counts=True)
            parts_k.append(u)
            parts_c.append(c)
            if sum(len(x) for x in parts_k) * 96 > budget:
                mk, mc = _merge(parts_k, parts_c)
                parts_k, parts_c = [mk], [mc]
        yield _merge(parts_k, parts_c) if parts_k else (np.empty(0, np.int64), np.empty(0, np.int64))


def _continuation_counts(ids, rows, V, n_rows, chunk, mem_budget_mb, log=None):
    """N1+(• g) for every kept order-k row g: the number of distinct words w with w g in the
    corpus. Each position j where a kept g starts (and g does not start with <s>) yields the
    pair (row of g, word at j - 1); the distinct pairs are counted per row."""
    m = len(rows)

    def pair_keys(s, e):
        s = max(s, 1)
        r = np.asarray(rows[s:e], dtype=np.int64)
        prev_word = np.asarray(ids[s - 1:e - 1], dtype=np.int64)
        ok = (r >= 0) & (np.asarray(ids[s:e]) != BOS)
        return np.where(ok, r * V + prev_word, -1)

    cont = np.zeros(n_rows, np.int32)
    for u, _ in _partitioned_unique(pair_keys, m, V, n_rows, chunk, mem_budget_mb, log, "continuations"):
        cont += np.bincount(u // V, minlength=n_rows).astype(np.int32)
    return cont


def _merge(parts_k, parts_c):
    """Sum counts of equal keys across several (sorted unique keys, counts) pairs."""
    if len(parts_k) == 1:
        return parts_k[0], parts_c[0]
    k = np.concatenate(parts_k)
    if len(k) == 0:
        return k, np.concatenate(parts_c)
    order = np.argsort(k, kind="stable")
    k = k[order]
    c = np.concatenate(parts_c)[order]
    del order
    starts = np.flatnonzero(np.r_[True, k[1:] != k[:-1]])
    return k[starts], np.add.reduceat(c, starts)


def _lookup(table, keys):
    keys = np.asarray(keys, dtype=np.int64)
    if len(table) == 0:
        return np.full(keys.shape, -1, np.int64)
    idx = np.minimum(np.searchsorted(table, keys), len(table) - 1)
    return np.where((np.asarray(table[idx]) == keys) & (keys >= 0), idx, -1)


def _alloc(workdir, name, n):
    if workdir is None:
        return np.empty(n, np.int32)
    return np.lib.format.open_memmap(os.path.join(workdir, name), mode="w+", dtype=np.int32, shape=(n,))


def _free(arr):
    if isinstance(arr, np.memmap):
        path = arr.filename
        del arr
        os.remove(path)
