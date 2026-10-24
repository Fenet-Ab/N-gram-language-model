"""MLE, Laplace and interpolated Kneser-Ney n-gram models over NGramCounts.

All three models score a batch of target words at once. Each target w comes with:
  hist_rows[k-1]  row of its k-word history in the order-k table (-1 if unseen), k = 1..n-1
  avail           number of history tokens available inside the sentence, capped at n-1
A model of order n conditions on the last min(avail, n-1) tokens. At sentence start the
history is just <s>, so the first word is predicted by a bigram even in a 4-gram model.
"""

import math

import numpy as np

from .counts import sentence_ids
from .vocab import BOS


class CountStats:
    """Per-order aggregates shared by every model built on the same NGramCounts, computed
    on first use so that a model touches only the orders it needs.

    Count kinds, per order-k row g:
      raw  occurrence count c(g)
      adj  Kneser-Ney continuation count N1+(• g), the number of distinct words seen right
           before g (NGramCounts.cont, counted before pruning). n-grams starting with <s>
           keep their raw count because nothing can precede <s>.
    The <s> unigram is zeroed in both: it is never predicted.

    level(kind, k) gives, per order-(k-1) history row h (scalars at k = 1):
      A(h)  total count of h as a history. For raw counts this is c(h •), which is exact
            even after pruning (see NGramCounts.history_totals); for adj it is the sum of
            the kept adj counts.
      S(h)  sum over kept w of max(a(h w) - D_k, 0), the discounted mass kept at order k.
    The interpolation weight of the lower order is then (A(h) - S(h)) / A(h): all mass
    removed by discounting or pruning moves to the lower order, so the model stays normalized.
    """

    def __init__(self, counts):
        self.counts = counts
        self._cache = {}

    def _memo(self, key, fn):
        if key not in self._cache:
            self._cache[key] = fn()
        return self._cache[key]

    def counts_of(self, kind, k):
        return self._memo((kind, k), lambda: self._raw(k) if kind == "raw" else self._adj(k))

    def _raw(self, k):
        c = self.counts.counts[k - 1]
        if k == 1:
            c = np.array(c)
            c[np.asarray(self.counts.keys[0]) == BOS] = 0
        return c

    def _adj(self, k):
        cnt = self.counts
        assert k <= len(cnt.cont), f"no continuation counts stored for order {k}"
        adj = np.array(cnt.cont[k - 1], dtype=np.int32)
        starts_bos = cnt.first_words(k) == BOS
        adj[starts_bos] = np.asarray(cnt.counts[k - 1])[starts_bos] if k > 1 else 0
        return adj

    def discount(self, kind, k):
        """D_k = n1 / (n1 + 2 n2) from the counts-of-counts n1, n2 of order k. For raw counts
        n1, n2 are recorded before pruning, so pruning does not change D_k."""
        def compute():
            if kind == "raw":
                n1, n2 = self.counts.meta["n1"][k - 1], self.counts.meta["n2"][k - 1]
            else:
                a = self.counts_of(kind, k)
                n1, n2 = int(np.count_nonzero(a == 1)), int(np.count_nonzero(a == 2))
            return n1 / (n1 + 2 * n2) if n1 + n2 > 0 else 0.5
        return self._memo(("D", kind, k), compute)

    def level(self, kind, k):
        def compute():
            a = self.counts_of(kind, k)
            disc = np.maximum(np.asarray(a, np.float64) - self.discount(kind, k), 0)
            if k == 1:
                return float(np.sum(a, dtype=np.int64)), float(disc.sum())
            h = np.asarray(self.counts.keys[k - 1]) // self.counts.V
            n_hist = self.counts.num_grams(k - 1)
            S = np.bincount(h, weights=disc, minlength=n_hist).astype(np.float32)
            if kind == "raw":
                A = self.counts.history_totals(k).astype(np.float32)
            else:
                A = np.bincount(h, weights=a, minlength=n_hist).astype(np.float32)
            return A, S
        return self._memo(("level", kind, k), compute)


def count_stats(counts):
    if getattr(counts, "_stats", None) is None:
        counts._stats = CountStats(counts)
    return counts._stats


class NGramModel:
    """Shared batching logic; subclasses implement `_probs`."""

    name = "base"

    def __init__(self, counts, order=None):
        self.counts = counts
        self.order = order or counts.order
        assert self.order <= counts.order
        self.stats = count_stats(counts)
        self.Vp = counts.V - 1  # predictable vocabulary: everything except <s>

    # ---- batch scoring ----------------------------------------------------------------

    def _probs(self, words, hist_rows, avail):
        raise NotImplementedError

    def _count_at(self, k, kind, words, hist):
        """For each word: a = count of (hist, w) at order k (0 if absent), and the history's
        A and S from CountStats.level. hist is ignored at k = 1."""
        c = self.counts
        tab = self.stats.counts_of(kind, k)
        A_all, S_all = self.stats.level(kind, k)
        g = c.find(1, words) if k == 1 else c.extend(k, hist, words)
        a = np.where(g >= 0, np.asarray(tab[np.maximum(g, 0)]), 0)
        if k == 1:
            return a, A_all, S_all
        have = hist >= 0
        hs = np.maximum(hist, 0)
        return a, np.where(have, A_all[hs], 0), np.where(have, S_all[hs], 0)

    def token_probs(self, ids):
        """P(ids[j] | history) for every position j with ids[j] != <s>.

        `ids` is an encoded corpus of `<s> ... </s>` sentences.
        """
        ids = np.asarray(ids, dtype=np.int64)
        targets = np.flatnonzero(ids != BOS)
        sid = sentence_ids(ids)
        starts = np.flatnonzero(ids == BOS)
        avail = np.minimum(targets - starts[sid[targets]], self.order - 1)
        rows = self.counts.positional_rows(ids, self.order - 1) if self.order > 1 else []
        hist_rows = []
        for k in range(1, self.order):
            src = targets - k
            r = np.full(len(targets), -1, np.int64)
            ok = src >= 0
            r[ok] = rows[k - 1][src[ok]]
            r[k > avail] = -1
            hist_rows.append(r)
        return self._probs(ids[targets], hist_rows, avail)

    def next_probs(self, context, words):
        """P(w | context) for each w in `words`; `context` is a list of ids starting with <s>."""
        words = np.asarray(words, dtype=np.int64)
        avail = min(len(context), self.order - 1)
        hist_rows = []
        for k in range(1, self.order):
            r = self.counts.row(context[-k:]) if k <= avail else -1
            hist_rows.append(np.full(len(words), r, np.int64))
        return self._probs(words, hist_rows, np.full(len(words), avail))

    # ---- evaluation helpers -----------------------------------------------------------

    def perplexity(self, ids):
        p = self.token_probs(ids)
        zeros = int(np.count_nonzero(p == 0))
        if zeros:
            return math.inf, zeros, len(p)
        return float(np.exp(-np.mean(np.log(p)))), 0, len(p)

    def sentence_report(self, vocab, tokens):
        """Per-token probabilities and the total probability of one tokenized sentence."""
        ids = vocab.encode_sentence(tokens)
        p = self.token_probs(ids)
        words = vocab.decode(ids[1:])
        logp = float(np.sum(np.log(p))) if np.all(p > 0) else -math.inf
        return list(zip(words, p.tolist())), logp


class MLEModel(NGramModel):
    """P(w | h) = c(h w) / c(h •); zero when h w is unseen or was pruned."""

    name = "mle"

    def _probs(self, words, hist_rows, avail):
        out = np.zeros(len(words))
        for L in np.unique(avail):
            sel = avail == L
            hist = hist_rows[L - 1][sel] if L > 0 else None
            a, A, _ = self._count_at(L + 1, "raw", words[sel], hist)
            out[sel] = np.divide(a, A, out=np.zeros(len(a)), where=np.asarray(A) > 0)
        return out


class LaplaceModel(NGramModel):
    """P(w | h) = (c(h w) + 1) / (c(h •) + V)."""

    name = "laplace"

    def _probs(self, words, hist_rows, avail):
        out = np.zeros(len(words))
        for L in np.unique(avail):
            sel = avail == L
            hist = hist_rows[L - 1][sel] if L > 0 else None
            a, A, _ = self._count_at(L + 1, "raw", words[sel], hist)
            out[sel] = (a + 1) / (A + self.Vp)
        return out


class KneserNeyModel(NGramModel):
    """Interpolated Kneser-Ney with one absolute discount D_k per order.

    P_k(w | h) = max(a(h w) - D_k, 0) / A(h) + (A(h) - S(h)) / A(h) * P_{k-1}(w | h')
    where a is the raw count at the model's highest order and the continuation count below
    it, A and S are defined in CountStats, and h' is h without its first word. Without
    pruning, A(h) - S(h) = D_k * #{w : a(h w) > 0}, the textbook interpolation weight.
    When A(h) = 0 the level passes P_{k-1} through. P_0 is uniform, 1 / V.
    """

    name = "kn"

    def _probs(self, words, hist_rows, avail):
        p = np.full(len(words), 1.0 / self.Vp)
        for k in range(1, self.order + 1):
            kind = "raw" if k == self.order else "adj"
            D = self.stats.discount(kind, k)
            hist = hist_rows[k - 2] if k > 1 else None
            a, A, S = self._count_at(k, kind, words, hist)
            A = np.broadcast_to(A, p.shape).astype(np.float64)
            safe = np.where(A > 0, A, 1.0)
            level = np.maximum(a - D, 0) / safe + (A - S) / safe * p
            p = np.where(A > 0, level, p)
        return p


MODELS = {"mle": MLEModel, "laplace": LaplaceModel, "kn": KneserNeyModel}


def make_model(counts, order, kind="kn"):
    return MODELS[kind](counts, order)
