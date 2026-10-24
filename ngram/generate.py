"""Random sentence generation and next-word suggestion."""

import bisect

import numpy as np

from .vocab import BOS, EOS, UNK


def generate(counts, vocab, order, max_len=30, rng=None, prefix=()):
    """Sample a sentence from the order-`order` MLE model.

    Each step samples w with probability c(h w) / c(h •) for the longest history h (up to
    order-1 words) that has continuations other than <unk>; <unk> is never sampled.
    Stops at </s> or max_len.
    """
    rng = rng or np.random.default_rng()
    ctx = [BOS] + [vocab.id(w) for w in prefix]
    out = list(prefix)
    while len(out) < max_len:
        words = cnts = None
        for k in range(min(order - 1, len(ctx)), -1, -1):
            if k == 0:
                words, cnts = counts.keys[0], counts.counts[0]
            else:
                r = counts.row(ctx[-k:])
                if r < 0:
                    continue
                words, cnts = counts.followers(k + 1, r)
            keep = (words != UNK) & (words != BOS)
            if keep.any():
                words, cnts = words[keep], cnts[keep]
                break
        w = int(rng.choice(words, p=cnts / cnts.sum()))
        if w == EOS:
            break
        out.append(vocab.words[w])
        ctx.append(w)
    return " ".join(out)


class PrefixIndex:
    """Vocabulary sorted by spelling, so the ids of all words starting with a prefix
    form one contiguous slice found by two binary searches."""

    def __init__(self, vocab):
        order = sorted(range(len(vocab)), key=vocab.words.__getitem__)
        self.sorted_words = [vocab.words[i] for i in order]
        self.ids = np.asarray(order, dtype=np.int64)

    def ids_with_prefix(self, prefix):
        lo = bisect.bisect_left(self.sorted_words, prefix)
        hi = bisect.bisect_left(self.sorted_words, prefix + "\U0010ffff")
        return self.ids[lo:hi]


class Suggester:
    """Top-k next-word suggestions from a language model.

    Candidates are the words seen after each suffix of the context in training plus the
    `pool` highest-count unigrams, so a suggestion exists even for an unseen context. When
    the user has typed part of the next word (`partial`), candidates are restricted to
    vocabulary words with that prefix.
    """

    def __init__(self, model, vocab, pool=200, exclude=()):
        self.model, self.vocab = model, vocab
        counts = model.counts
        self.frequent = counts.keys[0][np.argsort(-counts.counts[0])[:pool]]
        self.prefix_index = PrefixIndex(vocab)
        self.banned = np.asarray(sorted({BOS, EOS, UNK} | {vocab.id(w) for w in exclude}))

    def candidates(self, ctx, partial=""):
        counts = self.model.counts
        if partial:
            cands = self.prefix_index.ids_with_prefix(partial)
        else:
            parts = [self.frequent]
            for h in range(min(self.model.order - 1, len(ctx)), 0, -1):
                r = counts.row(ctx[-h:])
                if r >= 0:
                    parts.append(counts.followers(h + 1, r)[0])
            cands = np.unique(np.concatenate(parts))
        return cands[~np.isin(cands, self.banned)]

    def suggest(self, context_tokens, k=5, partial=""):
        """[(word, probability), ...] for the k most probable next words."""
        ctx = [BOS] + [self.vocab.id(w) for w in context_tokens]
        cands = self.candidates(ctx, partial)
        if len(cands) == 0:
            return []
        p = self.model.next_probs(ctx, cands)
        top = np.argsort(-p, kind="stable")[:k]
        return [(self.vocab.words[cands[i]], float(p[i])) for i in top]


def suggest(model, vocab, context_tokens, k=5, partial=""):
    """One-off suggestion call; build a Suggester once when calling repeatedly."""
    return Suggester(model, vocab).suggest(context_tokens, k, partial)
