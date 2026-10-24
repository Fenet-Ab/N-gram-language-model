"""Word <-> integer id mapping with sentence-boundary and unknown-word symbols."""

from array import array
from collections import Counter

import numpy as np

BOS, EOS, UNK = 0, 1, 2
SPECIALS = ["<s>", "</s>", "<unk>"]


class Vocab:
    def __init__(self, words):
        self.words = list(words)
        self.index = {w: i for i, w in enumerate(self.words)}

    @classmethod
    def from_counter(cls, counter, min_count=2):
        """Words with count >= min_count, most frequent first, after the special symbols."""
        kept = [w for w, c in counter.most_common() if c >= min_count and w not in SPECIALS]
        return cls(SPECIALS + kept)

    def __len__(self):
        return len(self.words)

    def id(self, word):
        return self.index.get(word, UNK)

    def encode_sentence(self, tokens):
        get = self.index.get
        return [BOS] + [get(w, UNK) for w in tokens] + [EOS]

    def encode(self, sentences):
        """Flat int32 array `<s> w1 .. wm </s> <s> ...` for an iterable of token lists."""
        buf = array("i")
        get = self.index.get
        for toks in sentences:
            buf.append(BOS)
            buf.extend(get(w, UNK) for w in toks)
            buf.append(EOS)
        return np.frombuffer(buf, dtype=np.int32).copy()

    def decode(self, ids):
        return [self.words[i] for i in ids]

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(self.words))

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            return cls(f.read().split("\n"))


def count_words(sentences):
    c = Counter()
    for toks in sentences:
        c.update(toks)
    return c
