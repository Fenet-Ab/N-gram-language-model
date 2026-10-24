from collections import Counter

import numpy as np
import pytest

from ngram.counts import NGramCounts
from ngram.models import KneserNeyModel, LaplaceModel, MLEModel
from ngram.normalize import normalize_chars, preprocess
from ngram.vocab import BOS, Vocab, count_words


def random_corpus(seed=0, n_sent=300, n_words=25):
    rng = np.random.default_rng(seed)
    words = [f"w{i}" for i in range(n_words)]
    # Zipf-like word frequencies so that counts of 1 and 2 both occur.
    p = 1 / np.arange(1, n_words + 1)
    p /= p.sum()
    return [list(rng.choice(words, size=rng.integers(1, 9), p=p)) for _ in range(n_sent)]


def brute_counts(ids, k):
    """Counter of k-grams that do not cross a sentence boundary."""
    out = Counter()
    sent = []
    for t in ids:
        if t == BOS and sent:
            out.update(tuple(sent[i:i + k]) for i in range(len(sent) - k + 1))
            sent = []
        sent.append(int(t))
    out.update(tuple(sent[i:i + k]) for i in range(len(sent) - k + 1))
    return out


@pytest.fixture(scope="module")
def setup():
    sents = random_corpus()
    vocab = Vocab.from_counter(count_words(sents), min_count=2)
    ids = vocab.encode(sents)
    counts = NGramCounts.build(ids, len(vocab), order=4)
    return sents, vocab, ids, counts


def test_counts_match_brute_force(setup):
    _, _, ids, counts = setup
    for k in range(1, 5):
        expected = brute_counts(ids, k)
        assert counts.num_grams(k) == len(expected)
        rows = np.arange(counts.num_grams(k))
        got = {tuple(g): int(c) for g, c in zip(counts.decode(k, rows).tolist(), counts.counts[k - 1])}
        assert got == dict(expected)


def test_meta_matches_brute_force(setup):
    _, _, ids, counts = setup
    for k in range(1, 5):
        c = brute_counts(ids, k)
        assert counts.meta["total"][k - 1] == sum(c.values())
        assert counts.meta["distinct"][k - 1] == len(c)
        assert counts.meta["n1"][k - 1] == sum(v == 1 for v in c.values())
        assert counts.meta["n2"][k - 1] == sum(v == 2 for v in c.values())


def test_partitioned_disk_build_matches_in_memory(setup, tmp_path):
    _, vocab, ids, counts = setup
    other = NGramCounts.build(ids, len(vocab), 4, workdir=str(tmp_path), mem_budget_mb=0.01, chunk=97)
    for k in range(1, 5):
        assert np.array_equal(other.keys[k - 1], counts.keys[k - 1])
        assert np.array_equal(other.counts[k - 1], counts.counts[k - 1])
        if k < 4:
            assert np.array_equal(other.cont[k - 1], counts.cont[k - 1])
    assert list(tmp_path.iterdir()) == []  # per-position row files are removed


def test_pruned_build_keeps_frequent_ngrams(setup):
    _, vocab, ids, _ = setup
    mins = [1, 1, 2, 2]
    pruned = NGramCounts.build(ids, len(vocab), 4, min_counts=mins, mem_budget_mb=0.01, chunk=97)
    for k in range(1, 5):
        expected = {g: c for g, c in brute_counts(ids, k).items() if c >= mins[k - 1]}
        rows = np.arange(pruned.num_grams(k))
        got = {tuple(g): int(c) for g, c in zip(pruned.decode(k, rows).tolist(), pruned.counts[k - 1])}
        assert got == expected
        full = brute_counts(ids, k)
        assert pruned.meta["distinct"][k - 1] == len(full)
        assert pruned.meta["n1"][k - 1] == sum(v == 1 for v in full.values())
        assert pruned.meta["total"][k - 1] == sum(full.values())


@pytest.mark.parametrize("order", [2, 3, 4])
def test_kneser_ney_normalizes_after_pruning(setup, order):
    sents, vocab, ids, _ = setup
    pruned = NGramCounts.build(ids, len(vocab), 4, min_counts=[1, 2, 2, 2])
    model = KneserNeyModel(pruned, order)
    all_words = np.arange(1, len(vocab))
    for toks in sents[:15]:
        ctx = vocab.encode_sentence(toks)[:-1]
        assert model.next_probs(ctx, all_words).sum() == pytest.approx(1.0)


def test_save_load_lower_order(setup, tmp_path):
    sents, vocab, _, counts = setup
    counts.save(tmp_path)
    loaded = NGramCounts.load(tmp_path, order=3)
    a, b = KneserNeyModel(loaded, 3), KneserNeyModel(counts, 3)
    ids = vocab.encode(sents[:30])
    assert np.allclose(a.token_probs(ids), b.token_probs(ids))


@pytest.mark.parametrize("mins", [[1, 1, 1, 1], [1, 2, 2, 2]])
def test_continuation_counts(setup, mins):
    _, vocab, ids, _ = setup
    counts = NGramCounts.build(ids, len(vocab), 4, min_counts=mins, mem_budget_mb=0.01, chunk=97)
    model = KneserNeyModel(counts, 4)
    for k in range(1, 4):
        upper = brute_counts(ids, k + 1)
        left = Counter(g[1:] for g in upper)
        rows = np.arange(counts.num_grams(k))
        for g, adj in zip(counts.decode(k, rows).tolist(), model.stats.counts_of("adj", k)):
            g = tuple(g)
            if g[0] == BOS:
                expected = 0 if k == 1 else brute_counts(ids, k)[g]
            else:
                expected = left[g]
            assert adj == expected, g


@pytest.mark.parametrize("order", [1, 2, 3, 4])
@pytest.mark.parametrize("cls", [KneserNeyModel, LaplaceModel])
def test_smoothed_models_normalize(setup, order, cls):
    sents, vocab, _, counts = setup
    model = cls(counts, order)
    all_words = np.arange(1, len(vocab))  # everything but <s>
    contexts = [[BOS], [BOS] + [vocab.id(w) for w in sents[0][:3]], [BOS, vocab.id("w1"), vocab.id("w1")],
                [BOS, 2, 2, 2]]  # unseen history of <unk>s
    for ctx in contexts:
        assert model.next_probs(ctx, all_words).sum() == pytest.approx(1.0)


def test_mle_matches_relative_frequency(setup):
    _, vocab, ids, counts = setup
    model = MLEModel(counts, 3)
    tri = brute_counts(ids, 3)
    hist_totals = Counter()
    for g, c in tri.items():
        hist_totals[g[:2]] += c
    for g in list(tri)[:50]:
        h, w = g[:2], g[2]
        ctx = list(h) if h[0] == BOS else [BOS, *h]
        assert model.next_probs(ctx, [w])[0] == pytest.approx(tri[g] / hist_totals[h])


def test_token_probs_agree_with_next_probs(setup):
    sents, vocab, _, counts = setup
    model = KneserNeyModel(counts, 4)
    for toks in sents[:20]:
        ids = vocab.encode_sentence(toks)
        batch = model.token_probs(ids)
        single = [model.next_probs(ids[:j], [ids[j]])[0] for j in range(1, len(ids))]
        assert np.allclose(batch, single)


def test_normalization_merges_homophones():
    assert normalize_chars("ሐገር") == normalize_chars("ሀገር") == "ሀገር"
    assert normalize_chars("ፀሐይ") == "ጸሀይ"
    assert normalize_chars("ዓለም") == "አለም"
    assert normalize_chars("ሠላም") == "ሰላም"


def test_split_numbers_and_abbreviations_are_rejoined():
    out = preprocess("በ1994 ዓ ም እ ኤ አ በ8 4 ቢ ዶላር ሸጠው፡፡ ቂ ቂ ቂ አለ፡፡")
    assert out[0] == ["በ<num>", "አም", "እኤአ", "በ<num>", "ቢ", "ዶላር", "ሸጠው"]
    assert out[1] == ["ቂ", "ቂ", "ቂ", "አለ"]  # runs that are not known abbreviations stay split
    assert preprocess("ዋጋው 1 000 000 ብር ነው።")[0] == ["ዋጋው", "<num>", "ብር", "ነው"]


def test_preprocess_sentences_and_numbers():
    out = preprocess("ኢትዮጵያ በአፍሪካ ቀንድ ትገኛለች። በ2015 ዓ.ም 3ኛ ሆነች፣ hello!")
    assert out[0] == ["ኢትዮጵያ", "በአፍሪካ", "ቀንድ", "ትገኛለች"]
    assert out[1][:2] == ["በ<num>", "አም"]
    assert "<num>ኛ" in out[1]
    assert "hello" not in out[1]
