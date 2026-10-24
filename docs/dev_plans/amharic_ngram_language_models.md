# Amharic n-gram language models (NLP Assignment 1)

## Goal

Build n-gram language models (n = 1..4) on the GPAC Amharic corpus and use them for
the four assignment tasks:

1. Corpus statistics, n-gram counts and probabilities, stopword analysis, word clouds,
   sentence probability, random generation.
2. Intrinsic evaluation: perplexity on held-out GPAC sentences.
3. Extrinsic evaluation: news-category classification, plus next-word keystroke savings.
4. A Streamlit next-word suggestion app.

Everything runs locally on the full corpus. The machine reports 5.2 GB of RAM with about
2 GB available. GPAC is a single 1.06 GB line of about 66M tokens, with `፡፡` as its
dominant sentence terminator.

## Layout

```
ngram/                 reusable package (imported by notebooks, scripts and the app)
  normalize.py         character normalization, sentence splitting, tokenization
  corpus.py            corpus preprocessing into train/dev/test, statistics, Vocab, encoding
  counts.py            NGramCounts: sorted-array n-gram tables (see "Count storage")
  models.py            MLE, Laplace and interpolated Kneser-Ney models
  generate.py          random sentence generation
  stopwords.py         curated Amharic stopword list
  viz.py               word clouds (with Ethiopic font download)
  evaluate.py          perplexity, keystroke savings, per-class LM classification
scripts/build_model.py CLI: preprocess a corpus file, train, and save a model
app/streamlit_app.py   Task 4 user interface
notebooks/             one notebook per task, Colab-ready
tests/                 pytest checks against brute-force counting on a tiny corpus
```

## Preprocessing

- **Character normalization.** Amharic has several letter series that are pronounced
  identically and used interchangeably in written text. We merge the ሐ and ኀ series
  into ሀ, ሠ into ሰ, ዐ into አ, and ፀ into ጸ, and also map ሃ to ሀ and ኣ/ዓ to አ. A
  homophone spelling variant otherwise becomes a separate vocabulary type with its own
  sparse counts. Notebook 1 reports vocabulary size before and after normalization.
- **Sentence splitting** on ። ፧ ? ! and the ASCII stand-in `::`.
- **Tokenization.** ፡ (the traditional word separator) becomes a space. Other Ethiopic
  and ASCII punctuation is removed. Digit runs (Arabic and Ge'ez numerals) become
  `<num>`. Tokens containing no Ethiopic letters are dropped and counted in the statistics.
- **Split.** 90/5/5 train/dev/test by sentence, assigned by a seeded random draw.
- **Vocabulary.** Training words with count ≥ 2. Every other word maps to `<unk>`, so
  the dev and test sets have a defined probability for unseen words.
- **Boundaries.** Each sentence becomes `<s> w1 … wm </s>` with a single `<s>`. The
  first word of a sentence is therefore predicted from the shorter history `<s>`, as in
  KenLM and SRILM. n-grams never cross sentence boundaries.

## Count storage

The obvious representation, a Python `dict` keyed by tuples of strings, costs roughly
200 bytes per entry. GPAC has tens of millions of distinct 3- and 4-grams, so it does
not fit in Colab's 12 GB.

Instead, words become integer ids w ∈ [0, V), where V is the vocabulary size. Each order
k has a sorted table of unique k-grams. The k-gram (h, w) — h the (k−1)-word history, w
the last word — is stored as one int64 key:

    key_k = idx_{k−1}(h) · V + w

Here idx_{k−1}(h) is the row of h in the order-(k−1) table. Every k-gram's prefix is
itself a (k−1)-gram inside the same sentence, so the row always exists.

This is a trie flattened into sorted arrays:
- Lookup is a `searchsorted` per order.
- All continuations of a history h occupy one contiguous key range
  [idx(h)·V, (idx(h)+1)·V). The suggester and the generator read candidates straight from
  that range.
- Storage is 8 bytes of key plus 4 bytes of count per entry.

## Bounded-memory build and pruning

Full GPAC has roughly 18M / 40M / 50M distinct 2- / 3- / 4-grams. The complete tables
alone would take about 1.8 GB, so the build streams from disk and prunes.

- **Streaming.** Preprocessing reads the one-line corpus in 1 MB pieces, each cut at a
  sentence terminator. Encoded ids are written straight into a memory-mapped int32 `.npy`.
- **Range-partitioned counting.** For order k, each position's key is formed from the
  order-(k−1) row stored at that position (an int32 file on disk) and the next word. The
  keys are split into P partitions by ranges of the prefix row, so partitions come out in
  ascending key order and need no global sort. Each pass over the corpus
  counts one partition with `np.unique`, so peak memory is about `mem_budget_mb` (512 MB)
  plus the kept tables.
- **Pruning.** All unigrams and bigrams are kept; 3- and 4-grams seen once are dropped.
  On a 6M-token sample, 93% of 3-grams were singletons. A k-gram occurs at most as often as
  its prefix and its suffix, so with non-decreasing thresholds every kept k-gram's prefix
  and suffix are kept, and the key encoding stays valid.
- **Exact statistics.** Thresholds are restricted to 1 or 2. Any k-gram whose prefix was
  pruned at threshold 2 then occurs exactly once, so positions with a pruned prefix are
  added to the totals, distinct counts and n1 without enumerating them. The test suite
  checks these against brute-force counts.
- **Continuation counts** N1+(• g), the number of distinct words seen immediately before
  g, are counted as distinct (row of g, preceding word) pairs over all positions, also
  range-partitioned. They are therefore exact under pruning; KenLM likewise computes them
  before pruning.
- **Loading.** Tables are saved as `.npy` files and opened with `mmap_mode="r"`. Lookups
  touch only a few pages, and the app loads orders 1–3 from the same directory.

Measured on full GPAC (77.5M tokens, 5.09M sentences), counting takes 6 minutes and
peaks at 1.13 GB of process memory (`RssAnon`), plus 0.91 GB of file pages the OS can
reclaim. The table output is identical to that of a hash-partitioned build.

## Models

Symbols used below:

| symbol | meaning |
|---|---|
| c(h w) | training count of the n-gram h followed by w |
| c(h •) | Σ_w c(h w), the number of times h occurs as a history |
| N1+(h •) | number of distinct w with c(h w) > 0 |
| N1+(• h w) | number of distinct words preceding h w (continuation count) |
| D_k | absolute discount at order k, D_k = n1 / (n1 + 2·n2), where n1 and n2 are the numbers of order-k n-grams seen exactly once and exactly twice |
| h′ | h with its first word dropped |

- **MLE.** P(w | h) = c(h w) / c(h •). Used for the top-10 lists, the 1.4 conditional
  probabilities and generation. On the 1.7 sentence it shows zero probabilities for
  unseen n-grams.
- **Laplace.** P(w | h) = (c(h w) + 1) / (c(h •) + V). This is the baseline that shows
  how add-one smoothing moves too much probability mass to unseen events.
- **Interpolated Kneser-Ney.** This is the main model for perplexity, classification
  and suggestion.
  - At the highest order it discounts the raw count:
    P(w | h) = max(c(h w) − D, 0) / c(h •) + γ(h) · P(w | h′).
  - γ(h) = 1 − Σ_kept w max(c(h w) − D, 0) / c(h •) is the leftover mass. Without
    pruning it equals the textbook D · N1+(h •) / c(h •). With pruning, it also carries
    the mass of the pruned n-grams, so the distribution still sums to 1.
  - c(h •) = c(h) unless h ends in `</s>`, so it is exact after pruning.
  - D at the highest order uses n1 and n2 counted before pruning.
  - Lower orders replace raw counts with continuation counts. An n-gram starting with
    `<s>` has no left context, so it keeps its raw count (KenLM's rule).
  - The unigram level interpolates with the uniform distribution 1/V.
  - If c(h •) = 0, the model uses P(w | h′) directly.

## Task 3: extrinsic evaluation

**Main result: news classification with one language model per class.** Data: the
Amharic News Text Classification dataset (about 50k articles in 6 categories), fetched by
the user.

For each category c, we train a Kneser-Ney model on that category's training articles,
sharing the GPAC vocabulary. A document d gets the label argmax_c log P(c) + log P(d | c).

To bring the GPAC models into the comparison, each class model is also interpolated with
the GPAC model:

    P_c(w | h) = λ · P_class(w | h) + (1 − λ) · P_GPAC(w | h)

λ is chosen on a dev split. We report accuracy, macro-F1 and confusion matrices for
n = 1..4, with and without GPAC interpolation.

**Secondary result: keystroke savings.** The GPAC models suggest the top-5 next words on
held-out test sentences. We report top-1 and top-5 accuracy and the fraction of
characters saved by accepting correct suggestions. This uses the same code path as the
Task 4 app.

## Alternatives considered

| choice | alternative | pros | cons | decision |
|---|---|---|---|---|
| Sorted-array trie | `dict[tuple, int]` / `nltk.lm` | simplest code; nltk has smoothing built in | 10–20× more memory; nltk.lm is very slow at GPAC scale | sorted arrays |
| Sorted-array trie | KenLM | fastest; industry standard | C++ build on Colab; hides the counting and smoothing the assignment grades | sorted arrays |
| Kneser-Ney | Stupid backoff | trivial to implement | not a normalized distribution, so perplexity is meaningless | KN (stupid backoff unnecessary) |
| Kneser-Ney | Good-Turing / Katz | classic textbook method | more fiddly; KN is the stronger baseline | KN, with Laplace as contrast |
| Homophone normalization | raw spelling | preserves the corpus's original spelling | inflates the vocabulary and splits counts across spelling variants | normalize, report both vocab sizes |
| Classification (Task 3) | keystroke savings only | no extra data | overlaps with Task 4 | both; classification is the headline |
| Classification (Task 3) | synthetic spelling-error detection | no extra data | artificial task | not chosen |
| Full corpus, prune 3-/4-gram singletons | 20–25% subset, exact counts | subset: simpler, no pruning approximation | uses a quarter of the data | full + pruning |
| Pair-based continuation counts | counts from kept (k+1)-gram suffixes | no extra pass | with 93% of 3-grams pruned, continuation counts collapse and distort lower KN orders | pair-based |
| Single `<s>` | (n−1) `<s>` padding | textbook (Jurafsky & Martin) presentation | spurious `<s> <s>` lower-order counts | single `<s>` |
