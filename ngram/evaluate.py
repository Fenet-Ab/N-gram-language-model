"""Intrinsic (perplexity) and extrinsic (keystroke savings, classification) evaluation."""

import numpy as np

from .counts import NGramCounts
from .models import KneserNeyModel, make_model
from .normalize import preprocess
from .vocab import BOS, UNK

# ---- intrinsic -------------------------------------------------------------------------


def perplexity_table(counts, ids, orders=(1, 2, 3, 4), kinds=("mle", "laplace", "kn")):
    """Perplexity of each (order, smoothing) model on an encoded held-out corpus.

    Perplexity is exp(-mean log P(w_j | history)) over every token except <s>, including </s>.
    MLE assigns zero to unseen n-grams, which makes its perplexity infinite; the number of
    zero-probability tokens is reported instead.
    """
    ids = np.asarray(ids)
    targets = ids[ids != BOS]
    rows = []
    for kind in kinds:
        for n in orders:
            ppl, zeros, n_tok = make_model(counts, n, kind).perplexity(ids)
            rows.append({"model": kind, "n": n, "perplexity": ppl, "zero_prob_tokens": zeros,
                         "tokens": n_tok, "oov_rate": float(np.mean(targets == UNK))})
    return rows


# ---- extrinsic: keystroke savings ------------------------------------------------------


def keystroke_savings(suggester, sentences, k=5, max_prefix=3):
    """Simulate typing `sentences` with a top-k suggestion list.

    Before each character of a word (up to `max_prefix` characters typed) the user looks at
    the suggestions and, if the word is listed, selects it with one keystroke; selection
    also inserts the following space. Otherwise the user types the whole word and a space.
    A character is one Ethiopic syllable (fidel).

    Returns top-1 and top-k next-word accuracy (with nothing of the word typed yet) and the
    keystroke saving rate 1 - keystrokes_with / keystrokes_without.
    """
    with_s = without_s = top1 = topk = n = 0
    for toks in sentences:
        context = []
        for word in toks:
            without_s += len(word) + 1
            cost = len(word) + 1
            for i in range(min(max_prefix, len(word)) + 1):
                sugg = [w for w, _ in suggester.suggest(context, k, partial=word[:i])]
                if i == 0:
                    n += 1
                    top1 += bool(sugg) and sugg[0] == word
                    topk += word in sugg
                if word in sugg:
                    cost = i + 1
                    break
            with_s += cost
            context.append(word)
    return {"top1_accuracy": top1 / max(n, 1), f"top{k}_accuracy": topk / max(n, 1),
            "keystroke_savings": 1 - with_s / max(without_s, 1), "words": n}


# ---- extrinsic: classification with one language model per class -----------------------


def encode_documents(vocab, texts):
    """Preprocess and encode raw documents one at a time; each becomes an int32 array of
    `<s> ... </s>` sentences. Holding ids instead of token strings keeps a 50k-article
    dataset at tens of megabytes instead of more than a gigabyte."""
    return [vocab.encode(preprocess(t)) for t in texts]


class ClassLMClassifier:
    """Assign a document to argmax_c [ log P(c) + log P(doc | c) ].

    P(doc | c) is the product over tokens of a Kneser-Ney n-gram model trained on class c's
    training documents. With a background model (the GPAC model), each class model is
    interpolated token by token: P_c(w | h) = lam * P_class(w | h) + (1 - lam) * P_bg(w | h).
    lam = 1 uses the class models alone. Documents are encoded id arrays (encode_documents).

    Class models are trained and applied one at a time and then discarded, so only one
    class model is in memory: fit_score returns, for each evaluation set, the per-token
    probabilities under every class, from which `predict` decides for any lam.
    """

    def __init__(self, V, order, background=None):
        self.V, self.order, self.background = V, order, background

    def fit_score(self, docs, labels, eval_sets):
        """Train on (docs, labels); return {name: (P, bg, doc_of_token)} for each
        name -> document list in eval_sets. P is (n_tokens, n_classes)."""
        self.classes = sorted(set(labels))
        labels = np.asarray(labels)
        self.log_prior = np.log([np.mean(labels == c) for c in self.classes])
        flat = {}
        for name, ds in eval_sets.items():
            ids = np.concatenate(ds)
            doc = np.repeat(np.arange(len(ds)), [len(d) for d in ds])[ids != BOS]
            bg = self.background.token_probs(ids) if self.background is not None else None
            flat[name] = (ids, np.empty((len(doc), len(self.classes))), bg, doc)
        for j, c in enumerate(self.classes):
            counts = NGramCounts.build(np.concatenate([d for d, l in zip(docs, labels) if l == c]),
                                       self.V, self.order)
            model = KneserNeyModel(counts, self.order)
            for ids, P, _, _ in flat.values():
                P[:, j] = model.token_probs(ids)
            del model, counts
        return {name: (P, bg, doc) for name, (_, P, bg, doc) in flat.items()}

    def predict(self, cached, lam=1.0):
        P, bg, doc = cached
        mix = P if bg is None or lam == 1.0 else lam * P + (1 - lam) * bg[:, None]
        logp = np.log(mix)
        n_docs = int(doc.max()) + 1
        ll = np.stack([np.bincount(doc, logp[:, c], minlength=n_docs)
                       for c in range(len(self.classes))], axis=1)
        return [self.classes[i] for i in (ll + self.log_prior).argmax(axis=1)]
