"""Amharic next-word suggestion (Task 4).

Run:  streamlit run app/streamlit_app.py -- --model-dir data/gpac --order 3

The last one or two words of the input (with a trigram model) form the history. When the
input does not end in a space, its last token is treated as a partly typed word and only
words starting with it are suggested. Clicking a suggestion appends it and recomputes.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ngram import pipeline  # noqa: E402
from ngram.generate import Suggester  # noqa: E402
from ngram.models import make_model  # noqa: E402
from ngram.normalize import normalize_chars, tokenize  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="data/gpac")
    ap.add_argument("--order", type=int, default=3)
    return ap.parse_args(sys.argv[1:])


@st.cache_resource
def load_suggesters(model_dir, max_order):
    vocab, counts = pipeline.load_counts(model_dir, max_order)
    return vocab, {n: Suggester(make_model(counts, n, "kn"), vocab) for n in range(1, max_order + 1)}


def split_input(text):
    """(complete words, partial last word) after the same normalization used in training."""
    text = normalize_chars(text)
    ends_with_space = text[-1:].isspace()
    words = tokenize(text)
    if words and not ends_with_space:
        return words[:-1], words[-1]
    return words, ""


def accept(word):
    words, _ = split_input(st.session_state.text)
    st.session_state.text = " ".join(words + [word]) + " "


def main():
    args = parse_args()
    st.set_page_config(page_title="Amharic next-word suggestion", layout="centered")
    st.title("Amharic next-word suggestion")
    vocab, suggesters = load_suggesters(args.model_dir, args.order)

    n = st.sidebar.select_slider("n-gram order", options=list(range(1, args.order + 1)), value=args.order)
    k = st.sidebar.slider("suggestions", 1, 10, 5)
    st.sidebar.caption(f"Vocabulary: {len(vocab):,} words · interpolated Kneser-Ney")

    st.session_state.setdefault("text", "")
    st.text_input("Type an incomplete Amharic sentence", key="text",
                  placeholder="ኢትዮጵያ ውስጥ የቡና ")
    words, partial = split_input(st.session_state.text)
    history = words[-(n - 1):] if n > 1 else []
    st.caption(f"history used: {' '.join(history) or '(none)'}"
               + (f" · completing: “{partial}”" if partial else ""))

    suggestions = suggesters[n].suggest(words, k=k, partial=partial)
    if not suggestions:
        st.info("No candidates for this prefix.")
        return
    cols = st.columns(len(suggestions))
    for col, (w, _) in zip(cols, suggestions):
        col.button(w, on_click=accept, args=(w,), use_container_width=True)
    st.dataframe(pd.DataFrame(suggestions, columns=["word", "P(word | history)"]),
                 hide_index=True, use_container_width=True)


if __name__ == "__main__":
    main()
