"""Terminal version of the next-word suggester (for Colab or machines without a browser).

Run:  python app/cli.py --model-dir data/gpac --order 3

Type words, then press Enter. Enter a number to append that suggestion, an empty line to
start a new sentence, or `q` to quit.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ngram import pipeline  # noqa: E402
from ngram.generate import Suggester  # noqa: E402
from ngram.models import make_model  # noqa: E402
from ngram.normalize import normalize_chars, tokenize  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="data/gpac")
    ap.add_argument("--order", type=int, default=3)
    ap.add_argument("-k", type=int, default=5)
    args = ap.parse_args()
    vocab, counts = pipeline.load_counts(args.model_dir, args.order)
    suggester = Suggester(make_model(counts, args.order, "kn"), vocab)

    words = []
    while True:
        suggestions = suggester.suggest(words, k=args.k)
        print("\n" + (" ".join(words) or "(new sentence)"))
        for i, (w, p) in enumerate(suggestions, 1):
            print(f"  {i}. {w}  ({p:.4f})")
        line = input("> ").strip()
        if line == "q":
            break
        if not line:
            words = []
        elif line.isdigit() and 1 <= int(line) <= len(suggestions):
            words.append(suggestions[int(line) - 1][0])
        else:
            words.extend(tokenize(normalize_chars(line)))


if __name__ == "__main__":
    main()
