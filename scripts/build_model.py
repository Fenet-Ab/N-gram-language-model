"""Preprocess a raw Amharic corpus and train n-gram counts.

Example:
    python scripts/build_model.py --corpus data/GPAC.txt --out data/gpac --order 4
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ngram import pipeline  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True, help="raw text file, or directory of .txt files")
    ap.add_argument("--out", required=True, help="model directory to write")
    ap.add_argument("--order", type=int, default=4)
    ap.add_argument("--min-count", type=int, default=2, help="vocabulary threshold; rarer words become <unk>")
    ap.add_argument("--min-counts", type=int, nargs="+", default=list(pipeline.DEFAULT_MIN_COUNTS),
                    help="per-order pruning thresholds, non-decreasing (default: 1 1 2 2)")
    ap.add_argument("--mem-budget-mb", type=int, default=512, help="working memory for counting")
    ap.add_argument("--max-chars", type=int, default=None, help="read only about the first N characters (for a quick sample)")
    ap.add_argument("--force", action="store_true", help="rebuild every stage even if outputs exist")
    args = ap.parse_args()
    vocab, counts, ids, stats = pipeline.build(
        args.corpus, args.out, args.order, args.min_count, args.min_counts, args.max_chars,
        args.force, args.mem_budget_mb)
    print(f"sentences={stats['sentences']:,} tokens={stats['tokens']:,} vocab={len(vocab):,}")
    for k in range(1, counts.order + 1):
        print(f"{k}-grams: {counts.meta['distinct'][k - 1]:,} distinct, {counts.num_grams(k):,} kept")


if __name__ == "__main__":
    main()
