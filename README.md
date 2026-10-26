# NLP Assignment 1 — Amharic n-gram language models

n-gram language models (n = 1..4) on the GPAC Amharic corpus: corpus analysis, intrinsic and
extrinsic evaluation, and a next-word suggestion app. The design and the alternatives considered
are in [`docs/dev_plans/amharic_ngram_language_models.md`](docs/dev_plans/amharic_ngram_language_models.md).

| Task | Where |
|---|---|
| 1 — n-gram LM, statistics, stopwords, word clouds, sentence probability, generation | `notebooks/01_ngram_language_model.ipynb` |
| 2 — intrinsic evaluation (perplexity) | `notebooks/02_intrinsic_evaluation.ipynb` |
| 3 — extrinsic evaluation (news classification, keystroke savings) | `notebooks/03_extrinsic_evaluation.ipynb` |
| 4 — next-word suggestion app | `notebooks/04_next_word_suggestion.ipynb`, `app/streamlit_app.py` |

## Setup

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
`````

## TO RUN THE PROJECT
source .venv/bin/activate
streamlit run app/streamlit_app.py -- --model-dir data/gpac --order 3



Put the corpus at `data/GPAC.txt`. A directory of `.txt` files or a `.zip` also works; set
`CORPUS` in the notebook setup cell. Counting streams the corpus from disk in key-range
partitions and drops 3- and 4-grams seen once. The full corpus (77M tokens) builds in about
25 minutes with about 1.1 GB of process memory, plus file cache the OS can reclaim. Set
`MAX_CHARS` for a smaller run.

```bash
python scripts/build_model.py --corpus data/GPAC.txt --out data/gpac --order 4  
streamlit run app/streamlit_app.py -- --model-dir data/gpac --order 3     
pytest
```

## Package

`ngram/` is shared by the notebooks, the script and the app:

- `normalize.py` — homophone normalization, sentence splitting, tokenization
- `corpus.py` — train/dev/test split and corpus statistics
- `vocab.py` — word ↔ id mapping, `<s>`, `</s>`, `<unk>`
- `counts.py` — n-gram counts stored as sorted int64 keys (`row(history) · V + word`)
- `models.py` — MLE, Laplace and interpolated Kneser-Ney models
- `generate.py` — sentence generation and next-word suggestion
- `evaluate.py` — perplexity, keystroke savings, per-class LM classifier
- `stopwords.py`, `viz.py`, `pipeline.py`
