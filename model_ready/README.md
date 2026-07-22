# Model-Ready Dataset (80:20) — legacy pipeline splits

**Legacy artifact** — these splits feed `../models/pipeline.py` (the initial Phase 2–4 study).
The current harness in `../experiments/` builds its own splits from `../Dataset/`.
Built from the cleaned intermediate of the master dataset (now `../Dataset/original/codeforces_master_dataset.csv`).
**No ground truth altered.** Per-task target filtering only, plus removal of 35 fully-redundant
exact-duplicate rows (identical statement AND identical labels).

```
model_ready/
├── rating_regression/   train.csv  test.csv   (target: rating)   8,673 / 2,192
└── tag_classification/  train.csv  test.csv   (target: tags_norm) 8,792 / 2,203
```

## What was applied
1. **Corrected features** — validated recomputed values under clean names (the 38%-wrong original `num_lessequal` is gone).
2. **`solvedCount` removed from `rating_regression/`** (−0.87 leak; kept for tags).
3. **80:20 grouped split** by `contestId`, with contests merged when they share a duplicate statement (union-find), seed 42 — **no contest or duplicate statement crosses train/test** (verified). Tune hyperparameters with GroupK-Fold CV *inside train*.
4. **Per-task filtering** — rating = rated rows only; tags = tagged rows only.

## Columns
**ID:** `contestId`, `index`, `name`, `url`
**Text:** `name_clean`, `statement_clean_norm` (raw cleaned), `statement_proc` (lowercased, stop-word-removed, Porter-stemmed — ready for TF-IDF)
**Numeric features (18):** `time_limit_sec`, `memory_limit_mb`, `statement_length_chars`, `statement_length_words`, `num_lessequal`, `num_greaterequal`, `num_equals`, `num_numeric_tokens`, `avg_word_length`, `sentence_count`, `avg_sentence_length_words`, `digit_count`, `uppercase_ratio`, `digit_ratio`, `punctuation_ratio`, `math_symbol_count`, `unique_word_ratio`, `example_count`
**Tag task extra:** `solvedCount`
**Target:** `rating` (regression) | `tags_norm` (multi-label, comma-separated)

## Trained models
See `../models/` — `pipeline.py` reproduces the legacy study (seed 42), `results_*.json` hold its metrics, and `predictions_*_test.csv` hold test-set predictions. For the project's final results see `../experiments/RESULTS.md` and `../Report/`.
