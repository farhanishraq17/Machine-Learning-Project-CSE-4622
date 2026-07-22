# Codeforces Problem **Rating** Dataset (Regression)

Task-ready dataset for predicting a Codeforces problem's **difficulty rating** (a regression target in
800–3500) from its statement and structural metadata. Problems span contests **#1 to #2236**.

**File:** `codeforces_rating_regression.csv` · **10,886 rows × 34 columns** · UTF-8
(All 10,886 rated problems; the 270 unrated problems from the master are excluded since they have no target.)

## Why these columns
This is the regression-specific cut of the master dataset. Two deliberate exclusions to keep the model honest:

- **`solvedCount` is removed** — it correlates with rating at **r ≈ −0.87 (log scale)** and is derived from the same contestant-performance signal that produces the rating. It is target leakage for "predict rating from the statement," so it is not included.
- **No tag-derived columns** (`tags`, `tags_norm`, `n_tags`, `is_special_tag`) — tags are the *other* task's output and are not part of "statement + structural metadata."

All count features use the **validated, recomputed** values (e.g. the original `num_lessequal` was only 38% correct in the source; here it is corrected).

## Target
| Column | Description |
|--------|-------------|
| `rating` | **Regression target.** Integer difficulty, 800–3500. |

## Identifiers (reference, not features)
`contestId`, `index`, `url`, `name` — plus `index_ordinal` (A→1, B→2 …; a usable difficulty-position prior) and `name_clean` (normalized title).

## Text features
| Column | Description |
|--------|-------------|
| `statement_clean_norm` | Cleaned statement (normalized Unicode/whitespace, formatting HTML stripped, math symbols & Cyrillic preserved). Use for TF-IDF or embeddings. |
| `statement_proc` | Lowercased, stop-word-removed, Porter-stemmed text. Drop-in for `TfidfVectorizer`. |

## Numeric / structural features
`time_limit_sec`, `memory_limit_mb`; `statement_length_chars`, `statement_length_words`, `num_lessequal`, `num_greaterequal`, `num_equals`, `num_numeric_tokens`; `avg_word_length`, `sentence_count`, `avg_sentence_length_words`, `digit_count`, `uppercase_ratio`, `digit_ratio`, `punctuation_ratio`, `math_symbol_count`, `unique_word_ratio`, `example_count`, `has_cyrillic`, `max_numeric_constant`, `log10_max_numeric_constant`, `has_mod_constant`, `has_power_notation`.

## Helpers
| Column | Description |
|--------|-------------|
| `low_quality_statement` | 1 = truncated/extraction-failed statement (rating still valid; text features are not). Optional filter. |
| `cv_group` | **Leakage-safe grouping key.** Split with `GroupKFold(groups=cv_group)` so no contest or duplicate statement crosses folds. |

## Suggested usage
Drop the ID columns from `X`, fit a `TfidfVectorizer` on `statement_proc` (train only), concatenate the numeric features, and evaluate with `GroupKFold` on `cv_group`. The full experiment harness (LightGBM champion, locked-test MAE ≈ 404, R² ≈ 0.50) is in `../../experiments/`; a legacy reference pipeline is in `../../models/`.
