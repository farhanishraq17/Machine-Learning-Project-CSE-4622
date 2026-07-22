# Codeforces Problem **Tag** Dataset (Multi-label Classification)

Task-ready dataset for predicting a Codeforces problem's **algorithmic tags** (multi-label) from its
statement and structural metadata. Problems span contests **#1 to #2236**.

**File:** `codeforces_tag_classification.csv` · **11,011 rows × 35 columns** · UTF-8
(All 11,011 tagged problems; the 145 untagged problems from the master are excluded since they have no target.)

## Target
| Column | Description |
|--------|-------------|
| `tags_norm` | **Multi-label target.** Comma-space separated, lowercased, de-duplicated, alphabetized tags. **38 distinct tags** (official Codeforces taxonomy). Binarize by splitting on `", "` (e.g. `MultiLabelBinarizer`). |

A problem carries 1–11 tags. The label set is steeply long-tailed (head: `greedy`, `math`, `implementation`, `dp`; rare: `schedules`, `communication`, `2-sat`). Use multi-label stratification, class weights, and report **macro-F1** alongside micro-F1. Note `*special` is a CF meta-marker rather than an algorithmic tag — you may treat it separately.

## Why these columns
This is the classification-specific cut of the master dataset.

- **`solvedCount` is kept** — for tag prediction it is legitimate structural metadata (not leakage, unlike the rating task), and weakly informative for difficulty-linked tags.
- **No `rating` column and no tag-derived features** (`n_tags`, `is_special_tag`) — `rating` is the other task's output, and `n_tags`/`is_special_tag` are computed from the target itself (they would leak the answer).

All count features use the **validated, recomputed** values (the source's `num_lessequal` was only 38% correct; corrected here).

## Identifiers (reference, not features)
`contestId`, `index`, `url`, `name`, plus `index_ordinal` and `name_clean`.

## Text features
| Column | Description |
|--------|-------------|
| `statement_clean_norm` | Cleaned statement (Unicode/whitespace normalized, formatting HTML stripped, math symbols & Cyrillic preserved). Use for TF-IDF or embeddings — the primary signal for this task. |
| `statement_proc` | Lowercased, stop-word-removed, Porter-stemmed text. Drop-in for `TfidfVectorizer`. |

## Numeric / structural features
`time_limit_sec`, `memory_limit_mb`, `solvedCount`; `statement_length_chars`, `statement_length_words`, `num_lessequal`, `num_greaterequal`, `num_equals`, `num_numeric_tokens`; `avg_word_length`, `sentence_count`, `avg_sentence_length_words`, `digit_count`, `uppercase_ratio`, `digit_ratio`, `punctuation_ratio`, `math_symbol_count`, `unique_word_ratio`, `example_count`, `has_cyrillic`, `max_numeric_constant`, `log10_max_numeric_constant`, `has_mod_constant`, `has_power_notation`.

`has_mod_constant` (mentions 998244353 / 1e9+7) and `math_symbol_count` are especially useful for distinguishing counting/DP/number-theory/geometry tags.

## Helpers
| Column | Description |
|--------|-------------|
| `low_quality_statement` | 1 = truncated/extraction-failed statement (tags still valid; text features are not). Optional filter. |
| `cv_group` | **Leakage-safe grouping key.** Split with `GroupKFold(groups=cv_group)` so no contest or duplicate statement crosses folds. |

## Suggested usage
Fit `TfidfVectorizer` on `statement_proc` (train only), `MultiLabelBinarizer` on `tags_norm`, train One-vs-Rest Logistic Regression or Linear SVM, evaluate with `GroupKFold` on `cv_group`. The full experiment harness (champion: OvR Logistic Regression with `class_weight='balanced'`, locked-test micro-F1 ≈ 0.48 / macro-F1 ≈ 0.35) is in `../../experiments/`; a legacy reference pipeline is in `../../models/`.
