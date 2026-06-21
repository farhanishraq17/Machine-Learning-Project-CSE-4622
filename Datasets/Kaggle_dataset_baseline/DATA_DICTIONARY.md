# Codeforces Master Dataset — Data Dictionary

**File:** `codeforces_master_dataset.csv` · **Rows:** 11,156 · **Columns:** 58 · **Encoding:** UTF-8 (RFC-4180 quoted)

This is the **baseline master dataset**: every problem, every feature, nothing deleted. For modeling, filter and drop columns per task (see notes). Targets are `rating` (regression) and `tags`/`tags_norm` (multi-label classification). Empty target cells are blank in the file and read as `NaN`.

## Identifiers
| Column | Type | Description |
|--------|------|-------------|
| `contestId` | int | Codeforces contest id. With `index` forms the unique key. |
| `index` | str | Problem index within contest (`A`, `B`, `F2`; numeric `01–14` for contest 921). |
| `index_ordinal` | int | `A→1, B→2, …`; numeric indices kept as-is. Difficulty position prior. |
| `url` | str | Canonical problem URL. |
| `name` | str | Original problem title. |
| `name_clean` | str | Title with whitespace/HTML/smart-quotes normalized. |

## Targets & tag fields
| Column | Type | Description |
|--------|------|-------------|
| `rating` | int / NaN | **Regression target.** 800–3500; `NaN` for 270 unrated problems. |
| `has_rating` | bool | True if `rating` present (use to filter the regression set). |
| `tags` | str | **Original tag list** (comma-separated). Untouched ground truth. |
| `tags_norm` | str | **Classification target.** Lowercased, de-duplicated, alphabetized, `, `-separated. |
| `n_tags` | int | Number of tags. |
| `has_tags` | bool | True if tags present (use to filter the classification set). |
| `is_special_tag` | bool | True if the meta-tag `*special` is present (consider excluding from label space). |

## Structural metadata
| Column | Type | Description |
|--------|------|-------------|
| `time_limit_sec` | float | Time limit (s). |
| `memory_limit_mb` | int | Memory limit (MB). |
| `solvedCount` | int | Number of accepted solvers. **⚠ Leakage for rating (r≈−0.87) — exclude from the rating model.** |

## Text
| Column | Type | Description |
|--------|------|-------------|
| `statement_raw` | str | Raw scraped statement (LaTeX `$$$…$$$`, newlines). Source of truth; not a direct feature. |
| `statement_clean` | str | Original "clean" field (still had residual newlines/HTML — kept for provenance). |
| `statement_clean_norm` | str | **Cleaned text**: folded Unicode whitespace, stripped formatting HTML, preserved ≤/≥/× and Cyrillic. Use for TF-IDF/embeddings. |
| `statement_proc` | str | NLP-ready: lowercased, stop-words removed, Porter-stemmed. Drop-in for `TfidfVectorizer`. |

## Engineered numeric features — original vs. recomputed
The dataset shipped precomputed counts; some were unreliable. Both are kept. **Use the `*_recomputed` columns** (validated, reproducible); `*_mismatch` flags rows where the original disagreed.
| Column group | Type | Description |
|--------------|------|-------------|
| `statement_length_chars` / `_recomputed` / `_mismatch` | int/int/0-1 | Character length of cleaned text. Original 100% correct. |
| `statement_length_words` / `_recomputed` / `_mismatch` | int/int/0-1 | Word count. Original 100% correct. |
| `num_lessequal` / `_recomputed` / `_mismatch` | int/int/0-1 | Count of `≤`/`<=`. **Original only 38% correct — use recomputed.** |
| `num_greaterequal` / `_recomputed` / `_mismatch` | int/int/0-1 | Count of `≥`/`>=`. Original 95%. |
| `num_equals` / `_recomputed` / `_mismatch` | int/int/0-1 | Count of standalone `=`. Original 100%. |
| `num_numeric_tokens` / `_recomputed` / `_mismatch` | int/int/0-1 | Count of integer tokens. Original 96%. |

## Additional engineered features
| Column | Type | Description |
|--------|------|-------------|
| `avg_word_length` | float | Mean token length in cleaned text. |
| `sentence_count` | int | Number of `.?!` runs. |
| `avg_sentence_length_words` | float | Words ÷ sentences. |
| `digit_count` | int | Digit characters. |
| `uppercase_ratio` | float | Uppercase ÷ alphabetic chars. |
| `digit_ratio` | float | Digits ÷ all chars. |
| `punctuation_ratio` | float | Punctuation ÷ all chars. |
| `math_symbol_count` | int | Count of `≤≥≠=<>+×∗*/·∑∏√∈∉∩∪`. |
| `unique_word_ratio` | float | Vocabulary richness (unique ÷ total words). |
| `example_count` | int | Mentions of "Example". |
| `has_cyrillic` | bool | Cyrillic present (bilingual/older statements). |
| `max_numeric_constant` | float | Largest integer in the statement (capped 1e18). Constraint-magnitude / difficulty signal. |
| `log10_max_numeric_constant` | float | `log10` of the above (model-friendly). |
| `has_mod_constant` | 0-1 | Statement mentions `998244353` / `1e9+7` → counting/DP/combinatorics signal. |
| `has_power_notation` | 0-1 | Statement uses `10^…`, `\cdot`, `\times 10` → constraint magnitudes. |

## Quality, duplicate & split helpers
| Column | Type | Description |
|--------|------|-------------|
| `dup_statement_exact` | 0-1 | Row's statement is byte-identical to another's (after normalization). |
| `dup_group_id` | int | Cluster id for exact-duplicate statements (−1 if unique). |
| `dup_name` | 0-1 | Title shared with another problem (names are reused; not a reliable dup signal alone). |
| `low_quality_statement` | 0-1 | Truncated/extraction-failed statement (<60 chars or <8 words). Targets still valid. |
| `cv_group` | int | **Leakage-safe grouping key.** Contests merged when they share a duplicate statement. Use with `GroupKFold` / group splits so no contest or duplicate crosses folds. |

## Recommended usage
- **Rating regression:** filter `has_rating`; drop `solvedCount`, `tags`, `tags_norm`, the `*_mismatch` flags, and the stale original count columns; split with `GroupKFold(groups=cv_group)`.
- **Tag classification:** filter `has_tags`; binarize `tags_norm` on `", "`; same grouped CV.
- Prefer `*_recomputed` features; treat `statement_proc` as the text input.
