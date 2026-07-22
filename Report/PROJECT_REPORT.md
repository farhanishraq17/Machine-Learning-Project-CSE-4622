# Codeforces Rating & Tag Prediction — Consolidated Project Report

**Project:** An Empirical Study on Rating and Tag Prediction for Codeforces Problems (CSE 4554 / 4622)
**Team:** Farhan Ishraq (220041217), Arifin Rafi (220041230), Tashin Mustakim (220041239)
**Report date:** 2026-06-22

This single document consolidates the full data-engineering and modeling pipeline: audit, validation, cleaning, leakage analysis, feature work, the data dictionary, and final model results. It reflects the **current state** of the project, whose primary deliverable is the Kaggle master dataset `codeforces_master_dataset.csv` (11,156 rows × 58 columns).

---

## 0. Deliverables & File Map (current status)

| Artifact | Location | Purpose |
|----------|----------|---------|
| **`codeforces_master_dataset.csv`** | `Dataset/original/` | **Baseline master — all features, 11,156 × 58.** The Kaggle upload. |
| `README.md`, `DATA_DICTIONARY.md` | `Dataset/original/` | Dataset description + column reference. |
| `cf_problem_metadata.csv` | `Dataset/original/` | Raw input (17 columns); all derived columns are deterministic functions of it. |
| `codeforces_rating_regression.csv` · `codeforces_tag_classification.csv` | `Dataset/regression/` · `Dataset/classification/` | The two used per-task, leakage-cleaned datasets. |
| `model_ready/` , `models/` | root | **Legacy** `pipeline.py` (Phase 2–4) + its 80:20 splits, metrics, and test predictions. Superseded by `experiments/`. |
| `experiments/` | root | The full non-transformer modeling harness + results (`RESULTS.md`). |
| `PROJECT_REPORT.md` (this file) · per-task `*_findings.md` | `Report/` , `Report/{regression,classification}/` | **This consolidated report** + the separated per-task findings. |

---

## 1. Dataset Audit

**Shape:** 11,156 rows × 17 raw columns. **Primary key `(contestId, index)` is unique** (0 duplicates); `url` is also unique. Problems span Codeforces contests **#1 through #2236** (latest included), across 1,980 contests.

### 1.1 Raw schema, types, missingness

| # | Column | Type | Missing | % |
|---|--------|------|--------:|--:|
| 1 | contestId | integer | 0 | 0.00 |
| 2 | index | string | 0 | 0.00 |
| 3 | name | string | 0 | 0.00 |
| 4 | rating | integer | 270 | 2.42 |
| 5 | tags | string | 145 | 1.30 |
| 6 | url | string | 0 | 0.00 |
| 7 | statement_raw | string | 0 | 0.00 |
| 8 | statement_clean | string | 0 | 0.00 |
| 9 | time_limit_sec | float | 0 | 0.00 |
| 10 | memory_limit_mb | integer | 0 | 0.00 |
| 11 | statement_length_chars | integer | 0 | 0.00 |
| 12 | statement_length_words | integer | 0 | 0.00 |
| 13 | num_lessequal | integer | 0 | 0.00 |
| 14 | num_greaterequal | integer | 0 | 0.00 |
| 15 | num_equals | integer | 0 | 0.00 |
| 16 | num_numeric_tokens | integer | 0 | 0.00 |
| 17 | solvedCount | integer | 0 | 0.00 |

Only the two **targets** have missing values: `rating` (270 unrated) and `tags` (145 untagged). Per policy **no rows were deleted**; boolean flags `has_rating` / `has_tags` let each task filter its usable subset.

### 1.2 Duplicates

| Check | Count |
|-------|------:|
| Fully identical rows | 0 |
| Duplicate `url` | 0 |
| Duplicate `(contestId, index)` | 0 |
| Duplicate `name` | 609 |
| Duplicate `statement_raw` / `statement_clean` | 97 / 97 |

97 problems share an identical statement; after normalization these form **52 exact-duplicate clusters covering 150 rows** — overwhelmingly the same problem appearing in both Div.1 and Div.2 (normal for Codeforces). 609 name collisions are reused titles (e.g. "Numbers"), not real duplicates. All flagged (`dup_statement_exact`, `dup_group_id`, `dup_name`), none deleted.

### 1.3 Structural / encoding issues

| Issue | name | statement_raw | statement_clean |
|-------|-----:|--------------:|----------------:|
| Leading / trailing whitespace | 1 / 52 | 5 / 1 | 0 / 0 |
| Double spaces | 4 | 6,112 | 0 |
| Tab characters | 0 | 2 | 0 |
| Embedded newlines | 0 | 6,623 | 6,623 |
| Repeated newlines | 0 | 2,312 | 0 |
| HTML-like tags | 1 | 48 | 48 |
| Invisible/space-variant Unicode | 0 | 5,111 | 5,111 |
| Control characters | 0 | 2 | 2 |
| Smart quotes | 10 | 102 | 102 |

Key findings: **`statement_clean` was not actually clean** — it retained 6,623 embedded newlines, 48 HTML-tag rows, 102 smart-quote rows, 2 control chars, and space-variant Unicode (thin space U+2009 ×96,542; no-break space U+00A0 ×14,319). These were normalized into `statement_clean_norm`. HTML was stripped **selectively**: real formatting (`<b>`, `<table>`, `<tr>`, `<td>`) removed, but content placeholders that are part of the text (`<number>`, `<hostname>`, `<expression>`, grammar tokens like `<term>`) preserved. Semantically important symbols (`≤` ×22,274, `≥`, `×`, `·`, Cyrillic) were kept. No mojibake, no `�`, no CRLF, no shifted columns. The numeric `index` values `01`–`14` (contest 921 "Labyrinth") are **valid**, not malformed.

### 1.4 Numeric range validation

| Column | Min | Max | Mean | Verdict |
|--------|----:|----:|-----:|---------|
| rating | 800 | 3500 | 1861.85 | Valid CF range. |
| time_limit_sec | 0.4 | 15 | 2.12 | Valid, positive. |
| memory_limit_mb | 4 | 1024 | 310.22 | Valid. |
| statement_length_chars | 2 | 10022 | 1923.50 | Min=2 = truncated outlier (flagged). |
| statement_length_words | 1 | 2163 | 360.69 | Min=1 = truncated outlier (flagged). |
| num_lessequal | 0 | 88 | 8.17 | Precomputed unreliable (see §2). |
| num_greaterequal | 0 | 18 | 0.12 | 93.7% zero (low variance). |
| num_equals | 0 | 83 | 2.88 | Recomputation matches. |
| num_numeric_tokens | 0 | 596 | 43.20 | 96% match. |
| solvedCount | 0 | 697330 | 10084.07 | Heavy right skew. |

No negative, impossible, or non-coercible values anywhere.

---

## 2. Feature Validation (recomputed vs. precomputed)

Every engineered count was recomputed from the text and compared to the shipped value (basis: original `statement_clean`).

| Feature | Formula | % match | Mismatches | Max abs diff | Reliable? |
|---------|---------|--------:|-----------:|-------------:|-----------|
| statement_length_chars | `len(clean)` | 100.00 | 0 | 0 | **YES** |
| statement_length_words | `len(clean.split())` | 100.00 | 0 | 0 | **YES** |
| num_lessequal | `count('≤')+count('<=')` | 38.24 | 6,890 | 88 | **NO** |
| num_greaterequal | `count('≥')+count('>=')` | 95.35 | 519 | 18 | **PARTIAL** |
| num_equals | standalone `=` regex | 100.00 | 0 | 0 | **YES** |
| num_numeric_tokens | `\b\d+\b` count | 96.45 | 396 | 94 | **PARTIAL** |

**Conclusion:** lengths and `num_equals` are fully trustworthy. **`num_lessequal` is only 38% correct — do not trust the shipped column;** use the corrected `num_lessequal_recomputed`. `num_greaterequal` and `num_numeric_tokens` are mostly right but should be taken from their `_recomputed` versions for reproducibility. Per-row `*_mismatch` flags are in the dataset. **Targets were never recomputed or altered.**

---

## 3. Data-Leakage Analysis (most important for model validity)

**3.1 `solvedCount` → `rating` — HIGH leakage.** Correlation r = **−0.87 (log scale)**, −0.48 raw. This is mechanical: rating is derived from contestant solving behavior, and solve count is the same difficulty signal. Since the task is "predict rating from the statement + metadata," `solvedCount` encodes the answer and would not be available for a new problem. **Excluded from the rating model** (kept for tags, where it is not leakage).

**3.2 Statement text → `tags` — expected signal, one near-trivial case.** The `interactive` tag: 299 problems carry it, 279 (93%) literally contain "interact". Legitimate (the model reads the statement) but near-trivially predictable — it inflates per-tag scores. Other algorithm names rarely appear ("greedy" 34, "geometry" 27, "binary search" 13, "dynamic programming" 1) = genuine signal, not leakage.

**3.3 Metadata checks (clean).** `url` holds only `contestId/index` (no target info); `name` encodes neither target; `contestId` weakly encodes era/difficulty — **always split by contest group** (see `cv_group`). The `*special` tag is a CF meta-marker (537 problems), not algorithmic — flagged via `is_special_tag`; consider excluding from the label space.

---

## 4. Class Imbalance

### 4.1 Rating (regression)

| Band | Count | Band | Count |
|------|------:|------|------:|
| 800–1000 | 1,420 | 1800–2000 | 1,020 |
| 1000–1200 | 833 | 2000–2200 | 934 |
| 1200–1400 | 917 | 2200–2400 | 870 |
| 1400–1600 | 924 | 2400–2600 | 864 |
| 1600–1800 | 1,027 | 2600–2900 | 885 |
| | | 2900–3500 | 966 |

**Reasonably balanced** (mean ≈ 1862, std ≈ 732); 800–1000 is largest because 800 is the floor. Healthy for regression. If binning into classes, use stratified splits + class weights for the sparse ≥2900 bands.

### 4.2 Tags (multi-label) — 38 official tags, steep long tail

| Tag | Count | Tag | Count | Tag | Count |
|-----|------:|-----|------:|-----|------:|
| greedy | 3,490 | bitmasks | 700 | matrices | 139 |
| math | 3,424 | two pointers | 640 | fft | 113 |
| implementation | 3,022 | *special | 537 | graph matchings | 106 |
| dp | 2,486 | geometry | 424 | string suffix structures | 101 |
| constructive algorithms | 2,074 | dsu | 408 | ternary search | 70 |
| data structures | 2,010 | divide and conquer | 359 | meet-in-the-middle | 52 |
| brute force | 1,972 | interactive | 299 | 2-sat | 39 |
| binary search | 1,282 | shortest paths | 293 | expression parsing | 38 |
| sortings | 1,229 | games | 292 | chinese remainder theorem | 25 |
| graphs | 1,203 | probabilities | 267 | schedules | 15 |
| dfs and similar | 1,059 | hashing | 237 | communication | 8 |
| trees | 954 | flows | 161 | | |
| number theory | 881 | combinatorics | 829 | strings | 810 |

**Rare tags (<50):** chinese remainder theorem (25), 2-sat (39), expression parsing (38), schedules (15), communication (8). **Cardinality:** 0–11 tags/problem (0→145, 1→1840, 2→3146, 3→2762, 4→1705, 5→886, 6→429, 7→167, 8→56, 9→14, 10→3, 11→3).

**Recommendations:** multi-label (iterative) stratification; class weights / `pos_weight = neg/pos` over naive oversampling; report **macro-F1**; threshold or merge tags with <15 examples; exploit label co-occurrence (`dfs and similar`↔`graphs`↔`trees`).

---

## 5. Outliers

| Type | Count | Decision |
|------|------:|----------|
| Truncated/near-empty (<60 chars or <8 words) | 23 | Keep + flag (`low_quality_statement`); targets still valid. |
| Statements <50 chars | 6 | Extraction failures (e.g. 120/H = "-1"). Flagged. |
| Very long (>6000 chars) | 17 | Legitimate hard problems. Keep. |
| Many tags (≥8) | 76 | Legitimate complex problems. Keep. |
| Very high numeric-token count (>300) | 10 | Keep; consider log-scaling. |
| 1.5×IQR length outliers | 304 | Mostly genuine. Keep. |
| Impossible values | 0 | None. |

No outlier rows removed.

---

## 6. Per-Column Quality Scores (0–100)

| Column | Score | Reasoning |
|--------|------:|-----------|
| url | 99 | Complete, unique, no leakage. |
| index | 96 | Complete; valid (incl. numeric indices). |
| contestId | 95 | Complete key part; memorization risk if used raw. |
| time_limit_sec | 94 | Complete, valid difficulty signal. |
| statement_clean_norm | 93 | Cleaned, semantics preserved; primary text feature. |
| memory_limit_mb | 92 | Complete, valid; lower variance. |
| tags | 92 | Target; clean taxonomy, 1.30% missing. |
| rating | 90 | Target; valid range, 2.42% missing. |
| num_equals | 90 | 100% reproducible. |
| statement_length_chars / _words | 97 / 97 | 100% reproducible, high value. |
| name | 88 | Minor noise (cleaned into `name_clean`). |
| num_numeric_tokens | 88 | 96% consistent; use recomputed. |
| statement_raw | 85 | Noisy source, not a direct feature. |
| statement_clean | 80 | Superseded by `statement_clean_norm`. |
| num_greaterequal | 70 | 95% consistent but 93.7% zeros. |
| solvedCount | 55 | Accurate but **leakage for rating**; skewed. |
| num_lessequal | 45 | **38% inconsistent — use recomputed.** |

Address first: `num_lessequal` (recompute), `solvedCount` (leakage), `num_greaterequal` (low variance), `statement_clean` (superseded).

---

## 7. Feature-Importance Estimates (heuristic)

| Tier | Rating regression | Tag classification |
|------|-------------------|--------------------|
| **High** | TF-IDF/embeddings of statement, statement_length_words, num_numeric_tokens, num_lessequal_recomputed, time/memory limits, max_numeric_constant | TF-IDF/embeddings, math_symbol_count, num_lessequal_recomputed, keyword flags |
| **Medium** | avg_word_length, sentence_count, math_symbol_count, unique_word_ratio, example_count, index_ordinal | statement_length_words, num_numeric_tokens, time_limit_sec |
| **Low** | num_greaterequal, digit_ratio, uppercase_ratio, has_cyrillic | num_greaterequal, uppercase_ratio, memory_limit_mb |
| **Harmful/leakage** | **solvedCount**, raw contestId | raw contestId |
| **Never use** | rating (target), tags (other target) | tags/rating of same row |

---

## 8. Feature-Engineering Review

Features marked **[in master]** are already in `codeforces_master_dataset.csv`; **[suggested]** are recommended next steps. Task: R=rating, C=tags, B=both.

**Text / length:** statement_length_words **[in master, B]**, statement_length_chars **[in master, B]**, avg_word_length **[in master, R]**, unique_word_ratio (vocab richness) **[in master, B]**, example_count **[in master, R]**, paragraph_count [suggested, R].

**Metadata / structural:** time_limit_sec & memory_limit_mb (raw) **[B]**, index_ordinal **[in master, R]**, is_special_tag **[in master, C]**, tl×ml interaction [suggested, R], contest-era bucket [suggested, R — use only with grouped CV].

**Statistical / numeric:** num_lessequal_recomputed **[in master, B]**, num_numeric_tokens_recomputed **[in master, B]**, digit_count **[in master, R]**, math_symbol_count **[in master, B]**, max_numeric_constant + log10 **[in master, R]**, has_power_notation **[in master, B]**, has_mod_constant **[in master, C]**.

**Lexical / keyword:** statement_proc (stemmed text, TF-IDF-ready) **[in master, B]**, algorithm-keyword flags [suggested, C], query_count [suggested, C], TF-IDF 1–2 grams [suggested → built in modeling, B].

**Semantic / embedding:** Word2Vec mean-pooled [built in modeling, B], sentence-transformer embeddings [suggested — blocked offline, B], topic-model loadings [suggested, C].

**Readability:** sentence_count **[in master, R]**, avg_sentence_length_words **[in master, R]**, uppercase_ratio / digit_ratio / punctuation_ratio **[in master, R]**, has_cyrillic **[in master, B]**, Flesch reading-ease [suggested, R], char entropy [suggested, R].

**Priority shortlist:** (1) TF-IDF of `statement_proc`; (2) `max_numeric_constant` (log) — already added; (3) `index_ordinal` — already added; (4) `has_mod_constant` + keyword flags; (5) sentence-transformer embeddings if online.

---

## 9. Data Dictionary — `codeforces_master_dataset.csv` (58 columns)

**Identifiers:** `contestId`, `index`, `index_ordinal` (A→1…), `url`, `name`, `name_clean`.
**Targets & tags:** `rating` (800–3500, NaN=unrated), `has_rating`, `tags` (original), `tags_norm` (lowercased/sorted/deduped, NaN=untagged), `n_tags`, `has_tags`, `is_special_tag`.
**Metadata:** `time_limit_sec`, `memory_limit_mb`, `solvedCount` (⚠ leakage for rating).
**Text:** `statement_raw`, `statement_clean` (provenance), `statement_clean_norm` (cleaned, semantics preserved — use for TF-IDF/embeddings), `statement_proc` (lowercased, stop-word-removed, Porter-stemmed).
**Original vs. recomputed counts (use the `_recomputed`):** `statement_length_chars`, `statement_length_words`, `num_lessequal`, `num_greaterequal`, `num_equals`, `num_numeric_tokens`, each with `_recomputed` and `_mismatch` variants.
**Engineered:** `avg_word_length`, `sentence_count`, `avg_sentence_length_words`, `digit_count`, `uppercase_ratio`, `digit_ratio`, `punctuation_ratio`, `math_symbol_count`, `unique_word_ratio`, `example_count`, `has_cyrillic`, `max_numeric_constant`, `log10_max_numeric_constant`, `has_mod_constant`, `has_power_notation`.
**Quality / duplicate / split helpers:** `dup_statement_exact`, `dup_group_id`, `dup_name`, `low_quality_statement`, `cv_group` (leakage-safe grouping key — use with `GroupKFold`).

**Recommended usage:** rating → filter `has_rating`, drop `solvedCount`/`tags`/`tags_norm`/`*_mismatch`/stale originals, split on `cv_group`. tags → filter `has_tags`, binarize `tags_norm` on `", "`, same grouped CV. Always prefer `*_recomputed`; treat `statement_proc` as text input.

---

## 10. Modeling Results

> **Canonical results.** The project's final numbers come from the **extended harness** (§10.4,
> `experiments/RESULTS.md`, and the per-task findings in `Report/regression/` and
> `Report/classification/`): rating champion **LightGBM** (locked MAE 404.04, R² 0.4965) and tag
> champion **LogReg with class_weight='balanced'** (locked micro-F1 0.4837, macro-F1 0.3493; with
> per-label thresholds the macro-F1 rises further — see the classification findings).
> §10.1–10.3 below document the **initial single-split pipeline** (`models/pipeline.py`); its
> Linear-SVM tag champion was superseded once the harness added class-weighted models on identical
> CV folds with significance testing. It is kept for provenance — its rating conclusion (LightGBM,
> MAE ≈ 404) matches the harness.

**Initial-pipeline setup (10.1–10.3):** 80:20 split grouped by `contestId` (duplicate-linked via `cv_group`), seed 42; hyperparameters tuned with grouped K-fold / holdout CV inside train only. 35 fully-redundant exact-duplicate rows (identical statement **and** labels) removed from the modeling set; no others dropped. **Features:** TF-IDF unigrams+bigrams (~28k dims, min_df=5, sublinear) + Word2Vec (100-d, skip-gram, mean-pooled) + 18 numerical/statistical features → unified matrix. **NLP transform:** lowercase → tokenize → English stop-word removal → **Porter stemming** (substitute for WordNet lemmatization, which could not download offline; functionally equivalent for TF-IDF). All metrics are on a held-out test sharing **no contest or duplicate statement** with training (rating: 8,673 train / 2,192 test; tags: 8,792 / 2,203, 38 labels).

### 10.1 Path A — Rating (regression) — initial pipeline

| Model | MAE | RMSE | R² |
|-------|----:|-----:|---:|
| Mean baseline | 608.53 | 728.85 | −0.00 |
| Ridge (unified) | 424.69 | 530.88 | 0.4695 |
| **LightGBM (unified) — champion** | **403.54** | **517.41** | **0.496** |

LightGBM cuts test MAE from 608.5 to 403.5 and explains ~50% of rating variance from statement + metadata alone. An MAE of ~404 on the 800–3500 scale is in line with published text-only difficulty prediction; rating is intrinsically noisy (it comes from contestant behavior, not text). This is a non-leaky ceiling for the current features.

### 10.2 Path B — Tags (multi-label, One-vs-Rest) — initial pipeline

| Model | micro-F1 | macro-F1 | Precision μ | Recall μ | Hamming | Subset Acc | Jaccard |
|-------|---------:|---------:|------------:|---------:|--------:|-----------:|--------:|
| TF-IDF + Logistic Regression | 0.4111 | 0.2607 | 0.6123 | 0.3095 | 0.0682 | 0.0781 | 0.2913 |
| TF-IDF + Linear SVM — *initial champion, superseded* | 0.4235 | 0.3075 | 0.5477 | 0.3452 | 0.0723 | 0.0749 | 0.3022 |

Linear SVM was the stronger of the two models this initial pipeline tried. Precision (~0.55–0.61) far exceeds recall (~0.31–0.35): the models predict a tag only when confident. Macro-F1 trails micro-F1 because long-tail tags are rarely recovered. **The extended harness (§10.4) then added class-weighted models on identical grouped folds, and `LogReg_bal` (class_weight='balanced') overtook Linear SVM — locked micro-F1 0.4837 / macro-F1 0.3493, avg. CV rank 1.0, Friedman p ≈ 4e-06 — making it the project's final tag champion.**

**Per-tag F1 highlights (initial Linear SVM):** strong on `interactive` (0.96), `trees` (0.73), `games` (0.73), `geometry` (0.64), `strings` (0.60), `probabilities` (0.58), `math` (0.53), `greedy` (0.52), `graphs` (0.51), `number theory` (0.50), `bitmasks` (0.50). Mid: `combinatorics` (0.44), `dfs and similar` (0.43), `*special` (0.41), `constructive` (0.41), `data structures` (0.40), `implementation` (0.39), `dp` (0.37). **Zero** on the rarest tags (`flows`, `matrices`, `graph matchings`, `fft`, `ternary search`, `expression parsing`, `meet-in-the-middle`, `2-sat`, `chinese remainder theorem`, `schedules`, `divide and conquer`) — the class-imbalance ceiling.

### 10.3 Duplicate analysis (summary of pair-level findings)

264 exact-duplicate statement pairs (150 share identical labels; **114 have conflicting rating/tags** — different problem versions sharing a statement, e.g. "Arithmancy (Hard)" 3100 vs "(Medium)" 2600) and 450 near-duplicate pairs (253 with Jaccard ≥ 0.8). All flagged, never auto-deleted; the grouped split guarantees none cross train/test.

---

## 11. Reproducibility & Known Limitations

- **Reproducible:** every derived column is a deterministic function of the preserved text; splits and models use seed 42. `models/pipeline.py` regenerates the initial Phase 2–4 study; `python experiments/src/run_experiment.py` regenerates the full harness (leaderboards, significance, thresholds/ordinal, error analysis, ablations, and this report's §10.4).
- **Offline substitutions:** Porter **stemming** replaces WordNet lemmatization, and **sentence-transformer embeddings were unavailable** (model downloads blocked in the build environment) — transformer embeddings are also outside this project's no-transformers scope.
- **Next-step status (all originally-planned upgrades executed in the harness):** tag recall via `class_weight='balanced'` ✔ (became the champion) and **per-label thresholds** ✔ (macro-F1 lift, applied to the champion — see `Report/classification/classification_findings.md`); ordinal binning ✔ (§10.4 study; the LightGBM champion already optimizes `regression_l1`, i.e. median/quantile-0.5 loss); LightGBM+linear ensembling ✔ (Stacking trails LightGBM, paired Wilcoxon p = 0.125 — not significantly better); tag-tail policy ✔ (all-38 reported as primary, ≥15-example floor reported alongside, `*special` reported separately). Remaining beyond scope: transformer embeddings.

**Publication check:** the dataset is audited, validated, de-leaked (by flagging, not deletion), enriched, documented, and parse-verified — clean, reproducible, and trustworthy, with the two offline substitutions disclosed above rather than hidden.

---

<!-- EXPERIMENTS-HARNESS-SUMMARY -->
### 10.4 Extended harness (experiments/, seed 42)

A broader roster on identical `GroupKFold(5)` folds + a locked test lives in `experiments/` (see `experiments/RESULTS.md`). Champions:

- **Rating:** `LightGBM` — locked MAE **404.04**, R² **0.4965** (CV MAE 405.8235±8.4551).
- **Tags:** `LogReg_bal` — locked micro-F1 **0.4837**, macro-F1 **0.3493** (CV micro-F1 0.4886±0.0049).

Adds significance tests (Friedman/Nemenyi + CD diagrams), per-label threshold tuning, an ordinal-rating study, feature-family/TF-IDF/Word2Vec ablations, and error analysis. <!-- EXPERIMENTS-HARNESS-SUMMARY -->
