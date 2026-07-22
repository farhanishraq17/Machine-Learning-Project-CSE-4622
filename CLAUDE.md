# CLAUDE.md — Project Instructions & Constraints

> **Read this file and `PROGRESS.md` at the start of every round. Update `PROGRESS.md`
> and `TODO.md` at the end of any round that changes the project state.**

## 1. What this project is

An empirical study for **CSE 4554 Machine Learning Lab** (IUT, Batch 22, 6th sem) that predicts
two things about Codeforces problems from the **problem statement text + metadata only**:

- **Track A — Rating regression:** predict problem `rating` (800–3500).
- **Track B — Tag classification:** multi-label prediction of 38 official algorithm tags (`tags_norm`).

Team: Farhan Ishraq (220041217), Arifin Rafi (220041230), Tashin Mustakim (220041239).

## 2. Hard constraints (do not violate)

1. **No transformers.** No BERT/RoBERTa/sentence-transformers/LLM embeddings of any kind.
   Allowed: TF-IDF, Word2Vec/FastText/Doc2Vec, LSA/TruncatedSVD, hand-crafted structural/
   statistical features, and classical ML (linear, SVM, trees, boosting, Naive Bayes, KNN).
2. **Seed 42 everywhere** (`random_state=42`). Results must be reproducible.
3. **Leakage-safe grouped splits.** Always split on `cv_group` (never a plain random split):
   locked 80/20 hold-out via `GroupShuffleSplit`, model selection via `GroupKFold(k=5)` **inside
   the training 80%**. No contest or duplicate statement may cross the train/test boundary.
4. **Fit-on-train-only.** Every `TfidfVectorizer`, `Word2Vec`, `StandardScaler`,
   `MultiLabelBinarizer`, and threshold calibration is `.fit()` on the train fold and
   `.transform()` on val/test. Never fit on the full dataset.
5. **`solvedCount` is leakage for rating** (r = −0.87). It is excluded from the rating model;
   it may be used for tags.
6. **Never delete rows or alter targets.** Data-quality issues are handled by *flagging*, not
   deletion (the one exception already applied: 35 fully-redundant exact-duplicate rows with
   identical statement AND labels were dropped from the modeling set — documented).
7. **Identical folds for all models** so paired significance tests stay valid — reuse the
   precomputed fold indices, don't regenerate per model.
8. **Honesty first.** Disclose limitations rather than hiding them (see §6).

## 3. Environment & how to run

- **OS/shell:** Windows 11, PowerShell (primary). Run all commands **from the repo root**.
- **Python:** 3.13. Key libs (see `experiments/results/versions.txt`): sklearn 1.9, lightgbm,
  xgboost, catboost, gensim, nltk, skmultilearn, scikit-posthocs, statsmodels.
- **Reproduce the full study:**
  ```bash
  python experiments/src/run_experiment.py
  ```
  Add `--fast` to skip the ablations. Feature blocks are cached under
  `experiments/results/cache/`, so re-runs are fast.
- **Config is the single source of truth:** `experiments/config.yaml` (seeds, paths, feature
  specs, tuning grids). Change experiment behavior there, not by hardcoding in scripts.

## 4. Repository map

| Path | What it holds |
|------|---------------|
| `Dataset/regression/` · `Dataset/classification/` | The two **used** per-task, leakage-cleaned CSVs (+ READMEs). Referenced by `experiments/config.yaml`. |
| `Dataset/original/` | Source data kept for provenance: `codeforces_master_dataset.csv` (11,156 × 58), `DATA_DICTIONARY.md`, and the raw `cf_problem_metadata.csv` (11,156 × 17). |
| `plan_for_Experiment.md` | The authoritative experiment execution plan (§0–§12), at repo root. |
| `experiments/config.yaml` | Central experiment config (seed 42). |
| `experiments/src/` | Harness (shared across both tasks): `make_splits`, `features`, `engine`, `metrics`, `models_rating`, `models_tags`, `evaluate`, `phase5`, `error_analysis`, `ablations`, `run_experiment`, `make_report`. |
| `experiments/results/` | Leaderboards, fold scores, significance, per-tag F1, figures, interpretability CSVs, ablations, cached features (kept, tracked). |
| `experiments/RESULTS.md` | Auto-generated combined experiment results (both tasks). |
| `Report/PROJECT_REPORT.md` | Consolidated data-engineering + modeling report (both tasks). |
| `Report/regression/` · `Report/classification/` | Per-task findings write-ups (`*_findings.md`). |
| `Report/*.pdf` | Submitted report (`217_230_239.pdf`) + project diagram (`Diagram_Comprehensive.pdf`). |
| `models/` , `model_ready/` | **Legacy** self-contained pipeline (`models/pipeline.py`) + its 80:20 splits and outputs. Kept at root for provenance; superseded by `experiments/`. |

## 5. Working conventions

- **Two implementations coexist:** the legacy `models/pipeline.py` (self-contained Phase 2–4) and
  the fuller `experiments/` harness (the plan in `plan_for_Experiment.md`). Prefer
  the `experiments/` harness for new modeling work; keep the two consistent when numbers change.
- **Leaderboards are append/rebuild artifacts** — `experiments/results/leaderboard_{rating,tags}.csv`.
  Every model runs on the same shared folds and reports **mean ± std across the 5 folds**, plus a
  single locked-test number for the champion.
- **Primary metrics:** Rating → **MAE** (secondary: RMSE, R², Spearman ρ, ±100/±200 bucket acc).
  Tags → **macro-F1 and micro-F1** (secondary: precision/recall, Hamming, subset acc, Jaccard, LRAP, per-tag F1).
- **Match the surrounding code style** in `experiments/src/` (compact, docstring'd, `common.CFG`-driven).
- Don't commit or push unless explicitly asked. If asked, branch first (never commit straight to `main`).

## 6. Known limitations to keep disclosed

- **Porter stemming** substitutes for WordNet lemmatization (`nltk_wordnet` unavailable offline).
- **Sentence-transformer embeddings unavailable** (downloads blocked offline) — and out of scope
  anyway per the no-transformers rule.
- Long-tail tags (<~50 train examples) are effectively unlearnable — report this as a ceiling,
  don't paper over it.

## 7. Current champions (verify against `PROGRESS.md` before quoting)

- **Rating:** LightGBM (unified features) — locked-test MAE ≈ **404**, R² ≈ **0.50**.
- **Tags:** LogReg (class_weight=balanced) on TF-IDF word 1–2grams — locked micro-F1 **0.4837**,
  macro-F1 **0.3493** (headline = base 0.5 threshold; per-label thresholds are *substitutes* for
  class weighting and are not adopted — see `Report/classification/classification_findings.md` §3).
  `Report/PROJECT_REPORT.md` §10 is reconciled: §10.1–10.3 = initial pipeline (marked superseded),
  §10.4 + per-task findings = canonical.
