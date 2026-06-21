# Experiment Execution Plan — Codeforces Rating & Tag Prediction

**Scope:** End-to-end plan to run, compare, and report many classical ML models on two tasks
— **rating regression** and **tag multi-label classification** — using only non-transformer
features (TF-IDF, Word2Vec, and structural/statistical features). Designed to be executed in
Claude Code from the datasets already prepared in this project.

---

## 0. Goals & Hard Constraints

- **Two tasks, run in parallel tracks:** Rating (regression) and Tags (multi-label classification).
- **Breadth of models:** train a large roster per task and compare them on a shared, fair test bed.
- **Multiple "scorelines":** judge each model on *several* metrics, not one — and look at agreement/disagreement between metrics.
- **No transformers.** No BERT/RoBERTa/sentence-transformers/LLM embeddings. Allowed: TF-IDF, Word2Vec/FastText/Doc2Vec, LSA/SVD, hand-crafted features, classical ML (linear, SVM, trees, boosting, NB, KNN).
- **Honesty first:** identical leakage-safe folds for every model; vectorizers/scalers fit on training folds only; documented seeds.

---

## 1. Inputs (already prepared)

| Dataset | File | Rows | Target |
|---------|------|-----:|--------|
| Regression | `Regression_Dataset/codeforces_rating_regression.csv` | 10,886 | `rating` (800–3500) |
| Classification | `Classification_Dataset/codeforces_tag_classification.csv` | 11,011 | `tags_norm` (38 labels) |
| Master (all features) | `Kaggle_dataset/codeforces_master_dataset.csv` | 11,156 | both |

Each task file already excludes the other task's leakage columns and ships `statement_proc`
(TF-IDF-ready text), `statement_clean_norm` (clean text), validated numeric features, and
`cv_group` (the leakage-safe grouping key). **Use `cv_group` for every split.**

---

## 2. Evaluation Protocol (the test bed — build this first)

This is the backbone. Get it right once and every model plugs into it.

1. **Hold-out test set:** one **GroupShuffleSplit** by `cv_group`, 80/20, `random_state=42`. Lock it; never tune on it.
2. **Model selection:** **GroupKFold (k=5)** on `cv_group` *within the training 80%*. All hyperparameter search and model comparison happen here.
3. **Fit-on-train-only rule:** `TfidfVectorizer`, `Word2Vec`, `StandardScaler`, `MultiLabelBinarizer`, threshold calibration — all `.fit()` on the train fold, `.transform()` on the val/test fold. No global fit.
4. **Identical folds for all models:** precompute the fold indices once and reuse, so model differences aren't fold noise. This enables paired statistical tests later.
5. **Report mean ± std across the 5 folds**, then a single final number on the locked test set for the chosen champion.
6. **Fixed seeds** everywhere (`random_state=42`); log library versions.

> Deliverable for this section: a `make_splits.py` that emits reusable fold indices + the locked test indices.

---

## 3. Feature Representations (build a small matrix of feature sets)

Treat features as an experimental axis. Build these blocks once, cache them:

**Text → sparse**
- `tfidf_word_1`: TF-IDF unigrams (`min_df=5`, `sublinear_tf=True`).
- `tfidf_word_12`: TF-IDF unigrams+bigrams (the default workhorse).
- `tfidf_char_35`: TF-IDF char n-grams (3–5) — robust to math tokens/typos.

**Text → dense (non-transformer)**
- `w2v_mean`: Word2Vec (100–300d, skip-gram) mean-pooled over tokens.
- `w2v_tfidf_mean`: TF-IDF-weighted Word2Vec mean (often beats plain mean).
- `lsa_300`: TruncatedSVD(300) on the TF-IDF matrix (a dense, decorrelated view).
- *(optional)* `doc2vec`: gensim Doc2Vec as an alternative dense encoder.

**Structural / statistical**
- `stats`: the numeric columns (lengths, counts, ratios, `max_numeric_constant`, flags, limits…), standardized.

**Feature-set combinations to evaluate (ablation grid):**
`stats only` · `tfidf_word_12 only` · `tfidf + stats` · `tfidf + w2v + stats (unified)` · `lsa + stats`.
This isolates how much each family contributes and tells you the best input for each model class.

> Note: linear/SVM/NB models like high-dimensional **sparse TF-IDF**; tree/boosting models prefer **dense, lower-dim** inputs (stats + LSA + w2v). Pair each model with the representation it likes (see rosters).

---

## 4. Track A — Rating Prediction (Regression)

### 4.1 Model roster (breadth)
| Family | Models | Preferred features |
|--------|--------|--------------------|
| Baseline | `DummyRegressor(mean/median)` | — |
| Linear | LinearRegression, **Ridge**, Lasso, ElasticNet, (Huber/SGDRegressor) | TF-IDF (+stats) |
| Instance | KNeighborsRegressor | LSA/w2v + stats |
| Kernel | SVR (RBF), LinearSVR | LSA/w2v + stats (scaled) |
| Trees | DecisionTree, RandomForest, ExtraTrees | stats (+LSA/w2v) |
| Boosting | GradientBoosting, AdaBoost, HistGradientBoosting, **XGBoost**, **LightGBM**, (CatBoost) | unified / dense |
| Ensemble | VotingRegressor / StackingRegressor over the best 3–4 | mixed |

### 4.2 Metrics (scorelines)
- **MAE** (primary — interpretable in rating points), **RMSE** (penalizes big misses), **R²**.
- **Median AE** (robust), **sMAPE** (scale-relative).
- **Rank quality:** Spearman ρ and Kendall τ between predicted and true rating (do we order problems correctly?).
- **Bucket accuracy:** % predictions within **±100** and within **±200** rating (≈ within one difficulty step). Very communicative for a report.

### 4.3 Tuning
- Per model, a focused grid/`RandomizedSearchCV` over `GroupKFold`, scoring on **neg-MAE**.
- Key knobs: Ridge/Lasso `alpha`; SVR `C`,`gamma`; RF/ET `n_estimators`,`max_depth`,`max_features`; boosting `learning_rate`,`num_leaves/max_depth`,`n_estimators` (+ early stopping on a grouped val).

### 4.4 Optional framing experiment
- Treat rating as **ordinal buckets** (e.g., 800–900, …) and try classification/ordinal regression; compare bucket-accuracy against the regressors. Report as a secondary study.

---

## 5. Track B — Tag Prediction (Multi-label Classification)

### 5.1 Problem-transformation strategies (an experimental axis)
- **Binary Relevance** (independent classifier per tag) — primary, simple, strong with TF-IDF.
- **One-vs-Rest** (sklearn `OneVsRestClassifier`) — equivalent BR wrapper; use for most base learners.
- **Classifier Chains** — models label dependencies (`dfs↔graphs↔trees`); compare vs BR.
- **Label Powerset** *(caution)* — only for a reduced/top-k tag subset; document the trade-off.

### 5.2 Base-learner roster (breadth)
| Family | Models | Notes |
|--------|--------|-------|
| Baseline | most-frequent / stratified dummy, label-prior thresholding | floor |
| Linear | **Logistic Regression**, **LinearSVC**, SGDClassifier(log/hinge) | TF-IDF champions |
| Naive Bayes | MultinomialNB, ComplementNB, BernoulliNB | strong, fast TF-IDF baselines |
| Instance | KNeighborsClassifier | dense features |
| Trees | DecisionTree, RandomForest, ExtraTrees | stats + LSA |
| Boosting | GradientBoosting, AdaBoost, **XGBoost**, **LightGBM** | per-label or native multilabel |

Wrap each in BR/OvR (and Classifier Chain for the top learners).

### 5.3 Metrics (scorelines)
- **F1:** micro, macro, weighted, samples (report all four — they tell different stories).
- **Precision & Recall** (micro & macro).
- **Hamming loss** (per-label error rate), **subset/exact-match accuracy** (strict), **Jaccard (samples)**.
- **Label-ranking:** LRAP (label-ranking average precision) and micro-AUC from decision scores.
- **Per-tag F1 table** + support — essential for seeing the head/tail story.

### 5.4 Imbalance & thresholds (high-leverage)
- `class_weight='balanced'` (LogReg/SVM/trees) and/or per-label `scale_pos_weight` (boosting).
- **Per-label threshold tuning:** optimize each tag's decision threshold on validation to maximize its F1 (big macro-F1 gains for rare tags). Use `decision_function`/`predict_proba`.
- **Rare-tag policy:** report with all 38 tags, and also with a frequency floor (e.g., tags with ≥15 train examples) so head-tag performance isn't hidden by unlearnable tails. Consider treating `*special` separately.
- Try multi-label **iterative stratification** as a sanity comparison to the grouped split (note: prefer grouped to avoid duplicate leakage; document the tension).

---

## 6. Model-Comparison Framework

1. **One leaderboard per task**, all models on identical folds, columns = every metric in §4.2 / §5.3, values = mean ± std across folds.
2. **Rank** by the primary metric (MAE for rating; macro-F1 *and* micro-F1 for tags) but **show all metrics** — call out where rankings disagree (e.g., a model great on micro-F1 but poor on macro-F1).
3. **Statistical significance:** since folds are shared, use **paired Wilcoxon signed-rank** (or the Nadeau–Bengio corrected resampled t-test) between the top models; for the whole roster use **Friedman test + Nemenyi post-hoc** and a **critical-difference diagram**. This makes "model A beats B" defensible.
4. **Efficiency columns:** train time and predict time — relevant for a fair "champion" choice.
5. **Champion selection:** pick per task using the primary metric + significance + simplicity, then run **once** on the locked test set and report final numbers.

---

## 7. Error Analysis & Interpretation

- **Rating:** residual plots (error vs true rating, vs length), worst over/under-predictions, calibration by bucket. Expect underprediction of very hard problems — quantify it.
- **Tags:** per-tag precision/recall, most-confused tag pairs, tags the model never predicts; inspect example statements for systematic misses.
- **Feature importance:** linear coefficients (top tokens per tag / for rating), tree gain importance, and **permutation importance** on the held-out set. Cross-check against the leakage notes.
- Save a handful of qualitative example predictions for the write-up.

---

## 8. Ablation Studies (what actually helps)

- **Feature-family ablation** (§3 grid): quantify lift from TF-IDF → +Word2Vec → +stats.
- **TF-IDF hyperparameters:** n-gram range (1 vs 1–2 vs char), `min_df`, `max_features`, `sublinear_tf`.
- **Word2Vec:** vector size (100/200/300), window, sg vs cbow, mean vs tf-idf-weighted pooling.
- **Preprocessing:** stemming vs none vs `*` (lemmatization if you can get WordNet locally); stopwords on/off.
- **Strategy ablation (tags):** BR vs Classifier Chain vs per-label thresholding.

---

## 9. Reproducibility & Engineering

- Single `config.yaml` (seeds, paths, grids); `random_state=42` throughout.
- Cache feature blocks (`.npz`/`.joblib`) so models reuse them.
- **Append every run to `results/leaderboard.csv`** with: task, model, feature_set, strategy, all metrics (mean±std), best params, train/predict time, timestamp, git hash.
- Save fitted champions (`joblib`) + their vectorizers.
- Pin library versions; note CPU/RAM. (Memory tip: TF-IDF + sparse models are light; dense w2v + boosting on full TF-IDF is the heavy combo — use LSA/stats for trees.)

---

## 10. Suggested Folder Structure (for Claude Code)

```
experiments/
├── config.yaml
├── src/
│   ├── make_splits.py        # locked test + GroupKFold indices (cv_group)
│   ├── features.py           # TF-IDF / Word2Vec / LSA / stats builders (fit-on-train)
│   ├── models_rating.py      # roster + tuning for regression
│   ├── models_tags.py        # roster + strategies + thresholds for multilabel
│   ├── evaluate.py           # all metrics + significance tests
│   └── run_experiment.py     # loops models × feature_sets, logs to leaderboard
├── results/
│   ├── leaderboard_rating.csv
│   ├── leaderboard_tags.csv
│   ├── per_tag_f1.csv
│   └── figures/              # CD diagram, residuals, confusion, importances
└── models/                   # saved champions + vectorizers
```

---

## 11. Phased Timeline

| Phase | Work | Output |
|------:|------|--------|
| **1. Test bed** | `make_splits.py`, metric module, leaderboard logger | Locked splits + evaluation harness |
| **2. Features** | Build & cache all feature blocks (§3) | Reusable feature matrices |
| **3. Baselines** | Dummy + 2–3 simple models per task, end-to-end | Pipeline validated, first numbers |
| **4. Full roster** | Train every model × suitable feature set, tune (§4–5) | Populated leaderboards |
| **5. Imbalance/thresholds** | Class weights + per-label thresholds (tags); ordinal study (rating) | Improved macro-F1 / bucket-acc |
| **6. Comparison** | Significance tests, CD diagram, champion pick, **test-set run** | Final ranked results |
| **7. Analysis & ablations** | Error analysis, feature importance, §8 ablations | Insight tables/plots |
| **8. Write-up** | Consolidate into report + figures | Final deliverable |

---

## 12. Definition of Done

- Both leaderboards filled with **all models × metrics (mean ± std)**, champions chosen with statistical backing.
- Per-tag F1 table and rating bucket-accuracy reported.
- Ablations quantify the value of each feature family and the tag strategy.
- One locked test-set number per task for each champion (reported once).
- Everything reproducible from `config.yaml` + cached splits (seed 42), with no transformer components anywhere.

---

### Quick reference — primary vs. secondary scorelines
- **Rating:** primary **MAE**; secondary RMSE, R², Spearman ρ, ±100/±200 bucket accuracy.
- **Tags:** primary **macro-F1 & micro-F1**; secondary precision/recall, Hamming, subset accuracy, Jaccard, LRAP, per-tag F1.
