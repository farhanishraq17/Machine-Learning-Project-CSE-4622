# PROGRESS.md — Project Status

*Last updated: 2026-07-22. See `CLAUDE.md` for constraints, `TODO.md` for the (now-empty) task queue.*

## Snapshot: ✅ PROJECT COMPLETE

Every phase of `plan_for_Experiment.md` (§1–§12 Definition of Done) is executed, documented, and
reconciled. Data engineering, both model rosters, significance testing, threshold/ordinal studies,
error analysis, all five ablations, per-task findings reports, and the repo restructure are done.
The work is **not yet committed** — the user will commit it themselves.

## Final champions (locked test, seed 42, grouped 80/20)

| Task | Champion | Result |
|------|----------|--------|
| Rating (regression) | **LightGBM** (unified) | MAE **404.04**, R² **0.4965** (baseline 608.8) |
| Tags (multi-label) | **LogReg_bal** (TF-IDF 1–2g, class_weight=balanced) | micro-F1 **0.4837**, macro-F1 **0.3493** (all 38 tags; ≥15-floor macro 0.3595) |

## Where everything lives

- `Report/PROJECT_REPORT.md` — consolidated report; §10 reconciled (harness canonical, initial
  pipeline marked superseded), §10.4 auto-refreshed.
- `Report/regression/regression_findings.md` · `Report/classification/classification_findings.md`
  — per-task findings with embedded figures, qualitative examples, and all study conclusions.
- `experiments/RESULTS.md` — auto-generated combined results (regenerate: `python experiments/src/make_report.py`).
- `experiments/results/` — leaderboards, foldscores, significance, phase5 JSONs, 5 ablation CSVs,
  figures, interpretability CSVs, cached features.

## Completed in the final round (2026-07-22)

- [x] **§10 reconciliation** — one canonical story (LogReg_bal tags / LightGBM rating); legacy
      pipeline.py narrative kept but explicitly marked superseded.
- [x] **P1 thresholds on the champion** — per-label thresholds applied to LogReg_bal itself:
      macro 0.3493→0.3515 (+0.002), micro 0.4837→0.3964. **Finding: class weighting and threshold
      tuning are substitutes** — headline stays the base champion. Floor-15 (36 tags) macro 0.3595;
      `*special` reported separately; tail policy documented (all-38 primary, nothing merged/dropped).
- [x] **P2 rating items** — Stacking not adopted (Wilcoxon p=0.125 vs LightGBM, worse mean);
      quantile framing noted (champion already optimizes L1 = quantile-0.5); hard-problem bias
      quantified (2900–3500 band under-predicted by ~862 on average; calibrated only in 1600–2000).
- [x] **P3 keyword ablation** — 15 keyword flags + query_count: macro +0.006, micro +0.000;
      recorded (`ablation_keyword_flags.csv`), not adopted (marginal).
- [x] **P4 polish** — figures embedded in findings + RESULTS.md; qualitative worst-prediction and
      confusion-pair tables added; stale paths fixed in Dataset/model_ready READMEs.
- [x] **Housekeeping** — deprecated `model_ready/*/val.csv` deleted; LightGBM feature-name warning
      flood silenced and UTF-8 console forced (both in `experiments/src/common.py`); keyword regexes
      use non-capturing groups.

## Earlier rounds (already complete)

- [x] Data engineering: audit, validation, leakage analysis, feature engineering, master dataset +
      per-task datasets + data dictionary (report §1–§9).
- [x] Full harness: splits, feature cache, both rosters, significance (Friedman/Nemenyi/Wilcoxon +
      CD diagrams), phase5, error analysis, feature-family/TF-IDF/W2V/strategy ablations.
- [x] Two harness bugs fixed (fp_histgbr sparse crash; ablation boosting rep → LightGBM).
- [x] Repo restructure: `Dataset/{regression,classification,original}`, `Report/` (+ per-task),
      root markdowns, `.gitignore`; all git renames; paths rewired and verified.

## Remaining

- **Commit** — intentionally left to the user.
- Optional/out-of-scope only: transformer embeddings (excluded by the no-transformers constraint).
