# Codeforces Rating & Tag Prediction (CSE 4554 ML Lab)

An empirical study predicting two things about Codeforces problems from the **problem statement +
metadata only**, using **classical (non-transformer) ML**:

- **Regression** — predict a problem's `rating` (800–3500).
- **Classification** — multi-label prediction of 38 official algorithm tags.

Team: Farhan Ishraq (220041217), Arifin Rafi (220041230), Tashin Mustakim (220041239).

## Repository layout

```
.
├── Dataset/
│   ├── regression/        codeforces_rating_regression.csv   (+ README)   ← used dataset
│   ├── classification/    codeforces_tag_classification.csv  (+ README)   ← used dataset
│   └── original/          master + raw source (codeforces_master_dataset.csv,
│                          cf_problem_metadata.csv, DATA_DICTIONARY.md, README)
│
├── experiments/           the full modeling harness (shared across both tasks)
│   ├── config.yaml        single source of truth (seed 42, paths, grids)
│   ├── src/               make_splits, features, engine, metrics, models_rating,
│   │                      models_tags, evaluate, phase5, error_analysis, ablations,
│   │                      run_experiment, make_report
│   ├── results/           leaderboards, significance, per-tag F1, ablations, figures,
│   │                      interpretability CSVs, cached feature blocks
│   └── RESULTS.md         auto-generated combined results (both tasks)
│
├── Report/
│   ├── PROJECT_REPORT.md          consolidated report (data engineering + both tasks)
│   ├── regression/                regression_findings.md
│   ├── classification/            classification_findings.md
│   ├── 217_230_239.pdf            submitted report
│   └── Diagram_Comprehensive.pdf  project diagram
│
├── models/ , model_ready/   legacy self-contained pipeline (Phase 2–4), kept for provenance
│
├── plan_for_Experiment.md   the experiment execution plan
├── CLAUDE.md                instructions + constraints (read each round)
├── PROGRESS.md              status: done / remaining
└── TODO.md                  prioritized next steps
```

## Reproduce

```bash
python experiments/src/run_experiment.py          # full study (roster + significance + ablations + report)
python experiments/src/run_experiment.py --fast   # skip the ablations
```

All paths resolve to the repo root regardless of the working directory (see `experiments/src/common.py`).
Seed 42 throughout; feature blocks are cached under `experiments/results/cache/` so re-runs are fast.

## Champions (locked test)

| Task | Champion | Result |
|------|----------|--------|
| Regression | LightGBM (unified) | MAE **404.0**, R² **0.50** |
| Classification | LogReg (balanced), TF-IDF 1–2g | micro-F1 **0.484**, macro-F1 **0.349** |

Full numbers and analysis in [`Report/`](Report/PROJECT_REPORT.md) and the per-task findings.

## Ground rules

No transformers. Leakage-safe grouped splits on `cv_group`. Fit-on-train-only. `solvedCount` excluded
from the rating model (leakage). Rows are never deleted / targets never altered — see `CLAUDE.md`.
