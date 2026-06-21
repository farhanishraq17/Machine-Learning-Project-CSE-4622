# Codeforces Problems — Rating & Tags Dataset

A cleaned, feature-rich dataset of **11,156 Codeforces problems** for two ML tasks: predicting a
problem's **difficulty rating** (regression) and its **algorithmic tags** (multi-label
classification), using only the problem statement and structural metadata. Problems span Codeforces
contests from **#1 up to #2236** (the latest contest included), across 1,980 contests.

Each row is one problem with **58 columns**, including:

- **Identifiers:** contestId, index, url, name
- **Targets:** rating (800–3500) and tags (38 distinct official tags, e.g. greedy, dp, math, graphs)
- **Metadata:** time/memory limits, solved count
- **Text:** raw statement, cleaned statement, and an NLP-processed (lowercased, stop-word-removed, stemmed) version
- **Engineered features:** statement length, word/number/symbol counts, vocabulary richness, max numeric constant, modulo/power-notation flags, and more
- **Helpers:** duplicate flags, missing-target flags, and a leakage-safe `cv_group` key for grouped cross-validation

Targets may be empty (270 problems unrated, 145 untagged) — filter per task.
Full column descriptions are in `DATA_DICTIONARY.md`.

**File:** `codeforces_master_dataset.csv` · 11,156 rows × 58 columns · UTF-8
