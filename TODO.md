# TODO.md — Next Steps

*Convention: `[ ]` open · `[x]` done. Keep this file and `PROGRESS.md` in sync.*

## Open items

- [ ] **Commit the completed work** (user will do this themselves — everything on disk is final:
      restructure + full study + reconciled reports + housekeeping).

That's it. All P0–P4 and housekeeping items are closed as of 2026-07-22 — see `PROGRESS.md`
("Completed in the final round") for what was done and where each result lives.

## Closed (final round, 2026-07-22)

- [x] P0 — §10 reconciled to the harness champions (legacy narrative marked superseded).
- [x] P1 — per-label thresholds applied to champion LogReg_bal (finding: substitutes for class
      weighting; headline stays base). Floor-15 reported; tail policy documented.
- [x] P2 — Stacking rejected with significance context; quantile-loss note (champion already L1);
      hard-problem under-prediction quantified (~−862 in the top band).
- [x] P3 — keyword flags + query_count ablation (macro +0.006; recorded, not adopted).
- [x] P4 — figures embedded, qualitative examples added, README/paths consistency pass.
- [x] Housekeeping — val.csv deleted; LGBM warning flood silenced; UTF-8 console.

## Ideas beyond scope (would break the project's constraints or scope)

- Transformer/sentence embeddings — explicitly excluded by the no-transformers rule.
- Non-median quantile prediction intervals — out of scope for a point-estimate leaderboard.
