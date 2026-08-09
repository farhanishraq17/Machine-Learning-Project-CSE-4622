# -*- coding: utf-8 -*-
"""One-command reproduction of the whole study (seed 42).

    python experiments/src/run_experiment.py [--fast]

Order: splits -> rating roster -> tags roster -> comparison/significance ->
imbalance/thresholds + ordinal -> error analysis -> ablations -> report.
Feature blocks are cached under results/cache, so re-runs are fast.
"""
from __future__ import annotations
import sys, time
from sklearn.preprocessing import MultiLabelBinarizer

from common import seed_everything, load_task, parse_tags, git_hash
import make_splits
import models_rating, models_tags
import evaluate, phase5, error_analysis, ablations
from make_splits import load_splits


def main(fast=False):
    t0 = time.time()
    seed_everything()
    print(f"== git {git_hash()} == building splits ==")
    make_splits.main()

    print("\n== RATING roster ==")
    dfr, sr = load_task("rating"), load_splits("rating")
    models_rating.run(dfr, sr)

    print("\n== TAGS roster ==")
    dft, st = load_task("tags"), load_splits("tags")
    mlb = MultiLabelBinarizer().fit(parse_tags(dft["tags_norm"]))
    models_tags.run(dft, st, mlb)

    print("\n== comparison + significance ==")
    evaluate.run_all()

    print("\n== imbalance/thresholds + ordinal ==")
    phase5.tags_thresholds(); phase5.ordinal_rating()

    print("\n== error analysis + importance ==")
    error_analysis.rating_analysis(); error_analysis.tags_analysis()

    if not fast:
        print("\n== ablations ==")
        ablations.feature_family_ablation(); ablations.tfidf_ablation()
        ablations.w2v_ablation(); ablations.tag_strategy_ablation()

    print("\n== building report ==")
    import make_report
    make_report.build()
    print(f"\nDONE in {round(time.time() - t0, 1)}s")


if __name__ == "__main__":
    main(fast="--fast" in sys.argv)
