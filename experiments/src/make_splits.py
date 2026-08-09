# -*- coding: utf-8 -*-
"""Phase 1 — leakage-safe splits (§2 of the plan).

For each task:
  * locked hold-out test  : GroupShuffleSplit(test_size, seed) on cv_group
  * model-selection folds : GroupKFold(k) on cv_group *within the training 80%*

Indices are absolute row positions into `load_task(task)` (which resets_index),
so every downstream script reproduces the exact same rows. Persisted to
results/splits_{task}.joblib. Asserts that no cv_group crosses train/test or folds.
"""
from __future__ import annotations
import numpy as np
import joblib
from sklearn.model_selection import GroupShuffleSplit, GroupKFold

from common import CFG, SEED, rp, load_task, seed_everything


def build_splits(task: str) -> dict:
    df = load_task(task)
    groups = df[CFG["split"]["group_col"]].to_numpy()
    n = len(df)
    idx = np.arange(n)

    # ---- locked 80/20 hold-out ----
    gss = GroupShuffleSplit(n_splits=1, test_size=CFG["split"]["test_size"], random_state=SEED)
    train_idx, test_idx = next(gss.split(idx, groups=groups))
    train_idx = np.sort(train_idx)
    test_idx = np.sort(test_idx)

    # ---- GroupKFold(k) inside the training 80% ----
    k = CFG["split"]["n_folds"]
    gkf = GroupKFold(n_splits=k)
    tr_groups = groups[train_idx]
    folds = []
    for fold_tr_rel, fold_va_rel in gkf.split(train_idx, groups=tr_groups):
        folds.append((train_idx[fold_tr_rel].copy(), train_idx[fold_va_rel].copy()))

    # ---- integrity assertions (leakage guard) ----
    g_train, g_test = set(groups[train_idx]), set(groups[test_idx])
    assert g_train.isdisjoint(g_test), f"{task}: cv_group leaks across train/test!"
    for i, (ftr, fva) in enumerate(folds):
        assert set(groups[ftr]).isdisjoint(set(groups[fva])), f"{task}: cv_group leaks in fold {i}!"
    # every training row is validated exactly once across folds
    va_all = np.concatenate([fva for _, fva in folds])
    assert np.array_equal(np.sort(va_all), np.sort(train_idx)), f"{task}: folds don't partition train"

    out = {
        "task": task, "seed": SEED, "n": n,
        "train_idx": train_idx, "test_idx": test_idx, "folds": folds,
        "n_groups_total": len(set(groups)),
        "n_groups_train": len(g_train), "n_groups_test": len(g_test),
    }
    joblib.dump(out, rp(CFG["paths"]["results_dir"], f"splits_{task}.joblib"))
    return out


def load_splits(task: str) -> dict:
    return joblib.load(rp(CFG["paths"]["results_dir"], f"splits_{task}.joblib"))


def main():
    seed_everything()
    for task in ["rating", "tags"]:
        s = build_splits(task)
        pct = 100 * len(s["test_idx"]) / s["n"]
        print(f"[{task}] n={s['n']}  train={len(s['train_idx'])}  test={len(s['test_idx'])} ({pct:.1f}%)  "
              f"folds={len(s['folds'])}  groups(total/tr/te)={s['n_groups_total']}/{s['n_groups_train']}/{s['n_groups_test']}")
        sizes = [len(fva) for _, fva in s["folds"]]
        print(f"        fold val sizes: {sizes}  -> leakage-free (asserted)")


if __name__ == "__main__":
    main()
