# -*- coding: utf-8 -*-
"""Shared CV engine (§2 protocol) used by baselines, rosters, and significance.

A model is a spec:
    {"name": str, "feat": <feature-set name>, "fit_predict": callable}
where fit_predict(Xtr, ytr, gtr, Xev) -> predictions on Xev (regression: yhat;
multilabel: (Y_pred_binary, Y_score_or_None)). The engine builds the model's
feature set per fold (fit-on-train-only, cached), calls fit_predict, times it, and
collects per-fold metrics on IDENTICAL folds so paired significance tests are valid.

cv_regression / cv_multilabel return:
    {"per_fold":[...], "agg":{metric:{mean,std}}, "fold_primary":[...], "time":{...}}
locked_regression / locked_multilabel fit on full train, score the locked test once.
"""
from __future__ import annotations
import time
import numpy as np
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from common import CFG, SEED
from features import get_feature_set
import metrics as M


# ------------------------------------------------------------ tuning helpers
def inner_group_holdout(groups, frac=0.15, seed=1):
    """Positional (fit_idx, val_idx) within a fold-train, grouped (for early stopping)."""
    gss = GroupShuffleSplit(n_splits=1, test_size=frac, random_state=seed)
    fit_rel, val_rel = next(gss.split(np.arange(len(groups)), groups=groups))
    return fit_rel, val_rel


def tune_scalar(make, grid, Xtr, ytr, gtr, scorer, k=3, maximize=True):
    """Pick the best scalar hyperparameter via inner GroupKFold on the fold-train."""
    best_p, best_s = None, None
    kk = min(k, len(np.unique(gtr)))
    gkf = GroupKFold(n_splits=max(2, kk))
    for p in grid:
        scores = []
        for a, b in gkf.split(Xtr, ytr, gtr):
            mdl = make(p)
            mdl.fit(Xtr[a] if hasattr(Xtr, "__getitem__") else Xtr[a], _row(ytr, a))
            scores.append(scorer(_row(ytr, b), mdl.predict(_row2(Xtr, b))))
        s = float(np.mean(scores))
        if best_s is None or (s > best_s if maximize else s < best_s):
            best_s, best_p = s, p
    return best_p


def _row(y, idx):
    return y[idx]


def _row2(X, idx):
    return X[idx]


# ------------------------------------------------------------ regression CV
def cv_regression(spec, task, df, splits):
    per_fold, fold_primary, fit_t, pred_t = [], [], 0.0, 0.0
    for fi, (ftr, fva) in enumerate(splits["folds"]):
        Xtr, Xev = get_feature_set(task, f"fold{fi}", df, ftr, fva, spec["feat"])
        ytr = df.iloc[ftr]["rating"].to_numpy(float)
        yva = df.iloc[fva]["rating"].to_numpy(float)
        gtr = df.iloc[ftr][CFG["split"]["group_col"]].to_numpy()
        t0 = time.time()
        yhat = spec["fit_predict"](Xtr, ytr, gtr, Xev)
        dt = time.time() - t0
        fit_t += dt
        mm = M.rating_metrics(yva, yhat)
        per_fold.append(mm)
        fold_primary.append(mm[M.RATING_PRIMARY])
    return {"per_fold": per_fold, "agg": M.aggregate(per_fold),
            "fold_primary": fold_primary, "time": {"cv_fit_predict_s": round(fit_t, 2)}}


def locked_regression(spec, task, df, splits):
    tr, te = splits["train_idx"], splits["test_idx"]
    Xtr, Xte = get_feature_set(task, "locked", df, tr, te, spec["feat"])
    ytr = df.iloc[tr]["rating"].to_numpy(float)
    yte = df.iloc[te]["rating"].to_numpy(float)
    gtr = df.iloc[tr][CFG["split"]["group_col"]].to_numpy()
    yhat = spec["fit_predict"](Xtr, ytr, gtr, Xte)
    return M.rating_metrics(yte, yhat), yhat, yte


# ------------------------------------------------------------ multilabel CV
def cv_multilabel(spec, task, df, splits, mlb):
    from common import parse_tags
    per_fold, fold_primary, fit_t = [], [], 0.0
    for fi, (ftr, fva) in enumerate(splits["folds"]):
        Xtr, Xev = get_feature_set(task, f"fold{fi}", df, ftr, fva, spec["feat"])
        Ytr = mlb.transform(parse_tags(df.iloc[ftr]["tags_norm"]))
        Yva = mlb.transform(parse_tags(df.iloc[fva]["tags_norm"]))
        t0 = time.time()
        Ypred, Yscore = spec["fit_predict"](Xtr, Ytr, Xev)
        fit_t += time.time() - t0
        mm = M.tag_metrics(Yva, Ypred, Yscore)
        per_fold.append(mm)
        fold_primary.append(mm["f1_macro"])
    return {"per_fold": per_fold, "agg": M.aggregate(per_fold),
            "fold_primary": fold_primary, "time": {"cv_fit_predict_s": round(fit_t, 2)}}


def locked_multilabel(spec, task, df, splits, mlb):
    from common import parse_tags
    tr, te = splits["train_idx"], splits["test_idx"]
    Xtr, Xte = get_feature_set(task, "locked", df, tr, te, spec["feat"])
    Ytr = mlb.transform(parse_tags(df.iloc[tr]["tags_norm"]))
    Yte = mlb.transform(parse_tags(df.iloc[te]["tags_norm"]))
    Ypred, Yscore = spec["fit_predict"](Xtr, Ytr, Xte)
    return M.tag_metrics(Yte, Ypred, Yscore), Ypred, Yscore, Yte
