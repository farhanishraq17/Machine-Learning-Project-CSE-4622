# -*- coding: utf-8 -*-
"""Phase 1 — metric suites (§4.2 rating, §5.3 tags).

rating_metrics(y_true, y_pred)                 -> dict of 9 scorelines
tag_metrics(Y_true, Y_pred, Y_score=None)      -> dict of label-based + ranking metrics
per_tag_f1(Y_true, Y_pred, classes)            -> {tag: {f1, precision, recall, support}}

All values are plain floats so they serialize cleanly to JSON/CSV.
"""
from __future__ import annotations
import numpy as np
from scipy.stats import spearmanr, kendalltau
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score, median_absolute_error,
    f1_score, precision_score, recall_score, hamming_loss, accuracy_score,
    jaccard_score, label_ranking_average_precision_score, roc_auc_score,
)


# ------------------------------------------------------------------ rating
def smape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, float); y_pred = np.asarray(y_pred, float)
    denom = (np.abs(y_true) + np.abs(y_pred))
    denom[denom == 0] = 1.0
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / denom) * 100.0)


def bucket_acc(y_true, y_pred, tol) -> float:
    return float(np.mean(np.abs(np.asarray(y_pred, float) - np.asarray(y_true, float)) <= tol))


def rating_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, float); y_pred = np.asarray(y_pred, float)
    rho = spearmanr(y_true, y_pred).correlation if len(np.unique(y_pred)) > 1 else 0.0
    tau = kendalltau(y_true, y_pred).correlation if len(np.unique(y_pred)) > 1 else 0.0
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
        "MedianAE": float(median_absolute_error(y_true, y_pred)),
        "sMAPE": smape(y_true, y_pred),
        "Spearman": float(rho if rho == rho else 0.0),   # guard NaN
        "Kendall": float(tau if tau == tau else 0.0),
        "acc_100": bucket_acc(y_true, y_pred, 100),
        "acc_200": bucket_acc(y_true, y_pred, 200),
    }


RATING_PRIMARY = "MAE"          # lower is better
RATING_LOWER_BETTER = {"MAE", "RMSE", "MedianAE", "sMAPE"}


# ------------------------------------------------------------------ tags
def tag_metrics(Y_true, Y_pred, Y_score=None) -> dict:
    Y_true = np.asarray(Y_true); Y_pred = np.asarray(Y_pred)
    m = {
        "f1_micro": float(f1_score(Y_true, Y_pred, average="micro", zero_division=0)),
        "f1_macro": float(f1_score(Y_true, Y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(Y_true, Y_pred, average="weighted", zero_division=0)),
        "f1_samples": float(f1_score(Y_true, Y_pred, average="samples", zero_division=0)),
        "precision_micro": float(precision_score(Y_true, Y_pred, average="micro", zero_division=0)),
        "precision_macro": float(precision_score(Y_true, Y_pred, average="macro", zero_division=0)),
        "recall_micro": float(recall_score(Y_true, Y_pred, average="micro", zero_division=0)),
        "recall_macro": float(recall_score(Y_true, Y_pred, average="macro", zero_division=0)),
        "hamming_loss": float(hamming_loss(Y_true, Y_pred)),
        "subset_accuracy": float(accuracy_score(Y_true, Y_pred)),
        "jaccard_samples": float(jaccard_score(Y_true, Y_pred, average="samples", zero_division=0)),
    }
    # ranking metrics need continuous scores
    if Y_score is not None:
        Y_score = np.asarray(Y_score, float)
        try:
            m["LRAP"] = float(label_ranking_average_precision_score(Y_true, Y_score))
        except Exception:
            m["LRAP"] = float("nan")
        # micro-AUC over labels that have both classes present
        try:
            keep = [j for j in range(Y_true.shape[1]) if 0 < Y_true[:, j].sum() < len(Y_true)]
            m["auc_micro"] = float(roc_auc_score(Y_true[:, keep], Y_score[:, keep], average="micro")) if keep else float("nan")
        except Exception:
            m["auc_micro"] = float("nan")
    else:
        m["LRAP"] = float("nan"); m["auc_micro"] = float("nan")
    return m


def per_tag_f1(Y_true, Y_pred, classes) -> dict:
    Y_true = np.asarray(Y_true); Y_pred = np.asarray(Y_pred)
    f1 = f1_score(Y_true, Y_pred, average=None, zero_division=0)
    pr = precision_score(Y_true, Y_pred, average=None, zero_division=0)
    rc = recall_score(Y_true, Y_pred, average=None, zero_division=0)
    out = {}
    for j, tag in enumerate(classes):
        out[tag] = {"f1": float(f1[j]), "precision": float(pr[j]), "recall": float(rc[j]),
                    "support": int(Y_true[:, j].sum())}
    return out


TAG_PRIMARY = ("f1_macro", "f1_micro")   # both reported; higher is better


def tune_label_thresholds(Y, S, steps=41):
    """Per-label decision threshold maximizing that label's F1 on (Y, S)."""
    Y = np.asarray(Y); S = np.asarray(S, float)
    thr = np.full(Y.shape[1], 0.5)
    for j in range(Y.shape[1]):
        col = S[:, j]
        if col.std() == 0 or Y[:, j].sum() == 0:
            continue
        lo, hi = np.percentile(col, [2, 98])
        best_t, best_f = thr[j], -1.0
        for t in np.linspace(lo, hi, steps):
            f = f1_score(Y[:, j], (col >= t).astype(int), zero_division=0)
            if f > best_f:
                best_f, best_t = f, t
        thr[j] = best_t
    return thr


def apply_thresholds(S, thr):
    return (np.asarray(S, float) >= np.asarray(thr, float)).astype(int)


def aggregate(dicts: list[dict]) -> dict:
    """mean±std across folds -> {metric: {'mean':.., 'std':..}} (NaN-safe)."""
    keys = dicts[0].keys()
    agg = {}
    for k in keys:
        vals = np.array([d[k] for d in dicts], float)
        agg[k] = {"mean": float(np.nanmean(vals)), "std": float(np.nanstd(vals))}
    return agg


if __name__ == "__main__":
    # tiny self-test
    yt = np.array([800, 1600, 2400, 3200]); yp = np.array([900, 1500, 2500, 3000])
    print("rating:", {k: round(v, 3) for k, v in rating_metrics(yt, yp).items()})
    Yt = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0]])
    Yp = np.array([[1, 0, 0], [0, 1, 0], [1, 0, 0]])
    Ys = np.array([[.9, .1, .4], [.2, .8, .3], [.7, .3, .2]])
    print("tags:", {k: round(v, 3) for k, v in tag_metrics(Yt, Yp, Ys).items()})
