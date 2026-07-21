# -*- coding: utf-8 -*-
"""Phase 3-4 — tag multilabel roster (§5). OvR / Binary-Relevance primary,
Classifier Chains for the top-2 linear learners.

fit_predict(Xtr, Ytr, Xev) -> (Y_pred_binary, Y_score_or_None)

Model / feature pairing (§3 note):
  Baseline_top3 -> stats(irrelevant)   LogReg/LinearSVC/ComplementNB -> tfidf_word_12
  KNN/RandomForest/LightGBM -> lsa+stats   Chain{LogReg,LinearSVC} -> tfidf_word_12

run(df, splits, mlb) -> {name: cv_result}; writes leaderboard_tags.csv,
foldscores_tags.json, and per_tag_f1.csv (champion, on the locked test).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import ComplementNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import ClassifierChain
from lightgbm import LGBMClassifier

from common import CFG, SEED, rp, dump_json, parse_tags
from engine import cv_multilabel, locked_multilabel
from features import canon
import metrics as M

LOGREG_C = CFG["grids"]["logreg_C"]
LINSVC_C = CFG["grids"]["linsvc_C"]


# ------------------------------------------------------------ score helpers
def _scores(model, Xev):
    if hasattr(model, "predict_proba"):
        try:
            P = model.predict_proba(Xev)
            return np.asarray(P, float)
        except Exception:
            pass
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(Xev), float)
    return None


def _tune_ovr_C(make, grid, Xtr, Ytr):
    """Pick C by a quick 80/20 random split maximizing micro-F1 (rows, not groups —
    tags cuts have far fewer duplicate rows, and this only sets a hyperparameter)."""
    rng = np.random.RandomState(SEED)
    n = Xtr.shape[0]; idx = rng.permutation(n); cut = int(0.8 * n)
    a, b = idx[:cut], idx[cut:]
    Xa, Xb = canon(Xtr[a]), canon(Xtr[b])
    best_c, best_f = grid[0], -1
    for c in grid:
        clf = OneVsRestClassifier(make(c), n_jobs=-1).fit(Xa, Ytr[a])
        f = M.f1_score(Ytr[b], clf.predict(Xb), average="micro", zero_division=0)
        if f > best_f:
            best_f, best_c = f, c
    return best_c


# ------------------------------------------------------------ fit_predict fns
def fp_baseline_top3(Xtr, Ytr, Xev):
    freq = Ytr.mean(axis=0)
    top = np.argsort(freq)[::-1][:3]
    P = np.zeros((Xev.shape[0], Ytr.shape[1]), int)
    P[:, top] = 1
    S = np.tile(freq, (Xev.shape[0], 1))
    return P, S


def fp_logreg(Xtr, Ytr, Xev):
    c = _tune_ovr_C(lambda C: LogisticRegression(C=C, max_iter=300, solver="liblinear"), LOGREG_C, Xtr, Ytr)
    clf = OneVsRestClassifier(LogisticRegression(C=c, max_iter=300, solver="liblinear"), n_jobs=-1).fit(Xtr, Ytr)
    return clf.predict(Xev), _scores(clf, Xev)


def fp_logreg_balanced(Xtr, Ytr, Xev):
    clf = OneVsRestClassifier(LogisticRegression(C=3, max_iter=300, solver="liblinear",
                              class_weight="balanced"), n_jobs=-1).fit(Xtr, Ytr)
    return clf.predict(Xev), _scores(clf, Xev)


def fp_linsvc(Xtr, Ytr, Xev):
    c = _tune_ovr_C(lambda C: LinearSVC(C=C), LINSVC_C, Xtr, Ytr)
    clf = OneVsRestClassifier(LinearSVC(C=c), n_jobs=-1).fit(Xtr, Ytr)
    return clf.predict(Xev), _scores(clf, Xev)


def fp_cnb(Xtr, Ytr, Xev):
    clf = OneVsRestClassifier(ComplementNB(alpha=0.3), n_jobs=-1).fit(Xtr, Ytr)
    return clf.predict(Xev), _scores(clf, Xev)


def fp_knn(Xtr, Ytr, Xev):
    clf = KNeighborsClassifier(n_neighbors=30, weights="distance").fit(Xtr, Ytr)
    P = clf.predict(Xev)
    try:
        S = np.asarray(clf.predict_proba(Xev))
        if S.ndim == 3:  # list-of-arrays shape guard
            S = S[:, :, 1].T
    except Exception:
        S = None
    return P, S


def fp_rf(Xtr, Ytr, Xev):
    clf = RandomForestClassifier(n_estimators=300, max_features="sqrt", n_jobs=-1,
                                 random_state=SEED).fit(Xtr, Ytr)
    P = clf.predict(Xev)
    proba = clf.predict_proba(Xev)  # list (per label) of (n,2)
    S = np.column_stack([p[:, 1] if p.shape[1] == 2 else np.zeros(Xev.shape[0]) for p in proba])
    return P, S


def fp_lightgbm(Xtr, Ytr, Xev):
    base = LGBMClassifier(n_estimators=200, learning_rate=0.1, num_leaves=31,
                          n_jobs=-1, verbose=-1, random_state=SEED)
    clf = OneVsRestClassifier(base, n_jobs=-1).fit(Xtr, Ytr)
    return clf.predict(Xev), _scores(clf, Xev)


def fp_chain_logreg(Xtr, Ytr, Xev):
    clf = ClassifierChain(LogisticRegression(C=3, max_iter=300, solver="liblinear"),
                          order="random", random_state=SEED).fit(Xtr, Ytr)
    S = np.asarray(clf.predict_proba(Xev), float)
    return (S >= 0.5).astype(int), S


def fp_chain_linsvc(Xtr, Ytr, Xev):
    clf = ClassifierChain(LinearSVC(C=1), order="random", random_state=SEED).fit(Xtr, Ytr)
    S = np.asarray(clf.decision_function(Xev), float)
    return (S >= 0.0).astype(int), S


def roster():
    return [
        {"name": "Baseline_top3", "family": "Baseline",   "feat": "stats",         "fit_predict": fp_baseline_top3},
        {"name": "ComplementNB",  "family": "NaiveBayes",  "feat": "tfidf_word_12", "fit_predict": fp_cnb},
        {"name": "LogReg",        "family": "Linear",      "feat": "tfidf_word_12", "fit_predict": fp_logreg},
        {"name": "LogReg_bal",    "family": "Linear",      "feat": "tfidf_word_12", "fit_predict": fp_logreg_balanced},
        {"name": "LinearSVC",     "family": "Linear",      "feat": "tfidf_word_12", "fit_predict": fp_linsvc},
        {"name": "KNN",           "family": "Instance",    "feat": "lsa+stats",     "fit_predict": fp_knn},
        {"name": "RandomForest",  "family": "Trees",       "feat": "lsa+stats",     "fit_predict": fp_rf},
        {"name": "LightGBM",      "family": "Boosting",    "feat": "lsa+stats",     "fit_predict": fp_lightgbm},
        {"name": "Chain_LogReg",  "family": "Chain",       "feat": "tfidf_word_12", "fit_predict": fp_chain_logreg},
        {"name": "Chain_LinSVC",  "family": "Chain",       "feat": "tfidf_word_12", "fit_predict": fp_chain_linsvc},
    ]


def run(df, splits, mlb, verbose=True):
    results, rows, foldscores = {}, [], {}
    champion, champ_f1 = None, -1
    for spec in roster():
        cv = cv_multilabel(spec, "tags", df, splits, mlb)
        locked, Ypred, Yscore, Yte = locked_multilabel(spec, "tags", df, splits, mlb)
        results[spec["name"]] = {"cv": cv, "locked": locked, "spec": spec,
                                 "locked_pred": Ypred, "locked_true": Yte}
        foldscores[spec["name"]] = cv["fold_primary"]
        a = cv["agg"]
        row = {"model": spec["name"], "family": spec["family"], "feature_set": spec["feat"]}
        for k, v in a.items():
            row[f"{k}_mean"] = round(v["mean"], 4); row[f"{k}_std"] = round(v["std"], 4)
        row["f1_macro_locked"] = round(locked["f1_macro"], 4)
        row["f1_micro_locked"] = round(locked["f1_micro"], 4)
        row["cv_time_s"] = cv["time"]["cv_fit_predict_s"]
        rows.append(row)
        if locked["f1_micro"] > champ_f1 and spec["family"] != "Baseline":
            champ_f1, champion = locked["f1_micro"], spec["name"]
        if verbose:
            print(f"  {spec['name']:14s} CV microF1={a['f1_micro']['mean']:.4f}±{a['f1_micro']['std']:.4f} "
                  f"macroF1={a['f1_macro']['mean']:.4f} | locked micro={locked['f1_micro']:.4f} "
                  f"macro={locked['f1_macro']:.4f} ({cv['time']['cv_fit_predict_s']}s)")
    lb = pd.DataFrame(rows).sort_values("f1_micro_mean", ascending=False).reset_index(drop=True)
    lb.to_csv(rp(CFG["paths"]["results_dir"], "leaderboard_tags.csv"), index=False)
    dump_json(foldscores, rp(CFG["paths"]["results_dir"], "foldscores_tags.json"))

    # per-tag F1 for the champion on the locked test
    ch = results[champion]
    ptf1 = M.per_tag_f1(ch["locked_true"], ch["locked_pred"], list(mlb.classes_))
    pd.DataFrame([{"tag": t, **d} for t, d in ptf1.items()]).sort_values(
        "support", ascending=False).to_csv(rp(CFG["paths"]["results_dir"], "per_tag_f1.csv"), index=False)
    return results, lb, champion


if __name__ == "__main__":
    from common import load_task, seed_everything
    from make_splits import load_splits
    from sklearn.preprocessing import MultiLabelBinarizer
    seed_everything()
    df = load_task("tags"); splits = load_splits("tags")
    mlb = MultiLabelBinarizer().fit(parse_tags(df["tags_norm"]))
    print(f"TAGS roster ({len(mlb.classes_)} labels; CV mean±std over 5 grouped folds, + locked test):")
    _, lb, champ = run(df, splits, mlb)
    print(f"\nChampion (locked micro-F1): {champ}")
    print(lb[["model", "family", "feature_set", "f1_micro_mean", "f1_macro_mean",
              "f1_micro_locked", "f1_macro_locked"]].to_string(index=False))
