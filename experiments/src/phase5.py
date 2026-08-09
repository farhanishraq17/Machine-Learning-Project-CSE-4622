# -*- coding: utf-8 -*-
"""Phase 5 — imbalance & thresholds (tags, §5.4) + ordinal framing (rating, §4.4).

Tags:
  * per-label decision-threshold tuning on a grouped validation split within train,
    applied to the locked test -> macro-F1 lift over the 0.5-threshold champion.
  * report all-38 tags vs a >=15-train-example floor; *special reported separately.
  * skmultilearn iterative-stratification sanity check (row-level micro-F1 vs grouped).
Rating:
  * bin rating into ordinal bands, train a classifier, compare band-accuracy (==band)
    and within-one-band accuracy against the regression champion's bucketed predictions.
Outputs: results/phase5_tags_thresholds.json, results/phase5_ordinal_rating.json.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LogisticRegression as LR
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer

from common import CFG, SEED, rp, dump_json, load_task, seed_everything, parse_tags
from make_splits import load_splits
from features import get_feature_set
import metrics as M


# ---------------------------------------------------------------- tags thresholds
def _champion_factory():
    """Roster champion from the tags leaderboard -> (name, estimator factory, base threshold).
    LogReg_bal scores are probabilities (base 0.5); LinearSVC margins (base 0.0)."""
    lb = pd.read_csv(rp(CFG["paths"]["results_dir"], "leaderboard_tags.csv"))
    name = lb[lb["family"] != "Baseline"].sort_values(
        "f1_micro_mean", ascending=False).iloc[0]["model"]
    if name == "LogReg_bal":
        return name, (lambda: OneVsRestClassifier(
            LogisticRegression(C=3, max_iter=300, solver="liblinear",
                               class_weight="balanced"), n_jobs=-1)), 0.5
    return "LinearSVC", (lambda: OneVsRestClassifier(LinearSVC(C=1), n_jobs=-1)), 0.0


def _scores(clf, X):
    return np.asarray(clf.predict_proba(X), float) if hasattr(clf, "predict_proba") \
        else np.asarray(clf.decision_function(X), float)


def tags_thresholds():
    df = load_task("tags"); s = load_splits("tags")
    tr, te = s["train_idx"], s["test_idx"]
    grp = df[CFG["split"]["group_col"]].to_numpy()
    mlb = MultiLabelBinarizer().fit(parse_tags(df["tags_norm"]))
    classes = list(mlb.classes_)

    # grouped inner val split within train for threshold calibration
    gss = GroupShuffleSplit(1, test_size=0.2, random_state=SEED)
    fit_rel, val_rel = next(gss.split(tr, groups=grp[tr]))
    fit_idx, val_idx = tr[fit_rel], tr[val_rel]

    Xfit, Xval = get_feature_set("tags", "p5_fit_val", df, fit_idx, val_idx, "tfidf_word_12")
    _, Xte = get_feature_set("tags", "locked", df, tr, te, "tfidf_word_12")  # cached; Xtr side unused
    Xtr_all, Xte2 = get_feature_set("tags", "locked", df, tr, te, "tfidf_word_12")

    Yfit = mlb.transform(parse_tags(df.iloc[fit_idx]["tags_norm"]))
    Yval = mlb.transform(parse_tags(df.iloc[val_idx]["tags_norm"]))
    Ytr = mlb.transform(parse_tags(df.iloc[tr]["tags_norm"]))
    Yte = mlb.transform(parse_tags(df.iloc[te]["tags_norm"]))

    # thresholds are applied to the ACTUAL roster champion (read from the leaderboard),
    # so the tuned locked-test number is a headline for the reported model, not a side study
    champ_name, make_clf, base_thr = _champion_factory()
    clf_val = make_clf().fit(Xfit, Yfit)
    Sval = _scores(clf_val, Xval)
    thr = M.tune_label_thresholds(Yval, Sval, steps=CFG["tags"]["threshold_grid_steps"])

    clf = make_clf().fit(Xtr_all, Ytr)
    Ste = _scores(clf, Xte2)
    Ypred_05 = (Ste >= base_thr).astype(int)
    Ypred_tuned = M.apply_thresholds(Ste, thr)

    base = M.tag_metrics(Yte, Ypred_05, Ste)
    tuned = M.tag_metrics(Yte, Ypred_tuned, Ste)

    # rare-tag floor (>=15 train examples) and *special split
    train_support = Ytr.sum(axis=0)
    keep = np.array([train_support[j] >= CFG["tags"]["rare_floor"] for j in range(len(classes))])
    special_j = classes.index(CFG["tags"]["special_tag"]) if CFG["tags"]["special_tag"] in classes else None

    def macro_on(mask, Yp):
        cols = np.where(mask)[0]
        return float(M.f1_score(Yte[:, cols], Yp[:, cols], average="macro", zero_division=0))

    out = {
        "champion": f"{champ_name} (OvR, tfidf_word_12)",
        "threshold_0.5": {k: round(base[k], 4) for k in ["f1_micro", "f1_macro", "precision_micro", "recall_micro"]},
        "threshold_tuned": {k: round(tuned[k], 4) for k in ["f1_micro", "f1_macro", "precision_micro", "recall_micro"]},
        "macro_f1_lift": round(tuned["f1_macro"] - base["f1_macro"], 4),
        "macro_f1_all38": {"base": round(macro_on(np.ones(len(classes), bool), Ypred_05), 4),
                           "tuned": round(macro_on(np.ones(len(classes), bool), Ypred_tuned), 4)},
        "macro_f1_floor15": {"n_tags": int(keep.sum()),
                             "base": round(macro_on(keep, Ypred_05), 4),
                             "tuned": round(macro_on(keep, Ypred_tuned), 4)},
    }
    if special_j is not None:
        out["special_tag"] = {"tag": classes[special_j],
                              "f1_base": round(M.f1_score(Yte[:, special_j], Ypred_05[:, special_j], zero_division=0), 4),
                              "f1_tuned": round(M.f1_score(Yte[:, special_j], Ypred_tuned[:, special_j], zero_division=0), 4),
                              "train_support": int(train_support[special_j])}

    # skmultilearn iterative stratification sanity check (row-level, non-grouped)
    try:
        from skmultilearn.model_selection import iterative_train_test_split
        Xall, _ = get_feature_set("tags", "locked", df, tr, te, "tfidf_word_12")
        idx = np.arange(len(tr)).reshape(-1, 1)
        itr, ytr_i, ite, yte_i = iterative_train_test_split(idx, Ytr, test_size=0.2)
        clf_i = OneVsRestClassifier(LinearSVC(C=1), n_jobs=-1).fit(Xall[itr.ravel()], ytr_i)
        Pi = clf_i.predict(Xall[ite.ravel()])
        out["iterative_stratification_sanity"] = {
            "note": "row-level split (allows duplicate leakage) — reference only; grouped split preferred",
            "f1_micro": round(M.f1_score(yte_i, Pi, average="micro", zero_division=0), 4),
            "f1_macro": round(M.f1_score(yte_i, Pi, average="macro", zero_division=0), 4)}
    except Exception as e:
        out["iterative_stratification_sanity"] = {"skipped": str(e)}

    dump_json(out, rp(CFG["paths"]["results_dir"], "phase5_tags_thresholds.json"))
    print(f"[tags-thresh] macro-F1 {out['threshold_0.5']['f1_macro']} -> {out['threshold_tuned']['f1_macro']} "
          f"(lift {out['macro_f1_lift']}); micro {out['threshold_0.5']['f1_micro']} -> {out['threshold_tuned']['f1_micro']}")
    return out


# ---------------------------------------------------------------- ordinal rating
def ordinal_rating():
    import models_rating as MR
    from engine import locked_regression
    df = load_task("rating"); s = load_splits("rating")
    tr, te = s["train_idx"], s["test_idx"]
    bins = CFG["rating"]["ordinal_bins"]
    yb_tr = pd.cut(df.iloc[tr]["rating"], bins=bins, labels=False, include_lowest=True).to_numpy()
    yb_te = pd.cut(df.iloc[te]["rating"], bins=bins, labels=False, include_lowest=True).to_numpy()

    Xtr, Xte = get_feature_set("rating", "locked", df, tr, te, "tfidf+stats")
    clf = LogisticRegression(C=10, max_iter=400, solver="liblinear")
    from sklearn.multiclass import OneVsRestClassifier as OVR
    clf = OVR(clf, n_jobs=-1).fit(Xtr, yb_tr)
    pb = clf.predict(Xte)

    def exact(a, b): return float(np.mean(a == b))
    def within1(a, b): return float(np.mean(np.abs(a - b) <= 1))

    # regression champion bucketed into the same bands
    champ = pd.read_csv(rp(CFG["paths"]["results_dir"], "leaderboard_rating.csv"))
    champ = champ[~champ["family"].isin(["Baseline"])].sort_values("MAE_mean").iloc[0]["model"]
    spec = next(x for x in MR.roster() if x["name"] == champ)
    _, yhat, yte = locked_regression(spec, "rating", df, s)
    reg_band = pd.cut(pd.Series(yhat), bins=bins, labels=False, include_lowest=True).to_numpy()
    reg_band = np.where(np.isnan(reg_band), 0, reg_band).astype(int)

    out = {"n_bands": len(bins) - 1, "bins": bins,
           "ordinal_classifier": {"exact_band_acc": round(exact(pb, yb_te), 4),
                                  "within_one_band_acc": round(within1(pb, yb_te), 4)},
           "regression_champion": {"model": champ,
                                   "exact_band_acc": round(exact(reg_band, yb_te), 4),
                                   "within_one_band_acc": round(within1(reg_band, yb_te), 4),
                                   "note": "champion regressor predictions bucketed into the same bands"}}
    dump_json(out, rp(CFG["paths"]["results_dir"], "phase5_ordinal_rating.json"))
    print(f"[ordinal] classifier exact={out['ordinal_classifier']['exact_band_acc']} "
          f"within1={out['ordinal_classifier']['within_one_band_acc']} | "
          f"regressor({champ}) exact={out['regression_champion']['exact_band_acc']} "
          f"within1={out['regression_champion']['within_one_band_acc']}")
    return out


if __name__ == "__main__":
    seed_everything()
    tags_thresholds()
    ordinal_rating()
