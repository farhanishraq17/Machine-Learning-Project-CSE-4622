# -*- coding: utf-8 -*-
"""Phase 7 — error analysis & feature importance (§7).

Rating (champion from leaderboard, on the locked test):
  * residuals vs true rating and vs statement length  -> figures/rating_residuals.png
  * calibration by rating band (mean pred, MAE)        -> figures/rating_calibration.png
  * worst over/under predictions                       -> results/rating_worst_predictions.csv
  * interpretable Ridge token coefficients             -> results/rating_top_tokens.csv
  * permutation importance of the 18 stats             -> figures/rating_stats_importance.png

Tags (champion, on the locked test):
  * per-tag F1 vs support                              -> figures/tags_per_tag_f1.png
  * most-confused tag pairs, never-predicted tags      -> results/tags_confusion_pairs.csv, tags_never_predicted.json
  * top tokens per tag (LinearSVC coefficients)        -> results/tags_top_tokens_per_tag.csv
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer

from common import CFG, SEED, rp, dump_json, load_task, seed_everything, parse_tags
from make_splits import load_splits

FIG = CFG["paths"]["figures_dir"]
RES = CFG["paths"]["results_dir"]
NUM = CFG["columns"]["numeric"]
PROC = CFG["columns"]["text_proc"]


def _champion(task, primary_col):
    lb = pd.read_csv(rp(RES, f"leaderboard_{task}.csv"))
    lb = lb[~lb["family"].isin(["Baseline"])]
    asc = task == "rating"
    return lb.sort_values(primary_col, ascending=asc).iloc[0]["model"]


# --------------------------------------------------------------- rating
def rating_analysis():
    import models_rating as MR
    df = load_task("rating"); s = load_splits("rating")
    tr, te = s["train_idx"], s["test_idx"]
    champ = _champion("rating", "MAE_mean")
    spec = next(x for x in MR.roster() if x["name"] == champ)
    from engine import locked_regression
    locked, yhat, yte = locked_regression(spec, "rating", df, s)
    sub = df.iloc[te].copy()
    sub["pred"] = yhat; sub["resid"] = yhat - yte; sub["abs_err"] = np.abs(sub["resid"])

    # residual figures
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].scatter(yte, sub["resid"], s=6, alpha=.3)
    ax[0].axhline(0, color="k", lw=.8); ax[0].set_xlabel("true rating"); ax[0].set_ylabel("pred - true")
    ax[0].set_title(f"Residuals vs true rating — {champ} (locked MAE={locked['MAE']:.0f})")
    ax[1].scatter(sub["statement_length_words"], sub["resid"], s=6, alpha=.3)
    ax[1].axhline(0, color="k", lw=.8); ax[1].set_xlabel("statement length (words)"); ax[1].set_ylabel("pred - true")
    ax[1].set_title("Residuals vs statement length")
    plt.tight_layout(); plt.savefig(rp(FIG, "rating_residuals.png"), dpi=130); plt.close()

    # calibration by band
    bins = CFG["rating"]["ordinal_bins"]
    sub["band"] = pd.cut(yte, bins=bins, include_lowest=True)
    cal = sub.groupby("band", observed=True).agg(n=("pred", "size"), mean_true=("rating", "mean"),
                                                 mean_pred=("pred", "mean"), mae=("abs_err", "mean")).reset_index()
    cal.to_csv(rp(RES, "rating_calibration.csv"), index=False)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    x = np.arange(len(cal))
    ax[0].plot(cal["mean_true"], cal["mean_true"], "k--", lw=1, label="ideal")
    ax[0].plot(cal["mean_true"], cal["mean_pred"], "o-", label="model")
    ax[0].set_xlabel("mean true rating (band)"); ax[0].set_ylabel("mean predicted"); ax[0].legend()
    ax[0].set_title("Calibration by rating band (underprediction of hard problems)")
    ax[1].bar(x, cal["mae"]); ax[1].set_xticks(x)
    ax[1].set_xticklabels([str(b) for b in cal["band"]], rotation=60, ha="right", fontsize=7)
    ax[1].set_ylabel("MAE"); ax[1].set_title("MAE by rating band")
    plt.tight_layout(); plt.savefig(rp(FIG, "rating_calibration.png"), dpi=130); plt.close()

    # worst predictions
    cols = ["contestId", "index", "name", "rating", "pred", "resid", "statement_length_words"]
    worst = pd.concat([sub.nlargest(10, "resid")[cols], sub.nsmallest(10, "resid")[cols]])
    worst.to_csv(rp(RES, "rating_worst_predictions.csv"), index=False)

    # interpretable Ridge token coefficients (fit on locked train tfidf)
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=30000, sublinear_tf=True)
    Xtr = vec.fit_transform(df.iloc[tr][PROC]); names = np.array(vec.get_feature_names_out())
    rg = Ridge(alpha=10).fit(Xtr, df.iloc[tr]["rating"].to_numpy(float))
    coef = rg.coef_
    top_hard = names[np.argsort(coef)[::-1][:25]]; top_easy = names[np.argsort(coef)[:25]]
    pd.DataFrame({"harder_tokens": top_hard, "easier_tokens": top_easy}).to_csv(
        rp(RES, "rating_top_tokens.csv"), index=False)

    # permutation importance of the 18 stats (compact model)
    sc = StandardScaler(); Str = sc.fit_transform(df.iloc[tr][NUM].to_numpy(float))
    Ste = sc.transform(df.iloc[te][NUM].to_numpy(float))
    hg = HistGradientBoostingRegressor(loss="absolute_error", random_state=SEED, max_iter=400).fit(
        Str, df.iloc[tr]["rating"].to_numpy(float))
    pi = permutation_importance(hg, Ste, yte, n_repeats=8, random_state=SEED, scoring="neg_mean_absolute_error")
    imp = pd.Series(pi.importances_mean, index=NUM).sort_values()
    plt.figure(figsize=(7, 6)); imp.plot.barh()
    plt.xlabel("Δ MAE when permuted"); plt.title("Permutation importance — 18 structural features (stats-only model)")
    plt.tight_layout(); plt.savefig(rp(FIG, "rating_stats_importance.png"), dpi=130); plt.close()
    imp[::-1].to_csv(rp(RES, "rating_stats_importance.csv"))
    print(f"[rating] champion={champ}; residuals/calibration/importance/tokens written.")
    return champ


# --------------------------------------------------------------- tags
def tags_analysis():
    import models_tags as MT
    df = load_task("tags"); s = load_splits("tags")
    tr, te = s["train_idx"], s["test_idx"]
    mlb = MultiLabelBinarizer().fit(parse_tags(df["tags_norm"]))
    classes = list(mlb.classes_)
    champ = _champion("tags", "f1_micro_mean")
    spec = next(x for x in MT.roster() if x["name"] == champ)
    from engine import locked_multilabel
    locked, Ypred, Yscore, Yte = locked_multilabel(spec, "tags", df, s, mlb)

    # per-tag F1 vs support figure
    ptf1 = pd.read_csv(rp(RES, "per_tag_f1.csv")).sort_values("f1")
    fig, ax = plt.subplots(figsize=(8, 9))
    ax.barh(ptf1["tag"], ptf1["f1"])
    for i, (f, sup) in enumerate(zip(ptf1["f1"], ptf1["support"])):
        ax.text(min(f + .01, .95), i, f"n={sup}", va="center", fontsize=6)
    ax.set_xlabel("F1 (locked test)"); ax.set_title(f"Per-tag F1 vs support — {champ}")
    plt.tight_layout(); plt.savefig(rp(FIG, "tags_per_tag_f1.png"), dpi=130); plt.close()

    # never-predicted tags
    never = [classes[j] for j in range(len(classes)) if Ypred[:, j].sum() == 0]
    dump_json({"never_predicted": never, "count": len(never)}, rp(RES, "tags_never_predicted.json"))

    # most-confused pairs: for each true tag a missed (FN), which tags were predicted on that row
    conf = {}
    for i in range(Yte.shape[0]):
        true_j = set(np.where(Yte[i] == 1)[0]); pred_j = set(np.where(Ypred[i] == 1)[0])
        for a in true_j - pred_j:          # missed true tags
            for b in pred_j - true_j:      # wrongly predicted tags on same row
                conf[(a, b)] = conf.get((a, b), 0) + 1
    pairs = sorted(conf.items(), key=lambda kv: -kv[1])[:25]
    pd.DataFrame([{"true_tag": classes[a], "predicted_instead": classes[b], "count": c}
                  for (a, b), c in pairs]).to_csv(rp(RES, "tags_confusion_pairs.csv"), index=False)

    # top tokens per tag (LinearSVC OvR on locked-train tfidf)
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=30000, sublinear_tf=True)
    Xtr = vec.fit_transform(df.iloc[tr][PROC]); names = np.array(vec.get_feature_names_out())
    Ytr = mlb.transform(parse_tags(df.iloc[tr]["tags_norm"]))
    ovr = OneVsRestClassifier(LinearSVC(C=1), n_jobs=-1).fit(Xtr, Ytr)
    rows = []
    for j, est in enumerate(ovr.estimators_):
        top = names[np.argsort(est.coef_.ravel())[::-1][:12]]
        rows.append({"tag": classes[j], "top_tokens": ", ".join(top)})
    pd.DataFrame(rows).to_csv(rp(RES, "tags_top_tokens_per_tag.csv"), index=False)
    print(f"[tags] champion={champ}; per-tag/confusion/never/tokens written. never_predicted={len(never)}")
    return champ


if __name__ == "__main__":
    seed_everything()
    rating_analysis()
    tags_analysis()
