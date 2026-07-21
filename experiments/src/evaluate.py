# -*- coding: utf-8 -*-
"""Phase 6 — comparison & significance (§6).

Consumes foldscores_{task}.json (per-fold primary scores per model, written by the
rosters) and the leaderboards. Produces:
  * paired Wilcoxon signed-rank between the top-2 models (shared folds)
  * Friedman omnibus test across the whole roster
  * Nemenyi post-hoc matrix + a critical-difference diagram (scikit-posthocs)
  * ranking-disagreement flags (primary vs secondary metric orderings)
Outputs: results/significance_{task}.json, figures/cd_{task}.png.

Note: with 5 folds the omnibus tests have low power; treat as directional evidence,
reported alongside the effect sizes in the leaderboard.
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, friedmanchisquare, rankdata
import scikit_posthocs as sp

from common import CFG, rp, dump_json


def _load_foldscores(task):
    with open(rp(CFG["paths"]["results_dir"], f"foldscores_{task}.json"), encoding="utf-8") as f:
        return json.load(f)


def significance(task, lower_better, exclude=("Dummy_mean", "Baseline_top3")):
    fs = _load_foldscores(task)
    models = [m for m in fs if m not in exclude]
    scores = {m: np.asarray(fs[m], float) for m in models}
    n_folds = len(next(iter(scores.values())))

    # goodness matrix (higher = better), folds x models
    good = np.column_stack([(-scores[m] if lower_better else scores[m]) for m in models])
    # per-fold ranks (rank 1 = best)
    ranks = np.column_stack([rankdata(-good[i]) for i in range(n_folds)]).T  # folds x models
    avg_rank = pd.Series(ranks.mean(axis=0), index=models).sort_values()

    # order models best->worst by mean goodness
    mean_good = pd.Series(good.mean(axis=0), index=models).sort_values(ascending=False)
    order = mean_good.index.tolist()

    out = {"task": task, "n_folds": n_folds, "models_ranked": order,
           "avg_rank": avg_rank.round(3).to_dict()}

    # Friedman omnibus
    try:
        stat, p = friedmanchisquare(*[good[:, j] for j in range(len(models))])
        out["friedman"] = {"stat": float(stat), "p": float(p)}
    except Exception as e:
        out["friedman"] = {"error": str(e)}

    # paired Wilcoxon between top-2
    if len(order) >= 2:
        a, b = order[0], order[1]
        try:
            w, pw = wilcoxon(scores[a], scores[b])
            out["wilcoxon_top2"] = {"a": a, "b": b, "stat": float(w), "p": float(pw),
                                    f"{a}_mean": float(scores[a].mean()), f"{b}_mean": float(scores[b].mean())}
        except Exception as e:
            out["wilcoxon_top2"] = {"a": a, "b": b, "error": str(e)}

    # Nemenyi post-hoc + CD diagram
    try:
        nem = sp.posthoc_nemenyi_friedman(good)   # folds x treatments
        nem.index = models; nem.columns = models
        out["nemenyi_min_p"] = float(np.nanmin(nem.values[~np.eye(len(models), dtype=bool)]))
        _cd_diagram(task, avg_rank, nem)
        out["cd_diagram"] = f"figures/cd_{task}.png"
    except Exception as e:
        out["nemenyi_error"] = str(e)

    dump_json(out, rp(CFG["paths"]["results_dir"], f"significance_{task}.json"))
    return out


def _cd_diagram(task, avg_rank, nem):
    plt.figure(figsize=(9, 2.6))
    sp.critical_difference_diagram(avg_rank, nem,
                                   label_fmt_left="{label}  ({rank:.2f})  ",
                                   label_fmt_right="  ({rank:.2f})  {label}")
    plt.title(f"Critical-difference diagram — {task} "
              f"({'lower MAE better' if task == 'rating' else 'higher macro-F1 better'})")
    plt.tight_layout()
    plt.savefig(rp(CFG["paths"]["figures_dir"], f"cd_{task}.png"), dpi=130)
    plt.close()


def ranking_disagreements(task, primary, others):
    lb = pd.read_csv(rp(CFG["paths"]["results_dir"], f"leaderboard_{task}.csv"))
    asc = task == "rating"
    base = lb.sort_values(primary, ascending=asc)["model"].tolist()
    flags = []
    for metric in others:
        if metric not in lb.columns:
            continue
        asc_o = metric.startswith(("MAE", "RMSE", "MedianAE", "sMAPE", "hamming"))
        order = lb.sort_values(metric, ascending=asc_o)["model"].tolist()
        if order[0] != base[0]:
            flags.append({"metric": metric, "primary_top": base[0], "this_top": order[0]})
    return {"primary": primary, "primary_order": base, "disagreements": flags}


def run_all():
    res = {}
    res["rating"] = {"significance": significance("rating", lower_better=True),
                     "disagreements": ranking_disagreements("rating", "MAE_mean",
                                     ["RMSE_mean", "R2_mean", "Spearman_mean", "acc_100_mean", "acc_200_mean"])}
    res["tags"] = {"significance": significance("tags", lower_better=False),
                   "disagreements": ranking_disagreements("tags", "f1_micro_mean",
                                   ["f1_macro_mean", "precision_micro_mean", "recall_micro_mean", "jaccard_samples_mean"])}
    dump_json(res, rp(CFG["paths"]["results_dir"], "significance_summary.json"))
    for task in ("rating", "tags"):
        s = res[task]["significance"]
        print(f"[{task}] ranked: {s['models_ranked'][:4]}...  Friedman p={s.get('friedman', {}).get('p')}")
        print(f"        Wilcoxon top-2: {s.get('wilcoxon_top2', {}).get('a')} vs "
              f"{s.get('wilcoxon_top2', {}).get('b')} p={s.get('wilcoxon_top2', {}).get('p')}")
        print(f"        disagreements: {res[task]['disagreements']['disagreements']}")
    return res


if __name__ == "__main__":
    run_all()
