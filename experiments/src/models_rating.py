# -*- coding: utf-8 -*-
"""Phase 3-4 — rating regression roster (§4). One tuned model per family.

Each roster entry pairs a model with its preferred feature set (§3 note):
  Dummy(mean)   -> stats        Ridge         -> tfidf+stats
  KNN           -> lsa+stats     LinearSVR     -> lsa+stats (scaled)
  RandomForest  -> lsa+stats     HistGBR       -> lsa+stats
  LightGBM      -> unified       XGBoost       -> unified
  Stacking      -> lsa+stats (Ridge+RF+HistGBR -> Ridge meta)

run(df, splits) -> {name: cv_result}; also fits each on full train for the locked test.
Writes results/leaderboard_rating.csv and results/foldscores_rating.json (for significance).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import LinearSVR
from sklearn.ensemble import (RandomForestRegressor, HistGradientBoostingRegressor,
                              StackingRegressor)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb
import xgboost as xgb

from common import CFG, SEED, rp, dump_json
from engine import cv_regression, locked_regression, tune_scalar, inner_group_holdout

G = CFG["grids"]


# ------------------------------------------------------------ fit_predict fns
def fp_dummy(Xtr, ytr, gtr, Xev):
    return np.full(Xev.shape[0], ytr.mean(), float)


def fp_ridge(Xtr, ytr, gtr, Xev):
    a = tune_scalar(lambda p: Ridge(alpha=p), G["ridge_alpha"], Xtr, ytr, gtr,
                    scorer=mean_absolute_error, maximize=False)
    m = Ridge(alpha=a).fit(Xtr, ytr)
    return m.predict(Xev)


def _scale(Xtr, Xev):
    sc = StandardScaler(with_mean=not sp.issparse(Xtr))
    return sc.fit_transform(Xtr), sc.transform(Xev)


def fp_knn(Xtr, ytr, gtr, Xev):
    Xtr2, Xev2 = _scale(Xtr, Xev)
    k = tune_scalar(lambda p: KNeighborsRegressor(n_neighbors=p, weights="distance"),
                    G["knn_k"], Xtr2, ytr, gtr, scorer=mean_absolute_error, maximize=False)
    return KNeighborsRegressor(n_neighbors=k, weights="distance").fit(Xtr2, ytr).predict(Xev2)


def fp_linsvr(Xtr, ytr, gtr, Xev):
    Xtr2, Xev2 = _scale(Xtr, Xev)
    c = tune_scalar(lambda p: LinearSVR(C=p, random_state=SEED, max_iter=5000),
                    G["linsvr_C"], Xtr2, ytr, gtr, scorer=mean_absolute_error, maximize=False)
    return LinearSVR(C=c, random_state=SEED, max_iter=5000).fit(Xtr2, ytr).predict(Xev2)


def fp_rf(Xtr, ytr, gtr, Xev):
    m = RandomForestRegressor(n_estimators=400, max_features="sqrt", n_jobs=-1,
                              random_state=SEED).fit(Xtr, ytr)
    return m.predict(Xev)


def fp_histgbr(Xtr, ytr, gtr, Xev):
    # HistGradientBoosting requires dense input; the roster pairs it with the dense
    # lsa+stats block, but the feature-family ablation feeds it sparse sets too.
    if sp.issparse(Xtr):
        Xtr = Xtr.toarray().astype(np.float32, copy=False)
        Xev = Xev.toarray().astype(np.float32, copy=False)
    m = HistGradientBoostingRegressor(loss="absolute_error", learning_rate=0.05,
                                      max_iter=1500, early_stopping=True, validation_fraction=0.12,
                                      n_iter_no_change=40, random_state=SEED).fit(Xtr, ytr)
    return m.predict(Xev)


def fp_lightgbm(Xtr, ytr, gtr, Xev):
    fit_rel, val_rel = inner_group_holdout(gtr, frac=0.15, seed=1)
    p = G["lgbm_rating"]
    params = dict(objective="regression_l1", metric="mae", learning_rate=p["learning_rate"],
                  num_leaves=p["num_leaves"], feature_fraction=0.7, bagging_fraction=0.8,
                  bagging_freq=1, min_child_samples=20, verbose=-1, seed=SEED, num_threads=-1)
    dtr = lgb.Dataset(Xtr[fit_rel], ytr[fit_rel])
    dva = lgb.Dataset(Xtr[val_rel], ytr[val_rel])
    m = lgb.train(params, dtr, p["n_estimators"], valid_sets=[dva],
                  callbacks=[lgb.early_stopping(p["early_stopping"], verbose=False)])
    return m.predict(Xev)


def fp_xgboost(Xtr, ytr, gtr, Xev):
    # paired with the dense lsa+stats block (fast); LightGBM covers sparse-unified boosting
    fit_rel, val_rel = inner_group_holdout(gtr, frac=0.15, seed=1)
    m = xgb.XGBRegressor(objective="reg:absoluteerror", n_estimators=1500, learning_rate=0.05,
                         max_depth=6, subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                         tree_method="hist", random_state=SEED, n_jobs=-1,
                         early_stopping_rounds=50, eval_metric="mae")
    m.fit(Xtr[fit_rel], ytr[fit_rel], eval_set=[(Xtr[val_rel], ytr[val_rel])], verbose=False)
    return m.predict(Xev)


def fp_stacking(Xtr, ytr, gtr, Xev):
    est = [("ridge", Ridge(alpha=10)),
           ("rf", RandomForestRegressor(n_estimators=300, max_features="sqrt", n_jobs=-1, random_state=SEED)),
           ("hgb", HistGradientBoostingRegressor(loss="absolute_error", learning_rate=0.06,
                                                  max_iter=600, random_state=SEED))]
    m = StackingRegressor(estimators=est, final_estimator=Ridge(alpha=1.0), n_jobs=-1, cv=3)
    return m.fit(Xtr, ytr).predict(Xev)


def roster():
    return [
        {"name": "Dummy_mean",   "family": "Baseline", "feat": "stats",       "fit_predict": fp_dummy},
        {"name": "Ridge",        "family": "Linear",   "feat": "tfidf+stats", "fit_predict": fp_ridge},
        {"name": "KNN",          "family": "Instance", "feat": "lsa+stats",   "fit_predict": fp_knn},
        {"name": "LinearSVR",    "family": "Kernel",   "feat": "lsa+stats",   "fit_predict": fp_linsvr},
        {"name": "RandomForest", "family": "Trees",    "feat": "lsa+stats",   "fit_predict": fp_rf},
        {"name": "HistGBR",      "family": "Boosting", "feat": "lsa+stats",   "fit_predict": fp_histgbr},
        {"name": "LightGBM",     "family": "Boosting", "feat": "unified",     "fit_predict": fp_lightgbm},
        {"name": "XGBoost",      "family": "Boosting", "feat": "lsa+stats",   "fit_predict": fp_xgboost},
        {"name": "Stacking",     "family": "Ensemble", "feat": "lsa+stats",   "fit_predict": fp_stacking},
    ]


def run(df, splits, verbose=True):
    results, rows, foldscores = {}, [], {}
    for spec in roster():
        cv = cv_regression(spec, "rating", df, splits)
        locked, yhat, yte = locked_regression(spec, "rating", df, splits)
        results[spec["name"]] = {"cv": cv, "locked": locked, "spec": spec}
        foldscores[spec["name"]] = cv["fold_primary"]
        a = cv["agg"]
        row = {"model": spec["name"], "family": spec["family"], "feature_set": spec["feat"]}
        for k, v in a.items():
            row[f"{k}_mean"] = round(v["mean"], 4); row[f"{k}_std"] = round(v["std"], 4)
        row["MAE_locked"] = round(locked["MAE"], 2)
        row["R2_locked"] = round(locked["R2"], 4)
        row["cv_time_s"] = cv["time"]["cv_fit_predict_s"]
        rows.append(row)
        if verbose:
            print(f"  {spec['name']:13s} CV MAE={a['MAE']['mean']:7.2f}±{a['MAE']['std']:5.2f} "
                  f"R2={a['R2']['mean']:.3f} | locked MAE={locked['MAE']:7.2f} "
                  f"({cv['time']['cv_fit_predict_s']}s)")
    lb = pd.DataFrame(rows).sort_values("MAE_mean").reset_index(drop=True)
    lb.to_csv(rp(CFG["paths"]["results_dir"], "leaderboard_rating.csv"), index=False)
    dump_json(foldscores, rp(CFG["paths"]["results_dir"], "foldscores_rating.json"))
    return results, lb


if __name__ == "__main__":
    from common import load_task, seed_everything
    from make_splits import load_splits
    seed_everything()
    df = load_task("rating"); splits = load_splits("rating")
    print("RATING roster (CV mean±std over 5 grouped folds, + locked test):")
    _, lb = run(df, splits)
    print("\nLeaderboard (ranked by CV MAE):")
    print(lb[["model", "family", "feature_set", "MAE_mean", "MAE_std", "R2_mean", "MAE_locked"]].to_string(index=False))
