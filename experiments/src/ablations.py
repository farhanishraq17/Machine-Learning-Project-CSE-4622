# -*- coding: utf-8 -*-
"""Phase 8 — ablation studies (§8): what actually helps.

  1. feature-family lift  : fixed model across {stats, tfidf, tfidf+stats, lsa+stats, unified}
  2. TF-IDF hyperparams   : word-1 vs word-1+2 vs char-3-5 (min_df/max_features from config)
  3. Word2Vec params      : vector_size {100,200,300} x sg{skip-gram,cbow} x pooling{mean,tfidf}
                            (rating, fold-0, quick Ridge probe — retraining w2v is the cost)
  4. tag strategy         : Binary-Relevance vs Classifier-Chain vs per-label thresholding
Outputs: results/ablation_*.csv.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.metrics import mean_absolute_error
from gensim.models import Word2Vec

from common import CFG, SEED, rp, load_task, seed_everything, parse_tags
from make_splits import load_splits
from features import get_feature_set, canon
from engine import cv_regression, cv_multilabel
import metrics as M
import models_rating as MR
import models_tags as MT

RES = CFG["paths"]["results_dir"]


# ------------------------------------------------------- 1. feature-family lift
def feature_family_ablation():
    df = load_task("rating"); s = load_splits("rating")
    rows = []
    combos = ["stats", "tfidf_word_12", "tfidf+stats", "lsa+stats", "unified"]
    # Ridge = linear representative (likes sparse TF-IDF); LightGBM = boosting
    # representative that ingests sparse natively — HistGBR needs dense input, so on the
    # 30k-wide TF-IDF sets it is both slow and memory-heavy and adds no insight the
    # champion LightGBM-unified doesn't already give.
    for feat in combos:
        for mname, fp in [("Ridge", MR.fp_ridge), ("LightGBM", MR.fp_lightgbm)]:
            spec = {"name": mname, "feat": feat, "fit_predict": fp}
            cv = cv_regression(spec, "rating", df, s)
            a = cv["agg"]
            rows.append({"model": mname, "feature_set": feat,
                         "MAE_mean": round(a["MAE"]["mean"], 2), "MAE_std": round(a["MAE"]["std"], 2),
                         "R2_mean": round(a["R2"]["mean"], 4)})
            print(f"  [feat-family] {mname:8s} {feat:14s} MAE={a['MAE']['mean']:.2f}")
    pd.DataFrame(rows).to_csv(rp(RES, "ablation_feature_family.csv"), index=False)


# ------------------------------------------------------- 2. TF-IDF hyperparams
def tfidf_ablation():
    rows = []
    # rating (Ridge) and tags (LinearSVC) over the 3 tfidf variants
    dfr, sr = load_task("rating"), load_splits("rating")
    for feat in ["tfidf_word_1", "tfidf_word_12", "tfidf_char_35"]:
        spec = {"name": "Ridge", "feat": feat, "fit_predict": MR.fp_ridge}
        a = cv_regression(spec, "rating", dfr, sr)["agg"]
        rows.append({"task": "rating", "vectorizer": feat, "metric": "MAE", "value": round(a["MAE"]["mean"], 2)})
        print(f"  [tfidf] rating Ridge {feat:14s} MAE={a['MAE']['mean']:.2f}")
    dft, st = load_task("tags"), load_splits("tags")
    mlb = MultiLabelBinarizer().fit(parse_tags(dft["tags_norm"]))
    for feat in ["tfidf_word_1", "tfidf_word_12", "tfidf_char_35"]:
        spec = {"name": "LinearSVC", "feat": feat, "fit_predict": MT.fp_linsvc}
        a = cv_multilabel(spec, "tags", dft, st, mlb)["agg"]
        rows.append({"task": "tags", "vectorizer": feat, "metric": "f1_micro", "value": round(a["f1_micro"]["mean"], 4)})
        rows.append({"task": "tags", "vectorizer": feat, "metric": "f1_macro", "value": round(a["f1_macro"]["mean"], 4)})
        print(f"  [tfidf] tags LinearSVC {feat:14s} microF1={a['f1_micro']['mean']:.4f} macroF1={a['f1_macro']['mean']:.4f}")
    pd.DataFrame(rows).to_csv(rp(RES, "ablation_tfidf.csv"), index=False)


# ------------------------------------------------------- 3. Word2Vec params (fold-0 probe)
def w2v_ablation():
    df = load_task("rating"); s = load_splits("rating")
    ftr, fva = s["folds"][0]
    ytr = df.iloc[ftr]["rating"].to_numpy(float); yva = df.iloc[fva]["rating"].to_numpy(float)
    tok_tr = df.iloc[ftr]["tokens"].tolist(); tok_va = df.iloc[fva]["tokens"].tolist()
    # idf for tfidf-weighted pooling
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(min_df=5).fit(df.iloc[ftr][CFG["columns"]["text_proc"]])
    idf = {w: vec.idf_[j] for w, j in vec.vocabulary_.items()}

    def pool(model, toks, weighted):
        dim = model.wv.vector_size; out = np.zeros((len(toks), dim), np.float32)
        dflt = float(np.mean(list(idf.values())))
        for i, ts in enumerate(toks):
            vs, ws = [], []
            for t in ts:
                if t in model.wv.key_to_index:
                    vs.append(model.wv[t]); ws.append(idf.get(t, dflt) if weighted else 1.0)
            if vs:
                w = np.asarray(ws, np.float32)
                out[i] = (np.asarray(vs, np.float32) * w[:, None]).sum(0) / (w.sum() + 1e-9)
        return out

    rows = []
    for size in [100, 200, 300]:
        for sg in [1, 0]:
            m = Word2Vec(tok_tr, vector_size=size, window=5, min_count=5, sg=sg, epochs=5, workers=1, seed=SEED)
            for weighted in [False, True]:
                Xtr = pool(m, tok_tr, weighted); Xva = pool(m, tok_va, weighted)
                sc = StandardScaler().fit(Xtr)
                rg = Ridge(alpha=10).fit(sc.transform(Xtr), ytr)
                mae = mean_absolute_error(yva, rg.predict(sc.transform(Xva)))
                rows.append({"vector_size": size, "algo": "skip-gram" if sg else "cbow",
                             "pooling": "tfidf" if weighted else "mean", "fold0_MAE": round(mae, 2)})
                print(f"  [w2v] size={size} {'sg' if sg else 'cbow'} pool={'tfidf' if weighted else 'mean'} MAE={mae:.2f}")
    pd.DataFrame(rows).to_csv(rp(RES, "ablation_w2v.csv"), index=False)


# ------------------------------------------------- 4b. keyword flags + query_count (tags)
def keyword_flags_ablation():
    """P3 — do deterministic algorithm-keyword flags + query_count lift the tag champion?"""
    df = load_task("tags"); s = load_splits("tags")
    mlb = MultiLabelBinarizer().fit(parse_tags(df["tags_norm"]))
    rows = []
    for feat in ["tfidf_word_12", "tfidf+kw"]:
        spec = {"name": "LogReg_bal", "feat": feat, "fit_predict": MT.fp_logreg_balanced}
        a = cv_multilabel(spec, "tags", df, s, mlb)["agg"]
        rows.append({"model": "LogReg_bal", "feature_set": feat,
                     "f1_micro": round(a["f1_micro"]["mean"], 4),
                     "f1_macro": round(a["f1_macro"]["mean"], 4),
                     "recall_micro": round(a["recall_micro"]["mean"], 4)})
        print(f"  [keywords] LogReg_bal {feat:14s} microF1={a['f1_micro']['mean']:.4f} "
              f"macroF1={a['f1_macro']['mean']:.4f}")
    pd.DataFrame(rows).to_csv(rp(RES, "ablation_keyword_flags.csv"), index=False)


# ------------------------------------------------------- 4. tag strategy
def fp_linsvc_thresholded(Xtr, Ytr, Xev):
    rng = np.random.RandomState(SEED)
    n = Xtr.shape[0]; idx = rng.permutation(n); cut = int(0.8 * n)
    a, b = idx[:cut], idx[cut:]
    Xa, Xb = canon(Xtr[a]), canon(Xtr[b])
    val_clf = OneVsRestClassifier(LinearSVC(C=1), n_jobs=-1).fit(Xa, Ytr[a])
    thr = M.tune_label_thresholds(Ytr[b], val_clf.decision_function(Xb),
                                  steps=CFG["tags"]["threshold_grid_steps"])
    clf = OneVsRestClassifier(LinearSVC(C=1), n_jobs=-1).fit(Xtr, Ytr)
    S = clf.decision_function(Xev)
    return M.apply_thresholds(S, thr), S


def tag_strategy_ablation():
    df = load_task("tags"); s = load_splits("tags")
    mlb = MultiLabelBinarizer().fit(parse_tags(df["tags_norm"]))
    rows = []
    strat = [("Binary-Relevance", MT.fp_linsvc), ("Classifier-Chain", MT.fp_chain_linsvc),
             ("Per-label-threshold", fp_linsvc_thresholded)]
    for name, fp in strat:
        spec = {"name": name, "feat": "tfidf_word_12", "fit_predict": fp}
        a = cv_multilabel(spec, "tags", df, s, mlb)["agg"]
        rows.append({"strategy": name, "base": "LinearSVC",
                     "f1_micro": round(a["f1_micro"]["mean"], 4), "f1_macro": round(a["f1_macro"]["mean"], 4),
                     "recall_micro": round(a["recall_micro"]["mean"], 4)})
        print(f"  [strategy] {name:20s} microF1={a['f1_micro']['mean']:.4f} macroF1={a['f1_macro']['mean']:.4f}")
    pd.DataFrame(rows).to_csv(rp(RES, "ablation_tag_strategy.csv"), index=False)


if __name__ == "__main__":
    seed_everything()
    print("1) feature-family ablation"); feature_family_ablation()
    print("2) tfidf ablation"); tfidf_ablation()
    print("3) w2v ablation"); w2v_ablation()
    print("4) tag-strategy ablation"); tag_strategy_ablation()
    print("5) keyword-flags ablation"); keyword_flags_ablation()
