# -*- coding: utf-8 -*-
"""Phase 2 — feature blocks (§3). Fit-on-train-only, cached per split.

Public API
----------
get_feature_set(task, split_id, df, train_pos, eval_pos, name, use_cache=True)
    -> (X_train, X_eval)   assembled feature matrices for a named feature set.

Named feature sets (combos of atomic blocks):
    stats · tfidf_word_12 · tfidf_word_1 · tfidf_char_35 · lsa_300
    w2v_mean · w2v_tfidf_mean · tfidf+stats · unified · lsa+stats

Every transformer (TfidfVectorizer, TruncatedSVD, Word2Vec, StandardScaler) is
fit strictly on df.iloc[train_pos] and applied to both train and eval rows, so
there is no information leak from the evaluation rows. Word2Vec uses workers=1 for
determinism (seed 42). Assembled matrices are cached to results/cache keyed by
(task, split_id, feature_set) so models and ablations reuse them.
"""
from __future__ import annotations
import os
import numpy as np
import scipy.sparse as sp
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from gensim.models import Word2Vec

from common import CFG, SEED, rp

NUMERIC = CFG["columns"]["numeric"]
PROC = CFG["columns"]["text_proc"]


# ----------------------------------------------------------------- atomic blocks
def _tfidf(kind, train_texts, eval_texts):
    p = CFG["features"][kind]
    vec = TfidfVectorizer(ngram_range=tuple(p["ngram_range"]), min_df=p["min_df"],
                          max_features=p["max_features"], sublinear_tf=p["sublinear_tf"],
                          analyzer=p["analyzer"])
    Xtr = vec.fit_transform(train_texts)
    Xev = vec.transform(eval_texts)
    return Xtr, Xev, vec


def _stats(df, train_pos, eval_pos):
    sc = StandardScaler()
    Ntr = sc.fit_transform(df.iloc[train_pos][NUMERIC].to_numpy(float))
    Nev = sc.transform(df.iloc[eval_pos][NUMERIC].to_numpy(float))
    return Ntr, Nev


def _lsa(Xtr_tfidf, Xev_tfidf):
    dim = min(CFG["features"]["lsa_dim"], Xtr_tfidf.shape[1] - 1)
    svd = TruncatedSVD(n_components=dim, random_state=SEED)
    Ltr = svd.fit_transform(Xtr_tfidf)
    Lev = svd.transform(Xev_tfidf)
    return Ltr, Lev


def _train_w2v(train_tokens):
    p = CFG["features"]["w2v"]
    return Word2Vec(sentences=list(train_tokens), vector_size=p["vector_size"], window=p["window"],
                    min_count=p["min_count"], sg=p["sg"], epochs=p["epochs"], workers=1, seed=SEED)


def _w2v_mean(model, tokens_list):
    dim = model.wv.vector_size
    out = np.zeros((len(tokens_list), dim), np.float32)
    for i, toks in enumerate(tokens_list):
        vecs = [model.wv[t] for t in toks if t in model.wv.key_to_index]
        if vecs:
            out[i] = np.mean(vecs, axis=0)
    return out


def _w2v_tfidf_mean(model, tokens_list, idf_map):
    dim = model.wv.vector_size
    out = np.zeros((len(tokens_list), dim), np.float32)
    default = float(np.mean(list(idf_map.values()))) if idf_map else 1.0
    for i, toks in enumerate(tokens_list):
        vecs, wts = [], []
        for t in toks:
            if t in model.wv.key_to_index:
                vecs.append(model.wv[t]); wts.append(idf_map.get(t, default))
        if vecs:
            w = np.asarray(wts, np.float32)
            out[i] = (np.asarray(vecs, np.float32) * w[:, None]).sum(0) / (w.sum() + 1e-9)
    return out


# ----------------------------------------------------------------- assembly
def _build(task, df, train_pos, eval_pos, name):
    tr_text = df.iloc[train_pos][PROC].tolist()
    ev_text = df.iloc[eval_pos][PROC].tolist()

    def tfidf12():
        return _tfidf("tfidf_word_12", tr_text, ev_text)

    if name == "stats":
        Ntr, Nev = _stats(df, train_pos, eval_pos)
        return Ntr, Nev

    if name in ("tfidf_word_12", "tfidf_word_1", "tfidf_char_35"):
        Xtr, Xev, _ = _tfidf(name, tr_text, ev_text)
        return Xtr, Xev

    if name == "lsa_300":
        Xtr, Xev, _ = tfidf12()
        Ltr, Lev = _lsa(Xtr, Xev)
        return Ltr, Lev

    if name in ("w2v_mean", "w2v_tfidf_mean"):
        model = _train_w2v(df.iloc[train_pos]["tokens"].tolist())
        if name == "w2v_mean":
            return _w2v_mean(model, df.iloc[train_pos]["tokens"].tolist()), \
                   _w2v_mean(model, df.iloc[eval_pos]["tokens"].tolist())
        vec = TfidfVectorizer(ngram_range=(1, 1), min_df=CFG["features"]["tfidf_word_1"]["min_df"])
        vec.fit(tr_text)
        idf_map = {w: vec.idf_[j] for w, j in vec.vocabulary_.items()}
        return _w2v_tfidf_mean(model, df.iloc[train_pos]["tokens"].tolist(), idf_map), \
               _w2v_tfidf_mean(model, df.iloc[eval_pos]["tokens"].tolist(), idf_map)

    if name == "tfidf+stats":
        Xtr, Xev, _ = tfidf12()
        Ntr, Nev = _stats(df, train_pos, eval_pos)
        return sp.hstack([Xtr, sp.csr_matrix(Ntr)]).tocsr(), sp.hstack([Xev, sp.csr_matrix(Nev)]).tocsr()

    if name == "lsa+stats":
        Xtr, Xev, _ = tfidf12()
        Ltr, Lev = _lsa(Xtr, Xev)
        Ntr, Nev = _stats(df, train_pos, eval_pos)
        return np.hstack([Ltr, Ntr]).astype(np.float32), np.hstack([Lev, Nev]).astype(np.float32)

    if name == "unified":   # tfidf + w2v_mean + stats  (the legacy pipeline's block)
        Xtr, Xev, _ = tfidf12()
        model = _train_w2v(df.iloc[train_pos]["tokens"].tolist())
        Wtr = _w2v_mean(model, df.iloc[train_pos]["tokens"].tolist())
        Wev = _w2v_mean(model, df.iloc[eval_pos]["tokens"].tolist())
        Ntr, Nev = _stats(df, train_pos, eval_pos)
        Utr = sp.hstack([Xtr, sp.csr_matrix(Wtr), sp.csr_matrix(Ntr)]).tocsr()
        Uev = sp.hstack([Xev, sp.csr_matrix(Wev), sp.csr_matrix(Nev)]).tocsr()
        return Utr, Uev

    raise ValueError(f"unknown feature set: {name}")


def canon(X):
    """Put a sparse matrix in canonical (sorted, deduped) form so downstream
    read-only checks (e.g. liblinear's np.max under loky memmap) don't try to
    sort in place. Fancy-index slicing resets this flag, so call it on slices too.
    """
    if sp.issparse(X):
        X = X.tocsr()
        X.sum_duplicates()
        X.sort_indices()
    return X


def get_feature_set(task, split_id, df, train_pos, eval_pos, name, use_cache=True):
    """Assemble (or load cached) a named feature set for one split."""
    cache_path = rp(CFG["paths"]["cache_dir"], f"{task}__{split_id}__{name}.joblib")
    if use_cache and os.path.exists(cache_path):
        d = joblib.load(cache_path)
        return canon(d["Xtr"]), canon(d["Xev"])
    Xtr, Xev = _build(task, df, train_pos, eval_pos, name)
    Xtr, Xev = canon(Xtr), canon(Xev)
    if use_cache:
        joblib.dump({"Xtr": Xtr, "Xev": Xev}, cache_path)
    return Xtr, Xev


def dim_of(X):
    return X.shape[1]


if __name__ == "__main__":
    from common import load_task, seed_everything
    from make_splits import load_splits
    seed_everything()
    df = load_task("rating")
    s = load_splits("rating")
    ftr, fva = s["folds"][0]
    # map absolute idx -> positional (df already reset_index in load_task, so idx==pos)
    for name in ["stats", "tfidf_word_12", "lsa_300", "w2v_mean", "w2v_tfidf_mean",
                 "tfidf+stats", "lsa+stats", "unified"]:
        Xtr, Xev = get_feature_set("rating", "smoke_fold0", df, ftr, fva, name, use_cache=False)
        print(f"{name:16s} train{tuple(Xtr.shape)} eval{tuple(Xev.shape)} "
              f"{'sparse' if sp.issparse(Xtr) else 'dense'}")
