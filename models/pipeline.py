# -*- coding: utf-8 -*-
"""
End-to-end reproducible pipeline (Phase 2-4) for the Codeforces rating & tag study.
Input : ../Kaggle_dataset/codeforces_master_dataset.csv   (the master dataset)
Output: results_*.json, predictions_*_test.csv

Deterministic (seed=42). Requires: pandas numpy scipy scikit-learn lightgbm gensim nltk
NLP note: Porter stemming is used in place of WordNet lemmatization (offline-safe);
swap in WordNetLemmatizer if nltk 'wordnet' data is available.
"""
import pandas as pd, numpy as np, re, json, warnings, scipy.sparse as sp
warnings.filterwarnings("ignore")
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, GridSearchCV
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
    f1_score, precision_score, recall_score, hamming_loss, accuracy_score, jaccard_score)
from nltk.stem import PorterStemmer
from gensim.models import Word2Vec
import lightgbm as lgb
SEED=42

# ---------- load + leakage-safe grouping ----------
df = pd.read_csv("../Kaggle_dataset/codeforces_master_dataset.csv")
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
df['tags_norm'] = df['tags_norm'].fillna('')
df['has_rating'] = df['rating'].notna()
df['has_tags']  = df['tags_norm'].str.strip()!=''
# prefer the validated recomputed count features when present
for _c in ['statement_length_chars','statement_length_words','num_lessequal',
           'num_greaterequal','num_equals','num_numeric_tokens']:
    if _c+'_recomputed' in df.columns: df[_c]=df[_c+'_recomputed']

# union-find over contestId linked by exact-duplicate statements -> split_group
parent={}
def find(x):
    parent.setdefault(x,x)
    while parent[x]!=x: parent[x]=parent[parent[x]]; x=parent[x]
    return x
def union(a,b):
    ra,rb=find(a),find(b)
    if ra!=rb: parent[ra]=rb
for c in df['contestId'].unique(): find(int(c))
for _,g in df.groupby('statement_clean_norm'):
    cids=g['contestId'].astype(int).tolist()
    for i in range(1,len(cids)): union(cids[0],cids[i])
df['split_group']=df['contestId'].astype(int).map(find)

# ---------- dedup fully-redundant rows ----------
df=df[~df.duplicated(subset=['statement_clean_norm','rating','tags_norm'],keep='first')].reset_index(drop=True)

# ---------- NLP transform ----------
ps=PorterStemmer(); tok=re.compile(r"[a-z]+"); STOP=set(ENGLISH_STOP_WORDS); cache={}
def stem(w):
    v=cache.get(w);  
    if v is None: v=ps.stem(w); cache[w]=v
    return v
df['tokens']=df['statement_clean_norm'].fillna('').map(
    lambda t:[stem(x) for x in tok.findall(t.lower()) if x not in STOP and len(x)>1])
df['statement_proc']=df['tokens'].map(' '.join)

# ---------- 80:20 grouped split ----------
tr,te=next(GroupShuffleSplit(1,test_size=0.2,random_state=SEED).split(df,groups=df['split_group']))
df['split']='train'; df.loc[df.index[te],'split']='test'

NUM=['time_limit_sec','memory_limit_mb','statement_length_chars','statement_length_words',
 'num_lessequal','num_greaterequal','num_equals','num_numeric_tokens','avg_word_length',
 'sentence_count','avg_sentence_length_words','digit_count','uppercase_ratio','digit_ratio',
 'punctuation_ratio','math_symbol_count','unique_word_ratio','example_count']
for c in NUM: df[c]=pd.to_numeric(df[c],errors='coerce').fillna(0.0)

def feats(a,b,with_dense):
    tf=TfidfVectorizer(ngram_range=(1,2),min_df=5,max_features=30000,sublinear_tf=True)
    Xa=tf.fit_transform(a['statement_proc']); Xb=tf.transform(b['statement_proc'])
    if not with_dense: return Xa,Xb
    w2v=Word2Vec(a['tokens'].tolist(),vector_size=100,window=5,min_count=5,sg=1,workers=4,epochs=5,seed=SEED)
    dv=lambda t:(np.mean([w2v.wv[x] for x in t if x in w2v.wv],0) if any(x in w2v.wv for x in t) else np.zeros(100,np.float32))
    Wa=np.vstack(a['tokens'].map(dv)); Wb=np.vstack(b['tokens'].map(dv))
    sc=StandardScaler(); Na=sc.fit_transform(a[NUM]); Nb=sc.transform(b[NUM])
    Ua=sp.hstack([Xa,sp.csr_matrix(Wa),sp.csr_matrix(Na)]).tocsr()
    Ub=sp.hstack([Xb,sp.csr_matrix(Wb),sp.csr_matrix(Nb)]).tocsr()
    return Ua,Ub

# ================= RATING =================
rr=df[df['has_rating']]; a=rr[rr.split=='train']; b=rr[rr.split=='test']
Ua,Ub=feats(a,b,with_dense=True)
ya=a['rating'].values; yb=b['rating'].values; ga=a['split_group'].values
M=lambda y,p:{'MAE':round(mean_absolute_error(y,p),2),'RMSE':round(float(np.sqrt(mean_squared_error(y,p))),2),'R2':round(r2_score(y,p),4)}
rating={'Mean_baseline':M(yb,np.full_like(yb,ya.mean()))}
rg=GridSearchCV(Ridge(),{'alpha':[1,10,30,100]},cv=list(GroupKFold(5).split(Ua,ya,ga)),scoring='neg_mean_absolute_error',n_jobs=-1).fit(Ua,ya)
rating['Ridge_unified']={**M(yb,rg.predict(Ub)),'best_alpha':rg.best_params_['alpha']}
ci,vi=next(GroupShuffleSplit(1,test_size=0.15,random_state=1).split(Ua,ya,ga))
gbm=lgb.train(dict(objective='regression_l1',metric='mae',learning_rate=0.05,num_leaves=63,
  feature_fraction=0.7,bagging_fraction=0.8,bagging_freq=1,min_child_samples=20,verbose=-1,seed=SEED),
  lgb.Dataset(Ua[ci],ya[ci]),2000,valid_sets=[lgb.Dataset(Ua[vi],ya[vi])],callbacks=[lgb.early_stopping(50,verbose=False)])
rating['LightGBM_unified']={**M(yb,gbm.predict(Ub)),'best_iter':gbm.best_iteration}
json.dump(rating,open('results_rating.json','w'),indent=2); print("RATING",rating)

# ================= TAGS =================
tg=df[df['has_tags']]; a=tg[tg.split=='train']; b=tg[tg.split=='test']
Xa,Xb=feats(a,b,with_dense=False)
mlb=MultiLabelBinarizer()
Ya=mlb.fit_transform(a['tags_norm'].map(lambda s:[t.strip() for t in s.split(',') if t.strip()]))
Yb=mlb.transform(b['tags_norm'].map(lambda s:[t.strip() for t in s.split(',') if t.strip()]))
ga=a['split_group'].values
ti,vi=next(GroupShuffleSplit(1,test_size=0.2,random_state=7).split(Xa,Ya,ga))
def tune(make,Cs):
    best=None
    for C in Cs:
        f=f1_score(Ya[vi],OneVsRestClassifier(make(C),n_jobs=-1).fit(Xa[ti],Ya[ti]).predict(Xa[vi]),average='micro',zero_division=0)
        if best is None or f>best[1]: best=(C,f)
    return best[0]
MET=lambda Y,P:{'f1_micro':round(f1_score(Y,P,average='micro',zero_division=0),4),'f1_macro':round(f1_score(Y,P,average='macro',zero_division=0),4),
  'precision_micro':round(precision_score(Y,P,average='micro',zero_division=0),4),'recall_micro':round(recall_score(Y,P,average='micro',zero_division=0),4),
  'hamming_loss':round(hamming_loss(Y,P),4),'subset_accuracy':round(accuracy_score(Y,P),4),'jaccard_samples':round(jaccard_score(Y,P,average='samples',zero_division=0),4)}
tags={}
C=tune(lambda C:LogisticRegression(C=C,max_iter=300,solver='liblinear'),[1,3,10])
tags['TFIDF_LogReg']={**MET(Yb,OneVsRestClassifier(LogisticRegression(C=C,max_iter=300,solver='liblinear'),n_jobs=-1).fit(Xa,Ya).predict(Xb)),'best_C':C}
C=tune(lambda C:LinearSVC(C=C),[0.3,1,3])
tags['TFIDF_LinearSVM']={**MET(Yb,OneVsRestClassifier(LinearSVC(C=C),n_jobs=-1).fit(Xa,Ya).predict(Xb)),'best_C':C}
json.dump(tags,open('results_tags.json','w'),indent=2); print("TAGS",tags)
