# -*- coding: utf-8 -*-
"""Does a STRONGER learned cue rescue cross-scene? Train a nonlinear learned cue
(HistGradientBoosting on 7 cheap features) and compare within-scene vs
Leave-One-Category-Out (LOCO, cross-scene) onset-AUC and gate recall vs periodic.
Uses pre-cached CDnet features (no video re-read)."""
import numpy as np, glob, os, math, csv
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
W=7; A=[0.1,0.2,0.333]; SEED=42; rng=np.random.default_rng(SEED)
FE=sorted(glob.glob("AwP-frame-scheduling-public/gates/data/features/*/*.npz"))
USE=[0,1,2,3,4,5,6]  # content features; idx7-9 are zeroed gating-decision features

def events(gt,md=2):
    ev=[];T=len(gt);t=0
    while t<T:
        if gt[t]:
            s=t
            while t<T and gt[t]: t+=1
            if t-s>=md: ev.append(s)
        else: t+=1
    return ev
def winlabel(gt):
    T=len(gt);y=np.zeros(T,bool)
    for o in events(gt): y[o:min(T,o+W+1)]=True
    return y
def recall_windowed(fired,ev,T):
    if not ev: return np.nan
    f=np.where(fired)[0]
    if len(f)==0: return 0.0
    d=0
    for o in ev:
        j=np.searchsorted(f,o)
        if j<len(f) and f[j]<=o+W: d+=1
    return d/len(ev)
def periodic_recall(ev,T,a):
    K=math.ceil(1/a); rs=[]
    for ph in range(K):
        fired=np.zeros(T,bool); fired[ph::K]=True
        rs.append(recall_windowed(fired,ev,T))
    return np.nanmean(rs)
def gate_recall(score,ev,T,a):
    b=int(round(a*T)); fired=np.zeros(T,bool)
    if b>0: fired[np.argsort(-score)[:b]]=True
    return recall_windowed(fired,ev,T)

# load per-video (ROI frames only)
V=[]
for f in FE:
    d=np.load(f,allow_pickle=True)
    roi=d['temporal_roi'].astype(bool); X=d['features'][:,USE][roi]; gt=d['gt_event'].astype(bool)[roi]
    if roi.sum()<20 or len(events(gt))<1: continue
    cat=os.path.basename(os.path.dirname(f))
    V.append(dict(cat=cat,X=X.astype(np.float32),gt=gt,y=winlabel(gt),ev=events(gt),T=int(roi.sum())))
cats=sorted(set(v['cat'] for v in V))
print(f"[data] {len(V)} videos, {len(cats)} categories, frames={sum(v['T'] for v in V)}, events={sum(len(v['ev']) for v in V)}")

def fit_predict(train,test):
    Xtr=np.vstack([v['X'] for v in train]); ytr=np.concatenate([v['y'] for v in train])
    m=HistGradientBoostingClassifier(max_iter=150,max_depth=4,learning_rate=0.1,
        l2_regularization=1.0,early_stopping=True,random_state=SEED)
    m.fit(Xtr,ytr)
    return [m.predict_proba(v['X'])[:,1] for v in test]

# ---- within-scene: 5-fold over videos (categories mixed) ----
idx=np.arange(len(V)); rng.shuffle(idx); folds=np.array_split(idx,5)
auc_w_learn=[]; auc_w_base=[]
for k in range(5):
    te=[V[i] for i in folds[k]]; tr=[V[i] for i in idx if i not in set(folds[k])]
    preds=fit_predict(tr,te)
    for v,p in zip(te,preds):
        if v['y'].any() and not v['y'].all():
            auc_w_learn.append(roc_auc_score(v['y'],p)); auc_w_base.append(roc_auc_score(v['y'],v['X'][:,0]))
# ---- cross-scene: Leave-One-Category-Out ----
auc_c_learn=[]; auc_c_base=[]; gate_rows=[]
loco_scores={}
for c in cats:
    te=[v for v in V if v['cat']==c]; tr=[v for v in V if v['cat']!=c]
    preds=fit_predict(tr,te)
    for v,p in zip(te,preds):
        loco_scores[id(v)]=p
        if v['y'].any() and not v['y'].all():
            auc_c_learn.append(roc_auc_score(v['y'],p)); auc_c_base.append(roc_auc_score(v['y'],v['X'][:,0]))

def boot(vals,B=1000,seed=1):
    vals=np.array([x for x in vals if not np.isnan(x)]); r=np.random.default_rng(seed)
    m=np.array([np.mean(vals[r.integers(0,len(vals),len(vals))]) for _ in range(B)])
    return vals.mean(),np.percentile(m,2.5),np.percentile(m,97.5)

print("\n=== onset-AUC (higher=better cue; eta=1-AUC) ===")
print(f"  within-scene : baseline(motion) {np.mean(auc_w_base):.3f}   learned-GBT {np.mean(auc_w_learn):.3f}")
print(f"  cross-scene  : baseline(motion) {np.mean(auc_c_base):.3f}   learned-GBT {np.mean(auc_c_learn):.3f}")
print("\n=== gate recall vs periodic (cross-scene LOCO learned cue) ===")
for a in A:
    gr=[gate_recall(loco_scores[id(v)],v['ev'],v['T'],a) for v in V]
    pr=[periodic_recall(v['ev'],v['T'],a) for v in V]
    gm,glo,ghi=boot(gr); pm,plo,phi=boot(pr)
    print(f"  a={a:.3f}: learned-gate {gm:.3f}[{glo:.3f},{ghi:.3f}]  periodic {pm:.3f}[{plo:.3f},{phi:.3f}]  gap {gm-pm:+.3f}")
    gate_rows.append(dict(a=a,learned_gate=round(gm,4),gate_ci=[round(glo,4),round(ghi,4)],
                          periodic=round(pm,4),periodic_ci=[round(plo,4),round(phi,4)],gap=round(gm-pm,4)))
# save
os.makedirs("results",exist_ok=True)
import json
json.dump(dict(n_videos=len(V),n_cats=len(cats),
   onset_auc=dict(within=dict(baseline=round(float(np.mean(auc_w_base)),4),learned=round(float(np.mean(auc_w_learn)),4)),
                  cross=dict(baseline=round(float(np.mean(auc_c_base)),4),learned=round(float(np.mean(auc_c_learn)),4))),
   gate_vs_periodic=gate_rows),
   open("results/learned_cue_probe.json","w"),indent=2)
print("\nsaved results/learned_cue_probe.json")
