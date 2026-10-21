# -*- coding: utf-8 -*-
"""Uncapped (MAX_FRAMES=None) 121-video pass. Resumable per-video cache."""
import os, re, glob, csv, json, sys, time
import numpy as np
import cv2
from multiprocessing import Pool

CODE = "/sessions/serene-trusting-brown/mnt/PhD Program/02. Working/03_AwP_Frame_Scheduling/code"
sys.path.insert(0, CODE)
from awp_experiments import (hedge_gate_fire, recall_windowed, periodic_certificate,
                             oblivious_random, bootstrap_ci, onset_auc, eta_from_auc,
                             A_GRID, LAMBDA_GRID, CORRUPT_GRID, W, B, SEED)

DS_ROOT = "/sessions/serene-trusting-brown/mnt/THS Programing/06.6 ASMAG Project/datasets"
CDNET   = "/sessions/serene-trusting-brown/mnt/THS Programing/06.6 ASMAG Project/dataset cdnet2014/archive/dataset"
OUT     = "/sessions/serene-trusting-brown/mnt/outputs/awp_uncapped"
RES     = os.path.join(OUT, "results"); os.makedirs(RES, exist_ok=True)
CACHE   = os.path.join(OUT, "cache");   os.makedirs(CACHE, exist_ok=True)
DOWNSCALE=(64,48); MAX_FRAMES=None; FG_THR=200; FG_FRAC_THR=0.0008
MIN_DURATION=2; N_WORKERS=8; TAG="64x48_fALL"
NAT=re.compile(r"(\d+)")
def natkey(s): return [int(x) if x.isdigit() else x for x in NAT.split(os.path.basename(s))]
def _cache_path(vid): return os.path.join(CACHE,"%s__%s.npz"%(vid,TAG))
def extract_one(job):
    vid_id,frames,gts=job; cache=_cache_path(vid_id)
    if os.path.exists(cache): return (vid_id,-1)
    frames=sorted(frames,key=natkey); gts=sorted(gts,key=natkey)
    if MAX_FRAMES: frames,gts=frames[:MAX_FRAMES],gts[:MAX_FRAMES]
    n=min(len(frames),len(gts))
    if n<5: return (vid_id,0)
    cue=np.zeros(n,dtype=np.float32); gt=np.zeros(n,dtype=bool); prev=None
    tot_px=DOWNSCALE[0]*DOWNSCALE[1]
    for t in range(n):
        g=cv2.imread(frames[t],cv2.IMREAD_REDUCED_GRAYSCALE_4)
        if g is None: cue[t]=cue[t-1] if t else 0.0
        else:
            g=cv2.resize(g,DOWNSCALE).astype(np.float32)
            cue[t]=float(np.mean(np.abs(g-prev))) if prev is not None else 0.0; prev=g
        m=cv2.imread(gts[t],cv2.IMREAD_REDUCED_COLOR_4)
        if m is not None:
            m=cv2.resize(m,DOWNSCALE); fg=int((m.max(axis=2)>FG_THR).sum())
            gt[t]=(fg>FG_FRAC_THR*tot_px)
    if n>1: cue[0]=cue[1]
    np.savez_compressed(cache,cue=cue,gt=gt); return (vid_id,n)
def extract_events(gt,min_duration=MIN_DURATION):
    events=[]; T=len(gt); t=0
    while t<T:
        if gt[t]:
            s=t
            while t<T and gt[t]: t+=1
            if (t-s)>=min_duration: events.append({"onset":s,"end":t-1,"duration":t-s})
        else: t+=1
    return events
def enumerate_videos():
    vids=[]
    for sub in ["synth1","synth2"]:
        base=os.path.join(DS_ROOT,"BMC",sub)
        if os.path.isdir(base):
            for v in sorted(os.listdir(base)):
                fdir=os.path.join(base,v,"frames"); gdir=os.path.join(base,v,"gt")
                if os.path.isdir(fdir) and os.path.isdir(gdir):
                    vids.append(("BMC","BMC_%s_%s"%(sub,v),glob.glob(fdir+"/*"),glob.glob(gdir+"/*")))
    las=os.path.join(DS_ROOT,"LASIESTA")
    if os.path.isdir(las):
        for gtd in sorted(glob.glob(las+"/*-GT")):
            base=gtd[:-3]
            if os.path.isdir(base):
                vids.append(("LASIESTA","LAS_"+os.path.basename(base),
                             glob.glob(base+"/*.bmp")+glob.glob(base+"/*.jpg"),glob.glob(gtd+"/*.png")))
    if os.path.isdir(CDNET):
        for cat in sorted(os.listdir(CDNET)):
            catd=os.path.join(CDNET,cat)
            if not os.path.isdir(catd): continue
            for v in sorted(os.listdir(catd)):
                vd=os.path.join(catd,v); ind=os.path.join(vd,"input"); gtd=os.path.join(vd,"groundtruth")
                if not (os.path.isdir(ind) and os.path.isdir(gtd)): continue
                fr=sorted(glob.glob(ind+"/*.jpg"),key=natkey); gg=sorted(glob.glob(gtd+"/*.png"),key=natkey)
                s,e=1,len(fr); roi=os.path.join(vd,"temporalROI.txt")
                if os.path.exists(roi):
                    try: s,e=[int(x) for x in open(roi).read().split()[:2]]
                    except Exception: pass
                fr=fr[s-1:e]; gg=gg[s-1:e]
                if fr and gg: vids.append(("CDnet","CDN_%s_%s"%(cat,v),fr,gg))
    return vids
def group_of(ds,vid):
    if ds=="CDnet": return "CDnet:"+vid.split("_",2)[1]
    return ds
def load_corpus(sel):
    corpus=[]
    for ds,vid,f,g in sel:
        cp=_cache_path(vid)
        if not os.path.exists(cp): continue
        z=np.load(cp); cue=z["cue"]; gt=z["gt"].astype(bool); ev=extract_events(gt)
        if len(ev)>=2:
            win=np.zeros(len(gt),bool)
            for e in ev: win[e["onset"]:min(len(gt),e["onset"]+W+1)]=True
            corpus.append(dict(grp=group_of(ds,vid),ds=ds,cue=cue,events=ev,win=win,T=len(gt)))
    return corpus
def main():
    allv=enumerate_videos()
    sel=[(ds,vid,f,g) for ds,vid,f,g in allv if f and g]
    todo=[(vid,f,g) for ds,vid,f,g in sel if not os.path.exists(_cache_path(vid))]
    print("[enum] %d videos; %d to extract, %d cached"%(len(sel),len(todo),len(sel)-len(todo)),flush=True)
    if todo:
        with Pool(N_WORKERS) as pool:
            done=0
            for vid_id,n in pool.imap_unordered(extract_one,todo):
                done+=1
                if n and n>0 and done%10==0:
                    print("  [%d/%d] %s n=%d"%(done,len(todo),vid_id,n),flush=True)
    n_cached=sum(1 for ds,vid,f,g in sel if os.path.exists(_cache_path(vid)))
    print("[extract] cached %d/%d"%(n_cached,len(sel)),flush=True)
    if n_cached<len(sel):
        print("[PARTIAL] rerun to finish remaining %d"%(len(sel)-n_cached),flush=True); return
    corpus=load_corpus(sel)
    print("[corpus] %d videos >=2ev; events=%d frames=%d"%(len(corpus),
          sum(len(c['events']) for c in corpus),sum(c['T'] for c in corpus)),flush=True)
    def grp_stats(rows,a):
        gate=np.mean([recall_windowed(hedge_gate_fire(c["cue"],c["T"],a,0.0),c["events"],W) for c in rows])
        per =np.mean([recall_windowed(hedge_gate_fire(c["cue"],c["T"],a,1.0),c["events"],W) for c in rows])
        return gate,per
    groups=sorted(set(c["grp"] for c in corpus)); out=[]
    for grp in groups+["ALL"]:
        rows=corpus if grp=="ALL" else [c for c in corpus if c["grp"]==grp]
        auc=float(np.mean([onset_auc(c["cue"],c["win"]) for c in rows]))
        nev=sum(len(c["events"]) for c in rows)
        g1,p1=grp_stats(rows,0.10); g2,p2=grp_stats(rows,0.20)
        out.append(dict(group=grp,n_videos=len(rows),n_events=nev,onset_auc=round(auc,3),
                        gate_a010=round(g1,3),periodic_a010=round(p1,3),gap_a010=round(g1-p1,3),
                        gate_a020=round(g2,3),periodic_a020=round(p2,3),gap_a020=round(g2-p2,3)))
    fields=["group","n_videos","n_events","onset_auc","gate_a010","periodic_a010","gap_a010",
            "gate_a020","periodic_a020","gap_a020"]
    with open(os.path.join(RES,"percategory_UNCAPPED.csv"),"w",newline="",encoding="utf-8") as fh:
        w=csv.DictWriter(fh,fieldnames=fields); w.writeheader(); [w.writerow(r) for r in out]
    deployed={}
    for a in A_GRID:
        cert,K=periodic_certificate(a,W)
        gate_per=[recall_windowed(hedge_gate_fire(c["cue"],c["T"],a,0.0),c["events"],W) for c in corpus]
        per_per =[recall_windowed(hedge_gate_fire(c["cue"],c["T"],a,1.0),c["events"],W) for c in corpus]
        gm,glo,ghi=bootstrap_ci(gate_per,B,SEED+int(a*1000))
        pm,plo,phi=bootstrap_ci(per_per,B,SEED+int(a*1000)+5)
        deployed["a%.3f"%a]=dict(real_cue_gate=round(gm,4),gate_ci=[round(glo,4),round(ghi,4)],
                                 periodic=round(pm,4),periodic_ci=[round(plo,4),round(phi,4)],
                                 certificate=round(cert,4),gate_minus_periodic=round(gm-pm,4))
    aucs=[onset_auc(c["cue"],c["win"]) for c in corpus]; by_ds={}
    for c in corpus: by_ds.setdefault(c["ds"],[]).append(onset_auc(c["cue"],c["win"]))
    summary=dict(source="REAL UNCAPPED (MAX_FRAMES=None) BMC+LASIESTA+CDnet",
        config=dict(seed=SEED,W=W,B=B,downscale=list(DOWNSCALE),max_frames=None,
                    n_videos_selected=len(sel),n_videos_with_events=len(corpus),
                    n_events=sum(len(c["events"]) for c in corpus),n_frames=sum(c["T"] for c in corpus)),
        cue_onset_auc=dict(mean=round(float(np.mean(aucs)),4),min=round(float(np.min(aucs)),4),
                           max=round(float(np.max(aucs)),4),
                           by_dataset={k:round(float(np.mean(v)),4) for k,v in by_ds.items()},
                           pa_reference=[0.49,0.79]),
        eta_deployed=round(eta_from_auc(float(np.mean(aucs))),4),deployed_gate_vs_periodic=deployed)
    json.dump(summary,open(os.path.join(RES,"metrics_summary_UNCAPPED.json"),"w"),indent=2,ensure_ascii=False,allow_nan=False)
    print("[done] wrote percategory_UNCAPPED.csv + metrics_summary_UNCAPPED.json",flush=True)
    for r in out: print("  %-26s n=%2d ev=%4d AUC=%.3f gap@.1=%+.3f gap@.2=%+.3f"%(
        r["group"],r["n_videos"],r["n_events"],r["onset_auc"],r["gap_a010"],r["gap_a020"]),flush=True)
if __name__=="__main__":
    t0=time.time(); main(); print("[elapsed] %.1fs"%(time.time()-t0),flush=True)
