# ==== C4 repo note (paths) ====
# Data paths below are ABSOLUTE to the author's machine (CDnet/LASIESTA/BMC/VIRAT
# detection dumps, frame_scores parquet, ASMAG NPZ/models). To re-run: edit the path
# constants and obtain the data (see README "Data & paths"). Canonical outputs are the
# committed ../../results/<name>.json. perscene_k.py also needs the deep0 "src" package;
# compute_vii*.py needs the asmag-rl-vs-deterministic modules (rl/, asmag_core/).
# ==============================
"""
C4 §VII low-a FRONTIER re-run. Reuse clean models (RF CALIB, DQN T4b), features idx8=0,
26-video held-out. NO retrain. Metric: fast_simulate event_recall (W=7), iso-energy
(matched activation), scene-independent. Bootstrap CI B=1000 seed=42.
Deterministic gates sweep a continuous score -> full frontier incl. low a; DQN uses its
lambda points (report min-reachable a).
"""
import json,sys,pickle,warnings,numpy as np
sys.path.insert(0,".")
from rl.fast_eval import fast_simulate, dqn_actions
from rl.agents import DQNAgent
from asmag_core.controllers import ACTION_SPACE
P_MISS,R_HIT,R_MAINT,R_QUIET,W=0.5,1.0,0.2,0.1,7
ACC,FAST,P3=ACTION_SPACE.index("ACC"),ACTION_SPACE.index("FAST"),ACTION_SPACE.index("P3_FALLBACK")
split=json.load(open("splits/S1_fold0.json")); test=split["test"]
tr=json.load(open("data/features/train_stats.json")); cols=tr["feature_columns"]
zm=np.array([tr["mean"][k] for k in cols],np.float32); zs=np.array([tr["std"][k] for k in cols],np.float32); zs=np.where(zs<1e-7,1.0,zs)
assert tr["mean"]["active_frame_rate"]==0.0,"PROVENANCE FAIL"
def npz(v):
    c,n=v.split("/");return np.load(f"data/features/{c}/{n}.npz",allow_pickle=True)
NP={v:npz(v) for v in test}
def sim(d,a):return fast_simulate(d,a,0.2,P_MISS,R_HIT,R_MAINT,R_QUIET,W)["episode"]
mRF=pickle.load(open("models/calib_S1_fold0_trainonly.pkl","rb")); mRF=mRF["model"] if isinstance(mRF,dict) else mRF
det_classes=[i for i,c in enumerate(mRF.classes_) if str(c) in ("FAST","ACC")]
# per-video continuous detect-scores
S_rf={}; S_mot={}
for v in test:
    d=NP[v];roi=d["temporal_roi"].astype(bool);feat=d["features"][roi].astype(np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore"); proba=mRF.predict_proba(feat)
    S_rf[v]=proba[:,det_classes].sum(axis=1)        # P(detect)
    S_mot[v]=feat[:,0]                                # motion_density_mean (S-gate)
ALLRF=np.concatenate([S_rf[v] for v in test]); ALLM=np.concatenate([S_mot[v] for v in test])
def recall_periodic(R):
    rec=[];act=[]
    for v in test:
        T=int(NP[v]["temporal_roi"].sum());a=np.full(T,P3,np.int32);a[::R]=ACC
        e=sim(NP[v],a);rec.append(e["event_recall"]);act.append(e["activation"])
    return np.array(rec),float(np.mean(act))
def recall_scoregate(Sdict,ALL,target_a):
    tau=np.quantile(ALL,1-target_a); rec=[];act=[]
    for v in test:
        op=Sdict[v]>tau;a=np.where(op,ACC,P3).astype(np.int32)
        e=sim(NP[v],a);rec.append(e["event_recall"]);act.append(e["activation"])
    return np.array(rec),float(np.mean(act))
def boot(diff,B=1000,seed=42):
    rng=np.random.RandomState(seed);n=len(diff);m=[diff[rng.randint(0,n,n)].mean() for _ in range(B)]
    return float(diff.mean()),float(np.percentile(m,2.5)),float(np.percentile(m,97.5))

GRID=[(10,0.10),(7,0.143),(5,0.20),(3,0.333),(2,0.50)]
print("FRONTIER recall (event_recall, W=7), 26 held-out videos:")
print(f"{'target_a':>9}{'periodic':>10}{'RF-gate':>9}{'S-gate':>9}")
frontier={}
for R,a in GRID:
    pr,pa=recall_periodic(R); rf,ra=recall_scoregate(S_rf,ALLRF,a); sg,sa=recall_scoregate(S_mot,ALLM,a)
    frontier[a]={"periodic":[float(pr.mean()),pa],"RF":[float(rf.mean()),ra],"S":[float(sg.mean()),sa]}
    print(f"{a:>9.2f}{pr.mean():>10.3f}{rf.mean():>9.3f}{sg.mean():>9.3f}")

# DQN reachable activations
print("\nDQN reachable (lambda points): activation + recall")
dqn_pts=[]
for lam in (0.05,0.10,0.20,0.40,0.80):
    seedrec=[];acts=[]
    for s in (0,1,2):
        ag=DQNAgent(seed=s);ag.load(f"outputs/rl/t4b/policy_fair_dqn_lam{lam:.2f}_seed{s}.pt");ag.eps=0.0
        rec=[];ac=[]
        for v in test:
            obs=(NP[v]["features"][NP[v]["temporal_roi"].astype(bool)].astype(np.float32)-zm)/zs
            e=sim(NP[v],dqn_actions(ag,obs));rec.append(e["event_recall"]);ac.append(e["activation"])
        seedrec.append(np.array(rec));acts.append(np.mean(ac))
    a=float(np.mean(acts));r=float(np.mean([x.mean() for x in seedrec]))
    dqn_pts.append((lam,a,r));print(f"  λ={lam:<5} a={a:.3f} recall={r:.3f}")
min_a_dqn=min(a for _,a,_ in dqn_pts)
print(f"  -> DQN min-reachable activation = {min_a_dqn:.3f} (cannot deep-skip to a<=0.20)")

# gaps @ matched activation 0.10,0.20,0.50 with CI (gate - periodic)
print("\ngate - periodic recall @ matched activation (CI95, B=1000 seed=42):")
gaps={}
for a in (0.10,0.20,0.50):
    R=round(1/a); pr,_=recall_periodic(R); rf,_=recall_scoregate(S_rf,ALLRF,a); sg,_=recall_scoregate(S_mot,ALLM,a)
    grf=boot(rf-pr); gsg=boot(sg-pr)
    gaps[a]={"RF":{"mean":grf[0],"ci":[grf[1],grf[2]]},"S":{"mean":gsg[0],"ci":[gsg[1],gsg[2]]},"periodic_recall":float(pr.mean())}
    print(f"  a={a:.2f}: RF-periodic {grf[0]:+.3f} [{grf[1]:+.3f},{grf[2]:+.3f}] | S-periodic {gsg[0]:+.3f} [{gsg[1]:+.3f},{gsg[2]:+.3f}]  (periodic recall={pr.mean():.3f})")

# iso-90%-recall activation per policy
def iso90_activation(kind):
    # fine sweep, return smallest activation achieving mean recall>=0.90
    if kind=="periodic":
        for R in range(40,0,-1):
            r,a=recall_periodic(R)
            if r.mean()>=0.90: return a
        return 1.0
    Sdict,ALL=(S_rf,ALLRF) if kind=="RF" else (S_mot,ALLM)
    for a in np.linspace(0.05,1.0,40):
        r,act=recall_scoregate(Sdict,ALL,a)
        if r.mean()>=0.90: return act
    return 1.0
iso={k:iso90_activation(k) for k in ("periodic","RF","S")}
print(f"\niso-90%-recall activation: periodic={iso['periodic']:.3f} RF={iso['RF']:.3f} S={iso['S']:.3f}")
print(f"  activation gap (gate-periodic, +=gate costs more=periodic wins): RF {iso['RF']-iso['periodic']:+.3f}, S {iso['S']-iso['periodic']:+.3f}")

# GO/NO-GO at low-a
beat=[(a,g) for a in (0.10,0.20) for g in ("RF","S") if gaps[a][g]["ci"][0]>0]
print(f"\nGO/NO-GO @ a<=0.20: {'STOP — gate CI-beats periodic: '+str(beat) if beat else 'GO — no gate CI-beats periodic at low a; debunk holds in deployable regime'}")

# extend artifact
res=json.load(open("report/a3_learned_result.json"))
res["frontier_low_a"]={"grid":frontier,"dqn_points":[{"lambda":l,"activation":a,"recall":r} for l,a,r in dqn_pts],
   "dqn_min_reachable_activation":min_a_dqn,"gaps_matched_activation":gaps,
   "iso90_activation":iso,"iso90_gap_gate_minus_periodic":{"RF":iso['RF']-iso['periodic'],"S":iso['S']-iso['periodic']},
   "go_nogo_low_a":("STOP" if beat else "GO")}
json.dump(res,open("report/a3_learned_result.json","w"),indent=2)
print("extended report/a3_learned_result.json")
