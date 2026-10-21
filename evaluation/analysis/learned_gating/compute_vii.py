# ==== C4 repo note (paths) ====
# Data paths below are ABSOLUTE to the author's machine (CDnet/LASIESTA/BMC/VIRAT
# detection dumps, frame_scores parquet, ASMAG NPZ/models). To re-run: edit the path
# constants and obtain the data (see README "Data & paths"). Canonical outputs are the
# committed ../../results/<name>.json. perscene_k.py also needs the deep0 "src" package;
# compute_vii*.py needs the asmag-rl-vs-deterministic modules (rl/, asmag_core/).
# ==============================
"""
C4 §VII compute: detection-grounded event-recall, iso-energy (matched activation),
scene-independent (26 held-out test videos). periodic vs det-gate(RF) vs RL-gate(DQN T4b).
Video-level bootstrap CI B=1000 seed=42. Clean de-oracled artifacts only.
Metric: fast_simulate event_recall (detection-availability w/ W=7 staleness) — the A3
gating-level detection-grounded recall. Detector itself out of scope (Deep 1).
"""
import json,sys,pickle,warnings,numpy as np
sys.path.insert(0,".")
from rl.fast_eval import fast_simulate, dqn_actions
from rl.agents import DQNAgent
from asmag_core.controllers import ACTION_SPACE
P_MISS,R_HIT,R_MAINT,R_QUIET,W=0.5,1.0,0.2,0.1,7
ACC,P3=ACTION_SPACE.index("ACC"),ACTION_SPACE.index("P3_FALLBACK")
split=json.load(open("splits/S1_fold0.json")); test=split["test"]
tr=json.load(open("data/features/train_stats.json")); cols=tr["feature_columns"]
zm=np.array([tr["mean"][k] for k in cols],np.float32); zs=np.array([tr["std"][k] for k in cols],np.float32); zs=np.where(zs<1e-7,1.0,zs)
assert tr["mean"]["active_frame_rate"]==0.0, "PROVENANCE FAIL: train_stats active != 0"
def npz(v):
    c,n=v.split("/"); return np.load(f"data/features/{c}/{n}.npz",allow_pickle=True)
NP={v:npz(v) for v in test}
def sim(d,actions):
    return fast_simulate(d,actions,0.2,P_MISS,R_HIT,R_MAINT,R_QUIET,W)["episode"]
def periodic(d,R):
    roi=d["temporal_roi"].astype(bool); T=int(roi.sum())
    a=np.full(T,P3,np.int32); a[::R]=ACC
    e=sim(d,a); return e["event_recall"],e["activation"]
def rf_gate(d,model):
    roi=d["temporal_roi"].astype(bool); feat=d["features"][roi].astype(np.float32)
    with warnings.catch_warnings(): warnings.simplefilter("ignore"); pr=model.predict(feat)
    a=np.array([ACTION_SPACE.index(str(p)) for p in pr],np.int32); e=sim(d,a); return e["event_recall"],e["activation"]
def dqn_gate(d,ag):
    roi=d["temporal_roi"].astype(bool); obs=(d["features"][roi].astype(np.float32)-zm)/zs
    e=sim(d,dqn_actions(ag,obs)); return e["event_recall"],e["activation"]
def boot(diff,B=1000,seed=42):
    rng=np.random.RandomState(seed); n=len(diff); m=[diff[rng.randint(0,n,n)].mean() for _ in range(B)]
    return float(np.mean(diff)), float(np.percentile(m,2.5)), float(np.percentile(m,97.5))
def periodic_recall_at(target_a):
    # choose R≈1/target_a; return per-video recall vector + actual mean activation
    R=max(1,round(1/target_a)); rec=[];act=[]
    for v in test:
        r,a=periodic(NP[v],R); rec.append(r);act.append(a)
    return np.array(rec),float(np.mean(act)),R

mRF=pickle.load(open("models/calib_S1_fold0_trainonly.pkl","rb")); mRF=mRF["model"] if isinstance(mRF,dict) else mRF

# oracle ceiling = always-on recall (activation 1)
orc=np.array([sim(NP[v],np.full(int(NP[v]["temporal_roi"].sum()),ACC,np.int32))["event_recall"] for v in test])

# det-gate RF
rf_rec=np.array([rf_gate(NP[v],mRF)[0] for v in test]); rf_act=float(np.mean([rf_gate(NP[v],mRF)[1] for v in test]))
pr_rf,pa_rf,Rrf=periodic_recall_at(rf_act); g_rf=boot(rf_rec-pr_rf)

results={"metric":"event_recall (detection-availability, W=7)","n_test_videos":len(test),
         "provenance":{"train_stats_active":tr["mean"]["active_frame_rate"],"calib":"random_forest","dqn":"t4b_deoracled"},
         "oracle_alwayson_recall_mean":float(orc.mean()),
         "det_gate_RF":{"activation":rf_act,"recall_mean":float(rf_rec.mean()),
            "periodic_matched":{"R":Rrf,"activation":pa_rf,"recall_mean":float(pr_rf.mean())},
            "gap_detgate_minus_periodic":{"mean":g_rf[0],"ci95":[g_rf[1],g_rf[2]]}},
         "RL_gate_by_lambda":[]}
print(f"PROVENANCE OK: train_stats active=0.0, RF (clean), DQN T4b de-oracled. test n={len(test)}")
print(f"oracle(always-on) recall = {orc.mean():.3f}")
print(f"\n{'config':<18}{'activation':>11}{'recall':>8}{'periodic@a':>11}{'gap':>8}{'ci95':>20}")
print(f"{'det-gate RF':<18}{rf_act:>11.3f}{rf_rec.mean():>8.3f}{pr_rf.mean():>11.3f}{g_rf[0]:>+8.3f}   [{g_rf[1]:+.3f},{g_rf[2]:+.3f}]")
stop=False
for lam in (0.05,0.10,0.20,0.40,0.80):
    seed_rec=[]
    for s in (0,1,2):
        ag=DQNAgent(seed=s); ag.load(f"outputs/rl/t4b/policy_fair_dqn_lam{lam:.2f}_seed{s}.pt"); ag.eps=0.0
        seed_rec.append(np.array([dqn_gate(NP[v],ag)[0] for v in test]))
        if s==0: act_lam=float(np.mean([dqn_gate(NP[v],ag)[1] for v in test]))
    rl_rec=np.mean(seed_rec,axis=0)  # seed-averaged per-video
    pr,pa,R=periodic_recall_at(act_lam); g=boot(rl_rec-pr)
    beat = g[1]>0  # CI lower bound > 0 => RL strictly beats periodic
    stop = stop or beat
    print(f"{'RL DQN λ='+str(lam):<18}{act_lam:>11.3f}{rl_rec.mean():>8.3f}{pr.mean():>11.3f}{g[0]:>+8.3f}   [{g[1]:+.3f},{g[2]:+.3f}]" + ("  <-- RL CI-BEATS!" if beat else ""))
    results["RL_gate_by_lambda"].append({"lambda":lam,"activation":act_lam,"recall_mean":float(rl_rec.mean()),
        "periodic_matched":{"R":R,"activation":pa,"recall_mean":float(pr.mean())},
        "gap_RL_minus_periodic":{"mean":g[0],"ci95":[g[1],g[2]],"ci_beats_periodic":bool(beat)}})
import os; os.makedirs("report",exist_ok=True)
json.dump(results,open("report/a3_learned_result.json","w"),indent=2)
print(f"\nGO/NO-GO: {'STOP — RL unexpectedly CI-beats periodic' if stop else 'GO — RL does NOT CI-beat periodic at equal energy (debunk holds for learned class)'}")
print("saved report/a3_learned_result.json")
