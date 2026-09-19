from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

os.environ.setdefault("OMP_NUM_THREADS","1")
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
os.environ.setdefault("MKL_NUM_THREADS","1")

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))

from frontier.phase1_projection import project_features
from frontier.phase5_featurebank import load_tier1_feature_bank
from frontier.phase5_domain_tools import deterministic_group_roles, no_group_leakage
from frontier.phase5_pairing import build_pair_selection
from frontier.phase5_summary_geometry import remap_classes, m3_distance, m3_upload_ratio
from frontier.phase5_eval import domain_risk_curve, bootstrap_gamma_ci
from frontier.phase4_eval import pair_decision_metrics
from frontier.phase5_gates import evaluate_phase5_gates
from frontier.phase5_reporting import write_report


def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()


def git_head() -> str:
    try:
        return subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True,stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "NO_GIT_HEAD"


def tracked_clean(rel: str) -> bool:
    a=subprocess.run(["git","ls-files","--error-unmatch","--",rel],cwd=ROOT,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    if a.returncode != 0: return False
    b=subprocess.run(["git","diff","--quiet","HEAD","--",rel],cwd=ROOT)
    return b.returncode == 0


def _authorities(protocol: dict) -> dict[str,str]:
    out={}
    for k,rel in protocol["authorities"].items():
        p=ROOT/rel
        if not p.exists(): raise RuntimeError(f"Missing Phase5 authority: {p}")
        out[k]=sha(p)
    return out


def _load_feature_manifest(protocol: dict):
    p=ROOT/protocol["paths"]["feature_manifest"]
    if not p.exists(): raise RuntimeError("Tier-1 feature manifest missing")
    m=json.loads(p.read_text(encoding="utf-8"))
    if m.get("status")!="FROZEN_AFTER_AUTHORIZED_UNBLIND":
        raise RuntimeError("Tier-1 feature manifest not frozen")
    entries={}
    for e in m["banks"]:
        q=ROOT/e["path"]
        if not q.exists(): raise FileNotFoundError(q)
        if sha(q)!=e["sha256"]: raise RuntimeError(f"Feature bank hash mismatch: {e['bank_id']}")
        z=dict(e); z["resolved_path"]=str(q); entries[e["bank_id"]]=z
    expected={
        f"{ds}__{enc}"
        for ds in protocol["tier1"]["datasets"]
        for enc in protocol["representations"]["encoders"]
    }
    if set(entries)!=expected:
        raise RuntimeError(f"Feature manifest IDs mismatch expected={sorted(expected)} actual={sorted(entries)}")
    return p,m,entries


def _smoke(protocol: dict):
    rng=np.random.default_rng(20260910)
    C,d=4,24
    rows=[]; X=[]; y=[]; dom=[]; grp=[]
    g=0
    for domain in range(3):
        for group in range(12):
            for c in range(C):
                n=8
                mu=np.zeros(d); mu[c]=1.5; mu[4]+=0.08*domain
                z=rng.normal(size=(n,d))+mu
                X.append(z); y.extend([c]*n); dom.extend([domain]*n); grp.extend([g]*n)
            g+=1
    X=np.vstack(X); y=np.asarray(y); dom=np.asarray(dom); grp=np.asarray(grp)
    p=json.loads(json.dumps(protocol))
    p["tier1"]["datasets"]["camelyon17"].update({
        "min_shared_classes":4,"min_fit_per_class":10,"min_eval_per_class":10,"max_shared_classes":4
    })
    sel=build_pair_selection("camelyon17","smoke__resnet18_imagenet1k_v1","resnet18_imagenet1k_v1",d,X,y,dom,grp,p)
    if not sel["selected_pairs"]:
        raise RuntimeError("Smoke pair selection returned no pairs")
    pr=sel["selected_pairs"][0]
    role=deterministic_group_roles("camelyon17",dom,grp,int(p["fixed"]["group_split_seed"]),float(p["fixed"]["group_fit_fraction"]))
    classes=pr["shared_classes"]
    a,b=pr["domain_a"],pr["domain_b"]
    af=(dom==a)&(role==0)&np.isin(y,classes); ae=(dom==a)&(role==1)&np.isin(y,classes)
    bf=(dom==b)&(role==0)&np.isin(y,classes); be=(dom==b)&(role==1)&np.isin(y,classes)
    Xaf,yaf,_=remap_classes(X[af],y[af],classes)
    Xae,yae,_=remap_classes(X[ae],y[ae],classes)
    Xbf,ybf,_=remap_classes(X[bf],y[bf],classes)
    Xbe,ybe,_=remap_classes(X[be],y[be],classes)
    ca,pa=domain_risk_curve(Xaf,yaf,Xae,yae,p["fixed"]["tau_grid"],p["fixed"]["alpha0"])
    cb,pb=domain_risk_curve(Xbf,ybf,Xbe,ybe,p["fixed"]["tau_grid"],p["fixed"]["alpha0"])
    met=pair_decision_metrics(ca,cb,p["fixed"]["tau_grid"])
    return {
        "selected_pairs":len(sel["selected_pairs"]),
        "no_group_leakage":sel["no_group_leakage"],
        "t2_distance":pr["t2_distance"],
        "gamma":met["deterministic_pair_minimax_regret"],
        "m3_distance":m3_distance(X[af],y[af],X[bf],y[bf],classes,p["fixed"]["m3_k"],p["fixed"]["m3_seed"]),
        "all_finite":bool(np.isfinite(ca).all() and np.isfinite(cb).all()),
    }


def _pair_freeze(protocol: dict, protocol_path: Path, authorities: dict):
    if not tracked_clean("configs/phase5_protocol.yaml"):
        raise RuntimeError("Pair freeze blocked: Phase5 protocol must be tracked and clean")
    mf_path,mf,entries=_load_feature_manifest(protocol)
    if not tracked_clean(str(mf_path.relative_to(ROOT)).replace("\\","/")):
        raise RuntimeError("Pair freeze blocked: feature manifest must be committed")
    pair_path=ROOT/protocol["paths"]["pair_manifest"]
    if pair_path.exists():
        raise RuntimeError(f"Refusing overwrite existing pair manifest: {pair_path}")

    conditions=[]
    for bid in sorted(entries):
        bank=load_tier1_feature_bank(entries[bid]["resolved_path"],entries[bid])
        for d in protocol["representations"]["projection_dims"]:
            Xp=project_features(bank.X,int(d),bank.encoder_id,int(protocol["representations"]["projection_seed"]))
            conditions.append(build_pair_selection(
                bank.dataset_id,bank.bank_id,bank.encoder_id,int(d),Xp,bank.y,bank.domain_id,bank.group_id,protocol
            ))
    payload={
        "freeze_type":"PHASE5_TIER1_PAIR_MANIFEST",
        "status":"FROZEN_BEFORE_RISK_EVALUATION",
        "generated_by_git_head":git_head(),
        "protocol_sha256":sha(protocol_path),
        "feature_manifest_sha256":sha(mf_path),
        "authority_sha256":authorities,
        "selection_uses_risk_outcomes":False,
        "selection_uses_m3":False,
        "conditions":conditions,
    }
    canon=json.dumps(payload,sort_keys=True,separators=(",",":")).encode()
    payload["payload_sha256"]=hashlib.sha256(canon).hexdigest()
    pair_path.parent.mkdir(parents=True,exist_ok=True)
    pair_path.write_text(json.dumps(payload,indent=2),encoding="utf-8")
    return payload


def _eval_one_condition(bank, Xp, cond: dict, protocol: dict):
    roles=deterministic_group_roles(
        bank.dataset_id,bank.domain_id,bank.group_id,
        int(protocol["fixed"]["group_split_seed"]),float(protocol["fixed"]["group_fit_fraction"])
    )
    taus=list(map(float,protocol["fixed"]["tau_grid"]))
    rows=[]; curves=[]
    for pr in cond["selected_pairs"]:
        classes=list(map(int,pr["shared_classes"]))
        a,b=int(pr["domain_a"]),int(pr["domain_b"])
        af=(bank.domain_id==a)&(roles==0)&np.isin(bank.y,classes)
        ae=(bank.domain_id==a)&(roles==1)&np.isin(bank.y,classes)
        bf=(bank.domain_id==b)&(roles==0)&np.isin(bank.y,classes)
        be=(bank.domain_id==b)&(roles==1)&np.isin(bank.y,classes)

        Xaf,yaf,_=remap_classes(Xp[af],bank.y[af],classes)
        Xae,yae,_=remap_classes(Xp[ae],bank.y[ae],classes)
        Xbf,ybf,_=remap_classes(Xp[bf],bank.y[bf],classes)
        Xbe,ybe,_=remap_classes(Xp[be],bank.y[be],classes)

        ca,pa=domain_risk_curve(Xaf,yaf,Xae,yae,taus,float(protocol["fixed"]["alpha0"]))
        cb,pb=domain_risk_curve(Xbf,ybf,Xbe,ybe,taus,float(protocol["fixed"]["alpha0"]))
        met=pair_decision_metrics(ca,cb,taus)
        ci_lo,ci_hi=bootstrap_gamma_ci(
            yae,pa,ybe,pb,taus,
            int(protocol["fixed"]["bootstrap_replicates"]),
            int(protocol["fixed"]["bootstrap_seed"]),
        )
        dm3=m3_distance(
            Xp[af],bank.y[af],Xp[bf],bank.y[bf],classes,
            int(protocol["fixed"]["m3_k"]),int(protocol["fixed"]["m3_seed"])
        )
        robust=bool(
            met["deterministic_pair_minimax_regret"] >= 0.005
            and ci_lo >= 0.0025
            and met["optimal_sets_disjoint"]
        )
        row={
            "condition_id":cond["condition_id"],
            "dataset_id":bank.dataset_id,
            "bank_id":bank.bank_id,
            "encoder_id":bank.encoder_id,
            "projection_dim":int(cond["projection_dim"]),
            "pair_id":pr["pair_id"],
            "domain_a":a,"domain_b":b,
            "selection_rank":int(pr["selection_rank"]),
            "eligible_pair_count":int(pr["eligible_pair_count"]),
            "selected_pair_count":int(pr["selected_pair_count"]),
            "n_shared_classes":int(pr["n_shared_classes"]),
            "t2_distance":float(pr["t2_distance"]),
            "t2_empirical_percentile":float(pr["t2_empirical_percentile"]),
            "m3_distance":float(dm3),
            **met,
            "gamma_bootstrap_ci95_low":float(ci_lo),
            "gamma_bootstrap_ci95_high":float(ci_hi),
            "robust_decision_critical":robust,
            "m3_upload_ratio_vs_t2":m3_upload_ratio(int(cond["projection_dim"]),int(protocol["fixed"]["m3_k"])),
            "no_group_leakage":bool(no_group_leakage(bank.domain_id,bank.group_id,roles)),
        }
        rows.append(row)
        for task,curve in [("domain_a",ca),("domain_b",cb)]:
            for tau,val in zip(taus,curve):
                curves.append({
                    "condition_id":cond["condition_id"],
                    "pair_id":pr["pair_id"],
                    "task":task,
                    "tau":float(tau),
                    "balanced_accuracy":float(val),
                })
    return rows,curves


def _formal(protocol: dict, protocol_path: Path, authorities: dict, out: Path, resume: bool):
    pair_path=ROOT/protocol["paths"]["pair_manifest"]
    if not pair_path.exists(): raise RuntimeError("Tier1 formal blocked: pair manifest missing")
    pair=json.loads(pair_path.read_text(encoding="utf-8"))
    if pair.get("status")!="FROZEN_BEFORE_RISK_EVALUATION":
        raise RuntimeError("Pair manifest invalid")
    pair_rel=str(pair_path.relative_to(ROOT)).replace("\\","/")
    if not tracked_clean(pair_rel):
        raise RuntimeError("Tier1 formal blocked: pair manifest must be committed and clean")
    mf_path,mf,entries=_load_feature_manifest(protocol)

    rows=[]; curves=[]
    shard_dir=out/"shards"; shard_dir.mkdir(parents=True,exist_ok=True)
    conds={c["condition_id"]:c for c in pair["conditions"]}
    for cid in sorted(conds):
        cond=conds[cid]
        shard=shard_dir/f"{cid}.json"
        if resume and shard.exists():
            z=json.loads(shard.read_text(encoding="utf-8"))
            rows.extend(z["rows"]); curves.extend(z["curves"]); continue
        bank=load_tier1_feature_bank(entries[cond["bank_id"]]["resolved_path"],entries[cond["bank_id"]])
        Xp=project_features(bank.X,int(cond["projection_dim"]),bank.encoder_id,int(protocol["representations"]["projection_seed"]))
        r,c=_eval_one_condition(bank,Xp,cond,protocol)
        shard.write_text(json.dumps({"rows":r,"curves":c},ensure_ascii=False),encoding="utf-8")
        rows.extend(r); curves.extend(c)

    df=pd.DataFrame(rows)
    cv=pd.DataFrame(curves)
    integrity={
        "manifest_ok":True,
        "pair_manifest_ok":True,
        "no_group_leakage":bool(df["no_group_leakage"].all()) if len(df) else False,
    }
    gate=evaluate_phase5_gates(df,protocol,integrity)
    return df,cv,gate,sha(mf_path),sha(pair_path)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--mode",choices=["smoke","pair-freeze","tier1-formal"],required=True)
    ap.add_argument("--resume",action="store_true")
    ap.add_argument("--protocol",default=str(ROOT/"configs"/"phase5_protocol.yaml"))
    ap.add_argument("--out-dir",default=str(ROOT/"runs"/"phase5"))
    args=ap.parse_args()

    protocol_path=Path(args.protocol)
    protocol=yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("version")!="1.0.0-FROZEN" or protocol.get("status")!="FROZEN_BEFORE_TIER1_OUTCOME_ACCESS":
        raise RuntimeError("Phase5 protocol not frozen")
    authorities=_authorities(protocol)
    out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    gate=None; mf_hash=None; pair_hash=None

    if args.mode=="smoke":
        smoke=_smoke(protocol)
        summary={
            "mode":"smoke","formal":False,"complete":bool(smoke["all_finite"] and smoke["no_group_leakage"]),
            "git_head":git_head(),"protocol_sha256":sha(protocol_path),
            "authority_sha256":authorities,"external_outcome_blackout":"ACTIVE",
            "smoke":smoke,
        }
    elif args.mode=="pair-freeze":
        auth=ROOT/protocol["paths"]["authorization"]
        if not auth.exists(): raise RuntimeError("Pair freeze blocked: unblind authorization missing")
        payload=_pair_freeze(protocol,protocol_path,authorities)
        summary={
            "mode":"pair-freeze","formal_pair_selection":True,"complete":True,
            "git_head":git_head(),"protocol_sha256":sha(protocol_path),
            "feature_manifest_sha256":payload["feature_manifest_sha256"],
            "pair_manifest_payload_sha256":payload["payload_sha256"],
            "authority_sha256":authorities,
            "risk_evaluation_executed":False,
        }
    else:
        auth=ROOT/protocol["paths"]["authorization"]
        if not auth.exists(): raise RuntimeError("Tier1 formal blocked: unblind authorization missing")
        df,cv,gate,mf_hash,pair_hash=_formal(protocol,protocol_path,authorities,out,args.resume)
        df.to_csv(out/"pair_results.csv",index=False)
        cv.to_csv(out/"risk_curves.csv",index=False)
        (out/"gate_summary.json").write_text(json.dumps(gate,indent=2),encoding="utf-8")
        pair=json.loads((ROOT/protocol["paths"]["pair_manifest"]).read_text(encoding="utf-8"))
        expected=sum(c["n_selected_pairs"] for c in pair["conditions"])
        summary={
            "mode":"tier1-formal","formal":True,
            "n_selected_pair_rows":int(len(df)),"expected_selected_pair_rows":int(expected),
            "n_risk_curve_rows":int(len(cv)),"expected_risk_curve_rows":int(expected*2*len(protocol["fixed"]["tau_grid"])),
            "complete":bool(len(df)==expected and len(cv)==expected*2*len(protocol["fixed"]["tau_grid"])),
            "git_head":git_head(),"protocol_sha256":sha(protocol_path),
            "feature_manifest_sha256":mf_hash,"pair_manifest_sha256":pair_hash,
            "authorization_sha256":sha(auth),"authority_sha256":authorities,
            "external_outcome_blackout":"OPENED_BY_COMMITTED_AUTHORIZATION",
        }

    (out/"run_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    (out/"environment_manifest.json").write_text(json.dumps({
        "python":sys.version,"platform":platform.platform(),"numpy":np.__version__,"pandas":pd.__version__
    },indent=2),encoding="utf-8")
    (out/"protocol_snapshot.yaml").write_text(protocol_path.read_text(encoding="utf-8"),encoding="utf-8")
    report=ROOT/"reports"/"phase5"/f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_phase5_{args.mode}_experiment_report.md"
    write_report(report,summary,gate," ".join(sys.argv))
    print(json.dumps(summary,indent=2))
    if gate is not None: print(json.dumps(gate,indent=2))
    print(f"REPORT={report}")

if __name__=="__main__":
    main()
