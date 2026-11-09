#!/usr/bin/env python
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

import pandas as pd

from poibench import ROOT
from poibench.baselines_simple import METHODS, TT_METHODS, SimpleBaselines
from poibench.cost import CostRecorder
from poibench.evaluate import candidates_path, load_policy_json


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("ds")
    ap.add_argument("split", choices=["train", "val", "test"])
    ap.add_argument("evidence_policy")
    ap.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS) + list(TT_METHODS),
                    help="hybrid-tt (exploratory, labelled) needs exports_ablation/<ds>/target_time_<split>.parquet")
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--n-jobs", type=int, default=8, help="worker processes (capped at 24)")
    ap.add_argument("--topk", type=int, default=50)
    a = ap.parse_args(argv)

    d = os.path.join(a.root, "data", a.ds)
    setup = CostRecorder()
    with setup.stage("load_data", one_time=True, notes="shared by all simple baselines of this run"):
        checkins = pd.read_parquet(os.path.join(d, "checkins.parquet"))
        venues = pd.read_parquet(os.path.join(d, "venues.parquet"))
        queries = pd.read_parquet(os.path.join(d, f"queries_{a.split}.parquet"))
        policy = load_policy_json(a.root, a.ds, a.evidence_policy)
        cpath = candidates_path(a.root, a.ds, a.split, policy)
        cands = pd.read_parquet(cpath) if os.path.exists(cpath) else None
    with setup.stage("fit_counts", one_time=True, notes="visible popularity + transition counts, shared"):
        model = SimpleBaselines(checkins, venues, policy=policy)
    queries = model.allowed_queries(queries)
    if cands is None:
        print(f"[run_simple] WARNING: {cpath} missing; writing end-to-end lists only")

    for m in a.methods:
        name = f"{m}@{a.evidence_policy}"
        out_dir = os.path.join(a.root, "preds", name, a.ds)
        os.makedirs(out_dir, exist_ok=True)
        rec = CostRecorder(method=name, ds=a.ds, split=a.split,
                           extra={"baseline": m, "evidence_policy": a.evidence_policy, "topk": a.topk, "seed": 0,
                                  "candidates_file": os.path.relpath(cpath, a.root) if cands is not None else None,
                                  "model_summary": model.summary()})
        rec.stages.extend(dict(s) for s in setup.stages)
        with rec.stage("predict", one_time=False, n_items=len(queries), n_gpus=0):
            tt = None
            if m in TT_METHODS:
                if a.split == "train":
                    raise SystemExit(f"{m} is not defined on the train split")
                tt = pd.read_parquet(os.path.join(a.root, "exports_ablation", a.ds, f"target_time_{a.split}.parquet"))
            preds = model.predict(m, queries, cands, topk=a.topk, n_jobs=a.n_jobs, target_time=tt)
        path = os.path.join(out_dir, f"{a.split}.parquet")
        preds.to_parquet(path + ".tmp", index=False)
        os.replace(path + ".tmp", path)
        rec.save(os.path.join(out_dir, f"{a.split}.cost.json"), n_queries=len(queries))
        print(f"[run_simple] {name} {a.ds}/{a.split}: {len(preds)} queries -> {path} "
              f"(predict {rec.stages[-1]['wall_s']:.1f}s)")


if __name__ == "__main__":
    main()
