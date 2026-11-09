#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C


def one(ds: str, policy: str) -> dict:
    import resource

    import numpy as np
    import pandas as pd

    from poibench.baselines_simple import SimpleBaselines
    from poibench.evaluate import candidates_path, load_policy_json

    d = C.ROOT / "data" / ds
    t0 = time.perf_counter()
    ck = pd.read_parquet(d / "checkins.parquet")
    venues = pd.read_parquet(d / "venues.parquet")
    q = pd.read_parquet(d / "queries_test.parquet")
    pol = load_policy_json(str(C.ROOT), ds, policy)
    cands = pd.read_parquet(candidates_path(str(C.ROOT), ds, "test", pol))
    load_s = time.perf_counter() - t0
    fits = []
    for _ in range(3):
        t0 = time.perf_counter()
        model = SimpleBaselines(ck, venues, policy=pol)
        fits.append(time.perf_counter() - t0)
    state = model.pop.nbytes + model.pop_order.nbytes + sum(a.nbytes + b.nbytes for a, b in model.markov.values())
    log = sum(getattr(model.A, k).nbytes for k in ("user", "venue", "t", "traj", "split", "traj_start"))
    t0 = time.perf_counter()
    pr = model.predict("hybrid", q, cands, topk=50, n_jobs=1)
    pred_s = time.perf_counter() - t0
    ref = pd.read_parquet(C.ROOT / "preds" / f"hybrid@{policy}" / ds / "test.parquet")
    rm = dict(zip(ref["query_id"], ref["ranked_venue_ids"]))
    match = float(np.mean([list(rm[a]) == list(b) for a, b in zip(pr["query_id"], pr["ranked_venue_ids"])]))
    rng = np.random.default_rng(0)
    idx = rng.choice(len(q), size=min(2000, len(q)), replace=False)
    cm = cands.set_index("query_id")
    lat = []
    for i in idx:
        qq = q.iloc[[i]]
        cc = cm.loc[[qq["query_id"].iloc[0]]].reset_index()
        t0 = time.perf_counter()
        model.predict("hybrid", qq, cc, topk=50, n_jobs=1)
        lat.append(time.perf_counter() - t0)
    lat = np.array(lat) * 1000
    return {"ds": ds, "policy": policy, "n_checkins_total": len(ck), "n_visible_train_checkins": int(model.pop_visible.sum()),
            "n_transitions": model.n_transitions, "n_venues": model.V, "n_users": int(len(model.A.user_first)),
            "n_test_queries": len(q), "load_s": load_s, "fit_s": float(np.median(fits)), "fit_s_all": json.dumps(fits),
            "predict_s": pred_s, "predict_ms_per_query": 1000 * pred_s / len(q), "throughput_qps_1core": len(q) / pred_s,
            "lat_p50_ms": float(np.percentile(lat, 50)), "lat_p95_ms": float(np.percentile(lat, 95)),
            "lat_p99_ms": float(np.percentile(lat, 99)), "lat_mean_ms": float(lat.mean()),
            "mean_visible_own_test": float(pr["n_visible_own"].mean()),
            "state_mb": state / 2 ** 20, "log_mb": log / 2 ** 20,
            "peak_rss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20, "match_preds": match,
            **{f"hw_{k}": v for k, v in C.hardware().items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(C.MAIN + C.BIG))
    ap.add_argument("--policies", nargs="+", default=["H-100", "H-10"])
    ap.add_argument("--child", nargs=2)
    a = ap.parse_args()
    if a.child:
        print("RESULT " + json.dumps(one(*a.child)))
        return
    rows = []
    for ds in a.datasets:
        for pol in a.policies:
            if not (C.ROOT / "preds" / f"hybrid@{pol}" / ds / "test.parquet").exists():
                continue
            env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
            p = subprocess.run([sys.executable, "-I", __file__, "--child", ds, pol], capture_output=True, text=True, env=env)
            line = [x for x in p.stdout.splitlines() if x.startswith("RESULT ")]
            if p.returncode or not line:
                print(f"{ds} {pol} FAILED rc={p.returncode}\n{p.stderr[-2000:]}")
                continue
            r = json.loads(line[0][7:])
            print(f"{ds:18s} {pol:6s} ck={r['n_checkins_total']:>9d} fit={r['fit_s']:.3f}s pred={r['predict_ms_per_query']:.4f}ms/q "
                  f"p50={r['lat_p50_ms']:.3f} p95={r['lat_p95_ms']:.3f} state={r['state_mb']:.1f}MB rss={r['peak_rss_gb']:.2f}GB "
                  f"match={r['match_preds']:.4f}", flush=True)
            rows.append(r)
    C.write_csv("e1_hybrid_bench.csv", rows)


if __name__ == "__main__":
    main()
