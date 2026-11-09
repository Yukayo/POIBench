#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

DS = "tky"


def root_of(frac: int) -> Path:
    return C.ROOT if frac == 100 else C.ROOT / "work" / "efficiency_scale" / f"u{frac}"


def hybrid(frac: int) -> dict:
    import numpy as np
    import pandas as pd
    from poibench.baselines_simple import SimpleBaselines
    from poibench.evaluate import candidates_path, load_policy_json
    sr = root_of(frac)
    d = sr / "data" / DS
    t0 = time.perf_counter()
    ck = pd.read_parquet(d / "checkins.parquet")
    venues = pd.read_parquet(d / "venues.parquet")
    q = pd.read_parquet(d / "queries_test.parquet")
    pol = load_policy_json(str(sr), DS, "H-100")
    cands = pd.read_parquet(candidates_path(str(sr), DS, "test", pol))
    load_s = time.perf_counter() - t0
    fits = []
    for _ in range(3):
        t0 = time.perf_counter()
        model = SimpleBaselines(ck, venues, policy=pol)
        fits.append(time.perf_counter() - t0)
    state = model.pop.nbytes + model.pop_order.nbytes + sum(a.nbytes + b.nbytes for a, b in model.markov.values())
    t0 = time.perf_counter()
    pr = model.predict("hybrid", q, cands, topk=50, n_jobs=1)
    pred_s = time.perf_counter() - t0
    out = sr / "preds" / "hybrid@H-100" / DS
    match = None
    if frac == 100:
        ref = pd.read_parquet(out / "test.parquet")
        rm = dict(zip(ref["query_id"], ref["ranked_venue_ids"]))
        match = float(np.mean([list(rm[a]) == list(b) for a, b in zip(pr["query_id"], pr["ranked_venue_ids"])]))
    else:
        out.mkdir(parents=True, exist_ok=True)
        pr[["query_id", "ranked_venue_ids"]].to_parquet(out / "test.parquet", index=False)
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
    return {"component": "hybrid", "frac": frac, "n_checkins": len(ck), "n_test_queries": len(q), "load_s": load_s,
            "build_s": float(np.median(fits)), "build_s_all": json.dumps(fits), "predict_s": pred_s,
            "batch_ms_per_query": 1000 * pred_s / len(q), "lat_p50_ms": float(np.percentile(lat, 50)),
            "lat_p95_ms": float(np.percentile(lat, 95)), "lat_p99_ms": float(np.percentile(lat, 99)),
            "state_mb": state / 2 ** 20, "match_preds_100": match,
            "peak_rss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20,
            **{f"hw_{k}": v for k, v in C.hardware().items()}}


def retriever(frac: int) -> dict:
    import numpy as np
    import pandas as pd
    from poibench import data as D
    sr = root_of(frac)
    d = sr / "data" / DS
    ck = pd.read_parquet(d / "checkins.parquet")
    venues = pd.read_parquet(d / "venues.parquet")
    q = pd.read_parquet(d / "queries_test.parquet", columns=["query_id", "target_row_id"])
    stored = pd.read_parquet(d / "candidates_test.parquet")
    A = D.Arrays.from_checkins(ck)
    builds = []
    for _ in range(3):
        t0 = time.perf_counter()
        st = D.prepare_retriever(A, venues, None)
        builds.append(time.perf_counter() - t0)
    idx_bytes = (st["pop"].nbytes + st["knn"].nbytes + st["cat_mask"].nbytes + st["global_order"].nbytes
                 + sum(a.nbytes + b.nbytes for a, b in st["trans"].values()))
    r = q["target_row_id"].to_numpy(np.int64)
    sl = D.evidence_slices(r, A, None)
    h = D.RETRIEVER["half_life_days"] * 86400.0
    t0g = int(A.t.min())
    smap = dict(zip(stored["query_id"].to_numpy(np.int64), stored["cand_venue_ids"]))
    smp = np.random.default_rng(0).choice(len(r), size=min(2000, len(r)), replace=False)
    lat, same = [], []
    for i in smp:
        t0 = time.perf_counter()
        a, b, c, e = sl["hist_lo"][i], sl["hist_hi"][i], sl["prefix_lo"][i], sl["prefix_hi"][i]
        ix = np.r_[np.arange(a, b), np.arange(c, e)]
        out = D._retrieve_one(int(A.venue[r[i] - 1]), A.venue[ix], np.exp2((A.t[ix] - t0g) / h), -1, st)
        lat.append(time.perf_counter() - t0)
        same.append(list(out) == list(smap[int(q["query_id"].iloc[i])]))
    lat = np.array(lat) * 1000
    return {"component": "retriever", "frac": frac, "n_checkins": len(ck), "n_test_queries": len(r),
            "n_catalog": int(st["cat_mask"].sum()), "build_s": float(np.median(builds)), "build_s_all": json.dumps(builds),
            "index_mb": idx_bytes / 2 ** 20, "lat_p50_ms": float(np.percentile(lat, 50)),
            "lat_p95_ms": float(np.percentile(lat, 95)), "lat_p99_ms": float(np.percentile(lat, 99)),
            "batch_ms_per_query": float(lat.mean()), "stored_match_sample": float(np.mean(same)),
            "peak_rss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20,
            **{f"hw_{k}": v for k, v in C.hardware().items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fracs", type=int, nargs="+", default=[20, 40, 60, 80, 100])
    ap.add_argument("--child", nargs=2)
    a = ap.parse_args()
    if a.child:
        f = {"hybrid": hybrid, "retriever": retriever}[a.child[0]]
        print("RESULT " + json.dumps(f(int(a.child[1]))))
        return
    rows = []
    for frac in a.fracs:
        for comp in ("hybrid", "retriever"):
            env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
            p = subprocess.run([sys.executable, "-I", __file__, "--child", comp, str(frac)], capture_output=True,
                               text=True, env=env)
            line = [x for x in p.stdout.splitlines() if x.startswith("RESULT ")]
            if p.returncode or not line:
                print(f"{comp} {frac} FAILED rc={p.returncode}\n{p.stderr[-2000:]}", flush=True)
                continue
            r = json.loads(line[0][7:])
            print(f"{comp:9s} {frac:3d}% ck={r['n_checkins']:>7d} build={r['build_s']:.3f}s "
                  f"q={r['batch_ms_per_query']:.4f}ms p50={r['lat_p50_ms']:.3f}ms rss={r['peak_rss_gb']:.2f}GB", flush=True)
            rows.append(r)
    C.write_csv("e7_scale_cpu.csv", rows)


if __name__ == "__main__":
    main()
