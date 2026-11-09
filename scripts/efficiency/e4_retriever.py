#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

import numpy as np
import pandas as pd

KS = (5, 10, 20, 50, 100)


def one(ds):
    from poibench import data as D
    from poibench.evaluate import assert_split_allowed
    assert_split_allowed(str(C.ROOT), "test")
    d = C.ROOT / "data" / ds
    ck = pd.read_parquet(d / "checkins.parquet")
    venues = pd.read_parquet(d / "venues.parquet")
    q = pd.read_parquet(d / "queries_test.parquet")
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

    def retrieve(i):
        a, b, c, e = sl["hist_lo"][i], sl["hist_hi"][i], sl["prefix_lo"][i], sl["prefix_hi"][i]
        ix = np.r_[np.arange(a, b), np.arange(c, e)]
        return D._retrieve_one(int(A.venue[r[i] - 1]), A.venue[ix], np.exp2((A.t[ix] - t0g) / h), -1, st)

    smp = np.random.default_rng(0).choice(len(r), size=min(2000, len(r)), replace=False)
    smap = dict(zip(stored["query_id"].to_numpy(np.int64), stored["cand_venue_ids"]))
    lat, same = [], []
    for i in smp:
        t0 = time.perf_counter()
        out = retrieve(i)
        lat.append(time.perf_counter() - t0)
        same.append(list(out) == list(smap[int(q["query_id"].iloc[i])]))
    lat = np.array(lat) * 1000
    old = D.RETRIEVER["out_len"]
    D.RETRIEVER["out_len"] = 100
    try:
        t0 = time.perf_counter()
        c100 = [retrieve(i) for i in range(len(r))]
        t100 = time.perf_counter() - t0
    finally:
        D.RETRIEVER["out_len"] = old
    tgt = q["target_venue_id"].to_numpy(np.int64)
    pref = np.mean([list(c[:50]) == list(smap[int(qid)]) for c, qid in zip(c100, q["query_id"].to_numpy(np.int64))])
    rec = {}
    for k in KS:
        if k <= 50:
            rec[f"recall@{k}"] = float(np.mean([t in smap[int(qid)][:k] for t, qid in zip(tgt, q["query_id"].to_numpy(np.int64))]))
        else:
            rec[f"recall@{k}"] = float(np.mean([t in c[:k] for t, c in zip(tgt, c100)]))
    row = {"ds": ds, "n_checkins": len(ck), "n_visible_train_checkins": int(A.is_train.sum()),
           "n_catalog_venues": int(st["cat_mask"].sum()), "n_test_queries": len(r), "build_s": float(np.median(builds)),
           "index_mb": idx_bytes / 2 ** 20, "lat_p50_ms": float(np.percentile(lat, 50)),
           "lat_p95_ms": float(np.percentile(lat, 95)), "lat_mean_ms": float(lat.mean()),
           "qps_1core": 1000 / float(lat.mean()), "out100_ms_per_query": 1000 * t100 / len(r),
           "stored50_match_sample": float(np.mean(same)), "prefix_match_out100": float(pref), **rec,
           **{f"hw_{k}": v for k, v in C.hardware().items()}}
    print(f"{ds:18s} build={row['build_s']:.2f}s idx={row['index_mb']:.1f}MB p50={row['lat_p50_ms']:.3f}ms "
          f"match={row['stored50_match_sample']:.4f}/{pref:.4f} " + " ".join(f"{k}={v:.4f}" for k, v in rec.items()), flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(C.MAIN + C.BIG))
    a = ap.parse_args()
    C.write_csv("e4_retriever.csv", [one(ds) for ds in a.datasets])


if __name__ == "__main__":
    main()
