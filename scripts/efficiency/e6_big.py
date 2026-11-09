#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

import pandas as pd

METHODS = ("hybrid", "markov", "repeat", "pop")
MET = ("acc@1", "acc@5", "ndcg@10", "mrr")


def one(ds):
    from poibench import evaluate as ev
    queries, cands, venues, checkins = ev.load_split(str(C.ROOT), ds, "test", "H-100")
    preds = {m: pd.read_parquet(C.ROOT / "preds" / f"{m}@H-100" / ds / "test.parquet") for m in METHODS}
    preds["retriever"] = cands[["query_id", "cand_venue_ids"]].rename(columns={"cand_venue_ids": "ranked_venue_ids"})
    res = ev.evaluate(preds, queries, candidates=cands, venues=venues, B=1000, seed=0, ref="hybrid", pairs="ref",
                      strata=["all"], split="test")
    R = pd.DataFrame(res["results"])
    P = pd.DataFrame(res["pairs"])
    rows = []
    for m in preds:
        row = {"ds": ds, "method": m, "n_queries": res["n_queries"], "n_users": res["n_users"]}
        for mode in ("e2e", "cand"):
            for met in MET:
                r = R[(R.method == m) & (R["mode"] == mode) & (R.metric == met)].iloc[0]
                row[f"{mode}:{met}"], row[f"{mode}:{met}_ci_lo"], row[f"{mode}:{met}_ci_hi"] = r.value, r.ci_lo, r.ci_hi
            if m != "hybrid":
                p = P[(P.a == m) & (P["mode"] == mode) & (P.metric == "acc@1")].iloc[0]
                row[f"{mode}:dacc1_vs_hybrid"] = p["diff"]
                row[f"{mode}:dacc1_vs_hybrid_ci_lo"], row[f"{mode}:dacc1_vs_hybrid_ci_hi"] = p.ci_lo, p.ci_hi
        rows.append(row)
    print(ds, " ".join(f"{r['method']}={r['e2e:acc@1']:.4f}" for r in rows), flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(C.BIG + C.MAIN))
    a = ap.parse_args()
    rows = []
    for ds in a.datasets:
        rows += one(ds)
    df = C.write_csv("e6_bigds.csv", rows)
    ret = pd.read_csv(C.OUT / "e4_retriever.csv").set_index("ds")
    for ds in a.datasets:
        if ds in ret.index:
            print(ds, "recall@20", f"{ret.loc[ds, 'recall@20']:.4f}", "recall@100", f"{ret.loc[ds, 'recall@100']:.4f}")
    return df


if __name__ == "__main__":
    main()
