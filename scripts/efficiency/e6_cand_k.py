#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

import numpy as np
import pandas as pd

KS = (5, 10, 20, 50, 100)
METHODS = ("pop", "repeat", "markov", "hybrid", "getnext", "getnext-zfeat", "graph-flashback", "ds-llm4poi-lcs",
           "llm4poi", "gnpr-sid")
OUTM = ("acc@1", "acc@5", "ndcg@10", "mrr")


def cand100(ds):
    p = C.OUT / "e6_cand100" / f"{ds}.parquet"
    if p.exists():
        return pd.read_parquet(p)
    from poibench import data as D
    from poibench.evaluate import assert_split_allowed
    assert_split_allowed(str(C.ROOT), "test")
    d = C.ROOT / "data" / ds
    ck = pd.read_parquet(d / "checkins.parquet")
    venues = pd.read_parquet(d / "venues.parquet")
    q = pd.read_parquet(d / "queries_test.parquet")
    stored = pd.read_parquet(d / "candidates_test.parquet")
    A = D.Arrays.from_checkins(ck)
    st = D.prepare_retriever(A, venues, None)
    r = q["target_row_id"].to_numpy(np.int64)
    sl = D.evidence_slices(r, A, None)
    h = D.RETRIEVER["half_life_days"] * 86400.0
    t0g = int(A.t.min())
    old = D.RETRIEVER["out_len"]
    D.RETRIEVER["out_len"] = 100
    try:
        out = []
        for i in range(len(r)):
            a, b, c, e = sl["hist_lo"][i], sl["hist_hi"][i], sl["prefix_lo"][i], sl["prefix_hi"][i]
            ix = np.r_[np.arange(a, b), np.arange(c, e)]
            out.append([int(x) for x in D._retrieve_one(int(A.venue[r[i] - 1]), A.venue[ix],
                                                        np.exp2((A.t[ix] - t0g) / h), -1, st)])
    finally:
        D.RETRIEVER["out_len"] = old
    smap = dict(zip(stored["query_id"].to_numpy(np.int64), stored["cand_venue_ids"]))
    qid = q["query_id"].to_numpy(np.int64)
    bad = sum(list(o[:50]) != [int(x) for x in smap[int(k)]] for o, k in zip(out, qid))
    assert bad == 0, f"{ds}: {bad} lists differ from the stored candidates in their first 50"
    df = pd.DataFrame({"query_id": qid, "cand_venue_ids": out})
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False)
    print(f"{ds}: 100-candidate lists for {len(df)} test queries, first 50 = stored (all)", flush=True)
    return df


def one(ds):
    from poibench import evaluate as ev
    ev.assert_split_allowed(str(C.ROOT), "test")
    queries, _, venues, checkins = ev.load_split(str(C.ROOT), ds, "test", "H-100")
    c100 = cand100(ds)
    preds = {}
    for m in METHODS:
        p = C.ROOT / "preds" / f"{m}@H-100" / ds / "test.parquet"
        if p.exists():
            preds[m] = pd.read_parquet(p)
    preds["retriever"] = c100.rename(columns={"cand_venue_ids": "ranked_venue_ids"})
    rows, perq = [], {}
    W = None
    for k in KS:
        es = ev.build_eval_set(queries, c100, venues, cand_k=k, checkins=checkins)
        if W is None:
            W = ev.bootstrap_weights(es.n_users, B=1000, seed=0)
            Cn = np.bincount(es.user_codes, minlength=es.n_users).astype(float)
            WC = W @ Cn
        hit = es.strata["cand=hit"].astype(float)
        S = np.bincount(es.user_codes, weights=hit, minlength=es.n_users)
        rb = (W @ S) / WC
        rec = (float(hit.mean()), *np.percentile(rb, [2.5, 97.5]))
        for m, df in preds.items():
            ranks, diag = ev.per_query_ranks(es, df)
            vals = ev.metrics_from_ranks(ranks["cand"])
            perq[(m, k)] = vals
            S = np.stack([np.bincount(es.user_codes, weights=vals[:, j], minlength=es.n_users)
                          for j in range(vals.shape[1])], axis=1)
            bt = (W @ S) / WC[:, None]
            row = {"ds": ds, "method": m, "k": k, "recall@k": rec[0], "recall_ci_lo": rec[1], "recall_ci_hi": rec[2],
                   "fill_rate": diag.get("cand_fill_rate"), "n_queries": len(es.queries)}
            for met in OUTM:
                j = ev.METRICS.index(met)
                row[met] = float(vals[:, j].mean())
                row[f"{met}_ci_lo"], row[f"{met}_ci_hi"] = (float(x) for x in np.percentile(bt[:, j], [2.5, 97.5]))
            row["share_of_ceiling"] = row["acc@1"] / rec[0] if rec[0] else np.nan
            row["_bt"] = bt[:, ev.METRICS.index("acc@1")]
            rows.append(row)
    idx = {(r["method"], r["k"]): r for r in rows}
    for r in rows:
        b20 = idx[(r["method"], 20)]["_bt"]
        d = r["_bt"] - b20
        r["dacc1_vs_k20"] = r["acc@1"] - idx[(r["method"], 20)]["acc@1"]
        r["dacc1_vs_k20_ci_lo"], r["dacc1_vs_k20_ci_hi"] = (float(x) for x in np.percentile(d, [2.5, 97.5]))
        if ("hybrid", r["k"]) in idx:
            hb = idx[("hybrid", r["k"])]
            d = r["_bt"] - hb["_bt"]
            r["dacc1_vs_hybrid"] = r["acc@1"] - hb["acc@1"]
            r["dacc1_vs_hybrid_ci_lo"], r["dacc1_vs_hybrid_ci_hi"] = (float(x) for x in np.percentile(d, [2.5, 97.5]))
    for r in rows:
        r.pop("_bt")
    print(ds, " ".join(f"{m}:{idx[(m, 20)]['acc@1']:.4f}->{idx[(m, 100)]['acc@1']:.4f}" for m in preds), flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(C.MAIN))
    a = ap.parse_args()
    rows = []
    for ds in a.datasets:
        rows += one(ds)
    C.write_csv("e6_cand_k.csv", rows)


if __name__ == "__main__":
    main()
