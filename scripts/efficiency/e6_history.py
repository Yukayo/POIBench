#!/usr/bin/env python
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

import numpy as np
import pandas as pd

EDGES = (1, 10, 30, 100, 300, 1000, 10 ** 9)
LABELS = ("1-9", "10-29", "30-99", "100-299", "300-999", ">=1000")
NPER = 200
LLM = {"ds-llm4poi-lcs": "work/ds_llm4poi/runs/{ds}/H-100/lcs/responses_test.parquet",
       "llm4poi": "work/llm4poi/{ds}/H-100/default/responses_test.parquet"}


def binof(n):
    return np.searchsorted(np.array(EDGES[1:]), n, side="right")


def one(ds):
    from poibench import data as D
    from poibench.baselines_simple import SimpleBaselines
    from poibench.evaluate import candidates_path, load_policy_json
    d = C.ROOT / "data" / ds
    ck = pd.read_parquet(d / "checkins.parquet")
    venues = pd.read_parquet(d / "venues.parquet")
    q = pd.read_parquet(d / "queries_test.parquet")
    qi = pd.read_parquet(C.ROOT / "exports" / ds / "H-100" / "query_inputs_test.parquet")[["query_id", "n_visible_own"]]
    q = q.merge(qi, on="query_id", how="left", validate="one_to_one")
    assert q["n_visible_own"].notna().all()
    pol = load_policy_json(str(C.ROOT), ds, "H-100")
    cands = pd.read_parquet(candidates_path(str(C.ROOT), ds, "test", pol)).set_index("query_id")
    model = SimpleBaselines(ck, venues, policy=pol)
    A = model.A
    ref = pd.read_parquet(C.ROOT / "preds" / "hybrid@H-100" / ds / "test.parquet").set_index("query_id")
    r_all = q["target_row_id"].to_numpy(np.int64)
    sl = D.evidence_slices(r_all, A, None)
    st = D.prepare_retriever(A, venues, None)
    h = D.RETRIEVER["half_life_days"] * 86400.0
    t0g = int(A.t.min())
    b = binof(q["n_visible_own"].to_numpy())
    rng = np.random.default_rng(0)
    pick = []
    for k in range(len(LABELS)):
        ix = np.flatnonzero(b == k)
        if len(ix):
            pick += sorted(rng.choice(ix, size=min(NPER, len(ix)), replace=False).tolist())
    rows = []
    pc = time.perf_counter
    for i in pick:
        r = r_all[i]
        vis = np.concatenate([A.venue[sl["hist_lo"][i]:sl["hist_hi"][i]], A.venue[sl["prefix_lo"][i]:sl["prefix_hi"][i]]])
        prev = int(A.venue[r - 1])
        t0 = pc()
        pool = model._pool("hybrid", vis, prev, 50)
        out = model._order("hybrid", pool, vis, prev, -1, False)[:50]
        core = pc() - t0
        qid = int(q["query_id"].iloc[i])
        assert list(out) == list(ref.loc[qid, "ranked_venue_ids"]), f"{ds} {qid}: core != preds/hybrid"
        qq = q.iloc[[i]]
        cc = cands.loc[[qid]].reset_index()
        t0 = pc()
        model.predict("hybrid", qq, cc, topk=50, n_jobs=1)
        call = pc() - t0
        a_, b_, c_, e_ = sl["hist_lo"][i], sl["hist_hi"][i], sl["prefix_lo"][i], sl["prefix_hi"][i]
        ix = np.r_[np.arange(a_, b_), np.arange(c_, e_)]
        t0 = pc()
        D._retrieve_one(prev, A.venue[ix], np.exp2((A.t[ix] - t0g) / h), -1, st)
        ret = pc() - t0
        rows.append({"ds": ds, "query_id": qid, "n_visible_own": int(q["n_visible_own"].iloc[i]), "bin": LABELS[b[i]],
                     "hybrid_core_ms": 1000 * core, "hybrid_call_ms": 1000 * call, "retriever_ms": 1000 * ret})
    S = pd.DataFrame(rows)
    tok = q[["query_id", "n_visible_own"]].copy()
    tok["bin"] = [LABELS[x] for x in b]
    for m, pat in LLM.items():
        p = C.ROOT / pat.format(ds=ds)
        if not p.exists():
            continue
        rr = pd.read_parquet(p, columns=["query_id", "n_prompt_tokens", "n_output_tokens"])
        tok = tok.merge(rr.rename(columns={"n_prompt_tokens": f"{m}_prompt_tokens",
                                           "n_output_tokens": f"{m}_output_tokens"}), on="query_id", how="left")
        rec = C.mt.load_cost(m, "H-100", ds)
        gq = rec["perq"]["test"]["gpu_s"] if rec else np.nan
        pt = tok[f"{m}_prompt_tokens"].astype(float)
        tok[f"{m}_gpu_ms_est"] = 1000 * gq * pt / pt.mean()
    out = []
    for k, lab in enumerate(LABELS):
        s = S[S["bin"] == lab]
        t = tok[tok["bin"] == lab]
        if not len(t):
            continue
        row = {"ds": ds, "bin": lab, "bin_lo": EDGES[k], "n_queries": len(t), "share_queries": len(t) / len(tok),
               "mean_n_visible_own": float(t["n_visible_own"].mean()), "n_sampled": len(s)}
        for c in ("hybrid_core_ms", "hybrid_call_ms", "retriever_ms"):
            row[c + "_p50"] = float(s[c].median()) if len(s) else np.nan
            row[c + "_mean"] = float(s[c].mean()) if len(s) else np.nan
            row[c + "_p95"] = float(np.percentile(s[c], 95)) if len(s) else np.nan
        for m in LLM:
            for c in ("prompt_tokens", "output_tokens", "gpu_ms_est"):
                col = f"{m}_{c}"
                if col in t:
                    row[col] = float(t[col].mean())
        out.append(row)
    print(ds, " ".join(f"{r['bin']}:{r['n_queries']}/{r['hybrid_core_ms_p50']:.3f}ms/"
                       f"{r.get('ds-llm4poi-lcs_prompt_tokens', float('nan')):.0f}tok" for r in out), flush=True)
    return out, rows


def main():
    bins, qs = [], []
    for ds in C.MAIN:
        a, b = one(ds)
        bins += a
        qs += b
    C.write_csv("e6_history_bins.csv", bins)
    df = C.write_csv("e6_history_queries.csv", qs)
    sl = []
    for c in ("hybrid_core_ms", "hybrid_call_ms", "retriever_ms"):
        g = {ds: list(zip(x["n_visible_own"], x[c])) for ds, x in df.groupby("ds")}
        bb, se, n, ng = C.loglog_fe(g)
        sl.append({"y": c, "x": "n_visible_own", "slope": bb, "se": se, "n": n, "n_cities": ng})
    B = pd.DataFrame(bins)
    for m in LLM:
        col = f"{m}_prompt_tokens"
        if col in B:
            g = {ds: list(zip(x["mean_n_visible_own"], x[col])) for ds, x in B.groupby("ds")}
            bb, se, n, ng = C.loglog_fe(g)
            sl.append({"y": col, "x": "n_visible_own (bin means)", "slope": bb, "se": se, "n": n, "n_cities": ng})
    C.write_csv("e6_history_slopes.csv", sl)


if __name__ == "__main__":
    main()
