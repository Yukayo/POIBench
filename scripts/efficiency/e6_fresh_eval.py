#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

import pandas as pd

MET = ("acc@1", "acc@5", "ndcg@10", "mrr")
FROZEN = ("hybrid", "llm4poi", "ds-llm4poi-lcs", "getnext-zfeat", "gnpr-sid")
FRESH_LLM = {"llm4poi-fresh": "llm4poi", "ds-llm4poi-lcs-fresh": "ds-llm4poi-lcs"}


def preds_of(ds):
    P = {}
    for m in FROZEN:
        p = C.ROOT / "preds" / f"{m}@H-100" / ds / "test.parquet"
        if p.exists():
            P[m] = pd.read_parquet(p)
    p = C.OUT / "fresh_hybrid" / ds / "test.parquet"
    if p.exists():
        P["hybrid-fresh"] = pd.read_parquet(p)
    for m in FRESH_LLM:
        p = C.OUT / "fresh_llm" / m / ds / "test.parquet"
        if p.exists():
            P[m] = pd.read_parquet(p)[["query_id", "ranked_venue_ids"]]
    return P


def cost_of(m, ds):
    if m in FRESH_LLM:
        p = C.OUT / "fresh_llm" / m / ds / "test.cost.json"
        if not p.exists():
            return {}
        j = json.load(open(p))
        inf = [s for s in j["stages"] if s["name"] == "infer"][0]
        n = j["prompt_stats"]["n"]
        return dict(gpu_ms_per_generated=1000 * inf["gpu_s"] / max(1, inf.get("n_generated", n)),
                    n_generated=inf.get("n_generated"), n_cache_hits=inf.get("n_cache_hits"),
                    prompt_tokens_mean=j["tokens"]["prompt_mean"], repro_identical=j["repro_check"]["identical_text_share"],
                    share_prompt_identical_to_frozen=j["prompt_stats"]["share_prompt_identical_to_frozen"],
                    host=j["host"], mean_n_hist=j["prompt_stats"].get("mean_n_hist"),
                    mean_n_bank=j["prompt_stats"].get("mean_n_bank"), mean_n_demos=j["prompt_stats"].get("mean_n_demos"))
    if m in ("llm4poi", "ds-llm4poi-lcs"):
        j = C.cost_json(m, "H-100", ds)
        inf = [s for s in j["stages"] if s["name"] == "infer"]
        g = sum(s["gpu_s"] for s in inf)
        gen = j.get("generation") or {}
        n_gen = gen.get("n_generated") or j["n_queries"]
        if gen.get("n_prompts") and gen["n_prompts"] > j["n_queries"]:
            n_gen = n_gen * j["n_queries"] / gen["n_prompts"]
        ps = j.get("prompt_stats") or {}
        return dict(gpu_ms_per_generated=1000 * g / max(1, n_gen), prompt_tokens_mean=j["tokens"]["prompt_mean"],
                    mean_n_hist=ps.get("mean_n_hist"), mean_n_bank=ps.get("mean_n_bank"),
                    mean_n_demos=ps.get("mean_n_demos"))
    return {}


def one(ds):
    from poibench import evaluate as ev
    queries, cands, venues, checkins = ev.load_split(str(C.ROOT), ds, "test", "H-100")
    P = preds_of(ds)
    res = ev.evaluate(P, queries, candidates=None, venues=venues, B=1000, seed=0, pairs="all", strata=["all"],
                      split="test")
    R, PR = pd.DataFrame(res["results"]), pd.DataFrame(res["pairs"])
    rows = []
    for m in P:
        row = {"ds": ds, "method": m, "n_queries": res["n_queries"], "n_users": res["n_users"]}
        for met in MET:
            r = R[(R.method == m) & (R["mode"] == "e2e") & (R.metric == met)].iloc[0]
            row[met], row[f"{met}_ci_lo"], row[f"{met}_ci_hi"] = r.value, r.ci_lo, r.ci_hi
        row.update(cost_of(m, ds))
        rows.append(row)

    def pair(a, b):
        x = PR[(PR["mode"] == "e2e") & (PR.metric == "acc@1") & (((PR.a == a) & (PR.b == b)) | ((PR.a == b) & (PR.b == a)))]
        if not len(x):
            return None
        x = x.iloc[0]
        s = 1 if x.a == a else -1
        lo, hi = (x.ci_lo, x.ci_hi) if s == 1 else (-x.ci_hi, -x.ci_lo)
        return dict(ds=ds, a=a, b=b, diff=s * x["diff"], ci_lo=lo, ci_hi=hi, excl0=bool(lo > 0 or hi < 0))

    prs = []
    for f, base in FRESH_LLM.items():
        for b in (base, "hybrid-fresh", "hybrid"):
            p = pair(f, b)
            if p:
                prs.append(p)
    for b in ("llm4poi", "ds-llm4poi-lcs", "hybrid"):
        p = pair("hybrid-fresh", b)
        if p:
            prs.append(p)
    print(ds, " ".join(f"{r['method']}={r['acc@1']:.4f}" for r in rows), flush=True)
    for p in prs:
        print(f"   {p['a']} - {p['b']}: {p['diff']:+.4f} [{p['ci_lo']:.4f}, {p['ci_hi']:.4f}]", flush=True)
    return rows, prs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(C.MAIN))
    a = ap.parse_args()
    rows, prs = [], []
    for ds in a.datasets:
        r, p = one(ds)
        rows += r
        prs += p
    C.write_csv("e6_fresh_llm.csv", rows)
    C.write_csv("e6_fresh_pairs.csv", prs)


if __name__ == "__main__":
    main()
