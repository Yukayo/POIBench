#!/usr/bin/env python
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from poibench import ROOT
from poibench import data as D
from poibench.cost import load_cost, summarize_cost
from poibench.evaluate import evaluate_files

SPLIT = "val"
DATASETS = ("nyc", "tky", "istanbul")
POLICIES = ("H-100", "H-10", "S-10")
METHODS = ("pop", "repeat", "markov", "hybrid")
METRICS = ("acc@1", "acc@5", "acc@10", "ndcg@5", "ndcg@10", "mrr", "map@20")


def run_preds(n_jobs):
    py = sys.executable
    for ds in DATASETS:
        for pol in POLICIES:
            cmd = [py, "-I", os.path.join(ROOT, "scripts", "run_simple.py"), ds, SPLIT, pol, "--n-jobs", str(n_jobs)]
            print("+", " ".join(cmd), flush=True)
            subprocess.run(cmd, check=True)


def preds_identical(ds, a, b):
    out = {}
    for m in METHODS:
        pa = pd.read_parquet(os.path.join(ROOT, "preds", f"{m}@{a}", ds, f"{SPLIT}.parquet"))
        pb = pd.read_parquet(os.path.join(ROOT, "preds", f"{m}@{b}", ds, f"{SPLIT}.parquet"))
        same = len(pa) == len(pb) and (pa.query_id.to_numpy() == pb.query_id.to_numpy()).all()
        for c in ("ranked_venue_ids", "cand_ranked_venue_ids"):
            same = same and all(list(x) == list(y) for x, y in zip(pa[c], pb[c]))
        out[m] = bool(same)
    return out


def in_history_under(ds, pol, ck_venue, q):
    ev = D.query_evidence(ds, SPLIT, pol).set_index("query_id").loc[q.query_id]
    tv = q.target_venue_id.to_numpy()
    hit = np.zeros(len(q), bool)
    for i, (a, b, c, d) in enumerate(zip(ev.hist_lo, ev.hist_hi, ev.prefix_lo, ev.prefix_hi)):
        hit[i] = (ck_venue[a:b] == tv[i]).any() or (ck_venue[c:d] == tv[i]).any()
    hl = (ev.hist_hi - ev.hist_lo + ev.prefix_hi - ev.prefix_lo).to_numpy()
    return hit, hl


def cand_recall(q, cands, mask=None):
    m = q[["query_id", "target_venue_id"]].merge(cands, on="query_id", how="left")
    tv = m.target_venue_id.to_numpy()
    r20 = np.array([t in list(c[:20]) for t, c in zip(tv, m.cand_venue_ids)])
    r50 = np.array([t in list(c[:50]) for t, c in zip(tv, m.cand_venue_ids)])
    if mask is None:
        mask = np.ones(len(q), bool)
    return float(r20[mask].mean()) if mask.any() else float("nan"), float(r50[mask].mean()) if mask.any() else float("nan")


def collect(n_boot, tag=""):
    summ = {"created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "split": SPLIT, "ds": {}}
    for ds in DATASETS:
        t0 = time.time()
        S = summ["ds"][ds] = {}
        q = D.load_queries(ds, SPLIT)
        ck = D.load_checkins(ds)
        ckv = ck.venue_id.to_numpy()
        S["n_queries"] = int(len(q))
        S["n_users"] = int(q.user_id.nunique())
        S["identical_S10_vs_H100"] = preds_identical(ds, "S-10", "H-100")
        S["policies"] = {}
        for pol in POLICIES:
            P = S["policies"][pol] = {}
            res = evaluate_files(ROOT, ds, SPLIT, [f"{m}@{pol}" for m in METHODS], policy=pol, B=n_boot,
                                 out_prefix=os.path.join(ROOT, "results", ds, SPLIT, f"simple__{pol}"))
            P["results"] = res["results"]
            P["pairs"] = [r for r in res["pairs"] if r["stratum"] == "all" and r["metric"] in ("acc@1", "ndcg@10")]
            P["diagnostics"] = res["diagnostics"]
            P["strata_sizes"] = res["strata_sizes"]
            P["candidates_file"] = res["config"]["candidates_file"]
            cands = pd.read_parquet(os.path.join(ROOT, P["candidates_file"]))
            ih100 = q.in_user_history.to_numpy(bool)
            ihp, hlp = in_history_under(ds, pol, ckv, q)
            P["in_history_share_policy"] = float(ihp.mean())
            P["in_history_share_H100_field"] = float(ih100.mean())
            P["mean_visible_own"] = float(hlp.mean())
            P["recall"] = {"all": cand_recall(q, cands), "repeat_policy": cand_recall(q, cands, ihp),
                           "explore_policy": cand_recall(q, cands, ~ihp)}
            P["cost"] = {}
            for m in METHODS:
                c = load_cost(os.path.join(ROOT, "preds", f"{m}@{pol}", ds, f"{SPLIT}.cost.json"))
                P["cost"][m] = summarize_cost(c)
        names = [f"{m}@{p}" for m in METHODS for p in ("H-100", "H-10")]
        res = evaluate_files(ROOT, ds, SPLIT, names, policy=None, e2e_only=True, B=n_boot,
                             out_prefix=os.path.join(ROOT, "results", ds, SPLIT, "simple__H-100_vs_H-10__e2e"))
        S["cross_policy_e2e"] = [r for r in res["pairs"] if r["stratum"] == "all" and r["metric"] in ("acc@1", "ndcg@10")
                                 and r["a"].split("@")[0] == r["b"].split("@")[0]]
        S["eval_wall_s"] = time.time() - t0
        print(f"[report] {ds} done in {S['eval_wall_s']:.1f}s", flush=True)
    with open(os.path.join(ROOT, "results", f"simple_val_summary{tag}.json"), "w") as f:
        json.dump(summ, f, indent=1, default=float)
    return summ


def _cell(r, ci=True):
    return f"{r['value']:.4f} [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]" if ci else f"{r['value']:.4f}"


def _lookup(results):
    return {(r["method"], r["mode"], r["stratum"], r["metric"]): r for r in results}


def write_md(summ, path):
    L = []
    w = L.append
    w("# Simple baselines on the validation split (integration run)")
    w("")
    w(f"Generated {summ['created_utc']} by `scripts/report_simple_val.py`. **Validation split only; no test-split "
      "prediction or metric was computed.** PROTOCOL.md is still DRAFT v0, so these numbers are sanity checks, "
      "not paper results. Seed 0, nothing tuned (fixed choices in the pre-registration log, 2026-10-27 12:20 and 12:55 UTC).")
    w("")
    w("Methods (`poibench/baselines_simple.py`): **pop** (visible train popularity), **repeat** (user visit frequency, "
      "recency tie-break), **markov** (first-order transitions from the previous venue, popularity back-off), "
      "**hybrid** (0.5/0.5 user-frequency + Markov-row mixture). Evidence policies (PROTOCOL section 5): "
      "**H-100** = full train-period history; **H-10** = only the latest 10% of each user's train-period check-ins "
      "are visible to anything (own history, popularity, transitions, candidates); **S-10** = full history visible, "
      "only 9% of train queries usable as gradient samples.")
    w("")
    w("Modes: **e2e** = the method's own top-10 list (MRR is MRR@10); **cand** = within the first 20 candidates of "
      "the common retriever of that policy (H-10 has its own candidate file; S-10 and H-100 share one). A target "
      "the retriever missed is a miss in cand mode, so cand Acc@k <= candidate R@20. "
      "Cells: value [95% user-level paired-bootstrap CI, B=1000, seed 0].")
    w("")
    w("## 1. Query sets, candidate recall and in-history share")
    w("")
    w("| dataset | policy | val queries | users | mean visible own check-ins | target in visible own history | "
      "cand R@20 | cand R@50 | R@20 repeat | R@20 explore |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for ds, S in summ["ds"].items():
        for pol, P in S["policies"].items():
            rc = P["recall"]
            w(f"| {ds} | {pol} | {S['n_queries']:,} | {S['n_users']:,} | {P['mean_visible_own']:.1f} | "
              f"{P['in_history_share_policy']:.3f} | {rc['all'][0]:.3f} | {rc['all'][1]:.3f} | "
              f"{rc['repeat_policy'][0]:.3f} | {rc['explore_policy'][0]:.3f} |")
    w("")
    w("\"Repeat\"/\"explore\" in this table use the visible history **of that policy**. The evaluator's "
      "`hist=` strata always use the H-100 field `queries.in_user_history` (PROTOCOL section 3), so under H-10 a "
      "query can be in `hist=repeat` while its target is invisible to the method.")
    w("")
    w("## 2. Results by dataset and policy")
    w("")
    for ds, S in summ["ds"].items():
        w(f"### {ds} ({S['n_queries']:,} val queries)")
        w("")
        for pol in POLICIES:
            P = S["policies"][pol]
            lk = _lookup(P["results"])
            w(f"**{pol}** (candidates: `{P['candidates_file']}`, R@20 = {P['recall']['all'][0]:.3f})")
            w("")
            w("| method | mode | Acc@1 | Acc@5 | Acc@10 | NDCG@10 | MRR |")
            w("|---|---|---|---|---|---|---|")
            for mode in ("e2e", "cand"):
                for m in METHODS:
                    k = f"{m}@{pol}"
                    cells = [_cell(lk[(k, mode, 'all', 'acc@1')])] + \
                            [_cell(lk[(k, mode, 'all', x)], ci=x == "ndcg@10") for x in METRICS[1:]]
                    w(f"| {m} | {mode} | " + " | ".join(cells) + " |")
            w("")
        if all(S["identical_S10_vs_H100"].values()):
            w("S-10 predictions are **identical** (both list columns, every query) to H-100 for all four methods, as "
              "expected: S-x only limits gradient samples and these baselines take no gradient steps. The S-10 rows "
              "therefore repeat H-100 and are kept only to show that the pipeline runs end to end under S-x.")
        else:
            w(f"**WARNING:** S-10 vs H-100 predictions differ: {S['identical_S10_vs_H100']}")
        w("")
    w("## 3. Hybrid by stratum (H-100, Acc@1)")
    w("")
    strata = ["all", "hist=repeat", "hist=explore", "pop=tail", "pop=mid", "pop=head", "hlen=short", "hlen=long",
              "pos=last", "pos=inner", "cand=hit", "cand=miss"]
    w("| dataset | mode | " + " | ".join(strata) + " |")
    w("|---|---|" + "---|" * len(strata))
    for ds, S in summ["ds"].items():
        P = S["policies"]["H-100"]
        lk = _lookup(P["results"])
        for mode in ("e2e", "cand"):
            cells = []
            for s in strata:
                r = lk.get(("hybrid@H-100", mode, s, "acc@1"))
                cells.append(f"{r['value']:.3f}" if r and r["n_queries"] else "-")
            w(f"| {ds} | {mode} | " + " | ".join(cells) + " |")
    w("")
    w("Stratum sizes (queries):")
    w("")
    w("| dataset | " + " | ".join(strata) + " |")
    w("|---|" + "---|" * len(strata))
    for ds, S in summ["ds"].items():
        sz = S["policies"]["H-100"]["strata_sizes"]
        w(f"| {ds} | " + " | ".join(f"{sz.get(s, 0):,}" for s in strata) + " |")
    w("")
    w("`pos=last` is LLM4POI's evaluation unit (last check-in of each trajectory); the original papers report only "
      "that stratum, so it is the closest comparison point to published numbers.")
    w("")
    w("## 4. History budget effect, H-10 minus H-100 (e2e only, paired)")
    w("")
    w("| dataset | method | dAcc@1 [95% CI] | dNDCG@10 [95% CI] |")
    w("|---|---|---|---|")
    for ds, S in summ["ds"].items():
        rows = {}
        for r in S["cross_policy_e2e"]:
            a, b = r["a"], r["b"]
            sign = 1.0
            if a.endswith("@H-100"):
                sign = -1.0
            m = a.split("@")[0]
            lo, hi = (r["ci_lo"], r["ci_hi"]) if sign > 0 else (-r["ci_hi"], -r["ci_lo"])
            rows.setdefault(m, {})[r["metric"]] = f"{sign * r['diff']:+.4f} [{lo:+.3f}, {hi:+.3f}]"
        for m in METHODS:
            w(f"| {ds} | {m} | {rows[m].get('acc@1', '-')} | {rows[m].get('ndcg@10', '-')} |")
    w("")
    w("Cand-mode differences across policies are not reported: H-10 and H-100 use different candidate sets, and "
      "the evaluator now refuses to score a method against a candidate file other than the one it ran on.")
    w("")
    w("## 5. Hybrid vs the next-best simple baseline (paired, H-100, Acc@1)")
    w("")
    w("| dataset | mode | comparison | diff [95% CI] | bootstrap p |")
    w("|---|---|---|---|---|")
    for ds, S in summ["ds"].items():
        P = S["policies"]["H-100"]
        for mode in ("e2e", "cand"):
            for r in P["pairs"]:
                if r["mode"] != mode or r["metric"] != "acc@1" or "hybrid@H-100" not in (r["a"], r["b"]):
                    continue
                if "repeat@H-100" not in (r["a"], r["b"]):
                    continue
                sign = 1.0 if r["a"] == "hybrid@H-100" else -1.0
                lo, hi = (r["ci_lo"], r["ci_hi"]) if sign > 0 else (-r["ci_hi"], -r["ci_lo"])
                w(f"| {ds} | {mode} | hybrid - repeat | {sign * r['diff']:+.4f} [{lo:+.3f}, {hi:+.3f}] | "
                  f"{r.get('p_boot', float('nan')):.3f} |")
    w("")
    w("p-values are not corrected for multiple comparisons (open issue).")
    w("")
    w("## 6. Cost (CPU only, no GPU)")
    w("")
    w("One-time = load data + fit counts (shared by the four baselines of one run, repeated in each method's "
      "cost.json); recurring = prediction of all val queries (16 worker processes, CPU pinned to 24 cores).")
    w("")
    w("| dataset | policy | method | one-time wall s | one-time CPU core-s | predict wall s | predict CPU core-s | "
      "CPU core-ms / query | peak mem GB | GPU s |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for ds, S in summ["ds"].items():
        for pol in ("H-100", "H-10"):
            for m in METHODS:
                c = S["policies"][pol]["cost"][m]
                w(f"| {ds} | {pol} | {m} | {c['one_time']['wall_s']:.2f} | {c['one_time']['cpu_core_s']:.2f} | "
                  f"{c['recurring']['wall_s']:.2f} | {c['recurring']['cpu_core_s']:.2f} | "
                  f"{1000 * c['recurring_per_query']['cpu_core_s']:.3f} | {c['peak_mem_gb']:.2f} | "
                  f"{c['one_time']['gpu_s'] + c['recurring']['gpu_s']:.0f} |")
    w("")
    w("S-10 cost rows equal H-100 up to timing noise and are omitted. The cost recorder's GPU path (NVML busy time, "
      "`peak_gpu_mem_gb`) is still untested; the first GPU job checks it (scripts/gpu_jobs/g0_cost_check.py).")
    w("")
    w("## 7. Diagnostics")
    w("")
    w("| dataset | policy | method | missing preds | lists < 10 | cand fill rate |")
    w("|---|---|---|---|---|---|")
    for ds, S in summ["ds"].items():
        for pol in POLICIES:
            for m in METHODS:
                d = S["policies"][pol]["diagnostics"][f"{m}@{pol}"]
                w(f"| {ds} | {pol} | {m} | {d['n_missing']} | {d['n_short_lists']} | {d.get('cand_fill_rate', float('nan')):.3f} |")
    w("")
    w("Result files: `results/<ds>/val/simple__<policy>.{json,csv,pairs.csv}`, "
      "`results/<ds>/val/simple__H-100_vs_H-10__e2e.*`, `results/simple_val_summary.json`.")
    w("")
    with open(path, "w") as f:
        f.write("\n".join(L))


def main():
    global DATASETS
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-run", action="store_true", help="reuse existing preds")
    ap.add_argument("--n-jobs", type=int, default=16)
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS),
                    help="datasets to run (2026-10-28: ca / sin / sao_paulo were added with --tag _newds)")
    ap.add_argument("--tag", default="", help="suffix of the summary json and report md (default: none)")
    a = ap.parse_args()
    DATASETS = tuple(a.datasets)
    if not a.skip_run:
        run_preds(min(a.n_jobs, 24))
    summ = collect(a.B, a.tag)
    os.makedirs(os.path.join(ROOT, "reports"), exist_ok=True)
    out = os.path.join(ROOT, "reports", f"REPORT_simple_val{a.tag}.md")
    write_md(summ, out)
    print("written:", out)


if __name__ == "__main__":
    main()
