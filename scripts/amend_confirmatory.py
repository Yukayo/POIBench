#!/usr/bin/env python
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from poibench import ROOT
from poibench import evaluate as ev

CHUNK = 2500


def amend(B, split):
    ev.assert_split_allowed(ROOT, split)
    conf = pd.read_csv(os.path.join(ROOT, "results", f"confirmatory_{split}.csv"))
    out = conf.copy()
    for c in ("B_amend", "p_amend", "p_holm_amend", "reject_amend"):
        out[c] = np.nan if c != "reject_amend" else None
    gate = {"max_abs_p_diff": 0.0, "max_abs_point_diff": 0.0, "n": 0}
    for ds, g in conf[conf.status == "run"].groupby("ds"):
        t0 = time.time()
        queries, cands, venues, checkins = ev.load_split(ROOT, ds, split, "H-100")
        es_e = ev.build_eval_set(queries, None, venues, cand_k=20, checkins=checkins)
        es_c = ev.build_eval_set(queries, cands, venues, cand_k=20, checkins=checkins) if (g["mode"] == "cand").any() else None
        if es_c is not None:
            assert es_c.n_users == es_e.n_users
        cache = {}

        def vals(name, mode):
            if (name, mode) not in cache:
                es = es_c if mode == "cand" else es_e
                ranks, _ = ev.per_query_ranks(es, ev.load_preds(ROOT, name, ds, split))
                cache[(name, mode)] = ev.metrics_from_ranks(ranks[mode])
            return cache[(name, mode)]

        m = es_e.strata["all"]
        nu = es_e.n_users
        C = np.bincount(es_e.user_codes[m], minlength=nu).astype(float)
        cols, point = [], []
        for _, t in g.iterrows():
            j = ev.METRICS.index(t.metric)
            d = vals(t["a"], t["mode"])[:, j] - vals(t["b"], t["mode"])[:, j]
            cols.append(np.bincount(es_e.user_codes[m], weights=d[m], minlength=nu))
            point.append(d[m].mean())
        S = np.stack(cols, 1)
        rng = np.random.default_rng(0)
        le = np.zeros(len(g))
        ge = np.zeros(len(g))
        first = None
        done = 0
        while done < B:
            k = min(CHUNK, B - done)
            W = np.empty((k, nu))
            for b in range(k):
                W[b] = np.bincount(rng.integers(0, nu, size=nu), minlength=nu)
            D = (W @ S) / (W @ C)[:, None]
            if first is None:
                first = D[:1000].copy()
            le += (D <= 0).sum(0)
            ge += (D >= 0).sum(0)
            done += k
        p1000 = np.minimum(1.0, 2 * np.minimum((first <= 0).mean(0), (first >= 0).mean(0)))
        dp = np.abs(p1000 - g.p_boot.to_numpy(float)).max()
        dpt = np.abs(np.array(point) - g["diff"].to_numpy(float)).max()
        gate["max_abs_p_diff"] = max(gate["max_abs_p_diff"], float(dp))
        gate["max_abs_point_diff"] = max(gate["max_abs_point_diff"], float(dpt))
        gate["n"] += len(g)
        if dp > 1e-12 or dpt > 1e-9:
            raise SystemExit(f"{ds}: the first 1000 resamples do not reproduce the frozen results "
                             f"(max |dp| = {dp}, max |d point| = {dpt}); amendment refused")
        pa = np.minimum(1.0, 2 * (np.minimum(le, ge) + 1) / (B + 1))
        out.loc[g.index, "B_amend"] = B
        out.loc[g.index, "p_amend"] = pa
        print(f"{ds}: {len(g)} tests, {nu} users, {time.time() - t0:.0f} s", flush=True)
    for fam, idx in out[out.status == "run"].groupby("family").groups.items():
        adj = ev.holm(list(out.loc[idx, "p_amend"].astype(float)))
        out.loc[idx, "p_holm_amend"] = adj
        out.loc[idx, "reject_amend"] = [bool(a < ev.CONFIRMATORY_ALPHA) for a in adj]
    path = os.path.join(ROOT, "results", f"confirmatory_{split}_B{B}.csv")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    out.to_csv(path, index=False)
    return conf, out, gate, path


def report(conf, out, gate, B, path):
    run = out[out.status == "run"]
    lines = ["# Confirmatory p-values, post-freeze amendment (analysis only)", "",
             "the pre-registration log 2026-11-03. `evaluate.py` computes p = 2 min(mean(d<=0), mean(d>=0)) over B = 1000 resamples with no",
             "add-one correction, so a test with no resample on the other side of zero prints p = 0 and Holm is applied to",
             "values that cannot resolve its thresholds. `scripts/amend_confirmatory.py` re-derives every pre-registered pair",
             f"with the frozen evaluator's functions, draws B = {B:,} resamples from the same generator, and uses",
             "p = 2(k+1)/(B+1), then Holm within the four families. Point estimates and 95% CIs stay the frozen B = 1000 ones.", "",
             f"Gate: the first 1000 resamples reproduce the frozen p_boot (max |difference| {gate['max_abs_p_diff']:.1e}) and",
             f"the frozen point estimates (max |difference| {gate['max_abs_point_diff']:.1e}) for all {gate['n']} tests.", "",
             "| family | tests run | reject, frozen B=1000 (raw Holm) | reject, amended | p = 0 at B=1000 | max amended p among rejections | amended p of the smallest non-rejection |",
             "|---|---|---|---|---|---|---|"]
    for fam, g in run.groupby("family"):
        rej_f = int(g.reject_h0.astype(str).eq("True").sum())
        rej_a = int(g.reject_amend.astype(str).eq("True").sum())
        z = int((g.p_boot.astype(float) == 0).sum())
        rr = g[g.reject_amend.astype(str).eq("True")]
        nr = g[~g.reject_amend.astype(str).eq("True")]
        mx = f"{rr.p_amend.astype(float).max():.1e}" if len(rr) else "-"
        mn = f"{nr.p_amend.astype(float).min():.4f}" if len(nr) else "-"
        lines.append(f"| {fam} | {len(g)} | {rej_f} | {rej_a} | {z} | {mx} | {mn} |")
    chg = run[run.reject_h0.astype(str) != run.reject_amend.astype(str)]
    lines += ["", f"Tests whose decision differs between the frozen and the amended p-values: {len(chg)}."]
    if len(chg):
        lines += ["", chg[["family", "ds", "a", "b", "p_boot", "p_amend", "p_holm_amend"]].to_string(index=False)]
    lines += ["", f"Full table: `{os.path.relpath(path, ROOT)}`.", ""]
    os.makedirs(os.path.join(ROOT, "reports"), exist_ok=True)
    with open(os.path.join(ROOT, "reports", "CONFIRMATORY_AMENDMENT.md"), "w") as f:
        f.write("\n".join(lines))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=100000)
    ap.add_argument("--split", default="test", choices=["test"])
    a = ap.parse_args(argv)
    conf, out, gate, path = amend(a.B, a.split)
    report(conf, out, gate, a.B, path)
    print(open(os.path.join(ROOT, "reports", "CONFIRMATORY_AMENDMENT.md")).read())


if __name__ == "__main__":
    main()
