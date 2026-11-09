#!/usr/bin/env python
import argparse
import csv
import glob
import json
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from poibench import ROOT
from poibench import evaluate as ev

POLICIES = ("H-100", "H-10", "H-70", "S-10", "S-70")
DATASETS = ev.MAIN_DATASETS + ("gowalla", "foursquare_global", "nyc_trainfilter", "tky_trainfilter")


def available(ds, split):
    out = {}
    for d in sorted(glob.glob(os.path.join(ROOT, "preds", "*@*"))):
        name = os.path.basename(d)
        pol = name.rsplit("@", 1)[1]
        if pol in POLICIES and os.path.exists(os.path.join(d, ds, f"{split}.parquet")):
            out.setdefault(pol, []).append(name)
    return out


def run_group(ds, split, tag, methods, policy, e2e_only):
    out = os.path.join(ROOT, "results", ds, split, f"paper_{tag}")
    try:
        res = ev.evaluate_files(ROOT, ds, split, methods, out_prefix=out, policy=policy, e2e_only=e2e_only,
                                B=1000, seed=0, pairs="all")
        return dict(ds=ds, split=split, tag=tag, ok=True, n_methods=len(methods), n_queries=res["n_queries"],
                    diag={m: res["diagnostics"][m] for m in methods})
    except Exception as e:
        return dict(ds=ds, split=split, tag=tag, ok=False, error=f"{type(e).__name__}: {e}",
                    tb=traceback.format_exc(limit=3))


def tasks_for(ds, split):
    av = available(ds, split)
    T = []
    for pol, ms in av.items():
        T.append((ds, split, pol, ms, pol, False))
    base = {}
    for pol, ms in av.items():
        for m in ms:
            base.setdefault(m.rsplit("@", 1)[0], []).append(m)
    for b, ms in sorted(base.items()):
        if len(ms) >= 2:
            ms = sorted(ms, key=lambda x: POLICIES.index(x.rsplit("@", 1)[1]))
            T.append((ds, split, f"policies-{b}_vs_", ms, None, True))
    return T


def confirmatory(split):
    rows = []
    cache = {}
    for t in ev.confirmatory_tests():
        tag = t["group"] + ("_vs_" if t["group"].startswith("policies-") else "")
        f = os.path.join(ROOT, "results", t["ds"], split, f"paper_{tag}.pairs.csv")
        if f not in cache:
            cache[f] = {}
            if os.path.exists(f):
                with open(f) as fh:
                    for r in csv.DictReader(fh):
                        cache[f][(r["a"], r["b"], r["mode"], r["stratum"], r["metric"])] = r
        P = cache[f]
        r, sign = P.get((t["a"], t["b"], t["mode"], t["stratum"], t["metric"])), 1.0
        if r is None:
            r, sign = P.get((t["b"], t["a"], t["mode"], t["stratum"], t["metric"])), -1.0
        rec = dict(t)
        if r is None or r.get("p_boot") in (None, ""):
            rec.update(diff=None, ci_lo=None, ci_hi=None, p_boot=None, status="not run")
        else:
            lo, hi = float(r["ci_lo"]), float(r["ci_hi"])
            rec.update(diff=sign * float(r["diff"]), ci_lo=min(sign * lo, sign * hi), ci_hi=max(sign * lo, sign * hi),
                       p_boot=float(r["p_boot"]), status="run")
        rows.append(rec)
    for fam in sorted({r["family"] for r in rows}):
        idx = [i for i, r in enumerate(rows) if r["family"] == fam]
        adj = ev.holm([rows[i]["p_boot"] for i in idx])
        for i, a in zip(idx, adj):
            rows[i]["p_holm"] = a
            rows[i]["m_family"] = sum(1 for j in idx if rows[j]["p_boot"] is not None)
            rows[i]["reject_h0"] = None if a is None else bool(a < ev.CONFIRMATORY_ALPHA)
    out = os.path.join(ROOT, "results", f"confirmatory_{split}.csv")
    keys = ["family", "ds", "a", "b", "mode", "stratum", "metric", "diff", "ci_lo", "ci_hi", "p_boot", "p_holm",
            "m_family", "reject_h0", "status", "group"]
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in keys})
    n_run = sum(r["status"] == "run" for r in rows)
    print(f"confirmatory ({split}): {n_run}/{len(rows)} tests run -> {out}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["val", "test"])
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    ap.add_argument("--jobs", type=int, default=12)
    a = ap.parse_args(argv)
    ev.assert_split_allowed(ROOT, a.split)
    T = [t for ds in a.datasets for t in tasks_for(ds, a.split)]
    print(f"{len(T)} evaluation calls")
    log = []
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        futs = [ex.submit(run_group, *t) for t in T]
        for f in as_completed(futs):
            r = f.result()
            log.append(r)
            print(("OK  " if r["ok"] else "ERR ") + f"{r['ds']}/{r['split']}/{r['tag']}"
                  + (f" ({r['n_methods']} methods, {r['n_queries']} queries)" if r["ok"] else f": {r['error']}"),
                  flush=True)
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, "results", f"eval_all_{a.split}.log.json"), "w") as fh:
        json.dump(sorted(log, key=lambda r: (r["ds"], r["tag"])), fh, indent=1, default=str)
    confirmatory(a.split)
    return 0 if all(r["ok"] for r in log) else 1


if __name__ == "__main__":
    sys.exit(main())
