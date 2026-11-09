#!/usr/bin/env python
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from poibench import ROOT
from poibench import evaluate as ev

DATASETS = ev.MAIN_DATASETS
DEFAULT = ("hybrid", "llm4poi", "llm4poi+hybridtail", "ds-llm4poi-lcs", "ds-llm4poi-lcs+hybridtail", "gnpr-sid",
           "getnext", "getnext-zfeat", "ga-llm")
TIMEGIVEN = ("hybrid-tt", "llm4poi-tt", "llm4poi-tt+hybridtttail", "ds-llm4poi-lcs-tt",
             "ds-llm4poi-lcs-tt+hybridtttail", "gnpr-sid-tt", "ga-llm-tt")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    ap.add_argument("--B", type=int, default=10000)
    a = ap.parse_args(argv)
    for ds in a.datasets:
        names = [f"{m}@H-100" for m in DEFAULT + TIMEGIVEN
                 if os.path.exists(os.path.join(ROOT, "preds", f"{m}@H-100", ds, "test.parquet"))]
        out = os.path.join(ROOT, "results", ds, "test", "exploratory_timeaware")
        res = ev.evaluate_files(ROOT, ds, "test", names, out_prefix=out, policy="H-100", e2e_only=True,
                                B=a.B, seed=0, pairs="all")
        json.dump({"ds": ds, "methods": names, "B": a.B, "n_queries": res["n_queries"]},
                  open(out + ".meta.json", "w"), indent=1)
        print(f"{ds}: {len(names)} methods, {res['n_queries']} queries -> {os.path.relpath(out, ROOT)}", flush=True)


if __name__ == "__main__":
    main()
