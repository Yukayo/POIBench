#!/usr/bin/env python
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from poibench import ROOT

DATASETS = ("nyc", "tky", "ca", "sin", "istanbul", "sao_paulo")
PAIRS = (
    ("llm4poi@H-100", "hybrid@H-100", "llm4poi+hybridtail@H-100"),
    ("ds-llm4poi-lcs@H-100", "hybrid@H-100", "ds-llm4poi-lcs+hybridtail@H-100"),
    ("llm4poi-tt@H-100", "hybrid-tt@H-100", "llm4poi-tt+hybridtttail@H-100"),
    ("ds-llm4poi-lcs-tt@H-100", "hybrid-tt@H-100", "ds-llm4poi-lcs-tt+hybridtttail@H-100"),
)


def compose(ds, split, m, t, name):
    fa = os.path.join(ROOT, "preds", m, ds, f"{split}.parquet")
    ft = os.path.join(ROOT, "preds", t, ds, f"{split}.parquet")
    if not (os.path.exists(fa) and os.path.exists(ft)):
        return None
    A, T = pd.read_parquet(fa), pd.read_parquet(ft)
    j = A[["query_id", "ranked_venue_ids"]].merge(T, on="query_id", suffixes=("_a", ""))
    if not (len(j) == len(A) == len(T)):
        raise SystemExit(f"{ds}/{split}: {m} ({len(A)}) and {t} ({len(T)}) do not cover the same queries ({len(j)})")
    top, cand = [], []
    for ans, tail, ctail in zip(j["ranked_venue_ids_a"], j["ranked_venue_ids"], j["cand_ranked_venue_ids"]):
        a = int(list(ans)[0])
        top.append([a] + [int(v) for v in tail if int(v) != a][:49])
        c = [int(v) for v in ctail]
        cand.append(([a] + [v for v in c if v != a]) if a in set(c) else c)
    out = pd.DataFrame({"query_id": j["query_id"].to_numpy(), "ranked_venue_ids": top, "cand_ranked_venue_ids": cand,
                        "n_visible_own": j["n_visible_own"].to_numpy()})
    d = os.path.join(ROOT, "preds", name, ds)
    os.makedirs(d, exist_ok=True)
    out.to_parquet(os.path.join(d, f"{split}.parquet.tmp"), index=False)
    os.replace(os.path.join(d, f"{split}.parquet.tmp"), os.path.join(d, f"{split}.parquet"))
    cost = json.load(open(os.path.join(ROOT, "preds", t, ds, f"{split}.cost.json")))
    meta = {"method": name, "ds": ds, "split": split, "composed_from": {"answer": m, "tail": t},
            "candidates_file": cost.get("candidates_file"), "stages": [], "n_queries": int(len(out)),
            "note": "exploratory list completion: single answer followed by the tail method's list (the pre-registration log 2026-11-03)"}
    json.dump(meta, open(os.path.join(d, f"{split}.cost.json"), "w"), indent=1)
    return len(out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    ap.add_argument("--split", default="test", choices=["val", "test"])
    a = ap.parse_args(argv)
    for ds in a.datasets:
        for m, t, name in PAIRS:
            n = compose(ds, a.split, m, t, name)
            print(f"{ds:10s} {name:42s} " + (f"{n} queries" if n else "skipped (inputs missing)"))


if __name__ == "__main__":
    main()
