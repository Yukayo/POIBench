#!/usr/bin/env python
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from poibench import ROOT
from poibench.hashing import sha256_file


def export_target_time(ds: str, root: str = ROOT) -> dict:
    src_dir = os.path.join(root, "data", ds)
    out_dir = os.path.join(root, "exports_ablation", ds)
    os.makedirs(out_dir, exist_ok=True)
    man = dict(purpose="labelled +target-time ablation only (PROTOCOL.md section 3)", ds=ds, files={}, source={})
    for split in ("val", "test"):
        src = os.path.join(src_dir, f"queries_{split}.parquet")
        q = pd.read_parquet(src, columns=["query_id", "target_t_local"]).sort_values("query_id")
        p = os.path.join(out_dir, f"target_time_{split}.parquet")
        q.to_parquet(p + ".tmp", index=False)
        os.replace(p + ".tmp", p)
        man["source"][f"queries_{split}.parquet"] = sha256_file(src)
        man["files"][os.path.basename(p)] = sha256_file(p)
    with open(os.path.join(out_dir, "MANIFEST.json.tmp"), "w") as f:
        json.dump(man, f, indent=1)
    os.replace(os.path.join(out_dir, "MANIFEST.json.tmp"), os.path.join(out_dir, "MANIFEST.json"))
    return man


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", nargs="+", required=True)
    a = ap.parse_args()
    for d in a.ds:
        print(json.dumps(export_target_time(d), indent=1))
