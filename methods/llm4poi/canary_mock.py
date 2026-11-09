#!/usr/bin/env python
import argparse
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
os.environ["LLM4POI_BACKEND"] = "mock"

import pandas as pd

from poibench import canary as C
from poibench import exports as E
from methods.llm4poi import prompts as PR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--n", type=int, default=300)
    a = ap.parse_args()
    rep = C.run_canary("methods/llm4poi/run.py:predict", a.ds, a.policy, "val", a.n, 0, keep=True,
                       tag="llm4poi-mock")
    work = sorted(d for d in os.listdir(os.path.join(ROOT, "work", "canary"))
                  if d.startswith(f"llm4poi-mock__{a.ds}__{a.policy}__val__"))[-1]
    wd = os.path.join(ROOT, "work", "canary", work)
    pa = pd.read_parquet(os.path.join(wd, "run_real", "prompts.parquet"))
    pb = pd.read_parquet(os.path.join(wd, "run_perturbed", "prompts.parquet"))
    same_prompts = int((pa["prompt"].to_numpy() == pb["prompt"].to_numpy()).sum())
    sa = PR.train_samples(E.Export(E.export_dir(a.ds, a.policy, ROOT)))
    sb = PR.train_samples(E.Export(os.path.join(wd, "export_perturbed")))
    sft_same = (list(sa["prompt"]) == list(sb["prompt"]) and list(sa["answer"]) == list(sb["answer"]))
    out = dict(result=rep["result"], n_mismatch=rep["n_mismatch"], lint=rep["lint"],
               exporter_invariance=all(rep["exporter_invariance"].values()), perturbation=rep["perturbation"],
               prompts_identical=f"{same_prompts}/{len(pa)}", sft_samples=len(sa), sft_identical=sft_same,
               s_targets_in_other_contexts=rep["s_targets_in_other_contexts"])
    p = os.path.join(ROOT, "logs", "llm4poi", f"canary_mock_{a.ds}_{a.policy}.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))
    shutil.rmtree(wd, ignore_errors=True)
    ok = rep["result"] == "PASS" and same_prompts == len(pa) and sft_same
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
