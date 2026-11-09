#!/usr/bin/env python
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

from poibench.canary import run_canary

ENTRY = "methods/qwen25_vl_ref/run.py:predict"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="vllm", choices=["vllm", "mock"])
    ap.add_argument("--arms", nargs="+", default=["A", "B", "C"])
    ap.add_argument("--policies", nargs="+", default=["H-100", "H-10"])
    ap.add_argument("--ds", default="nyc")
    ap.add_argument("--n", type=int, default=300)
    a = ap.parse_args(argv)
    os.environ["QWENVL_BACKEND"] = a.backend
    ok = True
    summary = []
    for pol in a.policies:
        for arm in a.arms:
            os.environ["QWENVL_ARM"] = arm
            tag = f"qwen25_vl_ref-{arm}" + ("-mock" if a.backend == "mock" else "")
            rep = run_canary(ENTRY, a.ds, pol, "val", a.n, 0, tag=tag)
            s = {k: rep[k] for k in ("result", "ds", "policy", "n_queries", "n_mismatch", "n_missing",
                                     "s_targets_in_other_contexts", "seconds")}
            s.update(arm=arm, backend=a.backend, lint=rep["lint"], invariance=all(rep["exporter_invariance"].values()))
            print(json.dumps(s), flush=True)
            summary.append(s)
            ok &= rep["result"] == "PASS"
    print(f"qwen2.5-vl-ref canary gate ({a.backend}): {'PASS' if ok else 'FAIL'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
