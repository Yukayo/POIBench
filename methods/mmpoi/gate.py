#!/usr/bin/env python
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

from poibench.canary import run_canary

ENTRY = "methods/mmpoi/run.py:predict"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=["image", "noimage"])
    ap.add_argument("--policies", nargs="+", default=["H-100", "H-10"])
    ap.add_argument("--ds", default="nyc")
    ap.add_argument("--n", type=int, default=300)
    a = ap.parse_args(argv)
    ok = True
    for pol in a.policies:
        for var in a.variants:
            subprocess.run([sys.executable, "-I", os.path.join(HERE, "run.py"), "--ds", a.ds, "--policy", pol,
                            "--seed", "0", "--variant", var, "--canary-prep"], check=True, cwd=ROOT)
            os.environ["MMPOI_VARIANT"] = var
            rep = run_canary(ENTRY, a.ds, pol, "val", a.n, 0, tag=f"mmpoi-{var}")
            s = {k: rep[k] for k in ("result", "ds", "policy", "n_queries", "n_mismatch", "n_missing",
                                     "s_targets_in_other_contexts", "seconds")}
            s.update(variant=var, lint=rep["lint"], invariance=all(rep["exporter_invariance"].values()))
            print(json.dumps(s), flush=True)
            ok &= rep["result"] == "PASS"
    print(f"mmpoi canary gate: {'PASS' if ok else 'FAIL'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
