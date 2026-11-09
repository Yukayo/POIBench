#!/usr/bin/env python
import argparse
import datetime as dt
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from poibench import data as D
from poibench.exports import EXTRA_POLICIES, POLICIES, export_inputs
from poibench.hashing import tree_sums


def _one(args):
    ds, pol, force = args
    t0 = dt.datetime.now()
    man = export_inputs(ds, pol, ROOT, force=force)
    s = (dt.datetime.now() - t0).total_seconds()
    return f"{ds}/{pol}: {'skipped (current)' if man.get('skipped') else 'written'} {man['counts']} ({s:.0f}s)"


def main(argv=None):
    ap = argparse.ArgumentParser()
    built = [d for d in D.ALL_DATASETS if os.path.exists(os.path.join(D.data_dir(d, ROOT), "meta.json"))]
    ap.add_argument("--datasets", nargs="+", default=built, choices=list(D.ALL_DATASETS))
    ap.add_argument("--policies", nargs="+", default=list(POLICIES), choices=list(POLICIES) + list(EXTRA_POLICIES))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    jobs = [(d, p, a.force) for d in a.datasets for p in a.policies]
    w = max(1, min(a.workers, 24, len(jobs)))
    if w == 1:
        msgs = map(_one, jobs)
    else:
        import multiprocessing as mp
        pool = mp.get_context("fork").Pool(w)
        msgs = pool.imap(_one, jobs)
    for m in msgs:
        print(f"[{dt.datetime.now(dt.timezone.utc):%H:%M:%S}] {m}", flush=True)
    sums = tree_sums(os.path.join(ROOT, "exports"), suffixes=(".parquet", ".json"))
    with open(os.path.join(ROOT, "exports", "SHA256SUMS"), "w") as f:
        f.write(sums)
    print(f"exports/SHA256SUMS: {sums.count(chr(10))} files")


if __name__ == "__main__":
    main()
