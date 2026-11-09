#!/usr/bin/env python
import argparse
import datetime as dt
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from poibench.hashing import sha256_file, sha256_text, tree_sums


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze-line", metavar="VERSION", default=None)
    a = ap.parse_args(argv)
    sums = tree_sums(os.path.join(ROOT, "data"))
    with open(os.path.join(ROOT, "data", "SHA256SUMS"), "w") as f:
        f.write(sums)
    data_fp = sha256_text(sums)
    eval_fp = sha256_file(os.path.join(ROOT, "poibench", "evaluate.py"))
    print(f"data/SHA256SUMS: {sums.count(chr(10))} files; data={data_fp}; eval={eval_fp}")
    if a.freeze_line:
        head = subprocess.run(["git", "-c", f"safe.directory={ROOT}", "-C", ROOT, "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-c", f"safe.directory={ROOT}", "-C", ROOT, "status", "--porcelain"],
                               capture_output=True, text=True, check=True).stdout.strip()
        if dirty:
            print("WARNING: uncommitted changes; commit before freezing", file=sys.stderr)
        print(f"Status: FROZEN v{a.freeze_line.lstrip('v')} {dt.datetime.now(dt.timezone.utc):%Y-%m-%d} "
              f"data={data_fp} eval={eval_fp} git={head}")


if __name__ == "__main__":
    main()
