import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from poibench.canary import _builtin_prev_then_pop

if __name__ == "__main__":
    export_dir, split, qfile, out = sys.argv[1:5]
    ids = [int(x) for x in open(qfile).read().split()]
    _builtin_prev_then_pop(export_dir, split, ids, None).to_parquet(out, index=False)
