import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MDIR = os.path.join(ROOT, "methods", "graph_flashback")
sys.path.insert(0, ROOT)
sys.path.insert(0, MDIR)

import numpy as np

import pipeline
import run

_orig = pipeline._write_triplet_input


def _leaky_write_triplet_input(ex, idx, path):
    n = _orig(ex, idx, path)
    ctx = ex.context("val").drop_duplicates("row_id").sort_values("row_id")
    users = ex.queries("val").drop_duplicates("query_id").set_index("query_id")["user_id"]
    ctx = ctx.assign(user_id=users.reindex(ctx["query_id"]).to_numpy())
    with open(path, "a") as f:
        for uid, g in ctx.groupby("user_id", sort=True):
            p = idx.pidx_of[g["venue_id"].to_numpy(np.int64)]
            p = p[p >= 0]
            if len(p):
                f.write(" ".join([str(idx.uidx(int(uid)))] + [str(int(x)) for x in p]) + " \n")
    return n


pipeline._write_triplet_input = _leaky_write_triplet_input
pipeline.ADAPTER_VERSION = "gf1-LEAKY-CONTROL"

if __name__ == "__main__":
    sys.exit(run.main())
