import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pandas as pd

from poibench.exports import Export


def predict(export_dir, split, query_ids, work_dir=None):
    c = Export(export_dir).candidates(split)
    cm = dict(zip(c["query_id"].astype(int), c["cand_venue_ids"]))
    out = pd.DataFrame({"query_id": [int(q) for q in query_ids],
                        "ranked_venue_ids": [[int(v) for v in cm[int(q)]][:10] for q in query_ids]})
    if not os.environ.get("CFG_ENTRY_SILENT"):
        out.attrs["adapter_config"] = {"cfg": os.environ.get("CFG_ENTRY_CFG", "default")}
    return out
