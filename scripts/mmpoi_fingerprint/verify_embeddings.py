#!/usr/bin/env python
import argparse
import json
import os
import sys
import zipfile

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


RELEASED = {"nyc": ("NYC_processed.zip", "NYC")}


def load(z, n, prefix="NYC"):
    d = {}
    for ln in z.read(f"{prefix}/{n}").split(b"\n"):
        if ln.strip():
            d.update(json.loads(ln))
    return d


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default="nyc")
    ds = ap.parse_args(argv).ds
    if ds not in RELEASED:
        print(f"VERIFY {ds}: no released vectors (the MMPOI authors released processed embeddings for NYC only)")
        return 2
    zname, prefix = RELEASED[ds]
    z = zipfile.ZipFile(os.path.join(ROOT, "data_raw", "mmpoi", "processed_dl", zname))
    ours = np.load(os.path.join(ROOT, "work", "mmpoi", "embed", ds, "embeddings.npz"))
    pos = {int(v): i for i, v in enumerate(ours["venue_id"])}
    idm = pd.read_parquet(os.path.join(ROOT, "data", ds, "id_map_venues.parquet"))
    raw2vid = dict(zip(idm["venue_raw"].str.lower(), idm["venue_id"].astype(int)))
    out = {}
    for mod, fn in (("image", "modal_image_embedding.json"), ("meta", "modal_meta_embedding.json"),
                    ("review", "modal_review_embedding.json")):
        theirs = load(z, fn, prefix)
        cos, n_both_zero, n_zero_mismatch = [], 0, 0
        for raw, v in theirs.items():
            t = np.asarray(v, dtype=np.float64).reshape(-1)
            i = pos.get(raw2vid.get(raw.lower(), -1))
            if i is None:
                continue
            o = ours[mod][i].astype(np.float64)
            tz, oz = len(t) != 768 or np.linalg.norm(t) == 0, np.linalg.norm(o) == 0
            if tz and oz:
                n_both_zero += 1
            elif tz != oz:
                n_zero_mismatch += 1
            else:
                cos.append(float(o @ t / np.linalg.norm(o) / np.linalg.norm(t)))
        c = np.asarray(cos)
        out[mod] = dict(n_compared=int(len(c)), n_both_missing=n_both_zero, n_missing_in_one_only=n_zero_mismatch,
                        cos_min=float(c.min()), cos_p01=float(np.percentile(c, 1)), cos_p05=float(np.percentile(c, 5)),
                        cos_median=float(np.median(c)), cos_mean=float(c.mean()),
                        share_ge_0999=float((c >= 0.999).mean()), share_ge_099=float((c >= 0.99).mean()))
    p = os.path.join(ROOT, "logs", "mmpoi_fingerprint",
                     "verify_embeddings.json" if ds == "nyc" else f"verify_embeddings_{ds}.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))
    ok = (out["image"]["cos_median"] >= 0.99 and out["image"]["share_ge_099"] >= 0.90
          and out["review"]["cos_median"] >= 0.999 and out["meta"]["cos_median"] >= 0.999)
    print("VERIFY", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
