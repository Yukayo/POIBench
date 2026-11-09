#!/usr/bin/env python
import argparse
import hashlib
import json
import os
import sys
import zipfile

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCES = {"nyc": ("NYC_WWW2024.zip", "24591ecd81b9fcb53d76b1e918ebb4124a7d96b06fb5241b59d4bfda85e623e9", "NYC"),
           "tky": ("TKY_WWW2024.zip", "7de8dcc762a0201ad58f457e805e6d636af25c58c7ea7d8fd8820565d1ba14a4", "TKY")}


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def parse_review_summary(blob: bytes) -> tuple:
    parts = {}
    for ln in blob.split(b"\n"):
        if not ln.strip():
            continue
        d = json.loads(ln)
        for key, tips in d.items():
            k, vid = key.split("_", 1)
            if not isinstance(tips, list) or not all(isinstance(t, str) for t in tips):
                raise ValueError(f"unexpected value type for {key}")
            parts.setdefault(vid.lower(), []).append((int(k), tips))
    return {v: [t for _, ts in sorted(p, key=lambda x: x[0]) for t in ts] for v, p in parts.items()}, \
        {v: len(p) for v, p in parts.items()}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default="nyc", choices=sorted(SOURCES))
    ds = ap.parse_args(argv).ds
    zname, ARCHIVE_SHA256, prefix = SOURCES[ds]
    ARCHIVE = os.path.join(ROOT, "data_raw", "mmpoi", "archives", zname)
    OUT = os.path.join(ROOT, "exports_mm", ds)
    got = sha256_file(ARCHIVE)
    if got != ARCHIVE_SHA256:
        raise SystemExit(f"{ARCHIVE}: sha256 {got} != {ARCHIVE_SHA256}")
    with zipfile.ZipFile(ARCHIVE) as z:
        tips, nkeys = parse_review_summary(z.read(f"{prefix}/review_summary.json"))
    idm = pd.read_parquet(os.path.join(ROOT, "data", ds, "id_map_venues.parquet"))
    raw2vid = dict(zip(idm["venue_raw"].str.lower(), idm["venue_id"].astype(int)))
    recs = []
    for raw, vid in sorted(raw2vid.items(), key=lambda x: x[1]):
        ts = tips.get(raw)
        recs.append(dict(venue_id=int(vid), has_tips=bool(ts), n_tips=len(ts or []), tips=list(ts or []),
                         n_source_keys=int(nkeys.get(raw, 0))))
    out = pd.DataFrame(recs)
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, "venue_texts.parquet")
    out.to_parquet(p + ".tmp.parquet", index=False)
    os.replace(p + ".tmp.parquet", p)
    man = dict(source=f"MMPOI {zname} {prefix}/review_summary.json", archive_sha256=ARCHIVE_SHA256,
               builder="scripts/mmpoi_texts.py", n_venues=int(len(out)), n_with_tips=int(out.has_tips.sum()),
               n_source_venues=len(tips), n_source_venues_not_in_catalog=int(len(set(tips) - set(raw2vid))),
               n_venues_multi_key=int((out.n_source_keys > 1).sum()), venue_texts_sha256=sha256_file(p))
    with open(os.path.join(OUT, "MANIFEST_texts.json"), "w") as f:
        json.dump(man, f, indent=1)
    print(json.dumps(man))
    return 0


if __name__ == "__main__":
    sys.exit(main())
