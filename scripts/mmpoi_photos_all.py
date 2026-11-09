#!/usr/bin/env python
import argparse
import hashlib
import json
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCES = {"nyc": ("data_raw/mmpoi/archives/NYC_WWW2024.zip",
                   "24591ecd81b9fcb53d76b1e918ebb4124a7d96b06fb5241b59d4bfda85e623e9", "NYC/image.tgz"),
           "tky": ("data_raw/mmpoi/archives/TKY_WWW2024.zip",
                   "7de8dcc762a0201ad58f457e805e6d636af25c58c7ea7d8fd8820565d1ba14a4", "TKY/image.tgz")}


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default="nyc", choices=sorted(SOURCES))
    ds = ap.parse_args(argv).ds
    ARCHIVE_REL, ARCHIVE_SHA256, TAR_MEMBER = SOURCES[ds]
    OUT = os.path.join(ROOT, "exports_mm", ds)
    inv = pd.read_parquet(os.path.join(ROOT, "data_raw", "mmpoi", "derived", f"{ds}_images.parquet"))
    n_all = len(inv)
    t = inv[inv["ok"] & (inv["venue_id"] >= 0)]
    t = t.sort_values(["venue_id", "j", "member"], kind="mergesort")
    out = pd.DataFrame({"venue_id": t["venue_id"].astype("int64").to_numpy(), "j": t["j"].astype("int64").to_numpy(),
                        "member": t["member"].to_numpy(), "sha256": t["sha256"].to_numpy(),
                        "nbytes": t["nbytes"].astype("int64").to_numpy()})
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, "venue_photos_all.parquet")
    out.to_parquet(p + ".tmp.parquet", index=False)
    os.replace(p + ".tmp.parquet", p)
    man = dict(source=f"MMPOI {ARCHIVE_REL} {TAR_MEMBER}", archive=ARCHIVE_REL, archive_sha256=ARCHIVE_SHA256,
               tar_member=TAR_MEMBER, builder="scripts/mmpoi_photos_all.py",
               rule="every decodable photo of the venue (no duplicate or size rule; MMPOI faithful)",
               n_photos_in_archive=int(n_all), n_photos=int(len(out)), n_venues=int(out["venue_id"].nunique()),
               n_undecodable_dropped=int((~inv["ok"]).sum()), venue_photos_all_sha256=sha256_file(p))
    with open(os.path.join(OUT, "MANIFEST_photos_all.json"), "w") as f:
        json.dump(man, f, indent=1)
    print(json.dumps(man))
    return 0


if __name__ == "__main__":
    sys.exit(main())
