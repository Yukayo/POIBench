#!/usr/bin/env python
import hashlib
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT, "envs", "tzlib"))
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
import timezonefinder

from poibench import data as D


def main():
    d = pd.read_csv(D.raw_path(D.GOWALLA_SNAP), sep="\t", header=None, usecols=[2, 3, 4],
                    names=["lat", "lon", "loc"], dtype={"lat": np.float64, "lon": np.float64, "loc": np.int64})
    v = d.groupby("loc")[["lat", "lon"]].median().reset_index()
    tf = timezonefinder.TimezoneFinder()
    v["valid_coords"] = v["lat"].between(-90, 90) & v["lon"].between(-180, 180)
    tz = [tf.timezone_at(lng=float(lo), lat=float(la)) if ok else "Etc/UTC"
          for la, lo, ok in zip(v["lat"], v["lon"], v["valid_coords"])]
    v["tz"] = pd.Series(tz, dtype=object)
    print(f"spots with out-of-range coordinates (tz Etc/UTC, flagged valid_coords=False): {int((~v['valid_coords']).sum())}")
    miss = v["tz"].isna()
    print(f"spots {len(v)}, zones {v['tz'].nunique()}, unresolved {int(miss.sum())}, "
          f"timezonefinder {timezonefinder.__version__ if hasattr(timezonefinder, '__version__') else '6.5.2'}")
    assert not miss.any()
    out = os.path.dirname(D.GOWALLA_TZ)
    os.makedirs(out, exist_ok=True)
    v["tz"] = v["tz"].astype(str)
    v.to_parquet(D.GOWALLA_TZ, index=False)
    h = hashlib.sha256(open(D.GOWALLA_TZ, "rb").read()).hexdigest()
    with open(os.path.join(out, "SHA256SUMS"), "w") as f:
        f.write(f"{h}  {os.path.basename(D.GOWALLA_TZ)}\n")
    print(v["tz"].value_counts().head(10).to_string())


if __name__ == "__main__":
    main()
