from __future__ import annotations
import os as _pb_os
def _pb_root():
    p = _pb_os.path.dirname(_pb_os.path.abspath(__file__))
    while not _pb_os.path.isfile(_pb_os.path.join(p, "poibench", "__init__.py")) and _pb_os.path.dirname(p) != p:
        p = _pb_os.path.dirname(p)
    return p
_PB_ROOT = _pb_os.environ.get("POIBENCH_ROOT") or _pb_root()
_PB_RAW = _pb_os.environ.get("POIBENCH_RAW_ROOT") or _pb_os.path.join(_PB_ROOT, "data_raw")

import hashlib
import json
import math
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import ROOT

RAW_ROOT = os.environ.get("POIBENCH_RAW_ROOT") or f"{_PB_RAW}"
DL_PREFIX = "data_raw/"
SPLITS = ("train", "val", "test")
DATASETS = ("nyc", "tky", "istanbul", "ca", "sin", "sao_paulo", "gowalla", "foursquare_global")
DATA_RAW = os.environ.get("POIBENCH_RAW_ROOT") or os.path.join(ROOT, "data_raw")


def raw_path(key: str, raw_root: str | None = None) -> str:
    if key.startswith(DL_PREFIX):
        return os.path.join(DATA_RAW if raw_root is None else raw_root, key[len(DL_PREFIX):])
    return os.path.join(RAW_ROOT if raw_root is None else raw_root, key)

MASSIVE_STEPS = {"istanbul": ("massive_steps_istanbul", "Europe/Istanbul"),
                 "sao_paulo": ("massive_steps_sao_paulo", "America/Sao_Paulo")}
GOWALLA_SNAP = "gowalla_snap/loc-gowalla_totalCheckins.txt.gz"
GOWALLA_SPOTS = "gowalla_liu/gowalla_spots_subset1.csv"
CA_POLYGONS_KEY = "data_raw/llm4poi_release/ca/us_state_polygon_json.json"
CA_POLYGONS = raw_path(CA_POLYGONS_KEY)
CA_STATES = ("california", "nevada")
CA_TZ = "America/Los_Angeles"
TIST_KEY = "data_raw/foursquare_tist2015"
TIST_DIR = raw_path(TIST_KEY)
TIST_COUNTRY = {"sin": "SG"}
FLASHBACK = {
    "gowalla": dict(min_poi=10, min_user=10, dedup=True, min_user_loader=101),
    "foursquare_global": dict(min_poi=50, min_user=100, dedup=False, min_user_loader=101),
}
WWW2019_KEY = "data_raw/foursquare_www2019"
WWW2019_DIR = raw_path(WWW2019_KEY)
GOWALLA_TZ_KEY = "data_raw/gowalla_global_derived/venue_tz.parquet"
GOWALLA_TZ = raw_path(GOWALLA_TZ_KEY)
GOWALLA_NO_CATEGORY = "Unknown"
MAIN = ("nyc", "tky", "ca", "sin", "istanbul", "sao_paulo")
APPENDIX = ("gowalla", "foursquare_global")
VARIANTS = {"nyc_trainfilter": "nyc", "tky_trainfilter": "tky"}
ALL_DATASETS = DATASETS + tuple(VARIANTS)
POLICY_X = (10, 70, 100)


def base_dataset(ds: str) -> str:
    return VARIANTS.get(ds, ds)

CFG = dict(
    min_poi_checkins=10,
    min_user_checkins=10,
    traj_gap_s=24 * 3600,
    train_end_frac=0.8,
    val_end_frac=0.9,
    istanbul_trail_prefix="2013",
    istanbul_tz="Europe/Istanbul",
)
RETRIEVER = dict(
    version="r1",
    list_len=50,
    out_len=50,
    main_len=20,
    half_life_days=14.0,
    geo_k=100,
    rrf_k=60,
)

RAW_FILES = {
    "nyc": ["tsmc2014/extracted/dataset_tsmc2014/dataset_TSMC2014_NYC.txt", "tsmc2014/dataset_tsmc2014.zip"],
    "tky": ["tsmc2014/extracted/dataset_tsmc2014/dataset_TSMC2014_TKY.txt", "tsmc2014/dataset_tsmc2014.zip"],
    "istanbul": [f"massive_steps_istanbul/{s}.parquet" for s in ("train", "validation", "test")],
    "sao_paulo": [f"massive_steps_sao_paulo/{s}.parquet" for s in ("train", "validation", "test")],
    "ca": [GOWALLA_SNAP, GOWALLA_SPOTS, CA_POLYGONS_KEY],
    "sin": [f"{TIST_KEY}/extracted/dataset_TIST2015_Checkins.txt", f"{TIST_KEY}/extracted/dataset_TIST2015_POIs.txt",
            f"{TIST_KEY}/dataset_TIST2015.zip"],
    "gowalla": [GOWALLA_SNAP, GOWALLA_SPOTS, GOWALLA_TZ_KEY],
    "foursquare_global": [f"{WWW2019_KEY}/extracted/raw_Checkins_anonymized.txt", f"{WWW2019_KEY}/extracted/raw_POIs.txt",
                          f"{WWW2019_KEY}/dataset_WWW2019.zip"],
}


def data_dir(ds: str, root: str = ROOT) -> str:
    return os.path.join(root, "data", ds)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def raw_checksums(ds: str, raw_root: str | None = None) -> dict:
    out = {}
    for key in RAW_FILES[base_dataset(ds)]:
        p = raw_path(key, raw_root)
        out[key] = {"sha256": sha256_file(p), "bytes": os.path.getsize(p)}
    return out


def load_raw(ds: str, raw_root: str | None = None) -> tuple[pd.DataFrame, dict]:
    stats = {}
    ds = base_dataset(ds)
    if ds in ("nyc", "tky"):
        path = raw_path(RAW_FILES[ds][0], raw_root)
        df = pd.read_csv(path, sep="\t", header=None, encoding="latin-1",
                         names=["user_raw", "venue_raw", "cat_id", "category", "lat", "lon", "tz_min", "utc"],
                         dtype={"user_raw": str, "venue_raw": str, "cat_id": str, "category": str})
        df["t_utc"] = pd.to_datetime(df["utc"], format="%a %b %d %H:%M:%S +0000 %Y")
        df["t_local"] = df["t_utc"] + pd.to_timedelta(df["tz_min"], unit="m")
        stats["raw_checkins"] = len(df)
    elif ds == "ca":
        df, stats = _load_gowalla_ca(raw_root)
    elif ds in TIST_COUNTRY:
        df, stats = _load_tist(ds, raw_root)
    elif ds == "gowalla":
        df, stats = _load_gowalla_global(raw_root)
    elif ds == "foursquare_global":
        df, stats = _load_foursquare_global(raw_root)
    elif ds in MASSIVE_STEPS:
        parts = [pd.read_parquet(raw_path(rel, raw_root)) for rel in RAW_FILES[ds]]
        df = pd.concat(parts, ignore_index=True)
        stats["raw_checkins_all_eras"] = len(df)
        df = df[df["trail_id"].astype(str).str.startswith(CFG["istanbul_trail_prefix"])].copy()
        stats["raw_checkins"] = len(df)
        df = df.rename(columns={"latitude": "lat", "longitude": "lon", "venue_category": "category"})
        df["user_raw"] = df["user_id"].astype(str)
        df["venue_raw"] = df["venue_id"].astype(str)
        df["t_utc"] = pd.to_datetime(df["timestamp"])
        tz = CFG["istanbul_tz"] if ds == "istanbul" else MASSIVE_STEPS[ds][1]
        df["t_local"] = (df["t_utc"].dt.tz_localize("UTC").dt.tz_convert(tz)
                         .dt.tz_localize(None))
        lat = df.groupby("venue_raw")["lat"].transform("first")
        lon = df.groupby("venue_raw")["lon"].transform("first")
        df["lat"], df["lon"] = lat, lon
        miss = df["lat"].isna() | df["lon"].isna()
        stats["dropped_no_coordinates_checkins"] = int(miss.sum())
        stats["dropped_no_coordinates_venues"] = int(df.loc[miss, "venue_raw"].nunique())
        df = df[~miss].copy()
    else:
        raise ValueError(ds)
    df = df.reset_index(drop=True)
    df["raw_order"] = np.arange(len(df), dtype=np.int64)
    stats["exact_duplicate_checkins_kept"] = int(df.duplicated(["user_raw", "venue_raw", "t_utc"]).sum())
    return df[["user_raw", "venue_raw", "category", "lat", "lon", "t_utc", "t_local", "raw_order"]], stats


def gowalla_category_names(raw_root: str | None = None) -> pd.Series:
    import ast
    sp = pd.read_csv(raw_path(GOWALLA_SPOTS, raw_root), usecols=["id", "spot_categories"])
    assert sp["id"].is_unique
    parsed = {}
    for s in pd.unique(sp["spot_categories"].astype(str)):
        cats = ast.literal_eval(s)
        assert len(cats) == 1, s
        parsed[s] = (cats[0]["url"].rsplit("/", 1)[-1], cats[0]["name"])
    ids_of_name: dict = {}
    for cid, name in set(parsed.values()):
        ids_of_name.setdefault(name, set()).add(cid)
    label = {s: (name if len(ids_of_name[name]) == 1 else f"{name} (gowalla:{cid})") for s, (cid, name) in parsed.items()}
    return pd.Series(sp["spot_categories"].astype(str).map(label).to_numpy(), index=sp["id"].to_numpy())


def _load_gowalla_ca(raw_root: str | None) -> tuple[pd.DataFrame, dict]:
    from matplotlib.path import Path
    d = pd.read_csv(raw_path(GOWALLA_SNAP, raw_root), sep="\t", header=None,
                    names=["user", "utc", "lat", "lon", "loc"],
                    dtype={"user": np.int64, "utc": str, "lat": np.float64, "lon": np.float64, "loc": np.int64})
    stats = {"snap_checkins_worldwide": int(len(d))}
    cat = d["loc"].map(gowalla_category_names(raw_root))
    d = d[cat.notna()].assign(category=cat[cat.notna()])
    stats["snap_checkins_with_public_category"] = int(len(d))
    with open(raw_path(CA_POLYGONS_KEY, raw_root)) as f:
        feats = json.load(f)["features"]
    pts = d[["lon", "lat"]].to_numpy()
    inside = np.zeros(len(d), dtype=bool)
    for ft in feats:
        if ft["properties"]["name"].lower() in CA_STATES:
            inside |= Path(np.asarray(ft["geometry"]["coordinates"][0], dtype=np.float64)).contains_points(pts)
    d = d[inside]
    stats["raw_checkins"] = int(len(d))
    t = pd.to_datetime(d["utc"], format="%Y-%m-%dT%H:%M:%SZ")
    df = pd.DataFrame({
        "user_raw": d["user"].astype(str), "venue_raw": d["loc"].astype(str), "category": d["category"].astype(str),
        "lat": d["lat"].to_numpy(), "lon": d["lon"].to_numpy(),
        "t_utc": t.astype("datetime64[us]"),
        "t_local": t.dt.tz_localize("UTC").dt.tz_convert(CA_TZ).dt.tz_localize(None).astype("datetime64[us]"),
    })
    return df, stats


def _load_tist(ds: str, raw_root: str | None = None) -> tuple[pd.DataFrame, dict]:
    ex = os.path.join(raw_path(TIST_KEY, raw_root), "extracted")
    pois = pd.read_csv(os.path.join(ex, "dataset_TIST2015_POIs.txt"), sep="\t", header=None, quoting=3,
                       names=["venue_raw", "lat", "lon", "category", "cc"],
                       dtype={"venue_raw": str, "lat": np.float64, "lon": np.float64, "category": str, "cc": str})
    assert pois["venue_raw"].is_unique
    pois = pois[pois["cc"] == TIST_COUNTRY[ds]].set_index("venue_raw")
    parts = []
    for ch in pd.read_csv(os.path.join(ex, "dataset_TIST2015_Checkins.txt"), sep="\t", header=None, quoting=3,
                          names=["user_raw", "venue_raw", "utc", "tz_min"], chunksize=4_000_000,
                          dtype={"user_raw": str, "venue_raw": str, "utc": str, "tz_min": np.int64}):
        parts.append(ch[ch["venue_raw"].isin(pois.index)])
    df = pd.concat(parts, ignore_index=True)
    stats = {"raw_checkins": int(len(df)), "country_venues_in_poi_file": int(len(pois))}
    df = df.join(pois[["lat", "lon", "category"]], on="venue_raw")
    df["t_utc"] = pd.to_datetime(df["utc"], format="%a %b %d %H:%M:%S +0000 %Y").astype("datetime64[us]")
    df["t_local"] = df["t_utc"] + pd.to_timedelta(df["tz_min"], unit="m")
    stats["rows_with_modal_offset"] = int((df["tz_min"] == df["tz_min"].mode().iloc[0]).sum())
    return df, stats


def flashback_filter(df: pd.DataFrame, ds: str, stats: dict) -> pd.DataFrame:
    fb = FLASHBACK[ds]
    it = 0
    while True:
        n = len(df)
        vc = df["venue_raw"].value_counts()
        df = df[df["venue_raw"].map(vc).to_numpy() >= fb["min_poi"]]
        uc = df["user_raw"].value_counts()
        df = df[df["user_raw"].map(uc).to_numpy() >= fb["min_user"]]
        it += 1
        if len(df) == n:
            break
    stats.update(flashback_core_iterations=it, flashback_core_checkins=int(len(df)),
                 flashback_core_users=int(df["user_raw"].nunique()),
                 flashback_core_venues=int(df["venue_raw"].nunique()))
    uc = df["user_raw"].value_counts()
    df = df[df["user_raw"].map(uc).to_numpy() >= fb["min_user_loader"]].copy()
    stats.update(flashback_loader_checkins=int(len(df)), flashback_loader_users=int(df["user_raw"].nunique()),
                 flashback_loader_venues=int(df["venue_raw"].nunique()))
    return df


def _load_gowalla_global(raw_root: str | None) -> tuple[pd.DataFrame, dict]:
    d = pd.read_csv(raw_path(GOWALLA_SNAP, raw_root), sep="\t", header=None,
                    names=["user", "utc", "lat", "lon", "loc"],
                    dtype={"user": np.int64, "utc": str, "lat": np.float64, "lon": np.float64, "loc": np.int64})
    stats = {"raw_checkins": int(len(d))}
    dup = d.duplicated(["user", "utc", "loc"])
    stats["dropped_exact_duplicates"] = int(dup.sum())
    d = d[~dup]
    d = d.assign(user_raw=d["user"].astype(str), venue_raw=d["loc"].astype(str))
    d = flashback_filter(d, "gowalla", stats)
    cat = d["loc"].map(gowalla_category_names(raw_root))
    stats["venues_without_liu_category"] = int(d.loc[cat.isna(), "venue_raw"].nunique())
    stats["checkins_at_venues_without_liu_category"] = int(cat.isna().sum())
    tz = pd.read_parquet(raw_path(GOWALLA_TZ_KEY, raw_root)).set_index("loc")["tz"]
    zone = d["loc"].map(tz)
    assert zone.notna().all(), "spot without a time zone in GOWALLA_TZ"
    t = pd.to_datetime(d["utc"], format="%Y-%m-%dT%H:%M:%SZ").astype("datetime64[us]")
    tl = pd.Series(np.empty(len(d), dtype="datetime64[us]"), index=d.index)
    for z, idx in zone.groupby(zone).groups.items():
        tl.loc[idx] = t.loc[idx].dt.tz_localize("UTC").dt.tz_convert(z).dt.tz_localize(None).to_numpy()
    stats["time_zones"] = int(zone.nunique())
    bad = ~(d["lat"].between(-90, 90) & d["lon"].between(-180, 180))
    stats["checkins_at_spots_with_out_of_range_coordinates"] = int(bad.sum())
    stats["spots_with_out_of_range_coordinates"] = int(d.loc[bad, "loc"].nunique())
    df = pd.DataFrame({
        "user_raw": d["user_raw"].to_numpy(), "venue_raw": d["venue_raw"].to_numpy(),
        "category": cat.fillna(GOWALLA_NO_CATEGORY).astype(str).to_numpy(),
        "lat": d["lat"].to_numpy(), "lon": d["lon"].to_numpy(),
        "t_utc": t.to_numpy(), "t_local": tl.to_numpy().astype("datetime64[us]"),
    })
    return df, stats


def _load_foursquare_global(raw_root: str | None = None) -> tuple[pd.DataFrame, dict]:
    ex = os.path.join(raw_path(WWW2019_KEY, raw_root), "extracted")
    d = pd.read_csv(os.path.join(ex, "raw_Checkins_anonymized.txt"), sep="\t", header=None, quoting=3,
                    names=["user_raw", "venue_raw", "utc", "tz_min"],
                    dtype={"user_raw": str, "venue_raw": str, "utc": str, "tz_min": str})
    stats = {"raw_checkins": int(len(d))}
    t = pd.to_datetime(d["utc"], format="%a %b %d %H:%M:%S +0000 %Y", errors="coerce")
    off = pd.to_numeric(d["tz_min"], errors="coerce")
    bad = t.isna().to_numpy() | off.isna().to_numpy()
    stats["dropped_unparseable_rows"] = int(bad.sum())
    d = d.drop(columns=["utc", "tz_min"]).assign(t_utc=t.astype("datetime64[us]"), tz_min=off)[~bad]
    d = flashback_filter(d, "foursquare_global", stats)
    pois = pd.read_csv(os.path.join(ex, "raw_POIs.txt"), sep="\t", header=None, quoting=3,
                       names=["venue_raw", "lat", "lon", "category", "cc"],
                       dtype={"venue_raw": str, "lat": np.float64, "lon": np.float64, "category": str, "cc": str})
    pois = pois[pois["venue_raw"].isin(pd.unique(d["venue_raw"]))]
    assert pois["venue_raw"].is_unique
    pois = pois.set_index("venue_raw")
    d = d.join(pois[["lat", "lon", "category"]], on="venue_raw")
    assert d[["lat", "lon", "category"]].notna().all().all(), "venue missing from raw_POIs.txt"
    stats["countries"] = int(pois["cc"].nunique())
    d["t_local"] = d["t_utc"] + pd.to_timedelta(d["tz_min"].astype(np.int64), unit="m")
    return d[["user_raw", "venue_raw", "category", "lat", "lon", "t_utc", "t_local"]], stats


def canonical_venues(df: pd.DataFrame) -> pd.DataFrame:
    cat = (df.groupby(["venue_raw", "category"]).size().rename("n").reset_index()
           .sort_values(["venue_raw", "n", "category"], ascending=[True, False, True])
           .drop_duplicates("venue_raw").set_index("venue_raw")["category"])
    ll = df.groupby("venue_raw")[["lat", "lon"]].median()
    return ll.join(cat)


def _sort_key_raw(ids: pd.Series) -> list:
    u = pd.unique(ids)
    try:
        return sorted(u, key=lambda s: (int(s), s))
    except ValueError:
        return sorted(u)


def preprocess(df: pd.DataFrame, stats: dict, cuts: tuple | None = None, freq_filter: bool = True
               ) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    venue_attr = canonical_venues(df)
    df = df.drop(columns=["category", "lat", "lon"]).join(venue_attr, on="venue_raw")

    if not freq_filter:
        assert cuts is None
        stats["frequency_filter"] = "flashback (applied in load_raw); PROTOCOL step 2 not applied"
    elif cuts is None:
        vc = df["venue_raw"].value_counts()
        df = df[df["venue_raw"].isin(vc.index[vc >= CFG["min_poi_checkins"]])]
        uc = df["user_raw"].value_counts()
        df = df[df["user_raw"].isin(uc.index[uc >= CFG["min_user_checkins"]])].copy()
    else:
        t_cut = pd.Timestamp(cuts[0])
        vc = df.loc[df["t_utc"] < t_cut, "venue_raw"].value_counts()
        df = df[df["venue_raw"].isin(vc.index[vc >= CFG["min_poi_checkins"]])]
        uc = df.loc[df["t_utc"] < t_cut, "user_raw"].value_counts()
        df = df[df["user_raw"].isin(uc.index[uc >= CFG["min_user_checkins"]])].copy()
        stats["filter_counts_only_before_utc"] = str(t_cut)
    stats["after_filter_checkins"] = len(df)
    stats["after_filter_users"] = int(df["user_raw"].nunique())
    stats["after_filter_venues"] = int(df["venue_raw"].nunique())

    df = df.sort_values(["t_utc", "raw_order"]).reset_index(drop=True)
    n = len(df)
    if cuts is None:
        t_val = df["t_utc"].iloc[int(n * CFG["train_end_frac"])]
        t_test = df["t_utc"].iloc[int(n * CFG["val_end_frac"])]
    else:
        t_val, t_test = pd.Timestamp(cuts[0]), pd.Timestamp(cuts[1])
    df["split"] = np.where(df["t_utc"] < t_val, "train", np.where(df["t_utc"] < t_test, "val", "test"))

    df = df.sort_values(["user_raw", "t_utc", "raw_order"]).reset_index(drop=True)
    same_prev = df["user_raw"].eq(df["user_raw"].shift(1))
    same_next = df["user_raw"].eq(df["user_raw"].shift(-1))
    gap_prev = (df["t_utc"] - df["t_utc"].shift(1)).dt.total_seconds()
    gap_next = (df["t_utc"].shift(-1) - df["t_utc"]).dt.total_seconds()
    close_prev = same_prev & (gap_prev <= CFG["traj_gap_s"])
    close_next = same_next & (gap_next <= CFG["traj_gap_s"])
    isolated = ~(close_prev | close_next)
    stats["removed_isolated_checkins"] = int(isolated.sum())
    df = df[~isolated].copy()

    users = _sort_key_raw(df["user_raw"])
    venues = _sort_key_raw(df["venue_raw"])
    umap = {u: i for i, u in enumerate(users)}
    vmap = {v: i for i, v in enumerate(venues)}
    df["user_id"] = df["user_raw"].map(umap).astype(np.int32)
    df["venue_id"] = df["venue_raw"].map(vmap).astype(np.int32)
    df = df.sort_values(["user_id", "t_utc", "raw_order"]).reset_index(drop=True)
    df["row_id"] = np.arange(len(df), dtype=np.int64)

    new_user = ~df["user_id"].eq(df["user_id"].shift(1))
    gap = (df["t_utc"] - df["t_utc"].shift(1)).dt.total_seconds()
    new_traj = new_user | (gap > CFG["traj_gap_s"])
    df["traj_id"] = (np.cumsum(new_traj.to_numpy()) - 1).astype(np.int32)

    ntr = df.loc[df["split"] == "train", "venue_id"].value_counts()
    vt = (pd.DataFrame({"venue_id": np.arange(len(venues), dtype=np.int32), "venue_raw": venues})
          .join(venue_attr, on="venue_raw"))
    vt["n_train_visits"] = vt["venue_id"].map(ntr).fillna(0).astype(np.int64)

    checkins = df[["row_id", "user_id", "venue_id", "category", "lat", "lon", "t_utc", "t_local",
                   "traj_id", "split"]].copy()
    meta = dict(t_val_start_utc=str(t_val), t_test_start_utc=str(t_test), n_filtered=n)
    id_maps = dict(users=pd.DataFrame({"user_id": np.arange(len(users), dtype=np.int32), "user_raw": users}),
                   venues=vt[["venue_id", "venue_raw"]].copy())
    return checkins, vt[["venue_id", "category", "lat", "lon", "n_train_visits"]], meta, id_maps


@dataclass
class Arrays:
    user: np.ndarray
    venue: np.ndarray
    t: np.ndarray
    traj: np.ndarray
    is_train: np.ndarray
    split: np.ndarray
    traj_start: np.ndarray
    user_first: np.ndarray
    user_ntrain: np.ndarray

    @classmethod
    def from_checkins(cls, ck: pd.DataFrame) -> "Arrays":
        ck = ck.sort_values("row_id")
        assert (ck["row_id"].to_numpy() == np.arange(len(ck))).all()
        user = ck["user_id"].to_numpy(np.int64)
        traj = ck["traj_id"].to_numpy(np.int64)
        split = ck["split"].map({"train": 0, "val": 1, "test": 2}).to_numpy(np.int8)
        n_users = int(user.max()) + 1
        rows = np.arange(len(ck), dtype=np.int64)
        ts = np.where(np.r_[True, traj[1:] != traj[:-1]], rows, 0)
        ts = np.maximum.accumulate(ts)
        uf = np.full(n_users, -1, dtype=np.int64)
        first = np.r_[True, user[1:] != user[:-1]]
        uf[user[first]] = rows[first]
        ntr = np.bincount(user[split == 0], minlength=n_users).astype(np.int64)
        lastpos = rows - uf[user]
        assert ((split == 0) == (lastpos < ntr[user])).all(), "train rows are not a per-user prefix"
        t = ck["t_utc"].to_numpy("datetime64[s]").astype(np.int64)
        return cls(user=user, venue=ck["venue_id"].to_numpy(np.int64), t=t, traj=traj,
                   is_train=split == 0, split=split, traj_start=ts, user_first=uf, user_ntrain=ntr)


def evidence_slices(target_rows: np.ndarray, A: Arrays, vis_from: np.ndarray | None = None) -> dict:
    r = np.asarray(target_rows, dtype=np.int64)
    u = A.user[r]
    first = A.user_first[u]
    lo = first if vis_from is None else np.maximum(first, vis_from[u])
    hist_end = first + A.user_ntrain[u]
    ts = A.traj_start[r]
    is_tr = A.split[r] == 0
    p_lo = np.where(is_tr, np.maximum(lo, ts), ts)
    h_lo = lo
    h_hi = np.where(is_tr, p_lo, np.maximum(lo, np.minimum(hist_end, ts)))
    return dict(hist_lo=h_lo, hist_hi=h_hi, prefix_lo=p_lo, prefix_hi=r)


def build_queries(ck: pd.DataFrame, venues: pd.DataFrame, A: Arrays) -> tuple[dict, dict]:
    rows = np.arange(len(A.user), dtype=np.int64)
    is_q = rows != A.traj_start
    train_users = np.zeros(len(A.user_first), dtype=bool)
    train_users[A.user_ntrain > 0] = True
    seen_venue = venues.sort_values("venue_id")["n_train_visits"].to_numpy() > 0
    out, st = {}, {}
    for si, s in enumerate(SPLITS):
        m = is_q & (A.split == si)
        st[f"{s}_candidate_targets"] = int(m.sum())
        if si > 0:
            keep = train_users[A.user] & seen_venue[A.venue]
            st[f"{s}_dropped_unseen_user_or_venue"] = int((m & ~keep).sum())
            m &= keep
        r = rows[m]
        sl = evidence_slices(r, A)
        hlen, inh = _hist_stats(r, sl, A)
        tl = ck["t_local"].to_numpy()
        out[s] = pd.DataFrame({
            "query_id": r, "user_id": A.user[r].astype(np.int32), "traj_id": A.traj[r].astype(np.int32),
            "target_row_id": r, "target_venue_id": A.venue[r].astype(np.int32),
            "target_t_local": tl[r], "prev_row_id": r - 1,
            "in_user_history": inh, "user_hist_len": hlen.astype(np.int32),
        })
    return out, st


def _hist_stats(r: np.ndarray, sl: dict, A: Arrays) -> tuple[np.ndarray, np.ndarray]:
    hlen = (sl["hist_hi"] - sl["hist_lo"]) + (sl["prefix_hi"] - sl["prefix_lo"])
    inh = np.zeros(len(r), dtype=bool)
    V = A.venue
    for i in range(len(r)):
        v = V[r[i]]
        a, b, c, d = sl["hist_lo"][i], sl["hist_hi"][i], sl["prefix_lo"][i], sl["prefix_hi"][i]
        inh[i] = bool((V[a:b] == v).any() or (V[c:d] == v).any())
    return hlen, inh


def make_policies(A: Arrays, queries_train: pd.DataFrame, t_val: str) -> dict:
    pol = {}
    n_users = len(A.user_first)
    qr = queries_train["target_row_id"].to_numpy(np.int64)
    for x in POLICY_X:
        keep = np.ceil(A.user_ntrain * x / 100.0).astype(np.int64)
        vis_from = A.user_first + A.user_ntrain - keep
        allowed = (qr - 1) >= vis_from[A.user[qr]]
        tq = qr[allowed]
        n_vis = int(keep.sum())
        common = dict(x=x, cutoff_t_utc=t_val, n_users=n_users,
                      n_train_queries=int(len(tq)), frac_train_queries=float(len(tq) / max(1, len(qr))),
                      train_query_ids=[int(v) for v in tq])
        pol[f"H-{x}"] = dict(policy=f"H-{x}", kind="H", **common,
                             n_visible_precutoff_checkins=n_vis,
                             frac_visible_precutoff_checkins=float(n_vis / max(1, A.user_ntrain.sum())),
                             user_visible_from_row_id={str(i): int(v) for i, v in enumerate(vis_from)})
        pol[f"S-{x}"] = dict(policy=f"S-{x}", kind="S", **common,
                             n_visible_precutoff_checkins=int(A.user_ntrain.sum()),
                             frac_visible_precutoff_checkins=1.0,
                             user_visible_from_row_id={str(i): int(v) for i, v in enumerate(A.user_first)})
    return pol


POLICY_TEXT = {
    "H": ("History budget: for every user only the latest x% (ceil) of their pre-cutoff (train-split) check-ins "
          "are visible to ANY component: graphs, transition/popularity statistics, retrieval stores, user profiles, "
          "prompts and training samples. A pre-cutoff row r of user u is visible iff r >= user_visible_from_row_id[u]. "
          "Train queries usable for gradient updates are listed in train_query_ids (previous check-in visible); the "
          "input prefix of a train query is truncated to visible rows. For val/test queries the current trajectory "
          "prefix is the query input and is always visible. Use the candidates listed under 'candidates'."),
    "S": ("Sample budget: all pre-cutoff evidence is visible (as in H-100), but only the train queries listed in "
          "train_query_ids may be used for gradient updates / as supervised examples (few-shot demonstrations count "
          "as samples). The list is exactly the H-x train-query set, so H-x vs S-x differ only in visible evidence "
          "and S-x vs S-100 differ only in the number of samples."),
}


SU = dict(policy="S-u", seed=0, target_sft_trajectories=51611,
          target_source="tky H-100: train trajectories with >= 1 train query (exports/tky/H-100, LLM4POI SFT set)")
SU_DATASETS = ("gowalla", "foursquare_global")
POLICY_TEXT_SU = (
    "User-subsample sample budget (S-u): all pre-cutoff evidence is visible (as in H-100 / S-x), but only the train "
    "queries of a seed-0 random subset of users may be used for gradient updates / as supervised examples. Users = the "
    "first k of numpy default_rng(0).permutation(sorted ids of users with >= 1 train query), k = the smallest prefix "
    "whose number of train trajectories with >= 1 train query reaches tky H-100's (51,611). Val/test queries: all.")


def make_su_policy(A: Arrays, queries_train: pd.DataFrame, t_val: str) -> dict:
    q = queries_train[["query_id", "user_id", "traj_id"]]
    users = np.sort(q["user_id"].unique().astype(np.int64))
    perm = np.random.default_rng(SU["seed"]).permutation(users)
    ntraj = q.groupby("user_id")["traj_id"].nunique().reindex(perm).to_numpy()
    k = int(np.searchsorted(np.cumsum(ntraj), SU["target_sft_trajectories"]) + 1)
    k = min(k, len(perm))
    chosen = np.sort(perm[:k])
    tq = np.sort(q.loc[q["user_id"].isin(set(chosen.tolist())), "query_id"].to_numpy(np.int64))
    n_users = len(A.user_first)
    return dict(policy=SU["policy"], kind="S", x=None, cutoff_t_utc=t_val, n_users=n_users,
                n_train_queries=int(len(tq)), frac_train_queries=float(len(tq) / max(1, len(q))),
                train_query_ids=[int(v) for v in tq],
                n_visible_precutoff_checkins=int(A.user_ntrain.sum()), frac_visible_precutoff_checkins=1.0,
                user_visible_from_row_id={str(i): int(v) for i, v in enumerate(A.user_first)},
                subsample=dict(seed=SU["seed"], k=k, n_candidate_users=int(len(users)), frac_users=float(k / len(users)),
                               users=[int(u) for u in chosen], n_sft_trajectories=int(ntraj[:k].sum()),
                               target_sft_trajectories=SU["target_sft_trajectories"],
                               target_source=SU["target_source"]),
                description=POLICY_TEXT_SU,
                public_description=POLICY_TEXT_SU,
                candidates={s: f"candidates_{s}.parquet" for s in SPLITS})


def policy_vis_from(pol: dict, n_users: int) -> np.ndarray:
    v = np.zeros(n_users, dtype=np.int64)
    for k, val in pol["user_visible_from_row_id"].items():
        v[int(k)] = val
    return v


_G: dict = {}


def _rank_key(ids, primary, pop):
    return np.lexsort((ids, -pop[ids], -primary))


def prepare_retriever(A: Arrays, venues: pd.DataFrame, vis_from: np.ndarray | None) -> dict:
    from sklearn.neighbors import BallTree
    nV = len(venues)
    rows = np.arange(len(A.user), dtype=np.int64)
    vis = A.is_train.copy()
    if vis_from is not None:
        vis &= rows >= vis_from[A.user]
    pop = np.bincount(A.venue[vis], minlength=nV).astype(np.float64)
    pm = np.r_[False, (A.traj[1:] == A.traj[:-1]) & vis[1:] & vis[:-1]]
    a, b = A.venue[rows[pm] - 1], A.venue[rows[pm]]
    pairs = pd.DataFrame({"a": a, "b": b}).value_counts().rename("c").reset_index()
    trans = {}
    for va, g in pairs.groupby("a", sort=False):
        ids = g["b"].to_numpy(np.int64)
        cnt = g["c"].to_numpy(np.float64)
        o = _rank_key(ids, cnt, pop)
        trans[int(va)] = (ids[o], cnt[o])
    cat_mask = venues.sort_values("venue_id")["n_train_visits"].to_numpy() > 0
    cat_ids = np.flatnonzero(cat_mask)
    vv = venues.sort_values("venue_id")
    ll = np.radians(vv[["lat", "lon"]].to_numpy())
    tree = BallTree(ll[cat_ids], metric="haversine")
    k = min(RETRIEVER["geo_k"], len(cat_ids))
    _, nn = tree.query(ll, k=k)
    knn = cat_ids[nn]
    gorder = cat_ids[_rank_key(cat_ids, pop[cat_ids], pop)]
    return dict(pop=pop, trans=trans, knn=knn, cat_mask=cat_mask, global_order=gorder)


def _topk(ids: np.ndarray, score: np.ndarray, pop: np.ndarray, k: int) -> np.ndarray:
    if len(ids) == 0:
        return ids
    o = _rank_key(ids, score, pop)
    return ids[o[:k]]


def _retrieve_one(prev_v: int, hist_venues: np.ndarray, hist_w: np.ndarray, loo_target: int, st: dict) -> np.ndarray:
    L, K = RETRIEVER["list_len"], RETRIEVER["rrf_k"]
    pop, cat = st["pop"], st["cat_mask"]
    if len(hist_venues):
        uv, inv = np.unique(hist_venues, return_inverse=True)
        uw = np.bincount(inv, weights=hist_w)
        m = cat[uv]
        U = _topk(uv[m], uw[m], pop, L)
    else:
        U = np.empty(0, dtype=np.int64)
    T = np.empty(0, dtype=np.int64)
    if prev_v in st["trans"]:
        ids, cnt = st["trans"][prev_v]
        if loo_target >= 0:
            cnt = cnt.copy()
            cnt[ids == loo_target] -= 1.0
            keep = cnt > 0
            T = _topk(ids[keep], cnt[keep], pop, L)
        else:
            T = ids[:L]
    g = st["knn"][prev_v]
    gp = pop[g]
    if loo_target >= 0:
        gp = gp - (g == loo_target)
    G = _topk(g, gp, pop, L)
    allv = np.concatenate([U, T, G])
    sc = np.concatenate([1.0 / (K + np.arange(1, len(U) + 1)), 1.0 / (K + np.arange(1, len(T) + 1)),
                         1.0 / (K + np.arange(1, len(G) + 1))])
    uv, inv = np.unique(allv, return_inverse=True)
    us = np.bincount(inv, weights=sc)
    out = _topk(uv, us, pop, RETRIEVER["out_len"])
    if len(out) < RETRIEVER["out_len"]:
        extra = st["global_order"][~np.isin(st["global_order"], out)][: RETRIEVER["out_len"] - len(out)]
        out = np.concatenate([out, extra])
    return out


def _retrieve_chunk(args):
    rows_chunk = args
    A, st, sl, loo = _G["A"], _G["st"], _G["sl"], _G["loo"]
    h = RETRIEVER["half_life_days"] * 86400.0
    t0 = _G["t0"]
    res = []
    for i in rows_chunk:
        r = int(sl["target"][i])
        a, b, c, d = sl["hist_lo"][i], sl["hist_hi"][i], sl["prefix_lo"][i], sl["prefix_hi"][i]
        idx = np.r_[np.arange(a, b), np.arange(c, d)]
        hv = A.venue[idx]
        hw = np.exp2((A.t[idx] - t0) / h)
        res.append((r, _retrieve_one(int(A.venue[r - 1]), hv, hw, int(A.venue[r]) if loo[i] else -1, st)))
    return res


def retrieve(A: Arrays, venues: pd.DataFrame, target_rows: np.ndarray, vis_from: np.ndarray | None,
             workers: int = 1, st: dict | None = None) -> pd.DataFrame:
    import multiprocessing as mp
    if st is None:
        st = prepare_retriever(A, venues, vis_from)
    r = np.asarray(target_rows, dtype=np.int64)
    sl = evidence_slices(r, A, vis_from)
    sl["target"] = r
    loo = A.split[r] == 0
    _G.update(A=A, st=st, sl=sl, loo=loo, t0=int(A.t.min()))
    idx = np.arange(len(r))
    if workers <= 1 or len(r) < 2000:
        res = _retrieve_chunk(idx)
    else:
        chunks = np.array_split(idx, workers * 8)
        with mp.get_context("fork").Pool(workers) as pool:
            res = [x for part in pool.map(_retrieve_chunk, chunks) for x in part]
    _G.clear()
    qid = np.array([x[0] for x in res], dtype=np.int64)
    cands = [x[1].astype(np.int64).tolist() for x in res]
    df = pd.DataFrame({"query_id": qid, "cand_venue_ids": cands})
    return df.sort_values("query_id").reset_index(drop=True)


def candidate_recall(queries: pd.DataFrame, cands: pd.DataFrame, ks=(20, 50), *, row_split,
                     root: str = ROOT) -> dict:
    sp = np.asarray(row_split)[queries["target_row_id"].to_numpy(np.int64)]
    if ((sp == 2) | (sp == "test")).any():
        from .evaluate import assert_split_allowed
        assert_split_allowed(root, "test")
    m = queries[["query_id", "target_venue_id"]].merge(cands, on="query_id", how="left")
    out = {}
    for k in ks:
        hit = np.array([t in c[:k] for t, c in zip(m["target_venue_id"], m["cand_venue_ids"])])
        out[f"recall@{k}"] = float(hit.mean())
    return out


def load_checkins(ds: str, root: str = ROOT) -> pd.DataFrame:
    return pd.read_parquet(os.path.join(data_dir(ds, root), "checkins.parquet"))


def load_venues(ds: str, root: str = ROOT) -> pd.DataFrame:
    return pd.read_parquet(os.path.join(data_dir(ds, root), "venues.parquet"))


def load_queries(ds: str, split: str, root: str = ROOT) -> pd.DataFrame:
    return pd.read_parquet(os.path.join(data_dir(ds, root), f"queries_{split}.parquet"))


def load_policy(ds: str, policy: str = "H-100", root: str = ROOT) -> dict:
    with open(os.path.join(data_dir(ds, root), "evidence", f"{policy}.json")) as f:
        return json.load(f)


def load_candidates(ds: str, split: str, policy: str = "H-100", root: str = ROOT) -> pd.DataFrame:
    if policy.startswith("H-") and policy != "H-100":
        p = os.path.join(data_dir(ds, root), "evidence", f"{policy}.candidates_{split}.parquet")
    else:
        p = os.path.join(data_dir(ds, root), f"candidates_{split}.parquet")
    return pd.read_parquet(p)


def query_evidence(ds: str, split: str, policy: str = "H-100", root: str = ROOT,
                   checkins: pd.DataFrame | None = None) -> pd.DataFrame:
    ck = load_checkins(ds, root) if checkins is None else checkins
    A = Arrays.from_checkins(ck)
    pol = load_policy(ds, policy, root)
    q = load_queries(ds, split, root)
    if split == "train":
        q = q[q["query_id"].isin(set(pol["train_query_ids"]))]
    vf = policy_vis_from(pol, len(A.user_first)) if pol["kind"] == "H" else None
    sl = evidence_slices(q["target_row_id"].to_numpy(), A, vf)
    return pd.DataFrame({"query_id": q["query_id"].to_numpy(), **sl})
