from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from poibench.data import Arrays, evidence_slices

VID = {c: i for i, c in enumerate("ABCDEF")}
NAME = {i: c for c, i in VID.items()}

HAND_ROWS = [
    (0, "A", "2012-01-01 10:00", 1, "train"),
    (0, "B", "2012-01-01 11:00", 1, "train"),
    (0, "C", "2012-01-01 12:00", 1, "train"),
    (0, "A", "2012-01-02 10:00", 2, "train"),
    (0, "B", "2012-01-02 11:00", 2, "train"),
    (0, "E", "2012-02-01 10:00", 6, "val"),
    (0, "A", "2012-02-01 11:00", 6, "val"),
    (0, "D", "2012-02-01 12:00", 6, "val"),
    (0, "C", "2012-03-01 10:00", 7, "test"),
    (0, "E", "2012-03-01 11:00", 7, "test"),
    (0, "B", "2012-03-01 12:00", 7, "test"),
    (1, "A", "2012-01-03 10:00", 3, "train"),
    (1, "C", "2012-01-03 11:00", 3, "train"),
    (1, "A", "2012-01-03 12:00", 3, "train"),
    (1, "D", "2012-01-04 10:00", 4, "train"),
    (1, "A", "2012-01-04 11:00", 4, "train"),
    (2, "B", "2012-01-05 10:00", 5, "train"),
    (2, "C", "2012-01-05 11:00", 5, "train"),
    (2, "B", "2012-03-02 10:00", 8, "test"),
    (2, "A", "2012-03-02 11:00", 8, "test"),
]


def build_tables(rows, n_venues=None):
    ck = pd.DataFrame(rows, columns=["user_id", "venue_id", "t_utc", "traj_id", "split"])
    ck["venue_id"] = [VID[v] if isinstance(v, str) else int(v) for v in ck["venue_id"]]
    ck["t_utc"] = pd.to_datetime(ck["t_utc"]).astype("datetime64[us]")
    ck = ck.sort_values(["user_id", "t_utc"], kind="mergesort").reset_index(drop=True)
    ck["row_id"] = np.arange(len(ck), dtype=np.int64)
    ck["user_id"] = ck["user_id"].astype(np.int32)
    ck["venue_id"] = ck["venue_id"].astype(np.int32)
    ck["traj_id"] = ck["traj_id"].astype(np.int32)
    ck["category"] = "cat" + (ck["venue_id"] % 3).astype(str)
    ck["t_local"] = ck["t_utc"] - pd.Timedelta(hours=5)
    ck["lat"] = 40.0 + ck["venue_id"] * 0.01
    ck["lon"] = -73.0 - ck["venue_id"] * 0.01
    ck = ck[["row_id", "user_id", "venue_id", "category", "lat", "lon", "t_utc", "t_local", "traj_id", "split"]]

    V = int(n_venues if n_venues is not None else ck.venue_id.max() + 1)
    train_counts = np.bincount(ck.venue_id[ck.split == "train"], minlength=V)
    venues = pd.DataFrame({"venue_id": np.arange(V, dtype=np.int32), "category": [f"cat{v % 3}" for v in range(V)],
                           "lat": 40.0 + np.arange(V) * 0.01, "lon": -73.0 - np.arange(V) * 0.01,
                           "n_train_visits": train_counts.astype(np.int64)})

    A = Arrays.from_checkins(ck)
    rows_all = np.arange(len(ck), dtype=np.int64)
    is_q = rows_all != A.traj_start
    queries = {}
    for si, s in enumerate(("train", "val", "test")):
        r = rows_all[is_q & (A.split == si)]
        sl = evidence_slices(r, A)
        hlen = (sl["hist_hi"] - sl["hist_lo"]) + (sl["prefix_hi"] - sl["prefix_lo"])
        inh = np.array([bool((A.venue[a:b] == A.venue[x]).any() or (A.venue[c:d] == A.venue[x]).any())
                        for x, a, b, c, d in zip(r, sl["hist_lo"], sl["hist_hi"], sl["prefix_lo"], sl["prefix_hi"])],
                       dtype=bool)
        queries[s] = pd.DataFrame({
            "query_id": r, "user_id": A.user[r].astype(np.int32), "traj_id": A.traj[r].astype(np.int32),
            "target_row_id": r, "target_venue_id": A.venue[r].astype(np.int32),
            "target_t_local": ck["t_local"].to_numpy()[r], "prev_row_id": r - 1,
            "in_user_history": inh, "user_hist_len": hlen.astype(np.int32)})

    order = np.lexsort((np.arange(V), -train_counts))[:50].tolist()
    cands = {s: pd.DataFrame({"query_id": q["query_id"], "cand_venue_ids": [order] * len(q)})
             for s, q in queries.items()}
    return {"checkins": ck, "venues": venues, "queries": queries, "candidates": cands}


def hand_tables():
    return build_tables(HAND_ROWS, n_venues=6)


def random_rows(n_users=40, n_venues=60, n_trajs_per_user=8, max_len=6, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    w = 1.0 / np.arange(1, n_venues + 1) ** 0.8
    w /= w.sum()
    base = pd.Timestamp("2012-04-01")
    traj = 0
    for u in range(n_users):
        fav = rng.choice(n_venues, size=5, replace=False)
        for k in range(n_trajs_per_user):
            day = base + pd.Timedelta(days=int(k * 10 + rng.integers(0, 5)))
            split = "train" if k < n_trajs_per_user * 0.6 else ("val" if k < n_trajs_per_user * 0.8 else "test")
            L = int(rng.integers(2, max_len + 1))
            for j in range(L):
                v = fav[rng.integers(0, 5)] if rng.random() < 0.5 else rng.choice(n_venues, p=w)
                t = day + pd.Timedelta(minutes=int(60 * j + rng.integers(0, 30)))
                rows.append((u, int(v), str(t), traj, split))
            traj += 1
    return rows, n_venues


def h_policy(tables, x, kind="H"):
    A = Arrays.from_checkins(tables["checkins"])
    keep = np.ceil(A.user_ntrain * x / 100.0).astype(np.int64)
    vis_from = A.user_first + A.user_ntrain - keep
    qr = tables["queries"]["train"]["target_row_id"].to_numpy(np.int64)
    tq = qr[(qr - 1) >= vis_from[A.user[qr]]]
    vf = vis_from if kind == "H" else A.user_first
    return {"policy": f"{kind}-{x}", "kind": kind, "x": x, "train_query_ids": [int(v) for v in tq],
            "user_visible_from_row_id": {str(i): int(v) for i, v in enumerate(vf)}}


def write_dataset(root, ds, tables, policies=None):
    d = os.path.join(root, "data", ds)
    os.makedirs(os.path.join(d, "evidence"), exist_ok=True)
    tables["checkins"].to_parquet(os.path.join(d, "checkins.parquet"), index=False)
    tables["venues"].to_parquet(os.path.join(d, "venues.parquet"), index=False)
    for s, q in tables["queries"].items():
        q.to_parquet(os.path.join(d, f"queries_{s}.parquet"), index=False)
        tables["candidates"][s].to_parquet(os.path.join(d, f"candidates_{s}.parquet"), index=False)
    for name, p in (policies or {"H-100": h_policy(tables, 100)}).items():
        with open(os.path.join(d, "evidence", f"{name}.json"), "w") as f:
            json.dump(p, f)
    return d
