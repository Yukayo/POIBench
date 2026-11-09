#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

import numpy as np
import pandas as pd

SCALE = C.ROOT / "work" / "efficiency_scale"
LINKS = ("poibench", "methods", "scripts", "third_party", "envs", "PROTOCOL.md", "data_raw")
FRACS = (20, 40, 60, 80)
DS = "tky"


def sample_users(n_users: int, frac: int) -> np.ndarray:
    perm = np.random.default_rng(0).permutation(n_users)
    return np.sort(perm[:int(round(frac / 100 * n_users))])


def scale_root(frac: int) -> Path:
    return SCALE / f"u{frac}"


def make_root(frac: int) -> Path:
    sr = scale_root(frac)
    sr.mkdir(parents=True, exist_ok=True)
    for name in LINKS:
        p = sr / name
        if not p.exists() and not p.is_symlink():
            p.symlink_to(C.ROOT / name)
    for d in ("data", "exports", "preds", "work", "logs"):
        (sr / d).mkdir(exist_ok=True)
    return sr


def graph_stats(ck: pd.DataFrame) -> dict:
    t = ck[ck["split"] == "train"].sort_values("row_id")
    v, tr = t["venue_id"].to_numpy(), t["traj_id"].to_numpy()
    same = tr[1:] == tr[:-1]
    e = np.unique(np.c_[v[:-1][same], v[1:][same]], axis=0)
    return dict(graph_nodes=int(np.unique(v).size), graph_edges=int(len(e)), n_transitions=int(same.sum()))


def build(frac: int, workers: int) -> dict:
    from poibench import data as D
    from poibench.exports import export_inputs
    src = C.ROOT / "data" / DS
    ck0 = pd.read_parquet(src / "checkins.parquet").sort_values("row_id").reset_index(drop=True)
    ven0 = pd.read_parquet(src / "venues.parquet").sort_values("venue_id").reset_index(drop=True)
    meta0 = json.load(open(src / "meta.json"))
    n_users0 = int(ck0["user_id"].max()) + 1
    users = sample_users(n_users0, frac)
    sr = make_root(frac)
    out = sr / "data" / DS
    (out / "evidence").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ck = ck0[ck0["user_id"].isin(users)].copy()
    umap = pd.Series(np.arange(len(users)), index=users)
    vids = np.unique(ck["venue_id"].to_numpy())
    vmap = pd.Series(np.arange(len(vids)), index=vids)
    tids = np.unique(ck["traj_id"].to_numpy())
    tmap = pd.Series(np.arange(len(tids)), index=tids)
    ck["orig_row_id"] = ck["row_id"]
    ck["user_id"] = umap.loc[ck["user_id"].to_numpy()].to_numpy().astype(np.int32)
    ck["venue_id"] = vmap.loc[ck["venue_id"].to_numpy()].to_numpy().astype(np.int32)
    ck["traj_id"] = tmap.loc[ck["traj_id"].to_numpy()].to_numpy().astype(np.int32)
    ck = ck.sort_values(["user_id", "orig_row_id"]).reset_index(drop=True)
    ck["row_id"] = np.arange(len(ck), dtype=np.int64)
    assert (np.diff(ck["orig_row_id"].to_numpy())[np.diff(ck["user_id"].to_numpy()) == 0] > 0).all()
    orig_rows = ck.pop("orig_row_id").to_numpy()
    ven = ven0.iloc[vids].copy()
    ven["venue_id"] = np.arange(len(vids), dtype=np.int32)
    ntr = ck.loc[ck["split"] == "train", "venue_id"].value_counts()
    ven["n_train_visits"] = ven["venue_id"].map(ntr).fillna(0).astype(np.int64)
    ven = ven.reset_index(drop=True)
    ck = ck[list(ck0.columns)]
    A = D.Arrays.from_checkins(ck)
    queries, qstats = D.build_queries(ck, ven, A)
    ck.to_parquet(out / "checkins.parquet", index=False)
    ven.to_parquet(out / "venues.parquet", index=False)
    pd.DataFrame({"row_id": np.arange(len(ck)), "orig_row_id": orig_rows}).to_parquet(out / "orig_rows.parquet", index=False)
    pd.DataFrame({"user_id": np.arange(len(users)), "orig_user_id": users}).to_parquet(out / "orig_users.parquet", index=False)
    pd.DataFrame({"venue_id": np.arange(len(vids)), "orig_venue_id": vids}).to_parquet(out / "orig_venues.parquet", index=False)
    for s, q in queries.items():
        q.to_parquet(out / f"queries_{s}.parquet", index=False)
    pols = D.make_policies(A, queries["train"], meta0["t_val_start_utc"])
    hp = pols["H-100"]
    tr0 = time.perf_counter()
    st = D.prepare_retriever(A, ven, None)
    ret_build_s = time.perf_counter() - tr0
    cand_paths, n_q = {}, {}
    for s in D.SPLITS:
        q = queries[s]
        if s == "train":
            q = q[q["query_id"].isin(set(hp["train_query_ids"]))]
        c = D.retrieve(A, ven, q["target_row_id"].to_numpy(), None, workers=workers, st=st)
        assert len(c) == len(q) and (c["cand_venue_ids"].map(len) == D.RETRIEVER["out_len"]).all()
        c.to_parquet(out / f"candidates_{s}.parquet", index=False)
        cand_paths[s] = f"candidates_{s}.parquet"
        n_q[s] = len(q)
    for name in ("H-100", "S-100"):
        p = dict(pols[name])
        p["description"] = D.POLICY_TEXT[p["kind"]]
        p["candidates"] = {s: f"candidates_{s}.parquet" for s in D.SPLITS}
        json.dump(p, open(out / "evidence" / f"{name}.json", "w"))
    counts = {}
    for s in D.SPLITS:
        sub = ck[ck["split"] == s]
        counts[s] = dict(checkins=int(len(sub)), users=int(sub.user_id.nunique()), venues=int(sub.venue_id.nunique()),
                         queries=int(len(queries[s])))
    meta = dict(dataset=DS, scale_fraction=frac, scale_seed=0, scale_rule="nested user sample, all check-ins kept",
                variant_of=DS, t_val_start_utc=meta0["t_val_start_utc"], t_test_start_utc=meta0["t_test_start_utc"],
                config=meta0.get("config"), retriever=meta0.get("retriever"), counts=counts,
                totals=dict(checkins=int(len(ck)), users=int(len(users)), venues=int(len(ven)),
                            catalog_venues=int((ven.n_train_visits > 0).sum()), trajectories=int(len(tids)),
                            categories=int(ven.category.nunique())),
                policies={k: dict(n_train_queries=v["n_train_queries"], frac_train_queries=v["frac_train_queries"],
                                  frac_visible_precutoff_checkins=v["frac_visible_precutoff_checkins"])
                          for k, v in pols.items() if k in ("H-100", "S-100")},
                preprocess_stats=qstats, built_utc=pd.Timestamp.now(tz="UTC").isoformat())
    json.dump(meta, open(out / "meta.json", "w"), indent=1, default=str)
    man = export_inputs(DS, "H-100", root=str(sr))
    row = dict(frac=frac, root=str(sr.relative_to(C.ROOT)), n_users=len(users), n_checkins=len(ck),
               n_train_checkins=counts["train"]["checkins"], n_venues=len(ven),
               n_catalog=meta["totals"]["catalog_venues"], n_traj=len(tids),
               n_train_queries=n_q["train"], n_val_queries=n_q["val"], n_test_queries=n_q["test"],
               retriever_build_once_s=ret_build_s, build_s=time.time() - t0, **graph_stats(ck))
    print(json.dumps(row), flush=True)
    return row


def full_row() -> dict:
    src = C.ROOT / "data" / DS
    ck = pd.read_parquet(src / "checkins.parquet")
    m = json.load(open(src / "meta.json"))
    c, t = m["counts"], m["totals"]
    qi = pd.read_parquet(C.ROOT / "exports" / DS / "H-100" / "query_inputs_train.parquet", columns=["query_id"])
    return dict(frac=100, root=".", n_users=t["users"], n_checkins=t["checkins"], n_train_checkins=c["train"]["checkins"],
                n_venues=t["venues"], n_catalog=t["catalog_venues"], n_traj=t["trajectories"],
                n_train_queries=len(qi), n_val_queries=c["val"]["queries"], n_test_queries=c["test"]["queries"],
                **graph_stats(ck))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fracs", type=int, nargs="+", default=list(FRACS))
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    rows = [build(f, a.workers) for f in a.fracs] + [full_row()]
    C.write_csv("e7_scale_sizes.csv", rows)


if __name__ == "__main__":
    main()
