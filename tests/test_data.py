from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd

from poibench import data as D

CK_COLS = ["row_id", "user_id", "venue_id", "category", "lat", "lon", "t_utc", "t_local", "traj_id"]
V_COLS = ["venue_id", "category", "lat", "lon", "n_train_visits"]
Q_COLS = ["query_id", "user_id", "traj_id", "target_row_id", "target_venue_id", "target_t_local", "prev_row_id",
          "in_user_history", "user_hist_len"]


def check(ds: str, n_sample: int = 3000):
    rng = np.random.default_rng(0)
    ck = D.load_checkins(ds)
    ven = D.load_venues(ds)
    assert all(c in ck.columns for c in CK_COLS), ck.columns
    assert list(ven.columns) == V_COLS, ven.columns
    assert (ck.row_id.to_numpy() == np.arange(len(ck))).all()
    assert ck[["lat", "lon"]].notna().all().all() and ven[["lat", "lon"]].notna().all().all()
    u = ck.user_id.to_numpy()
    assert (np.diff(u) >= 0).all()
    t = ck.t_utc.to_numpy("datetime64[s]").astype(np.int64)
    same_u = u[1:] == u[:-1]
    assert (np.diff(t)[same_u] >= 0).all()
    tr = ck.traj_id.to_numpy()
    newt = tr[1:] != tr[:-1]
    assert (np.diff(tr) >= 0).all()
    assert ((np.diff(t) > 86400) == newt)[same_u].all(), "trajectory boundaries != 24h gaps"
    assert ck.groupby("traj_id").size().min() >= 2, "single-check-in trajectory survived"
    assert (ck.groupby("traj_id").user_id.nunique() == 1).all()
    import json
    meta = json.load(open(os.path.join(D.data_dir(ds), "meta.json")))
    tv, tt = pd.Timestamp(meta["t_val_start_utc"]), pd.Timestamp(meta["t_test_start_utc"])
    exp = np.where(ck.t_utc < tv, "train", np.where(ck.t_utc < tt, "val", "test"))
    assert (ck.split.to_numpy() == exp).all()
    ntr = ck[ck.split == "train"].venue_id.value_counts()
    assert (ven.set_index("venue_id").n_train_visits == ntr.reindex(ven.venue_id).fillna(0).astype(int).values).all()
    assert (ck.venue_id.isin(ven.venue_id)).all()
    A = D.Arrays.from_checkins(ck)
    cat = set(ven.venue_id[ven.n_train_visits > 0])
    tr_users = set(ck.user_id[ck.split == "train"])
    for s in D.SPLITS:
        q = D.load_queries(ds, s)
        assert list(q.columns) == Q_COLS, q.columns
        assert q.query_id.is_unique and (q.query_id == q.target_row_id).all()
        r = q.target_row_id.to_numpy()
        assert (ck.split.to_numpy()[r] == s).all()
        assert (A.traj[r - 1] == A.traj[r]).all() and (q.prev_row_id == r - 1).all()
        assert (A.venue[r] == q.target_venue_id.to_numpy()).all()
        assert (A.user[r] == q.user_id.to_numpy()).all()
        if s != "train":
            assert q.target_venue_id.isin(cat).all() and q.user_id.isin(tr_users).all()
        idx = rng.choice(len(q), size=min(n_sample, len(q)), replace=False)
        for i in idx:
            row = q.iloc[i]
            tr_ = int(row.target_row_id)
            uu = A.user[tr_]
            rows_u = np.flatnonzero(A.user == uu)
            if s == "train":
                vis = rows_u[rows_u < tr_]
            else:
                pre = rows_u[A.split[rows_u] == 0]
                pref = np.arange(A.traj_start[tr_], tr_)
                vis = np.union1d(pre[pre < A.traj_start[tr_]], pref) if len(pre) else pref
                vis = np.union1d(vis, pre)
            assert row.user_hist_len == len(vis), (ds, s, tr_, row.user_hist_len, len(vis))
            assert bool(row.in_user_history) == bool((A.venue[vis] == A.venue[tr_]).any())
        c = D.load_candidates(ds, s)
        assert len(c) == len(q) and set(c.query_id) == set(q.query_id)
        L = np.stack(c.cand_venue_ids.to_numpy())
        assert L.shape[1] == 50
        assert all(len(set(x)) == 50 for x in L[:2000])
        assert np.isin(L, list(cat)).all()
    print(f"{ds}: schema / ordering / split / queries / candidates OK")

    qv = D.load_queries(ds, "val")
    last = qv.sort_values("target_row_id").groupby("traj_id").tail(1)
    sub = last.sample(n=min(1500, len(last)), random_state=0)
    r = sub.target_row_id.to_numpy()
    base = D.load_candidates(ds, "val").set_index("query_id").loc[sub.query_id]
    A2 = D.Arrays.from_checkins(ck)
    catl = np.array(sorted(cat))
    A2.venue = A2.venue.copy()
    A2.venue[r] = rng.choice(catl, size=len(r))
    c2 = D.retrieve(A2, ven, r, None, workers=1).set_index("query_id")
    assert all(list(a) == list(b) for a, b in zip(base.cand_venue_ids, c2.loc[sub.query_id].cand_venue_ids))
    print(f"{ds}: val candidates invariant to target venues OK")

    pol = D.load_policy(ds, "H-10")
    vf = D.policy_vis_from(pol, len(A.user_first))
    rows = np.arange(len(A.user))
    invisible = A.is_train & (rows < vf[A.user])
    keep_prefix = np.zeros(len(rows), dtype=bool)
    for tr_ in r:
        keep_prefix[A.traj_start[tr_]:tr_] = True
    mut = invisible & ~keep_prefix
    A3 = D.Arrays.from_checkins(ck)
    A3.venue = A3.venue.copy()
    A3.venue[mut] = rng.choice(catl, size=int(mut.sum()))
    b10 = D.load_candidates(ds, "val", "H-10").set_index("query_id").loc[sub.query_id]
    c3 = D.retrieve(A3, ven, r, vf, workers=1).set_index("query_id")
    assert all(list(a) == list(b) for a, b in zip(b10.cand_venue_ids, c3.loc[sub.query_id].cand_venue_ids))
    print(f"{ds}: H-10 val candidates invariant to {int(mut.sum())} invisible rows OK")

    qtr = D.load_queries(ds, "train")
    for x in D.POLICY_X:
        h, s_ = D.load_policy(ds, f"H-{x}"), D.load_policy(ds, f"S-{x}")
        assert h["train_query_ids"] == s_["train_query_ids"]
        ids = np.array(h["train_query_ids"])
        assert np.isin(ids, qtr.query_id.to_numpy()).all()
        vf = D.policy_vis_from(h, len(A.user_first))
        assert (ids - 1 >= vf[A.user[ids]]).all()
        keep = A.user_first + A.user_ntrain - vf
        exp_keep = np.ceil(A.user_ntrain * x / 100.0)
        assert (keep == exp_keep).all()
        if x == 100:
            assert len(ids) == len(qtr)
    print(f"{ds}: policies OK")


def check_variant(ds: str):
    import json
    base = D.base_dataset(ds)
    m, bm = (json.load(open(os.path.join(D.data_dir(x), "meta.json"))) for x in (ds, base))
    assert (m["t_val_start_utc"], m["t_test_start_utc"]) == (bm["t_val_start_utc"], bm["t_test_start_utc"])
    assert m["variant_of"] == base
    raw, _ = D.load_raw(ds)
    tv = pd.Timestamp(m["t_val_start_utc"])
    pre = raw[raw.t_utc < tv]
    vc = pre.venue_raw.value_counts()
    idv = pd.read_parquet(os.path.join(D.data_dir(ds), "id_map_venues.parquet"))
    assert (idv.venue_raw.map(vc).fillna(0) >= D.CFG["min_poi_checkins"]).all()
    pre = pre[pre.venue_raw.isin(vc.index[vc >= D.CFG["min_poi_checkins"]])]
    uc = pre.user_raw.value_counts()
    idu = pd.read_parquet(os.path.join(D.data_dir(ds), "id_map_users.parquet"))
    assert (idu.user_raw.map(uc).fillna(0) >= D.CFG["min_user_checkins"]).all()
    assert m["future_filter"]["catalog_venues_below_min_before_cutoff"] == 0
    print(f"{ds}: trainfilter variant OK")


def check_su(ds: str):
    import hashlib
    import json
    su = D.load_policy(ds, "S-u")
    s100 = D.load_policy(ds, "S-100")
    assert su["kind"] == "S" and su["user_visible_from_row_id"] == s100["user_visible_from_row_id"]
    assert su["candidates"] == s100["candidates"]
    qtr = D.load_queries(ds, "train")
    users = sorted(set(int(u) for u in qtr.user_id))
    perm = [int(u) for u in np.random.default_rng(0).permutation(np.array(users, dtype=np.int64))]
    ntraj = qtr.groupby("user_id").traj_id.nunique().to_dict()
    tot, k = 0, 0
    while tot < D.SU["target_sft_trajectories"] and k < len(perm):
        tot += ntraj[perm[k]]
        k += 1
    chosen = set(perm[:k])
    assert su["subsample"]["k"] == k and set(su["subsample"]["users"]) == chosen, (k, su["subsample"]["k"])
    assert su["subsample"]["n_sft_trajectories"] == tot >= D.SU["target_sft_trajectories"]
    exp_ids = sorted(int(x) for x in qtr.loc[qtr.user_id.isin(chosen), "query_id"])
    assert su["train_query_ids"] == exp_ids
    print(f"{ds}: S-u policy OK (k={k}, {tot} SFT trajectories, {len(exp_ids)} train queries)")
    ed = os.path.join(D.ROOT, "exports", ds)
    if os.path.exists(os.path.join(ed, "S-u", "MANIFEST.json")):
        def sha(pol, fn):
            with open(os.path.join(ed, pol, fn), "rb") as f:
                return hashlib.sha256(f.read()).hexdigest()
        for fn in ["visible_rows.parquet", "venues_public.parquet"] + [
                f"{t}_{sp}.parquet" for t in ("query_inputs", "context", "candidates") for sp in ("val", "test")]:
            assert sha("S-u", fn) == sha("S-100", fn), fn
        qi = pd.read_parquet(os.path.join(ed, "S-u", "query_inputs_train.parquet"))
        assert sorted(qi.query_id.astype(int).tolist()) == exp_ids
        q100 = pd.read_parquet(os.path.join(ed, "S-100", "query_inputs_train.parquet")).set_index("query_id")
        assert (qi.set_index("query_id").sort_index() == q100.loc[sorted(exp_ids)]).all().all()
        lab = pd.read_parquet(os.path.join(ed, "S-u", "train_labels.parquet"))
        assert sorted(lab.query_id.astype(int).tolist()) == exp_ids
        pp = json.load(open(os.path.join(ed, "S-u", "policy_public.json")))
        assert pp["policy"] == "S-u" and "user_visible_from_row_id" not in pp
        print(f"{ds}: S-u export = S-100 export except the train-query set OK")


import unittest


class TestBuiltData(unittest.TestCase):

    def _run(self, ds):
        if not os.path.exists(os.path.join(D.data_dir(ds), "checkins.parquet")):
            self.skipTest(f"data/{ds} not built")
        check(ds)

    def test_nyc(self):
        self._run("nyc")

    def test_tky(self):
        self._run("tky")

    def test_istanbul(self):
        self._run("istanbul")

    def test_ca(self):
        self._run("ca")

    def test_sin(self):
        self._run("sin")

    def test_sao_paulo(self):
        self._run("sao_paulo")

    def test_gowalla(self):
        self._run("gowalla")

    def test_foursquare_global(self):
        self._run("foursquare_global")

    def test_su_policies(self):
        for ds in D.SU_DATASETS:
            if os.path.exists(os.path.join(D.data_dir(ds), "evidence", "S-u.json")):
                check_su(ds)

    def test_nyc_trainfilter(self):
        self._run("nyc_trainfilter")
        check_variant("nyc_trainfilter")

    def test_tky_trainfilter(self):
        self._run("tky_trainfilter")
        check_variant("tky_trainfilter")


if __name__ == "__main__":
    for ds in (sys.argv[1:] or D.DATASETS):
        check(ds)
        if ds in D.SU_DATASETS and os.path.exists(os.path.join(D.data_dir(ds), "evidence", "S-u.json")):
            check_su(ds)
        if ds in D.VARIANTS:
            check_variant(ds)
