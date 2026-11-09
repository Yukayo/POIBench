from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import ROOT
from .data import POLICY_TEXT, SPLITS, Arrays, evidence_slices, policy_vis_from
from .hashing import sha256_file

EXPORT_VERSION = "e1"
POLICIES = ("H-10", "H-70", "H-100", "S-10", "S-70", "S-100")
EXTRA_POLICIES = ("S-u",)
ROW_COLS = ["row_id", "user_id", "venue_id", "category", "lat", "lon", "t_utc", "t_local", "traj_id"]
CTX_COLS = ["query_id", "pos", "row_id", "venue_id", "category", "lat", "lon", "t_utc", "t_local"]
QIN_COLS = ["query_id", "user_id", "traj_id", "prev_row_id", "prev_venue_id", "prev_t_utc", "prev_t_local",
            "hist_lo", "hist_hi", "prefix_lo", "prefix_hi", "n_visible_own"]
VEN_COLS = ["venue_id", "category", "lat", "lon", "in_catalog", "n_visible_train_visits"]
LABEL_COLS = ["query_id", "label_row_id", "label_venue_id", "label_t_utc", "label_t_local"]
FORBIDDEN_COLS = ("split", "target_row_id", "target_venue_id", "target_t_local", "in_user_history",
                  "user_hist_len", "n_train_visits", "user_visible_from_row_id")
SOURCE_FILES = ["checkins.parquet", "venues.parquet"] + [f"queries_{s}.parquet" for s in SPLITS]

RULES = ("Adapters read only this directory (poibench.exports.Export), never data/<ds>/. Population statistics "
         "come only from visible_rows. A val/test query may use visible_rows and its own context rows "
         "(query_id == q) only. train_labels are the only targets. Model selection on val only through "
         "poibench.evaluate.val_feedback. Ids are keys, not features. Checked by poibench/canary.py.")
PUBLIC_POLICY_TEXT = {
    "H": "History budget: only the latest x% (ceil) of each user's pre-cutoff check-ins are visible to ANY "
         "component (visible_rows.parquet already applies this). Supervised examples: the train queries listed in "
         "query_inputs_train.parquet. A val/test query's own trajectory prefix (context file) is always visible.",
    "S": "Sample budget: all pre-cutoff check-ins are visible (visible_rows.parquet), but only the train queries "
         "listed in query_inputs_train.parquet (the H-x train-query set) may be used for gradient updates or as "
         "supervised examples / few-shot demonstrations.",
}


def export_dir(ds: str, policy: str, root: str = ROOT) -> str:
    return os.path.join(root, "exports", ds, policy)


def _ranges_concat(lo: np.ndarray, hi: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = (hi - lo).astype(np.int64)
    tot = int(n.sum())
    owner = np.repeat(np.arange(len(lo)), n)
    start = np.repeat(np.cumsum(n) - n, n)
    pos = np.arange(tot, dtype=np.int64) - start
    return np.repeat(lo, n) + pos, owner, pos


def build_export(ck: pd.DataFrame, venues: pd.DataFrame, queries: dict, policy: dict, cands: dict,
                 splits=SPLITS) -> dict:
    ck = ck.sort_values("row_id").reset_index(drop=True)
    A = Arrays.from_checkins(ck)
    n_users = len(A.user_first)
    rows_all = np.arange(len(A.user), dtype=np.int64)
    if policy["kind"] == "H":
        vf = policy_vis_from(policy, n_users)
        vis = A.is_train & (rows_all >= vf[A.user])
    elif policy["kind"] == "S":
        vf = None
        vis = A.is_train.copy()
    else:
        raise ValueError(policy["kind"])
    out = {}
    vrows = ck.loc[vis, ROW_COLS].reset_index(drop=True)
    out["visible_rows"] = vrows

    V = len(venues)
    vt = venues.sort_values("venue_id").reset_index(drop=True)
    if not (vt["venue_id"].to_numpy() == np.arange(V)).all():
        raise ValueError("venue_id must be dense 0..V-1")
    out["venues_public"] = pd.DataFrame({
        "venue_id": vt["venue_id"].to_numpy(np.int32), "category": vt["category"].to_numpy(),
        "lat": vt["lat"].to_numpy(), "lon": vt["lon"].to_numpy(),
        "in_catalog": vt["n_train_visits"].to_numpy() > 0,
        "n_visible_train_visits": np.bincount(A.venue[vis], minlength=V).astype(np.int64)})

    tq = set(int(x) for x in policy["train_query_ids"])
    t_utc = ck["t_utc"].to_numpy()
    t_loc = ck["t_local"].to_numpy()
    for s in splits:
        q = queries[s]
        if s == "train":
            q = q[q["query_id"].isin(tq)]
        q = q.sort_values("query_id").reset_index(drop=True)
        r = q["target_row_id"].to_numpy(np.int64)
        if len(r) and not (A.split[r] == SPLITS.index(s)).all():
            raise ValueError(f"queries_{s} has targets outside split {s}")
        sl = evidence_slices(r, A, vf)
        prev = r - 1
        if ((sl["prefix_hi"] - sl["prefix_lo"]) < 1).any():
            raise ValueError("previous check-in not visible for some query")
        out[f"query_inputs_{s}"] = pd.DataFrame({
            "query_id": q["query_id"].to_numpy(np.int64), "user_id": A.user[r].astype(np.int32),
            "traj_id": A.traj[r].astype(np.int32), "prev_row_id": prev,
            "prev_venue_id": A.venue[prev].astype(np.int32), "prev_t_utc": t_utc[prev], "prev_t_local": t_loc[prev],
            "hist_lo": sl["hist_lo"].astype(np.int64), "hist_hi": sl["hist_hi"].astype(np.int64),
            "prefix_lo": sl["prefix_lo"].astype(np.int64), "prefix_hi": sl["prefix_hi"].astype(np.int64),
            "n_visible_own": ((sl["hist_hi"] - sl["hist_lo"]) + (sl["prefix_hi"] - sl["prefix_lo"])).astype(np.int64)})
        if s == "train":
            out["train_labels"] = pd.DataFrame({
                "query_id": q["query_id"].to_numpy(np.int64), "label_row_id": r,
                "label_venue_id": A.venue[r].astype(np.int32), "label_t_utc": t_utc[r], "label_t_local": t_loc[r]})
        else:
            idx, owner, pos = _ranges_concat(sl["prefix_lo"], sl["prefix_hi"])
            c = ck.iloc[idx]
            out[f"context_{s}"] = pd.DataFrame({
                "query_id": q["query_id"].to_numpy(np.int64)[owner], "pos": pos.astype(np.int32),
                "row_id": idx, "venue_id": c["venue_id"].to_numpy(np.int32), "category": c["category"].to_numpy(),
                "lat": c["lat"].to_numpy(), "lon": c["lon"].to_numpy(), "t_utc": c["t_utc"].to_numpy(),
                "t_local": c["t_local"].to_numpy()})
        cs = cands[s]
        cs = cs[cs["query_id"].isin(set(q["query_id"]))].sort_values("query_id").reset_index(drop=True)
        if len(cs) != len(q) or not (cs["query_id"].to_numpy() == q["query_id"].to_numpy()).all():
            raise ValueError(f"candidate file does not cover the {s} queries")
        out[f"candidates_{s}"] = cs[["query_id", "cand_venue_ids"]]
    out["policy_public"] = dict(policy=policy["policy"], kind=policy["kind"], x=policy["x"],
                                cutoff_t_utc=policy.get("cutoff_t_utc"),
                                n_train_queries=int(len(out.get("query_inputs_train", ()))),
                                description=policy.get("public_description", PUBLIC_POLICY_TEXT[policy["kind"]]))
    check_export(out, A, vis)
    return out


def check_export(out: dict, A: Arrays, vis: np.ndarray) -> None:
    for name, df in out.items():
        if isinstance(df, pd.DataFrame):
            bad = [c for c in df.columns if c in FORBIDDEN_COLS]
            if bad:
                raise AssertionError(f"{name} has forbidden columns {bad}")
    vr = out["visible_rows"]["row_id"].to_numpy(np.int64)
    if not (vis[vr].all() and A.is_train[vr].all() and len(vr) == int(vis.sum())):
        raise AssertionError("visible_rows is not exactly the policy-visible train rows")
    ven = out["venues_public"]
    if not (ven["n_visible_train_visits"].to_numpy() == np.bincount(A.venue[vr], minlength=len(ven))).all():
        raise AssertionError("venue counts not computed from visible rows")
    vset = np.zeros(len(A.user), dtype=bool)
    vset[vr] = True
    for s in SPLITS:
        qi = out.get(f"query_inputs_{s}")
        if qi is None:
            continue
        qid = qi["query_id"].to_numpy(np.int64)
        for lo, hi in (("hist_lo", "hist_hi"), ) + ((("prefix_lo", "prefix_hi"),) if s == "train" else ()):
            idx, _, _ = _ranges_concat(qi[lo].to_numpy(), qi[hi].to_numpy())
            if not vset[idx].all():
                raise AssertionError(f"{s}: {lo}..{hi} reaches rows outside visible_rows")
        if (qi["prefix_hi"].to_numpy() != qid).any() or (qi["prev_row_id"].to_numpy() != qid - 1).any():
            raise AssertionError(f"{s}: evidence must end right before the target")
        if s != "train":
            ctx = out[f"context_{s}"]
            cq = ctx["query_id"].to_numpy(np.int64)
            cr = ctx["row_id"].to_numpy(np.int64)
            if not ((cr < cq).all() and (A.traj[cr] == A.traj[cq]).all()):
                raise AssertionError(f"context_{s} contains a row at/after its query's target or another trajectory")
            if len(ctx) != int((qi["prefix_hi"] - qi["prefix_lo"]).sum()):
                raise AssertionError(f"context_{s} size != sum of prefix lengths")


def _write_parquet(df: pd.DataFrame, path: str) -> None:
    df.to_parquet(path + ".tmp", index=False)
    os.replace(path + ".tmp", path)


def source_hashes(ds: str, policy: str, root: str = ROOT) -> dict:
    d = os.path.join(root, "data", ds)
    with open(os.path.join(d, "evidence", f"{policy}.json")) as f:
        pol = json.load(f)
    files = list(SOURCE_FILES) + [f"evidence/{policy}.json"] + sorted(set((pol.get("candidates") or {}).values()))
    return {rel: sha256_file(os.path.join(d, rel)) for rel in files}


def load_sources(ds: str, policy: str, root: str = ROOT) -> tuple:
    d = os.path.join(root, "data", ds)
    ck = pd.read_parquet(os.path.join(d, "checkins.parquet"))
    venues = pd.read_parquet(os.path.join(d, "venues.parquet"))
    queries = {s: pd.read_parquet(os.path.join(d, f"queries_{s}.parquet")) for s in SPLITS}
    with open(os.path.join(d, "evidence", f"{policy}.json")) as f:
        pol = json.load(f)
    cands = {s: pd.read_parquet(os.path.join(d, (pol.get("candidates") or {}).get(s, f"candidates_{s}.parquet")))
             for s in SPLITS}
    with open(os.path.join(d, "meta.json")) as f:
        meta = json.load(f)
    return ck, venues, queries, pol, cands, meta


def write_export(out: dict, path: str, manifest_extra: dict | None = None) -> dict:
    os.makedirs(path, exist_ok=True)
    mf = os.path.join(path, "MANIFEST.json")
    if os.path.exists(mf):
        os.remove(mf)
    files, counts = {}, {}
    for name, obj in out.items():
        if isinstance(obj, pd.DataFrame):
            fn = f"{name}.parquet"
            _write_parquet(obj, os.path.join(path, fn))
            counts[name] = int(len(obj))
        else:
            fn = f"{name}.json"
            with open(os.path.join(path, fn + ".tmp"), "w") as f:
                json.dump(obj, f, indent=1, default=str)
            os.replace(os.path.join(path, fn + ".tmp"), os.path.join(path, fn))
        files[fn] = sha256_file(os.path.join(path, fn))
    man = dict(export_version=EXPORT_VERSION, files=files, counts=counts,
               rules=RULES, **(manifest_extra or {}))
    with open(mf + ".tmp", "w") as f:
        json.dump(man, f, indent=1, default=str)
    os.replace(mf + ".tmp", mf)
    return man


def export_is_current(ds: str, policy: str, root: str = ROOT, out_root: str | None = None,
                      src: dict | None = None) -> bool:
    p = os.path.join(export_dir(ds, policy, out_root or root), "MANIFEST.json")
    if not os.path.exists(p):
        return False
    with open(p) as f:
        man = json.load(f)
    return (man.get("export_version") == EXPORT_VERSION
            and man.get("source_sha256") == (src or source_hashes(ds, policy, root)))


def export_inputs(ds: str, policy: str, root: str = ROOT, out_root: str | None = None, force: bool = False) -> dict:
    src = source_hashes(ds, policy, root)
    path = export_dir(ds, policy, out_root or root)
    if not force and export_is_current(ds, policy, root, out_root, src):
        with open(os.path.join(path, "MANIFEST.json")) as f:
            return dict(json.load(f), skipped=True)
    ck, venues, queries, pol, cands, meta = load_sources(ds, policy, root)
    out = build_export(ck, venues, queries, pol, cands)
    return write_export(out, path, dict(ds=ds, policy=policy, t_val_start_utc=meta["t_val_start_utc"],
                                        source_sha256=src, source_dir=f"data/{ds}"))


@dataclass
class QueryInputs:
    split: str
    table: pd.DataFrame
    hist_pos: np.ndarray
    prefix_pos: np.ndarray
    prefix_source: pd.DataFrame
    rows: pd.DataFrame

    def own(self, i: int, col: str = "venue_id") -> np.ndarray:
        a, b = self.hist_pos[i]
        c, d = self.prefix_pos[i]
        h = self.rows[col].to_numpy()[a:b]
        p = self.prefix_source[col].to_numpy()[c:d]
        return np.concatenate([h, p])


class Export:

    def __init__(self, path: str):
        self.path = path
        with open(os.path.join(path, "MANIFEST.json")) as f:
            self.manifest = json.load(f)
        with open(os.path.join(path, "policy_public.json")) as f:
            self.policy = json.load(f)
        self._cache = {}

    def _pq(self, name: str) -> pd.DataFrame:
        if name not in self._cache:
            self._cache[name] = pd.read_parquet(os.path.join(self.path, f"{name}.parquet"))
        return self._cache[name]

    @property
    def rows(self) -> pd.DataFrame:
        return self._pq("visible_rows")

    @property
    def venues(self) -> pd.DataFrame:
        return self._pq("venues_public")

    def queries(self, split: str) -> pd.DataFrame:
        return self._pq(f"query_inputs_{split}")

    def context(self, split: str) -> pd.DataFrame:
        if split == "train":
            raise ValueError("train prefixes are in visible_rows (use query_inputs('train'))")
        return self._pq(f"context_{split}")

    def candidates(self, split: str) -> pd.DataFrame:
        return self._pq(f"candidates_{split}")

    def train_labels(self) -> pd.DataFrame:
        return self._pq("train_labels")

    def query_inputs(self, split: str) -> QueryInputs:
        q = self.queries(split)
        rid = self.rows["row_id"].to_numpy(np.int64)
        hist = np.stack([np.searchsorted(rid, q["hist_lo"].to_numpy()),
                         np.searchsorted(rid, q["hist_hi"].to_numpy())], axis=1)
        if split == "train":
            src = self.rows
            pre = np.stack([np.searchsorted(rid, q["prefix_lo"].to_numpy()),
                            np.searchsorted(rid, q["prefix_hi"].to_numpy())], axis=1)
        else:
            src = self.context(split)
            cq = src["query_id"].to_numpy(np.int64)
            qq = q["query_id"].to_numpy(np.int64)
            pre = np.stack([np.searchsorted(cq, qq, side="left"), np.searchsorted(cq, qq, side="right")], axis=1)
        return QueryInputs(split=split, table=q, hist_pos=hist, prefix_pos=pre, prefix_source=src, rows=self.rows)
