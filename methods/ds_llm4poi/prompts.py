from __future__ import annotations

import hashlib
import importlib.util
import json
import multiprocessing as mp
import os
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from poibench.exports import Export
from methods.ds_llm4poi import apply_patches

NUM_DEMO = 5
RANDOM_SEED = 0
DATA_BANK = "the_user"
CFGS = {
    "lcs": dict(demo_selection="similarity", similarity="lcs", target_time=False),
    "random": dict(demo_selection="random", similarity="", target_time=False),
    "lcs-tt": dict(demo_selection="similarity", similarity="lcs", target_time=True),
}
WEEKDAYS = np.array(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], dtype=object)

_UP = None


def upstream() -> SimpleNamespace:
    global _UP
    if _UP is None:
        d = apply_patches.ensure_patched()
        mods = {}
        for name in ("data_utils", "prompt_utils", "demo_selection"):
            spec = importlib.util.spec_from_file_location(f"dsllm4poi_upstream_{name}",
                                                          os.path.join(d, "src", "utils", f"{name}.py"))
            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
            mods[name] = m
        with open(os.path.join(d, "PATCHED.json")) as f:
            info = json.load(f)
        _UP = SimpleNamespace(path=d, info=info, **mods)
    return _UP


def upstream_config(cfg: str) -> SimpleNamespace:
    c = CFGS[cfg]
    return SimpleNamespace(prompting=SimpleNamespace(
        type="DS-LLM4POI", data_bank=DATA_BANK, demo_selection=c["demo_selection"], num_demo=NUM_DEMO,
        random_seed=RANDOM_SEED, similarity=c["similarity"], reverse_order=False))


def code_fingerprint() -> str:
    h = hashlib.sha256()
    with open(os.path.abspath(__file__), "rb") as f:
        h.update(f.read())
    h.update(json.dumps(upstream().info["patched_sha256"], sort_keys=True).encode())
    return h.hexdigest()


def time_features(t_local) -> tuple[np.ndarray, np.ndarray]:
    t = pd.Series(pd.to_datetime(pd.Series(t_local)).to_numpy())
    st = t.dt.strftime("%I:%M %p").to_numpy(dtype=object)
    hour = t.dt.hour.to_numpy()
    minute = t.dt.minute.to_numpy()
    for h, ampm in ((0, "AM"), (12, "PM")):
        for k in np.flatnonzero(hour == h):
            st[k] = f"00:{int(minute[k]):02d} {ampm}"
    return st, WEEKDAYS[t.dt.weekday.to_numpy()]


def llm4poi_frame(df: pd.DataFrame, user_id=None) -> pd.DataFrame:
    st, dow = time_features(df["t_local"])
    out = pd.DataFrame({
        "UserId": (df["user_id"].to_numpy() if user_id is None else user_id),
        "PoiId": df["venue_id"].to_numpy(np.int64),
        "PoiCategoryName": df["category"].to_numpy(dtype=object),
        "Latitude": df["lat"].to_numpy(), "Longitude": df["lon"].to_numpy(),
        "UTCTimeOffset": df["t_local"].to_numpy(),
        "UTCTimeOffsetEpoch": pd.to_datetime(df["t_utc"]).to_numpy().astype("datetime64[s]").astype(np.int64),
        "pseudo_session_trajectory_id": (df["traj_id"].to_numpy(np.int64) if "traj_id" in df.columns else -1),
        "start_time": st, "day_of_week": dow,
    }, index=pd.Index(df["row_id"].to_numpy(np.int64)))
    return out


def lcs_len(a, b) -> int:
    if len(a) == 0 or len(b) == 0:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else (prev[j] if prev[j] >= cur[j - 1] else cur[j - 1]))
        prev = cur
    return prev[-1]


def merge_similarity(traj_ids, scores, higher_is_more_similar: bool = True) -> list:
    order = sorted(range(len(traj_ids)), key=lambda k: scores[k], reverse=higher_is_more_similar)
    return [int(traj_ids[k]) for k in order]


class PromptBuilder:

    def __init__(self, export_dir: str, split: str, cfg: str, query_ids=None, target_time: dict | None = None):
        if split not in ("val", "test"):
            raise ValueError("prompts are built for val/test queries (zero-shot method: no training queries needed)")
        if cfg not in CFGS:
            raise ValueError(f"unknown cfg {cfg}")
        if CFGS[cfg]["target_time"] and target_time is None:
            raise ValueError(f"cfg {cfg} needs the target-time ablation file (load_target_time)")
        if not CFGS[cfg]["target_time"] and target_time is not None:
            raise ValueError(f"cfg {cfg} must not receive target times")
        self.U = upstream()
        self.cfg, self.split = cfg, split
        self.ucfg = upstream_config(cfg)
        self.target_time = target_time
        ex = Export(export_dir)
        qi = ex.query_inputs(split)
        tab = qi.table
        self.qpos = pd.Series(np.arange(len(tab)), index=tab["query_id"].to_numpy(np.int64))
        idx = np.arange(len(tab)) if query_ids is None else self.qpos.loc[[int(q) for q in query_ids]].to_numpy()
        self.idx = idx
        self.query_ids = tab["query_id"].to_numpy(np.int64)
        self.user_ids = tab["user_id"].to_numpy(np.int64)
        self.hist_pos = qi.hist_pos
        self.R = llm4poi_frame(ex.rows)
        self.R_traj = self.R["pseudo_session_trajectory_id"].to_numpy(np.int64)
        self.R_rid = self.R.index.to_numpy(np.int64)
        self.train_q = np.sort(ex.queries("train")["query_id"].to_numpy(np.int64))
        ctx = qi.prefix_source
        pp = qi.prefix_pos[idx]
        keep = np.concatenate([np.arange(a, b) for a, b in pp]) if len(pp) else np.zeros(0, np.int64)
        sub = ctx.iloc[keep]
        self.C = llm4poi_frame(sub, user_id=-1)
        traj_of_q = tab["traj_id"].to_numpy(np.int64)
        owner = np.repeat(idx, pp[:, 1] - pp[:, 0])
        self.C["pseudo_session_trajectory_id"] = traj_of_q[owner]
        self.C["UserId"] = self.user_ids[owner]
        starts = np.cumsum(np.r_[0, (pp[:, 1] - pp[:, 0])])
        self.cpos = {int(i): (int(starts[k]), int(starts[k + 1])) for k, i in enumerate(idx)}

    def _bank(self, i: int) -> pd.DataFrame:
        a, b = self.hist_pos[i]
        if b <= a:
            return self.R.iloc[0:0]
        tr = self.R_traj[a:b]
        rid = self.R_rid[a:b]
        last = np.r_[tr[1:] != tr[:-1], True]
        ends_rid, ends_tr = rid[last], tr[last]
        j = np.minimum(np.searchsorted(self.train_q, ends_rid), max(len(self.train_q) - 1, 0))
        ok = (self.train_q[j] == ends_rid) if len(self.train_q) else np.zeros(len(ends_rid), bool)
        keep_tr = ends_tr[ok]
        H = self.R.iloc[a:b]
        return H[np.isin(tr, keep_tr)]

    def build(self, i: int) -> dict:
        U, P = self.U, self.U.prompt_utils
        qid = int(self.query_ids[i])
        c0, c1 = self.cpos[int(i)]
        cur = self.C.iloc[c0:c1]
        bank = self._bank(i)
        bank_tr = bank["pseudo_session_trajectory_id"].unique().tolist()
        if self.ucfg.prompting.demo_selection == "random":
            demo_ids = U.demo_selection.random_selection(cur, bank, self.ucfg)
            scores = None
        else:
            seq = cur["PoiId"].tolist()
            btr = bank["pseudo_session_trajectory_id"].to_numpy(np.int64)
            bpoi = bank["PoiId"].to_numpy(np.int64)
            scores = []
            for t in bank_tr:
                demo_ctx = bpoi[btr == t][:-1].tolist()
                scores.append(lcs_len(seq, demo_ctx))
            merged = merge_similarity(bank_tr, scores, higher_is_more_similar=True)
            key = str(int(cur["pseudo_session_trajectory_id"].iloc[0]))
            demo_ids = U.demo_selection.similarity_based_selection(cur, bank, {key: merged}, self.ucfg)
        demos = P.create_demos_prompt(bank, demo_ids)
        if self.target_time is None:
            inp = P.create_case_dict_no_target_time(cur)
            prompt = P.create_prompt_no_target_time(demos=demos, input=inp)
        else:
            tt = pd.Timestamp(self.target_time[qid])
            st, dow = time_features([tt])
            trow = pd.DataFrame({c: [None] for c in cur.columns}, index=pd.Index([qid]))
            trow["start_time"], trow["day_of_week"] = st[0], dow[0]
            trow["pseudo_session_trajectory_id"] = cur["pseudo_session_trajectory_id"].iloc[0]
            cur_t = pd.concat([cur.astype(object), trow.astype(object)])
            inp = P.create_case_dict(cur_t, is_test=True)
            prompt = P.create_prompt(demos=demos, input=inp)
        return dict(query_id=qid, prompt=prompt, n_context=int(len(cur)), n_bank=int(len(bank_tr)),
                    n_demos=int(len(demo_ids)), demo_traj_ids=[int(x) for x in demo_ids],
                    demo_lcs=([int(scores[bank_tr.index(t)]) for t in demo_ids] if scores is not None else []))


_G = {}


def _work(chunk):
    b = _G["builder"]
    return [b.build(int(i)) for i in chunk]


def build_prompts(export_dir: str, split: str, cfg: str, query_ids=None, n_jobs: int = 1,
                  target_time: dict | None = None) -> pd.DataFrame:
    b = PromptBuilder(export_dir, split, cfg, query_ids, target_time)
    idx = b.idx
    if n_jobs <= 1 or len(idx) < 200:
        recs = [b.build(int(i)) for i in idx]
    else:
        _G["builder"] = b
        chunks = [idx[k:k + 256] for k in range(0, len(idx), 256)]
        with mp.get_context("fork").Pool(n_jobs) as pool:
            recs = [r for part in pool.map(_work, chunks, chunksize=1) for r in part]
        _G.clear()
    df = pd.DataFrame(recs)
    df["prompt_sha256"] = [hashlib.sha256(p.encode()).hexdigest() for p in df["prompt"]]
    return df


def load_target_time(ds: str, split: str, root: str = ROOT) -> dict:
    p = os.path.join(root, "exports_ablation", ds, f"target_time_{split}.parquet")
    if not os.path.exists(p):
        raise FileNotFoundError(f"{p} missing: run scripts/export_target_time.py --ds {ds} (lcs-tt ablation only)")
    df = pd.read_parquet(p)
    return dict(zip(df["query_id"].astype(np.int64), df["target_t_local"]))  # canary-allow: lcs-tt ablation only
