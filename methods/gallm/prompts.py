from __future__ import annotations

import hashlib
import os
import types

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
UPSTREAM_DIR = os.path.join(ROOT, "third_party", "GA-LLM")
UPSTREAM_COMMIT = "360f5c3e4d70b5191d45f8ef389a3fdb05bbcc9f"
UPSTREAM = {
    "llm": ("data-preprocess/process_llm_gps.py", "71db05c6d04495945ebf320610e4ea0f335ca518c1b8fd5d9897794933f330dd"),
    "stage1": ("data-preprocess/process_projector_gps_stage1.py",
               "1f34cde9f604015cd5754ee6a2f40025648420a9b62539b8df9fa31649e132c9"),
}
TIME_FMT = "%Y-%m-%d %H:%M:%S"
SENT = "@@TARGET_TIME@@"
CFGS = {"default": dict(target_time=False, method="ga-llm"),
        "tt": dict(target_time=True, method="ga-llm-tt")}
_RANGE_SRC = "{'NYC': 4981, 'TKY': 7833, 'CA': 9690}[args.dataset_name.upper()]"
_FN = {}


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def upstream_fn(kind: str):
    if kind in _FN:
        return _FN[kind]
    rel, sha = UPSTREAM[kind]
    p = os.path.join(UPSTREAM_DIR, rel)
    got = _sha256(p)
    if got != sha:
        raise RuntimeError(f"{p} sha256 {got} != pinned {sha} (GA-LLM {UPSTREAM_COMMIT})")
    src = open(p).read()
    edits = [("from tqdm import tqdm", "tqdm = lambda x, **k: x"), (_RANGE_SRC, "args.n_poi")]
    if kind == "llm":
        edits.append(('            question_parts = ["<question>: "]\n',
                      '            num_similar_traj = num_early_traj = 0; num_current_traj = len(user_trajectory_data) - 1\n'
                      '            question_parts = ["<question>: "]\n'))
    for a, b in edits:
        if src.count(a) != 1:
            raise RuntimeError(f"{rel}: edit target found {src.count(a)} times: {a!r}")
        src = src.replace(a, b)
    ns = {"__name__": f"gallm_upstream_{kind}"}
    exec(compile(src, p, "exec"), ns)
    _FN[kind] = ns["generate_qa_pairs"]
    return _FN[kind]


def category_ids(venues: pd.DataFrame) -> dict:
    cats = sorted(set(str(c) for c in venues["category"].astype(str)))
    return {c: i for i, c in enumerate(cats)}


def _frame(df: pd.DataFrame, user: int, traj: int, cat_id: dict) -> pd.DataFrame:
    t = pd.to_datetime(df["t_local"])
    return pd.DataFrame({
        "UserId": int(user), "pseudo_session_trajectory_id": int(traj),
        "UTCTimeOffsetEpoch": (pd.to_datetime(df["t_utc"]).astype("int64") // 10**9).to_numpy()
        if len(df) else np.array([], dtype=np.int64),
        "UTCTimeOffset": t.dt.strftime(TIME_FMT).to_numpy(dtype=object) if len(df) else np.array([], dtype=object),
        "PoiId": df["venue_id"].astype(np.int64).to_numpy(),
        "Latitude": df["lat"].astype(float).to_numpy(), "Longitude": df["lon"].astype(float).to_numpy(),
        "PoiCategoryName": df["category"].astype(str).to_numpy(dtype=object),
        "PoiCategoryId": np.array([cat_id[str(c)] for c in df["category"].astype(str)], dtype=np.int64),
    })


def _one(kind: str, cur: pd.DataFrame, hist: pd.DataFrame, user: int, traj: int, target: dict, n_poi: int,
         cat_id: dict) -> tuple[str, str]:
    main = _frame(cur, user, traj, cat_id)
    tgt = pd.DataFrame({"UserId": [int(user)], "pseudo_session_trajectory_id": [int(traj)],
                        "UTCTimeOffsetEpoch": [int(target["t_epoch"])], "UTCTimeOffset": [target["t_local_str"]],
                        "PoiId": [int(target["venue_id"])], "Latitude": [float(target["lat"])],
                        "Longitude": [float(target["lon"])], "PoiCategoryName": ["nan"], "PoiCategoryId": [-1]})
    main = pd.concat([main, tgt], ignore_index=True)
    h = _frame(hist, user, -1, cat_id)
    args = types.SimpleNamespace(dataset_name="poibench", n_poi=int(n_poi))
    qa, _ = upstream_fn(kind)(main, kqt=None, historical_data=h, args=args, use_sim=False)
    if len(qa) != 1:
        raise AssertionError(f"upstream returned {len(qa)} samples for one trajectory")
    return qa[0]


def _strip_tt(q: str, a: str, user: int) -> tuple[str, str]:
    clause = f"At {SENT}, "
    if q.count(clause) != 1:
        raise AssertionError(f"target-time clause found {q.count(clause)} times in the question")
    q = q.replace(clause, "")
    if a.count(clause) == 1:
        a = a.replace(clause, "")
    if SENT in q or SENT in a:
        raise AssertionError("target-time sentinel left in the text")
    return q, a


def _target_epoch_after(cur: pd.DataFrame) -> int:
    return int(pd.to_datetime(cur["t_utc"]).astype("int64").max() // 10**9) + 1 if len(cur) else 0


def train_samples(ex, kind: str, cfg: str = "default", query_ids=None) -> pd.DataFrame:
    qi = ex.query_inputs("train")
    q = qi.table
    lab = ex.train_labels().set_index("query_id")
    keep = np.ones(len(q), dtype=bool)
    if query_ids is not None:
        keep = q["query_id"].isin(set(int(x) for x in query_ids)).to_numpy()
    last = q.loc[keep].groupby("traj_id", sort=True)["query_id"].max()
    pos = np.flatnonzero(q["query_id"].isin(set(last.to_numpy().tolist())).to_numpy())
    rows = ex.rows
    V = len(ex.venues)
    cat_id = category_ids(ex.venues)
    ven = ex.venues.set_index("venue_id")
    out = []
    for i in pos:
        a, b = qi.hist_pos[i]
        c, d = qi.prefix_pos[i]
        if d <= c:
            raise AssertionError("train query without a trajectory prefix")
        qid = int(q["query_id"].iat[i])
        u, tr = int(q["user_id"].iat[i]), int(q["traj_id"].iat[i])
        cur, hist = rows.iloc[c:d], rows.iloc[a:b]
        vid = int(lab.at[qid, "label_venue_id"])
        tt = CFGS[cfg]["target_time"]
        target = dict(venue_id=vid, lat=float(ven.at[vid, "lat"]), lon=float(ven.at[vid, "lon"]),
                      t_local_str=(pd.Timestamp(lab.at[qid, "label_t_local"]).strftime(TIME_FMT) if tt else SENT),
                      t_epoch=int(pd.Timestamp(lab.at[qid, "label_t_utc"]).value // 10**9))
        qs, an = _one(kind, cur, hist, u, tr, target, V, cat_id)
        if not tt:
            qs, an = _strip_tt(qs, an, u)
        out.append(dict(query_id=qid, user_id=u, traj_id=tr, question=qs, answer=an, n_hist=int(b - a),
                        n_cur=int(d - c)))
    return pd.DataFrame(out)


def query_prompts(ex, split: str, query_ids, cfg: str = "default", target_time: dict | None = None) -> pd.DataFrame:
    if split == "train":
        raise ValueError("query prompts are for val/test")
    qi = ex.query_inputs(split)
    q = qi.table
    idx = pd.Index(q["query_id"].to_numpy(np.int64))
    want = np.asarray([int(x) for x in query_ids], dtype=np.int64)
    pos = idx.get_indexer(want)
    if (pos < 0).any():
        raise KeyError(f"{int((pos < 0).sum())} query ids not in query_inputs_{split}")
    rows, ctx = ex.rows, qi.prefix_source
    cq = ctx["query_id"].to_numpy(np.int64)
    V = len(ex.venues)
    cat_id = category_ids(ex.venues)
    tt = CFGS[cfg]["target_time"]
    out = []
    for i, qid in zip(pos, want):
        a, b = qi.hist_pos[i]
        c, d = qi.prefix_pos[i]
        if not (cq[c:d] == qid).all():
            raise AssertionError("context slice of another query")
        u, tr = int(q["user_id"].iat[i]), int(q["traj_id"].iat[i])
        cur, hist = ctx.iloc[c:d], rows.iloc[a:b]
        target = dict(venue_id=-1, lat=0.0, lon=0.0,
                      t_local_str=(pd.Timestamp(target_time[int(qid)]).strftime(TIME_FMT) if tt else SENT),
                      t_epoch=_target_epoch_after(cur))
        qs, _ = _one("llm", cur, hist, u, tr, target, V, cat_id)
        if not tt:
            qs, _ = _strip_tt(qs, "", u)
        p = inference_prompt(qs, cfg)
        out.append(dict(query_id=int(qid), user_id=u, question=qs, prompt=p,
                        prompt_sha256=hashlib.sha256(p.encode()).hexdigest(), n_hist=int(b - a), n_cur=int(d - c)))
    return pd.DataFrame(out)


def inference_prompt(question: str, cfg: str) -> str:
    prompt1 = question
    tmp1, tmp2 = prompt1.split('Which POI id will user ')
    user_id = tmp2.split(' visit')[0]
    time = tmp1[-24:] if CFGS[cfg]["target_time"] else ""
    return prompt1 + '<answer>:' + f'{time} user {user_id} will visit POI id '


def load_target_time(ds: str, split: str) -> dict:
    p = os.path.join(ROOT, "exports_ablation", ds, f"target_time_{split}.parquet")
    t = pd.read_parquet(p)
    return dict(zip(t["query_id"].astype(np.int64), t.iloc[:, 1]))   # canary-allow: labelled tt flip only


def gps_mapping(ex) -> pd.DataFrame:
    v = ex.venues.sort_values("venue_id")
    if not (v["venue_id"].to_numpy() == np.arange(len(v))).all():
        raise AssertionError("venue ids are not dense 0..V-1")
    return pd.DataFrame({"GPS_id": v["venue_id"].astype(np.int64).to_numpy(),
                         "Latitude": v["lat"].astype(float).to_numpy(), "Longitude": v["lon"].astype(float).to_numpy()})


def code_fingerprint() -> str:
    h = hashlib.sha256()
    with open(os.path.abspath(__file__), "rb") as f:
        h.update(f.read())
    for _, sha in UPSTREAM.values():
        h.update(sha.encode())
    return h.hexdigest()[:16]
