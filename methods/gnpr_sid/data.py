from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import pandas as pd

from poibench import ROOT
from poibench.exports import Export

HIST_WINDOW = 50
MIN_TRAIN_LEN = 20
KEEP_LAST_K = 5
TIME_FMT = "%Y-%m-%d %H:%M:%S"

INSTRUCTION_UPSTREAM = ("Here is a record of a user's POI accesses, your task is based on the history "
                        "to predict the POI that the user is likely to access at the specified time.")
INSTRUCTION = ("Here is a record of a user's POI accesses, your task is based on the history "
               "to predict the POI that the user is likely to access next.")
CFGS = {"default": dict(target_time=False, instruction=INSTRUCTION),
        "tt": dict(target_time=True, instruction=INSTRUCTION_UPSTREAM)}


def encode_item(item) -> str:
    labels = ["a", "b", "c", "d"]
    if len(item) > 4:
        raise ValueError(f"SID longer than 4 levels: {item}")
    return "".join(f"<{labels[i]}_{int(v)}>" for i, v in enumerate(item))


def input_text(uid: int, times, codes, target_time: str | None = None) -> str:
    history_text = ", ".join(f"{t} visited {p}" for t, p in zip(times, codes))
    if target_time is None:
        return (f"User_{uid} checkin history: {history_text}.\n"
                f"Next, user_{uid} is likely to visit:")
    return (f"User_{uid} checkin history: {history_text}.\n"
            f"When {target_time} user_{uid} is likely to visit:")


def load_target_time(ds: str, split: str) -> dict:
    p = os.path.join(ROOT, "exports_ablation", ds, f"target_time_{split}.parquet")
    if not os.path.exists(p):
        raise FileNotFoundError(f"{p} missing: run scripts/export_target_time.py --ds {ds} (cfg tt only)")
    t = pd.read_parquet(p)
    return dict(zip(t["query_id"].astype(np.int64), t.iloc[:, 1]))   # canary-allow: labelled tt ablation only


def prompt_text(inp: str, instruction: str = INSTRUCTION) -> str:
    return (f"### Instruction:\n{instruction.strip()}\n\n"
            f"### Input:\n{inp.strip()}\n\n"
            f"### Response:\n")


def _fmt_times(t) -> np.ndarray:
    return pd.to_datetime(pd.Series(t)).dt.strftime(TIME_FMT).to_numpy(dtype=object)


def poi_info(ex: Export) -> pd.DataFrame:
    ven = ex.venues
    cat = ven[ven["in_catalog"].to_numpy()].sort_values("venue_id")
    rows = ex.rows
    hours = pd.to_datetime(rows["t_local"]).dt.hour.to_numpy()
    v = rows["venue_id"].to_numpy(np.int64)
    key, cnt = np.unique(v * 24 + hours, return_counts=True)
    per = {}
    for k, c in zip(key, cnt):
        per.setdefault(int(k // 24), []).append((int(k % 24), int(c)))
    hist = []
    for pid in cat["venue_id"].to_numpy(np.int64):
        items = sorted(per.get(int(pid), []), key=lambda x: (-x[1], x[0]))
        hist.append(str({h: c for h, c in items}))
    return pd.DataFrame({"pid": cat["venue_id"].to_numpy(np.int64), "category": cat["category"].to_numpy(),
                         "latitude": cat["lat"].to_numpy(), "longitude": cat["lon"].to_numpy(),
                         "visit_time_and_count": hist})


def poi_info_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def sft_samples(ex: Export, pid2code: dict | None, cfg: str = "default"):
    C = CFGS[cfg]
    rows = ex.rows.sort_values("row_id")
    rid = rows["row_id"].to_numpy(np.int64)
    uid = rows["user_id"].to_numpy(np.int64)
    tid = rows["traj_id"].to_numpy(np.int64)
    vid = rows["venue_id"].to_numpy(np.int64)
    tstr = _fmt_times(rows["t_local"])
    trainq = set(int(x) for x in ex.queries("train")["query_id"].to_numpy(np.int64))
    brk = np.r_[True, (uid[1:] != uid[:-1]) | (tid[1:] != tid[:-1])]
    seg_start = np.flatnonzero(brk)
    seg_end = np.r_[seg_start[1:], len(rid)]
    ustart = {}
    for i in np.flatnonzero(np.r_[True, uid[1:] != uid[:-1]]):
        ustart[int(uid[i])] = int(i)
    per_user = {}
    st = dict(n_segments=int(len(seg_start)), n_lt2=0, n_lt_min_len=0, n_target_not_train_query=0)
    for s, e in zip(seg_start, seg_end):
        u = int(uid[s])
        lo = max(ustart[u], e - HIST_WINDOW)
        n = e - lo
        if n < 2:
            st["n_lt2"] += 1
            continue
        if n < MIN_TRAIN_LEN:
            st["n_lt_min_len"] += 1
            continue
        if int(rid[e - 1]) not in trainq:
            st["n_target_not_train_query"] += 1
            continue
        per_user.setdefault(u, []).append((lo, e))
    out, meta = [], []
    for u in sorted(per_user):
        for lo, e in per_user[u][-KEEP_LAST_K:]:
            hv, ht = vid[lo:e - 1], tstr[lo:e - 1]
            tgt = int(vid[e - 1])
            if pid2code is None:
                codes, target = [str(int(x)) for x in hv], str(tgt)
            else:
                codes, target = [pid2code[int(x)] for x in hv], pid2code[tgt]
            tt = tstr[e - 1] if C["target_time"] else None
            out.append({"instruction": C["instruction"], "input": input_text(u, ht, codes, tt), "output": target})
            meta.append((u, int(rid[e - 1]), int(e - lo)))
    st.update(n_samples=len(out), n_users=len(per_user),
              mean_len=float(np.mean([m[2] for m in meta])) if meta else 0.0,
              target_row_ids_sha256=hashlib.sha256(",".join(str(m[1]) for m in meta).encode()).hexdigest())
    return out, st


def samples_sha256(samples: list) -> str:
    return hashlib.sha256(json.dumps(samples, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def eval_prompts(ex: Export, split: str, query_ids, pid2code: dict, cfg: str = "default",
                 target_time: dict | None = None) -> pd.DataFrame:
    C = CFGS[cfg]
    if C["target_time"] and target_time is None:
        raise ValueError("cfg tt needs target_time (load_target_time)")
    qi = ex.query_inputs(split)
    tab = qi.table
    pos = pd.Series(np.arange(len(tab)), index=tab["query_id"].to_numpy(np.int64))
    hv_all = qi.rows["venue_id"].to_numpy(np.int64)
    ht_all = qi.rows["t_local"].to_numpy()
    pv_all = qi.prefix_source["venue_id"].to_numpy(np.int64)
    pt_all = qi.prefix_source["t_local"].to_numpy()
    users = tab["user_id"].to_numpy(np.int64)
    recs = []
    for q in query_ids:
        i = int(pos[int(q)])
        a, b = qi.hist_pos[i]
        c, d = qi.prefix_pos[i]
        v = np.concatenate([hv_all[a:b], pv_all[c:d]])
        t = np.concatenate([ht_all[a:b], pt_all[c:d]])
        known = np.array([int(x) in pid2code for x in v], dtype=bool)
        n_drop = int((~known).sum())
        v, t = v[known][-(HIST_WINDOW - 1):], t[known][-(HIST_WINDOW - 1):]
        tt = _fmt_times([target_time[int(q)]])[0] if C["target_time"] else None
        inp = input_text(int(users[i]), _fmt_times(t), [pid2code[int(x)] for x in v], tt)
        recs.append(dict(query_id=int(q), prompt=prompt_text(inp, C["instruction"]), n_items=int(len(v)),
                         n_dropped_no_sid=n_drop))
    return pd.DataFrame(recs)
