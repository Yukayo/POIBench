#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from poibench.exports import Export, export_dir

TOPK = 50


def _stats(ex: Export):
    ven = ex.venues
    V = len(ven)
    pop = ven.sort_values("venue_id")["n_visible_train_visits"].to_numpy(np.int64)
    rows = ex.rows
    rid = rows["row_id"].to_numpy(np.int64)
    tr = rows["traj_id"].to_numpy(np.int64)
    v = rows["venue_id"].to_numpy(np.int64)
    pm = np.r_[False, (np.diff(rid) == 1) & (tr[1:] == tr[:-1])]
    a, b = v[np.flatnonzero(pm) - 1], v[pm]
    markov = {}
    if len(a):
        uk, cnt = np.unique(a * V + b, return_counts=True)
        pa, pb = uk // V, uk % V
        bounds = np.r_[0, np.flatnonzero(np.diff(pa)) + 1, len(pa)]
        for s, e in zip(bounds[:-1], bounds[1:]):
            markov[int(pa[s])] = (pb[s:e].astype(np.int64), cnt[s:e].astype(np.int64))
    return V, pop, np.lexsort((np.arange(V), -pop)), markov


def _order(codes, vis, prev, pop, markov):
    popv = pop[codes]
    uf = np.zeros(len(codes))
    ur = np.full(len(codes), -1.0)
    if len(vis):
        uv, cnt = np.unique(vis, return_counts=True)
        _, first_rev = np.unique(vis[::-1], return_index=True)
        last = len(vis) - 1 - first_rev
        j = np.minimum(np.searchsorted(uv, codes), len(uv) - 1)
        hit = uv[j] == codes
        uf[hit] = cnt[j[hit]]
        ur[hit] = last[j[hit]]
    mk = np.zeros(len(codes))
    sm = 0.0
    succ = markov.get(int(prev))
    if succ is not None:
        sv, sc = succ
        sm = float(sc.sum())
        j = np.minimum(np.searchsorted(sv, codes), len(sv) - 1)
        hit = sv[j] == codes
        mk[hit] = sc[j[hit]]
    su = float(len(vis))
    score = (uf / su if su > 0 else 0.0) + (mk / sm if sm > 0 else 0.0)
    return codes[np.lexsort((codes, -popv, -ur, -score))]


def predict(export_dir, split, query_ids, work_dir=None):
    if split == "train":
        raise ValueError("export_hybrid predicts val/test queries only")
    ex = Export(export_dir)
    V, pop, pop_order, markov = _stats(ex)
    qi = ex.query_inputs(split)
    pos = pd.Series(np.arange(len(qi.table)), index=qi.table["query_id"].to_numpy())
    cmap = dict(zip(ex.candidates(split)["query_id"], ex.candidates(split)["cand_venue_ids"]))
    prev_v = qi.table["prev_venue_id"].to_numpy(np.int64)
    top, cand = [], []
    for qid in query_ids:
        i = int(pos[int(qid)])
        vis = qi.own(i, "venue_id").astype(np.int64)
        prev = prev_v[i]
        parts = [pop_order[:TOPK + 1], vis]
        if int(prev) in markov:
            parts.append(markov[int(prev)][0])
        pool = np.unique(np.concatenate(parts))
        top.append(_order(pool, vis, prev, pop, markov)[:TOPK].tolist())
        c = np.asarray(cmap[int(qid)], dtype=np.int64)
        _, first = np.unique(c, return_index=True)
        cand.append(_order(c[np.sort(first)], vis, prev, pop, markov).tolist())
    return pd.DataFrame({"query_id": [int(x) for x in query_ids], "ranked_venue_ids": top,
                         "cand_ranked_venue_ids": cand})


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--compare", action="store_true",
                    help="compare with preds/hybrid@<policy>/<ds>/<split>.parquet (poibench/baselines_simple.py)")
    a = ap.parse_args(argv)
    ed = export_dir(a.ds, a.policy, ROOT)
    qids = Export(ed).queries(a.split)["query_id"].tolist()
    p = predict(ed, a.split, qids)
    if a.compare:
        ref = pd.read_parquet(os.path.join(ROOT, "preds", f"hybrid@{a.policy}", a.ds, f"{a.split}.parquet"))
        ref = ref.set_index("query_id").loc[p["query_id"]]
        same_top = sum(list(x) == list(y) for x, y in zip(p["ranked_venue_ids"], ref["ranked_venue_ids"]))
        same_c = sum(list(x) == list(y) for x, y in zip(p["cand_ranked_venue_ids"], ref["cand_ranked_venue_ids"]))
        print(f"{a.ds}/{a.policy}/{a.split}: {len(p)} queries; identical ranked_venue_ids {same_top}, "
              f"identical cand_ranked_venue_ids {same_c}")
        return 0 if same_top == len(p) and same_c == len(p) else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
