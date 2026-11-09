#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

import numpy as np
import pandas as pd

FRESH = C.ROOT / "work" / "efficiency_fresh"
POL = "H-100"
RULE = ("FRESH (exploratory, EFFICIENCY.md E6 G8): a test query sees all check-ins of its user strictly before the "
        "query (train rows + val/test rows of earlier trajectories + the current prefix); population statistics "
        "(venues_public) unchanged (train rows); test split only.")


def fresh_dir(ds: str) -> Path:
    return FRESH / "exports" / ds / POL


def _ranges(lo, hi):
    from poibench.exports import _ranges_concat
    return _ranges_concat(np.asarray(lo, np.int64), np.asarray(hi, np.int64))


def make_fresh(out: dict, ck: pd.DataFrame) -> tuple[dict, dict]:
    from poibench.data import Arrays
    from poibench.exports import QIN_COLS, ROW_COLS
    ck = ck.sort_values("row_id").reset_index(drop=True)
    A = Arrays.from_checkins(ck)
    qi = out["query_inputs_test"].copy()
    r = qi["query_id"].to_numpy(np.int64)
    u = A.user[r]
    ts = A.traj_start[r]
    assert (qi["prefix_lo"].to_numpy() == ts).all() and (qi["prefix_hi"].to_numpy() == r).all()
    assert (qi["hist_lo"].to_numpy() == A.user_first[u]).all(), "H-100 history must start at the user's first row"
    old_hi = qi["hist_hi"].to_numpy(np.int64)
    idx, _, _ = _ranges(old_hi, ts)
    extra = np.unique(idx)
    assert (~A.is_train[extra]).all(), "added history rows must be val/test rows"
    vr = pd.concat([out["visible_rows"], ck.loc[extra, ROW_COLS]], ignore_index=True).sort_values("row_id")
    vr = vr.reset_index(drop=True)
    assert vr["row_id"].is_unique
    qi["hist_hi"] = ts
    qi["n_visible_own"] = ((qi["hist_hi"] - qi["hist_lo"]) + (qi["prefix_hi"] - qi["prefix_lo"])).astype(np.int64)
    last = extra[np.r_[A.traj[extra][1:] != A.traj[extra][:-1], True]]
    last = last[(A.traj_start[last] < last)]
    nxt = np.minimum(last + 1, len(A.traj) - 1)
    full = (last == len(A.traj) - 1) | (A.traj[nxt] != A.traj[last])
    last = last[full]
    t_utc, t_loc = ck["t_utc"].to_numpy(), ck["t_local"].to_numpy()
    pu = A.user[last]
    pseudo = pd.DataFrame({
        "query_id": last, "user_id": pu.astype(np.int32), "traj_id": A.traj[last].astype(np.int32),
        "prev_row_id": last - 1, "prev_venue_id": A.venue[last - 1].astype(np.int32), "prev_t_utc": t_utc[last - 1],
        "prev_t_local": t_loc[last - 1], "hist_lo": A.user_first[pu], "hist_hi": A.traj_start[last],
        "prefix_lo": A.traj_start[last], "prefix_hi": last,
        "n_visible_own": (last - A.user_first[pu]).astype(np.int64)})[QIN_COLS]
    qtr = pd.concat([out["query_inputs_train"], pseudo], ignore_index=True).sort_values("query_id").reset_index(drop=True)
    assert qtr["query_id"].is_unique
    new = {k: v for k, v in out.items() if not k.endswith("_val")}
    new["visible_rows"] = vr
    new["query_inputs_test"] = qi
    new["query_inputs_train"] = qtr
    new["policy_public"] = dict(out["policy_public"], fresh_rule=RULE)
    vpos = np.searchsorted(vr["row_id"].to_numpy(np.int64), np.arange(len(A.user)))
    inv = np.zeros(len(A.user), bool)
    inv[vr["row_id"].to_numpy(np.int64)] = True
    hidx, hown, _ = _ranges(qi["hist_lo"], qi["hist_hi"])
    pidx, pown, _ = _ranges(qi["prefix_lo"], qi["prefix_hi"])
    t = A.t
    st = dict(n_test_queries=int(len(r)), n_added_rows=int(len(extra)), n_pseudo_train_entries=int(len(pseudo)),
              hist_rows_in_visible=bool(inv[hidx].all()),
              hist_same_user=bool((A.user[hidx] == u[hown]).all()), prefix_same_user=bool((A.user[pidx] == u[pown]).all()),
              hist_strictly_before_target=bool((t[hidx] < t[r[hown]]).all()),
              prefix_strictly_before_target=bool((t[pidx] < t[r[pown]]).all()),
              n_prefix_rows_same_second_as_target=int((t[pidx] >= t[r[pown]]).sum()),
              hist_before_traj_start=bool((hidx < ts[hown]).all() and (t[hidx] < t[ts[hown]]).all()),
              mean_visible_own_frozen=float(((old_hi - qi["hist_lo"].to_numpy()) + (r - ts)).mean()),
              mean_visible_own_fresh=float(qi["n_visible_own"].mean()),
              share_queries_with_added_rows=float((old_hi < ts).mean()))
    assert st["hist_rows_in_visible"] and st["hist_same_user"] and st["prefix_same_user"]
    assert st["hist_strictly_before_target"] and st["hist_before_traj_start"]
    del vpos
    return new, st


def write_fresh(new: dict, path: Path, ds: str, extra: dict) -> dict:
    from poibench.exports import write_export
    base = json.load(open(C.ROOT / "exports" / ds / POL / "MANIFEST.json"))
    man = dict(ds=ds, policy=POL, fresh=True, fresh_rule=RULE, t_val_start_utc=base.get("t_val_start_utc"),
               source_sha256=base.get("source_sha256"), source_dir=base.get("source_dir"),
               protocol_export_files=base.get("files"), **extra)
    if path.exists():
        shutil.rmtree(path)
    return write_export(new, str(path), man)


def cmd_build(datasets):
    from poibench.exports import Export, build_export, load_sources
    rows = []
    for ds in datasets:
        t0 = time.time()
        ck, venues, queries, pol, cands, meta = load_sources(ds, POL)
        out = build_export(ck, venues, queries, pol, cands)
        ref = Export(str(C.ROOT / "exports" / ds / POL))
        for name in ("visible_rows", "query_inputs_test", "context_test", "candidates_test", "query_inputs_train"):
            assert ref._pq(name).equals(out[name].reset_index(drop=True)), f"{ds}: rebuilt {name} != stored export"
        new, st = make_fresh(out, ck)
        write_fresh(new, fresh_dir(ds), ds, dict(build_checks=st))
        st.update(ds=ds, seconds=round(time.time() - t0, 1))
        print(json.dumps(st), flush=True)
        rows.append(st)
    C.write_csv("e6_fresh_export.csv", rows)


def perturb_fresh(ck, venues, queries, S, seed=0):
    from poibench.data import Arrays
    rng = np.random.default_rng(seed + 1)
    ck = ck.sort_values("row_id").reset_index(drop=True)
    A = Arrays.from_checkins(ck)
    protect = np.zeros(len(A.user), bool)
    for q in S:
        protect[A.user_first[A.user[q]]:q] = True
    mut = (~A.is_train) & ~protect
    idx = np.flatnonzero(mut)
    vt = venues.sort_values("venue_id").reset_index(drop=True)
    catalog = vt["venue_id"].to_numpy(np.int64)[vt["n_train_visits"].to_numpy() > 0]
    old = A.venue[idx]
    new = catalog[rng.integers(0, len(catalog), size=len(idx))]
    same = new == old
    while same.any():
        new[same] = catalog[rng.integers(0, len(catalog), size=int(same.sum()))]
        same = new == old
    ck2 = ck.copy()
    ck2.loc[idx, "venue_id"] = new.astype(ck["venue_id"].dtype)
    for c in ("category", "lat", "lon"):
        ck2.loc[idx, c] = vt[c].to_numpy()[new]
    shift_h = rng.integers(1, 13, size=int(A.traj.max()) + 1)
    delta = pd.to_timedelta(shift_h[A.traj[idx]], unit="h")
    ck2.loc[idx, "t_utc"] = ck["t_utc"].to_numpy()[idx] + delta.to_numpy()
    ck2.loc[idx, "t_local"] = ck["t_local"].to_numpy()[idx] + delta.to_numpy()
    q2 = {}
    v2 = ck2["venue_id"].to_numpy()
    for s, q in queries.items():
        q = q.copy()
        q["target_venue_id"] = v2[q["target_row_id"].to_numpy(np.int64)].astype(q["target_venue_id"].dtype)
        q2[s] = q
    return ck2, q2, dict(n_rows_replaced=int(len(idx)), n_protected_rows=int(protect.sum()),
                         n_protected_valtest_rows=int((protect & ~A.is_train).sum()))


def _import(path, name):
    import importlib.util
    sp = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(mod)
    return mod


def _leaky_target(export_dir, split, query_ids, work_dir=None):
    from poibench.exports import Export
    ex = Export(export_dir)
    vr = ex.rows.set_index("row_id")["venue_id"]
    q = ex.queries(split).set_index("query_id")
    out = []
    for qid in query_ids:
        t = vr.get(int(qid))
        pv = int(q.at[int(qid), "prev_venue_id"])
        out.append([int(t), pv] if t is not None and int(t) != pv else [pv])
    return pd.DataFrame({"query_id": list(query_ids), "ranked_venue_ids": out})


def cmd_canary(datasets, n):
    from poibench.canary import _frames_equal, _norm_list, choose_queries, lint_sources
    from poibench.exports import Export, build_export, load_sources
    os.environ.setdefault("LLM4POI_BACKEND", "mock")
    os.environ.setdefault("DSLLM4POI_BACKEND", "mock")
    os.environ.setdefault("LLM4POI_CACHE_DIR", str(FRESH / "canary_mock_cache" / "llm4poi"))
    os.environ.setdefault("DSLLM4POI_CACHE_DIR", str(FRESH / "canary_mock_cache" / "ds"))
    l4p = _import(C.ROOT / "methods" / "llm4poi" / "run.py", "e6_l4p_run")
    dsm = _import(C.ROOT / "methods" / "ds_llm4poi" / "run.py", "e6_ds_run")
    rows = []
    for ds in datasets:
        t0 = time.time()
        real = fresh_dir(ds)
        assert (real / "MANIFEST.json").exists(), f"build the fresh export of {ds} first"
        ck, venues, queries, pol, cands, meta = load_sources(ds, POL)
        S = choose_queries(queries["test"], n, 0)
        ck2, q2, pinfo = perturb_fresh(ck, venues, queries, S)
        work = FRESH / "canary" / ds
        if work.exists():
            shutil.rmtree(work)
        pert = work / "export_perturbed"
        new2, _ = make_fresh(build_export(ck2, venues, q2, pol, cands), ck2)
        write_fresh(new2, pert, ds, dict(perturbed_for_canary=True, n_protected_queries=int(len(S))))
        A0, A1 = Export(str(real)), Export(str(pert))
        sset = set(int(x) for x in S)
        inv = {}
        qa, qb = A0.queries("test"), A1.queries("test")
        inv["query_inputs_test[S]"] = _frames_equal(qa[qa.query_id.isin(sset)].reset_index(drop=True),
                                                    qb[qb.query_id.isin(sset)].reset_index(drop=True))
        ca, cb = A0.context("test"), A1.context("test")
        inv["context_test[S]"] = _frames_equal(ca[ca.query_id.isin(sset)].reset_index(drop=True),
                                               cb[cb.query_id.isin(sset)].reset_index(drop=True))
        inv["candidates_test"] = _frames_equal(A0.candidates("test"), A1.candidates("test"))
        inv["venues_public"] = _frames_equal(A0.venues, A1.venues)
        qs = qa[qa.query_id.isin(sset)]
        hidx, _, _ = _ranges(qs["hist_lo"], qs["hist_hi"])
        ra, rb = A0.rows.set_index("row_id"), A1.rows.set_index("row_id")
        inv["history_rows[S]"] = _frames_equal(ra.loc[hidx].reset_index(), rb.loc[hidx].reset_index())
        exposed = int(ra.index.isin(list(sset)).sum())
        exposed_changed = int(sum(ra.at[q, "venue_id"] != rb.at[q, "venue_id"] for q in sset if q in ra.index))
        qids = [int(x) for x in S]
        res = {}
        pa = l4p.PR.query_prompts(A0, "test", qids, "default")
        pb = l4p.PR.query_prompts(A1, "test", qids, "default")
        res["llm4poi_prompt_mismatch"] = int((pa["prompt"].to_numpy() != pb["prompt"].to_numpy()).sum())
        res["llm4poi_mean_n_hist"] = float(pa["n_hist"].mean())
        da = dsm.PR.build_prompts(str(real), "test", "lcs", qids, n_jobs=1)
        db = dsm.PR.build_prompts(str(pert), "test", "lcs", qids, n_jobs=1)
        res["ds_prompt_mismatch"] = int((da["prompt"].to_numpy() != db["prompt"].to_numpy()).sum())
        res["ds_mean_n_bank"] = float(da["n_bank"].mean())
        for tag, fn in (("llm4poi", l4p.predict), ("ds", dsm.predict), ("leaky_control", _leaky_target)):
            ya = fn(export_dir=str(real), split="test", query_ids=qids, work_dir=None).set_index("query_id")
            yb = fn(export_dir=str(pert), split="test", query_ids=qids, work_dir=None).set_index("query_id")
            res[f"{tag}_pred_mismatch"] = int(sum(_norm_list(ya.at[q, "ranked_venue_ids"]) !=
                                                  _norm_list(yb.at[q, "ranked_venue_ids"]) for q in qids))
        lint = lint_sources([str(C.ROOT / "methods" / "llm4poi"), str(C.ROOT / "methods" / "ds_llm4poi")])
        ok = (all(inv.values()) and res["llm4poi_prompt_mismatch"] == 0 and res["ds_prompt_mismatch"] == 0
              and res["llm4poi_pred_mismatch"] == 0 and res["ds_pred_mismatch"] == 0 and not lint
              and res["leaky_control_pred_mismatch"] > 0)
        rep = dict(result="PASS" if ok else "FAIL", rule=RULE, ds=ds, policy=POL, split="test", n_queries=len(qids),
                   seed=0, created_utc=pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ"), perturbation=pinfo,
                   exporter_invariance=inv, s_targets_in_fresh_visible_rows=exposed,
                   s_target_copies_changed=exposed_changed, **res,
                   lint=[dict(file=f, line=i, why=w, text=t) for f, i, w, t in lint],
                   real_export_files=A0.manifest.get("files"), seconds=round(time.time() - t0, 1))
        d = C.ROOT / "logs" / "canary"
        d.mkdir(parents=True, exist_ok=True)
        json.dump(rep, open(d / f"e6_fresh__{ds}__{POL}__test.json", "w"), indent=1)
        print(json.dumps({k: rep[k] for k in rep if k not in ("lint", "real_export_files", "rule")}), flush=True)
        rows.append({k: v for k, v in rep.items() if not isinstance(v, (dict, list))} | {"n_lint": len(lint)})
        shutil.rmtree(work, ignore_errors=True)
    C.write_csv("e6_fresh_canary.csv", rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "canary"])
    ap.add_argument("--datasets", nargs="+", default=list(C.MAIN))
    ap.add_argument("--n", type=int, default=300)
    a = ap.parse_args()
    if a.cmd == "build":
        cmd_build(a.datasets)
    else:
        cmd_canary(a.datasets, a.n)


if __name__ == "__main__":
    main()
