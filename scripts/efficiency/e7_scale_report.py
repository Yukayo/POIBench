#!/usr/bin/env python
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

import numpy as np
import pandas as pd

DS = "tky"
FRACS = (20, 40, 60, 80, 100)
GPU_M = ("getnext", "getnext-zfeat", "graph-flashback")
GN_CFG = {"getnext": "released", "getnext-zfeat": "zfeat"}


def root_of(frac):
    return C.ROOT / "work" / "efficiency_scale" / f"u{frac}"


def du_mb(paths):
    return sum(p.stat().st_size for p in paths if p.is_file()) / 2 ** 20


def gpu_row(m, frac):
    sr = root_of(frac)
    pdir = sr / "preds" / f"{m}@H-100" / DS
    p = pdir / "test.cost.json"
    if not p.exists():
        return None
    j = json.load(open(p))
    st = j["stages"]
    once = [s for s in st if s.get("one_time")]
    tr = [s for s in st if s["name"] == "train"]
    pt = [s for s in st if s["name"] == "predict_test"]
    row = dict(method=m, frac=frac, host=j["hardware"]["host"], gpu_model=j["hardware"].get("gpu_model"),
               once_wall_s=sum(s["wall_s"] for s in once), once_gpu_s=sum(s.get("gpu_s", 0) for s in once),
               once_cpu_core_s=sum(s.get("cpu_core_s", 0) for s in once),
               train_wall_s=sum(s["wall_s"] for s in tr), train_gpu_busy_s=sum(s.get("gpu_busy_s", 0) or 0 for s in tr),
               peak_host_gb=max(s.get("peak_mem_gb", 0) or 0 for s in st),
               peak_gpu_gb_torch=max(s.get("peak_gpu_mem_gb", 0) or 0 for s in st),
               peak_gpu_gb_nvml=max(s.get("peak_gpu_mem_nvml_gb", 0) or 0 for s in st),
               query_ms=1000 * pt[0]["wall_s"] / pt[0]["n_items"] if pt else None,
               query_gpu_busy_ms=1000 * (pt[0].get("gpu_busy_s") or 0) / pt[0]["n_items"] if pt else None,
               n_test_queries=pt[0]["n_items"] if pt else None)
    if m in GN_CFG:
        row["epochs_run"] = j.get("epochs")
        row["best_epoch"] = j.get("best_epoch")
        row["train_items"] = j.get("train_sequences")
        w = sr / "work" / "getnext" / DS / "H-100"
        row["model_mb"] = du_mb([w / GN_CFG[m] / "train" / "best.pt"])
        try:
            g = np.load(w / "prepare" / "graph_A.npz")
            a = g[g.files[0]] if len(g.files) == 1 else None
            if a is not None:
                row["graph_nodes"] = int(a.shape[0])
                row["graph_nnz"] = int(np.count_nonzero(a))
            else:
                row["graph_files"] = ",".join(g.files)
        except Exception as e:
            row["graph_err"] = str(e)[:80]
    else:
        ms = j.get("model_summary") or {}
        pr = ms.get("prepare", {}).get("graphs", {})
        row["graph_nodes"] = (pr.get("loc_graph_shape") or [None])[0]
        row["graph_nnz"] = pr.get("loc_graph_nnz")
        row["user_graph_nnz"] = pr.get("user_loc_graph_nnz")
        row["train_items"] = ms.get("train", {}).get("n_loss_positions")
        row["best_epoch"] = ms.get("chosen_epoch")
        row["epochs_run"] = len(ms.get("val_history") or [])
        td = sr / j["train_dir"]
        row["model_mb"] = du_mb(list(td.glob("*.pt")) + list(td.glob("*.pth")) + list(td.glob("*.pkl"))) if td.exists() else None
        row["kge_cpu_core_s"] = sum(s.get("cpu_core_s", 0) for s in st if s["name"].startswith("kge"))
    return row


def accuracy(frac, methods):
    from poibench import evaluate as ev
    sr = root_of(frac)
    queries, cands, venues, _ = ev.load_split(str(sr), DS, "test", "H-100")
    P = {"hybrid": pd.read_parquet(sr / "preds" / "hybrid@H-100" / DS / "test.parquet")
         if frac != 100 or (sr / "preds" / "hybrid@H-100" / DS / "test.parquet").exists()
         else pd.read_parquet(C.ROOT / "preds" / "hybrid@H-100" / DS / "test.parquet"),
         "retriever": cands[["query_id", "cand_venue_ids"]].rename(columns={"cand_venue_ids": "ranked_venue_ids"})}
    for m in methods:
        p = sr / "preds" / f"{m}@H-100" / DS / "test.parquet"
        if p.exists():
            P[m] = pd.read_parquet(p)
    res = ev.evaluate(P, queries, candidates=None, venues=venues, B=1000, seed=0, ref="hybrid", pairs="ref",
                      strata=["all"], split="test", root=str(C.ROOT))
    R, PR = pd.DataFrame(res["results"]), pd.DataFrame(res["pairs"])
    out = {}
    for m in P:
        r = R[(R.method == m) & (R["mode"] == "e2e") & (R.metric == "acc@1")].iloc[0]
        d = dict(acc1=r.value, acc1_ci_lo=r.ci_lo, acc1_ci_hi=r.ci_hi, n_users_eval=res["n_users"],
                 n_queries_eval=res["n_queries"])
        if m != "hybrid":
            x = PR[(PR["mode"] == "e2e") & (PR.metric == "acc@1") & (PR.a == m)]
            if len(x):
                d.update(dacc1_vs_hybrid=x.iloc[0]["diff"], dacc1_ci_lo=x.iloc[0].ci_lo, dacc1_ci_hi=x.iloc[0].ci_hi)
        out[m] = d
    return out


def main():
    sizes = pd.read_csv(C.OUT / "e7_scale_sizes.csv").set_index("frac")
    cpu = pd.read_csv(C.OUT / "e7_scale_cpu.csv")
    rows = []
    for frac in FRACS:
        sz = sizes.loc[frac].to_dict()
        base = dict(frac=frac, n_users=sz["n_users"], n_checkins=sz["n_checkins"], n_train_checkins=sz["n_train_checkins"],
                    n_catalog=sz["n_catalog"], n_traj=sz["n_traj"], n_train_queries=sz["n_train_queries"],
                    n_test_queries_all=sz["n_test_queries"], poi_graph_edges=sz["graph_edges"])
        for comp in ("hybrid", "retriever"):
            c = cpu[(cpu.component == comp) & (cpu.frac == frac)]
            if not len(c):
                continue
            c = c.iloc[0]
            rows.append(dict(base, method=comp, host=c["hw_host"], once_wall_s=c["build_s"], once_cpu_core_s=c["build_s"],
                             once_gpu_s=0.0, peak_host_gb=c["peak_rss_gb"],
                             model_mb=c["state_mb"] if comp == "hybrid" else c["index_mb"],
                             query_ms=c["batch_ms_per_query"], query_p50_ms=c["lat_p50_ms"],
                             query_p95_ms=c["lat_p95_ms"]))
        for m in GPU_M:
            g = gpu_row(m, frac)
            if g:
                rows.append(dict(base, **g))
        acc = accuracy(frac, GPU_M)
        for r in rows:
            if r["frac"] == frac and r["method"] in acc:
                r.update(acc[r["method"]])
    df = C.write_csv("e7_scale.csv", rows, cols=["method", "frac", "n_users", "n_checkins", "once_wall_s", "once_gpu_s",
                                                  "peak_host_gb", "peak_gpu_gb_torch", "model_mb", "query_ms", "acc1"])
    sl = []
    for m, g in df.groupby("method"):
        for x in ("n_checkins", "n_train_checkins"):
            for y in ("once_wall_s", "once_gpu_s", "train_wall_s", "peak_host_gb", "peak_gpu_gb_torch", "model_mb",
                      "query_ms", "graph_nnz"):
                if y not in g or g[y].isna().all():
                    continue
                b, se, r2, n = C.loglog(g[x].tolist(), g[y].tolist())
                if b is not None:
                    sl.append(dict(method=m, x=x, y=y, slope=b, se=se, r2=r2, n=n,
                                   ratio_100_over_20=(float(g[g.frac == 100][y].iloc[0] / g[g.frac == 20][y].iloc[0])
                                                      if (g.frac == 100).any() and (g.frac == 20).any() else None)))
    C.write_csv("e7_scale_slopes.csv", sl)
    for _, r in df.iterrows():
        print(f"{r['method']:16s} {r['frac']:3d}% ck={r['n_checkins']:>7d} once={r['once_wall_s']:.1f}s "
              f"mem={r.get('peak_host_gb', float('nan')):.2f}GB model={r.get('model_mb', float('nan')):.2f}MB "
              f"q={r.get('query_ms', float('nan')):.4f}ms acc1={r.get('acc1', float('nan')):.4f}")


if __name__ == "__main__":
    main()
