#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

POLS = ("H-10", "H-70", "H-100", "S-10", "S-70")


def native_train_items(m: str, j: dict | None):
    if not j:
        return None
    try:
        if m.startswith("getnext"):
            return j["train_sequences"]["n_labels_used"]
        if m == "graph-flashback":
            return j["model_summary"]["train"]["n_loss_positions"]
        if m.startswith("gnpr-sid"):
            return j["train_meta"]["fingerprint"]["n_sft_samples"]
        if m.startswith("ga-llm"):
            return j["sft"]["n_samples_stage2"] * j["sft"]["epochs"]
        if m.startswith("llm4poi"):
            s = j["sft"]
            return int(round(s["global_step"] * s["effective_batch"] / float(s["args"]["num_train_epochs"])))
    except (KeyError, TypeError, ValueError):
        return None
    return None


def prompt_tokens(m: str, j: dict | None):
    if not j:
        return None
    t = j.get("tokens") or {}
    return t.get("prompt_mean")


def records():
    rows = []
    for m in C.METHODS:
        for ds in C.MAIN + (C.BIG if m == "hybrid" else ()):
            for pol in POLS:
                c = C.mt.load_cost(m, pol, ds)
                if c is None:
                    continue
                j = C.cost_json(m, pol, ds, "test")
                q = c["perq"].get("test") or {}
                size_pol = pol if pol != "S-100" else "H-100"
                r = {"method": m, "family": C.FAMILY[m], "policy": pol, "ds": ds, "source": c["source"],
                     "colocated": c["colocated"], **C.data_sizes(ds, size_pol),
                     "native_train_items": native_train_items(m, j), "prompt_tokens_mean": prompt_tokens(m, j),
                     "once_gpu_s": c["once"]["gpu_s"], "once_gpu_busy_s": c["once"].get("gpu_busy_s"),
                     "once_wall_s": c["once"]["wall_s"], "once_cpu_core_s": c["once"]["cpu_core_s"],
                     "perq_gpu_s": q.get("gpu_s"), "perq_gpu_busy_s": q.get("gpu_busy_s"), "perq_wall_s": q.get("wall_s"),
                     "perq_cpu_core_s": q.get("cpu_core_s"), "peak_gpu_mem_gb": c["peak_gpu_mem"],
                     "peak_host_mem_gb": max([c["once_peak_mem"]] + [v.get("peak_mem") or 0 for v in c["perq"].values()])}
                rows.append(r)
    return rows


def slopes(df):
    out = []
    trained = [m for m in C.METHODS if C.FAMILY[m] != "counting" and m != "ds-llm4poi-lcs"]

    def add(m, analysis, xv, yv, res, note=""):
        b, se = res[0], res[1]
        out.append({"method": m, "analysis": analysis, "x": xv, "y": yv, "slope": b, "se": se,
                    "r2": res[2] if analysis.startswith("across") else None,
                    "n_points": res[3] if analysis.startswith("across") else res[2],
                    "n_groups": None if analysis.startswith("across") else res[3], "note": note})

    h100 = df[(df["policy"] == "H-100") & (df["ds"].isin(C.MAIN))]
    for m in C.METHODS:
        d = h100[h100["method"] == m]
        if m in trained:
            add(m, "across_cities_H-100", "n_train_queries", "once_gpu_s", C.loglog(d["n_train_queries"], d["once_gpu_s"]))
            add(m, "across_cities_H-100", "n_train_checkins", "once_gpu_s", C.loglog(d["n_train_checkins"], d["once_gpu_s"]))
            add(m, "across_cities_H-100", "native_train_items", "once_gpu_s",
                C.loglog(d["native_train_items"], d["once_gpu_s"]))
        if m != "hybrid":
            add(m, "across_cities_H-100", "mean_visible_own_test", "perq_gpu_s",
                C.loglog(d["mean_visible_own_test"], d["perq_gpu_s"]))
            add(m, "across_cities_H-100", "n_catalog_venues", "perq_gpu_s", C.loglog(d["n_catalog_venues"], d["perq_gpu_s"]))
        if d["prompt_tokens_mean"].notna().sum() >= 2:
            add(m, "across_cities_H-100", "prompt_tokens_mean", "perq_gpu_s",
                C.loglog(d["prompt_tokens_mean"], d["perq_gpu_s"]))
        if m == "hybrid":
            add(m, "across_cities_H-100", "mean_visible_own_test", "perq_cpu_core_s",
                C.loglog(d["mean_visible_own_test"], d["perq_cpu_core_s"]))

    s = df[df["policy"].isin(["S-10", "S-70", "H-100"]) & df["ds"].isin(["nyc", "tky", "istanbul"])]
    for m in trained:
        d = s[s["method"] == m]
        g = {ds: list(zip(x["n_train_queries"], x["once_gpu_s"])) for ds, x in d.groupby("ds")}
        add(m, "within_city_samples(S-x,H-100)", "n_train_queries", "once_gpu_s", C.loglog_fe(g))
        g = {ds: list(zip(x["native_train_items"], x["once_gpu_s"])) for ds, x in d.groupby("ds")}
        add(m, "within_city_samples(S-x,H-100)", "native_train_items", "once_gpu_s", C.loglog_fe(g))
    h = df[df["policy"].isin(["H-10", "H-70", "H-100"]) & df["ds"].isin(["nyc", "tky", "istanbul"])]
    for m in C.METHODS:
        d = h[h["method"] == m]
        if m == "hybrid":
            g = {ds: list(zip(x["mean_visible_own_test"], x["perq_cpu_core_s"])) for ds, x in d.groupby("ds")}
            add(m, "within_city_history(H-x)", "mean_visible_own_test", "perq_cpu_core_s", C.loglog_fe(g))
            continue
        g = {ds: list(zip(x["mean_visible_own_test"], x["perq_gpu_s"])) for ds, x in d.groupby("ds")}
        add(m, "within_city_history(H-x)", "mean_visible_own_test", "perq_gpu_s", C.loglog_fe(g))
        if d["prompt_tokens_mean"].notna().sum() >= 2:
            g = {ds: list(zip(x["prompt_tokens_mean"], x["perq_gpu_s"])) for ds, x in d.groupby("ds")}
            add(m, "within_city_history(H-x)", "prompt_tokens_mean", "perq_gpu_s", C.loglog_fe(g))
    for m in C.METHODS:
        d = df[df["method"] == m].set_index(["ds", "policy"])
        for ds in ("nyc", "tky", "istanbul"):
            for x in ("10", "70"):
                if (ds, f"H-{x}") in d.index and (ds, f"S-{x}") in d.index:
                    a, b = d.loc[(ds, f"H-{x}")], d.loc[(ds, f"S-{x}")]
                    yk = "perq_cpu_core_s" if m == "hybrid" else "perq_gpu_s"
                    out.append({"method": m, "analysis": f"ratio_H-{x}/S-{x}", "x": ds, "y": yk,
                                "slope": (a[yk] / b[yk]) if b[yk] else None, "note": "per-query cost ratio, same samples"})
                    if m in trained:
                        out.append({"method": m, "analysis": f"ratio_H-{x}/S-{x}", "x": ds, "y": "once_gpu_s",
                                    "slope": (a["once_gpu_s"] / b["once_gpu_s"]) if b["once_gpu_s"] else None,
                                    "note": "one-time cost ratio, same samples"})
    hb = C.OUT / "e1_hybrid_bench.csv"
    if hb.exists():
        import pandas as pd
        b = pd.read_csv(hb)
        b = b[b["policy"] == "H-100"]
        for xv, yv in (("n_visible_train_checkins", "fit_s"), ("n_visible_train_checkins", "predict_ms_per_query"),
                       ("n_visible_train_checkins", "state_mb"), ("n_visible_train_checkins", "peak_rss_gb"),
                       ("mean_visible_own_test", "predict_ms_per_query")):
            add("hybrid", "across_datasets_measured(8)", xv, yv, C.loglog(b[xv], b[yv]))
    return out


def main():
    import pandas as pd
    rows = records()
    df = C.write_csv("e1_cost_records.csv", rows)
    for col in ("n_train_queries", "once_gpu_s", "perq_gpu_s", "prompt_tokens_mean", "mean_visible_own_test",
                "perq_cpu_core_s", "n_train_checkins", "n_catalog_venues", "native_train_items"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    sl = C.write_csv("e1_slopes.csv", slopes(df))
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(sl[["method", "analysis", "x", "y", "slope", "se", "n_points"]].to_string())


if __name__ == "__main__":
    main()
