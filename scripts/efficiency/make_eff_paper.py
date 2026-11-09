#!/usr/bin/env python
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

import numpy as np
import pandas as pd

mt = C.mt
OUT = C.OUT
NUM: dict = {}
ORDER = ("hybrid", "getnext", "graph-flashback", "ds-llm4poi-lcs", "llm4poi", "gnpr-sid", "ga-llm")
COL = dict(zip(ORDER, (mt.PAL[0], mt.PAL[1], mt.PAL[2], mt.PAL[3], mt.PAL[4], mt.PAL[6], mt.PAL[7])))
COL["hybrid-fresh"] = mt.PAL[0]
MK = dict(zip(ORDER, ("o", "s", "^", "D", "v", "X", "P")))
MK["hybrid-fresh"] = "o"
LAB = dict(C.MLABEL, **{"hybrid-fresh": "Hybrid, kept fresh"})
LONG = ("sin", "istanbul", "sao_paulo")


def rd(name):
    p = OUT / name
    return pd.read_csv(p) if p.exists() else None


def f4(x):
    return mt.DASH if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.4f}".replace("-", "$-$")


def s4(x):
    return mt.sf4(x) if x is not None and not (isinstance(x, float) and np.isnan(x)) else mt.DASH


def q2(x):
    return mt.q2(x) if x is not None and not (isinstance(x, float) and np.isnan(x)) else mt.DASH


def num(k, v):
    NUM[k] = v


def e5():
    out = {}
    d = OUT / "e5"
    for p in sorted(d.glob("*.json")) if d.exists() else []:
        if p.name.endswith(".mock.json"):
            continue
        j = json.load(open(p))
        if not j.get("mock"):
            out[j["method"]] = j
    return out


def numbers(rec, sl, hb, up, drift, e3, ret, sv):
    if sl is not None:
        for _, r in sl.iterrows():
            if r["analysis"].startswith("ratio") or pd.isna(r["slope"]):
                continue
            a = {"across_cities_H-100": "across", "within_city_samples(S-x,H-100)": "withinS",
                 "within_city_history(H-x)": "withinH", "across_datasets_measured(8)": "measured"}.get(r["analysis"])
            if a:
                num(f"eff:slope:{a}:{r['method']}:{r['x']}:{r['y']}", f4(float(r["slope"])))
    if hb is not None:
        b = hb[hb["policy"] == "H-100"]
        for _, r in b.iterrows():
            num(f"eff:hyb:{r['ds']}:fit", q2(r["fit_s"]))
            num(f"eff:hyb:{r['ds']}:msq", f"{r['predict_ms_per_query']:.2f}")
            num(f"eff:hyb:{r['ds']}:p50ms", f"{r['lat_p50_ms']:.2f}")
            num(f"eff:hyb:{r['ds']}:ckM", f"{r['n_visible_train_checkins'] / 1e6:.2f}")
            num(f"eff:hyb:{r['ds']}:rss", q2(r["peak_rss_gb"]))
            num(f"eff:hyb:{r['ds']}:statemb", q2(r["state_mb"]))
        num("eff:hyb:msq:min", f"{b['predict_ms_per_query'].min():.2f}")
        num("eff:hyb:msq:max", f"{b['predict_ms_per_query'].max():.2f}")
        num("eff:hyb:qps1core:min", f"{1000 / b['predict_ms_per_query'].max():,.0f}".replace(",", "{,}"))
        num("eff:hyb:match", f4(float(hb["match_preds"].min())))
    if up is not None:
        num("eff:upd:us:min", q2(up["insert_us_mean"].min()))
        num("eff:upd:us:max", q2(up["insert_us_mean"].max()))
        num("eff:upd:p99:max", q2(up["insert_us_p99"].max()))
        num("eff:upd:week:max", q2(up["week_of_inserts_cpu_s"].max()))
        num("eff:upd:refit:max", q2(up["refit_all_s"].max()))
        num("eff:upd:resort:max", q2(up["pop_resort_ms"].max()))
        num("eff:upd:inserts:total", mt.intc(int(up["n_inserts"].sum())))
        for _, r in up.iterrows():
            num(f"eff:upd:{r['ds']}:us", q2(r["insert_us_mean"]))
            num(f"eff:upd:{r['ds']}:weeks", q2(r["week_of_inserts_cpu_s"]))
    if rec is not None:
        h = rec[(rec["policy"] == "H-100") & rec["ds"].isin(C.MAIN)]
        for m in ("getnext", "graph-flashback", "llm4poi", "gnpr-sid", "ga-llm"):
            g = h[h["method"] == m]["once_gpu_s"] / 3600
            if len(g):
                num(f"eff:retrain:{m}:min", q2(g.min()))
                num(f"eff:retrain:{m}:max", q2(g.max()))
    if drift is not None:
        for _, r in drift.iterrows():
            k = f"eff:drift:{r['ds']}:{r['method']}"
            num(k + ":acc1", f4(r["acc1"]))
            num(k + ":slope", s4(r["slope_per_week"]))
            if not pd.isna(r.get("dslope_vs_hybrid", np.nan)):
                lo, hi = r["dslope_vs_hybrid_ci_lo"], r["dslope_vs_hybrid_ci_hi"]
                num(k + ":dslope", s4(r["dslope_vs_hybrid"]))
                num(k + ":dslopeci", f"[{lo:.4f}, {hi:.4f}]".replace("-", "$-$"))
                num(k + ":dslopesig", "1" if (lo > 0 or hi < 0) else "0")
            if r["method"] == "hybrid-fresh":
                num(f"eff:fresh:{r['ds']}:gain", s4(r["dacc1_vs_hybrid"]))
                num(f"eff:fresh:{r['ds']}:gainci", f"[{r['dacc1_vs_hybrid_ci_lo']:.4f}, {r['dacc1_vs_hybrid_ci_hi']:.4f}]")
            if "fresh_minus_m_acc1" in r and not pd.isna(r.get("fresh_minus_m_acc1", np.nan)):
                lo, hi = r["fresh_minus_m_ci_lo"], r["fresh_minus_m_ci_hi"]
                num(f"eff:freshvs:{r['ds']}:{r['method']}", s4(r["fresh_minus_m_acc1"]))
                num(f"eff:freshvs:{r['ds']}:{r['method']}:ci", f"[{lo:.4f}, {hi:.4f}]".replace("-", "$-$"))
                num(f"eff:freshvs:{r['ds']}:{r['method']}:sig", "1" if (lo > 0 or hi < 0) else ("-1" if hi < 0 else "0"))
        f = drift[drift["method"] == "hybrid-fresh"]
        if len(f):
            num("eff:fresh:gain:min", s4(f["dacc1_vs_hybrid"].min()))
            num("eff:fresh:gain:max", s4(f["dacc1_vs_hybrid"].max()))
            num("eff:fresh:gain:npos", str(int((f["dacc1_vs_hybrid_ci_lo"] > 0).sum())))
    if e3 is not None:
        for _, r in e3.iterrows():
            k = f"eff:size:{r['ds']}:{r['family']}:{r['backbone']}"
            num(k + ":acc1", f4(r["acc1"]))
            num(k + ":perK", q2(r["gpus_per_1k"]))
            if r["once_gpu_h"] and r["once_gpu_h"] > 0:
                num(k + ":onceh", q2(r["once_gpu_h"]))
            if not pd.isna(r["peak_gpu_mem_gb_torch"]):
                num(k + ":memgb", q2(r["peak_gpu_mem_gb_torch"]))
        for fam, pk in (("ds-llm4poi-lcs", "qwen2.5-7b"), ("llm4poi", "qwen2.5-7b"), ("gnpr-sid", "llama3-8b")):
            g = e3[e3["family"] == fam]
            base = g[g["backbone"] == pk].set_index("ds")["gpus_per_1k"]
            for k, x in g.groupby("backbone"):
                if k == pk:
                    continue
                r = (x.set_index("ds")["gpus_per_1k"] / base).dropna()
                if len(r):
                    num(f"eff:size:{fam}:{k}:perKratio", q2(float(r.mean())))
                    num(f"eff:size:{fam}:{k}:ncity", str(len(r)))
    if ret is not None:
        for _, r in ret.iterrows():
            k = f"eff:ret:{r['ds']}"
            num(k + ":build", q2(r["build_s"]))
            num(k + ":p50ms", f"{r['lat_p50_ms']:.2f}")
            num(k + ":idxmb", q2(r["index_mb"]))
            for kk in (5, 10, 20, 50, 100):
                num(k + f":r{kk}", f4(r[f"recall@{kk}"]))
        m = ret[ret["ds"].isin(C.MAIN)]
        for kk in (5, 10, 20, 50, 100):
            num(f"eff:ret:r{kk}:min", f4(m[f"recall@{kk}"].min()))
            num(f"eff:ret:r{kk}:max", f4(m[f"recall@{kk}"].max()))
        num("eff:ret:build:max", q2(ret["build_s"].max()))
        num("eff:ret:p50ms:max", f"{ret['lat_p50_ms'].max():.2f}")
    for m, j in sv.items():
        k = f"eff:serve:{m}"
        num(k + ":p50ms", q2(j["batch1"]["p50_ms"]))
        num(k + ":p95ms", q2(j["batch1"]["p95_ms"]))
        num(k + ":load", q2(j.get("load_s")))
        for b, t in j["throughput"].items():
            num(k + f":qps{b}", q2(t["qps"]))
            num(k + f":b{b}p95ms", q2(t.get("batch_p95_ms")))


def table(rec, sl, hb, up, drift, sv):
    def slope(m, analysis, x, y):
        if sl is None:
            return None
        r = sl[(sl["method"] == m) & (sl["analysis"] == analysis) & (sl["x"] == x) & (sl["y"] == y)]
        return None if not len(r) or pd.isna(r["slope"].iloc[0]) else float(r["slope"].iloc[0])

    def dr(m):
        if drift is None:
            return mt.DASH
        cells = []
        for ds in ("sin", "istanbul"):
            r = drift[(drift["ds"] == ds) & (drift["method"] == m)]
            if not len(r) or m == "hybrid":
                cells.append(mt.DASH if m != "hybrid" else "0")
                continue
            r = r.iloc[0]
            sig = r["dslope_vs_hybrid_ci_lo"] > 0 or r["dslope_vs_hybrid_ci_hi"] < 0
            cells.append(s4(1000 * r["dslope_vs_hybrid"] / 1000) + (r"$^\dagger$" if sig else ""))
        return " / ".join(cells)

    L = [r"\scriptsize", r"\setlength{\tabcolsep}{2.4pt}", r"\begin{tabular}{@{}lrrrrr@{}}", r"\toprule",
         r" & \multicolumn{2}{c}{growth exponent} & \multicolumn{2}{c}{serving, batch 1 / 128} & \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
         r"method & train & query & p50 (ms) & q/s & refresh cost \\", r"\midrule"]
    for m in ORDER:
        if m == "hybrid":
            tr = slope(m, "across_datasets_measured(8)", "n_visible_train_checkins", "fit_s")
            qu = slope(m, "within_city_history(H-x)", "mean_visible_own_test", "perq_cpu_core_s")
            ref = (q2(up["insert_us_mean"].max()) + r"\,$\mu$s/ins.") if up is not None else mt.DASH
        else:
            tr = slope(m, "across_cities_H-100", "native_train_items", "once_gpu_s") if m != "ds-llm4poi-lcs" else None
            qu = slope(m, "within_city_history(H-x)", "mean_visible_own_test", "perq_gpu_s")
            if m == "ds-llm4poi-lcs":
                ref = "none"
            elif rec is not None:
                g = rec[(rec["method"] == m) & (rec["policy"] == "H-100") & rec["ds"].isin(C.MAIN)]["once_gpu_s"] / 3600
                ref = f"{q2(g.min())}--{q2(g.max())}\\,h" if len(g) else mt.DASH
            else:
                ref = mt.DASH
        s = sv.get(m)
        p50 = q2(s["batch1"]["p50_ms"]) if s else mt.DASH
        qps = q2(s["throughput"].get("128", {}).get("qps")) if s and "128" in s["throughput"] else mt.DASH
        trs = "--" if m == "ds-llm4poi-lcs" else f4(tr)
        L.append(" & ".join([LAB[m], trs, f4(qu), p50, qps, ref]) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    mt.TAB.mkdir(parents=True, exist_ok=True)
    (mt.TAB / "tab_eff.tex").write_text("% AUTO-GENERATED by scripts/efficiency/make_eff_paper.py; do not edit.\n"
                                       + "\n".join(L) + "\n")


def figure(rec, hb, fresh, sv):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 6.5, "axes.linewidth": 0.5, "pdf.fonttype": 42, "font.family": "serif"})
    fig, axs = plt.subplots(2, 2, figsize=(3.45, 3.25))
    (a, b), (c, d) = axs

    def style(ax, xl, yl, title):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, color="#e6e6e3", lw=0.4)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(length=2, width=0.4, labelsize=5.8, pad=1.5)
        ax.set_xlabel(xl, fontsize=6, labelpad=1)
        ax.set_ylabel(yl, fontsize=6, labelpad=1)
        ax.set_title(title, fontsize=6.5, pad=2.5)

    if rec is not None:
        for m in ("getnext", "graph-flashback", "llm4poi", "gnpr-sid", "ga-llm"):
            g = rec[(rec["method"] == m) & rec["ds"].isin(C.MAIN) & rec["native_train_items"].notna()]
            ok = g["colocated"].astype(str) != "True"
            a.scatter(g.loc[ok, "native_train_items"], g.loc[ok, "once_gpu_s"], s=9, marker=MK[m], color=COL[m],
                      edgecolors="white", linewidths=0.3, zorder=3)
            a.scatter(g.loc[~ok, "native_train_items"], g.loc[~ok, "once_gpu_s"], s=9, marker=MK[m], facecolors="none",
                      edgecolors=COL[m], linewidths=0.6, zorder=3)
    if hb is not None:
        h = hb[hb["policy"] == "H-100"]
        a.scatter(h["n_visible_train_checkins"], h["fit_s"], s=9, marker=MK["hybrid"], color=COL["hybrid"],
                  edgecolors="white", linewidths=0.3, zorder=3)
    style(a, "training items (Hybrid: check-ins)", "one-time compute (s)", "(a) build / train cost")
    if rec is not None:
        for m in ("hybrid", "getnext", "ds-llm4poi-lcs", "llm4poi", "gnpr-sid", "ga-llm"):
            g = rec[(rec["method"] == m) & rec["ds"].isin(C.MAIN) & rec["policy"].isin(["H-10", "H-70", "H-100"])]
            y = 1000 * (g["perq_cpu_core_s"] if m == "hybrid" else g["perq_gpu_s"])
            b.scatter(g["mean_visible_own_test"], y, s=9, marker=MK[m], color=COL[m], edgecolors="white",
                      linewidths=0.3, zorder=3)
    style(b, "visible check-ins per query", "ms per query", "(b) inference cost")
    for m in ORDER:
        s = sv.get(m)
        if not s:
            continue
        bs = sorted((int(k), v["qps"]) for k, v in s["throughput"].items() if k.isdigit())
        c.plot([x for x, _ in bs], [y for _, y in bs], "-", marker=MK[m], ms=3, lw=1.0, color=COL[m])
    style(c, "batch size", "queries / s", "(c) serving throughput (NYC)")
    from matplotlib.ticker import NullFormatter
    c.set_xticks([1, 8, 32, 128])
    c.set_xticklabels(["1", "8", "32", "128"])
    c.xaxis.set_minor_formatter(NullFormatter())
    c.yaxis.set_minor_formatter(NullFormatter())
    fp = rd("e6_fresh_pairs.csv")
    pairs = (("llm4poi-fresh", "llm4poi", "llm4poi", -0.2), ("ds-llm4poi-lcs-fresh", "ds-llm4poi-lcs", "ds-llm4poi-lcs", 0.0),
             ("hybrid-fresh", "hybrid", "hybrid", 0.2))
    for i, ds in enumerate(C.MAIN):
        for fa, fb, m, off in pairs:
            r = fp[(fp["ds"] == ds) & (fp["a"] == fa) & (fp["b"] == fb)] if fp is not None else []
            if not len(r):
                continue
            r = r.iloc[0]
            v, lo, hi = r["diff"], r["ci_lo"], r["ci_hi"]
            d.errorbar([i + off], [v], yerr=[[v - lo], [hi - v]], fmt=MK[m], ms=2.8, color=COL[m], ecolor=COL[m],
                       elinewidth=0.7, capsize=1.1, capthick=0.6, mew=0.7)
    d.axhline(0, color="#555555", lw=0.6)
    d.set_xticks(range(len(C.MAIN)))
    d.set_xticklabels([C.LABEL[x].replace("Sao Paulo", "SP").replace("Istanbul", "IST") for x in C.MAIN], fontsize=5.6)
    d.grid(True, axis="y", color="#e6e6e3", lw=0.4)
    d.set_axisbelow(True)
    for sp in ("top", "right"):
        d.spines[sp].set_visible(False)
    d.tick_params(length=2, width=0.4, labelsize=5.8, pad=1.5)
    d.set_ylabel("Acc@1 fresh $-$ frozen", fontsize=6, labelpad=1)
    d.set_title("(d) fresh evidence (no retraining)", fontsize=6.2, pad=2.5)
    from matplotlib.lines import Line2D
    present = [m for m in ORDER if m != "ga-llm" or (rec is not None and (rec["method"] == "ga-llm").any()) or sv.get(m)]
    hs = [Line2D([], [], color=COL[m], marker=MK[m], ms=3, lw=0.8) for m in present]
    fig.legend(hs, [LAB[m] for m in present], loc="upper center", ncol=4, frameon=False,
               fontsize=5.6, handlelength=1.3, columnspacing=0.8, handletextpad=0.3, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(pad=0.25, h_pad=0.5, w_pad=0.5, rect=(0, 0, 1, 0.9))
    mt.FIG.mkdir(exist_ok=True)
    fig.savefig(mt.FIG / "fig_eff.pdf")
    fig.savefig(C.SCRATCH / "fig_eff.png", dpi=220)
    plt.close(fig)


def main():
    rec, sl, hb = rd("e1_cost_records.csv"), rd("e1_slopes.csv"), rd("e1_hybrid_bench.csv")
    up, drift, wk = rd("e2_update_cost.csv"), rd("e2_drift.csv"), rd("e2_weekly_acc.csv")
    e3, ret = rd("e3_model_size.csv"), rd("e4_retriever.csv")
    sv = e5()
    numbers(rec, sl, hb, up, drift, e3, ret, sv)
    table(rec, sl, hb, up, drift, sv)
    figure(rec, hb, drift[drift["method"] == "hybrid-fresh"] if drift is not None else None, sv)
    import make_eff6
    make_eff6.run(num, COL, MK, LAB)
    import make_eff7
    make_eff7.run(num, COL, MK, LAB)
    L = ["% AUTO-GENERATED by scripts/efficiency/make_eff_paper.py; do not edit.",
         r"% \V{eff:...}: EXPLORATORY efficiency / scalability study (results/efficiency/, reports/EFFICIENCY.md)"]
    for k, v in sorted(NUM.items()):
        L.append(rf"\expandafter\def\csname V:{k}\endcsname{{{v}}}")
    (mt.TAB / "numbers_eff.tex").write_text("\n".join(L) + "\n")
    print(f"{len(NUM)} macros, tab_eff.tex, fig_eff.pdf; serving rows: {sorted(sv)}")


if __name__ == "__main__":
    main()
