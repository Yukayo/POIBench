from __future__ import annotations

import numpy as np
import pandas as pd

import common as C

mt = C.mt
SC_M = ("hybrid", "retriever", "getnext", "graph-flashback")
FRESH_PAIRS = (("llm4poi-fresh", "llm4poi"), ("ds-llm4poi-lcs-fresh", "ds-llm4poi-lcs"), ("hybrid-fresh", "hybrid"))
SHORT = {"llm4poi-fresh": "llm4poi", "ds-llm4poi-lcs-fresh": "ds", "hybrid-fresh": "hybrid"}
CITY = {"nyc": "NYC", "tky": "TKY", "ca": "CA", "sin": "SIN", "istanbul": "IST", "sao_paulo": "SP"}


def rd(name):
    p = C.OUT / name
    return pd.read_csv(p) if p.exists() else None


def ok(x):
    return x is not None and not (isinstance(x, float) and np.isnan(x))


def f4(x):
    return f"{x:.4f}".replace("-", "$-$") if ok(x) else mt.DASH


def s4(x):
    return mt.sf4(x) if ok(x) else mt.DASH


def q2(x):
    return mt.q2(x) if ok(x) else mt.DASH


def ci(lo, hi):
    return f"[{lo:.4f}, {hi:.4f}]".replace("-", "$-$")


def sig(lo, hi):
    return "1" if lo > 0 else ("-1" if hi < 0 else "0")


def numbers(num):
    fl, fp, fx, fc = rd("e6_fresh_llm.csv"), rd("e6_fresh_pairs.csv"), rd("e6_fresh_export.csv"), rd("e6_fresh_canary.csv")
    if fc is not None:
        num("eff:freshllm:canary:npass", str(int((fc["result"] == "PASS").sum())))
        num("eff:freshllm:canary:n", str(int(fc["n_queries"].iloc[0])))
        num("eff:freshllm:canary:leakyfail", str(int((fc["leaky_control_pred_mismatch"] > 0).sum())))
    if fx is not None:
        for _, r in fx.iterrows():
            num(f"eff:freshllm:{r['ds']}:visfrozen", q2(r["mean_visible_own_frozen"]))
            num(f"eff:freshllm:{r['ds']}:visfresh", q2(r["mean_visible_own_fresh"]))
            num(f"eff:freshllm:{r['ds']}:shareadded", f"{100 * r['share_queries_with_added_rows']:.2f}")
        num("eff:freshllm:shareadded:min", f"{100 * fx['share_queries_with_added_rows'].min():.2f}")
        num("eff:freshllm:shareadded:max", f"{100 * fx['share_queries_with_added_rows'].max():.2f}")
    if fl is not None:
        for _, r in fl.iterrows():
            k = f"eff:freshllm:{r['ds']}:{r['method']}"
            num(k + ":acc1", f4(r["acc@1"]))
            for c, n in (("gpu_ms_per_generated", "gpums"), ("prompt_tokens_mean", "tok")):
                if c in r and ok(r[c]):
                    num(k + f":{n}", q2(r[c]))
            for c, n in (("repro_identical", "repro"), ("share_prompt_identical_to_frozen", "sameprompt")):
                if c in r and ok(r[c]):
                    num(k + f":{n}", f"{100 * r[c]:.2f}")
        for m in ("llm4poi-fresh", "ds-llm4poi-lcs-fresh"):
            g = fl[fl["method"] == m]
            if len(g) and "repro_identical" in g:
                num(f"eff:freshllm:{SHORT[m]}:repro:min", f"{100 * g['repro_identical'].min():.2f}")
    if fp is not None:
        for _, r in fp.iterrows():
            k = f"eff:freshllm:{r['ds']}:{r['a']}:vs:{r['b']}"
            num(k, s4(r["diff"]))
            num(k + ":ci", ci(r["ci_lo"], r["ci_hi"]))
            num(k + ":sig", sig(r["ci_lo"], r["ci_hi"]))
        for a, b in FRESH_PAIRS + (("llm4poi-fresh", "hybrid-fresh"), ("ds-llm4poi-lcs-fresh", "hybrid-fresh")):
            g = fp[(fp["a"] == a) & (fp["b"] == b)]
            if len(g):
                k = f"eff:freshllm:{a}:vs:{b}"
                num(k + ":min", s4(g["diff"].min()))
                num(k + ":max", s4(g["diff"].max()))
                num(k + ":npos", str(int((g["ci_lo"] > 0).sum())))
                num(k + ":nneg", str(int((g["ci_hi"] < 0).sum())))
                num(k + ":ncity", str(len(g)))
    sc, sl, sz = rd("e7_scale.csv"), rd("e7_scale_slopes.csv"), rd("e7_scale_sizes.csv")
    if sz is not None:
        for _, r in sz.iterrows():
            k = f"eff:scale:size:{int(r['frac'])}"
            num(k + ":users", mt.intc(int(r["n_users"])))
            num(k + ":ck", mt.intc(int(r["n_checkins"])))
            num(k + ":pois", mt.intc(int(r["n_catalog"])))
            num(k + ":edges", mt.intc(int(r["graph_edges"])))
            num(k + ":trainq", mt.intc(int(r["n_train_queries"])))
    if sc is not None:
        for _, r in sc[sc["method"].isin(SC_M)].iterrows():
            k = f"eff:scale:{r['method']}:{int(r['frac'])}"
            num(k + ":once", q2(r["once_wall_s"]))
            if ok(r.get("once_gpu_s")) and r["once_gpu_s"] > 0:
                num(k + ":gpuh", q2(r["once_gpu_s"] / 3600))
            num(k + ":mem", q2(r.get("peak_host_gb")))
            if ok(r.get("peak_gpu_gb_torch")) and r["peak_gpu_gb_torch"] > 0:
                num(k + ":gpumem", q2(r["peak_gpu_gb_torch"]))
            num(k + ":model", q2(r.get("model_mb")))
            num(k + ":qms", f"{r['query_ms']:.4f}" if ok(r.get("query_ms")) else mt.DASH)
            if ok(r.get("acc1")):
                num(k + ":acc1", f4(r["acc1"]))
                num(k + ":acc1ci", ci(r["acc1_ci_lo"], r["acc1_ci_hi"]))
            if ok(r.get("graph_nnz")):
                num(k + ":graphnnz", mt.intc(int(r["graph_nnz"])))
    if sl is not None:
        for _, r in sl[(sl["x"] == "n_checkins") & sl["method"].isin(SC_M) & (sl["n"] == 5)].iterrows():
            num(f"eff:scale:{r['method']}:slope:{r['y']}", f4(r["slope"]))
            if ok(r.get("ratio_100_over_20")):
                num(f"eff:scale:{r['method']}:ratio:{r['y']}", q2(r["ratio_100_over_20"]))


def tab_scale(LAB):
    sc, sl = rd("e7_scale.csv"), rd("e7_scale_slopes.csv")
    if sc is None:
        return

    def at(m, f, c):
        r = sc[(sc["method"] == m) & (sc["frac"] == f)]
        return float(r[c].iloc[0]) if len(r) and c in r and ok(r[c].iloc[0]) else None

    def slope(m, y):
        if sl is None:
            return None
        r = sl[(sl["method"] == m) & (sl["x"] == "n_checkins") & (sl["y"] == y) & (sl["n"] == 5)]
        return float(r["slope"].iloc[0]) if len(r) else None

    def pair(a, b, fmt=q2):
        return f"{fmt(a)}/{fmt(b)}" if ok(a) or ok(b) else mt.DASH

    lab = dict(LAB, **{"retriever": "Retriever"})
    L = [r"\scriptsize", r"\setlength{\tabcolsep}{2.0pt}", r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
         r" & \multicolumn{2}{c}{build / train} & mem. & model & query & Acc@1 \\",
         r"\cmidrule(lr){2-3}",
         r"method & s, 20\%/100\% & exp. & GB & MB & ms & 20\%/100\% \\", r"\midrule"]
    for m in SC_M:
        if not (sc["method"] == m).any():
            continue
        mem = "peak_gpu_gb_torch" if m in ("getnext", "graph-flashback") else "peak_host_gb"
        L.append(" & ".join([
            lab.get(m, m), pair(at(m, 20, "once_wall_s"), at(m, 100, "once_wall_s")), f4(slope(m, "once_wall_s")),
            pair(at(m, 20, mem), at(m, 100, mem)), pair(at(m, 20, "model_mb"), at(m, 100, "model_mb")),
            pair(at(m, 20, "query_ms"), at(m, 100, "query_ms"), lambda x: f"{x:.3f}" if ok(x) else mt.DASH),
            pair(at(m, 20, "acc1"), at(m, 100, "acc1"), f4)]) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (mt.TAB / "tab_scale.tex").write_text("% AUTO-GENERATED by scripts/efficiency/make_eff7.py; do not edit.\n"
                                         + "\n".join(L) + "\n")


def tab_freshllm():
    fl, fp = rd("e6_fresh_llm.csv"), rd("e6_fresh_pairs.csv")
    if fl is None or fp is None or not fl["method"].isin(["llm4poi-fresh", "ds-llm4poi-lcs-fresh"]).any():
        return

    def acc(ds, m):
        r = fl[(fl["ds"] == ds) & (fl["method"] == m)]
        return float(r["acc@1"].iloc[0]) if len(r) else None

    def gain(ds, a, b):
        r = fp[(fp["ds"] == ds) & (fp["a"] == a) & (fp["b"] == b)]
        if not len(r):
            return mt.DASH
        r = r.iloc[0]
        return s4(r["diff"]) + (r"$^\dagger$" if (r["ci_lo"] > 0 or r["ci_hi"] < 0) else "")

    L = [r"\scriptsize", r"\setlength{\tabcolsep}{2.2pt}", r"\begin{tabular}{@{}lrrrrrrr@{}}", r"\toprule",
         r" & \multicolumn{2}{c}{LLM4POI} & \multicolumn{2}{c}{DS-LLM4POI} & \multicolumn{2}{c}{Hybrid} & LLM4POI \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
         r"city & fresh & gain & fresh & gain & fresh & gain & $-$Hyb.\,fr. \\", r"\midrule"]
    for ds in C.MAIN:
        L.append(" & ".join([C.LABEL[ds], f4(acc(ds, "llm4poi-fresh")), gain(ds, "llm4poi-fresh", "llm4poi"),
                             f4(acc(ds, "ds-llm4poi-lcs-fresh")), gain(ds, "ds-llm4poi-lcs-fresh", "ds-llm4poi-lcs"),
                             f4(acc(ds, "hybrid-fresh")), gain(ds, "hybrid-fresh", "hybrid"),
                             gain(ds, "llm4poi-fresh", "hybrid-fresh")]) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (mt.TAB / "tab_freshllm.tex").write_text("% AUTO-GENERATED by scripts/efficiency/make_eff7.py; do not edit.\n"
                                            + "\n".join(L) + "\n")


def figure(COL, MK, LAB):
    sc, fp = rd("e7_scale.csv"), rd("e6_fresh_pairs.csv")
    if sc is None and fp is None:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({"font.size": 6.5, "axes.linewidth": 0.5, "pdf.fonttype": 42, "font.family": "serif"})
    fig, axs = plt.subplots(2, 2, figsize=(3.45, 3.25))
    (a, b), (c, d) = axs
    col = dict(COL, **{"retriever": "#7a7a7a"})
    mk = dict(MK, **{"retriever": "o"})
    lab = dict(LAB, **{"retriever": "Retriever"})

    from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, NullFormatter, NullLocator
    xt = sorted(sc["n_checkins"].unique()) if sc is not None else []

    def style(ax, xl, yl, title, logx=True):
        if logx:
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.xaxis.set_major_locator(FixedLocator(xt))
            ax.xaxis.set_minor_locator(NullLocator())
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: "" if len(xt) == 5 and v == xt[3] else f"{v / 1e3:.0f}k"))
            ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            ax.yaxis.set_minor_formatter(NullFormatter())
        ax.grid(True, color="#e6e6e3", lw=0.4)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(length=2, width=0.4, labelsize=5.8, pad=1.5)
        ax.set_xlabel(xl, fontsize=6, labelpad=1)
        ax.set_ylabel(yl, fontsize=6, labelpad=1)
        ax.set_title(title, fontsize=6.5, pad=2.5)

    if sc is not None:
        for m in SC_M:
            g = sc[sc["method"] == m].sort_values("n_checkins")
            if not len(g):
                continue
            kw = dict(marker=mk[m], ms=2.8, lw=0.9, color=col[m], mfc=("white" if m == "retriever" else col[m]),
                      ls=(":" if m == "retriever" else "-"))
            a.plot(g["n_checkins"], g["once_wall_s"], **kw)
            b.plot(g["n_checkins"], g["query_ms"], **kw)
            if g["model_mb"].notna().any():
                c.plot(g["n_checkins"], g["model_mb"], **kw)
        style(a, "check-ins (TKY, 20-100% of users)", "build / train time (s)", "(a) one-time cost")
        a.yaxis.set_major_locator(LogLocator(base=10))
        style(b, "check-ins (TKY, 20-100% of users)", "ms per query", "(b) inference latency")
        style(c, "check-ins (TKY, 20-100% of users)", "model / state size (MB)", "(c) space")
    ck = rd("e6_cand_k.csv")
    dm = ("hybrid", "getnext", "llm4poi", "gnpr-sid", "ds-llm4poi-lcs")
    if ck is not None:
        from matplotlib.ticker import NullFormatter as _NF
        rc = ck[ck["method"] == "hybrid"].groupby("k")["recall@k"].mean()
        d.plot(rc.index, rc.values, ls="--", color="#555555", lw=0.9, zorder=2)
        d.annotate("recall@k ceiling", (rc.index[-2], rc.values[-2]), xytext=(-2, 4), textcoords="offset points",
                   ha="right", fontsize=5.4, color="#555555")
        for m in dm:
            g = ck[ck["method"] == m].groupby("k")["acc@1"].mean()
            if len(g):
                d.plot(g.index, g.values, "-", marker=mk[m], ms=2.6, lw=0.9, color=col[m], zorder=3)
        style(d, "candidates k", "candidate-mode Acc@1", "(d) candidate-set size (6 cities)", logx=False)
        d.set_xscale("log")
        ks = sorted(int(k) for k in rc.index)
        d.set_xticks(ks)
        d.set_xticklabels([str(k) for k in ks])
        d.xaxis.set_minor_formatter(_NF())
        d.set_ylim(0, None)
    ms = [m for m in SC_M if sc is not None and (sc["method"] == m).any()]
    hs = [Line2D([], [], color=col[m], marker=mk[m], ms=2.8, lw=0.8, mfc=("white" if m == "retriever" else col[m]),
                 ls=(":" if m == "retriever" else "-")) for m in ms]
    names = [lab[m] for m in ms]
    for m in ("llm4poi", "gnpr-sid", "ds-llm4poi-lcs"):
        hs.append(Line2D([], [], color=col[m], marker=MK[m], ms=2.8, lw=0.8))
        names.append(lab[m] + " (d)")
    fig.legend(hs, names, loc="upper center", ncol=4, frameon=False, fontsize=5.6, handlelength=1.3,
               columnspacing=0.8, handletextpad=0.3, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(pad=0.25, h_pad=0.5, w_pad=0.5, rect=(0, 0, 1, 0.88))
    fig.savefig(mt.FIG / "fig_eff3.pdf")
    p = C.ROOT / "work" / "efficiency2"
    p.mkdir(parents=True, exist_ok=True)
    fig.savefig(p / "fig_eff3.png", dpi=220)
    plt.close(fig)


def run(num, COL, MK, LAB):
    numbers(num)
    tab_scale(LAB)
    tab_freshllm()
    figure(COL, MK, LAB)
