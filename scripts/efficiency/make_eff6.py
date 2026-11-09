from __future__ import annotations

import numpy as np
import pandas as pd

import common as C

mt = C.mt
CX_ORDER = ("hybrid", "retriever", "getnext", "graph-flashback", "ds-llm4poi-lcs", "llm4poi", "gnpr-sid")
CANDK_M = ("hybrid", "getnext", "graph-flashback", "ds-llm4poi-lcs", "llm4poi", "gnpr-sid")
KS = (5, 10, 20, 50, 100)
TEXSYM = {
    "hybrid": dict(build=r"$O(N)$", query=r"$O(h + d + K\log K)$"),
    "retriever": dict(build=r"$O(N + V\log V)$ (100-NN)", query=r"$O(h + K\log K)$"),
    "getnext": dict(build=r"$O(E_p S L^2 D + E_p V^2 D)$", query=r"$O(L^2 D + V D)$"),
    "graph-flashback": dict(build=r"$O(E_k t_r D + E_p S L D^2)$", query=r"$O(L D^2 + V D)$"),
    "ds-llm4poi-lcs": dict(build="none (zero-shot)", query=r"$O(B L^2 + T P)$"),
    "llm4poi": dict(build=r"$O(E_p \textstyle\sum_S T\, P)$", query=r"$O(T P)$, $T$ capped"),
    "gnpr-sid": dict(build=r"$O(E_p V D + E_p \textstyle\sum_S T\, P)$", query=r"$O(B_m T P)$"),
}


def rd(name):
    p = C.OUT / name
    return pd.read_csv(p) if p.exists() else None


def f4(x):
    return mt.DASH if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.4f}".replace("-", "$-$")


def s4(x):
    return mt.sf4(x) if x is not None and not (isinstance(x, float) and np.isnan(x)) else mt.DASH


def q2(x):
    return mt.q2(x) if x is not None and not (isinstance(x, float) and np.isnan(x)) else mt.DASH


def ci(lo, hi):
    return f"[{lo:.4f}, {hi:.4f}]".replace("-", "$-$")


def rngs(a, b, unit=""):
    if a is None or (isinstance(a, float) and np.isnan(a)):
        return mt.DASH
    s = q2(a) if abs(a - b) < 1e-12 or q2(a) == q2(b) else f"{q2(a)}--{q2(b)}"
    return s + (f"\\,{unit}" if unit else "")


def numbers(num):
    cx = rd("e6_complexity.csv")
    if cx is not None:
        for _, r in cx.iterrows():
            k = f"eff:cx:{r['method']}"
            for a in ("build", "query", "update", "mem"):
                num(f"{k}:{a}:min", q2(r[f"{a}_min"]))
                num(f"{k}:{a}:max", q2(r[f"{a}_max"]))
            num(f"{k}:gpugb:max", q2(r["peak_gpu_gb_max"]))
    st = rd("e6_stages.csv")
    if st is not None:
        once = st[st["unit"] == "once"]
        for m, g in once.groupby("method"):
            tot = g.groupby("ds")["wall_s"].sum()
            for s_, x in g.groupby("stage"):
                share = x["wall_s"].to_numpy() / tot.loc[x["ds"]].to_numpy()
                num(f"eff:stage:{m}:{s_}:share", mt.pct(float(np.mean(share))))
                num(f"eff:stage:{m}:{s_}:gpuh:max", q2(float(x["gpu_s"].max() / 3600)))
                num(f"eff:stage:{m}:{s_}:cpus:max", q2(float(x["cpu_core_s"].max())))
        pq = st[(st["unit"] != "once") & (st["stage"] == "infer")]
        for m, g in pq.groupby("method"):
            num(f"eff:stage:{m}:infer:gpusperK:min", q2(float(g["gpu_s"].min())))
            num(f"eff:stage:{m}:infer:gpusperK:max", q2(float(g["gpu_s"].max())))
        pp = st[(st["unit"] != "once") & (st["stage"] == "prepare")]
        for m, g in pp.groupby("method"):
            num(f"eff:stage:{m}:prepare:cpusperK:max", q2(float(g["cpu_core_s"].max())))
    so = rd("e6_storage.csv")
    if so is not None:
        for m, g in so.groupby("method"):
            per = g.groupby("ds")["disk_mb"].sum()
            num(f"eff:store:{m}:mb:min", q2(float(per.min())))
            num(f"eff:store:{m}:mb:max", q2(float(per.max())))
            num(f"eff:store:{m}:gb:max", q2(float(per.max() / 1024)))
    hb = rd("e6_history_bins.csv")
    if hb is not None:
        for c in ("hybrid_core_ms_p50", "hybrid_call_ms_p50", "retriever_ms_p50"):
            lo = hb[hb["bin"] == "1-9"][c]
            hi = hb[hb["bin"] == ">=1000"][c]
            num(f"eff:hist:{c}:short:max", f"{lo.max():.2f}")
            num(f"eff:hist:{c}:long:max", f"{hi.max():.2f}")
        for m in ("ds-llm4poi-lcs", "llm4poi"):
            c = f"{m}_prompt_tokens"
            if c in hb:
                a = hb[hb["bin"] == "1-9"].set_index("ds")[c]
                b = hb[hb["bin"] == ">=1000"].set_index("ds")[c]
                rr = (b / a).dropna()
                num(f"eff:hist:{m}:tok:short:mean", mt.intc(round(float(a.mean()))))
                num(f"eff:hist:{m}:tok:long:mean", mt.intc(round(float(b.mean()))))
                num(f"eff:hist:{m}:tok:ratio:min", q2(float(rr.min())))
                num(f"eff:hist:{m}:tok:ratio:max", q2(float(rr.max())))
                e = f"{m}_gpu_ms_est"
                num(f"eff:hist:{m}:gpums:long:max", q2(float(hb[hb['bin'] == '>=1000'][e].max())))
                num(f"eff:hist:{m}:gpums:short:min", q2(float(hb[hb['bin'] == '1-9'][e].min())))
        sh = hb[hb["bin"] == ">=1000"].set_index("ds")["share_queries"]
        num("eff:hist:longshare:max", mt.pct(float(sh.max())))
    sl = rd("e6_history_slopes.csv")
    if sl is not None:
        for _, r in sl.iterrows():
            num(f"eff:hist:slope:{r['y']}", f4(r["slope"]))
    ck = rd("e6_cand_k.csv")
    if ck is not None:
        for _, r in ck.iterrows():
            k = f"eff:candk:{r['ds']}:{r['method']}:k{int(r['k'])}"
            num(k, f4(r["acc@1"]))
            if int(r["k"]) in (5, 100) and r["method"] != "retriever":
                num(k + ":d20", s4(r["dacc1_vs_k20"]))
                num(k + ":d20ci", ci(r["dacc1_vs_k20_ci_lo"], r["dacc1_vs_k20_ci_hi"]))
        for kk in KS:
            x = ck[(ck["k"] == kk) & (ck["method"] == "hybrid")]
            num(f"eff:candk:recall:k{kk}:mean", f4(float(x["recall@k"].mean())))
        for m, g in ck.groupby("method"):
            for kk in KS:
                x = g[g["k"] == kk]
                num(f"eff:candk:{m}:k{kk}:mean", f4(float(x["acc@1"].mean())))
                num(f"eff:candk:{m}:k{kk}:share:mean", mt.pct(float(x["share_of_ceiling"].mean())))
            x = g[g["k"] == 100]
            num(f"eff:candk:{m}:d100:min", s4(float(x["dacc1_vs_k20"].min())))
            num(f"eff:candk:{m}:d100:max", s4(float(x["dacc1_vs_k20"].max())))
            num(f"eff:candk:{m}:d100:npos", str(int((x["dacc1_vs_k20_ci_lo"] > 0).sum())))
            num(f"eff:candk:{m}:d100:nneg", str(int((x["dacc1_vs_k20_ci_hi"] < 0).sum())))
            x = g[g["k"] == 5]
            num(f"eff:candk:{m}:d5:min", s4(float(x["dacc1_vs_k20"].min())))
            num(f"eff:candk:{m}:d5:max", s4(float(x["dacc1_vs_k20"].max())))
    bg = rd("e6_bigds.csv")
    if bg is not None:
        for _, r in bg.iterrows():
            k = f"eff:big:{r['ds']}:{r['method']}"
            for met in ("acc@1", "acc@5", "ndcg@10", "mrr"):
                num(f"{k}:{met}", f4(r[f"e2e:{met}"]))
            num(f"{k}:acc@1:ci", ci(r["e2e:acc@1_ci_lo"], r["e2e:acc@1_ci_hi"]))
            if r["method"] != "hybrid" and not pd.isna(r.get("e2e:dacc1_vs_hybrid", np.nan)):
                num(f"{k}:dhyb", s4(r["e2e:dacc1_vs_hybrid"]))
                num(f"{k}:dhybci", ci(r["e2e:dacc1_vs_hybrid_ci_lo"], r["e2e:dacc1_vs_hybrid_ci_hi"]))


def gallm_complexity_row():
    import json as _json
    once, perq, peak = [], [], []
    for ds in C.MAIN:
        c = mt.load_cost("ga-llm", "H-100", ds)
        if not c:
            continue
        once.append(c["once"]["gpu_s"] / 3600)
        q = (c["perq"].get("test") or {}).get("gpu_s")
        if q:
            perq.append(1000 * q)
        if c.get("peak_gpu_mem"):
            peak.append(c["peak_gpu_mem"])
    b = rngs(min(once), max(once), "GPU-h") if once else mt.DASH
    qc = rngs(min(perq), max(perq), "GPU-ms") if perq else mt.DASH
    sp = C.OUT / "e5" / "ga-llm.json"
    if sp.exists():
        j = _json.load(open(sp))
        if not j.get("mock") and (j.get("batch1") or {}).get("p50_ms"):
            qc += f"; {q2(j['batch1']['p50_ms'])}\\,ms b1"
    up = ("retrain " + rngs(min(once), max(once), "GPU-h")) if once else mt.DASH
    mem = mt.DASH + "; " + (q2(max(peak)) + r"\,GB" if peak else mt.DASH)
    return " & ".join(["GA-LLM", r"$O(2E_p \textstyle\sum_S T\, P)$", b, r"$O(T P)$", qc, up, mem]) + r" \\"


def tab_complexity():
    cx = rd("e6_complexity.csv")
    if cx is None:
        return
    cx = cx.set_index("method")

    L = [r"\scriptsize", r"\setlength{\tabcolsep}{2.4pt}", r"\begin{tabular}{@{}lllllll@{}}", r"\toprule",
         r"method & build & build cost & query & query cost & update (cost) & memory (disk; peak GPU) \\",
         r"\midrule"]
    unit = {"CPU s": "s", "GPU h": "GPU-h", "CPU ms": "ms", "GPU ms": "GPU-ms"}
    for m in CX_ORDER:
        if m not in cx.index:
            continue
        r = cx.loc[m]

        def sym(k):
            return TEXSYM[m][k]
        b = rngs(r["build_min"], r["build_max"], unit.get(r["build_unit"], ""))
        if m == "ds-llm4poi-lcs":
            b = "load " + rngs(r["build_min"] * 3600, r["build_max"] * 3600, "s")
        qc = rngs(r["query_min"], r["query_max"], unit.get(r["query_unit"], ""))
        if not pd.isna(r.get("serve_p50_ms_nyc", np.nan)):
            qc += f"; {q2(r['serve_p50_ms_nyc'])}\\,ms b1"
        if r["update_unit"] == "us/insert":
            up = rngs(r["update_min"], r["update_max"], r"$\mu$s/insert")
        elif r["update_unit"] == "none":
            up = "none"
        elif r["update_unit"] == "CPU s rebuild":
            up = "rebuild " + rngs(r["update_min"], r["update_max"], "s")
        else:
            up = "retrain " + rngs(r["update_min"], r["update_max"], "GPU-h")
        if r["mem_unit"].startswith("MB") and r["mem_max"] >= 1024:
            mem = rngs(r["mem_min"] / 1024, r["mem_max"] / 1024, "GB")
        else:
            mem = rngs(r["mem_min"], r["mem_max"], "MB")
        mem += "; " + (q2(r["peak_gpu_gb_max"]) + r"\,GB" if r["peak_gpu_gb_max"] > 0 else "CPU")
        L.append(" & ".join([C.MLABEL[m], sym("build"), b, sym("query"), qc, up, mem]) + r" \\")
    L.append(gallm_complexity_row())
    L += [r"\bottomrule", r"\end{tabular}"]
    (mt.TAB / "tab_complexity.tex").write_text("% AUTO-GENERATED by scripts/efficiency/make_eff6.py "
                                              "(via make_eff_paper.py); do not edit.\n" + "\n".join(L) + "\n")


def tab_bigds():
    bg = rd("e6_bigds.csv")
    if bg is None:
        return
    lab = {"hybrid": "Hybrid", "markov": "Markov", "repeat": "Repeat", "pop": "Pop", "retriever": "Retriever (R@50 list)"}
    dss = [d for d in C.BIG if d in set(bg["ds"])]
    L = [r"\scriptsize", r"\setlength{\tabcolsep}{2.4pt}",
         r"\begin{tabular}{@{}l" + "rrrr" * len(dss) + "@{}}", r"\toprule",
         " & ".join([""] + [rf"\multicolumn{{4}}{{c}}{{{C.LABEL[d]} ({mt.intc(bg[bg.ds == d]['n_queries'].iloc[0])} q.)}}"
                            for d in dss]) + r" \\",
         "".join(rf"\cmidrule(lr){{{2 + 4 * i}-{5 + 4 * i}}}" for i in range(len(dss))),
         " & ".join(["method"] + ["Acc@1", "Acc@5", "NDCG@10", "MRR"] * len(dss)) + r" \\", r"\midrule"]
    for m in ("hybrid", "markov", "repeat", "retriever", "pop"):
        cells = []
        for d in dss:
            r = bg[(bg.ds == d) & (bg.method == m)]
            if not len(r):
                cells += [mt.DASH] * 4
                continue
            r = r.iloc[0]
            best = {met: bg[bg.ds == d][f"e2e:{met}"].max() for met in ("acc@1", "acc@5", "ndcg@10", "mrr")}
            for met in ("acc@1", "acc@5", "ndcg@10", "mrr"):
                v = r[f"e2e:{met}"]
                cells.append(mt.f4(v, bold=round(v, 4) == round(best[met], 4)))
        L.append(" & ".join([lab[m]] + cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (mt.TAB / "tab_bigds.tex").write_text("% AUTO-GENERATED by scripts/efficiency/make_eff6.py "
                                         "(via make_eff_paper.py); do not edit.\n" + "\n".join(L) + "\n")


def figure(COL, MK, LAB):
    ck, hb = rd("e6_cand_k.csv"), rd("e6_history_bins.csv")
    if ck is None and hb is None:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({"font.size": 6.5, "axes.linewidth": 0.5, "pdf.fonttype": 42, "font.family": "serif"})
    fig, (a, b) = plt.subplots(1, 2, figsize=(3.45, 1.75))

    def style(ax, xl, yl, title, logy=False):
        ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")
        ax.grid(True, color="#e6e6e3", lw=0.4)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(length=2, width=0.4, labelsize=5.8, pad=1.5)
        ax.set_xlabel(xl, fontsize=6, labelpad=1)
        ax.set_ylabel(yl, fontsize=6, labelpad=1)
        ax.set_title(title, fontsize=6.5, pad=2.5)

    from matplotlib.ticker import NullFormatter
    if ck is not None:
        rec = ck[ck["method"] == "hybrid"].groupby("k")["recall@k"].mean()
        a.plot(rec.index, rec.values, ls="--", color="#555555", lw=0.9, marker="", zorder=2)
        a.annotate("recall@k ceiling", (rec.index[-2], rec.values[-2]), xytext=(-2, 4), textcoords="offset points",
                   ha="right", fontsize=5.4, color="#555555")
        for m in CANDK_M:
            g = ck[ck["method"] == m].groupby("k")["acc@1"].mean()
            if len(g):
                a.plot(g.index, g.values, "-", marker=MK[m], ms=2.6, lw=0.9, color=COL[m], zorder=3)
        style(a, "candidates k", "candidate-mode Acc@1", "(a) accuracy vs. candidate-set size")
        a.set_xticks(list(KS))
        a.set_xticklabels([str(k) for k in KS])
        a.xaxis.set_minor_formatter(NullFormatter())
        a.set_ylim(0, None)
    if hb is not None:
        order = ["1-9", "10-29", "30-99", "100-299", "300-999", ">=1000"]
        g = hb.groupby("bin").mean(numeric_only=True).reindex(order).dropna(how="all")
        x = g["mean_n_visible_own"]
        b.plot(x, g["hybrid_core_ms_p50"], "-", marker=MK["hybrid"], ms=2.6, lw=0.9, color=COL["hybrid"])
        b.plot(x, g["retriever_ms_p50"], ":", marker="o", ms=2.6, lw=0.9, color="#7a7a7a", mfc="white")
        for m in ("ds-llm4poi-lcs", "llm4poi"):
            c = f"{m}_gpu_ms_est"
            if c in g:
                b.plot(x, g[c], "--", marker=MK[m], ms=2.6, lw=0.9, color=COL[m])
        style(b, "visible check-ins of the user", "ms per query", "(b) per-query cost vs. history", logy=True)
    hs = [Line2D([], [], color=COL[m], marker=MK[m], ms=2.6, lw=0.8) for m in CANDK_M]
    hs.append(Line2D([], [], color="#7a7a7a", marker="o", mfc="white", ms=2.6, lw=0.8, ls=":"))
    fig.legend(hs, [LAB[m] for m in CANDK_M] + ["Retriever"], loc="upper center", ncol=4, frameon=False,
               fontsize=5.4, handlelength=1.3, columnspacing=0.8, handletextpad=0.3, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(pad=0.25, w_pad=0.6, rect=(0, 0, 1, 0.8))
    fig.savefig(mt.FIG / "fig_eff2.pdf")
    p = C.ROOT / "work" / "efficiency2"
    p.mkdir(parents=True, exist_ok=True)
    fig.savefig(p / "fig_eff2.png", dpi=220)
    plt.close(fig)


def run(num, COL, MK, LAB):
    numbers(num)
    tab_complexity()
    tab_bigds()
    figure(COL, MK, LAB)
