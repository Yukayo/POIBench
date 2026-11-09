#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_tables as mt

ROOT, TAB, FIG = mt.ROOT, mt.TAB, mt.FIG
DS = ("nyc", "tky", "ca", "sin", "istanbul", "sao_paulo")
LABEL = {"nyc": "NYC", "tky": "TKY", "ca": "CA", "sin": "SIN", "istanbul": "Istanbul", "sao_paulo": r"S\~ao Paulo"}
METRICS = ("acc@1", "acc@5", "acc@10", "ndcg@10")
PAIRS = (
    ("hybrid-tt", "hybrid"),
    ("llm4poi", "hybrid"), ("gnpr-sid", "hybrid"), ("ds-llm4poi-lcs", "hybrid"),
    ("llm4poi-tt", "hybrid-tt"), ("gnpr-sid-tt", "hybrid-tt"), ("ds-llm4poi-lcs-tt", "hybrid-tt"),
    ("llm4poi-tt", "llm4poi"), ("gnpr-sid-tt", "gnpr-sid"), ("ds-llm4poi-lcs-tt", "ds-llm4poi-lcs"),
    ("llm4poi-tt", "hybrid"),
    ("llm4poi+hybridtail", "hybrid"), ("ds-llm4poi-lcs+hybridtail", "hybrid"),
    ("llm4poi-tt+hybridtttail", "hybrid-tt"), ("ds-llm4poi-lcs-tt+hybridtttail", "hybrid-tt"),
    ("getnext-zfeat", "hybrid"), ("getnext-zfeat", "getnext"),
    ("llm4poi", "getnext-zfeat"), ("gnpr-sid", "getnext-zfeat"),
    ("ga-llm", "hybrid"), ("ga-llm", "llm4poi"), ("ga-llm", "gnpr-sid"), ("ga-llm", "getnext-zfeat"),
    ("ga-llm-tt", "hybrid-tt"), ("ga-llm-tt", "ga-llm"), ("ga-llm-tt", "llm4poi-tt"),
)
ROWS = (
    ("hybrid", "Hybrid", "unknown"), ("ds-llm4poi-lcs", "DS-LLM4POI", "unknown"), ("llm4poi", "LLM4POI", "unknown"),
    ("gnpr-sid", "GNPR-SID", "unknown"), ("ga-llm", "GA-LLM", "unknown"),
    ("hybrid-tt", "Hybrid-TT", "given"), ("ds-llm4poi-lcs-tt", "DS-LLM4POI-TT", "given"),
    ("llm4poi-tt", "LLM4POI-TT", "given"), ("gnpr-sid-tt", "GNPR-SID-TT", "given"), ("ga-llm-tt", "GA-LLM-TT", "given"),
)
WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}
NUM = {}


def load(ds):
    base = ROOT / "results" / ds / "test" / "exploratory_timeaware"
    if not Path(str(base) + ".csv").exists():
        return None, None
    res = pd.read_csv(str(base) + ".csv")
    res = res[(res["mode"] == "e2e") & (res["stratum"] == "all")]
    pr = pd.read_csv(str(base) + ".pairs.csv")
    pr = pr[(pr["mode"] == "e2e") & (pr["stratum"] == "all")]
    return res, pr


def value(res, m, metric):
    if res is None:
        return None
    r = res[(res["method"] == f"{m}@H-100") & (res["metric"] == metric)]
    return float(r["value"].iloc[0]) if len(r) else None


def pair(pr, a, b, metric):
    if pr is None:
        return None
    A, B = f"{a}@H-100", f"{b}@H-100"
    r = pr[(pr["a"] == A) & (pr["b"] == B) & (pr["metric"] == metric)]
    if len(r):
        r = r.iloc[0]
        return float(r["diff"]), float(r["ci_lo"]), float(r["ci_hi"])
    r = pr[(pr["a"] == B) & (pr["b"] == A) & (pr["metric"] == metric)]
    if len(r):
        r = r.iloc[0]
        return -float(r["diff"]), -float(r["ci_hi"]), -float(r["ci_lo"])
    return None


def dropped_shares():
    import json
    for ds in DS:
        f = ROOT / "data" / ds / "meta.json"
        if f.exists():
            st = json.load(open(f))["preprocess_stats"]
            NUM[f"ds:{ds}:dropshare"] = mt.pct(st["test_dropped_unseen_user_or_venue"] / st["test_candidate_targets"])


def collect():
    dropped_shares()
    D = {ds: load(ds) for ds in DS}
    for ds, (res, pr) in D.items():
        for m in {x for p in PAIRS for x in p} | {k for k, _, _ in ROWS}:
            for met in METRICS:
                v = value(res, m, met)
                if v is not None:
                    NUM[f"xv:{ds}:{m}:{met}"] = f"{v:.4f}"
    for a, b in PAIRS:
        for met in METRICS:
            got = []
            for ds in DS:
                p = pair(D[ds][1], a, b, met)
                if p is None:
                    continue
                d, lo, hi = p
                NUM[f"x:{a}:{b}:{ds}:{met}:diff"] = mt.sf4(d)
                NUM[f"x:{a}:{b}:{ds}:{met}:ci"] = f"[{lo:.4f}, {hi:.4f}]"
                NUM[f"x:{a}:{b}:{ds}:{met}:sig"] = "1" if (lo > 0 or hi < 0) else "0"
                got.append((d, lo, hi))
            if got:
                ds_ = [g[0] for g in got]
                NUM[f"x:{a}:{b}:{met}:min"] = mt.sf4(min(ds_))
                NUM[f"x:{a}:{b}:{met}:max"] = mt.sf4(max(ds_))
                NUM[f"x:{a}:{b}:{met}:n"] = str(len(got))
                NUM[f"x:{a}:{b}:{met}:nword"] = WORDS.get(len(got), str(len(got)))
                NUM[f"x:{a}:{b}:{met}:npos"] = str(sum(1 for g in got if g[1] > 0))
                NUM[f"x:{a}:{b}:{met}:nneg"] = str(sum(1 for g in got if g[2] < 0))
    return D


def strata_values():
    mx = {}
    for ds in DS:
        f = ROOT / "results" / ds / "test" / "exploratory_timeaware.csv"
        if not f.exists():
            continue
        r = pd.read_csv(f)
        r = r[(r["mode"] == "e2e") & (r["metric"] == "acc@1") & (r["stratum"].isin(["hist=repeat", "hist=explore"]))]
        for _, x in r.iterrows():
            m = x["method"].replace("@H-100", "")
            st = "repeat" if x["stratum"] == "hist=repeat" else "explore"
            NUM[f"xs:{ds}:{m}:{st}"] = f"{x['value']:.4f}"
            if st == "explore":
                mx[m] = max(mx.get(m, 0.0), float(x["value"]))
    for m, v in mx.items():
        NUM[f"xsmax:{m}:explore"] = f"{v:.4f}"


def icpc_extra(D):
    def acc(ds, name):
        v = value(D[ds][0], name, "acc@1")
        return v
    for ds in DS:
        n_cost = mt.load_cost("hybrid-tt", "H-100", ds)
        for m in ("llm4poi", "gnpr-sid"):
            c = mt.load_cost(m, "H-100", ds)
            a_m, a_r = acc(ds, f"{m}-tt"), acc(ds, "hybrid-tt")
            if c is None or n_cost is None or a_m is None or a_r is None:
                continue
            n = c["n"].get("test")
            for inf in (False, True):
                v = mt.icpc(c, n_cost, a_m, a_r, None if inf else n, "gpu_s")
                if v is not None:
                    NUM[f"icpctt:{ds}:{m}:{'inf' if inf else 'test'}"] = mt.q2(v)
        cg = mt.load_cost("getnext", "H-100", ds)
        c = mt.load_cost("llm4poi", "H-100", ds)
        res = mt.R(ds, "test")
        a_m, a_r = res.get("llm4poi@H-100", "acc@1"), res.get("getnext@H-100", "acc@1")
        if cg is not None and c is not None and a_m is not None and a_r is not None:
            n = c["n"].get("test")
            for inf in (False, True):
                v = mt.icpc(c, cg, a_m, a_r, None if inf else n, "gpu_s")
                if v is not None:
                    NUM[f"icpcgn:{ds}:llm4poi:{'inf' if inf else 'test'}"] = mt.q2(v)


def table(D):
    cols = "l" + "r" * len(DS)
    L = [r"% AUTO-GENERATED by paper/scripts/make_extra.py; do not edit.", r"\scriptsize",
         r"\setlength{\tabcolsep}{4pt}", rf"\begin{{tabular}}{{@{{}}{cols}@{{}}}}", r"\toprule",
         "method & " + " & ".join(LABEL[d] for d in DS) + r" \\"]
    block = None
    heads = {"unknown": r"Request time unknown (default protocol)", "given": r"Request time given to the method"}
    for key, lab, blk in ROWS:
        if blk != block:
            L.append(r"\midrule")
            L.append(rf"\multicolumn{{{len(DS) + 1}}}{{@{{}}l}}{{\textit{{{heads[blk]}}}}} \\")
            block = blk
        cells = []
        for ds in DS:
            v = value(D[ds][0], key, "acc@1")
            cells.append(f"{v:.4f}" if v is not None else mt.DASH)
        L.append(lab + " & " + " & ".join(cells) + r" \\")
    L.append(r"\midrule")
    L.append(rf"\multicolumn{{{len(DS) + 1}}}{{@{{}}l}}{{\textit{{Acc@1 lead of the fine-tuned LLM over the Hybrid of the same condition ($^\dagger$ 95\% CI excludes 0)}}}} \\")
    for a, b, lab in (("llm4poi", "hybrid", "LLM4POI $-$ Hybrid"), ("gnpr-sid", "hybrid", "GNPR-SID $-$ Hybrid"),
                      ("llm4poi-tt", "hybrid-tt", "LLM4POI-TT $-$ Hybrid-TT"),
                      ("gnpr-sid-tt", "hybrid-tt", "GNPR-SID-TT $-$ Hybrid-TT")):
        cells = []
        for ds in DS:
            p = pair(D[ds][1], a, b, "acc@1")
            cells.append(mt.DASH if p is None else mt.sf4(p[0]) + (r"$^\dagger$" if (p[1] > 0 or p[2] < 0) else ""))
        L.append(lab + " & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    mt.write("tab_timeaware", "\n".join(L) + "\n")


def figure(D):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"matplotlib unavailable ({e}); figure skipped")
        return
    import numpy as np
    plt.rcParams.update({"font.size": 7.5, "axes.linewidth": 0.6, "pdf.fonttype": 42, "font.family": "serif"})
    panels = (("Request time unknown (default protocol)", "hybrid",
               (("ds-llm4poi-lcs", "DS-LLM4POI"), ("llm4poi", "LLM4POI"), ("gnpr-sid", "GNPR-SID"))),
              ("Request time given to every method", "hybrid-tt",
               (("ds-llm4poi-lcs-tt", "DS-LLM4POI-TT"), ("llm4poi-tt", "LLM4POI-TT"), ("gnpr-sid-tt", "GNPR-SID-TT"))))
    colors = ("#9ecae1", "#0072B2", "#E69F00")
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.2), sharey=True)
    x0 = np.arange(len(DS))
    w = 0.26
    for ax, (title, ref, meths) in zip(axes, panels):
        for k, (m, lab) in enumerate(meths):
            xs, ys, lo, hi = [], [], [], []
            for i, ds in enumerate(DS):
                p = pair(D[ds][1], m, ref, "acc@1")
                if p is None:
                    continue
                xs.append(x0[i] + (k - 1) * w)
                ys.append(p[0])
                lo.append(p[0] - p[1])
                hi.append(p[2] - p[0])
            ax.bar(xs, ys, w, color=colors[k], linewidth=0)
            ax.errorbar(xs, ys, yerr=[lo, hi], fmt="none", ecolor="black", elinewidth=0.5, capsize=1.2, capthick=0.5)
        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_xticks(x0)
        ax.set_xticklabels([LABEL[d].replace(r"\~a", "\u00e3") for d in DS], fontsize=6.8)
        ax.set_title(title + "\nlead over " + ("Hybrid" if ref == "hybrid" else "Hybrid-TT"), fontsize=7.5)
        ax.tick_params(length=2, width=0.5)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
    from matplotlib.patches import Patch
    fig.legend([Patch(color=c) for c in colors], ["DS-LLM4POI", "LLM4POI", "GNPR-SID"], loc="upper center", ncol=3,
               frameon=False, fontsize=7, handlelength=1.0, columnspacing=1.6, bbox_to_anchor=(0.5, 1.0))
    axes[0].set_ylabel("Acc@1 difference")
    fig.tight_layout(pad=0.3, w_pad=0.8, rect=(0, 0, 1, 0.9))
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / "fig_leads.pdf")
    plt.close(fig)


DN_DS = ("nyc", "tky", "istanbul")
DN_POL = ("H-10", "S-10", "H-70", "S-70", "H-100")
DN_METHODS = (
    ("hybrid", "Hybrid", "#2e2e2e", "o"), ("getnext", "GETNext", "#5e7f5d", "s"),
    ("graph-flashback", "Graph-Flashback", "#6f8dba", "P"), ("ds-llm4poi-lcs", "DS-LLM4POI", "#c26b67", "v"),
    ("llm4poi", "LLM4POI", "#7b55a0", "^"), ("gnpr-sid", "GNPR-SID", "#daa113", "X"),
)
FINETUNED = ("llm4poi", "gnpr-sid")
SCRATCH = ("getnext", "graph-flashback")


def dataneeds_acc():
    out = {}
    nhist, nft, nsamp, nsc = 0, 0, 0, 0
    r20d = []
    for ds in DN_DS:
        res = mt.R(ds, "test")
        for pol in DN_POL:
            rc = res.recall.get(pol)
            if rc is not None:
                NUM[f"dna:{ds}:r20:{pol}"] = mt.f4(rc)
        rc = {pol: res.recall.get(pol) for pol in DN_POL}
        if None not in (rc["H-10"], rc["S-10"], rc["H-100"]):
            NUM[f"dna:{ds}:r20:dHS"] = mt.sf4(rc["H-10"] - rc["S-10"])
            NUM[f"dna:{ds}:r20:dS"] = mt.sf4(rc["S-10"] - rc["H-100"])
            r20d.append(rc["H-10"] - rc["S-10"])
        for m, _, _, _ in DN_METHODS:
            v = {pol: res.get(f"{m}@{pol}", "acc@1") for pol in DN_POL}
            if m == "hybrid":
                v["S-10"] = v["S-10"] if v["S-10"] is not None else v["H-100"]
            out[(ds, m)] = v
            for pol in DN_POL:
                if v[pol] is not None:
                    NUM[f"dna:{ds}:{m}:{pol}"] = mt.f4(v[pol])
            eff = {}
            for key, a, b in (("dS", "S-10", "H-100"), ("dHS", "H-10", "S-10"), ("dH", "H-10", "H-100")):
                if v[a] is None or v[b] is None:
                    continue
                pr = res.pair(f"{m}@{a}", f"{m}@{b}", "acc@1")
                d = v[a] - v[b]
                NUM[f"dna:{ds}:{m}:{key}"] = mt.sf4(d)
                eff[key] = d
                if pr is not None and not (m == "hybrid" and key == "dS"):
                    NUM[f"dna:{ds}:{m}:{key}:ci"] = f"[{pr['ci_lo']:.4f}, {pr['ci_hi']:.4f}]".replace("-", "$-$")
            if "dHS" in eff and "dH" in eff and eff["dH"] < 0 and eff["dHS"] <= 0:
                NUM[f"dna:{ds}:{m}:histshare"] = mt.pct(eff["dHS"] / eff["dH"])
            if m in FINETUNED and "dHS" in eff and "dS" in eff:
                nft += 1
                nhist += abs(eff["dHS"]) > abs(eff["dS"])
            if m in SCRATCH and "dHS" in eff and "dS" in eff:
                nsc += 1
                nsamp += abs(eff["dS"]) > abs(eff["dHS"])
    for ds in DN_DS:
        d70 = [out[(ds, m)]["H-70"] - out[(ds, m)]["H-100"] for m, _, _, _ in DN_METHODS
               if out[(ds, m)]["H-70"] is not None and out[(ds, m)]["H-100"] is not None]
        if d70:
            NUM[f"dna:{ds}:h70:minloss"] = mt.sf4(min(d70))
            NUM[f"dna:{ds}:h70:n"] = str(len(d70))
    NUM["dna:ft:nhist"], NUM["dna:ft:n"] = str(nhist), str(nft)
    NUM["dna:scratch:nsamp"], NUM["dna:scratch:n"] = str(nsamp), str(nsc)
    if r20d:
        NUM["dna:r20:dHS:min"], NUM["dna:r20:dHS:max"] = mt.sf4(max(r20d)), mt.sf4(min(r20d))
    return out


def dataneeds_table(V):
    L = [r"% AUTO-GENERATED by paper/scripts/make_extra.py; do not edit.", r"\scriptsize",
         r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
         " & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{LABEL[d]}}}" for d in DN_DS) + r" \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
         "method & " + " & ".join([r"$\Delta_S$", r"$\Delta_{HS}$"] * len(DN_DS)) + r" \\", r"\midrule"]
    for m, lab, _, _ in DN_METHODS:
        cells = []
        for ds in DN_DS:
            v = V[(ds, m)]
            dS = None if v["S-10"] is None or v["H-100"] is None else v["S-10"] - v["H-100"]
            dHS = None if v["H-10"] is None or v["S-10"] is None else v["H-10"] - v["S-10"]
            cells += ["=" if m == "hybrid" else mt.sf4(dS), mt.sf4(dHS)]
        L.append(lab + " & " + " & ".join(cells) + r" \\")
    L.append(r"\midrule")
    cells = []
    for ds in DN_DS:
        cells += ["=", NUM.get(f"dna:{ds}:r20:dHS", mt.DASH)]
    L.append(r"Retriever R@20 & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    mt.write("tab_dataneeds_acc", "\n".join(L) + "\n")


def dataneeds_figure(V):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
        from matplotlib.lines import Line2D
    except Exception as e:
        print(f"matplotlib unavailable ({e}); data-needs figure skipped")
        return
    plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
                         "mathtext.fontset": "stix", "pdf.fonttype": 42, "axes.linewidth": 0.6,
                         "axes.edgecolor": "#4d4d4d", "xtick.color": "#1a1a1a", "ytick.color": "#1a1a1a",
                         "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5,
                         "ytick.major.size": 2.5})
    xs = (10, 70, 100)
    FIG.mkdir(parents=True, exist_ok=True)

    def runs(pts):
        out, cur = [], []
        for x, y in pts:
            if y is None:
                if cur:
                    out.append(cur)
                cur = []
            else:
                cur.append((x, y))
        if cur:
            out.append(cur)
        return out

    for tag, ds in zip(("a", "b", "c"), DN_DS):
        fig, ax = plt.subplots(figsize=(167.07 / 72, 111.6 / 72))
        fig.subplots_adjust(left=33.8 / 167.07, right=161.3 / 167.07, bottom=1 - 87.8 / 111.6, top=1 - 4.3 / 111.6)
        series = []
        for m, lab, col, mk in DN_METHODS:
            v = V[(ds, m)]
            h = [(x, v[f"H-{x}"]) for x in xs]
            s_ = [(10, v["S-10"]), (70, v["S-70"]), (100, v["H-100"])] if m != "hybrid" else None
            series.append((m, col, mk, h, s_))
        for m, col, mk, h, s_ in series:
            if s_ is None or s_[0][1] is None:
                continue
            for run in runs(s_):
                if len(run) > 1:
                    ax.plot([p[0] for p in run], [p[1] for p in run], color=col, lw=1.0, ls=(0, (3.2, 1.6)), zorder=2)
        for m, col, mk, h, s_ in series:
            for run in runs(h):
                ax.plot([p[0] for p in run], [p[1] for p in run], color=col, lw=1.3, marker=mk, ms=4 if mk != "X" else 4.6,
                        mew=1.0, mec=col, mfc=col, zorder=3 if m == "hybrid" else 4)
        for m, col, mk, h, s_ in series:
            if s_ is None:
                continue
            pts = [p for p in s_[:2] if p[1] is not None]
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts], ls="none", marker=mk, ms=4 if mk != "X" else 4.6,
                        mew=0.9, mec=col, mfc="white", zorder=5)
        ax.set_xlim(4, 104)
        ax.set_xticks(xs)
        ax.grid(True, color="#e6e6e6", lw=0.6)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(labelsize=8.5)
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5, steps=[1, 2, 5, 10]))
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        ax.set_xlabel(r"Share $x$ (%)", fontsize=9, labelpad=1.5)
        ax.set_ylabel("Acc@1", fontsize=9, labelpad=2)
        fig.savefig(FIG / f"poibench_dataneeds_{tag}_{ds}_r1.pdf")
        plt.close(fig)
    fig = plt.figure(figsize=(514.07 / 72, 25.92 / 72))
    hs = [Line2D([], [], color=col, lw=1.3, marker=mk, ms=4 if mk != "X" else 4.6, mfc=col, mec=col)
          for _, _, col, mk in DN_METHODS]
    labs = [lab for _, lab, _, _ in DN_METHODS]
    hs += [Line2D([], [], color="#2e2e2e", lw=1.3), Line2D([], [], color="#555555", lw=1.0, ls=(0, (3.2, 1.6)),
                                                          marker="o", ms=4, mfc="white", mec="#555555", mew=0.9)]
    labs += [r"H-$x$", r"S-$x$"]
    order = [0, 3, 1, 4, 2, 5, 6, 7]
    fig.legend([hs[i] for i in order], [labs[i] for i in order], loc="center", ncol=4, frameon=False, fontsize=9,
               handlelength=2.4, columnspacing=1.6, handletextpad=0.5, labelspacing=0.25, borderaxespad=0)
    fig.savefig(FIG / "poibench_dataneeds_key_r1.pdf")
    plt.close(fig)


def main():
    D = collect()
    icpc_extra(D)
    strata_values()
    table(D)
    figure(D)
    V = dataneeds_acc()
    dataneeds_table(V)
    dataneeds_figure(V)
    L = ["% AUTO-GENERATED by paper/scripts/make_extra.py; do not edit.",
         r"% \V{key}: exploratory time-aware analysis (results/<ds>/test/exploratory_timeaware.*)"]
    for k, v in sorted(NUM.items()):
        L.append(rf"\expandafter\def\csname V:{k}\endcsname{{{v}}}")
    mt.write("numbers_extra", "\n".join(L) + "\n")
    print(f"{len(NUM)} macros, tab_timeaware.tex, tab_dataneeds_acc.tex, fig_leads.pdf, poibench_dataneeds_*_r1.pdf")


if __name__ == "__main__":
    main()
