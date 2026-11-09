#!/usr/bin/env python
from __future__ import annotations

import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import numpy as np
import pandas as pd

import make_tables as mt

MAIN = ("nyc", "tky", "ca", "sin", "istanbul", "sao_paulo")
CITY = {"nyc": "NYC", "tky": "TKY", "ca": "CA", "sin": "SIN", "istanbul": "IST", "sao_paulo": "SP"}
FAMILIES = [
    ("ds-llm4poi-lcs", "ds-llm4poi-lcs", "ds-llm4poi-lcs-",
     {"qwen2.5-3b": 3.09, "qwen2.5-7b": 7.62, "qwen2.5-14b": 14.77, "qwen2.5-32b": 32.76, "llama3.1-8b": 8.03,
      "mistral-7b-v0.3": 7.25}, "qwen2.5-7b"),
    ("llm4poi", "llm4poi", "llm4poi-",
     {"qwen2.5-1.5b": 1.54, "qwen2.5-3b": 3.09, "qwen2.5-7b": 7.62, "llama3.1-8b": 8.03}, "qwen2.5-7b"),
    ("gnpr-sid", "gnpr-sid", "gnpr-sid-",
     {"qwen2.5-7b": 7.62, "qwen2.5-14b": 14.77, "qwen2.5-32b": 32.76, "llama3-8b": 8.03}, "llama3-8b"),
]
FLAB = {"ds-llm4poi-lcs": "DS-LLM4POI (zero-shot)", "llm4poi": "LLM4POI (full FT)", "gnpr-sid": "GNPR-SID (LoRA)"}
FCOL = {"ds-llm4poi-lcs": mt.PAL[3], "llm4poi": mt.PAL[4], "gnpr-sid": mt.PAL[6]}
FMK = {"ds-llm4poi-lcs": "D", "llm4poi": "v", "gnpr-sid": "X"}
OTHER_MK = {"llama3.1-8b": "s", "llama3-8b": "s", "mistral-7b-v0.3": "^"}
NUM: dict = {}


def num(k, v):
    NUM[k] = v


def f4(x):
    return mt.DASH if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.4f}"


def s4(x):
    return mt.DASH if x is None or (isinstance(x, float) and np.isnan(x)) else mt.sf4(x)


def q2(x):
    return mt.DASH if x is None or (isinstance(x, float) and np.isnan(x)) else mt.q2(x)


def rows():
    from poibench import evaluate as ev
    ev.assert_split_allowed(str(ROOT), "test")
    out = []
    for ds in MAIN:
        names = {}
        for fam, prim, pre, keys, pk in FAMILIES:
            for k, nb in keys.items():
                m = prim if k == pk else f"{pre}{k}"
                if (ROOT / "preds" / f"{m}@H-100" / ds / "test.parquet").exists():
                    names[m] = (fam, k, nb, k == pk)
        queries, cands, venues, checkins = ev.load_split(str(ROOT), ds, "test", "H-100")
        es = ev.build_eval_set(queries, cands, venues, checkins=checkins)
        W = ev.bootstrap_weights(es.n_users, B=1000, seed=0)
        uc, nu = es.user_codes, es.n_users
        Wn = W @ np.bincount(uc, minlength=nu).astype(float)

        def hits(m):
            df = pd.read_parquet(ROOT / "preds" / f"{m}@H-100" / ds / "test.parquet",
                                 columns=["query_id", "ranked_venue_ids"])
            rk, _ = ev.per_query_ranks(es, df)
            return (rk["e2e"] <= 1).astype(float)
        yh = hits("hybrid")
        bh = W @ np.bincount(uc, weights=yh, minlength=nu) / Wn
        out.append({"family": "hybrid", "backbone": "", "params_b": None, "primary": True, "method": "hybrid",
                    "ds": ds, "acc1": float(yh.mean())})
        for m, (fam, k, nb, prim) in names.items():
            y = hits(m)
            b = W @ np.bincount(uc, weights=y, minlength=nu) / Wn
            c = mt.load_cost(m, "H-100", ds)
            q = ((c or {}).get("perq", {}).get("test") or {})
            out.append({"family": fam, "backbone": k, "params_b": nb, "primary": prim, "method": m, "ds": ds,
                        "acc1": float(y.mean()), "d": float(y.mean() - yh.mean()),
                        "d_lo": float(np.percentile(b - bh, 2.5)), "d_hi": float(np.percentile(b - bh, 97.5)),
                        "once_gpu_h": (c["once"]["gpu_s"] / 3600) if c else None,
                        "gpus_per_1k": 1000 * (q.get("gpu_s") or 0) if c else None})
            print(f"{ds:10s} {m:32s} acc1={y.mean():.4f} d={y.mean() - yh.mean():+.4f}", flush=True)
    return pd.DataFrame(out)


def numbers(df):
    for _, r in df.iterrows():
        if r["family"] == "hybrid":
            num(f"sc:{r['ds']}:hybrid:acc1", f4(r["acc1"]))
            continue
        k = f"sc:{r['ds']}:{r['family']}:{r['backbone']}"
        num(f"{k}:acc1", f4(r["acc1"]))
        num(f"{k}:d", s4(r["d"]))
        num(f"{k}:dci", f"[{s4(r['d_lo'])}, {s4(r['d_hi'])}]")
        num(f"{k}:sig", "yes" if (r["d_lo"] > 0 or r["d_hi"] < 0) else "no")
        num(f"{k}:perK", q2(r["gpus_per_1k"]))
        if r["family"] != "ds-llm4poi-lcs":
            num(f"{k}:onceh", q2(r["once_gpu_h"]))
    s = df[df["family"] != "hybrid"]
    for fam, _, _, keys, pk in FAMILIES:
        g = s[s["family"] == fam]
        for k in keys:
            x = g[g["backbone"] == k]
            if not len(x):
                continue
            kk = f"sc:{fam}:{k}"
            num(f"{kk}:ncity", str(len(x)))
            num(f"{kk}:d:min", s4(x["d"].min()))
            num(f"{kk}:d:max", s4(x["d"].max()))
            num(f"{kk}:npos", str(int((x["d_lo"] > 0).sum())))
            num(f"{kk}:nneg", str(int((x["d_hi"] < 0).sum())))
            num(f"{kk}:ntie", str(int(((x["d_lo"] <= 0) & (x["d_hi"] >= 0)).sum())))
            p = g[g["backbone"] == pk].set_index("ds")
            xx = x.set_index("ds")
            com = [d for d in xx.index if d in p.index]
            if com and k != pk:
                rq = np.mean([xx.loc[d, "gpus_per_1k"] / p.loc[d, "gpus_per_1k"] for d in com
                              if p.loc[d, "gpus_per_1k"]])
                num(f"{kk}:perKratio", f"{rq:.2f}")
                if fam != "ds-llm4poi-lcs":
                    ro = np.mean([xx.loc[d, "once_gpu_h"] / p.loc[d, "once_gpu_h"] for d in com if p.loc[d, "once_gpu_h"]])
                    num(f"{kk}:onceratio", f"{ro:.2f}")
                dd = np.mean([xx.loc[d, "acc1"] - p.loc[d, "acc1"] for d in com])
                num(f"{kk}:dprim:mean", s4(dd))
                num(f"{kk}:dprim:min", s4(min(xx.loc[d, "acc1"] - p.loc[d, "acc1"] for d in com)))
                num(f"{kk}:dprim:max", s4(max(xx.loc[d, "acc1"] - p.loc[d, "acc1"] for d in com)))
                num(f"{kk}:dprim:absmax", f4(max(abs(xx.loc[d, "acc1"] - p.loc[d, "acc1"]) for d in com)))
            if fam == "ds-llm4poi-lcs" or len(x) > 1:
                num(f"{kk}:perK:min", q2(x["gpus_per_1k"].min()))
                num(f"{kk}:perK:max", q2(x["gpus_per_1k"].max()))
            if fam != "ds-llm4poi-lcs":
                num(f"{kk}:onceh:min", q2(x["once_gpu_h"].min()))
                num(f"{kk}:onceh:max", q2(x["once_gpu_h"].max()))
    for ds in ("nyc", "tky"):
        g = s[(s["ds"] == ds) & (s["family"] != "ds-llm4poi-lcs")]
        if len(g):
            b = g.loc[g["acc1"].idxmax()]
            num(f"sc:{ds}:bestft", f"{FLAB[b['family']].split(' (')[0]} {label(b['backbone'])}")
            num(f"sc:{ds}:bestft:acc1", f4(b["acc1"]))


def label(k):
    return {"qwen2.5-1.5b": "Qwen2.5-1.5B", "qwen2.5-3b": "Qwen2.5-3B", "qwen2.5-7b": "Qwen2.5-7B",
            "qwen2.5-14b": "Qwen2.5-14B", "qwen2.5-32b": "Qwen2.5-32B", "llama3.1-8b": "Llama-3.1-8B",
            "llama3-8b": "Llama-3-8B", "mistral-7b-v0.3": "Mistral-7B"}.get(k, k)


def figure(df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter, NullLocator
    plt.rcParams.update({"font.size": 6.5, "axes.linewidth": 0.5, "pdf.fonttype": 42, "font.family": "serif"})
    fig, axs = plt.subplots(2, 2, figsize=(3.45, 3.1))
    (a, b), (c, d) = axs
    s = df[df["family"] != "hybrid"]
    xt = [1.5, 3, 7, 14, 32]

    def style(ax, xl, yl, title, logy=False):
        ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            ax.yaxis.set_minor_formatter(NullFormatter())
        ax.xaxis.set_major_locator(FixedLocator(xt))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.grid(True, color="#e6e6e3", lw=0.4)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(length=2, width=0.4, labelsize=5.8, pad=1.5)
        ax.set_xlabel(xl, fontsize=6, labelpad=1)
        ax.set_ylabel(yl, fontsize=6, labelpad=1)
        ax.set_title(title, fontsize=6.5, pad=2.5)

    def ladder(ax, g, ycol, fam, **kw):
        q = g[g["backbone"].str.startswith("qwen")].sort_values("params_b")
        if len(q):
            ax.plot(q["params_b"], q[ycol], "-", marker=FMK[fam], ms=2.8, lw=0.9, color=FCOL[fam], **kw)
        o = g[~g["backbone"].str.startswith("qwen")]
        for _, r in o.iterrows():
            ax.plot([r["params_b"]], [r[ycol]], ls="none", marker=OTHER_MK.get(r["backbone"], "o"), ms=3,
                    mfc="white", mec=FCOL[fam], mew=0.8)

    for ax, ds, t in ((a, "nyc", "(a) NYC"), (b, "tky", "(b) TKY")):
        g = s[s["ds"] == ds]
        for fam in FLAB:
            ladder(ax, g[g["family"] == fam], "acc1", fam)
        h = df[(df["family"] == "hybrid") & (df["ds"] == ds)]
        if len(h):
            ax.axhline(h["acc1"].iloc[0], color=mt.PAL[0], lw=0.8, ls="--")
        style(ax, "parameters (B)", "Acc@1 (end-to-end)", t)
    g = s[s["family"] == "ds-llm4poi-lcs"]
    for i, ds in enumerate(MAIN):
        q = g[(g["ds"] == ds) & g["backbone"].str.startswith("qwen")].sort_values("params_b")
        c.plot(q["params_b"], q["d"], "-", lw=0.6, color=FCOL["ds-llm4poi-lcs"], alpha=0.55)
    m = g[g["backbone"].str.startswith("qwen")].groupby("params_b")["d"].mean()
    c.plot(m.index, m.values, "-", marker="D", ms=2.8, lw=1.3, color=FCOL["ds-llm4poi-lcs"])
    for k in ("llama3.1-8b", "mistral-7b-v0.3"):
        o = g[g["backbone"] == k]
        if len(o):
            c.plot([o["params_b"].iloc[0]], [o["d"].mean()], ls="none", marker=OTHER_MK[k], ms=3, mfc="white",
                   mec=FCOL["ds-llm4poi-lcs"], mew=0.8)
    c.axhline(0, color=mt.PAL[0], lw=0.8, ls="--")
    style(c, "parameters (B)", "Acc@1 $-$ Hybrid", "(c) zero-shot, six cities")
    for fam in FLAB:
        gg = s[s["family"] == fam]
        mm = gg.groupby(["backbone", "params_b"], as_index=False)["gpus_per_1k"].mean()
        ladder(d, mm, "gpus_per_1k", fam)
    style(d, "parameters (B)", "GPU-s per 1k queries", "(d) inference cost", logy=True)
    hs = [Line2D([], [], color=FCOL[f], marker=FMK[f], ms=2.8, lw=0.9) for f in FLAB]
    names = [FLAB[f] for f in FLAB]
    hs += [Line2D([], [], color="#555555", marker="s", ms=3, lw=0, mfc="white"),
           Line2D([], [], color="#555555", marker="^", ms=3, lw=0, mfc="white"),
           Line2D([], [], color=mt.PAL[0], lw=0.8, ls="--")]
    names += ["Llama-3(.1)-8B", "Mistral-7B", "Hybrid"]
    fig.legend(hs, names, loc="upper center", ncol=3, frameon=False, fontsize=5.6, handlelength=1.3,
               columnspacing=0.8, handletextpad=0.3, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(pad=0.25, h_pad=0.5, w_pad=0.5, rect=(0, 0, 1, 0.88))
    fig.savefig(mt.FIG / "fig_scale.pdf")
    plt.close(fig)


def main():
    df = rows()
    df.to_csv(mt.TAB / "scale_data.csv", index=False)
    numbers(df)
    figure(df)
    L = ["% AUTO-GENERATED by paper/scripts/make_scale.py; do not edit.",
         r"% \V{sc:...}: model size and family (EXPLORATORY; test, H-100, end-to-end, B = 1000, seed 0)"]
    for k, v in sorted(NUM.items()):
        L.append(rf"\expandafter\def\csname V:{k}\endcsname{{{v}}}")
    (mt.TAB / "numbers_scale.tex").write_text("\n".join(L) + "\n")
    print(f"{len(NUM)} macros, fig_scale.pdf, {len(df)} rows")


if __name__ == "__main__":
    main()
