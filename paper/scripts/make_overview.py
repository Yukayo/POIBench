#!/usr/bin/env python3
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parents[1] / "figures" / "fig_overview.pdf"
INK, MUTED, EDGE = "#1f1f1e", "#5f5e58", "#9a998f"
FILL = {"data": "#eaf2fc", "proc": "#f4f4f1", "gate": "#fdeee7", "out": "#e8f6ef"}

BOXES = {
    "raw": (0.0, 0.30, 2.0, 0.50, "data", "Raw check-ins",
            "TSMC2014 NYC, TKY\nMassive-STEPS Istanbul\n(sha256-pinned)"),
    "build": (2.25, 0.30, 2.0, 0.50, "proc", "Build (once)",
              "filter $\\geq$10 / $\\geq$10\nglobal 80/10/10 time cut\n24 h trajectories, queries"),
    "ret": (2.25, -0.45, 2.0, 0.50, "proc", "Common retriever",
            "user, transition, geo lists\nRRF, top-20 / 50\nno ground-truth insertion"),
    "exp": (4.50, 0.30, 2.0, 0.50, "proc", "Exporter per policy",
            "H-x / S-x visible rows\nquery inputs (no targets)\ncandidates, train labels"),
    "canary": (6.75, -0.45, 2.0, 0.50, "gate", "Leakage canary",
               "hidden + future rows replaced:\nidentical output required"),
    "adapt": (6.75, 0.30, 2.0, 0.50, "proc", "Method adapters",
              "pinned upstream code\nseed 0, author defaults\nval-only selection"),
    "eval": (9.00, 0.30, 2.0, 0.50, "gate", "Locked evaluator",
             "test split locked until\nfreeze (data/code hashes)"),
    "out": (9.00, -0.45, 2.0, 0.50, "out", "Reported",
            "Acc@k, NDCG, MRR, strata\nuser bootstrap CIs\nper-stage cost, ICPC"),
}
ARROWS = [("raw", "build"), ("build", "exp"), ("build", "ret"), ("ret", "exp"), ("exp", "adapt"),
          ("canary", "adapt"), ("adapt", "eval"), ("eval", "out")]


def anchor(b, side):
    x, y, w, h = b[:4]
    return {"r": (x + w, y + h / 2), "l": (x, y + h / 2), "b": (x + w / 2, y), "t": (x + w / 2, y + h)}[side]


def main():
    fig, ax = plt.subplots(figsize=(7.16, 1.55))
    for x, y, w, h, kind, title, body in BOXES.values():
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.05",
                                    fc=FILL[kind], ec=EDGE, lw=0.7))
        ax.text(x + w / 2, y + h - 0.07, title, ha="center", va="top", fontsize=6.6, color=INK, weight="bold")
        ax.text(x + w / 2, y + h - 0.19, body, ha="center", va="top", fontsize=5.0, color=MUTED, linespacing=1.15)
    for a, b in ARROWS:
        A, B = BOXES[a], BOXES[b]
        if A[1] == B[1]:
            p, q = anchor(A, "r"), anchor(B, "l")
        elif A[0] == B[0]:
            p, q = (anchor(A, "b"), anchor(B, "t")) if A[1] > B[1] else (anchor(A, "t"), anchor(B, "b"))
        elif a == "ret":
            p, q = anchor(A, "r"), anchor(B, "b")
        else:
            p, q = anchor(A, "r"), anchor(B, "b")
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=7, lw=0.8, color=MUTED,
                                     connectionstyle="arc3,rad=0" if (A[1] == B[1] or A[0] == B[0]) else "angle,angleA=0,angleB=90"))
    ax.set_xlim(-0.05, 11.05)
    ax.set_ylim(-0.5, 0.85)
    ax.axis("off")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.02)
    print(OUT)


if __name__ == "__main__":
    main()
