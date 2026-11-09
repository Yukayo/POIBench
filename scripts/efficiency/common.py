from __future__ import annotations

import json
import math
import os
import platform
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "paper" / "scripts"))

import make_tables as mt

OUT = ROOT / "results" / "efficiency"
OUT.mkdir(parents=True, exist_ok=True)
SCRATCH = Path(__import__("os").environ.get("POIBENCH_SCRATCH") or str(ROOT / "work" / "efficiency_scratch"))
SCRATCH.mkdir(parents=True, exist_ok=True)

MAIN = ("nyc", "tky", "ca", "sin", "istanbul", "sao_paulo")
BIG = ("gowalla", "foursquare_global")
LABEL = {"nyc": "NYC", "tky": "TKY", "ca": "CA", "sin": "SIN", "istanbul": "Istanbul", "sao_paulo": "Sao Paulo",
         "gowalla": "Gowalla", "foursquare_global": "Foursquare"}
POLICIES = ("H-10", "H-70", "H-100", "S-10", "S-70")
METHODS = ("hybrid", "getnext", "getnext-zfeat", "graph-flashback", "ds-llm4poi-lcs", "llm4poi", "gnpr-sid", "ga-llm")
MLABEL = {"hybrid": "Hybrid", "getnext": "GETNext", "getnext-zfeat": "GETNext-z", "graph-flashback": "Graph-Flashback",
          "ds-llm4poi-lcs": "DS-LLM4POI", "llm4poi": "LLM4POI", "gnpr-sid": "GNPR-SID", "retriever": "Retriever", "ga-llm": "GA-LLM"}
FAMILY = {"hybrid": "counting", "getnext": "trained", "getnext-zfeat": "trained", "graph-flashback": "trained",
          "ds-llm4poi-lcs": "llm-icl", "llm4poi": "llm-ft", "gnpr-sid": "llm-ft", "ga-llm": "llm-ft"}


def hardware() -> dict:
    cpu = None
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return {"host": platform.node(), "cpu_model": cpu, "cpu_count": os.cpu_count(), "python": platform.python_version()}


def meta(ds: str) -> dict:
    return json.load(open(ROOT / "data" / ds / "meta.json"))


def cost_json(method: str, policy: str, ds: str, split: str = "test") -> dict | None:
    p = ROOT / "preds" / f"{method}@{policy}" / ds / f"{split}.cost.json"
    return json.load(open(p)) if p.exists() else None


def data_sizes(ds: str, policy: str = "H-100") -> dict:
    m = meta(ds)
    c, t = m["counts"], m["totals"]
    pol = m.get("policies", {}).get(policy, {})
    out = {"n_users": t["users"], "n_catalog_venues": t["catalog_venues"], "n_checkins_total": t["checkins"],
           "n_train_checkins": c["train"]["checkins"], "n_train_queries_all": c["train"]["queries"],
           "n_val_queries": c["val"]["queries"], "n_test_queries": c["test"]["queries"],
           "n_train_queries": pol.get("n_train_queries"),
           "n_visible_train_checkins": (round(pol["frac_visible_precutoff_checkins"] * c["train"]["checkins"])
                                        if "frac_visible_precutoff_checkins" in pol else None)}
    qi = ROOT / "exports" / ds / policy / "query_inputs_test.parquet"
    if qi.exists():
        import pandas as pd
        q = pd.read_parquet(qi, columns=["n_visible_own"])
        out["mean_visible_own_test"] = float(q["n_visible_own"].mean())
    return out


def loglog(x, y):
    import numpy as np
    pts = [(math.log10(a), math.log10(b)) for a, b in zip(x, y) if a and b and a > 0 and b > 0]
    n = len(pts)
    if n < 2:
        return None, None, None, n
    X = np.array([p[0] for p in pts])
    Y = np.array([p[1] for p in pts])
    if np.ptp(X) == 0:
        return None, None, None, n
    A = np.c_[X, np.ones(n)]
    coef, *_ = np.linalg.lstsq(A, Y, rcond=None)
    res = Y - A @ coef
    ss_tot = float(((Y - Y.mean()) ** 2).sum())
    r2 = 1 - float((res ** 2).sum()) / ss_tot if ss_tot > 0 else None
    se = math.sqrt(float((res ** 2).sum()) / (n - 2) / float(((X - X.mean()) ** 2).sum())) if n > 2 else None
    return float(coef[0]), se, r2, n


def loglog_fe(groups):
    import numpy as np
    xs, ys = [], []
    for pts in groups.values():
        pts = [(math.log10(a), math.log10(b)) for a, b in pts if a and b and a > 0 and b > 0]
        if len(pts) < 2:
            continue
        X = np.array([p[0] for p in pts])
        Y = np.array([p[1] for p in pts])
        xs.append(X - X.mean())
        ys.append(Y - Y.mean())
    if not xs:
        return None, None, 0, 0
    X, Y = np.concatenate(xs), np.concatenate(ys)
    if float((X ** 2).sum()) == 0:
        return None, None, len(X), len(xs)
    b = float((X * Y).sum() / (X ** 2).sum())
    dof = len(X) - len(xs) - 1
    se = math.sqrt(float(((Y - b * X) ** 2).sum()) / dof / float((X ** 2).sum())) if dof > 0 else None
    return b, se, len(X), len(xs)


def write_csv(name: str, rows: list, cols: list | None = None):
    import pandas as pd
    df = pd.DataFrame(rows)
    if cols:
        df = df[[c for c in cols if c in df.columns] + [c for c in df.columns if c not in cols]]
    p = OUT / name
    df.to_csv(p, index=False)
    print(f"wrote {p.relative_to(ROOT)} ({len(df)} rows)")
    return df
