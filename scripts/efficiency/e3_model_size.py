#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

import numpy as np
import pandas as pd

FAMILIES = [
    ("ds-llm4poi-lcs", "ds-llm4poi-lcs", "ds-llm4poi-lcs-",
     {"qwen2.5-3b": 3.09, "qwen2.5-7b": 7.62, "qwen2.5-14b": 14.77, "qwen2.5-32b": 32.76, "llama3.1-8b": 8.03,
      "mistral-7b-v0.3": 7.25}, "qwen2.5-7b"),
    ("llm4poi", "llm4poi", "llm4poi-",
     {"qwen2.5-1.5b": 1.54, "qwen2.5-3b": 3.09, "qwen2.5-7b": 7.62, "llama3.1-8b": 8.03}, "qwen2.5-7b"),
    ("gnpr-sid", "gnpr-sid", "gnpr-sid-",
     {"qwen2.5-7b": 7.62, "qwen2.5-14b": 14.77, "qwen2.5-32b": 32.76, "llama3-8b": 8.03}, "llama3-8b"),
    ("ga-llm", "ga-llm", "ga-llm-", {"llama2-7b-longlora": 6.74}, "llama2-7b-longlora"),
]


def peak_torch(method, ds):
    best = None
    for sp in ("val", "test"):
        j = C.cost_json(method, "H-100", ds, sp)
        for st in (j or {}).get("stages", []):
            v = st.get("peak_gpu_mem_gb")
            if v:
                best = max(best or 0.0, float(v))
    return best


def main():
    from poibench import evaluate as ev
    ev.assert_split_allowed(str(C.ROOT), "test")
    rows = []
    for ds in C.MAIN:
        names = {}
        for fam, prim, pre, keys, pk in FAMILIES:
            for k, nb in keys.items():
                m = prim if k == pk else f"{pre}{k}"
                if (C.ROOT / "preds" / f"{m}@H-100" / ds / "test.parquet").exists():
                    names[m] = (fam, k, nb, k == pk)
        if not names:
            continue
        queries, cands, venues, checkins = ev.load_split(str(C.ROOT), ds, "test", "H-100")
        es = ev.build_eval_set(queries, cands, venues, checkins=checkins)
        W = ev.bootstrap_weights(es.n_users, B=1000, seed=0)
        uc, nu = es.user_codes, es.n_users
        n_u = np.bincount(uc, minlength=nu).astype(float)
        Wn = W @ n_u

        def hits(m):
            df = pd.read_parquet(C.ROOT / "preds" / f"{m}@H-100" / ds / "test.parquet", columns=["query_id", "ranked_venue_ids"])
            rk, _ = ev.per_query_ranks(es, df)
            return (rk["e2e"] <= 1).astype(float)
        yh = hits("hybrid")
        bh = W @ np.bincount(uc, weights=yh, minlength=nu) / Wn
        for m, (fam, k, nb, prim) in names.items():
            y = hits(m)
            b = W @ np.bincount(uc, weights=y, minlength=nu) / Wn
            c = C.mt.load_cost(m, "H-100", ds)
            j = C.cost_json(m, "H-100", ds, "test") or {}
            q = (c or {}).get("perq", {}).get("test") or {}
            tok = j.get("tokens") or {}
            rows.append({"family": fam, "backbone": k, "params_b": nb, "primary": prim, "method": m, "ds": ds,
                         "acc1": float(y.mean()), "acc1_ci_lo": float(np.percentile(b, 2.5)),
                         "acc1_ci_hi": float(np.percentile(b, 97.5)), "dacc1_vs_hybrid": float(y.mean() - yh.mean()),
                         "dacc1_ci_lo": float(np.percentile(b - bh, 2.5)), "dacc1_ci_hi": float(np.percentile(b - bh, 97.5)),
                         "once_gpu_h": (c["once"]["gpu_s"] / 3600) if c else None,
                         "gpus_per_1k": 1000 * (q.get("gpu_s") or 0) if c else None,
                         "peak_gpu_mem_gb_torch": peak_torch(m, ds), "peak_gpu_mem_gb_any": c["peak_gpu_mem"] if c else None,
                         "prompt_tokens_mean": tok.get("prompt_mean"), "output_tokens_mean": tok.get("output_mean"),
                         "colocated": c.get("colocated") if c else None, "cost_source": c.get("source") if c else None})
            print(f"{ds:10s} {m:34s} acc1={y.mean():.4f} once={rows[-1]['once_gpu_h']} perK={rows[-1]['gpus_per_1k']}", flush=True)
    C.write_csv("e3_model_size.csv", rows)


if __name__ == "__main__":
    main()
