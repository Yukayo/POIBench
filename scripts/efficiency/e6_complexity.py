#!/usr/bin/env python
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

import pandas as pd

SYM = {
    "hybrid": dict(build="O(N)", query="O(h + d + K log K)", update="O(d) per insert (sorted-array insert)",
                   memory="O(V + E + N)"),
    "retriever": dict(build="O(N + V log V + 100 V log V) (kNN table)", query="O(h + K log K)",
                      update="rebuild O(N + V log V)", memory="O(100 V + E)"),
    "getnext": dict(build="O(Ep S (L^2 D + L D^2) + Ep V^2 D) (dense POI graph)", query="O(L^2 D + V D)",
                    update="retrain", memory="O(V^2 + V D + params)"),
    "getnext-zfeat": dict(build="as GETNext", query="as GETNext", update="retrain", memory="as GETNext"),
    "graph-flashback": dict(build="O(Ep_kge tr D) + O(Ep S L D^2)", query="O(L D^2 + V D)", update="retrain",
                            memory="O(V k + U V_u + V D)"),
    "ds-llm4poi-lcs": dict(build="none (zero-shot)", query="O(B L^2) LCS + O(T P) prefill/decode",
                           update="none (reads evidence at query time)", memory="O(P)"),
    "llm4poi": dict(build="O(Ep sum_S T P) full fine-tune", query="O(T P), T capped (50 history entries)",
                    update="retrain O(Ep sum_S T P)", memory="O(P) serve; ~16 P bytes train"),
    "gnpr-sid": dict(build="O(Ep V D) SID + O(Ep sum_S T P) LoRA SFT", query="O(Bm T P) beam search",
                     update="retrain (SID + SFT)", memory="O(P + LoRA + V)"),
}
ORDER = tuple(SYM)


def rng(s):
    s = pd.Series(s).dropna()
    return (float(s.min()), float(s.max())) if len(s) else (None, None)


def main():
    rec = pd.read_csv(C.OUT / "e1_cost_records.csv")
    rec = rec[(rec["policy"] == "H-100") & rec["ds"].isin(C.MAIN)]
    hb = pd.read_csv(C.OUT / "e1_hybrid_bench.csv")
    hb = hb[(hb["policy"] == "H-100") & hb["ds"].isin(C.MAIN)]
    up = pd.read_csv(C.OUT / "e2_update_cost.csv")
    up = up[up["ds"].isin(C.MAIN)]
    ret = pd.read_csv(C.OUT / "e4_retriever.csv")
    ret = ret[ret["ds"].isin(C.MAIN)]
    sto = pd.read_csv(C.OUT / "e6_storage.csv")
    e5 = {}
    for p in sorted((C.OUT / "e5").glob("*.json")):
        j = json.load(open(p))
        if not j.get("mock"):
            e5[j["method"]] = j
    rows = []
    for m in ORDER:
        r = {"method": m, "label": C.MLABEL[m], **{f"sym_{k}": v for k, v in SYM[m].items()}}
        g = rec[rec["method"] == m]
        if m == "hybrid":
            r["build_unit"], (r["build_min"], r["build_max"]) = "CPU s", rng(hb["fit_s"])
            r["query_unit"], (r["query_min"], r["query_max"]) = "CPU ms", rng(hb["predict_ms_per_query"])
            r["update_unit"], (r["update_min"], r["update_max"]) = "us/insert", rng(up["insert_us_mean"])
            r["mem_unit"], (r["mem_min"], r["mem_max"]) = "MB state", rng(hb["state_mb"])
            r["peak_gpu_gb_min"] = r["peak_gpu_gb_max"] = 0.0
        elif m == "retriever":
            r["build_unit"], (r["build_min"], r["build_max"]) = "CPU s", rng(ret["build_s"])
            r["query_unit"], (r["query_min"], r["query_max"]) = "CPU ms", rng(ret["lat_mean_ms"])
            r["update_unit"], (r["update_min"], r["update_max"]) = "CPU s rebuild", rng(ret["build_s"])
            r["mem_unit"], (r["mem_min"], r["mem_max"]) = "MB index", rng(ret["index_mb"])
            r["peak_gpu_gb_min"] = r["peak_gpu_gb_max"] = 0.0
        else:
            r["build_unit"], (r["build_min"], r["build_max"]) = "GPU h", rng(g["once_gpu_s"] / 3600)
            r["query_unit"], (r["query_min"], r["query_max"]) = "GPU ms", rng(1000 * g["perq_gpu_s"])
            if m == "ds-llm4poi-lcs":
                r["update_unit"], r["update_min"], r["update_max"] = "none", 0.0, 0.0
            else:
                r["update_unit"], (r["update_min"], r["update_max"]) = "GPU h retrain", rng(g["once_gpu_s"] / 3600)
            s = sto[sto["method"] == m]
            per_city = s.groupby("ds")["disk_mb"].sum()
            r["mem_unit"], (r["mem_min"], r["mem_max"]) = "MB on disk", rng(per_city)
            (r["peak_gpu_gb_min"], r["peak_gpu_gb_max"]) = rng(g["peak_gpu_mem_gb"])
        sv = e5.get(m)
        r["serve_p50_ms_nyc"] = sv["batch1"]["p50_ms"] if sv else None
        r["serve_qps128_nyc"] = sv["throughput"].get("128", {}).get("qps") if sv else None
        rows.append(r)
    C.write_csv("e6_complexity.csv", rows)


if __name__ == "__main__":
    main()
