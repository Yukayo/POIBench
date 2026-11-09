#!/usr/bin/env python
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

import numpy as np
import pandas as pd

R = C.ROOT
RT3 = Path(__import__("os").environ.get("POIBENCH_RUNTIME_DIR") or str(R / "work" / "runtime"))
STAGE_MAP = {
    "hybrid": {"load_data": "prepare", "fit_counts": "index", "predict": "infer"},
    "getnext": {"prepare": "prepare", "train": "train", "predict_test": "infer", "predict_val": "infer"},
    "getnext-zfeat": {"prepare": "prepare", "train": "train", "predict_test": "infer", "predict_val": "infer"},
    "graph-flashback": {"kge_triplets": "prepare", "train_data": "prepare", "kge_transe": "index", "graphs": "index",
                        "train": "train", "predict_test": "infer", "predict_val": "infer"},
    "gnpr-sid": {"prepare": "prepare", "sid": "index", "sft": "train", "model_load": "load", "infer": "infer",
                 "infer_cached": "infer"},
    "ds-llm4poi-lcs": {"prepare_prompts": "prepare", "model_load": "load", "infer": "infer", "infer_cached": "infer"},
    "llm4poi": {"prepare_prompts": "prepare", "train": "train", "infer": "infer", "infer_cached": "infer"},
}
CANON = ("prepare", "index", "train", "load", "infer")


def records(m, ds):
    exc = {sp: R / "results" / "cost_exclusive" / f"{m}@H-100" / ds / f"{sp}.cost.json" for sp in ("val", "test")}
    src = "exclusive" if all(p.exists() for p in exc.values()) else "preds"
    out = {}
    for sp in ("val", "test"):
        p = exc[sp] if src == "exclusive" else R / "preds" / f"{m}@H-100" / ds / f"{sp}.cost.json"
        if p.exists():
            out[sp] = json.load(open(p))
    return out, src


def stages():
    rows = []
    for m, mp in STAGE_MAP.items():
        for ds in C.MAIN:
            recs, src = records(m, ds)
            if "test" not in recs:
                continue
            nq = recs["test"].get("n_queries") or 0
            acc = {(c, o): {"gpu_s": 0.0, "cpu_core_s": 0.0, "wall_s": 0.0, "peak_gpu_mem_gb": 0.0,
                            "peak_gpu_mem_nvml_gb": 0.0, "peak_mem_gb": 0.0} for c in CANON for o in (True, False)}
            seen = set()
            for sp in ("val", "test"):
                for st in recs.get(sp, {}).get("stages", []):
                    c = mp.get(st["name"])
                    if c is None:
                        raise KeyError(f"unmapped stage {m} {st['name']}")
                    if st.get("one_time"):
                        if st["name"] in seen:
                            continue
                        seen.add(st["name"])
                        scale = 1.0
                    else:
                        if sp != "test":
                            continue
                        scale = 1000.0 / nq
                    a = acc[(c, bool(st.get("one_time")))]
                    a["peak_gpu_mem_nvml_gb"] = max(a["peak_gpu_mem_nvml_gb"], float(st.get("peak_gpu_mem_nvml_gb") or 0.0))
                    for k in ("gpu_s", "cpu_core_s", "wall_s"):
                        a[k] += scale * float(st.get(k) or 0.0)
                    a["peak_gpu_mem_gb"] = max(a["peak_gpu_mem_gb"], float(st.get("peak_gpu_mem_gb") or 0.0))
                    a["peak_mem_gb"] = max(a["peak_mem_gb"], float(st.get("peak_mem_gb") or 0.0))
            for (c, o), a in acc.items():
                if not any(a[k] for k in ("gpu_s", "cpu_core_s", "wall_s")):
                    continue
                rows.append({"method": m, "ds": ds, "stage": c, "unit": "once" if o else "per 1k test queries",
                             "source": src, "n_test_queries": nq, **a})
    ret = pd.read_csv(C.OUT / "e4_retriever.csv")
    for _, r in ret[ret["ds"].isin(C.MAIN)].iterrows():
        rows.append({"method": "retriever", "ds": r["ds"], "stage": "index", "unit": "once", "source": "e4",
                     "n_test_queries": r["n_test_queries"], "gpu_s": 0.0, "cpu_core_s": r["build_s"],
                     "wall_s": r["build_s"]})
        rows.append({"method": "retriever", "ds": r["ds"], "stage": "infer", "unit": "per 1k test queries",
                     "source": "e4", "n_test_queries": r["n_test_queries"], "gpu_s": 0.0,
                     "cpu_core_s": r["lat_mean_ms"], "wall_s": r["lat_mean_ms"]})
    df = C.write_csv("e6_stages.csv", rows)
    return df


def fsize(paths):
    tot = 0
    for p in paths:
        p = Path(p)
        if p.is_file():
            tot += p.stat().st_size
        elif p.is_dir():
            tot += sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    return tot


def model_bytes(d: Path):
    return sum(f.stat().st_size for f in d.glob("*.safetensors"))


def storage():
    hb = pd.read_csv(C.OUT / "e1_hybrid_bench.csv")
    hb = hb[hb["policy"] == "H-100"].set_index("ds")
    ret = pd.read_csv(C.OUT / "e4_retriever.csv").set_index("ds")
    q7 = model_bytes(RT3 / "models" / "Qwen2.5-7B-Instruct")
    l8 = model_bytes(RT3 / "models_x" / "Llama-3.1-8B-Instruct")
    rows = []
    for ds in C.MAIN:
        cj = {m: (C.cost_json(m, "H-100", ds) or {}) for m in STAGE_MAP}
        vis = R / "exports" / ds / "H-100" / "visible_rows.parquet"

        def peak(m, key):
            recs, _ = records(m, ds)
            v = [float(s.get(key) or 0) for r in recs.values() for s in r.get("stages", [])]
            return max(v) if v else None

        def add(m, artefact, nbytes, kind, note="", mem_gb=None, gpu_gb=None):
            rows.append({"method": m, "ds": ds, "artefact": artefact, "disk_mb": nbytes / 2 ** 20, "kind": kind,
                         "note": note, "inmem_gb": mem_gb, "peak_gpu_gb": gpu_gb})

        add("hybrid", "statistics (popularity, transitions; rebuilt from the check-in table)",
            hb.loc[ds, "state_mb"] * 2 ** 20, "measured", "in-memory state; on disk only the check-in table",
            hb.loc[ds, "peak_rss_gb"], 0.0)
        add("hybrid", "visible check-in table (exports visible_rows.parquet)", fsize([vis]), "measured")
        add("retriever", "index (popularity, transitions, 100-NN table)", ret.loc[ds, "index_mb"] * 2 ** 20,
            "measured", "in-memory index", None, 0.0)
        for m, sub in (("getnext", "released"), ("getnext-zfeat", "zfeat")):
            d = R / "work" / "getnext" / ds / "H-100"
            add(m, "model (best.pt)", fsize([d / sub / "train" / "best.pt"]), "measured", "",
                peak(m, "peak_mem_gb"), peak(m, "peak_gpu_mem_gb"))
            add(m, "POI graph + features (graph_A.npz, graph_X.csv)",
                fsize([d / "prepare" / "graph_A.npz", d / "prepare" / "graph_X.csv"]), "measured")
        g = cj["graph-flashback"]
        if g:
            tdir = R / g["train_dir"]
            pdir = R / g["prepare_dir"]
            add("graph-flashback", "model (model.pt)", fsize([tdir / "model.pt"]), "measured", "",
                peak("graph-flashback", "peak_mem_gb"), peak("graph-flashback", "peak_gpu_mem_gb"))
            add("graph-flashback", "KGE + POI graphs (transe.ckpt, *.pkl, index.npz)",
                fsize([pdir / "transe.ckpt", pdir / "index.npz"] + list(pdir.glob("*.pkl"))), "measured")
        add("ds-llm4poi-lcs", "base LLM Qwen2.5-7B-Instruct (shared by all cities, no per-city state)", q7,
            "measured", "vLLM reserves ~94 GB (NVML); model weights 14.19 GiB", None,
            peak("ds-llm4poi-lcs", "peak_gpu_mem_nvml_gb"))
        add("llm4poi", "full fine-tuned Qwen2.5-7B checkpoint (one per city)", q7, "proxy",
            "second-host copy only; same shapes and dtype as Qwen2.5-7B-Instruct", None,
            peak("llm4poi", "peak_gpu_mem_gb"))
        gd = R / "work" / "gnpr_sid" / "runs" / ds / "H-100"
        add("gnpr-sid", "LoRA adapter (adapter_model.safetensors)", fsize([gd / "sft" / "adapter_model.safetensors"]),
            "measured", "", None, peak("gnpr-sid", "peak_gpu_mem_gb"))
        add("gnpr-sid", "semantic-ID codebook + POI embeddings (sid/)", fsize([gd / "sid"]), "measured")
        add("gnpr-sid", "base LLM Meta-Llama-3-8B-Instruct (shared)", l8, "proxy",
            "second-host copy only; Llama-3.1-8B-Instruct safetensors here: same parameter count and dtype")
    return C.write_csv("e6_storage.csv", rows)


if __name__ == "__main__":
    stages()
    storage()
