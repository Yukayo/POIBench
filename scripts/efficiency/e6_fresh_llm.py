#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import socket
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

import numpy as np
import pandas as pd

FRESH = C.ROOT / "work" / "efficiency_fresh"
MOCK = bool(os.environ.get("E6_MOCK"))
BACKEND = "mock" if MOCK else "vllm"
OUTB = (FRESH / "mock_out") if MOCK else (C.OUT / "fresh_llm")
NAME = {"llm4poi": "llm4poi-fresh", "ds": "ds-llm4poi-lcs-fresh"}
REPRO_N = 200


def _import(path, name):
    sp = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(mod)
    return mod


def _canary_ok(ds):
    p = C.ROOT / "logs" / "canary" / f"e6_fresh__{ds}__H-100__test.json"
    return p.exists() and json.load(open(p))["result"] == "PASS"


def _now():
    return pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


def _write(name, ds, out, cost):
    d = OUTB / name / ds
    d.mkdir(parents=True, exist_ok=True)
    out.to_parquet(d / "test.parquet.tmp", index=False)
    os.replace(d / "test.parquet.tmp", d / "test.parquet")
    json.dump(cost, open(d / "test.cost.json", "w"), indent=1, default=str)
    print(f"wrote {d.relative_to(C.ROOT)} ({len(out)} queries)", flush=True)


def _gpus():
    v = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    return [g for g in v.split(",") if g.strip()]


def run_llm4poi(datasets):
    R = _import(C.ROOT / "methods" / "llm4poi" / "run.py", "e6_l4p_run")
    from poibench.exports import Export
    ckroot = Path(os.environ.get("POIBENCH_LLM4POI_CKPT_ROOT") or (C.ROOT / "work" / "llm4poi" / "ckpt"))
    base_cache = os.environ.get("LLM4POI_CACHE_DIR")
    for ds in datasets:
        if not _canary_ok(ds):
            raise SystemExit(f"{ds}: fresh canary not PASS")
        rec = json.load(open(C.ROOT / "preds" / "llm4poi@H-100" / ds / "test.cost.json"))
        done = json.load(open(ckroot / ds / "H-100" / "default" / "DONE.json"))
        rel = done["final_ckpt"].split("/H-100/default/", 1)[1]
        model_dir = str(ckroot / ds / "H-100" / "default" / rel)
        t0 = time.perf_counter()
        mfp = R.model_fingerprint(model_dir)
        if mfp != rec["model_fingerprint"]:
            raise SystemExit(f"{ds}: checkpoint fingerprint {mfp[:16]} != recorded {rec['model_fingerprint'][:16]}")
        wd = FRESH / ("mock_llm4poi" if MOCK else "llm4poi") / ds
        wd.mkdir(parents=True, exist_ok=True)
        fp = pd.read_parquet(C.ROOT / "work" / "llm4poi" / ds / "H-100" / "default" / "prompts_test.parquet")
        fr = pd.read_parquet(C.ROOT / "work" / "llm4poi" / ds / "H-100" / "default" / "responses_test.parquet")
        smp = np.sort(np.random.default_rng(0).choice(len(fp), size=min(REPRO_N, len(fp)), replace=False))
        os.environ["LLM4POI_CACHE_DIR"] = str(FRESH / "repro_cache" / "llm4poi")
        rr, _ = R.generate(fp["prompt"].iloc[smp].tolist(), None if MOCK else model_dir, "mock-v1" if MOCK else mfp,
                           BACKEND, str(wd / "repro"))
        ft = dict(zip(fr["query_id"], fr["text"]))
        repro = float(np.mean([a["text"] == ft[q] for a, q in zip(rr, fp["query_id"].iloc[smp])]))
        if base_cache:
            os.environ["LLM4POI_CACHE_DIR"] = base_cache
        else:
            os.environ.pop("LLM4POI_CACHE_DIR", None)
        print(f"[llm4poi {ds}] reproduction of {len(smp)} frozen answers: {repro:.4f}", flush=True)
        ex = Export(str(FRESH / "exports" / ds / "H-100"))
        qids = [int(x) for x in ex.queries("test")["query_id"].to_numpy(np.int64)]
        t1 = time.perf_counter()
        pr = R.PR.query_prompts(ex, "test", qids, "default")
        prep_s = time.perf_counter() - t1
        pr.to_parquet(wd / "prompts_test.parquet", index=False)
        frozen_same = float(np.mean(pr["prompt_sha256"].to_numpy() == fp.set_index("query_id").loc[qids, "prompt_sha256"].to_numpy()))
        res, gst = R.generate(pr["prompt"].tolist(), None if MOCK else model_dir, "mock-v1" if MOCK else mfp,
                              BACKEND, str(wd))
        out = R.parse_outputs(qids, res, ex, "test")
        out.assign(prompt_sha256=pr["prompt_sha256"].to_numpy()).to_parquet(wd / "responses_test.parquet", index=False)
        preds = pd.DataFrame({"query_id": out["query_id"], "ranked_venue_ids": out["ranked_lenient"],
                              "answer_venue_id": out["answer_lenient"]})
        ngpu = max(1, gst.get("n_workers") or 0)
        cost = dict(method=NAME["llm4poi"], ds=ds, split="test", exploratory=True, evidence_rule="fresh",
                    created_utc=_now(), host=socket.gethostname(), gpus=_gpus(), checkpoint=model_dir,
                    model_fingerprint=mfp, frozen_run=rec.get("created_utc"), cfg="default", seed=0,
                    sampling=gst.get("sampling"), history_limit=R.PR.HISTORY_LIMIT,
                    repro_check=dict(n=int(len(smp)), identical_text_share=repro),
                    prompt_stats=dict(n=len(pr), mean_n_hist=float(pr["n_hist"].mean()),
                                      mean_n_cur=float(pr["n_cur"].mean()),
                                      share_prompt_identical_to_frozen=frozen_same),
                    frozen_prompt_stats=rec.get("prompt_stats"),
                    generation=gst, tokens=dict(prompt_mean=float(out["n_prompt_tokens"].mean()),
                                                output_mean=float(out["n_output_tokens"].mean())),
                    frozen_tokens=rec.get("tokens"),
                    parse_rates=dict(lenient_valid=float(out["valid_lenient"].mean()),
                                     strict_valid=float(out["valid_strict"].mean())),
                    stages=[dict(name="prepare_prompts", wall_s=prep_s, gpu_s=0.0, n_items=len(pr)),
                            dict(name="infer", wall_s=gst["gen_wall_s"], gpu_s=gst["gen_wall_s"] * ngpu, n_gpus=ngpu,
                                 n_items=len(pr), n_generated=gst["n_generated"], n_cache_hits=gst["n_cache_hits"],
                                 notes="vLLM data-parallel incl. engine start; generated prompts only")],
                    total_wall_s=time.perf_counter() - t0)
        _write(NAME["llm4poi"], ds, preds, cost)


def run_ds(datasets):
    D = _import(C.ROOT / "methods" / "ds_llm4poi" / "run.py", "e6_ds_run")
    from poibench.cost import CostRecorder
    from poibench.exports import Export
    base_cache = os.environ.get("DSLLM4POI_CACHE_DIR")
    for ds in datasets:
        if not _canary_ok(ds):
            raise SystemExit(f"{ds}: fresh canary not PASS")
        rec = json.load(open(C.ROOT / "preds" / "ds-llm4poi-lcs@H-100" / ds / "test.cost.json"))
        t0 = time.perf_counter()
        wd = FRESH / ("mock_ds" if MOCK else "ds") / ds
        wd.mkdir(parents=True, exist_ok=True)
        fdir = C.ROOT / "work" / "ds_llm4poi" / "runs" / ds / "H-100" / "lcs"
        fp = pd.read_parquet(fdir / "prompts_test.parquet")
        fr = pd.read_parquet(fdir / "responses_test.parquet")
        smp = np.sort(np.random.default_rng(0).choice(len(fp), size=min(REPRO_N, len(fp)), replace=False))
        os.environ["DSLLM4POI_CACHE_DIR"] = str(FRESH / "repro_cache" / "ds")
        rr, _, _ = D.run_generation(fp.iloc[smp].reset_index(drop=True), BACKEND, f"e6-repro/{ds}", None)
        ft = dict(zip(fr["query_id"], fr["text"]))
        repro = float(np.mean([a["text"] == ft[q] for a, q in zip(rr, fp["query_id"].iloc[smp])]))
        if base_cache:
            os.environ["DSLLM4POI_CACHE_DIR"] = base_cache
        else:
            os.environ.pop("DSLLM4POI_CACHE_DIR", None)
        print(f"[ds {ds}] reproduction of {len(smp)} frozen answers: {repro:.4f}", flush=True)
        fx = str(FRESH / "exports" / ds / "H-100")
        ex = Export(fx)
        t1 = time.perf_counter()
        df = D.PR.build_prompts(fx, "test", "lcs", None, n_jobs=D._n_jobs(8))
        prep_s = time.perf_counter() - t1
        df.to_parquet(wd / "prompts_test.parquet", index=False)
        frozen_same = float(np.mean(df["prompt_sha256"].to_numpy() ==
                                    fp.set_index("query_id").loc[df["query_id"].to_numpy(), "prompt_sha256"].to_numpy()))
        cr = CostRecorder(method=NAME["ds"], ds=ds, split="test")
        cr.add_stage(name="prepare_prompts", wall_s=prep_s, cpu_core_s=0.0, gpu_s=0.0, peak_mem_gb=0.0, one_time=False,
                     n_items=len(df))
        res, gst, be = D.run_generation(df, BACKEND, f"e6-fresh/{ds}", cr)
        out = D.parse_outputs(df, res, ex, "test")
        out.assign(prompt_sha256=df["prompt_sha256"].to_numpy()).to_parquet(wd / "responses_test.parquet", index=False)
        preds = pd.DataFrame({"query_id": out["query_id"], "ranked_venue_ids": out["ranked_lenient"],
                              "answer_venue_id": out["answer_lenient"]})
        cost = dict(method=NAME["ds"], ds=ds, split="test", exploratory=True, evidence_rule="fresh", created_utc=_now(),
                    host=socket.gethostname(), gpus=_gpus(), backend=be.id, cfg="lcs", seed=0,
                    repro_check=dict(n=int(len(smp)), identical_text_share=repro),
                    prompt_stats=dict(n=len(df), mean_n_demos=float(df["n_demos"].mean()),
                                      mean_n_bank=float(df["n_bank"].mean()),
                                      mean_n_context=float(df["n_context"].mean()),
                                      share_no_demo=float((df["n_demos"] == 0).mean()),
                                      share_prompt_identical_to_frozen=frozen_same),
                    frozen_prompt_stats=rec.get("prompt_stats"), generation=gst,
                    tokens=dict(prompt_mean=float(out["n_prompt_tokens"].mean()),
                                output_mean=float(out["n_output_tokens"].mean())),
                    frozen_tokens=rec.get("tokens"),
                    parse_rates=dict(lenient_valid=float(out["valid_lenient"].mean()),
                                     strict_valid=float(out["valid_strict"].mean())),
                    stages=cr.stages, total_wall_s=time.perf_counter() - t0)
        _write(NAME["ds"], ds, preds, cost)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("method", choices=["llm4poi", "ds"])
    ap.add_argument("--datasets", nargs="+", default=list(C.MAIN))
    a = ap.parse_args()
    todo = [d for d in a.datasets if not (OUTB / NAME[a.method] / d / "test.parquet").exists()]
    print(f"[e6 fresh {a.method}] {_now()} host={socket.gethostname()} gpus={_gpus()} todo={todo}", flush=True)
    (run_llm4poi if a.method == "llm4poi" else run_ds)(todo)
    print(f"[e6 fresh {a.method}] {_now()} done", flush=True)


if __name__ == "__main__":
    main()
