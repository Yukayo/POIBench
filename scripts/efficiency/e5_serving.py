#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np
import pandas as pd

DS, POL, SPLIT = "nyc", "H-100", "test"
BATCHES = (1, 8, 32, 128)
WARMUP = 8
OUT = ROOT / "results" / "efficiency" / "e5"


def sample_qids(n: int) -> list:
    from poibench.exports import Export, export_dir
    ex = Export(export_dir(DS, POL, str(ROOT)))
    q = ex.queries(SPLIT)["query_id"].to_numpy(np.int64)
    return sorted(int(x) for x in np.random.default_rng(0).choice(q, size=min(n, len(q)), replace=False))


def pct(a, p):
    return float(np.percentile(np.asarray(a), p)) if len(a) else None


class Bench:

    def __init__(self, name, items, qids, f, sync=lambda: None):
        self.name, self.items, self.qids, self.f, self.sync = name, items, qids, f, sync

    def call(self, idx):
        self.sync()
        t0 = time.perf_counter()
        out = self.f([self.items[i] for i in idx])
        self.sync()
        return time.perf_counter() - t0, out

    def run(self, n_lat: int) -> dict:
        n = len(self.items)
        rep = {"n_queries": n, "n_lat": min(n_lat, n), "warmup": WARMUP}
        self.call(list(range(min(WARMUP, n))))
        top1 = [None] * n
        lat = []
        for i in range(min(n_lat, n)):
            dt, o = self.call([i])
            lat.append(dt)
            top1[i] = o[0]
        rep["batch1"] = {"p50_ms": 1000 * pct(lat, 50), "p95_ms": 1000 * pct(lat, 95), "p99_ms": 1000 * pct(lat, 99),
                         "mean_ms": 1000 * float(np.mean(lat)), "qps": len(lat) / float(np.sum(lat))}
        print(f"[e5 {self.name}] batch 1: p50 {rep['batch1']['p50_ms']:.1f} ms, p95 {rep['batch1']['p95_ms']:.1f} ms, "
              f"{rep['batch1']['qps']:.2f} q/s", flush=True)
        rep["throughput"] = {"1": {"qps": rep["batch1"]["qps"], "batch_p50_ms": rep["batch1"]["p50_ms"],
                                   "batch_p95_ms": rep["batch1"]["p95_ms"], "n": len(lat)}}
        for B in [b for b in BATCHES if b > 1] + [n]:
            bl, tot = [], 0.0
            for s in range(0, n, B):
                idx = list(range(s, min(n, s + B)))
                dt, o = self.call(idx)
                bl.append(dt)
                tot += dt
                for i, v in zip(idx, o):
                    top1[i] = v if top1[i] is None else top1[i]
            key = str(B) if B in BATCHES else "all"
            rep["throughput"][key] = {"batch": B, "qps": n / tot, "total_s": tot, "batch_p50_ms": 1000 * pct(bl, 50),
                                      "batch_p95_ms": 1000 * pct(bl, 95), "n": n}
            print(f"[e5 {self.name}] batch {B}: {n / tot:.2f} q/s, batch p50 {1000 * pct(bl, 50):.1f} ms", flush=True)
        ref = ROOT / "preds" / f"{self.name}@{POL}" / DS / f"{SPLIT}.parquet"
        if ref.exists():
            r = pd.read_parquet(ref, columns=["query_id", "ranked_venue_ids"])
            rm = dict(zip(r["query_id"].astype(int), r["ranked_venue_ids"].map(lambda x: int(x[0]) if len(x) else -1)))
            same = [rm.get(q) == t for q, t in zip(self.qids, top1) if t is not None]
            rep["sanity_top1_equals_preds"] = float(np.mean(same)) if same else None
        return rep


def gpu_mem(rep: dict, torch=None):
    if torch is not None and torch.cuda.is_available():
        rep["torch_max_allocated_gb"] = torch.cuda.max_memory_allocated() / 1e9
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        rep["nvml_used_gb"] = pynvml.nvmlDeviceGetMemoryInfo(h).used / 1e9
        rep["gpu_name"] = pynvml.nvmlDeviceGetName(h)
        rep["gpu_name"] = rep["gpu_name"].decode() if isinstance(rep["gpu_name"], bytes) else rep["gpu_name"]
    except Exception as e:
        rep["nvml"] = f"unavailable: {type(e).__name__}"


def first_int(text: str):
    import re
    m = re.search(r"\d+", text or "")
    return int(m.group()) if m else None


def run_ds(a, qids):
    from methods.ds_llm4poi import llm as L
    from methods.ds_llm4poi import parse as P
    pdf = pd.read_parquet(ROOT / "work" / "ds_llm4poi" / "runs" / DS / POL / "lcs" / f"prompts_{SPLIT}.parquet")
    pm = dict(zip(pdf["query_id"].astype(int), pdf["prompt"]))
    prompts = [pm[q] for q in qids]
    t0 = time.perf_counter()
    be = L.MockBackend().load() if a.mock else L.VLLMBackend().load()
    load_s = time.perf_counter() - t0
    def f(ps):
        out = []
        for r in be.generate(ps):
            try:
                v = P.parse_lenient(r["text"])
                out.append(int(v) if v is not None else None)
            except (TypeError, ValueError):
                out.append(None)
        return out
    rep = Bench("ds-llm4poi-lcs", prompts, qids, f).run(a.n_lat)
    rep.update(load_s=load_s, backend=getattr(be, "id", "mock"), engine="vLLM offline LLM.chat, default engine args")
    return rep


def run_llm4poi(a, qids):
    from methods.llm4poi import run as R
    from methods.llm4poi import prompts as PR
    wd = ROOT / "work" / "llm4poi" / DS / POL / "default"
    pdf = pd.read_parquet(wd / f"prompts_{SPLIT}.parquet")
    pm = dict(zip(pdf["query_id"].astype(int), pdf["prompt"]))
    prompts = [pm[q] for q in qids]
    ck = json.load(open(wd / "DONE_train.json"))["final_ckpt"]
    rep0 = {"checkpoint": ck}
    t0 = time.perf_counter()
    if a.mock:
        def gen(ps):
            return [R.mock_text(p) for p in ps]
    else:
        if not os.path.isdir(ck):
            raise SystemExit(f"checkpoint {ck} not found on this host")
        from vllm import LLM, SamplingParams
        tok = ck if os.path.exists(os.path.join(ck, "tokenizer_config.json")) else R.base_model()["path"]
        llm = LLM(model=ck, tokenizer=tok, seed=R.SEED, dtype="bfloat16")
        diff = dict(llm.llm_engine.model_config.get_diff_sampling_param())
        sp = SamplingParams(**dict(diff, **R.SAMPLING))
        rep0["sampling"] = str(sp)

        def gen(ps):
            outs = llm.chat([PR.messages(p) for p in ps], sampling_params=sp, use_tqdm=False)
            return [o.outputs[0].text for o in outs]
    load_s = time.perf_counter() - t0

    def f(ps):
        return [R.parse_lenient(t) for t in gen(ps)]
    rep = Bench("llm4poi", prompts, qids, f).run(a.n_lat)
    rep.update(rep0, load_s=load_s, engine="vLLM offline LLM.chat (one engine, one GPU), the adapter's sampling")
    return rep


def run_gnpr(a, qids):
    from methods.gnpr_sid import data as D
    from methods.gnpr_sid import sid as S
    from poibench.exports import Export, export_dir
    wd = ROOT / "work" / "gnpr_sid" / "runs" / DS / POL
    ex = Export(export_dir(DS, POL, str(ROOT)))
    sids, _ = S.load_sids(str(wd / "sid"))
    p2c = {int(k): D.encode_item(v) for k, v in sids.items()}
    pr = D.eval_prompts(ex, SPLIT, qids, p2c)
    assert pr["query_id"].astype(int).tolist() == qids
    cand = ex.candidates(SPLIT)
    cmap = dict(zip(cand["query_id"].astype(np.int64), cand["cand_venue_ids"]))
    items = list(zip(pr["prompt"].tolist(), [cmap[q] for q in qids]))
    torch = None
    t0 = time.perf_counter()
    if a.mock:
        def f(it):
            return [int(c[0]) for _, c in it]
    else:
        import torch
        from methods.gnpr_sid import llm as L
        g = L.Generator(str(wd / "sft" / "final_sft"), p2c, device="cuda")

        def f(it):
            with torch.inference_mode():
                res = g._batch([p for p, _ in it], [c for _, c in it])
            return [int(b[0][0]) if b else None for b, _ in res]
    load_s = time.perf_counter() - t0
    sync = (lambda: torch.cuda.synchronize()) if torch is not None else (lambda: None)
    rep = Bench("gnpr-sid", items, qids, f, sync).run(a.n_lat)
    rep.update(load_s=load_s, engine="HF transformers + merged LoRA, SID-constrained beam search (10) + candidate "
                                      "scoring (the adapter's Generator._batch; CUDA OOM splits a batch in halves)")
    if not a.mock:
        rep["oom_splits"] = int(getattr(g, "oom_splits", 0))
    gpu_mem(rep, torch)
    return rep


def run_getnext(a, qids):
    from methods.getnext import run as G
    from poibench.exports import Export, export_dir
    import torch
    ex = Export(export_dir(DS, POL, str(ROOT)))
    dev = "cpu" if a.mock else a.device
    ck = ROOT / "work" / "getnext" / DS / POL / "released" / "train" / "best.pt"
    t0 = time.perf_counter()
    model = G.load_model(str(ck), dev)
    load_s = time.perf_counter() - t0
    qs_all = G.query_sequences(ex, SPLIT, qids, model.meta)
    cd = ex.candidates(SPLIT)
    cands = dict(zip(cd["query_id"].to_numpy(np.int64).tolist(), cd["cand_venue_ids"]))
    keys = ("query_id", "user", "poi", "t")

    def f(idx_items):
        idx = list(idx_items)
        qs = {k: [qs_all[k][i] for i in idx] for k in keys}
        qs["n_dropped"] = 0
        sc = G.predict_scores(model, qs, chunk=max(len(idx), 1))
        df, _ = G.rank_lists(model, qs, sc, {q: cands[q] for q in qs["query_id"]})
        return [int(x[0]) for x in df["ranked_venue_ids"]]
    sync = (lambda: torch.cuda.synchronize()) if dev.startswith("cuda") else (lambda: None)
    rep = Bench("getnext", list(range(len(qids))), qids, f, sync).run(a.n_lat)
    rep.update(load_s=load_s, checkpoint=str(ck.relative_to(ROOT)), device=dev,
               engine="the adapter's predict_scores + rank_lists (released cfg), torch eval mode")
    gpu_mem(rep, torch if dev.startswith("cuda") else None)
    return rep


def run_hybrid(a, qids):
    from poibench.baselines_simple import SimpleBaselines
    from poibench.evaluate import candidates_path, load_policy_json
    d = ROOT / "data" / DS
    ck = pd.read_parquet(d / "checkins.parquet")
    venues = pd.read_parquet(d / "venues.parquet")
    q = pd.read_parquet(d / f"queries_{SPLIT}.parquet").set_index("query_id").loc[qids].reset_index()
    pol = load_policy_json(str(ROOT), DS, POL)
    cands = pd.read_parquet(candidates_path(str(ROOT), DS, SPLIT, pol)).set_index("query_id")
    t0 = time.perf_counter()
    model = SimpleBaselines(ck, venues, policy=pol)
    load_s = time.perf_counter() - t0

    def f(idx):
        qq = q.iloc[list(idx)]
        cc = cands.loc[qq["query_id"].tolist()].reset_index()
        pr = model.predict("hybrid", qq, cc, topk=50, n_jobs=1)
        return [int(x[0]) for x in pr["ranked_venue_ids"]]
    rep = Bench("hybrid", list(range(len(qids))), qids, f).run(a.n_lat)
    rep.update(load_s=load_s, device="cpu (1 core)", engine="poibench.baselines_simple.SimpleBaselines.predict, n_jobs=1")
    return rep


def run_gallm(a, qids):
    import importlib.util
    import types
    spec = importlib.util.spec_from_file_location("gallm_infer", ROOT / "methods" / "gallm" / "infer.py")
    I = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(I)
    wd = ROOT / "work" / "gallm" / DS / POL / "default"
    pdf = pd.read_parquet(wd / f"prompts_{SPLIT}.parquet")
    pm = dict(zip(pdf["query_id"].astype(int), pdf["prompt"]))
    prompts = [pm[q] for q in qids]
    ck = ROOT / "work" / "gallm" / "ckpt" / DS / POL / "default"
    s1 = json.load(open(ck / "stage1" / "DONE.json"))
    enc = str(ck / "stage1" / os.path.basename(s1["encoder"]))
    base = os.path.join(os.environ["POIBENCH_MODEL_DIR_X"], "Llama-2-7b-longlora-32k-ft")
    src = sorted(str(p) for p in (ROOT / "work" / "gallm").glob("src_*") if (p / "poibench_vec_geo.py").exists())[-1]
    V = len(pd.read_parquet(ROOT / "exports" / DS / POL / "venues_public.parquet", columns=["venue_id"]))
    args = types.SimpleNamespace(src=src, base_model=base, stage2=str(ck / "stage2"), encoder=enc,
                                 gps_mapping=str(wd / "gps_mapping.csv"), seq_len=32768, context_size=32768,
                                 max_new_tokens=max(4, len(str(V - 1))))
    rep0 = {"checkpoint": str(ck), "encoder": enc}
    t0 = time.perf_counter()
    if a.mock:
        def one(p, q):
            return {"text": str(abs(hash(p)) % 997), "skipped": False}
    else:
        model, tok, gen, _, dev = I.load_model(args)
        import torch

        def one(p, q):
            return I.generate_one(model, tok, gen, dev, p, q, 0)
    load_s = time.perf_counter() - t0
    sync = (lambda: torch.cuda.synchronize()) if not a.mock else (lambda: None)
    n = len(prompts)
    rep = {"n_queries": n, "n_lat": min(a.n_lat, n), "warmup": WARMUP}
    for i in range(min(WARMUP, n)):
        one(prompts[i], qids[i])
    lat, top1, skipped = [], [], 0
    for i in range(min(a.n_lat, n)):
        sync()
        t1 = time.perf_counter()
        r = one(prompts[i], qids[i])
        sync()
        lat.append(time.perf_counter() - t1)
        skipped += int(bool(r.get("skipped")))
        d = "".join(ch for ch in (r.get("text") or "") if ch.isdigit())
        top1.append(int(d) if d and int(d) < V else None)
    rep["batch1"] = {"p50_ms": 1000 * pct(lat, 50), "p95_ms": 1000 * pct(lat, 95), "p99_ms": 1000 * pct(lat, 99),
                     "mean_ms": 1000 * float(np.mean(lat)), "qps": len(lat) / float(np.sum(lat))}
    rep["throughput"] = {"1": {"qps": rep["batch1"]["qps"], "batch_p50_ms": rep["batch1"]["p50_ms"],
                               "batch_p95_ms": rep["batch1"]["p95_ms"], "n": len(lat)}}
    rep["batch_gt1"] = "not supported by the released inference (batch 1 only); not measured"
    rep["skipped_ge_32768_tokens"] = skipped
    print(f"[e5 ga-llm] batch 1: p50 {rep['batch1']['p50_ms']:.1f} ms, p95 {rep['batch1']['p95_ms']:.1f} ms, "
          f"{rep['batch1']['qps']:.2f} q/s ({skipped} skipped)", flush=True)
    ref = ROOT / "preds" / f"ga-llm@{POL}" / DS / f"{SPLIT}.parquet"
    if ref.exists():
        r = pd.read_parquet(ref, columns=["query_id", "answer_venue_id"])
        rm = dict(zip(r["query_id"].astype(int), r["answer_venue_id"].astype(int)))
        same = [rm.get(q) == (t if t is not None else -1) for q, t in zip(qids, top1)]
        rep["sanity_top1_equals_preds"] = float(np.mean(same)) if same else None
    rep.update(rep0, load_s=load_s, engine="HF transformers generate, batch 1 (authors' test_new_fusion_gps.py), "
                                             "4-bit NF4 + LoRA, one GPU; batch > 1 not supported by the release")
    if not a.mock:
        gpu_mem(rep, torch)
    return rep


RUN = {"ds-llm4poi-lcs": run_ds, "llm4poi": run_llm4poi, "gnpr-sid": run_gnpr, "getnext": run_getnext,
       "hybrid": run_hybrid, "ga-llm": run_gallm}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("method", choices=sorted(RUN))
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--n-lat", type=int, default=200)
    ap.add_argument("--mock", action="store_true", help="CPU dry run with a stand-in model (no weights)")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    import platform
    qids = sample_qids(a.n)
    t0 = time.time()
    rep = RUN[a.method](a, qids)
    rep.update(method=a.method, ds=DS, policy=POL, split=SPLIT, mock=a.mock, host=platform.node(),
               created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), wall_total_s=time.time() - t0,
               cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"), qids_sha=hash(tuple(qids)) & 0xFFFFFFFF)
    if a.method in ("ds-llm4poi-lcs", "llm4poi") and not a.mock:
        try:
            import torch
            gpu_mem(rep, torch)
        except Exception:
            gpu_mem(rep)
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / (f"{a.method}{'.mock' if a.mock else ''}.json")
    with open(out, "w") as fh:
        json.dump(rep, fh, indent=1, default=str)
    print(f"[e5] wrote {out.relative_to(ROOT)} ({rep['wall_total_s']:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
