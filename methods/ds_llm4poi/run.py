#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd

from poibench.cost import CostRecorder
from poibench.exports import Export, export_dir
from methods.ds_llm4poi import apply_patches
from methods.ds_llm4poi import llm as L
from methods.ds_llm4poi import parse as PA
from methods.ds_llm4poi import prompts as PR

METHOD = "ds-llm4poi"
TOPK = 50
SEED = 0


def log(*a):
    print(f"[ds_llm4poi {time.strftime('%H:%M:%S')}]", *a, flush=True)


def method_tag(cfg: str, strict: bool = False, model_key: str = L.DEFAULT_MODEL_KEY) -> str:
    mk = "" if model_key == L.DEFAULT_MODEL_KEY else f"-{model_key}"
    return f"{METHOD}-{cfg}{mk}" + ("-strictparse" if strict else "")


def candidates_file_decl(ex: Export, split: str) -> str:
    keys = [k for k in ex.manifest["source_sha256"] if k.endswith(f"candidates_{split}.parquet")]
    if len(keys) != 1:
        raise RuntimeError(f"cannot identify the candidate file of {split} in the export manifest: {keys}")
    return os.path.join(ex.manifest["source_dir"], keys[0])


def _sha_ids(ids) -> str:
    return hashlib.sha256(",".join(str(int(x)) for x in ids).encode()).hexdigest()


def _write_parquet(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path + ".tmp", index=False)
    os.replace(path + ".tmp", path)


def _write_json(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f, indent=1, default=str)
    os.replace(path + ".tmp", path)


_BACKENDS = {}


def get_backend(kind: str, model_key: str = L.DEFAULT_MODEL_KEY):
    if (kind, model_key) not in _BACKENDS:
        _BACKENDS[(kind, model_key)] = L.BACKENDS[kind](*L.MODELS[model_key]) if kind == "vllm" else L.BACKENDS[kind]()
    return _BACKENDS[(kind, model_key)]


def n_gpus_for(backend_kind: str) -> int:
    return 1 if backend_kind == "vllm" else 0


def prepare_split(ex_dir: str, ex: Export, ds: str, split: str, cfg: str, qids, wd: str, n_jobs: int):
    path = os.path.join(wd, f"prompts_{split}.parquet")
    mpath = os.path.join(wd, f"prompts_{split}.json")
    fp = dict(export_files=ex.manifest.get("files"), cfg=cfg, code=PR.code_fingerprint(),
              query_ids_sha256=_sha_ids(qids), n_queries=len(qids), num_demo=PR.NUM_DEMO, seed=PR.RANDOM_SEED)
    if os.path.exists(path) and os.path.exists(mpath):
        with open(mpath) as f:
            meta = json.load(f)
        if meta.get("fingerprint") == fp:
            log(f"{split}: reusing prompts {os.path.relpath(path, ROOT)}")
            return pd.read_parquet(path), meta
    rec = CostRecorder()
    with rec.stage("prepare_prompts", one_time=False, n_gpus=0, n_items=len(qids),
                   notes="demo selection + prompt construction on CPU (per query)"):
        tt = PR.load_target_time(ds, split) if PR.CFGS[cfg]["target_time"] else None
        df = PR.build_prompts(ex_dir, split, cfg, qids, n_jobs=n_jobs, target_time=tt)
    st = rec.stages[0]
    stats = dict(n_queries=int(len(df)), mean_n_demos=float(df["n_demos"].mean()),
                 share_no_demo=float((df["n_demos"] == 0).mean()), share_lt5_demos=float((df["n_demos"] < 5).mean()),
                 mean_n_bank=float(df["n_bank"].mean()), mean_n_context=float(df["n_context"].mean()),
                 mean_prompt_chars=float(df["prompt"].str.len().mean()),
                 max_prompt_chars=int(df["prompt"].str.len().max()))
    meta = dict(fingerprint=fp, stage=st, stats=stats)
    _write_parquet(df, path)
    _write_json(meta, mpath)
    log(f"{split}: built {len(df)} prompts in {st['wall_s']:.1f} s -> {os.path.relpath(path, ROOT)} {stats}")
    return df, meta


def parse_outputs(prompts_df: pd.DataFrame, res: list, ex: Export, split: str) -> pd.DataFrame:
    V = len(ex.venues)
    cmap = dict(zip(ex.candidates(split)["query_id"].astype(np.int64), ex.candidates(split)["cand_venue_ids"]))
    rows = []
    for qid, r in zip(prompts_df["query_id"].to_numpy(np.int64), res):
        a_len, a_str = PA.parse_lenient(r["text"]), PA.parse_strict(r["text"])
        lst_l, ok_l = PA.ranked_list(a_len, cmap[int(qid)], V, TOPK)
        lst_s, ok_s = PA.ranked_list(a_str, cmap[int(qid)], V, TOPK)
        rows.append(dict(query_id=int(qid), ranked_lenient=lst_l, ranked_strict=lst_s,
                         answer_lenient=int(a_len) if ok_l else -1, answer_strict=int(a_str) if ok_s else -1,
                         parsed_lenient=a_len is not None, parsed_strict=a_str is not None,
                         valid_lenient=ok_l, valid_strict=ok_s, text=r["text"],
                         n_prompt_tokens=int(r["n_prompt_tokens"]), n_output_tokens=int(r["n_output_tokens"]),
                         finish_reason=str(r["finish_reason"])))
    return pd.DataFrame(rows)


def run_generation(prompts_df: pd.DataFrame, backend_kind: str, job: str, rec: CostRecorder | None,
                   model_key: str = L.DEFAULT_MODEL_KEY):
    be = get_backend(backend_kind, model_key)
    cache = L.ResponseCache(be.id, cache_dir=os.environ.get("DSLLM4POI_CACHE_DIR") or None)
    ng = n_gpus_for(backend_kind)
    keys = [L.cache_key(be.id, p) for p in prompts_df["prompt"]]
    pre = {k for k in set(keys) if k in cache.mem}

    def on_load(f):
        if rec is None:
            return f()
        with rec.stage("model_load", one_time=True, n_gpus=ng,
                       notes=f"{be.id}; once per process (recorded in the first split's cost.json)"):
            f()

    if rec is None:
        res, gst = L.generate_cached(list(prompts_df["prompt"]), be, cache, ng, job=job, log=log, on_load=on_load)
    else:
        loaded_here = False
        if not be.loaded and any(k not in cache.mem for k in set(keys)):
            on_load(be.load)
            loaded_here = True
        with rec.stage("infer", one_time=False, n_gpus=ng, n_items=len(prompts_df),
                       notes="LLM generation of the prompts not in the response cache") as st:
            res, gst = L.generate_cached(list(prompts_df["prompt"]), be, cache, ng, job=job, log=log,
                                         on_load=on_load)
            st["n_generated"] = gst["n_generated"]
            st["n_cache_hits"] = gst["n_cache_hits"]
        gst["loaded"] = gst["loaded"] or loaded_here
        if pre:
            g = sum(float(cache.mem[k]["gen_s"]) for k in pre)
            gg = sum(float(cache.mem[k]["gen_s"]) * float(cache.mem[k]["n_gpus"]) for k in pre)
            rec.add_stage("infer_cached", wall_s=g, cpu_core_s=0.0, gpu_s=gg, peak_mem_gb=0.0, one_time=False,
                          n_items=len(pre), notes="generation time attributed from the cache shards that produced "
                                                  "these answers (earlier attempt or another job with the same prompt)")
    return res, gst, be


def predict_split(a, split: str, ex_dir: str, ex: Export, qids, wd: str, out_base: str, write_preds: bool,
                  pdir: str | None = None):
    pdf, pmeta = prepare_split(ex_dir, ex, a.ds, split, a.cfg, qids, pdir or wd, _n_jobs(a.n_jobs))
    tag = method_tag(a.cfg, model_key=a.model)
    rec = CostRecorder(method=f"{tag}@{a.policy}", ds=a.ds, split=split)
    rec.add_stage(**{k: v for k, v in pmeta["stage"].items() if k != "name"}, name="prepare_prompts")
    res, gst, be = run_generation(pdf, a.backend, f"{tag}@{a.policy}/{a.ds}/{split}", rec, a.model)
    out = parse_outputs(pdf, res, ex, split)
    _write_parquet(out.assign(prompt_sha256=pdf["prompt_sha256"].to_numpy()),
                   os.path.join(wd, f"responses_{split}.parquet"))
    n = len(out)
    fr = out["finish_reason"].value_counts().to_dict()
    up = PR.upstream().info
    common = dict(
        evidence_policy=a.policy, cfg=a.cfg, seed=SEED, zero_shot=True, backend=be.id,
        model=L.MODELS[a.model][0], model_key=a.model, model_revision=getattr(be, "revision", None),
        model_revision_pinned=L.MODELS[a.model][1],
        model_source=getattr(be, "src", None), sampling=L.SAMPLING, engine_seed=L.ENGINE_SEED,
        num_demo=PR.NUM_DEMO, data_bank=PR.DATA_BANK, demo_selection=PR.CFGS[a.cfg]["demo_selection"],
        similarity=PR.CFGS[a.cfg]["similarity"], target_time_in_prompt=PR.CFGS[a.cfg]["target_time"],
        candidates_file=candidates_file_decl(ex, split), output="[model answer] + policy candidates (dedup, top 50)",
        upstream_commit=up["commit"], upstream_patches=up["patches"], export_manifest_files=ex.manifest.get("files"),
        prompt_stats=pmeta["stats"], generation=dict(gst, finish_reasons=fr),
        tokens=dict(prompt_total=int(out["n_prompt_tokens"].sum()), output_total=int(out["n_output_tokens"].sum()),
                    prompt_mean=float(out["n_prompt_tokens"].mean()), output_mean=float(out["n_output_tokens"].mean())),
        parse_rates=dict(lenient_parsed=float(out["parsed_lenient"].mean()),
                         lenient_valid=float(out["valid_lenient"].mean()),
                         strict_parsed=float(out["parsed_strict"].mean()),
                         strict_valid=float(out["valid_strict"].mean())),
        fill_share=dict(lenient=float(1 - out["valid_lenient"].mean()), strict=float(1 - out["valid_strict"].mean())))
    paths = []
    for strict in (False, True):
        t = method_tag(a.cfg, strict, a.model)
        d = os.path.join(out_base, f"{t}@{a.policy}", a.ds)
        col = "strict" if strict else "lenient"
        preds = pd.DataFrame({"query_id": out["query_id"], "ranked_venue_ids": out[f"ranked_{col}"],
                              "answer_venue_id": out[f"answer_{col}"]})
        cr = CostRecorder(method=f"{t}@{a.policy}", ds=a.ds, split=split,
                          extra=dict(common, parse=col, primary=not strict,
                                     shares_generations_with=(f"{tag}@{a.policy}" if strict else None)))
        cr.stages = list(rec.stages)
        if write_preds:
            _write_parquet(preds, os.path.join(d, f"{split}.parquet"))
            cr.save(os.path.join(d, f"{split}.cost.json"), n_queries=n)
            paths.append(os.path.relpath(os.path.join(d, f"{split}.parquet"), ROOT))
    log(f"{split}: {n} queries; parse lenient {common['parse_rates']['lenient_valid']:.3f} valid, strict "
        f"{common['parse_rates']['strict_valid']:.3f} valid; finish {fr}; generated {gst['n_generated']}, "
        f"cache hits {gst['n_cache_hits']} -> {paths}")
    return out, common


def _n_jobs(n: int) -> int:
    return 1 if any(getattr(b, "loaded", False) for b in _BACKENDS.values()) else n


def predict(export_dir, split, query_ids, work_dir=None):
    cfg = os.environ.get("DSLLM4POI_CFG", "lcs")
    if cfg not in PR.CFGS:
        raise ValueError(f"DSLLM4POI_CFG={cfg!r} is not a DS-LLM4POI configuration ({sorted(PR.CFGS)})")
    kind = os.environ.get("DSLLM4POI_BACKEND", "vllm")
    mkey = os.environ.get("DSLLM4POI_MODEL", L.DEFAULT_MODEL_KEY)
    if PR.CFGS[cfg]["target_time"]:
        raise ValueError(f"cfg {cfg} is the labelled target-time ablation; the leakage gate covers lcs/random only")
    qids = [int(x) for x in query_ids]
    nj = 1 if any(getattr(b, "loaded", False) for b in _BACKENDS.values()) else int(os.environ.get("DSLLM4POI_NJOBS", "4"))
    df = PR.build_prompts(export_dir, split, cfg, qids, n_jobs=nj)
    if work_dir:
        _write_parquet(df, os.path.join(work_dir, "prompts.parquet"))
    res, _, _ = run_generation(df, kind, f"predict:{cfg}", None, mkey)
    out = parse_outputs(df, res, Export(export_dir), split)
    if work_dir:
        _write_parquet(out, os.path.join(work_dir, "responses.parquet"))
    ret = pd.DataFrame({"query_id": out["query_id"], "ranked_venue_ids": out["ranked_lenient"]})
    ret.attrs["adapter_config"] = dict(cfg=cfg, backend=kind, model=mkey,
                                       demo_selection=PR.CFGS[cfg]["demo_selection"])
    return ret


def token_stats(a) -> dict:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(L.MODEL, revision=L.REVISION, local_files_only=True)
    ed = export_dir(a.ds, a.policy, ROOT)
    ex = Export(ed)
    out = {}
    for split in a.split:
        q = ex.queries(split)["query_id"].to_numpy(np.int64)
        if a.sample and a.sample < len(q):
            q = np.sort(np.random.default_rng(SEED).choice(q, size=a.sample, replace=False))
        tt = PR.load_target_time(a.ds, split) if PR.CFGS[a.cfg]["target_time"] else None
        t0 = time.perf_counter()
        df = PR.build_prompts(ed, split, a.cfg, q, n_jobs=a.n_jobs, target_time=tt)
        t_build = time.perf_counter() - t0
        texts = [tok.apply_chat_template(L.messages(p), tokenize=False, add_generation_prompt=True)
                 for p in df["prompt"]]
        n = np.array([len(x) for x in tok(texts, add_special_tokens=False)["input_ids"]])
        ans = '```json\n{"place_id": 1234, "place_category": "American Restaurant"}\n```'
        n_out = len(tok(ans, add_special_tokens=False)["input_ids"]) + 1
        n_all = int(len(ex.queries(split)))
        out[split] = dict(n_sampled=int(len(q)), n_queries_split=n_all, prompt_tokens_mean=float(n.mean()),
                          p50=float(np.median(n)), p95=float(np.percentile(n, 95)), max=int(n.max()),
                          min=int(n.min()), est_output_tokens=n_out,
                          est_prompt_tokens_split=float(n.mean() * n_all),
                          est_gpu_h_split_at_prefill_tok_s={str(r): float(n.mean() * n_all / r / 3600)
                                                            for r in (5000, 10000, 20000)},
                          mean_n_demos=float(df["n_demos"].mean()), share_no_demo=float((df["n_demos"] == 0).mean()),
                          prompt_build_s=round(t_build, 1), prompt_build_s_per_query=round(t_build / len(df), 5))
        log(f"tokens {a.ds}/{a.policy}/{a.cfg}/{split}: {out[split]}")
    rep = dict(ds=a.ds, policy=a.policy, cfg=a.cfg, tokenizer=f"{L.MODEL}@{L.REVISION}",
               chat_template="tokenizer default (Qwen2.5 adds its default system message)", splits=out)
    p = os.path.join(ROOT, "logs", "ds_llm4poi", f"tokens_{a.ds}_{a.policy}_{a.cfg}.json")
    _write_json(rep, p)
    log(f"-> {os.path.relpath(p, ROOT)}")
    return rep


def main(argv=None):
    ap = argparse.ArgumentParser(description=(__doc__ or " "), formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", nargs="?", choices=["prepare", "train", "predict", "all", "smoke", "tokens"])
    ap.add_argument("--ds", required=True, nargs="+")
    ap.add_argument("--policy", required=True)
    ap.add_argument("--split", "--splits", dest="split", nargs="+", default=["val", "test"], choices=["val", "test"])
    ap.add_argument("--cfg", default="lcs", choices=sorted(PR.CFGS))
    ap.add_argument("--model", default=L.DEFAULT_MODEL_KEY, choices=sorted(L.MODELS),
                    help="backbone key (exploratory LLM-scaling study; default = the primary Qwen2.5-7B-Instruct)")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--backend", default=None, choices=sorted(L.BACKENDS),
                    help="default: vllm on cuda, mock on cpu (mock output never goes to preds/)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-jobs", type=int, default=8, help="CPU workers for prompt building (<= 24)")
    ap.add_argument("--limit", type=int, default=0, help="only n seed-0 sampled queries per split (smoke; work/ only)")
    ap.add_argument("--n", type=int, default=50, help="smoke: number of queries")
    ap.add_argument("--sample", type=int, default=0, help="tokens: queries sampled per split (0 = all)")
    ap.add_argument("--canary-prep", action="store_true")
    a = ap.parse_args(argv)
    if a.seed != 0:
        ap.error("seed 0 only (owner rule)")
    a.n_jobs = max(1, min(a.n_jobs, 24))
    if a.canary_prep:
        log(f"zero-shot method: nothing to train for the canary; patched upstream at {apply_patches.ensure_patched()}")
        return 0
    if a.cmd is None:
        ap.error("a subcommand is required (prepare|train|predict|all|smoke|tokens) unless --canary-prep")
    if a.backend is None:
        a.backend = "vllm" if a.device == "cuda" else "mock"
    if a.backend == "vllm" and a.device != "cuda":
        ap.error("the vllm backend needs --device cuda")
    rc = 0
    for ds in a.ds:
        b = argparse.Namespace(**vars(a))
        b.ds = ds
        rc = rc or _main_one(b)
    return rc


def _main_one(a) -> int:
    if a.cmd == "tokens":
        token_stats(a)
        return 0
    if a.cmd == "smoke":
        a.limit, a.backend = a.n, "mock"
        if a.device != "cpu":
            raise SystemExit("smoke is the CPU check (--device cpu)")
    ed = export_dir(a.ds, a.policy, ROOT)
    if not os.path.exists(os.path.join(ed, "MANIFEST.json")):
        raise SystemExit(f"{ed} missing: run scripts/export_inputs.py --datasets {a.ds} --policies {a.policy}")
    ex = Export(ed)
    mode = "smoke" if a.limit else ("runs" if a.backend == "vllm" else "mock")
    pdir = os.path.join(ROOT, "work", "ds_llm4poi", mode, a.ds, a.policy, a.cfg)
    wd = pdir if a.model == L.DEFAULT_MODEL_KEY else f"{pdir}@{a.model}"
    write_preds_to_root = (mode == "runs")
    out_base = os.path.join(ROOT, "preds") if write_preds_to_root else os.path.join(wd, "preds")
    log(f"{a.cmd} ds={a.ds} policy={a.policy} cfg={a.cfg} model={a.model} splits={a.split} backend={a.backend} mode={mode} "
        f"upstream={apply_patches.ensure_patched()}")

    def qids_for(split):
        q = ex.queries(split)["query_id"].to_numpy(np.int64)
        if a.limit and a.limit < len(q):
            q = np.sort(np.random.default_rng(SEED).choice(q, size=a.limit, replace=False))
        return [int(x) for x in q]

    if a.cmd in ("prepare", "all", "predict", "smoke"):
        for s in a.split:
            prepare_split(ed, ex, a.ds, s, a.cfg, qids_for(s), pdir, _n_jobs(a.n_jobs))
    if a.cmd in ("train", "all"):
        log("train: DS-LLM4POI is zero-shot (in-context demonstrations only); nothing to train")
        _write_json(dict(zero_shot=True, note="no training stage; demonstrations are selected per query"),
                    os.path.join(wd, "train.json"))
    if a.cmd in ("predict", "all", "smoke"):
        summary = {}
        for s in a.split:
            out, common = predict_split(a, s, ed, ex, qids_for(s), wd, out_base, write_preds=True, pdir=pdir)
            summary[s] = dict(n=len(out), parse_rates=common["parse_rates"], prompt_stats=common["prompt_stats"],
                              tokens=common["tokens"], generation=common["generation"])
        if a.cmd == "smoke":
            ex_prompt = pd.read_parquet(os.path.join(pdir, f"prompts_{a.split[0]}.parquet"))["prompt"].iloc[0]
            with open(os.path.join(wd, "example_prompt.txt"), "w") as f:
                f.write(ex_prompt + "\n")
            _write_json(summary, os.path.join(wd, "smoke_summary.json"))
            log(f"smoke summary -> {os.path.relpath(os.path.join(wd, 'smoke_summary.json'), ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
