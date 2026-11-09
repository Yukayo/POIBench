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
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("WANDB_DISABLED", "true")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd

from poibench.cost import CostRecorder
from poibench.exports import Export, export_dir
from methods.gnpr_sid import data as D
from methods.gnpr_sid import upstream as U

METHOD = "gnpr-sid"
CFG_ENV = "GNPR_SID_CFG"


def method_name(cfg: str = "default") -> str:
    bb = U.backbone()
    return (METHOD if cfg == "default" else f"{METHOD}-{cfg}") + ("" if bb == U.DEFAULT_BACKBONE else f"-{bb}")
VERSION = "s2"
SEED = 0
TOPK = 50
SHARD = 1024
SMOKE_SFT_STEPS = 100
SMOKE_N_QUERIES = 200
SMOKE_MIN_DISTINCT_TOP1 = 10
CANARY_SID_EPOCHS = 250
CANARY_SFT_STEPS = 10


def log(*a):
    print(f"[gnpr_sid {time.strftime('%H:%M:%S')}]", *a, flush=True)


def _write_parquet(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path + ".tmp", index=False)
    os.replace(path + ".tmp", path)


def _write_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f, indent=1, default=str)
    os.replace(path + ".tmp", path)


def _read_json(path):
    with open(path) as f:
        return json.load(f)


def candidates_file_decl(ex: Export, split: str) -> str:
    src = ex.manifest.get("source_sha256", {})
    keys = [k for k in src if k.endswith(f"candidates_{split}.parquet")]
    if len(keys) != 1:
        raise RuntimeError(f"cannot identify the {split} candidate file in the export MANIFEST: {keys}")
    return f"{ex.manifest['source_dir']}/{keys[0]}"


def work_dir_for(mode: str, ds: str, policy: str, cfg: str = "default") -> str:
    base = os.environ.get("GNPR_SID_WORK") or os.path.join(ROOT, "work", "gnpr_sid")
    bb = U.backbone()
    sub = (mode if cfg == "default" else f"{mode}_{cfg}") + ("" if bb == U.DEFAULT_BACKBONE else f"_bb-{bb}")
    return os.path.join(base, sub, ds, policy)


def reuse_primary_sid(wd: str, ds: str, policy: str, cfg: str, ex: Export | None = None) -> None:
    import shutil
    if U.backbone() == U.DEFAULT_BACKBONE or os.path.exists(os.path.join(wd, "sid", "SID_DONE.json")):
        return
    base = os.environ.get("GNPR_SID_WORK") or os.path.join(ROOT, "work", "gnpr_sid")
    src = os.path.join(base, "runs" if cfg == "default" else f"runs_{cfg}", ds, policy)
    if not os.path.exists(os.path.join(src, "sid", "SID_DONE.json")):
        log(f"no primary SIDs at {src}: training SIDs for this backbone")
        return
    if ex is not None:
        pf = _read_json(os.path.join(src, "prepare.json"))["fingerprint"]
        fp, _, _ = inputs_fingerprint(ex, cfg)
        for k in ("poi_info_sha256", "sft_samples_venue_sha256"):
            if fp[k] != pf[k]:
                raise RuntimeError(f"cannot reuse the primary SIDs of {src}: {k} differs ({fp[k][:12]} vs {pf[k][:12]})")
    os.makedirs(wd, exist_ok=True)
    tmp = os.path.join(wd, "sid.tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    for dp, _, fns in os.walk(os.path.join(src, "sid")):
        rel = os.path.relpath(dp, os.path.join(src, "sid"))
        d = tmp if rel == "." else os.path.join(tmp, rel)
        os.makedirs(d, exist_ok=True)
        for fn in fns:
            shutil.copyfile(os.path.join(dp, fn), os.path.join(d, fn))
    os.replace(tmp, os.path.join(wd, "sid"))
    st = _read_json(os.path.join(src, "train_cost.json"))["stages"]
    stages = {k: dict(st[k], notes=st[k].get("notes", "") + f" [copied from the primary run {os.path.relpath(src, ROOT)}; "
                      "backbone-independent]") for k in ("prepare", "sid") if k in st}
    _write_json(dict(stages=stages, sid_copied_from=os.path.relpath(src, ROOT)), os.path.join(wd, "train_cost.json"))
    log(f"SIDs copied from the primary run {os.path.relpath(src, ROOT)}")


def inputs_fingerprint(ex: Export, cfg: str = "default") -> dict:
    pi = D.poi_info(ex)
    samples, st = D.sft_samples(ex, None, cfg)
    return dict(poi_info_sha256=hashlib.sha256(D.poi_info_csv_bytes(pi)).hexdigest(),
                sft_samples_venue_sha256=D.samples_sha256(samples), n_sft_samples=st["n_samples"],
                n_catalog=int(len(pi))), pi, st


def prepare(ex: Export, wd: str, cfg: str = "default") -> dict:
    p = os.path.join(wd, "prepare.json")
    if os.path.exists(p) and os.path.exists(os.path.join(wd, "poi_info.csv")):
        meta = _read_json(p)
        if meta.get("export_files") == ex.manifest.get("files"):
            return meta
    rec = CostRecorder()
    with rec.stage("prepare", one_time=True, n_gpus=0, notes="poi_info + SFT-sample construction (CPU)"):
        fp, pi, st = inputs_fingerprint(ex, cfg)
        os.makedirs(wd, exist_ok=True)
        with open(os.path.join(wd, "poi_info.csv.tmp"), "wb") as f:
            f.write(D.poi_info_csv_bytes(pi))
        os.replace(os.path.join(wd, "poi_info.csv.tmp"), os.path.join(wd, "poi_info.csv"))
    meta = dict(fingerprint=fp, sft_stats=st, export_files=ex.manifest.get("files"), stage=rec.stages[0])
    _write_json(meta, p)
    log(f"prepare: {fp['n_catalog']} catalog POIs, {st['n_samples']} SFT samples from {st['n_users']} users "
        f"(segments {st['n_segments']}, <20 rows {st['n_lt_min_len']}, target not a train query "
        f"{st['n_target_not_train_query']})")
    return meta


def train(ex: Export, wd: str, device: str, sid_epochs=None, sft_steps=None, cfg: str = "default") -> dict:
    from methods.gnpr_sid import llm as L
    from methods.gnpr_sid import sid as S
    pm = prepare(ex, wd, cfg)
    cpath = os.path.join(wd, "train_cost.json")
    stages = _read_json(cpath)["stages"] if os.path.exists(cpath) else {}
    stages.setdefault("prepare", pm["stage"])
    if "sid" not in stages or not os.path.exists(os.path.join(wd, "sid", "SID_DONE.json")):
        rec = CostRecorder()
        with rec.stage("sid", one_time=True, n_gpus=1 if device.startswith("cuda") else 0,
                       notes="category2vec + POI2emb + CRQVAE training (authors' Trainer) + SID assignment"):
            S.run_sid(os.path.join(wd, "poi_info.csv"), os.path.join(wd, "sid"), device, epochs=sid_epochs,
                      seed=SEED, log=log)
        stages["sid"] = rec.stages[0]
        _write_json(dict(stages=stages), cpath)
    sids, sid_stats = S.load_sids(os.path.join(wd, "sid"))
    pid2code = {int(k): D.encode_item(v) for k, v in sids.items()}
    samples, st = D.sft_samples(ex, pid2code, cfg)
    sp = os.path.join(wd, "sft_train.json")
    if not os.path.exists(sp):
        _write_json(samples, sp)
    sft_dir = os.path.join(wd, "sft")
    if "sft" not in stages or not os.path.exists(os.path.join(sft_dir, "SFT_DONE.json")):
        rec = CostRecorder()
        with rec.stage("sft", one_time=True, n_gpus=1, n_items=len(samples),
                       notes=f"LoRA SFT {L.LLM_REPO}, {L.SFT_CALL}, max_steps={sft_steps}") as srec:
            meta = L.sft(samples, sft_dir, max_steps=sft_steps, seed=SEED, log=log)
            srec["resumed_from"] = meta.get("resumed_from")
        prev = stages.get("sft_attempts", [])
        stages["sft_attempts"] = prev + [rec.stages[0]]
        st_all = stages["sft_attempts"]
        stages["sft"] = dict(rec.stages[0], wall_s=sum(s["wall_s"] for s in st_all),
                             cpu_core_s=sum(s["cpu_core_s"] for s in st_all),
                             gpu_s=sum(s["gpu_s"] for s in st_all),
                             peak_mem_gb=max(s["peak_mem_gb"] for s in st_all), n_attempts=len(st_all))
        _write_json(dict(stages=stages), cpath)
    sft_meta = _read_json(os.path.join(sft_dir, "SFT_DONE.json"))
    train_meta = dict(fingerprint=pm["fingerprint"], sft_samples_sha256=D.samples_sha256(samples),
                      sid_stats=sid_stats, sft=sft_meta, upstream=U.info(), adapter_version=VERSION,
                      sid_epochs_override=sid_epochs, sft_max_steps=sft_steps, cfg=cfg)
    _write_json(train_meta, os.path.join(wd, "train_meta.json"))
    return dict(pid2code=pid2code, stages=stages, meta=train_meta, adapter_dir=os.path.join(sft_dir, "final_sft"))


_GEN = {}


def L_infer_batch() -> int:
    from methods.gnpr_sid import llm as L
    return L.INFER_BATCH


def get_generator(adapter_dir: str, pid2code: dict, rec: CostRecorder | None):
    from methods.gnpr_sid import llm as L
    key = (os.path.abspath(adapter_dir), hashlib.sha256(json.dumps(sorted(pid2code.items())).encode()).hexdigest())
    if key not in _GEN:
        _GEN.clear()
        dev = os.environ.get("GNPR_SID_DEVICE", "cuda")
        if rec is None:
            _GEN[key] = L.Generator(adapter_dir, pid2code, device=dev)
        else:
            with rec.stage("model_load", one_time=True, n_gpus=1,
                           notes="base model + LoRA merge + SID trie (once per process)"):
                _GEN[key] = L.Generator(adapter_dir, pid2code, device=dev)
    return _GEN[key]


def rank_outputs(qids, results, cand_map) -> pd.DataFrame:
    rows = []
    for q, (beam, cs) in zip(qids, results):
        cands = [int(x) for x in cand_map[int(q)]]
        lst = []
        for v, _ in beam:
            if v not in lst:
                lst.append(v)
        nb = len(lst)
        for v in cands:
            if len(lst) >= TOPK:
                break
            if v not in lst:
                lst.append(v)
        _, first = np.unique(np.array(cands), return_index=True)
        uc = [cands[i] for i in sorted(first)]
        sc = {v: s for v, s in zip(cands, cs)}
        cr = [v for _, v in sorted(((-(sc[v] if np.isfinite(sc[v]) else -1e30), i), v) for i, v in enumerate(uc))]
        rows.append(dict(query_id=int(q), ranked_venue_ids=lst[:TOPK], cand_ranked_venue_ids=cr, n_beam=nb,
                         beam_scores=[float(s) for _, s in beam]))
    return pd.DataFrame(rows)


def predict_split(ex: Export, split: str, qids, tr: dict, wd: str, rec: CostRecorder, write_to: str | None,
                  cfg: str = "default"):
    pid2code = tr["pid2code"]
    tt = D.load_target_time(ex.manifest.get("ds"), split) if D.CFGS[cfg]["target_time"] else None
    prompts = D.eval_prompts(ex, split, qids, pid2code, cfg, tt)
    cand = ex.candidates(split)
    cand_map = dict(zip(cand["query_id"].astype(np.int64), cand["cand_venue_ids"]))
    fp = hashlib.sha256((tr["meta"]["sft_samples_sha256"] + json.dumps(tr["meta"]["sid_stats"], sort_keys=True,
                         default=str) + "".join(prompts["prompt"]) + VERSION).encode()).hexdigest()[:16]
    sdir = os.path.join(wd, f"infer_{split}_{fp}")
    os.makedirs(sdir, exist_ok=True)
    parts = []
    gen = None

    def _cached(k, chunk):
        sp = os.path.join(sdir, f"shard_{k // SHARD:05d}.parquet")
        return os.path.exists(sp) and pd.read_parquet(sp, columns=["query_id"])["query_id"].tolist() == \
            chunk["query_id"].tolist()

    if any(not _cached(k, prompts.iloc[k:k + SHARD]) for k in range(0, len(prompts), SHARD)):
        gen = get_generator(tr["adapter_dir"], pid2code, rec)
    with rec.stage("infer", one_time=False, n_gpus=1, n_items=len(prompts),
                   notes="SID-constrained beam search (10) + scoring of the 50 candidates; shards not in the "
                         "resume cache") as st:
        n_new, reused_wall = 0, 0.0
        for k in range(0, len(prompts), SHARD):
            sp = os.path.join(sdir, f"shard_{k // SHARD:05d}.parquet")
            chunk = prompts.iloc[k:k + SHARD]
            if os.path.exists(sp):
                df = pd.read_parquet(sp)
                if df["query_id"].tolist() == chunk["query_id"].tolist():
                    parts.append(df)
                    reused_wall += float(df["shard_wall_s"].iloc[0])
                    continue
            if gen is None:
                gen = get_generator(tr["adapter_dir"], pid2code, rec)
            t0 = time.time()
            res = gen.run(chunk["prompt"].tolist(), [cand_map[int(q)] for q in chunk["query_id"]], log=log)
            df = rank_outputs(chunk["query_id"].tolist(), res, cand_map)
            df["shard_wall_s"] = time.time() - t0
            _write_parquet(df, sp)
            parts.append(df)
            n_new += len(chunk)
            log(f"{split}: shard {k // SHARD} ({len(chunk)} queries) {time.time() - t0:.0f} s")
        st["n_generated"] = n_new
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if reused_wall > 0:
        rec.add_stage("infer_cached", wall_s=reused_wall, cpu_core_s=0.0, gpu_s=reused_wall, peak_mem_gb=0.0,
                      one_time=False, n_items=int(len(out) - st["n_generated"]), n_gpus=1,
                      notes="wall time of inference shards computed by an earlier attempt (resume)")
    stats = dict(n_queries=int(len(out)), oom_splits=(None if gen is None else int(getattr(gen, "oom_splits", 0))),
                 infer_batch=int(L_infer_batch()), mean_beam=float(out["n_beam"].mean()),
                 share_full_beam=float((out["n_beam"] >= 10).mean()),
                 fill_share_top10=float(np.mean([max(0, 10 - min(n, 10)) / 10 for n in out["n_beam"]])),
                 n_distinct_top1=int(out["ranked_venue_ids"].map(lambda x: x[0]).nunique()),
                 mean_prompt_items=float(prompts["n_items"].mean()),
                 share_rows_dropped_no_sid=float(prompts["n_dropped_no_sid"].sum() /
                                                 max(1, prompts["n_items"].sum() + prompts["n_dropped_no_sid"].sum())))
    if write_to:
        _write_parquet(out[["query_id", "ranked_venue_ids", "cand_ranked_venue_ids", "n_beam"]],
                       os.path.join(write_to, f"{split}.parquet"))
    return out, stats


def write_cost(ex: Export, split: str, a, tr: dict, rec: CostRecorder, stats: dict, out_dir: str, n: int):
    from methods.gnpr_sid import llm as L
    cr = CostRecorder(method=f"{method_name(a.cfg)}@{a.policy}", ds=a.ds, split=split, extra=dict(
        evidence_policy=a.policy, seed=SEED, adapter_version=VERSION, model=L.LLM_REPO,
        model_revision=L.base_model_revision(), model_revision_pinned=L.LLM_REVISION,
        candidates_file=candidates_file_decl(ex, split),
        output="ranked_venue_ids = beam SIDs (<= 10, constrained beam search) + policy candidates (dedup, top 50); "
               "cand_ranked_venue_ids = the 50 policy candidates by length-normalised SID log-probability",
        cfg=a.cfg, target_time_in_prompt=D.CFGS[a.cfg]["target_time"], sft_call=L.SFT_CALL, num_beams=L.NUM_BEAMS, infer_batch=L.INFER_BATCH,
        upstream=U.info(), export_manifest_files=ex.manifest.get("files"), pred_stats=stats,
        train_meta={k: tr["meta"][k] for k in ("fingerprint", "sft_samples_sha256", "sid_stats")},
        sft_labels=tr["meta"]["sft"].get("labels"), one_time_note="one-time stages repeated in val and test cost.json"))
    st = tr["stages"]
    for name in ("prepare", "sid", "sft"):
        s = st.get(name)
        if s is not None:
            cr.add_stage(**{k: v for k, v in s.items() if k != "name"}, name=name)
    for s in rec.stages:
        cr.stages.append(s)
    cr.save(os.path.join(out_dir, f"{split}.cost.json"), n_queries=n)


def predict(export_dir, split, query_ids, work_dir=None):
    from methods.gnpr_sid import sid as S
    ex = Export(export_dir)
    ds, pol = ex.manifest.get("ds"), ex.manifest.get("policy")
    cfg = os.environ.get(CFG_ENV, "default")
    ck = os.environ.get("GNPR_SID_CKPT") or work_dir_for("canary", ds, pol, cfg)
    tm = _read_json(os.path.join(ck, "train_meta.json"))
    if tm.get("cfg", "default") != cfg:
        raise RuntimeError(f"checkpoint {ck} was trained with cfg {tm.get('cfg', 'default')}, not {cfg}")
    fp, _, _ = inputs_fingerprint(ex, cfg)
    for k in ("poi_info_sha256", "sft_samples_venue_sha256"):
        if fp[k] != tm["fingerprint"][k]:
            raise RuntimeError(f"training input {k} derived from {export_dir} differs from the checkpoint's: the "
                               f"training stages would see different data (leak through training)")
    sids, _ = S.load_sids(os.path.join(ck, "sid"))
    pid2code = {int(k): D.encode_item(v) for k, v in sids.items()}
    qids = [int(x) for x in query_ids]
    tt = D.load_target_time(ds, split) if D.CFGS[cfg]["target_time"] else None
    prompts = D.eval_prompts(ex, split, qids, pid2code, cfg, tt)
    if work_dir:
        _write_parquet(prompts, os.path.join(work_dir, "prompts.parquet"))
    cand = ex.candidates(split)
    cand_map = dict(zip(cand["query_id"].astype(np.int64), cand["cand_venue_ids"]))
    gen = get_generator(os.path.join(ck, "sft", "final_sft"), pid2code, None)
    res = gen.run(prompts["prompt"].tolist(), [cand_map[q] for q in qids], log=log)
    if getattr(gen, "oom_splits", 0):
        log(f"canary predict: {gen.oom_splits} OOM batch splits so far (batch composition differs from an OOM-free run)")
    out = rank_outputs(qids, res, cand_map)
    if work_dir:
        _write_parquet(out, os.path.join(work_dir, "preds.parquet"))
    ret = out[["query_id", "ranked_venue_ids", "cand_ranked_venue_ids"]].copy()
    ret.attrs["adapter_config"] = dict(cfg=cfg, checkpoint_cfg=tm.get("cfg", "default"),
                                       target_time_in_prompt=bool(D.CFGS[cfg]["target_time"]),
                                       backbone=os.environ.get("GNPR_SID_BACKBONE"))
    return ret


def main(argv=None):
    ap = argparse.ArgumentParser(description=(__doc__ or " "), formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", nargs="?", default="all", choices=["prepare", "train", "predict", "all"])
    ap.add_argument("--ds", required=True)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--splits", "--split", dest="splits", nargs="+", default=["val", "test"], choices=["val", "test"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--canary-prep", action="store_true")
    ap.add_argument("--cfg", default="default", choices=sorted(D.CFGS))
    ap.add_argument("--backbone", default=None, help="LLM backbone key (llm.BACKBONES; default $GNPR_SID_BACKBONE or "
                                                     "llama3-8b)")
    ap.add_argument("--probe-mem", action="store_true")
    ap.add_argument("--probe-one", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if a.seed != 0:
        ap.error("seed 0 only (owner rule)")
    if a.backbone:
        if "methods.gnpr_sid.llm" in sys.modules and sys.modules["methods.gnpr_sid.llm"].BACKBONE != a.backbone:
            raise SystemExit("llm module already imported with another backbone")
        os.environ["GNPR_SID_BACKBONE"] = a.backbone
    from methods.gnpr_sid import llm as L
    if L.BACKBONE not in L.BACKBONES:
        raise SystemExit(f"unknown backbone {L.BACKBONE}")
    ed = export_dir(a.ds, a.policy, ROOT)
    if not os.path.exists(os.path.join(ed, "MANIFEST.json")):
        raise SystemExit(f"{ed} missing: run scripts/export_inputs.py --datasets {a.ds} --policies {a.policy}")
    ex = Export(ed)
    U.verify_clone()
    if D.CFGS[a.cfg]["target_time"] and not os.path.exists(os.path.join(ROOT, "exports_ablation", a.ds,
                                                                          "MANIFEST.json")):
        raise SystemExit(f"cfg {a.cfg} needs exports_ablation/{a.ds}/ (scripts/export_target_time.py --ds {a.ds})")
    if a.canary_prep:
        wd = work_dir_for("canary", a.ds, a.policy, a.cfg)
        log(f"canary-prep {a.ds}/{a.policy} cfg={a.cfg} -> {os.path.relpath(wd, ROOT)}")
        train(ex, wd, a.device, sid_epochs=CANARY_SID_EPOCHS, sft_steps=CANARY_SFT_STEPS, cfg=a.cfg)
        return 0
    mode = "smoke" if a.smoke else ("probe" if (a.probe_mem or a.probe_one) else "runs")
    wd = work_dir_for(mode, a.ds, a.policy, a.cfg)
    if a.probe_mem:
        return probe_mem(ex, wd, a.cfg)
    if a.probe_one:
        return probe_one(ex, wd, a.cfg)
    if mode == "runs":
        reuse_primary_sid(wd, a.ds, a.policy, a.cfg, ex)
    log(f"{a.cmd} ds={a.ds} policy={a.policy} cfg={a.cfg} splits={a.splits} mode={mode} resume={os.environ.get('POIBENCH_RESUME')} "
        f"attempt={os.environ.get('POIBENCH_ATTEMPT')} wd={os.path.relpath(wd, ROOT)}")
    if a.cmd == "prepare":
        prepare(ex, wd, a.cfg)
        return 0
    tr = train(ex, wd, a.device, sft_steps=SMOKE_SFT_STEPS if a.smoke else None, cfg=a.cfg)
    if a.cmd == "train":
        return 0
    summary = {}
    for split in a.splits:
        q = ex.queries(split)["query_id"].to_numpy(np.int64)
        if a.smoke:
            q = np.sort(np.random.default_rng(SEED).choice(q, size=min(SMOKE_N_QUERIES, len(q)), replace=False))
        qids = [int(x) for x in q]
        rec = CostRecorder()
        out_dir = (os.path.join(wd, "preds") if a.smoke
                   else os.path.join(ROOT, "preds", f"{method_name(a.cfg)}@{a.policy}", a.ds))
        out, stats = predict_split(ex, split, qids, tr, wd, rec, out_dir, a.cfg)
        write_cost(ex, split, a, tr, rec, stats, out_dir, len(out))
        summary[split] = dict(stats, stages={s["name"]: round(s["wall_s"], 1) for s in rec.stages})
        log(f"{split}: {stats} -> {os.path.relpath(out_dir, ROOT)}")
    if a.smoke:
        summary["projection"] = smoke_projection(ex, wd, tr, summary)
        _write_json(summary, os.path.join(wd, "smoke_summary.json"))
        log(f"smoke summary -> {os.path.relpath(os.path.join(wd, 'smoke_summary.json'), ROOT)}: "
            f"{json.dumps(summary['projection'])}")
        v = summary.get("val", {})
        ok = v.get("share_full_beam") == 1.0 and v.get("n_distinct_top1", 0) >= SMOKE_MIN_DISTINCT_TOP1
        log(f"smoke validity: {'PASS' if ok else 'FAIL'} (share_full_beam {v.get('share_full_beam')}, "
            f"n_distinct_top1 {v.get('n_distinct_top1')})")
        return 0 if ok else 3
    return 0


PROBE_MAX_ALLOC_GIB = 86.0
PROBE_LADDER = (8, 4, 2, 1)


def probe_mem(ex: Export, wd: str, cfg: str) -> int:
    import subprocess
    os.makedirs(wd, exist_ok=True)
    ds, pol = ex.manifest.get("ds"), ex.manifest.get("policy")
    attempts, chosen = [], None
    for b in PROBE_LADDER:
        one = os.path.join(wd, f"probe_b{b}.json")
        if os.path.exists(one):
            os.remove(one)
        env = dict(os.environ, GNPR_SID_MICRO_BATCH=str(b), GNPR_SID_PROBE_OUT=one)
        cmd = [sys.executable, "-I", os.path.abspath(__file__), "--ds", ds, "--policy", pol, "--seed", "0",
               "--cfg", cfg, "--backbone", U.backbone(), "--probe-one"]
        t0 = time.time()
        rc = subprocess.run(cmd, env=env, cwd=ROOT).returncode
        r = _read_json(one) if os.path.exists(one) else dict(oom=None, error=f"child rc {rc}, no result file")
        r.update(micro_batch=b, rc=rc, child_wall_s=time.time() - t0)
        attempts.append(r)
        log(f"probe micro-batch {b}: rc {rc}, oom {r.get('oom')}, peak allocated {r.get('peak_alloc_gib')}")
        if rc == 0 and r.get("oom") is False and r.get("peak_alloc_gib", 1e9) <= PROBE_MAX_ALLOC_GIB:
            chosen = b
            break
    rep = dict(backbone=U.backbone(), micro_batch=chosen, max_alloc_gib=PROBE_MAX_ALLOC_GIB, attempts=attempts)
    _write_json(rep, os.path.join(wd, "probe_mem.json"))
    log(f"probe_mem: micro_batch={chosen}")
    return 0


def probe_one(ex: Export, wd: str, cfg: str) -> int:
    import shutil
    import torch
    from methods.gnpr_sid import llm as L
    from methods.gnpr_sid import sid as S
    src = os.path.join(ROOT, "work", "gnpr_sid", "runs", ex.manifest.get("ds"), ex.manifest.get("policy"), "sid")
    sids, _ = S.load_sids(src)
    pid2code = {int(k): D.encode_item(v) for k, v in sids.items()}
    samples, _ = D.sft_samples(ex, pid2code, cfg)
    n = 16 * 2
    samples = sorted(samples, key=lambda x: -len(x["input"]))[:n]
    out = os.path.join(wd, f"probe_work_b{L.SFT_CALL['batch_size']}")
    shutil.rmtree(out, ignore_errors=True)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    oom, meta = False, {}
    try:
        meta = L.sft(samples, out, max_steps=2, seed=SEED, log=log)
    except torch.cuda.OutOfMemoryError as e:
        oom, meta = True, dict(error=str(e)[:500])
    rep = dict(backbone=L.BACKBONE, sft_call=L.SFT_CALL, n_samples=n, wall_s=time.time() - t0, oom=oom,
               peak_alloc_gib=torch.cuda.max_memory_allocated() / 2**30,
               peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
               total_gib=torch.cuda.get_device_properties(0).total_memory / 2**30,
               labels=meta.get("labels"), log_history=meta.get("log_history"))
    _write_json(rep, os.environ["GNPR_SID_PROBE_OUT"])
    log(f"probe_one: {json.dumps({k: v for k, v in rep.items() if k != 'log_history'}, default=str)}")
    shutil.rmtree(out, ignore_errors=True)
    return 0


def smoke_projection(ex: Export, wd: str, tr: dict, summary: dict) -> dict:
    from methods.gnpr_sid import llm as L
    sft = tr["meta"]["sft"]
    st = tr["stages"]
    rt = [h for h in sft.get("log_history", []) if "train_runtime" in h]
    if rt:
        s_per_step = rt[-1]["train_runtime"] / SMOKE_SFT_STEPS
        sft_overhead = max(0.0, st["sft"]["wall_s"] - rt[-1]["train_runtime"])
    else:
        s_per_step, sft_overhead = st["sft"]["wall_s"] / SMOKE_SFT_STEPS, 0.0
    inf = summary.get("val", {})
    s_per_q = inf.get("stages", {}).get("infer", 0.0) / max(1, inf.get("n_queries", 1))
    eff = L.SFT_CALL["batch_size"] * L.SFT_CALL["grad_accum"]
    root = os.path.join(ROOT, "exports")
    proj = dict(s_per_sft_step=s_per_step, sft_overhead_s=sft_overhead, s_per_query=s_per_q,
                sid_s=st["sid"]["wall_s"], jobs={})
    for ds in ("nyc", "tky"):
        for pol in ("H-100", "H-10", "S-10"):
            e = Export(os.path.join(root, ds, pol))
            _, ss = D.sft_samples(e, None)
            n_steps = int(np.ceil(ss["n_samples"] / eff)) * L.SFT_CALL["num_train_epochs"]
            nq = len(e.queries("val")) + len(e.queries("test"))
            h = (st["sid"]["wall_s"] + sft_overhead + n_steps * s_per_step + nq * s_per_q) / 3600
            proj["jobs"][f"{ds}/{pol}"] = dict(n_samples=ss["n_samples"], sft_steps=n_steps, n_queries=nq,
                                               gpu_h=round(h, 2))
    proj["total_gpu_h"] = round(sum(j["gpu_h"] for j in proj["jobs"].values()), 2)
    return proj


if __name__ == "__main__":
    sys.exit(main())
