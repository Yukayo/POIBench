#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import socket
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

import numpy as np
import pandas as pd

FRESH = C.ROOT / "work" / "efficiency_fresh"
NAME = "ga-llm-fresh"
POL = "H-100"


def _import(path, name):
    sp = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(mod)
    return mod


def cmd_canary(datasets, n):
    from poibench.canary import _frames_equal, _norm_list, choose_queries, lint_sources
    from poibench.exports import Export, build_export, load_sources
    E = _import(C.ROOT / "scripts" / "efficiency" / "e6_fresh_export.py", "e6_fresh_export")
    R = _import(C.ROOT / "methods" / "gallm" / "run.py", "e8_gallm_run")
    os.environ["GALLM_BACKEND"] = "mock"
    for ds in datasets:
        t0 = time.time()
        real = E.fresh_dir(ds)
        ck, venues, queries, pol, cands, meta = load_sources(ds, POL)
        S = choose_queries(queries["test"], n, 0)
        ck2, q2, pinfo = E.perturb_fresh(ck, venues, queries, S)
        work = FRESH / "canary_gallm" / ds
        shutil.rmtree(work, ignore_errors=True)
        pert = work / "export_perturbed"
        new2, _ = E.make_fresh(build_export(ck2, venues, q2, pol, cands), ck2)
        E.write_fresh(new2, pert, ds, dict(perturbed_for_canary=True, n_protected_queries=int(len(S))))
        A0, A1 = Export(str(real)), Export(str(pert))
        sset = set(int(x) for x in S)
        qa, qb = A0.queries("test"), A1.queries("test")
        inv = {"query_inputs_test[S]": _frames_equal(qa[qa.query_id.isin(sset)].reset_index(drop=True),
                                                     qb[qb.query_id.isin(sset)].reset_index(drop=True))}
        ca, cb = A0.context("test"), A1.context("test")
        inv["context_test[S]"] = _frames_equal(ca[ca.query_id.isin(sset)].reset_index(drop=True),
                                               cb[cb.query_id.isin(sset)].reset_index(drop=True))
        inv["candidates_test"] = _frames_equal(A0.candidates("test"), A1.candidates("test"))
        inv["venues_public"] = _frames_equal(A0.venues, A1.venues)
        qids = [int(x) for x in S]
        pa = R.PR.query_prompts(A0, "test", qids, "default")
        pb = R.PR.query_prompts(A1, "test", qids, "default")
        res = dict(gallm_prompt_mismatch=int((pa["prompt"].to_numpy() != pb["prompt"].to_numpy()).sum()),
                   gallm_mean_n_hist=float(pa["n_hist"].mean()))
        for tag, fn in (("gallm", R.predict), ("leaky_control", E._leaky_target)):
            ya = fn(export_dir=str(real), split="test", query_ids=qids, work_dir=None).set_index("query_id")
            yb = fn(export_dir=str(pert), split="test", query_ids=qids, work_dir=None).set_index("query_id")
            res[f"{tag}_pred_mismatch"] = int(sum(_norm_list(ya.at[q, "ranked_venue_ids"]) !=
                                                  _norm_list(yb.at[q, "ranked_venue_ids"]) for q in qids))
        lint = lint_sources([str(C.ROOT / "methods" / "gallm")])
        ok = (all(inv.values()) and res["gallm_prompt_mismatch"] == 0 and res["gallm_pred_mismatch"] == 0
              and not lint and res["leaky_control_pred_mismatch"] > 0)
        rep = dict(result="PASS" if ok else "FAIL", rule=E.RULE, ds=ds, policy=POL, split="test", n_queries=len(qids),
                   seed=0, created_utc=pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ"), perturbation=pinfo,
                   exporter_invariance=inv, **res, lint=[dict(file=f, line=i, why=w, text=t) for f, i, w, t in lint],
                   real_export_files=A0.manifest.get("files"), seconds=round(time.time() - t0, 1))
        d = C.ROOT / "logs" / "canary"
        json.dump(rep, open(d / f"e8_fresh_gallm__{ds}__{POL}__test.json", "w"), indent=1)
        print(json.dumps({k: rep[k] for k in rep if k not in ("lint", "real_export_files", "rule", "perturbation")}),
              flush=True)
        shutil.rmtree(work, ignore_errors=True)


def cmd_run(datasets):
    from poibench.exports import Export
    R = _import(C.ROOT / "methods" / "gallm" / "run.py", "e8_gallm_run")
    for ds in datasets:
        cp = C.ROOT / "logs" / "canary" / f"e8_fresh_gallm__{ds}__{POL}__test.json"
        if not (cp.exists() and json.load(open(cp))["result"] == "PASS"):
            raise SystemExit(f"{ds}: GA-LLM fresh canary not PASS")
        rec = json.load(open(C.ROOT / "preds" / f"ga-llm@{POL}" / ds / "test.cost.json"))
        _, ck = R.run_dirs(ds, POL, "default", False)
        s1 = R._load_done(os.path.join(ck, "stage1", "DONE.json"))
        s2 = R._load_done(os.path.join(ck, "stage2", "DONE.json"))
        mfp = R.model_fingerprint(s2["out"])
        if mfp != rec["generation"]["model_fingerprint"]:
            raise SystemExit(f"{ds}: model fingerprint {mfp} != recorded {rec['generation']['model_fingerprint']}")
        t0 = time.perf_counter()
        wd = FRESH / "gallm" / ds
        wd.mkdir(parents=True, exist_ok=True)
        ex = Export(str(FRESH / "exports" / ds / POL))
        qids = [int(x) for x in ex.queries("test")["query_id"].to_numpy(np.int64)]
        t1 = time.perf_counter()
        pr = R.PR.query_prompts(ex, "test", qids, "default")
        prep_s = time.perf_counter() - t1
        pr.to_parquet(wd / "prompts_test.parquet", index=False)
        R.PR.gps_mapping(ex).to_csv(wd / "gps_mapping.csv", index=False)
        fp = pd.read_parquet(C.ROOT / "work" / "gallm" / ds / POL / "default" / "prompts_test.parquet")
        same = float(np.mean(pr["prompt_sha256"].to_numpy() ==
                             fp.set_index("query_id").loc[qids, "prompt_sha256"].to_numpy()))
        res, gst = R.run_infer(pr, str(wd), s1, s2["out"], ds, len(ex.venues), "hf", "fresh_test")
        out = R.parse_outputs(qids, res, ex, "test")
        out.assign(prompt_sha256=pr["prompt_sha256"].to_numpy()).to_parquet(wd / "responses_test.parquet", index=False)
        preds = pd.DataFrame({"query_id": out["query_id"], "ranked_venue_ids": out["ranked_venue_ids"],
                              "answer_venue_id": out["answer_venue_id"]})
        cost = dict(method=NAME, ds=ds, split="test", exploratory=True, evidence_rule="fresh",
                    created_utc=pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ"), host=socket.gethostname(),
                    gpus=os.environ.get("CUDA_VISIBLE_DEVICES"), checkpoint=s2["out"], model_fingerprint=mfp,
                    frozen_run=rec.get("created_utc"), cfg="default", seed=0,
                    prompt_stats=dict(n=len(pr), mean_n_hist=float(pr["n_hist"].mean()),
                                      mean_n_cur=float(pr["n_cur"].mean()), share_prompt_identical_to_frozen=same),
                    generation={k: v for k, v in gst.items() if k != "load_s"},
                    parse_rates=dict(valid=float(out["valid"].mean()), skipped=float(out["skipped"].mean())),
                    stages=[dict(name="prepare_prompts", wall_s=prep_s, gpu_s=0.0, n_items=len(pr)),
                            dict(name="infer", wall_s=gst["wall_s"], gpu_s=gst["gpu_s"], n_gpus=gst["n_workers"],
                                 n_items=len(pr), n_generated=gst.get("n_generated"), n_cache_hits=gst.get("n_cached"),
                                 notes="HF generate batch 1, one process per visible GPU; generated prompts only")],
                    total_wall_s=time.perf_counter() - t0)
        d = C.OUT / "fresh_llm" / NAME / ds
        d.mkdir(parents=True, exist_ok=True)
        preds.to_parquet(d / "test.parquet.tmp", index=False)
        os.replace(d / "test.parquet.tmp", d / "test.parquet")
        json.dump(cost, open(d / "test.cost.json", "w"), indent=1, default=str)
        print(f"wrote {d.relative_to(C.ROOT)} ({len(preds)} queries; same prompt as frozen {same:.4f})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["canary", "run"])
    ap.add_argument("--datasets", nargs="+", default=list(C.MAIN))
    ap.add_argument("--n", type=int, default=300)
    a = ap.parse_args()
    cmd_canary(a.datasets, a.n) if a.cmd == "canary" else cmd_run(a.datasets)


if __name__ == "__main__":
    main()
