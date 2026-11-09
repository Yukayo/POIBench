#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for _p in (ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ.setdefault("TQDM_MININTERVAL", "30")


def _pipeline():
    import pipeline
    return pipeline


def canary_pointer(ds: str, policy: str) -> str:
    return os.path.join(ROOT, "work", "graph_flashback", ds, policy, "CANARY_MODEL.json")


def predict(export_dir, split, query_ids, work_dir=None):
    pl = _pipeline()
    ex = pl.Export(export_dir)
    e = pl.export_info(ex)
    with open(canary_pointer(e["ds"], e["policy"])) as f:
        ptr = json.load(f)
    b, meta, _ = pl.load_bundle(os.path.join(ROOT, ptr["train_dir"]), os.environ.get("GF_CANARY_DEVICE", "cpu"))
    p = pl.predict_windows(b, ex, split, query_ids)
    return p[["query_id", "ranked_venue_ids", "cand_ranked_venue_ids"]]


def _query_ids(ex, split, cfg, path=None):
    import numpy as np
    if path:
        with open(path) as f:
            return [int(x) for x in f.read().split()]
    q = ex.queries(split)["query_id"].to_numpy(np.int64)
    if cfg["smoke"] and cfg["smoke_val_queries"] and len(q) > cfg["smoke_val_queries"]:
        q = np.sort(np.random.default_rng(cfg["seed"]).choice(q, size=cfg["smoke_val_queries"], replace=False))
    return q.tolist()


def _write_preds(pl, ex, cfg, split, p, out, pdir, tdir, pred_stage, val_fb=None, cost_out=None, cost_only=False):
    from poibench.cost import CostRecorder
    e = pl.export_info(ex)
    with open(os.path.join(pdir, "DONE.json")) as f:
        prep = json.load(f)
    with open(os.path.join(tdir, "train_meta.json")) as f:
        tm = json.load(f)
    extra = dict(method_family="graph_flashback", evidence_policy=e["policy"], seed=cfg["seed"], topk=pl.TOPK,
                 smoke=cfg["smoke"], candidates_file=pl.candidates_file_decl(ex, split),
                 adapter_version=pl.ADAPTER_VERSION, src_key=prep.get("src"), config=cfg,
                 prepare_dir=os.path.relpath(pdir, ROOT), train_dir=os.path.relpath(tdir, ROOT),
                 model_summary=dict(prepare={k: prep.get(k) for k in ("n_users", "n_train_users", "n_triplet_users",
                                                                      "n_venues", "n_triplets", "triplets_by_relation",
                                                                      "graphs")},
                                    train=tm["data"], chosen_epoch=tm["chosen_epoch"], selection=tm["selection"],
                                    val_history=[h.get("val") for h in tm["history"] if h.get("val")]),
                 n_fallback=int(p.attrs.get("n_fallback", 0)),
                 n_dropped_noncatalog_inputs=int(p.attrs.get("n_dropped_noncatalog", 0)),
                 target_time_used=False, export=dict(ds=e["ds"], policy=e["policy"], perturbed=e["perturbed"]))
    if val_fb is not None:
        extra["smoke_val_feedback"] = val_fb
    rec = CostRecorder(method=f"{pl.METHOD}@{e['policy']}", ds=e["ds"], split=split, extra=extra)
    for st in prep.get("cost_stages", []):
        st = dict(st)
        st["notes"] = (st.get("notes", "") + " prepare stage (run once per visible-row set, shared by policies "
                       "with identical visible rows)").strip()
        rec.add_stage(**st)
    for st in tm.get("cost_stages", []):
        rec.add_stage(**st)
    rec.stages.append(pred_stage)
    cost = cost_out or (out[:-len(".parquet")] + ".cost.json" if out.endswith(".parquet") else out + ".cost.json")
    if cost_only:
        from poibench.exclusive import assert_outside_preds
        assert_outside_preds(out)
        assert_outside_preds(cost)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    p.to_parquet(out + ".tmp", index=False)
    os.replace(out + ".tmp", out)
    rec.save(cost, n_queries=len(p))
    pl.log(f"wrote {os.path.relpath(out, ROOT)} ({len(p)} queries, {extra['n_fallback']} fallback) + "
           f"{os.path.relpath(cost, ROOT)}")
    return cost


def _finish_cost_only(pl, a, e, wdir_cfg, chk, written):
    from poibench import exclusive as X
    os.makedirs(wdir_cfg, exist_ok=True)
    log_p = os.path.join(wdir_cfg, "colocation_checks.jsonl")
    with open(log_p, "a") as f:
        f.write(json.dumps(dict(attempt=os.environ.get("POIBENCH_ATTEMPT"), cmd=a.cmd, device=a.device,
                                check=chk)) + "\n")
    with open(log_p) as f:
        checks = [json.loads(ln) for ln in f if ln.strip()]
    flags = [c["check"]["colocated"] if c["check"] else None for c in checks if c.get("cmd") != "prepare"]
    colocated = True if any(v is True for v in flags) else (False if flags and all(v is False for v in flags)
                                                           else None)
    for split, (cost, out) in written.items():
        with open(cost) as f:
            c = json.load(f)
        c.update(X.record(chk, colocated=colocated, colocation_checks_all_attempts=checks,
                          work_dir=os.path.relpath(a.work_dir_eff, ROOT), rerun_preds=os.path.relpath(out, ROOT),
                          preds_parity=X.preds_parity(out, os.path.join(ROOT, "preds", f"{pl.METHOD}@{e['policy']}",
                                                                        e["ds"], f"{split}.parquet")),
                          rerun_of="main-table job of the same method/policy/ds (preds/ of record)"))
        with open(cost + ".tmp", "w") as f:
            json.dump(c, f, indent=2, default=str)
        os.replace(cost + ".tmp", cost)
        pl.log(f"cost-only: {os.path.relpath(cost, ROOT)} colocated={colocated} gpu={c.get('gpu_model')}")


def run(a) -> int:
    pl = _pipeline()
    from poibench.exports import export_dir
    import torch
    torch.set_num_threads(a.threads)
    cfg = pl.make_config(smoke=a.smoke, epochs=a.epochs, kge_epochs=a.kge_epochs)
    ed = a.export_dir or export_dir(a.ds, a.policy, ROOT)
    ex = pl.Export(ed)
    e = pl.export_info(ex)
    if a.ds and e["ds"] and a.ds != e["ds"] or a.policy and e["policy"] and a.policy != e["policy"]:
        raise SystemExit(f"--ds/--policy {a.ds}/{a.policy} do not match the export {e['ds']}/{e['policy']}")
    default_root = os.path.join(ROOT, "work", "cost_exclusive", "graph_flashback") if a.cost_only else \
        os.path.join(ROOT, "work", "graph_flashback")
    work_root = a.work_dir or default_root
    if a.cost_only:
        from poibench import exclusive as X
        X.assert_outside_preds(work_root)
        for split in a.split:
            X.cost_path(f"{pl.METHOD}@{e['policy']}", e["ds"], split, a.cost_root)
        a.work_dir_eff = work_root
        mon = X.GpuShareMonitor().start() if a.device.startswith("cuda") else None
        written = {}
        try:
            return _run_stages(pl, a, cfg, ex, e, work_root, written)
        finally:
            chk = mon.stop() if mon is not None else None
            _finish_cost_only(pl, a, e, os.path.join(work_root, e["ds"], e["policy"]), chk, written)
    return _run_stages(pl, a, cfg, ex, e, work_root, {})


def _run_stages(pl, a, cfg, ex, e, work_root, written) -> int:
    if a.cmd in ("prepare", "train", "all"):
        pdir = pl.prepare(ex, cfg, work_root, a.device)
    else:
        pdir = pl.prepare_dir(ex, cfg, work_root)
        if not os.path.exists(os.path.join(pdir, "DONE.json")):
            raise SystemExit(f"prepare not done: {pdir} (run `prepare` first)")
    if a.cmd == "prepare":
        return 0
    if a.cmd in ("train", "all"):
        tdir = pl.train(ex, cfg, work_root, pdir, a.device)
    else:
        tdir = pl.train_dir(ex, cfg, work_root)
        if not os.path.exists(os.path.join(tdir, "model.pt")):
            raise SystemExit(f"train not done: {tdir} (run `train` first)")
    if a.cmd == "train":
        return 0
    from poibench.cost import CostRecorder
    b, meta, _ = pl.load_bundle(tdir, a.device)
    for split in a.split:
        if split not in ("val", "test"):
            raise SystemExit("predict on val or test")
        qids = _query_ids(ex, split, cfg, a.query_ids)
        rec = CostRecorder()
        with rec.stage(f"predict_{split}", one_time=False, n_items=len(qids),
                       n_gpus=1 if a.device.startswith("cuda") else 0):
            p = pl.predict_windows(b, ex, split, qids)
        val_fb = None
        if cfg["smoke"] and split == "val" and not e["perturbed"] and not a.out and not a.cost_only:
            from poibench.evaluate import val_feedback
            val_fb = val_feedback(e["ds"], p[["query_id", "ranked_venue_ids"]], query_ids=qids)
            pl.log(f"smoke val feedback (sanity only, {len(qids)} queries): {val_fb}")
        cost_out = None
        if a.cost_only:
            from poibench import exclusive as X
            out = os.path.join(work_root, e["ds"], e["policy"], "cost_only_preds", f"{split}.parquet")
            cost_out = X.cost_path(f"{pl.METHOD}@{e['policy']}", e["ds"], split, a.cost_root)
        elif a.out:
            out = a.out
        elif cfg["smoke"]:
            out = os.path.join(ROOT, "work", "graph_flashback", e["ds"], e["policy"], "smoke_preds", f"{split}.parquet")
        else:
            out = os.path.join(ROOT, "preds", f"{pl.METHOD}@{e['policy']}", e["ds"], f"{split}.parquet")
        cost = _write_preds(pl, ex, cfg, split, p, out, pdir, tdir, rec.stages[0], val_fb, cost_out=cost_out,
                            cost_only=a.cost_only)
        if a.cost_only:
            written[split] = (cost, out)
    return 0


def canary_prep(a) -> int:
    pl = _pipeline()
    from poibench.exports import export_dir
    import torch
    torch.set_num_threads(a.threads)
    cfg = pl.make_config(smoke=True)
    ex = pl.Export(export_dir(a.ds, a.policy, ROOT))
    work_root = os.path.join(ROOT, "work", "graph_flashback", "canary_prep")
    pdir = pl.prepare(ex, cfg, work_root, "cpu")
    tdir = pl.train(ex, cfg, work_root, pdir, "cpu")
    ptr = canary_pointer(a.ds, a.policy)
    os.makedirs(os.path.dirname(ptr), exist_ok=True)
    with open(ptr, "w") as f:
        json.dump(dict(train_dir=os.path.relpath(tdir, ROOT), prepare_dir=os.path.relpath(pdir, ROOT),
                       config=cfg), f, indent=1)
    pl.log(f"canary-prep: smoke checkpoint {os.path.relpath(tdir, ROOT)}")
    if a.no_full_pipeline:
        return 0
    from poibench.canary import run_canary
    py = sys.executable
    entry = (f"cmd:{py} -I methods/graph_flashback/run.py all --export-dir {{export_dir}} --work-dir {{work_dir}} "
             f"--smoke --split {{split}} --query-ids {{query_ids}} --out {{out}} --device cpu --threads {a.threads}")
    rep = run_canary(entry, a.ds, a.policy, "val", a.canary_n, 0, lint_paths=[HERE],
                     tag="graph_flashback_fullpipeline")
    pl.log(f"full-pipeline canary {a.ds}/{a.policy}: {rep['result']} mismatches={rep['n_mismatch']} "
           f"missing={rep['n_missing']} invariance={all(rep['exporter_invariance'].values())} lint={len(rep['lint'])}")
    return 0 if rep["result"] == "PASS" else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Graph-Flashback adapter (see module docstring)")
    ap.add_argument("cmd", nargs="?", default="all", choices=["prepare", "train", "predict", "all"])
    ap.add_argument("--ds")
    ap.add_argument("--policy")
    ap.add_argument("--split", "--splits", nargs="+", default=["val", "test"], dest="split")
    ap.add_argument("--device", default="cpu", help="cpu | cuda | cuda:<i>")
    ap.add_argument("--seed", type=int, default=0, help="owner rule: 0 only")
    ap.add_argument("--threads", type=int, default=8, help="torch CPU threads")
    ap.add_argument("--smoke", action="store_true", help="tiny CPU smoke config (100-user subsample, 1 KGE epoch, "
                                                         "2 Flashback epochs, 500 val queries, last epoch)")
    ap.add_argument("--epochs", type=int, help=argparse.SUPPRESS)
    ap.add_argument("--kge-epochs", type=int, help=argparse.SUPPRESS)
    ap.add_argument("--export-dir", help="read this export instead of exports/<ds>/<policy> (canary)")
    ap.add_argument("--work-dir", help="work root (default work/graph_flashback)")
    ap.add_argument("--query-ids", help="file with query ids (default: all queries of the split)")
    ap.add_argument("--out", help="prediction parquet (default preds/graph-flashback@<policy>/<ds>/<split>.parquet)")
    ap.add_argument("--canary-prep", action="store_true", help="smoke checkpoint for predict() + full-pipeline canary")
    ap.add_argument("--no-full-pipeline", action="store_true", help="with --canary-prep: skip the full-pipeline canary")
    ap.add_argument("--canary-n", type=int, default=300)
    ap.add_argument("--cost-only", action="store_true",
                    help="exclusive cost re-run (Q3): identical pipeline in --work-dir (default "
                         "work/cost_exclusive/graph_flashback); writes only results/cost_exclusive/"
                         "graph-flashback@<policy>/<ds>/<split>.cost.json, never preds/")
    ap.add_argument("--cost-root", help="with --cost-only: root of the cost records (default results/cost_exclusive)")
    a = ap.parse_args(argv)
    if a.cost_only:
        if a.canary_prep or a.out or a.query_ids or a.cmd == "train" or a.cmd == "predict":
            ap.error("--cost-only: cmd prepare or all, no --canary-prep/--out/--query-ids")
        if (a.smoke or a.export_dir) and not a.cost_root:
            ap.error("--cost-only with --smoke/--export-dir only for tests, with an explicit --cost-root")
    elif a.cost_root:
        ap.error("--cost-root only with --cost-only")
    if a.seed != 0:
        ap.error("seed 0 only (owner rule)")
    if (a.epochs or a.kge_epochs) and not a.smoke:
        ap.error("--epochs/--kge-epochs only with --smoke (defaults are never changed for real runs)")
    if a.out and len(a.split) != 1:
        ap.error("--out needs exactly one --split")
    if not a.export_dir and not (a.ds and a.policy):
        ap.error("--ds and --policy are required (or --export-dir)")
    if a.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if a.canary_prep:
        return canary_prep(a)
    return run(a)


if __name__ == "__main__":
    sys.exit(main())
