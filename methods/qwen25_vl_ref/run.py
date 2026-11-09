#!/usr/bin/env python
from __future__ import annotations
import os as _pb_os
def _pb_root():
    p = _pb_os.path.dirname(_pb_os.path.abspath(__file__))
    while not _pb_os.path.isfile(_pb_os.path.join(p, "poibench", "__init__.py")) and _pb_os.path.dirname(p) != p:
        p = _pb_os.path.dirname(p)
    return p
_PB_ROOT = _pb_os.environ.get("POIBENCH_ROOT") or _pb_root()
_PB_CACHE = _pb_os.environ.get("POIBENCH_CACHE") or _pb_os.path.join(_PB_ROOT, "work", "cache")

import argparse
import json
import math
import os
import re
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
from methods.qwen25_vl_ref import vlm as V

METHOD = "qwen2.5-vl-ref"
ARMS = ("A", "B", "C")
SEED = 0
N_HIST = 20
N_HIST_PHOTO = 4
N_CAND = 20
TOPK = 50
CITY = {"nyc": "New York City", "tky": "Tokyo", "istanbul": "Istanbul"}

def _caption_texts():
    import ast as _ast
    p = os.path.join(ROOT, "third_party", "LLM-Augmented-POIRec", "main", "llm_based", "embedding_utils",
                     "photo_summary_openai.py")
    if not os.path.exists(p):
        raise FileNotFoundError(f"{p} missing: run bash scripts/fetch_upstream.sh")
    with open(p, encoding="utf-8") as f:
        tree = _ast.parse(f.read())
    system = prompt = None
    for node in _ast.walk(tree):
        if isinstance(node, _ast.Assign) and any(isinstance(t, _ast.Name) and t.id == "text_prompt" for t in node.targets):
            prompt = _ast.literal_eval(node.value)
        if isinstance(node, _ast.Dict):
            kv = {k.value: v for k, v in zip(node.keys, node.values) if isinstance(k, _ast.Constant)}
            if isinstance(kv.get("role"), _ast.Constant) and kv["role"].value == "system" and "content" in kv:
                system = _ast.literal_eval(kv["content"])
    if system is None or prompt is None:
        raise RuntimeError(f"caption prompt not found in {p}")
    return system, prompt


def log(*a):
    print(f"[qwen25_vl_ref {time.strftime('%H:%M:%S')}]", *a, flush=True)


def _write_parquet(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path + ".tmp", index=False)
    os.replace(path + ".tmp", path)


def _write_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f, indent=1, default=str)
    os.replace(path + ".tmp", path)


def mm_dir(ds: str) -> str:
    return os.path.join(ROOT, "exports_mm", ds)


IMG_CACHE = os.environ.get("POIBENCH_QWENVL_IMG_CACHE", f"{_PB_CACHE}/qwenvl_imgs")
_STAGED: dict = {}


def _sha256_file(path: str) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def stage_images(ds: str, imgs: dict) -> dict:
    if not os.path.isdir(os.path.dirname(IMG_CACHE)):
        return imgs
    dst_dir = os.path.join(IMG_CACHE, ds)
    os.makedirs(dst_dir, exist_ok=True)
    out = {}
    for v, (src, sha) in imgs.items():
        dst = os.path.join(dst_dir, f"{sha}.png")
        if not (os.path.exists(dst) and _sha256_file(dst) == sha):
            tmp = f"{dst}.tmp{os.getpid()}"
            with open(src, "rb") as fi, open(tmp, "wb") as fo:
                fo.write(fi.read())
            if _sha256_file(tmp) != sha:
                raise RuntimeError(f"{src}: sha256 mismatch after copy")
            os.replace(tmp, dst)
        out[v] = (dst, sha)
    return out


def load_images(ds: str) -> dict:
    if ds not in _STAGED:
        t = pd.read_parquet(os.path.join(mm_dir(ds), "venue_images.parquet"))
        t = t[t.has_image]
        imgs = {int(v): (os.path.join(mm_dir(ds), p), s) for v, p, s in zip(t.venue_id, t.image, t.sha256)}
        _STAGED[ds] = stage_images(ds, imgs)
    return dict(_STAGED[ds])


def n_gpus_for(kind: str) -> int:
    return 1 if kind == "vllm" else 0


def caption_request(path: str, sha: str) -> list:
    return [("system", [("text", _caption_texts()[0])]), ("user", [("text", _caption_texts()[1])]),
            ("user", [("image", path, sha)])]


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```\s*$", re.S)


def parse_caption(text: str) -> tuple[str, bool]:
    t = _FENCE.sub("", text.strip())
    try:
        d = json.loads(t)
        summ = str(d.get("summary", "")).strip()
        kw = d.get("keywords", [])
        kw = [str(x).strip() for x in kw if str(x).strip()][:5] if isinstance(kw, list) else []
        if summ:
            return summ + (f" Keywords: {', '.join(kw)}." if kw else ""), True
    except (json.JSONDecodeError, AttributeError, TypeError):
        pass
    m = re.search(r'"summary"\s*:\s*"((?:[^"\\]|\\.)*)"', t)
    if m:
        return m.group(1).strip(), False
    return re.sub(r"\s+", " ", t)[:300], False


def captions_path(ds: str, backend_id: str) -> str:
    return os.path.join(ROOT, "work", "qwen25_vl_ref", "captions", V._slug(backend_id), ds, "captions.parquet")


def ensure_captions(ds: str, kind: str, rec: CostRecorder | None = None, job_stage: bool = False) -> pd.DataFrame:
    be = V.get_backend(kind)
    p = captions_path(ds, be.id)
    imgs = load_images(ds)
    if os.path.exists(p):
        t = pd.read_parquet(p)
        if len(t) == len(imgs) and dict(zip(t.venue_id.astype(int), t.sha256)) == {k: v[1] for k, v in imgs.items()}:
            return t
    vids = sorted(imgs)
    reqs = [caption_request(*imgs[v]) for v in vids]
    cache = V.ResponseCache(be.id, "caption")
    ng = n_gpus_for(kind)

    def on_load(f):
        if rec is None:
            return f()
        with rec.stage("model_load", one_time=True, n_gpus=ng, notes=f"{be.id}; once per process"):
            f()

    t0 = time.perf_counter()
    if rec is not None and not be.loaded and V.n_pending(reqs, "caption", be, cache):
        on_load(be.load)
    if job_stage and rec is not None:
        with rec.stage("caption_job", one_time=True, n_gpus=ng, notes="caption every venue photo"):
            res, st = V.generate_cached(reqs, "caption", be, cache, ng, job=f"caption/{ds}", log=log, on_load=on_load)
    else:
        res, st = V.generate_cached(reqs, "caption", be, cache, ng, job=f"caption/{ds}", log=log, on_load=on_load)
    caps = [parse_caption(r["text"]) for r in res]
    t = pd.DataFrame(dict(venue_id=vids, sha256=[imgs[v][1] for v in vids], caption=[c[0] for c in caps],
                          parsed=[c[1] for c in caps], text=[r["text"] for r in res],
                          n_prompt_tokens=[int(r["n_prompt_tokens"]) for r in res],
                          n_output_tokens=[int(r["n_output_tokens"]) for r in res],
                          finish_reason=[str(r["finish_reason"]) for r in res],
                          gen_s=[float(r["gen_s"]) for r in res], n_gpus=[int(r["n_gpus"]) for r in res]))
    _write_parquet(t, p)
    _write_json(dict(ds=ds, backend=be.id, n=len(t), parsed_share=float(t.parsed.mean()),
                     finish_reasons=t.finish_reason.value_counts().to_dict(),
                     prompt_tokens=int(t.n_prompt_tokens.sum()), output_tokens=int(t.n_output_tokens.sum()),
                     gen_s_total=float(t.gen_s.sum()), gpu_s_total=float((t.gen_s * t.n_gpus).sum()),
                     generated_now=st["n_generated"], wall_s_now=round(time.perf_counter() - t0, 1),
                     sampling=V.SAMPLING["caption"], prompt_source="LLM-Augmented-POIRec c7dac9da "
                     "photo_summary_openai.py l.92-109 (verbatim), metadata block omitted"),
                p.replace(".parquet", ".json"))
    log(f"captions {ds}: {len(t)} venues, parsed {t.parsed.mean():.3f}, generated now {st['n_generated']} -> "
        f"{os.path.relpath(p, ROOT)}")
    return t


def _hav_km(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * math.asin(math.sqrt(min(1.0, a)))


def _dedup(seq) -> list:
    return list(dict.fromkeys(int(x) for x in seq))


def build_requests(ex: Export, split: str, qids, arm: str, images: dict, captions: dict | None) -> list:
    if arm not in ARMS:
        raise ValueError(arm)
    if arm == "B" and captions is None:
        raise ValueError("arm B needs captions")
    ds = ex.manifest["ds"]
    ven = ex.venues.set_index("venue_id")
    vcat = ven["category"].to_dict()
    vlat, vlon = ven["lat"].to_dict(), ven["lon"].to_dict()
    qi = ex.query_inputs(split)
    pos = pd.Series(np.arange(len(qi.table)), index=qi.table["query_id"].to_numpy(np.int64))
    cands = ex.candidates(split)
    cmap = dict(zip(cands["query_id"].astype(np.int64), cands["cand_venue_ids"]))
    out = []
    for qid in qids:
        qid = int(qid)
        i = int(pos[qid])
        hv = qi.own(i, "venue_id").astype(np.int64)[-N_HIST:]
        ht = qi.own(i, "t_local")[-N_HIST:]
        allc = _dedup(cmap[qid])
        cand, rest = allc[:N_CAND], allc[N_CAND:TOPK]
        prev = int(hv[-1])
        photo_pos = {}
        for k in range(len(hv) - 1, -1, -1):
            v = int(hv[k])
            if v in images and v not in photo_pos:
                photo_pos[v] = k
                if len(photo_pos) == N_HIST_PHOTO:
                    break
        photo_rows = set(photo_pos.values())
        segs = [("text", f"Here are the most recent check-ins of one user in {CITY.get(ds, ds)} and {len(cand)} "
                         f"candidate places. Predict which candidate place the user will check in at next.\n\n"
                         f"Recent check-ins of the user (oldest first; local time, place id, category, "
                         f"latitude/longitude):\n")]
        nimg = ncap = 0

        def annotate(v):
            nonlocal nimg, ncap
            if arm == "B":
                c = captions.get(v)
                if c:
                    segs.append(("text", f"\n   photo description: {c}"))
                    ncap += 1
            elif arm == "C":
                segs.append(("text", "\n   photo: "))
                segs.append(("image", images[v][0], images[v][1]))
                nimg += 1

        for k, (v, t) in enumerate(zip(hv, ht)):
            v = int(v)
            segs.append(("text", f"{k + 1}. {pd.Timestamp(t):%a %Y-%m-%d %H:%M}, place {v}, {vcat.get(v, 'unknown')}, "
                                 f"({vlat[v]:.4f}, {vlon[v]:.4f})"))
            if k in photo_rows:
                annotate(v)
            segs.append(("text", "\n"))
        segs.append(("text", "The next check-in comes right after the last one above.\n\n"
                             "Candidate places (in random order; place id, category, latitude/longitude, distance "
                             "from the last check-in place):\n"))
        order = np.random.default_rng([SEED, qid]).permutation(len(cand))
        for k in order:
            v = cand[int(k)]
            segs.append(("text", f"- place {v}: {vcat.get(v, 'unknown')}, ({vlat[v]:.4f}, {vlon[v]:.4f}), "
                                 f"{_hav_km(vlat[prev], vlon[prev], vlat[v], vlon[v]):.1f} km"))
            if v in images:
                annotate(v)
            segs.append(("text", "\n"))
        segs.append(("text", f"\nRank all {len(cand)} candidate places from most to least likely to be the user's next "
                             f"check-in. Answer with a JSON list of the {len(cand)} place ids only, most likely "
                             f"first."))
        merged = []
        for s in segs:
            if s[0] == "text" and merged and merged[-1][0] == "text":
                merged[-1] = ("text", merged[-1][1] + s[1])
            else:
                merged.append(s)
        out.append(dict(query_id=qid, req=[("user", merged)], cand=cand, rest=rest, n_images=nimg, n_captions=ncap))
    return out


_LIST = re.compile(r"\[([^\[\]]*)\]")


def parse_ranking(text: str, cand: list) -> tuple[list, int, bool]:
    cs = set(cand)
    ids, found = [], False
    for m in _LIST.finditer(text):
        got = [int(x) for x in re.findall(r"\d+", m.group(1))]
        if any(g in cs for g in got):
            ids, found = got, True
            break
    if not found:
        ids = [int(x) for x in re.findall(r"\d+", text)]
    model = [x for x in dict.fromkeys(ids) if x in cs]
    return model + [c for c in cand if c not in set(model)], len(model), found


def _generate(reqs, kind, job, rec):
    be = V.get_backend(kind)
    cache = V.ResponseCache(be.id, "rank")
    ng = n_gpus_for(kind)

    def on_load(f):
        if rec is None:
            return f()
        with rec.stage("model_load", one_time=True, n_gpus=ng,
                       notes=f"{be.id}; once per process (recorded in the first split's cost.json)"):
            f()

    if rec is None:
        return V.generate_cached(reqs, "rank", be, cache, ng, job=job, log=log, on_load=on_load)[0], be
    if not be.loaded and V.n_pending(reqs, "rank", be, cache):
        on_load(be.load)
    with rec.stage("infer", one_time=False, n_gpus=ng, n_items=len(reqs),
                   notes="VLM generation of the requests not in the response cache") as st:
        res, gst = V.generate_cached(reqs, "rank", be, cache, ng, job=job, log=log, on_load=on_load)
        st["n_generated"], st["n_cache_hits"] = gst["n_generated"], gst["n_cache_hits"]
    pre = gst["cached_keys"]
    if pre:
        rec.add_stage("infer_cached", wall_s=sum(float(cache.mem[k]["gen_s"]) for k in pre), cpu_core_s=0.0,
                      gpu_s=sum(float(cache.mem[k]["gen_s"]) * float(cache.mem[k]["n_gpus"]) for k in pre),
                      peak_mem_gb=0.0, one_time=False, n_items=len(pre),
                      notes="generation time attributed from the cache shards that produced these answers")
    return res, be


def _assemble(items, res):
    rows = []
    for it, r in zip(items, res):
        rk, nv, found = parse_ranking(r["text"], it["cand"])
        rows.append(dict(query_id=it["query_id"], cand_ranked_venue_ids=rk,
                         ranked_venue_ids=_dedup(rk + it["rest"])[:TOPK], n_model_ids=nv, list_found=found,
                         n_images=it["n_images"], n_captions=it["n_captions"], text=r["text"],
                         n_prompt_tokens=int(r["n_prompt_tokens"]), n_output_tokens=int(r["n_output_tokens"]),
                         finish_reason=str(r["finish_reason"])))
    return pd.DataFrame(rows)


def _captions_dict(ds, kind, rec=None):
    t = ensure_captions(ds, kind, rec)
    return dict(zip(t.venue_id.astype(int), t.caption)), t


def predict(export_dir, split, query_ids, work_dir=None):
    arm = os.environ.get("QWENVL_ARM", "C")
    kind = os.environ.get("QWENVL_BACKEND", "vllm")
    ex = Export(export_dir)
    ds = ex.manifest["ds"]
    caps = _captions_dict(ds, kind)[0] if arm == "B" else None
    items = build_requests(ex, split, query_ids, arm, load_images(ds), caps)
    res, _ = _generate([it["req"] for it in items], kind, f"predict:{arm}", None)
    out = _assemble(items, res)
    if work_dir:
        _write_parquet(out, os.path.join(work_dir, "responses.parquet"))
    return out[["query_id", "ranked_venue_ids", "cand_ranked_venue_ids"]]


def candidates_file_decl(ex: Export, split: str) -> str:
    keys = [k for k in ex.manifest["source_sha256"] if k.endswith(f"candidates_{split}.parquet")]
    if len(keys) != 1:
        raise RuntimeError(f"cannot identify the candidate file of {split} in the export manifest: {keys}")
    return os.path.join(ex.manifest["source_dir"], keys[0])


def run_split(a, split, ex, qids, out_base, wd, captions_tab):
    tag = f"{METHOD}-{a.arm}"
    rec = CostRecorder(method=f"{tag}@{a.policy}", ds=a.ds, split=split)
    imgs = load_images(a.ds)
    caps = dict(zip(captions_tab.venue_id.astype(int), captions_tab.caption)) if captions_tab is not None else None
    with rec.stage("prepare_requests", one_time=False, n_gpus=0, n_items=len(qids), notes="prompt construction (CPU)"):
        items = build_requests(ex, split, qids, a.arm, imgs, caps)
    if a.arm == "B":
        ct = captions_tab
        rec.add_stage("caption", wall_s=float(ct.gen_s.sum()), cpu_core_s=0.0, gpu_s=float((ct.gen_s * ct.n_gpus).sum()),
                      peak_mem_gb=0.0, one_time=True, n_items=int(len(ct)),
                      prompt_tokens=int(ct.n_prompt_tokens.sum()), output_tokens=int(ct.n_output_tokens.sum()),
                      notes="one caption per venue photo by the same VLM (all venues of exports_mm; attributed from "
                            "the caption cache); one-time, shared by val and test (counted in both files)")
    res, be = _generate([it["req"] for it in items], a.backend, f"{tag}@{a.policy}/{a.ds}/{split}", rec)
    out = _assemble(items, res)
    _write_parquet(out, os.path.join(wd, f"responses_{split}.parquet"))
    n = len(out)
    meta = dict(evidence_policy=a.policy, arm=a.arm, seed=SEED, zero_shot=True, backend=be.id, model=V.MODEL,
                model_revision=getattr(be, "revision", None), model_revision_pinned=V.REVISION, engine=V.ENGINE,
                sampling=V.SAMPLING["rank"], n_hist=N_HIST, n_hist_photo=N_HIST_PHOTO, n_cand=N_CAND,
                candidates_file=candidates_file_decl(ex, split), primary_output="cand_ranked_venue_ids",
                output="cand_ranked_venue_ids = model ranking of the first 20 retriever candidates (unranked appended "
                       "in retriever order); ranked_venue_ids = that + retriever candidates 21-50",
                images=dict(source="exports_mm/" + a.ds, manifest=json.load(open(os.path.join(mm_dir(a.ds), "MANIFEST.json")))),
                export_manifest_files=ex.manifest.get("files"),
                stats=dict(mean_images=float(out.n_images.mean()), mean_captions=float(out.n_captions.mean()),
                           list_found=float(out.list_found.mean()), mean_model_ids=float(out.n_model_ids.mean()),
                           full_ranking=float((out.n_model_ids == out.cand_ranked_venue_ids.map(len)).mean()),
                           no_valid_id=float((out.n_model_ids == 0).mean()),
                           finish_reasons=out.finish_reason.value_counts().to_dict(),
                           prompt_tokens_mean=float(out.n_prompt_tokens.mean()),
                           output_tokens_mean=float(out.n_output_tokens.mean())))
    d = os.path.join(out_base, f"{tag}@{a.policy}", a.ds)
    rec.meta.update(meta)
    _write_parquet(out[["query_id", "ranked_venue_ids", "cand_ranked_venue_ids"]], os.path.join(d, f"{split}.parquet"))
    rec.save(os.path.join(d, f"{split}.cost.json"), n_queries=n)
    log(f"{tag}@{a.policy}/{a.ds}/{split}: {n} queries; {meta['stats']} -> {os.path.relpath(d, ROOT)}")
    return meta


def run_cost_only(a, ex, caps, cost_root=None, wdir=None, limit=None, log=log):
    from poibench import exclusive as X
    tag = f"{METHOD}-{a.arm}"
    mname = f"{tag}@{a.policy}"
    wdir = wdir or os.path.join(ROOT, "work", "cost_exclusive", "qwen25_vl_ref", a.ds, a.policy, a.arm)
    X.assert_outside_preds(wdir)
    old_cache = os.environ.get("QWENVL_CACHE_DIR")
    os.environ["QWENVL_CACHE_DIR"] = os.path.join(wdir, "cache")
    out_base = X.assert_outside_preds(os.path.join(wdir, "cost_only_out"))
    cjs = {s: X.cost_path(mname, a.ds, s, cost_root) for s in a.split}
    mon = X.GpuShareMonitor().start() if a.device.startswith("cuda") else None
    try:
        for s in a.split:
            q = ex.queries(s)["query_id"].to_numpy(np.int64)
            if limit:
                q = q[:limit]
            run_split(a, s, ex, [int(x) for x in q], out_base, wdir, caps)
    finally:
        if old_cache is None:
            os.environ.pop("QWENVL_CACHE_DIR", None)
        else:
            os.environ["QWENVL_CACHE_DIR"] = old_cache
        chk = mon.stop() if mon is not None else None
        os.makedirs(wdir, exist_ok=True)
        with open(os.path.join(wdir, "colocation_checks.jsonl"), "a") as f:
            f.write(json.dumps(dict(attempt=os.environ.get("POIBENCH_ATTEMPT"), device=a.device, check=chk)) + "\n")
    with open(os.path.join(wdir, "colocation_checks.jsonl")) as f:
        checks = [json.loads(ln) for ln in f if ln.strip()]
    flags = [c["check"]["colocated"] if c["check"] else None for c in checks]
    colocated = True if any(v is True for v in flags) else (False if flags and all(v is False for v in flags) else None)
    for s in a.split:
        d = os.path.join(out_base, mname, a.ds)
        with open(os.path.join(d, f"{s}.cost.json")) as f:
            c = json.load(f)
        pq = os.path.join(d, f"{s}.parquet")
        c.update(X.record(chk, colocated=colocated, colocation_checks_all_attempts=checks,
                          work_dir=os.path.relpath(wdir, ROOT), rerun_preds=os.path.relpath(pq, ROOT),
                          preds_parity=X.preds_parity(pq, os.path.join(ROOT, "preds", mname, a.ds, f"{s}.parquet")),
                          rerun_of="main-table job of the same arm/policy/ds (preds/ of record)",
                          cost_only_reason="photos read from the local SSD (stage_images); the main run of arm C on nyc "
                                           "read them from the cross-region share (I/O-bound)"))
        _write_json(c, cjs[s])
        log(f"cost-only: wrote {cjs[s]} (colocated={colocated})")
    return cjs


def main(argv=None):
    ap = argparse.ArgumentParser(description=(__doc__ or " "), formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", nargs="?", choices=["caption", "all", "predict", "smoke"])
    ap.add_argument("--ds", required=True)
    ap.add_argument("--policy", default="H-100")
    ap.add_argument("--arm", default="C", choices=ARMS)
    ap.add_argument("--split", "--splits", dest="split", nargs="+", default=["val", "test"], choices=["val", "test"])
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=50, help="smoke: seed-0 sampled queries per split")
    ap.add_argument("--canary-prep", action="store_true")
    ap.add_argument("--cost-only", action="store_true",
                    help="exclusive cost-of-record re-run (cmd all): costs only, preds/ never written")
    ap.add_argument("--cost-root", help="with --cost-only: root of the cost records (default results/cost_exclusive)")
    ap.add_argument("--work-dir", help="with --cost-only: work dir (default work/cost_exclusive/qwen25_vl_ref/...)")
    ap.add_argument("--limit", type=int, help="with --cost-only: first N queries of each split (tests only)")
    a = ap.parse_args(argv)
    if a.seed != 0:
        ap.error("seed 0 only (owner rule)")
    if (a.cost_root or a.work_dir or a.limit) and not a.cost_only:
        ap.error("--cost-root/--work-dir/--limit only with --cost-only")
    if a.cost_only and (a.cmd != "all" or a.canary_prep):
        ap.error("--cost-only is a mode of `all`")
    if a.limit and not a.cost_root:
        ap.error("--limit only with an explicit --cost-root (tests)")
    if a.canary_prep:
        log("zero-shot method: nothing to prepare for the canary")
        return 0
    if a.cmd is None:
        ap.error("a subcommand is required")
    a.backend = "vllm" if a.device == "cuda" else "mock"
    if a.cmd == "smoke" and a.device != "cpu":
        ap.error("smoke is the CPU check (--device cpu)")
    if a.cmd == "caption":
        rec = CostRecorder(method=f"{METHOD}-caption", ds=a.ds, split="-")
        t = ensure_captions(a.ds, a.backend, rec, job_stage=True)
        rec.save(captions_path(a.ds, V.get_backend(a.backend).id).replace(".parquet", ".cost.json"), n_queries=len(t))
        return 0
    ed = export_dir(a.ds, a.policy, ROOT)
    ex = Export(ed)
    if a.cost_only:
        caps = ensure_captions(a.ds, a.backend) if a.arm == "B" else None
        run_cost_only(a, ex, caps, a.cost_root, a.work_dir, a.limit)
        return 0
    real = a.backend == "vllm" and a.cmd != "smoke"
    wd = os.path.join(ROOT, "work", "qwen25_vl_ref", "runs" if real else "smoke", a.ds, a.policy, a.arm)
    out_base = os.path.join(ROOT, "preds") if real else os.path.join(wd, "preds")
    caps = None
    if a.arm == "B":
        caps = ensure_captions(a.ds, a.backend)
    summary = {}
    for s in a.split:
        q = ex.queries(s)["query_id"].to_numpy(np.int64)
        if a.cmd == "smoke" and a.n < len(q):
            q = np.sort(np.random.default_rng(SEED).choice(q, size=a.n, replace=False))
        summary[s] = run_split(a, s, ex, [int(x) for x in q], out_base, wd, caps)["stats"]
    if a.cmd == "smoke":
        items = build_requests(ex, a.split[0], [int(ex.queries(a.split[0])["query_id"].iloc[0])], a.arm,
                               load_images(a.ds), dict(zip(caps.venue_id.astype(int), caps.caption)) if caps is not None else None)
        with open(os.path.join(wd, "example_request.txt"), "w") as f:
            for role, segs in items[0]["req"]:
                f.write(f"<{role}>\n" + "".join(s[1] if s[0] == "text" else f"<IMAGE {os.path.basename(s[1])}>"
                                                  for s in segs) + "\n")
        _write_json(summary, os.path.join(wd, "smoke_summary.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
