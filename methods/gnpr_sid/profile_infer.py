#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np

from methods.gnpr_sid import data as D
from methods.gnpr_sid import llm as L
from methods.gnpr_sid import sid as S
from poibench.exports import Export, export_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "work", "gnpr_sid", "smoke", "nyc", "H-100"))
    ap.add_argument("--compare", default=None, help="s1 predictions parquet of the same checkpoint (equality check)")
    a = ap.parse_args()
    import torch
    os.makedirs(os.path.join(ROOT, "logs", "gnpr_sid"), exist_ok=True)
    ex = Export(export_dir("nyc", "H-100", ROOT))
    sids, _ = S.load_sids(os.path.join(a.ckpt, "sid"))
    p2c = {int(k): D.encode_item(v) for k, v in sids.items()}
    q = ex.queries("val")["query_id"].to_numpy(np.int64)
    q = np.sort(np.random.default_rng(0).choice(q, size=a.n, replace=False))
    pr = D.eval_prompts(ex, "val", q, p2c)
    cand = ex.candidates("val")
    cmap = dict(zip(cand["query_id"].astype(np.int64), cand["cand_venue_ids"]))
    t0 = time.time()
    g = L.Generator(os.path.join(a.ckpt, "sft", "final_sft"), p2c)
    rep = dict(load_s=time.time() - t0, keep_kw=g.keep_kw, attn=getattr(g.model.config, "_attn_implementation", None),
               n=a.n, batch=L.INFER_BATCH, trie_depth=g.trie.max_depth)
    sync = torch.cuda.synchronize
    prompts = pr["prompt"].tolist()
    tim = dict(prefill=0.0, cand=0.0, beam=0.0)
    with torch.inference_mode():
        for bi in range(0, len(prompts), L.INFER_BATCH):
            ids = list(range(bi, min(bi + L.INFER_BATCH, len(prompts))))
            sync(); t = time.time()
            cache, mask, logp0, _ = g._prefill([prompts[i] for i in ids])
            sync(); tim["prefill"] += time.time() - t; t = time.time()
            g.score_candidates(cache, mask, logp0, [cmap[int(q[i])] for i in ids])
            sync(); tim["cand"] += time.time() - t; t = time.time()
            g.beam_search(cache, mask, logp0)
            sync(); tim["beam"] += time.time() - t
            del cache
        rep["seconds"] = tim
        rep["s_per_query"] = {k: v / len(prompts) for k, v in tim.items()}
        from torch.profiler import ProfilerActivity, profile
        ids = list(range(L.INFER_BATCH))
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            cache, mask, logp0, _ = g._prefill([prompts[i] for i in ids])
            g.score_candidates(cache, mask, logp0, [cmap[int(q[i])] for i in ids])
            g.beam_search(cache, mask, logp0)
            sync()
        rep["profile_cuda"] = prof.key_averages().table(sort_by="cuda_time_total", row_limit=25)
        rep["profile_cpu"] = prof.key_averages().table(sort_by="cpu_time_total", row_limit=25)
        if a.compare:
            import pandas as pd
            from methods.gnpr_sid import run as R
            ref = pd.read_parquet(a.compare)
            qq = ref["query_id"].astype(int).tolist()
            pr2 = D.eval_prompts(ex, "val", qq, p2c)
            t = time.time()
            res = g.run(pr2["prompt"].tolist(), [cmap[x] for x in qq])
            sync()
            rep["compare_wall_s"] = time.time() - t
            rep["compare_s_per_query"] = rep["compare_wall_s"] / len(qq)
            new = R.rank_outputs(qq, res, cmap)
            same_top10 = [list(x)[:10] == list(y)[:10] for x, y in zip(new["ranked_venue_ids"], ref["ranked_venue_ids"])]
            same_set10 = [set(list(x)[:10]) == set(list(y)[:10]) for x, y in zip(new["ranked_venue_ids"], ref["ranked_venue_ids"])]
            same_c = [list(x) == list(y) for x, y in zip(new["cand_ranked_venue_ids"], ref["cand_ranked_venue_ids"])]
            gaps = []
            for (beam, _), ok in zip(res, same_top10):
                sc = [s for _, s in beam]
                gaps.append(float(min(abs(np.diff(sc)))) if len(sc) > 1 else float("nan"))
            rep["compare"] = dict(n=len(qq), same_top10_order=float(np.mean(same_top10)),
                                  same_top10_set=float(np.mean(same_set10)), same_cand_order=float(np.mean(same_c)),
                                  min_adjacent_beam_gap_of_mismatches=[g_ for g_, ok in zip(gaps, same_top10) if not ok][:20])
            new.to_parquet(os.path.join(ROOT, "logs", "gnpr_sid", "profile_compare_s2.parquet"), index=False)
    rep["peak_gpu_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
    out = os.path.join(ROOT, "logs", "gnpr_sid", f"profile_infer_{time.strftime('%Y%m%dT%H%M%S')}.json")
    with open(out, "w") as f:
        json.dump(rep, f, indent=1)
    print(json.dumps({k: v for k, v in rep.items() if not k.startswith("profile")}, indent=1))
    print(rep["profile_cuda"])
    print(rep["profile_cpu"])


if __name__ == "__main__":
    main()
