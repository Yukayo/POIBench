#!/usr/bin/env python
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import resource
import shutil
import socket
import subprocess
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(HERE))
from poibench.cost import CostRecorder
from poibench.exports import Export, export_dir as _export_dir
from gallm import prompts as PR

ADAPTER_VERSION = "g1"
SEED = 0
TOPK = 50
UP = PR.UPSTREAM_DIR
LLM4POI_DIR = os.path.join(ROOT, "third_party", "LLM4POI")
LLM4POI_COMMIT = "83a44f4eece089828f58bae258b4692592eb3213"
BASE_REPO = "Yukang/Llama-2-7b-longlora-32k-ft"
BASE_REVISION = "ab48674ffc55568ffe2a1207ef0e711c2febbaaf"
BASE_NAME = "Llama-2-7b-longlora-32k-ft"
PY_GALLM = os.environ.get("PY_GALLM") or os.environ.get("POIBENCH_PY_GALLM") or "python"
README_ARGS = ["--bf16", "True", "--lora_r", "256", "--lora_alpha", "512", "--model_max_length", "32768",
               "--use_flash_attn", "True", "--low_rank_training", "True", "--num_train_epochs", "<EPOCHS>",
               "--per_device_train_batch_size", "1", "--per_device_eval_batch_size", "2",
               "--gradient_accumulation_steps", "1", "--evaluation_strategy", "no", "--save_strategy", "steps",
               "--save_steps", "1000", "--save_total_limit", "2", "--learning_rate", "2e-5", "--weight_decay", "0.0",
               "--warmup_steps", "20", "--lr_scheduler_type", "constant_with_warmup", "--logging_steps", "1",
               "--deepspeed", "ds_configs/stage2.json", "--tf32", "True"]
README_EPOCHS = 3
EPOCHS = 1
README_ARGS = [EPOCHS and str(EPOCHS) if x == "<EPOCHS>" else x for x in README_ARGS]
GEN = dict(max_new_tokens=4, do_sample=True, temperature=0.6, top_k=40, top_p=0.1, seq_len=32768, context_size=32768)
STAGE_SCRIPT = {"stage1": "finetune_FusionGeoEncoder.py", "stage2": "finetune_LLM_with_FusionGeoEncoder.py"}
SRC_FILES = ["finetune_FusionGeoEncoder.py", "finetune_LLM_with_FusionGeoEncoder.py", "test_new_fusion_gps.py",
             "llama_attn_replace_sft.py", "geo_model/fusion_geo_encoder.py", "zero_to_fp32.py", "LICENSE"]
PATCHES = {
    "finetune_FusionGeoEncoder.py": [
        ("set_seed(42)\n", "set_seed(int(os.environ.get('POIBENCH_SEED', '0')))  # POIBench P2: seed 0\n"),
        ('    if "nyc" in data_args.data_path.lower():\n',
         '    if True:  # POIBench P1: GPS_id = venue id 0..V-1 (upstream NYC branch), any dataset name\n'),
        ("    trainer.train(resume_from_checkpoint=False)\n",
         "    trainer.train(resume_from_checkpoint=(os.environ.get('POIBENCH_RESUME_CKPT') or False))  # POIBench P3\n"),
    ],
    "finetune_LLM_with_FusionGeoEncoder.py": [
        ('    if "nyc" in data_args.data_path.lower():\n',
         '    if True:  # POIBench P1: GPS_id = venue id 0..V-1 (upstream NYC branch), any dataset name\n'),
        ("    trainer.train(resume_from_checkpoint=False)\n",
         "    trainer.train(resume_from_checkpoint=(os.environ.get('POIBENCH_RESUME_CKPT') or False))  # POIBench P3\n"),
    ],
}
PATCHES_P5 = {'finetune_FusionGeoEncoder.py': [('from geo_model.fusion_geo_encoder import FusionGeoEncoder4096 as FusionGeoEncoder\n', 'from geo_model.fusion_geo_encoder import FusionGeoEncoder4096 as FusionGeoEncoder\nfrom poibench_vec_geo import vec_fusion  # POIBench P5 (batched GPS encoder, implementation only)\n'), ('        geo_embeddings = []\n        for lat, lng in gps_coordinates:\n            embedding_4096 = module.geo_encoder(latitude=lat, longitude=lng)  # [1,4096]\n            geo_embeddings.append(embedding_4096.squeeze(0))\n        geo_embeddings = torch.stack(geo_embeddings).to(module.weight.device)  # [num_gps_tokens, 4096]\n', '        geo_embeddings = vec_fusion(module.geo_encoder, gps_coordinates).to(module.weight.device)  # POIBench P5\n')], 'finetune_LLM_with_FusionGeoEncoder.py': [('from geo_model.fusion_geo_encoder import FusionGeoEncoder4096 as FusionGeoEncoder\n', 'from geo_model.fusion_geo_encoder import FusionGeoEncoder4096 as FusionGeoEncoder\nfrom poibench_vec_geo import vec_fusion  # POIBench P5 (batched GPS encoder, implementation only)\n'), ('        fusion_embeddings = []\n        for lat, lng in gps_coordinates:\n            emb = fusion_geo_encoder(lat, lng)  # [1,4096]\n            fusion_embeddings.append(emb.squeeze(0))\n\n        fusion_embeddings = torch.stack(fusion_embeddings).to(module.weight.device)\n', '        fusion_embeddings = vec_fusion(fusion_geo_encoder, gps_coordinates).to(module.weight.device)  # POIBench P5\n')], 'test_new_fusion_gps.py': [('from geo_model.fusion_geo_encoder import FusionGeoEncoder4096 as FusionGeoEncoder\n', 'from geo_model.fusion_geo_encoder import FusionGeoEncoder4096 as FusionGeoEncoder\nfrom poibench_vec_geo import vec_fusion  # POIBench P5 (batched GPS encoder, implementation only)\n'), ('        fusion_embeddings = []\n        for lat, lng in gps_coordinates:\n            # Use the FusionGeoEncoder to generate embeddings\n            emb_4096 = fusion_geo_encoder(latitude=lat, longitude=lng)  # [1,4096]\n            fusion_embeddings.append(emb_4096.squeeze(0))\n\n        fusion_embeddings = torch.stack(fusion_embeddings).to(module.weight.device)\n', '        fusion_embeddings = vec_fusion(fusion_geo_encoder, gps_coordinates).to(module.weight.device)  # POIBench P5\n')]}
for _f, _p in PATCHES_P5.items():
    PATCHES.setdefault(_f, []).extend(_p)
CANARY_KEY = ("nyc", "H-100", "default")
NPROC = int(os.environ.get("POIBENCH_GALLM_NPROC", "1"))


def log(*a):
    print(f"[gallm {time.strftime('%H:%M:%S', time.gmtime())}]", *a, flush=True)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _write_json(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f, indent=1, default=str)
    os.replace(path + ".tmp", path)


def _read_json(path: str):
    with open(path) as f:
        return json.load(f)


def _write_parquet(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path + ".tmp", index=False)
    os.replace(path + ".tmp", path)


def _sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def method_tag(cfg: str) -> str:
    return PR.CFGS[cfg]["method"]


def visible_gpus() -> list:
    v = os.environ.get("CUDA_VISIBLE_DEVICES")
    if v is not None and v.strip() != "":
        return [x for x in v.split(",") if x.strip() != ""]
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], capture_output=True,
                             text=True, timeout=60).stdout.split()
        return out
    except Exception:
        return []


def host() -> str:
    return socket.gethostname()


def ensure_src() -> str:
    h = hashlib.sha256()
    for f in SRC_FILES:
        h.update(f.encode())
        h.update(_sha(os.path.join(UP, f)).encode())
    h.update(json.dumps(PATCHES, sort_keys=True).encode())
    h.update(_sha(os.path.join(LLM4POI_DIR, "ds_configs", "stage2.json")).encode())
    h.update(_sha(os.path.join(HERE, "src_add", "poibench_vec_geo.py")).encode())
    fp = h.hexdigest()[:12]
    d = os.path.join(ROOT, "work", "gallm", f"src_{fp}")
    if os.path.exists(os.path.join(d, ".complete")):
        return d
    tmp = d + f".tmp{os.getpid()}"
    shutil.rmtree(tmp, ignore_errors=True)
    for f in SRC_FILES:
        os.makedirs(os.path.dirname(os.path.join(tmp, f)), exist_ok=True)
        src = open(os.path.join(UP, f), "rb").read()
        if f in PATCHES:
            s = src.decode()
            for a, b in PATCHES[f]:
                if s.count(a) != 1:
                    raise RuntimeError(f"patch target in {f} found {s.count(a)} times: {a!r}")
                s = s.replace(a, b)
            src = s.encode()
        with open(os.path.join(tmp, f), "wb") as g:
            g.write(src)
    shutil.copyfile(os.path.join(HERE, "src_add", "poibench_vec_geo.py"), os.path.join(tmp, "poibench_vec_geo.py"))
    os.makedirs(os.path.join(tmp, "ds_configs"), exist_ok=True)
    shutil.copyfile(os.path.join(LLM4POI_DIR, "ds_configs", "stage2.json"), os.path.join(tmp, "ds_configs", "stage2.json"))
    with open(os.path.join(tmp, ".complete"), "w") as g:
        g.write(json.dumps(dict(upstream=PR.UPSTREAM_COMMIT, llm4poi_ds_config=LLM4POI_COMMIT, fp=fp)))
    try:
        os.rename(tmp, d)
    except OSError:
        shutil.rmtree(tmp, ignore_errors=True)
        if not os.path.exists(os.path.join(d, ".complete")):
            raise
    return d


def base_model() -> dict:
    for var in ("POIBENCH_MODEL_DIR_X", "POIBENCH_MODEL_DIR"):
        b = os.environ.get(var)
        if not b:
            continue
        p = os.path.join(b, BASE_NAME)
        mk = os.path.join(p, ".poibench_complete")
        if os.path.isfile(mk):
            m = _read_json(mk)
            if m.get("sha") != BASE_REVISION:
                raise RuntimeError(f"{p}: revision {m.get('sha')} != pinned {BASE_REVISION}")
            return dict(path=p, revision=m["sha"], repo=m.get("repo"))
    raise RuntimeError(f"{BASE_NAME}: no verified local copy (.poibench_complete) under $POIBENCH_MODEL_DIR[_X]")


def run_dirs(ds: str, policy: str, cfg: str, smoke: bool, mock: bool = False) -> tuple[str, str]:
    tag = cfg + ("_mock" if mock else "") + ("_smoke" if smoke else "")
    wd = os.path.join(ROOT, "work", "gallm", ds, policy, tag)
    ck_root = os.environ.get("POIBENCH_GALLM_CKPT_ROOT") or os.path.join(ROOT, "work", "gallm", "ckpt")
    return wd, os.path.join(ck_root, ds, policy, tag)


def prepare(ex: Export, ds: str, cfg: str, wd: str, splits: list, smoke_n: int = 0) -> dict:
    t0, c0 = time.perf_counter(), time.process_time()
    splits = ["test", "val"]
    fp = dict(code=PR.code_fingerprint(), export=ex.manifest.get("files"), cfg=cfg, smoke_n=smoke_n,
              splits=sorted(splits))
    fpp = os.path.join(wd, "prepare_meta.json")
    if os.path.exists(fpp) and _read_json(fpp).get("fingerprint") == fp:
        return _read_json(fpp)
    os.makedirs(wd, exist_ok=True)
    qids = None
    if smoke_n:
        q = ex.queries("train")
        qids = q["query_id"].to_numpy()[:: max(1, len(q) // (4 * smoke_n))][: 4 * smoke_n]
    meta = dict(fingerprint=fp, created=_now(), host=host(), files={})
    for kind, name in (("stage1", "stage1_train.json"), ("llm", "stage2_train.json")):
        s = PR.train_samples(ex, kind, cfg, query_ids=qids)
        if smoke_n:
            s = s.iloc[:smoke_n]
        recs = [{"question": a, "answer": b} for a, b in zip(s["question"], s["answer"])]
        with open(os.path.join(wd, name + ".tmp"), "w") as f:
            json.dump(recs, f)
        os.replace(os.path.join(wd, name + ".tmp"), os.path.join(wd, name))
        meta["files"][name] = dict(n=len(recs), sha256=_sha(os.path.join(wd, name)),
                                   n_hist_mean=float(s["n_hist"].mean()), n_cur_mean=float(s["n_cur"].mean()))
    g = PR.gps_mapping(ex)
    g.to_csv(os.path.join(wd, "gps_mapping.csv"), index=False)
    meta["files"]["gps_mapping.csv"] = dict(n=len(g), sha256=_sha(os.path.join(wd, "gps_mapping.csv")))
    tt = PR.CFGS[cfg]["target_time"]
    for s in splits:
        q = ex.queries(s)
        ids = q["query_id"].to_numpy(np.int64)
        if smoke_n:
            ids = ids[:smoke_n]
        p = PR.query_prompts(ex, s, ids, cfg, PR.load_target_time(ds, s) if tt else None)
        _write_parquet(p, os.path.join(wd, f"prompts_{s}.parquet"))
        meta["files"][f"prompts_{s}.parquet"] = dict(n=len(p), sha256=_sha(os.path.join(wd, f"prompts_{s}.parquet")))
    meta["stage"] = dict(wall_s=time.perf_counter() - t0, cpu_core_s=time.process_time() - c0, gpu_s=0.0,
                         peak_mem_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, one_time=True)
    _write_json(meta, fpp)
    return meta


def _latest_ckpt(out_dir: str):
    best = None
    for p in glob.glob(os.path.join(out_dir, "checkpoint-*")):
        try:
            n = int(p.rsplit("-", 1)[1])
        except ValueError:
            continue
        if os.path.isfile(os.path.join(p, "trainer_state.json")) and os.path.isfile(os.path.join(p, "latest")):
            if best is None or n > best[0]:
                best = (n, p)
    return best[1] if best else None


def gpu_env(n: int = 1) -> dict:
    env = dict(os.environ)
    g = visible_gpus()
    if len(g) < n:
        raise RuntimeError(f"needs {n} GPU(s), visible {g}")
    env["CUDA_VISIBLE_DEVICES"] = ",".join(g[:n])
    return env


def _rebase(p):
    if not p or os.path.exists(p) or "/work/gallm/ckpt/" not in p:
        return p
    ck_root = os.environ.get("POIBENCH_GALLM_CKPT_ROOT") or os.path.join(ROOT, "work", "gallm", "ckpt")
    return os.path.join(ck_root, p.split("/work/gallm/ckpt/", 1)[1])


def _load_done(donep: str) -> dict:
    d = _read_json(donep)
    for k in ("out", "encoder"):
        if k in d:
            d[k] = _rebase(d[k])
    return d


def train_stage(stage: str, src: str, wd: str, ckdir: str, extra: list, max_steps: int = 0) -> dict:
    out = os.path.join(ckdir, stage)
    donep = os.path.join(out, "DONE.json")
    if os.path.exists(donep):
        return _load_done(donep)
    os.makedirs(out, exist_ok=True)
    bm = base_model()
    data = os.path.join(wd, "stage1_train.json" if stage == "stage1" else "stage2_train.json")
    nproc = NPROC
    cmd = [PY_GALLM, "-m", "torch.distributed.run", f"--nproc_per_node={nproc}", f"--master_port={_free_port()}",
           STAGE_SCRIPT[stage], "--model_name_or_path", bm["path"], "--output_dir", out,
           "--gps_mapping_path", os.path.join(wd, "gps_mapping.csv"), "--data_path", data, "--seed", str(SEED),
           "--data_seed", str(SEED)] + README_ARGS + extra
    if max_steps:
        cmd += ["--max_steps", str(max_steps)]
    env = gpu_env(nproc)
    env["POIBENCH_SEED"] = str(SEED)
    env.pop("POIBENCH_RESUME_CKPT", None)
    sess_path = os.path.join(out, "sessions.json")
    sessions = _read_json(sess_path) if os.path.exists(sess_path) else []
    rc_ = _latest_ckpt(out)
    if rc_:
        env["POIBENCH_RESUME_CKPT"] = rc_
    t0, c0 = time.perf_counter(), resource.getrusage(resource.RUSAGE_CHILDREN)
    log(f"{stage}: start (resume={rc_}) -> {out}")
    lp = os.path.join(out, f"train_{len(sessions)}.log")
    samp = _NvSampler(env["CUDA_VISIBLE_DEVICES"])
    samp.start()
    with open(lp, "w") as f:
        f.write(" ".join(cmd) + "\n")
        f.flush()
        p = subprocess.run(cmd, cwd=src, env=env, stdout=f, stderr=subprocess.STDOUT)
    samp.stop()
    c1 = resource.getrusage(resource.RUSAGE_CHILDREN)
    wall = time.perf_counter() - t0
    sessions.append(dict(start=t0, wall_s=wall, peak_gpu_mem_nvml_gb=samp.peak_gb, cpu_core_s=(c1.ru_utime + c1.ru_stime) - (c0.ru_utime + c0.ru_stime),
                         rc=p.returncode, resume=rc_, host=host(), log=lp, n_gpus=nproc, ended=_now()))
    _write_json(sessions, sess_path)
    if p.returncode != 0:
        raise RuntimeError(f"{stage} failed rc={p.returncode}; log {lp}")
    st = {}
    sp = os.path.join(out, "trainer_state.json")
    if os.path.exists(sp):
        st = _read_json(sp)
    peak = max([x.get("peak_gpu_mem_nvml_gb") or 0 for x in sessions] or [0]) or None
    done = dict(stage=stage, out=out, finished=_now(), host=host(), global_step=st.get("global_step"),
                train_wall_s=sum(s["wall_s"] for s in sessions), train_cpu_core_s=sum(s["cpu_core_s"] for s in sessions),
                train_gpu_s=sum(s["wall_s"] * s["n_gpus"] for s in sessions), sessions=sessions,
                peak_gpu_mem_gb=peak, base=bm, max_steps=max_steps or None, data_sha256=_sha(data))
    if stage == "stage1":
        enc = os.path.join(out, "fusion_geo_encoder_final_merged.pth")
        if not os.path.exists(enc):
            raise RuntimeError("stage1 finished without fusion_geo_encoder_final_merged.pth")
        done["encoder"] = enc
        done["encoder_sha256"] = _sha(enc)
    else:
        if not os.path.exists(os.path.join(out, "trainable_params.bin")):
            raise RuntimeError("stage2 finished without trainable_params.bin")
        done["model_fingerprint"] = model_fingerprint(out)
    for c in glob.glob(os.path.join(out, "checkpoint-*")):
        shutil.rmtree(c, ignore_errors=True)
    _write_json(done, donep)
    log(f"{stage}: done in {done['train_wall_s'] / 3600:.2f} h, step {done['global_step']}")
    return done


class _NvSampler:

    def __init__(self, gpu: str):
        import threading
        self.gpu, self.peak_gb, self._stop = gpu, None, threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            try:
                o = subprocess.run(["nvidia-smi", "-i", str(self.gpu), "--query-gpu=memory.used",
                                    "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=60)
                v = max(float(x) for x in o.stdout.split()) * 1.048576e-3
                self.peak_gb = max(self.peak_gb or 0.0, v)
            except Exception:
                pass
            self._stop.wait(20)

    def start(self):
        self._t.start()

    def stop(self):
        self._stop.set()
        self._t.join(timeout=70)


def _free_port() -> int:
    s = socket.socket()
    s.bind(("", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def model_fingerprint(stage2_dir: str) -> str:
    h = hashlib.sha256()
    for f in sorted(os.listdir(stage2_dir)):
        if f.startswith("adapter_") or f == "trainable_params.bin":
            h.update(f.encode())
            h.update(_sha(os.path.join(stage2_dir, f)).encode())
    s1 = os.path.join(os.path.dirname(stage2_dir), "stage1", "DONE.json")
    if os.path.exists(s1):
        h.update(_read_json(s1).get("encoder_sha256", "").encode())
    return h.hexdigest()[:20]


def _cache_key(mfp: str, qid: int, prompt: str, max_new: int) -> str:
    return hashlib.sha256(f"{mfp}|{SEED}|{qid}|{max_new}|{json.dumps(GEN, sort_keys=True)}|{prompt}".encode()).hexdigest()


class ResponseCache:

    def __init__(self, mfp: str):
        self.dir = os.path.join(ROOT, "work", "gallm", "llm_cache", mfp)
        os.makedirs(self.dir, exist_ok=True)
        self.d = {}
        self.run_of = {}
        for f in sorted(glob.glob(os.path.join(self.dir, "*.parquet"))):
            t = pd.read_parquet(f)
            self.d.update({k: json.loads(v) for k, v in zip(t["key"], t["rec"])})
            if "run" in t.columns:
                self.run_of.update({k: r for k, r in zip(t["key"], t["run"]) if isinstance(r, str) and r})

    def add(self, recs: dict, run: str | None = None) -> None:
        if not recs:
            return
        f = os.path.join(self.dir, f"part_{_now().replace(':', '')}_{os.getpid()}_{len(self.d)}.parquet")
        cols = {"key": list(recs), "rec": [json.dumps(v) for v in recs.values()]}
        if run:
            cols["run"] = [run] * len(recs)
        _write_parquet(pd.DataFrame(cols), f)
        self.d.update(recs)
        if run:
            self.run_of.update({k: run for k in recs})

    def ledger(self, run: str) -> dict | None:
        p = os.path.join(self.dir, "runs", f"{run}.json")
        return _read_json(p) if os.path.exists(p) else None


def _weights_mtime(stage2_dir: str | None) -> float | None:
    if not stage2_dir or not os.path.isdir(stage2_dir):
        return None
    t = [os.path.getmtime(os.path.join(stage2_dir, f)) for f in os.listdir(stage2_dir)
         if f.startswith("adapter_") or f == "trainable_params.bin"]
    return max(t) if t else None


def vet_resume(wd: str, tag: str, todo: pd.DataFrame, n: int, backend: str, stage2_dir: str | None) -> dict:
    inp = os.path.join(wd, "gen", f"{tag}_in.parquet")
    old_prompt, old_key = {}, {}
    if os.path.exists(inp):
        t = pd.read_parquet(inp)
        old_prompt = dict(zip(t["query_id"].astype(int), t["prompt"]))
        if "key" in t.columns:
            old_key = dict(zip(t["query_id"].astype(int), t["key"]))
    want = dict(zip(todo["query_id"].astype(int), todo["key"]))
    prompt = dict(zip(todo["query_id"].astype(int), todo["prompt"]))
    tq = [int(q) for q in todo["query_id"]]
    wm = _weights_mtime(stage2_dir) if backend == "hf" else None
    resumed, dropped, stamped = {}, 0, 0
    for k in range(n):
        o = os.path.join(wd, "gen", f"{tag}_out{k}.jsonl")
        if not os.path.exists(o):
            continue
        mine = set(tq[k * len(tq) // n:(k + 1) * len(tq) // n])
        fresh_file = wm is None or os.path.getmtime(o) >= wm
        lines, changed = [], False
        for line in open(o):
            try:
                r = json.loads(line)
            except ValueError:
                changed = True
                continue
            if not line.endswith("\n"):
                line += "\n"
                changed = True
            q = None if r.get("kind") == "meta" else int(r["query_id"])
            if q is None or q not in want:
                lines.append(line)
                continue
            if "key" in r:
                ok = r["key"] == want[q]
            else:
                ok = (q in old_key and old_key[q] == want[q] and old_prompt.get(q) == prompt[q] and fresh_file
                      and not (backend == "hf" and not r.get("skipped") and not r.get("gen_s")))
                if ok:
                    line = json.dumps(dict(r, key=want[q])) + "\n"
                    stamped += 1
                    changed = True
            if not ok:
                dropped += 1
                changed = True
                continue
            lines.append(line)
            if q in mine:
                resumed[q] = float(r.get("gen_s") or 0.0)
        if changed:
            with open(o + ".tmp", "w") as f:
                f.writelines(lines)
            os.replace(o + ".tmp", o)
    if dropped or stamped:
        log(f"infer {tag}: resume check: {dropped} answer(s) to another request removed (regenerated), "
            f"{stamped} legacy answer(s) verified against the previous input's request key and keyed")
    return resumed


def run_infer(prompts: pd.DataFrame, wd: str, stage1: dict | None, stage2_dir: str | None, ds: str, n_venues: int,
              backend: str, tag: str, info: dict | None = None) -> tuple[list, dict]:
    os.makedirs(os.path.join(wd, "gen"), exist_ok=True)
    if backend != "hf":
        tag = f"{tag}_{backend}"
    inp = os.path.join(wd, "gen", f"{tag}_in.parquet")
    max_new = max(GEN["max_new_tokens"], len(str(n_venues - 1)))
    mfp = model_fingerprint(stage2_dir) if backend == "hf" else "mock-v1"
    cache = ResponseCache(mfp)
    keys = [_cache_key(mfp, int(q), p, max_new) for q, p in zip(prompts["query_id"], prompts["prompt"])]
    if info is not None:
        info.update(keys=keys, cached_keys={k for k in keys if k in cache.d}, cache=cache, run=None)
    todo_mask = [k not in cache.d for k in keys]
    todo = prompts.loc[todo_mask, ["query_id", "prompt"]].assign(key=[k for k, m in zip(keys, todo_mask) if m])
    log(f"infer {tag}: {len(prompts)} prompts, {len(prompts) - len(todo)} cached, model {mfp}")
    if len(todo) == 0:
        res = [dict(cache.d[k], cached=True) for k in keys]
        return res, dict(wall_s=0.0, cpu_core_s=0.0, n_workers=0, gpu_s=0.0, max_new_tokens=max_new,
                         peak_gpu_mem_gb=None, load_s=[], keys_loaded=None, n_generated=0, n_cached=len(keys),
                         model_fingerprint=mfp)
    t0 = time.perf_counter()
    c0 = resource.getrusage(resource.RUSAGE_CHILDREN)
    gpus = visible_gpus() if backend == "hf" else ["0"]
    gpus = gpus[:max(1, min(len(gpus), len(todo) // 64 or 1))]
    n = len(gpus)
    resumed = vet_resume(wd, tag, todo, n, backend, stage2_dir)
    _write_parquet(todo, inp)
    procs, outs = [], []
    for k, g in enumerate(gpus):
        o = os.path.join(wd, "gen", f"{tag}_out{k}.jsonl")
        outs.append(o)
        cmd = [PY_GALLM if backend == "hf" else sys.executable, "-I", os.path.join(HERE, "infer.py"), "--src", ensure_src(),
               "--inp", inp, "--out", o, "--shard", str(k), "--nshards", str(n), "--seed", str(SEED),
               "--max_new_tokens", str(max_new), "--backend", backend]
        if backend == "hf":
            cmd += ["--base_model", base_model()["path"], "--stage2", stage2_dir, "--encoder", stage1["encoder"],
                    "--gps_mapping", os.path.join(wd, "gps_mapping.csv")]
        env = dict(os.environ)
        if backend == "hf":
            env["CUDA_VISIBLE_DEVICES"] = str(g)
        lf = open(os.path.join(wd, "gen", f"{tag}_out{k}.log"), "a")
        procs.append((subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT), lf))
    rcs = []
    for p, lf in procs:
        rcs.append(p.wait())
        lf.close()
    if any(rcs):
        raise RuntimeError(f"inference failed rcs={rcs}; logs {wd}/gen/{tag}_out*.log")
    c1 = resource.getrusage(resource.RUSAGE_CHILDREN)
    res = {}
    stats = dict(peak_gpu_mem_gb=None, load_s=[], keys_loaded=None)
    want = dict(zip(todo["query_id"].astype(int), todo["key"]))
    for o in outs:
        for line in open(o):
            r = json.loads(line)
            if r.get("kind") == "meta":
                stats["load_s"].append(r.get("load_s"))
                stats["keys_loaded"] = r.get("trainable_keys_matched")
                if r.get("peak_gpu_mem_gb") is not None:
                    stats["peak_gpu_mem_gb"] = max(stats["peak_gpu_mem_gb"] or 0, r["peak_gpu_mem_gb"])
                continue
            q = int(r["query_id"])
            if q in want and r.get("key") != want[q]:
                continue
            res[q] = {x: v for x, v in r.items() if x != "key"}
    missing = [q for q in todo["query_id"] if int(q) not in res]
    if missing:
        raise RuntimeError(f"{len(missing)} prompts without output")
    new = {k: res[int(q)] for k, q in zip(keys, prompts["query_id"]) if k not in cache.d}
    wall = time.perf_counter() - t0
    gst = dict(wall_s=wall, cpu_core_s=(c1.ru_utime + c1.ru_stime) - (c0.ru_utime + c0.ru_stime), n_workers=n,
               gpu_s=wall * n if backend == "hf" else 0.0, max_new_tokens=max_new, n_generated=len(todo),
               n_cached=len(keys) - len(todo), model_fingerprint=mfp, n_resumed=len(resumed),
               resumed_gen_s=sum(resumed.values()), **stats)
    run = f"{_now().replace(':', '')}_{host()}_{os.getpid()}_{tag}"[:200]
    _write_json(dict(run=run, tag=tag, host=host(), backend=backend, finished=_now(), n_new_keys=len(new),
                     **{k_: gst[k_] for k_ in ("wall_s", "cpu_core_s", "gpu_s", "n_workers", "n_generated",
                                              "n_resumed", "resumed_gen_s", "peak_gpu_mem_gb", "load_s")}),
                os.path.join(cache.dir, "runs", f"{run}.json"))
    cache.add(new, run=run)
    if info is not None:
        info["run"] = run
    return [cache.d[k] for k in keys], gst


def parse_answer(text: str):
    import re
    d = re.sub(r"[^0-9]", "", text or "")
    return int(d) if d else None


def ranked_list(answer, candidates, n_venues: int, k: int = TOPK) -> tuple[list, bool]:
    valid = answer is not None and 0 <= int(answer) < n_venues
    out, seen = [], set()
    for v in ([int(answer)] if valid else []) + [int(x) for x in candidates]:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out[:k], valid


def parse_outputs(qids, res: list, ex: Export, split: str) -> pd.DataFrame:
    V = len(ex.venues)
    c = ex.candidates(split)
    cmap = dict(zip(c["query_id"].astype(np.int64), c["cand_venue_ids"]))
    rows = []
    for qid, r in zip(qids, res):
        a = parse_answer(r["text"]) if not r.get("skipped") else None
        lst, ok = ranked_list(a, cmap[int(qid)], V)
        rows.append(dict(query_id=int(qid), ranked_venue_ids=lst, answer_venue_id=int(a) if ok else -1, valid=ok,
                         text=r["text"], skipped=bool(r.get("skipped")), n_prompt_tokens=int(r["n_prompt_tokens"]),
                         gen_s=float(r.get("gen_s") or 0)))
    return pd.DataFrame(rows)


def predict(export_dir, split, query_ids, work_dir=None):
    backend = os.environ.get("GALLM_BACKEND", "hf")
    cfg = os.environ.get("GALLM_CFG", "default")
    if cfg not in PR.CFGS:
        raise ValueError(f"GALLM_CFG={cfg!r} is not a GA-LLM configuration ({sorted(PR.CFGS)})")
    ex = Export(export_dir)
    ds = ex.manifest.get("ds")
    qids = [int(x) for x in query_ids]
    tt = PR.CFGS[cfg]["target_time"]
    df = PR.query_prompts(ex, split, qids, cfg, PR.load_target_time(ds, split) if tt else None)
    wd = work_dir or os.path.join(ROOT, "work", "gallm", "predict_tmp")
    os.makedirs(wd, exist_ok=True)
    s1 = s2 = None
    if backend == "hf":
        if cfg != CANARY_KEY[2] and not (os.environ.get("GALLM_STAGE1_DONE") and os.environ.get("GALLM_STAGE2")):
            raise ValueError(f"GALLM_CFG={cfg}: the fixed canary checkpoint is trained with cfg {CANARY_KEY[2]}; set "
                             f"GALLM_STAGE1_DONE and GALLM_STAGE2 to a {cfg} checkpoint (or use GALLM_BACKEND=mock)")
        _, ck = run_dirs(*CANARY_KEY, smoke=True)
        s1 = _load_done(os.environ.get("GALLM_STAGE1_DONE") or os.path.join(ck, "stage1", "DONE.json"))
        s2 = os.environ.get("GALLM_STAGE2") or os.path.join(ck, "stage2")
    g = PR.gps_mapping(ex)
    g.to_csv(os.path.join(wd, "gps_mapping.csv"), index=False)
    tag = hashlib.sha256(("|".join(df["prompt_sha256"])).encode()).hexdigest()[:12]
    res, _ = run_infer(df, wd, s1, s2, ds, len(ex.venues), backend, f"canary_{split}_{tag}")
    out = parse_outputs(qids, res, ex, split)
    if work_dir:
        _write_parquet(df, os.path.join(work_dir, "prompts.parquet"))
        _write_parquet(out, os.path.join(work_dir, "responses.parquet"))
    ret = pd.DataFrame({"query_id": out["query_id"], "ranked_venue_ids": out["ranked_venue_ids"]})
    ret.attrs["adapter_config"] = dict(cfg=cfg, backend=backend, target_time_in_prompt=bool(tt),
                                       model_fingerprint=None if s2 is None else model_fingerprint(s2))
    return ret


def keys_digest(keys) -> str:
    return hashlib.sha256("\n".join(keys).encode()).hexdigest()


def keep_shard_measurement(path: str, new: dict) -> dict:
    if not os.path.exists(path):
        return new
    try:
        old = _read_json(path)
    except (OSError, ValueError):
        return new
    same = all(old.get(k) == new.get(k) for k in ("shard", "nshards", "n", "model_fingerprint"))
    if old.get("keys_sha256") and old["keys_sha256"] != new.get("keys_sha256"):
        same = False
    h_old, h_new = old.get("n_cached"), new.get("n_cached")
    if not same or not h_new or h_old is None or h_old >= h_new:
        return new
    run = {k: new.get(k) for k in ("wall_s", "cpu_core_s", "gpu_s", "n_workers", "n_generated", "n_cached", "host",
                                   "finished", "gpus", "ledger_run")}
    return dict(old, reruns=list(old.get("reruns") or []) + [run])


def cached_answer_costs(split_of: list, info: dict, shards: list) -> dict:
    keys, pre, cache = info["keys"], info["cached_keys"], info["cache"]
    n = len(keys)
    covered = set()
    for x in shards:
        K, N = int(x["shard"]), int(x["nshards"])
        lo, hi = K * n // N, (K + 1) * n // N
        if "cached_positions" in x:
            not_measured = set(int(i) for i in x["cached_positions"])
            covered.update(k for i, k in enumerate(keys[lo:hi]) if i not in not_measured)
        elif not (x.get("n_cached") and int(x["n_cached"]) >= int(x.get("n") or hi - lo)):
            covered.update(keys[lo:hi])
    ledgers, seen, out = {}, set(), {}
    acc = {}
    for i, k in enumerate(keys):
        if k not in pre or k in covered or k in seen:
            continue
        seen.add(k)
        sp = split_of[i]
        a = acc.setdefault(sp, dict(wall_s=0.0, cpu_core_s=0.0, gpu_s=0.0, n=0, n_ledger=0, n_gen_s=0, runs=set()))
        run = cache.run_of.get(k)
        if run and run not in ledgers:
            ledgers[run] = cache.ledger(run)
        lg = ledgers.get(run) if run else None
        if lg and lg.get("n_generated"):
            m = float(lg["n_generated"])
            a["wall_s"] += (float(lg["wall_s"]) + float(lg.get("resumed_gen_s") or 0.0)) / m
            a["cpu_core_s"] += float(lg["cpu_core_s"]) / m
            a["gpu_s"] += (float(lg["gpu_s"]) + float(lg.get("resumed_gen_s") or 0.0)) / m
            a["n_ledger"] += 1
            a["runs"].add(run)
        else:
            g = float(cache.d[k].get("gen_s") or 0.0)
            a["wall_s"] += g
            a["gpu_s"] += g
            a["n_gen_s"] += 1
        a["n"] += 1
    for sp, a in acc.items():
        if not a["n"]:
            continue
        out[sp] = dict(name="infer_cached", wall_s=a["wall_s"], cpu_core_s=a["cpu_core_s"], gpu_s=a["gpu_s"],
                       peak_mem_gb=0.0, one_time=False, n_items=a["n"],
                       notes=f"answers taken from the response cache: {a['n_ledger']} attributed from the ledger of "
                             f"the {len(a['runs'])} execution(s) that generated them (model load included), "
                             f"{a['n_gen_s']} from their stored generation time (gen_s, 1 GPU, no model load)")
    return out


def candidates_file_decl(ex: Export, split: str) -> str:
    keys = [k for k in ex.manifest["source_sha256"] if k.endswith(f"candidates_{split}.parquet")]
    if len(keys) != 1:
        raise RuntimeError(f"cannot identify the candidate file of {split} in the export manifest: {keys}")
    return os.path.join(ex.manifest["source_dir"], keys[0])


def write_outputs(a, ex: Export, split: str, out: pd.DataFrame, pmeta: dict, stages: list, common: dict,
                  out_base: str) -> str:
    t = method_tag(a.cfg)
    d = os.path.join(out_base, f"{t}@{a.policy}", a.ds)
    preds = pd.DataFrame({"query_id": out["query_id"], "ranked_venue_ids": out["ranked_venue_ids"],
                          "answer_venue_id": out["answer_venue_id"]})
    cr = CostRecorder(method=f"{t}@{a.policy}", ds=a.ds, split=split, extra=common)
    cr.add_stage(name="prepare", **pmeta["stage"])
    for s in stages:
        cr.add_stage(**s)
    _write_parquet(preds, os.path.join(d, f"{split}.parquet"))
    cr.save(os.path.join(d, f"{split}.cost.json"), n_queries=len(preds),
            keep_measured=("method", "ds", "split", "cfg", "n_queries", "generation.model_fingerprint",
                           "export_manifest_files"))
    return os.path.relpath(os.path.join(d, f"{split}.parquet"), ROOT)


def main(argv=None):
    ap = argparse.ArgumentParser(description=(__doc__ or " "), formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ds", required=True)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--splits", nargs="+", default=["val", "test"], choices=["val", "test"])
    ap.add_argument("--cfg", default="default", choices=sorted(PR.CFGS))
    ap.add_argument("--stage", default="all", choices=["all", "prepare", "train", "predict"])
    ap.add_argument("--backend", default="hf", choices=["hf", "mock"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", type=int, default=0, help="number of SFT samples / queries (work/ only)")
    ap.add_argument("--max-steps", type=int, default=0, help="smoke only")
    ap.add_argument("--infer-shard", default=None, help="K/N: generate only the K-th of N contiguous query shards "
                    "(fills the response cache, writes gen/shard_K_of_N_<splits>.json; no preds). A later plain run "
                    "assembles the predictions from the cache and adds the shards' GPU time to the infer stage.")
    ap.add_argument("--canary-prep", action="store_true",
                    help="H-100: train the fixed canary checkpoint (nyc H-100 smoke: 64 samples, 20 steps per stage); "
                         "other policies: prepare their data only")
    a = ap.parse_args(argv)
    if a.canary_prep:
        if a.policy == CANARY_KEY[1] and a.ds == CANARY_KEY[0]:
            return main(["--ds", a.ds, "--policy", a.policy, "--smoke", "64", "--max-steps", "20", "--stage", "train"])
        return main(["--ds", a.ds, "--policy", a.policy, "--smoke", "64", "--stage", "prepare"])
    if a.seed != SEED:
        raise SystemExit("seed 0 only")
    if a.max_steps and not a.smoke:
        raise SystemExit("--max-steps is for smoke runs only")
    ex = Export(_export_dir(a.ds, a.policy))
    wd, ckdir = run_dirs(a.ds, a.policy, a.cfg, bool(a.smoke), mock=a.backend == "mock")
    os.makedirs(wd, exist_ok=True)
    src = ensure_src()
    log(f"{a.ds} {a.policy} cfg={a.cfg} smoke={a.smoke} host={host()} src={src} ckpt={ckdir}")
    pmeta = prepare(ex, a.ds, a.cfg, wd, a.splits, a.smoke)
    log(f"prepared: {json.dumps({k: v.get('n') for k, v in pmeta['files'].items()})} "
        f"({pmeta['stage']['wall_s']:.0f} s)")
    if a.stage == "prepare":
        return 0
    if a.backend == "hf":
        s1 = train_stage("stage1", src, wd, ckdir, [], a.max_steps)
        s2 = train_stage("stage2", src, wd, ckdir, ["--fusion_geoencoder_path", s1["encoder"]], a.max_steps)
    else:
        s1, s2 = None, None
    if a.stage == "train":
        return 0
    out_base = os.path.join(wd, "preds") if (a.smoke or a.backend != "hf") else os.path.join(ROOT, "preds")
    prompts = {s: pd.read_parquet(os.path.join(wd, f"prompts_{s}.parquet")) for s in a.splits}
    allp = pd.concat([prompts[s] for s in a.splits], ignore_index=True)
    sp_tag = "_".join(a.splits)
    if a.infer_shard:
        K, N = (int(x) for x in a.infer_shard.split("/"))
        n = len(allp)
        part = allp.iloc[K * n // N:(K + 1) * n // N]
        sinfo = {}
        _, g = run_infer(part, wd, s1, s2["out"] if s2 else None, a.ds, len(ex.venues), a.backend,
                         f"shard{K}of{N}_{sp_tag}", info=sinfo)
        rec = dict(g, host=host(), shard=K, nshards=N, n=len(part), finished=_now(),
                   gpus=os.environ.get("CUDA_VISIBLE_DEVICES"), keys_sha256=keys_digest(sinfo["keys"]),
                   cached_positions=[i for i, k in enumerate(sinfo["keys"]) if k in sinfo["cached_keys"]],
                   ledger_run=sinfo["run"])
        sp = os.path.join(wd, "gen", f"shard_{K}_of_{N}_{sp_tag}.json")
        _write_json(keep_shard_measurement(sp, rec), sp)
        log(f"shard {K}/{N}: {len(part)} prompts done")
        return 0
    info = {}
    res, gst = run_infer(allp, wd, s1, s2["out"] if s2 else None, a.ds, len(ex.venues), a.backend,
                         f"main_{sp_tag}", info=info)
    execution = dict(wall_s=gst["wall_s"], cpu_core_s=gst["cpu_core_s"], gpu_s=gst["gpu_s"],
                     n_generated=gst["n_generated"], n_cache_hits=len(info["cached_keys"]), n_workers=gst["n_workers"],
                     backend=a.backend, host=host(), finished=_now(), ledger_run=info["run"])
    shards = []
    for f in sorted(glob.glob(os.path.join(wd, "gen", f"shard_*_of_*_{sp_tag}.json"))):
        x = _read_json(f)
        K, N, n = int(x["shard"]), int(x["nshards"]), len(info["keys"])
        if x.get("keys_sha256") and x["keys_sha256"] != keys_digest(info["keys"][K * n // N:(K + 1) * n // N]):
            log(f"WARNING {os.path.basename(f)}: generated for other requests (prompts / model changed); ignored")
            continue
        shards.append(x)
    cached_cost = cached_answer_costs([s for s in a.splits for _ in range(len(prompts[s]))], info, shards)
    if shards:
        gst = dict(gst, wall_s=gst["wall_s"] + sum(x["wall_s"] for x in shards),
                   gpu_s=gst["gpu_s"] + sum(x["gpu_s"] for x in shards),
                   cpu_core_s=gst["cpu_core_s"] + sum(x["cpu_core_s"] for x in shards),
                   n_workers=max([gst["n_workers"]] + [x["n_workers"] for x in shards]),
                   resumed_gen_s=(gst.get("resumed_gen_s") or 0.0) + sum(x.get("resumed_gen_s") or 0.0 for x in shards),
                   peak_gpu_mem_gb=max([gst.get("peak_gpu_mem_gb") or 0] + [x.get("peak_gpu_mem_gb") or 0 for x in shards]) or None,
                   shards=[{k: x.get(k) for k in ("shard", "nshards", "n", "host", "wall_s", "gpu_s", "n_generated",
                                                   "gpus")} for x in shards])
        log(f"assembled {len(shards)} shard(s): hosts {sorted(set(x['host'] for x in shards))}")
    k, ntot, summary = 0, max(1, len(allp)), {}
    for s in a.splits:
        n = len(prompts[s])
        r = res[k:k + n]
        k += n
        qids = prompts[s]["query_id"].astype(np.int64).tolist()
        out = parse_outputs(qids, r, ex, s)
        _write_parquet(out.assign(prompt_sha256=prompts[s]["prompt_sha256"].to_numpy()),
                       os.path.join(wd, f"responses_{s}.parquet"))
        share = n / ntot
        stages = []
        for st in (s1, s2):
            if st is None:
                continue
            stages.append(dict(name=f"train_{st['stage']}", wall_s=st["train_wall_s"], cpu_core_s=st["train_cpu_core_s"],
                               gpu_s=st["train_gpu_s"], peak_mem_gb=0.0, one_time=True,
                               n_gpus=max(x.get("n_gpus", 1) for x in st["sessions"]),
                               peak_gpu_mem_gb=st.get("peak_gpu_mem_gb"),
                               notes=f"{st['stage']}: {st['global_step']} optimizer steps, {len(st['sessions'])} "
                                     f"session(s), host(s) {sorted(set(x['host'] for x in st['sessions']))}"))
        stages.append(dict(name="infer", wall_s=gst["wall_s"] * share, cpu_core_s=gst["cpu_core_s"] * share,
                           gpu_s=gst["gpu_s"] * share, peak_mem_gb=0.0, one_time=False, n_items=n,
                           n_gpus=gst["n_workers"] if a.backend == "hf" else 0,
                           peak_gpu_mem_gb=gst.get("peak_gpu_mem_gb"),
                           notes=f"HF generate, batch 1, {gst['n_workers']} GPU process(es) incl. model load; one pass "
                                 f"over {'+'.join(a.splits)}, attributed by prompt count"))
        if gst.get("resumed_gen_s"):
            stages.append(dict(name="infer_resumed", wall_s=gst["resumed_gen_s"] * share, cpu_core_s=0.0,
                               gpu_s=gst["resumed_gen_s"] * share, peak_mem_gb=0.0, one_time=False, n_items=n,
                               notes="answers an interrupted earlier attempt left in the generation files (kept by "
                                     "infer.py on restart): their stored generation time (gen_s, 1 GPU, no model "
                                     "load), attributed by prompt count"))
        if cached_cost.get(s) is not None:
            stages.append(cached_cost[s])
        common = dict(evidence_policy=a.policy, cfg=a.cfg, seed=SEED, exploratory=True, host=host(),
                      backend=a.backend, execution=execution,
                      colocated=False,
                      model=BASE_REPO, base_revision=BASE_REVISION, upstream_commit=PR.UPSTREAM_COMMIT,
                      adapter=ADAPTER_VERSION, src=os.path.basename(src), stage1=s1 and s1["out"],
                      stage2=s2 and s2["out"], encoder_sha256=s1 and s1.get("encoder_sha256"),
                      target_time_in_prompt=PR.CFGS[a.cfg]["target_time"], generation=dict(GEN, **{
                          k_: v for k_, v in gst.items() if k_ != "load_s"}),
                      readme_args=README_ARGS, candidates_file=candidates_file_decl(ex, s),
                      sft=dict(n_samples_stage1=pmeta["files"]["stage1_train.json"]["n"],
                               n_samples_stage2=pmeta["files"]["stage2_train.json"]["n"], epochs=EPOCHS,
                               readme_epochs=README_EPOCHS,
                               n_gpus_train=max([x.get("n_gpus", 1) for st in (s1, s2) if st for x in st["sessions"]] or [1]),
                               global_steps={st["stage"]: st["global_step"] for st in (s1, s2) if st}),
                      output="[model answer] + policy candidates (dedup, top 50)",
                      export_manifest_files=ex.manifest.get("files"),
                      tokens=dict(prompt_total=int(out["n_prompt_tokens"].sum()),
                                  prompt_mean=float(out["n_prompt_tokens"].mean())),
                      parse_rates=dict(valid=float(out["valid"].mean()), skipped=float(out["skipped"].mean())),
                      fill_share=float(1 - out["valid"].mean()))
        p = write_outputs(a, ex, s, out, pmeta, stages, common, out_base)
        summary[s] = dict(n=n, valid=float(out["valid"].mean()), path=p)
        log(f"{s}: {n} queries, valid {out['valid'].mean():.4f} -> {p}")
    _write_json(dict(summary=summary, gen=gst, finished=_now(), host=host()), os.path.join(wd, "run_summary.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
