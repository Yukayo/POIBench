#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import resource
import shutil
import socket
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd

from poibench.cost import CostRecorder, cached_answers_stage
from poibench.exports import Export, export_dir
from methods.llm4poi import prompts as PR

ADAPTER_VERSION = "l1"
SEED = 0
TOPK = 50
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
BASE_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
MODEL_TYPE = "qwen2_5"
DEFAULT_BASE = "qwen2.5-7b"
_QWEN_SYSTEM = PR.SYSTEM
BASES = {
    "qwen2.5-7b": (BASE_MODEL, BASE_REVISION, "qwen2_5", _QWEN_SYSTEM),
    "qwen2.5-1.5b": ("Qwen/Qwen2.5-1.5B-Instruct", None, "qwen2_5", _QWEN_SYSTEM),
    "qwen2.5-3b": ("Qwen/Qwen2.5-3B-Instruct", None, "qwen2_5", _QWEN_SYSTEM),
    "llama3.1-8b": ("meta-llama/Llama-3.1-8B-Instruct", None, "llama3_1", None),
}
BASE_KEY = DEFAULT_BASE
_PR_MESSAGES = PR.messages


def _messages_no_system(prompt: str, answer: str | None = None) -> list:
    return [dict(m, content=m["content"].strip()) for m in _PR_MESSAGES(prompt, answer) if m["role"] != "system"]


def set_base(key: str) -> None:
    global BASE_KEY, BASE_MODEL, BASE_REVISION, MODEL_TYPE
    if key not in BASES:
        raise SystemExit(f"unknown LLM4POI base {key}; known: {sorted(BASES)}")
    BASE_KEY = key
    BASE_MODEL, BASE_REVISION, MODEL_TYPE, system = BASES[key]
    PR.SYSTEM = system
    PR.messages = _PR_MESSAGES if system is not None else _messages_no_system
    os.environ["POIBENCH_LLM4POI_BASE"] = key


set_base(os.environ.get("POIBENCH_LLM4POI_BASE", DEFAULT_BASE))
EFFECTIVE_BATCH = 1024
MICRO_BATCH = int(os.environ.get("POIBENCH_LLM4POI_MICRO_BATCH", "4"))
SFT_ARGS = {
    "train_type": "full", "target_modules": "all-linear", "torch_dtype": "bfloat16", "num_train_epochs": "3",
    "streaming": "false", "learning_rate": "1e-5", "packing": "false", "logging_steps": "1",
    "max_length": "12024", "warmup_ratio": "0.05", "dataloader_num_workers": "8", "dataset_num_proc": "8",
    "save_total_limit": "2", "deepspeed": "zero3", "use_liger_kernel": "true", "attn_impl": "flash_attn",
}
SAVE_STEPS_DEFAULT = 20
SAMPLING = {"temperature": 0.0, "max_tokens": 36}
GEN_CONFIG_KEYS = ("repetition_penalty", "temperature", "top_k", "top_p", "min_p", "max_new_tokens")
GEN_CHUNK = 2048
SMOKE_LONGEST = 64
SMOKE_VAL = 500
CANARY_KEY = ("nyc", "H-100", "default")


def log(*a):
    print(f"[llm4poi {time.strftime('%H:%M:%S')}]", *a, flush=True)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


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


def _sha_ids(ids) -> str:
    return hashlib.sha256(",".join(str(int(x)) for x in ids).encode()).hexdigest()


def method_tag(cfg: str, strict: bool = False) -> str:
    return PR.CFGS[cfg]["method"] + ("" if BASE_KEY == DEFAULT_BASE else f"-{BASE_KEY}") + \
        ("-strictparse" if strict else "")


def visible_gpus() -> list:
    v = os.environ.get("CUDA_VISIBLE_DEVICES")
    if v is None:
        try:
            import torch
            return [str(i) for i in range(torch.cuda.device_count())]
        except Exception:
            return []
    return [x for x in v.split(",") if x.strip() != ""]


def run_dirs(ds: str, policy: str, cfg: str, smoke: bool) -> tuple[str, str]:
    c = cfg if BASE_KEY == DEFAULT_BASE else f"{cfg}@{BASE_KEY}"
    sub = os.path.join("smoke", ds, policy, c) if smoke else os.path.join(ds, policy, c)
    wd = os.path.join(ROOT, "work", "llm4poi", sub)
    ck_root = os.environ.get("POIBENCH_LLM4POI_CKPT_ROOT") or os.path.join(ROOT, "work", "llm4poi", "ckpt")
    return wd, os.path.join(ck_root, sub)


def candidates_file_decl(ex: Export, split: str) -> str:
    keys = [k for k in ex.manifest["source_sha256"] if k.endswith(f"candidates_{split}.parquet")]
    if len(keys) != 1:
        raise RuntimeError(f"cannot identify the candidate file of {split} in the export manifest: {keys}")
    return os.path.join(ex.manifest["source_dir"], keys[0])


def base_model() -> dict:
    for var in ("POIBENCH_MODEL_DIR", "POIBENCH_MODEL_DIR_X"):
        if var == "POIBENCH_MODEL_DIR_X" and BASE_KEY == DEFAULT_BASE:
            continue
        d = os.path.join(os.environ.get(var, ""), BASE_MODEL.split("/")[-1])
        mk = os.path.join(d, ".poibench_complete")
        if os.environ.get(var) and os.path.exists(mk):
            m = _read_json(mk)
            if m.get("repo") not in (None, BASE_MODEL):
                raise RuntimeError(f"{d} holds {m.get('repo')}, not {BASE_MODEL}")
            return dict(path=d, revision=m.get("sha"), local=True,
                        matches_pin=BASE_REVISION is None or m.get("sha") == BASE_REVISION)
    if BASE_REVISION is None:
        raise RuntimeError(f"{BASE_MODEL}: no verified local copy (.poibench_complete) under $POIBENCH_MODEL_DIR[_X]")
    return dict(path=BASE_MODEL, revision=BASE_REVISION, local=False, matches_pin=True)


def tokenizer():
    from transformers import AutoTokenizer
    bm = base_model()
    if bm["local"]:
        return AutoTokenizer.from_pretrained(bm["path"])
    return AutoTokenizer.from_pretrained(BASE_MODEL, revision=BASE_REVISION, local_files_only=True)


def n_tokens(tok, msgs_list: list, add_generation_prompt: bool) -> np.ndarray:
    texts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=add_generation_prompt)
             for m in msgs_list]
    out = []
    for s in range(0, len(texts), 4096):
        out += [len(x) for x in tok(texts[s:s + 4096], add_special_tokens=False)["input_ids"]]
    return np.asarray(out, dtype=np.int64)


def sft_fingerprint(ex: Export, cfg: str, smoke_n: int) -> dict:
    return dict(adapter=ADAPTER_VERSION, export_files=ex.manifest.get("files"), cfg=cfg, smoke_n=int(smoke_n),
                code=PR.code_fingerprint(), history_limit=PR.HISTORY_LIMIT, system=PR.SYSTEM)


def prepare_sft(ex: Export, cfg: str, wd: str, smoke_n: int = 0, with_tokens: bool = True) -> dict:
    path, mpath = os.path.join(wd, "sft_train.jsonl"), os.path.join(wd, "sft_train.json")
    fp = sft_fingerprint(ex, cfg, smoke_n)
    if os.path.exists(path) and os.path.exists(mpath):
        meta = _read_json(mpath)
        if meta.get("fingerprint") == fp and (meta.get("tokens") or not with_tokens):
            log(f"reusing SFT data {os.path.relpath(path, ROOT)} ({meta['n_samples']} samples)")
            return meta
    t0 = time.perf_counter()
    df = PR.train_samples(ex, cfg)
    n_all = len(df)
    tok_stats, ntok = None, None
    if with_tokens:
        tok = tokenizer()
        ntok = n_tokens(tok, [PR.messages(p, a) for p, a in zip(df["prompt"], df["answer"])], False)
        df["n_tokens"] = ntok
    if smoke_n:
        key = df["n_tokens"] if "n_tokens" in df else df["prompt"].str.len()
        longest = set(df.loc[key.sort_values(ascending=False, kind="stable").index[:SMOKE_LONGEST], "query_id"])
        rest = df.loc[~df["query_id"].isin(longest), "query_id"].to_numpy()
        pick = np.random.default_rng(SEED).choice(rest, size=max(0, min(len(rest), smoke_n - len(longest))),
                                                  replace=False)
        df = df[df["query_id"].isin(longest | set(int(x) for x in pick))].reset_index(drop=True)
    os.makedirs(wd, exist_ok=True)
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        for p, a in zip(df["prompt"], df["answer"]):
            f.write(json.dumps({"messages": PR.messages(p, a)}, ensure_ascii=False) + "\n")
    os.replace(path + ".tmp", path)
    if with_tokens:
        n = df["n_tokens"].to_numpy()
        tok_stats = dict(total=int(n.sum()), mean=float(n.mean()), p50=float(np.median(n)),
                         p95=float(np.percentile(n, 95)), max=int(n.max()),
                         n_over_max_length=int((n > int(SFT_ARGS["max_length"])).sum()))
    meta = dict(fingerprint=fp, n_samples=int(len(df)), n_samples_full=int(n_all),
                n_train_queries_policy=int(len(ex.queries("train"))), mean_n_hist=float(df["n_hist"].mean()),
                mean_n_cur=float(df["n_cur"].mean()), max_n_cur=int(df["n_cur"].max()),
                sha256=hashlib.sha256(open(path, "rb").read()).hexdigest(), tokens=tok_stats,
                query_ids_sha256=_sha_ids(df["query_id"]), build_s=round(time.perf_counter() - t0, 1),
                created_utc=_now())
    _write_json(meta, mpath)
    log(f"SFT data: {meta['n_samples']} samples -> {os.path.relpath(path, ROOT)}; tokens {tok_stats}")
    return meta


def prepare_prompts(ex: Export, ds: str, split: str, cfg: str, qids, wd: str) -> tuple[pd.DataFrame, dict]:
    path, mpath = os.path.join(wd, f"prompts_{split}.parquet"), os.path.join(wd, f"prompts_{split}.json")
    fp = dict(adapter=ADAPTER_VERSION, export_files=ex.manifest.get("files"), cfg=cfg, code=PR.code_fingerprint(),
              query_ids_sha256=_sha_ids(qids), n_queries=len(qids))
    if os.path.exists(path) and os.path.exists(mpath):
        meta = _read_json(mpath)
        if meta.get("fingerprint") == fp:
            return pd.read_parquet(path), meta
    t0 = time.perf_counter()
    c0 = resource.getrusage(resource.RUSAGE_SELF)
    tt = PR.load_target_time(ds, split) if PR.CFGS[cfg]["target_time"] else None
    df = PR.query_prompts(ex, split, qids, cfg, tt)
    c1 = resource.getrusage(resource.RUSAGE_SELF)
    wall = time.perf_counter() - t0
    stage = dict(wall_s=wall, cpu_core_s=(c1.ru_utime + c1.ru_stime) - (c0.ru_utime + c0.ru_stime), gpu_s=0.0,
                 peak_mem_gb=0.0, one_time=False, n_items=len(df), notes="prompt construction on CPU (per query)")
    meta = dict(fingerprint=fp, stage=stage, stats=dict(n=int(len(df)), mean_n_hist=float(df["n_hist"].mean()),
                                                        mean_n_cur=float(df["n_cur"].mean()),
                                                        mean_chars=float(df["prompt"].str.len().mean()),
                                                        max_chars=int(df["prompt"].str.len().max())))
    _write_parquet(df, path)
    _write_json(meta, mpath)
    log(f"{split}: {len(df)} prompts in {wall:.1f} s -> {os.path.relpath(path, ROOT)}")
    return df, meta


def _complete_ckpts(sft_dir: str, with_states: bool = True) -> list:
    out = []
    if not os.path.isdir(sft_dir):
        return out
    for n in os.listdir(sft_dir):
        m = re.fullmatch(r"checkpoint-(\d+)", n)
        if not m:
            continue
        p = os.path.join(sft_dir, n)
        lt = os.path.join(p, "latest")
        if not (os.path.exists(os.path.join(p, "trainer_state.json")) and os.path.exists(lt)
                and any(f.endswith(".safetensors") for f in os.listdir(p))):
            continue
        with open(lt) as f:
            tag = f.read().strip()
        if with_states and not os.path.isdir(os.path.join(p, tag)):
            continue
        out.append((int(m.group(1)), p))
    return sorted(out)


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def check_template(model_path: str, sample_msgs: list) -> dict:
    from swift.llm import get_model_tokenizer, get_template
    if BASE_KEY != DEFAULT_BASE:
        sys.path.insert(0, os.path.join(HERE, "swift_entry"))
        import sft_fixed_date
        sys.path.pop(0)
        sft_fixed_date.patch_templates()
    kw = {} if BASE_KEY == DEFAULT_BASE else dict(model_type=MODEL_TYPE)
    _, tok = get_model_tokenizer(model_path, load_model=False, **kw)
    tpl = get_template(tok.model_meta.template, tok, max_length=int(SFT_ARGS["max_length"]))
    tpl.set_mode("train")
    enc = tpl.encode({"messages": sample_msgs})
    ids, lab = list(enc["input_ids"]), list(enc["labels"])
    k = next(i for i, x in enumerate(lab) if x != -100)
    hf = list(tok.apply_chat_template(sample_msgs[:-1], tokenize=True, add_generation_prompt=True))
    res = dict(template=tok.model_meta.template, prefix_equal=ids[:k] == hf,
               label_text=tok.decode([x for x in lab if x != -100]))
    if not res["prefix_equal"]:
        raise RuntimeError(f"swift training template != HF chat template used by vLLM: {res}")
    return res


def swift_cmd(data: str, out_dir: str, n_gpus: int, epochs: str, save_steps: int, resume: str | None) -> list:
    bm = base_model()
    acc = EFFECTIVE_BATCH // (MICRO_BATCH * n_gpus)
    if acc * MICRO_BATCH * n_gpus != EFFECTIVE_BATCH:
        raise ValueError(f"effective batch {EFFECTIVE_BATCH} not divisible by {MICRO_BATCH} x {n_gpus} GPUs")
    args = dict(SFT_ARGS, num_train_epochs=epochs, model=bm["path"], model_type=MODEL_TYPE, dataset=data,
                per_device_train_batch_size=str(MICRO_BATCH), gradient_accumulation_steps=str(acc),
                save_steps=str(save_steps), save_only_model="false", output_dir=out_dir, seed=str(SEED),
                data_seed=str(SEED), add_version="false", report_to="tensorboard")
    if resume:
        args["resume_from_checkpoint"] = resume
    entry = ["-m", "swift.cli.sft"] if BASE_KEY == DEFAULT_BASE else \
        [os.path.join(HERE, "swift_entry", "sft_fixed_date.py")]
    cmd = [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node", str(n_gpus), "--master_port",
           str(_free_port())] + entry
    for k, v in args.items():
        cmd += [f"--{k}", str(v)]
    return cmd


def run_logged(cmd: list, log_path: str, env: dict | None = None) -> tuple[int, float, float]:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    env = dict(env if env is not None else os.environ, MPLBACKEND="Agg")
    c0 = resource.getrusage(resource.RUSAGE_CHILDREN)
    t0 = time.perf_counter()
    with open(log_path, "a", buffering=1) as lf:
        lf.write(f"\n# {_now()} {' '.join(cmd)}\n")
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, text=True, bufsize=1,
                             cwd=os.path.dirname(log_path))
        for line in p.stdout:
            lf.write(line)
            sys.stdout.write(line)
        rc = p.wait()
    c1 = resource.getrusage(resource.RUSAGE_CHILDREN)
    return rc, time.perf_counter() - t0, (c1.ru_utime + c1.ru_stime) - (c0.ru_utime + c0.ru_stime)


def train(wd: str, ckdir: str, sft_meta: dict, epochs: str = "3", save_steps: int = SAVE_STEPS_DEFAULT,
          drop_optimizer_states: bool = True) -> dict:
    sft_dir = os.path.join(ckdir, "sft")
    done_p = os.path.join(ckdir, "DONE.json")
    st_p = os.path.join(wd, "train_state.json")
    key = dict(sft_sha256=sft_meta["sha256"], epochs=epochs, base=base_model()["revision"],
               effective_batch=EFFECTIVE_BATCH, sft_args=SFT_ARGS, adapter=ADAPTER_VERSION)
    if os.path.exists(done_p):
        d = _read_json(done_p)
        if d.get("key") == key:
            log(f"train: DONE ({d['final_ckpt']})")
            return d
        raise RuntimeError(f"{done_p} belongs to another training configuration; move it away by hand")
    st = _read_json(st_p) if os.path.exists(st_p) else dict(key=key, sessions=[])
    if st.get("key") != key:
        raise RuntimeError(f"{st_p} belongs to another training configuration")
    gpus = visible_gpus()
    if not gpus:
        raise RuntimeError("training needs GPUs (CUDA_VISIBLE_DEVICES is empty)")
    if "template_check" not in st:
        with open(os.path.join(wd, "sft_train.jsonl")) as f:
            st["template_check"] = check_template(base_model()["path"], json.loads(f.readline())["messages"])
        log(f"template check: {st['template_check']}")
    ck = _complete_ckpts(sft_dir)
    resume = ck[-1][1] if ck else None
    cmd = swift_cmd(os.path.join(wd, "sft_train.jsonl"), sft_dir, len(gpus), epochs, save_steps, resume)
    log(f"train: {len(gpus)} GPUs, resume from {resume}; {' '.join(cmd)}")
    sess = dict(start_utc=_now(), resume_from=resume, n_gpus=len(gpus), host=socket.gethostname(),
                attempt=os.environ.get("POIBENCH_ATTEMPT"))
    rc, wall, cpu = run_logged(cmd, os.path.join(wd, "train.log"))
    sess.update(end_utc=_now(), rc=rc, wall_s=wall, cpu_core_s=cpu, gpu_s=wall * len(gpus))
    st["sessions"].append(sess)
    _write_json(st, st_p)
    if rc != 0:
        raise RuntimeError(f"swift sft failed rc={rc} (log {os.path.join(wd, 'train.log')})")
    ck = _complete_ckpts(sft_dir)
    if not ck:
        raise RuntimeError("swift finished without a checkpoint")
    final = ck[-1][1]
    ts = _read_json(os.path.join(final, "trainer_state.json"))
    if ts.get("global_step") != ts.get("max_steps"):
        raise RuntimeError(f"last checkpoint {final} is step {ts.get('global_step')} of {ts.get('max_steps')}")
    logs = [r for r in ts.get("log_history", []) if "loss" in r]
    mem = []
    lj = os.path.join(sft_dir, "logging.jsonl")
    if os.path.exists(lj):
        with open(lj) as f:
            for x in f:
                try:
                    v = json.loads(x).get("memory(GiB)")
                except ValueError:
                    continue
                if v is not None:
                    mem.append(float(v))
    d = dict(key=key, final_ckpt=final, global_step=ts["global_step"], max_steps=ts["max_steps"],
             sessions=st["sessions"], template_check=st["template_check"], finished_utc=_now(),
             train_wall_s=float(sum(s["wall_s"] for s in st["sessions"])),
             train_gpu_s=float(sum(s["gpu_s"] for s in st["sessions"])),
             train_cpu_core_s=float(sum(s["cpu_core_s"] for s in st["sessions"])),
             n_logged_steps=len(logs), model_fingerprint=model_fingerprint(final),
             peak_gpu_mem_gib_swift=max(mem) if mem else None)
    if drop_optimizer_states:
        for step, p in ck:
            for n in sorted(os.listdir(p)):
                if re.fullmatch(r"global_step\d+", n):
                    shutil.rmtree(os.path.join(p, n))
            if p != final:
                shutil.rmtree(p)
        d["optimizer_states_dropped"] = True
    _write_json(d, done_p)
    _write_json(d, os.path.join(wd, "DONE_train.json"))
    log(f"train: finished at step {d['global_step']}, final {final}, {d['train_wall_s'] / 3600:.2f} h wall")
    return d


def model_fingerprint(model_dir: str) -> str:
    h = hashlib.sha256()
    for n in sorted(os.listdir(model_dir)):
        p = os.path.join(model_dir, n)
        if n == "config.json":
            h.update(open(p, "rb").read())
        elif n.endswith(".safetensors"):
            sz = os.path.getsize(p)
            with open(p, "rb") as f:
                f.seek(max(0, sz // 2 - (1 << 19)))
                h.update(n.encode() + str(sz).encode() + f.read(1 << 20))
    return h.hexdigest()


def effective_sampling(model_dir: str | None) -> dict:
    d = {}
    p = os.path.join(model_dir, "generation_config.json") if model_dir else None
    if p and os.path.exists(p):
        g = _read_json(p)
        for k in GEN_CONFIG_KEYS:
            if g.get(k) is not None:
                d["max_tokens" if k == "max_new_tokens" else k] = g[k]
    d.update(SAMPLING)
    return d


def cache_key(model_fp: str, prompt: str, sampling: dict | None = None) -> str:
    s = json.dumps({"model": model_fp, "sampling": sampling or SAMPLING, "messages": PR.messages(prompt)},
                   sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(s.encode()).hexdigest()


class ResponseCache:

    def __init__(self, model_fp: str, cache_dir: str | None = None):
        base = os.environ.get("LLM4POI_CACHE_DIR") or os.path.join(ROOT, "work", "llm4poi", "llm_cache")
        self.dir = cache_dir or os.path.join(base, model_fp[:24])
        os.makedirs(self.dir, exist_ok=True)
        self.mem = {}
        self.reload()

    def reload(self):
        for f in sorted(os.listdir(self.dir)):
            if f.startswith("shard_") and f.endswith(".parquet"):
                for r in pd.read_parquet(os.path.join(self.dir, f)).to_dict("records"):
                    self.mem.setdefault(r["key"], r)

    def put(self, recs: list) -> None:
        name = f"shard_{datetime.now(timezone.utc):%Y%m%dT%H%M%S}_{os.getpid()}_{uuid.uuid4().hex[:12]}.parquet"
        p = os.path.join(self.dir, name)
        pd.DataFrame(recs).to_parquet(p + ".tmp", index=False)
        os.replace(p + ".tmp", p)
        for r in recs:
            self.mem.setdefault(r["key"], r)


def mock_text(prompt: str) -> str:
    h = int(hashlib.sha256(prompt.encode()).hexdigest(), 16)
    ids = re.findall(r"\| (\d+) \|", prompt) or ["0"]
    pid = ids[h % len(ids)]
    return [pid, f"{pid}<|endoftext|>", f"POI {pid}", "unknown", f" {pid}\n"][(h >> 64) % 5]


def gen_worker(a) -> int:
    df = pd.read_parquet(a.inp)
    cache = ResponseCache(a.model_fp, a.cache_dir)
    df = df[~df["key"].isin(set(cache.mem))]
    if not len(df):
        return 0
    from vllm import LLM, SamplingParams
    t0 = time.perf_counter()
    tok = a.model if os.path.exists(os.path.join(a.model, "tokenizer_config.json")) else base_model()["path"]
    llm = LLM(model=a.model, tokenizer=tok, seed=SEED, dtype="bfloat16")
    want = json.loads(a.sampling)
    diff = dict(llm.llm_engine.model_config.get_diff_sampling_param())
    eff = dict(diff, **SAMPLING)
    if {k: float(v) for k, v in eff.items()} != {k: float(v) for k, v in want.items()}:
        raise RuntimeError(f"vLLM generation defaults {diff} + {SAMPLING} != expected {want}")
    sp = SamplingParams(**eff)
    print(f"[gen] sampling {sp}", flush=True)
    print(f"[gen {os.environ.get('CUDA_VISIBLE_DEVICES')}] engine ready in {time.perf_counter() - t0:.1f} s; "
          f"{len(df)} prompts", flush=True)
    keys, prompts = df["key"].tolist(), df["prompt"].tolist()
    for s in range(0, len(keys), GEN_CHUNK):
        t1 = time.perf_counter()
        outs = llm.chat([PR.messages(p) for p in prompts[s:s + GEN_CHUNK]], sampling_params=sp, use_tqdm=False)
        dt = time.perf_counter() - t1
        recs = []
        for k, o in zip(keys[s:s + GEN_CHUNK], outs):
            c = o.outputs[0]
            recs.append(dict(key=k, text=c.text, n_prompt_tokens=len(o.prompt_token_ids or []),
                             n_output_tokens=len(c.token_ids), finish_reason=str(c.finish_reason),
                             gen_s=dt / len(outs), created_utc=_now(), host=socket.gethostname()))
        cache.put(recs)
        print(f"[gen {os.environ.get('CUDA_VISIBLE_DEVICES')}] {s + len(recs)}/{len(keys)} in {dt:.1f} s "
              f"({sum(r['n_prompt_tokens'] for r in recs) / dt:.0f} prompt tok/s)", flush=True)
    return 0


def generate(prompts: list, model_dir: str | None, model_fp: str, backend: str, wd: str,
             info: dict | None = None) -> tuple[list, dict]:
    cache = ResponseCache(model_fp)
    sampling = effective_sampling(model_dir) if backend == "vllm" else dict(SAMPLING)
    keys = [cache_key(model_fp, p, sampling) for p in prompts]
    first = {}
    for k, p in zip(keys, prompts):
        first.setdefault(k, p)
    todo = [k for k in first if k not in cache.mem]
    if info is not None:
        info.update(keys=keys, cached_keys={k for k in first if k in cache.mem}, cache=cache)
    stats = dict(n_prompts=len(prompts), n_unique=len(first), n_cache_hits=len(first) - len(todo), n_generated=0,
                 gen_wall_s=0.0, n_workers=0, sampling=sampling)
    t0 = time.perf_counter()
    if todo and backend == "mock":
        cache.put([dict(key=k, text=mock_text(first[k]), n_prompt_tokens=len(first[k]) // 4, n_output_tokens=2,
                        finish_reason="stop", gen_s=0.0, created_utc=_now(), host=socket.gethostname())
                   for k in todo])
    elif todo:
        gpus = visible_gpus() or ["0"]
        stats["n_workers"] = len(gpus)
        tmp = os.path.join(wd, "gen_tmp", uuid.uuid4().hex[:8])
        os.makedirs(tmp, exist_ok=True)
        procs = []
        for i, g in enumerate(gpus):
            part = todo[i::len(gpus)]
            if not part:
                continue
            inp = os.path.join(tmp, f"todo_{i}.parquet")
            pd.DataFrame({"key": part, "prompt": [first[k] for k in part]}).to_parquet(inp, index=False)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=g, MPLBACKEND="Agg")
            cmd = [sys.executable, "-I", os.path.abspath(__file__), "_gen", "--ds", "-", "--policy", "-",
                   "--model", model_dir, "--model-fp", model_fp, "--cache-dir", cache.dir, "--inp", inp,
                   "--sampling", json.dumps(sampling)]
            lf = open(os.path.join(wd, f"gen_worker_{i}.log"), "a")
            procs.append((subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT, cwd=tmp), lf, i))
        bad = []
        for p, lf, i in procs:
            rc = p.wait()
            lf.close()
            if rc != 0:
                bad.append((i, rc))
        if bad:
            raise RuntimeError(f"generation workers failed {bad} (logs {wd}/gen_worker_*.log)")
        cache.reload()
        shutil.rmtree(tmp, ignore_errors=True)
    stats["gen_wall_s"] = time.perf_counter() - t0
    stats["n_generated"] = sum(1 for k in todo if k in cache.mem)
    missing = [k for k in first if k not in cache.mem]
    if missing:
        raise RuntimeError(f"{len(missing)} prompts without a response")
    return [cache.mem[k] for k in keys], stats


def parse_strict(text: str):
    s = (text or "").strip().split("<")[0]
    return int(s) if s.isdigit() and s == str(int(s)) else None


_INT = re.compile(r"\d+")


def parse_lenient(text: str):
    m = _INT.search(text or "")
    return int(m.group(0)) if m else None


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
        al, ast_ = parse_lenient(r["text"]), parse_strict(r["text"])
        ll, okl = ranked_list(al, cmap[int(qid)], V)
        ls, oks = ranked_list(ast_, cmap[int(qid)], V)
        rows.append(dict(query_id=int(qid), ranked_lenient=ll, ranked_strict=ls,
                         answer_lenient=int(al) if okl else -1, answer_strict=int(ast_) if oks else -1,
                         valid_lenient=okl, valid_strict=oks, text=r["text"],
                         n_prompt_tokens=int(r["n_prompt_tokens"]), n_output_tokens=int(r["n_output_tokens"]),
                         finish_reason=str(r["finish_reason"])))
    return pd.DataFrame(rows)


def default_canary_ckpt() -> str:
    _, ck = run_dirs(*CANARY_KEY, smoke=True)
    return os.path.join(ck, "DONE.json")


def predict(export_dir, split, query_ids, work_dir=None):
    backend = os.environ.get("LLM4POI_BACKEND", "vllm")
    cfg = os.environ.get("LLM4POI_CFG", "default")
    if cfg not in PR.CFGS:
        raise ValueError(f"LLM4POI_CFG={cfg!r} is not an LLM4POI configuration ({sorted(PR.CFGS)})")
    ex = Export(export_dir)
    ds = ex.manifest.get("ds")
    qids = [int(x) for x in query_ids]
    tt = PR.CFGS[cfg]["target_time"]
    df = PR.query_prompts(ex, split, qids, cfg, PR.load_target_time(ds, split) if tt else None)
    if backend == "mock":
        model_dir, mfp = None, "mock-v1"
    else:
        if cfg != CANARY_KEY[2] and not os.environ.get("LLM4POI_CKPT"):
            raise ValueError(f"LLM4POI_CFG={cfg}: the fixed canary checkpoint is trained with cfg {CANARY_KEY[2]}; set "
                             f"LLM4POI_CKPT to a {cfg} checkpoint (or use LLM4POI_BACKEND=mock)")
        model_dir = os.environ.get("LLM4POI_CKPT") or _read_json(default_canary_ckpt())["final_ckpt"]
        mfp = model_fingerprint(model_dir)
    wd = work_dir or os.path.join(ROOT, "work", "llm4poi", "predict_tmp")
    os.makedirs(wd, exist_ok=True)
    if work_dir:
        _write_parquet(df, os.path.join(work_dir, "prompts.parquet"))
    res, _ = generate(df["prompt"].tolist(), model_dir, mfp, backend, wd)
    out = parse_outputs(qids, res, ex, split)
    if work_dir:
        _write_parquet(out, os.path.join(work_dir, "responses.parquet"))
    log(f"predict {ds}/{split}: {len(out)} queries, lenient valid {out['valid_lenient'].mean():.3f}")
    ret = pd.DataFrame({"query_id": out["query_id"], "ranked_venue_ids": out["ranked_lenient"]})
    ret.attrs["adapter_config"] = dict(cfg=cfg, backend=backend, target_time_in_prompt=bool(tt), model_fingerprint=mfp)
    return ret


def write_outputs(a, ex: Export, split: str, out: pd.DataFrame, pmeta: dict, stages: list, common: dict,
                  out_base: str) -> list:
    paths = []
    for strict in (False, True):
        t = method_tag(a.cfg, strict)
        col = "strict" if strict else "lenient"
        d = os.path.join(out_base, f"{t}@{a.policy}", a.ds)
        preds = pd.DataFrame({"query_id": out["query_id"], "ranked_venue_ids": out[f"ranked_{col}"],
                              "answer_venue_id": out[f"answer_{col}"]})
        cr = CostRecorder(method=f"{t}@{a.policy}", ds=a.ds, split=split,
                          extra=dict(common, parse=col, primary=not strict,
                                     shares_generations_with=(f"{method_tag(a.cfg)}@{a.policy}" if strict else None)))
        cr.add_stage(name="prepare_prompts", **{k: v for k, v in pmeta["stage"].items()})
        for s in stages:
            cr.add_stage(**s)
        _write_parquet(preds, os.path.join(d, f"{split}.parquet"))
        cr.save(os.path.join(d, f"{split}.cost.json"), n_queries=len(preds),
                keep_measured=("method", "ds", "split", "cfg", "n_queries", "model_fingerprint", "sampling",
                               "export_manifest_files"))
        paths.append(os.path.relpath(os.path.join(d, f"{split}.parquet"), ROOT))
    return paths


def do_predict(a, ex: Export, wd: str, done: dict | None, prompts: dict, pmetas: dict, out_base: str) -> dict:
    if a.backend == "mock":
        model_dir, mfp = None, "mock-v1"
    else:
        model_dir = done["final_ckpt"]
        mfp = done.get("model_fingerprint") or model_fingerprint(model_dir)
    allp = [p for s in a.splits for p in prompts[s]["prompt"].tolist()]
    c0 = resource.getrusage(resource.RUSAGE_CHILDREN)
    info = {}
    res, gst = generate(allp, model_dir, mfp, a.backend, wd, info=info)
    c1 = resource.getrusage(resource.RUSAGE_CHILDREN)
    gen_cpu = (c1.ru_utime + c1.ru_stime) - (c0.ru_utime + c0.ru_stime)
    execution = dict(wall_s=gst["gen_wall_s"], cpu_core_s=gen_cpu, n_generated=gst["n_generated"],
                     n_cache_hits=gst["n_cache_hits"], n_workers=gst["n_workers"], backend=a.backend,
                     host=socket.gethostname(), finished=_now())
    seen_cached = set()
    summary, k = {}, 0
    ntot = max(1, len(allp))
    for s in a.splits:
        n = len(prompts[s])
        r = res[k:k + n]
        k += n
        qids = prompts[s]["query_id"].astype(np.int64).tolist()
        out = parse_outputs(qids, r, ex, s)
        _write_parquet(out.assign(prompt_sha256=prompts[s]["prompt_sha256"].to_numpy()),
                       os.path.join(wd, f"responses_{s}.parquet"))
        share = n / ntot
        n_gpu = max(1, gst["n_workers"]) if a.backend == "vllm" else 0
        stages = []
        if done is not None and a.backend == "vllm":
            stages.append(dict(name="train", wall_s=done["train_wall_s"], cpu_core_s=done["train_cpu_core_s"],
                               gpu_s=done["train_gpu_s"], peak_mem_gb=0.0, one_time=True,
                               n_gpus=done["sessions"][-1]["n_gpus"],
                               peak_gpu_mem_gb=(done.get("peak_gpu_mem_gib_swift") or 0) * 1.073741824,
                               notes=f"swift full fine-tune, {done['global_step']} optimizer steps, "
                                     f"{len(done['sessions'])} session(s) incl. checkpoint saves/loads"))
        stages.append(dict(name="infer", wall_s=gst["gen_wall_s"] * share, cpu_core_s=gen_cpu * share,
                           gpu_s=gst["gen_wall_s"] * share * n_gpu, peak_mem_gb=0.0, one_time=False, n_items=n,
                           n_gpus=n_gpu,
                           notes=f"vLLM data-parallel over {gst['n_workers']} GPU(s) incl. engine start; one pass "
                                 f"over {'+'.join(a.splits)}, attributed by prompt count; generated "
                                 f"{gst['n_generated']}, cache hits {gst['n_cache_hits']}"))
        ks = [x for x in info["keys"][k - n:k] if x in info["cached_keys"] and x not in seen_cached]
        ks = list(dict.fromkeys(ks))
        seen_cached.update(ks)
        cst = cached_answers_stage([info["cache"].mem[x] for x in ks],
                                   notes="answers taken from the response cache (earlier attempt or another job with "
                                         "the same prompt): their stored generation time (gen_s per answer = its "
                                         "share of a timed vLLM chat call on one GPU; engine start not included)")
        if cst is not None and a.backend == "vllm":
            stages.append(cst)
        cached = [x for x in r if float(x.get("gen_s") or 0) > 0]
        common = dict(evidence_policy=a.policy, cfg=a.cfg, seed=SEED, backend=a.backend, execution=execution,
                      model=BASE_MODEL, base_key=BASE_KEY,
                      model_type=MODEL_TYPE,
                      base_revision=(base_model()["revision"] if a.backend == "vllm" or BASE_REVISION
                                     else None), base_revision_pinned=BASE_REVISION,
                      checkpoint=model_dir, model_fingerprint=mfp, sampling=gst["sampling"], system=PR.SYSTEM,
                      target_time_in_prompt=PR.CFGS[a.cfg]["target_time"], history_limit=PR.HISTORY_LIMIT,
                      candidates_file=candidates_file_decl(ex, s),
                      output="[model answer] + policy candidates (dedup, top 50)",
                      upstream_commit=PR.UPSTREAM_COMMIT, adapter=ADAPTER_VERSION,
                      export_manifest_files=ex.manifest.get("files"), prompt_stats=pmetas[s]["stats"],
                      sft=dict(args=SFT_ARGS, effective_batch=EFFECTIVE_BATCH, micro_batch=MICRO_BATCH,
                               global_step=(done or {}).get("global_step")),
                      generation=dict(gst, finish_reasons=out["finish_reason"].value_counts().to_dict(),
                                      gen_s_sum_of_answers=float(sum(float(x.get("gen_s") or 0) for x in cached))),
                      tokens=dict(prompt_total=int(out["n_prompt_tokens"].sum()),
                                  output_total=int(out["n_output_tokens"].sum()),
                                  prompt_mean=float(out["n_prompt_tokens"].mean())),
                      parse_rates=dict(lenient_valid=float(out["valid_lenient"].mean()),
                                       strict_valid=float(out["valid_strict"].mean())),
                      fill_share=dict(lenient=float(1 - out["valid_lenient"].mean()),
                                      strict=float(1 - out["valid_strict"].mean())))
        paths = write_outputs(a, ex, s, out, pmetas[s], stages, common, out_base)
        summary[s] = dict(n=n, parse_rates=common["parse_rates"], tokens=common["tokens"], paths=paths)
        log(f"{s}: {n} queries; valid lenient {common['parse_rates']['lenient_valid']:.3f} strict "
            f"{common['parse_rates']['strict_valid']:.3f} -> {paths}")
    return dict(summary=summary, generation=gst)


def smoke(a, ex: Export) -> int:
    wd, ckdir = run_dirs(a.ds, a.policy, a.cfg, smoke=True)
    rep_p = os.path.join(ROOT, "logs", "llm4poi", f"smoke_{a.ds}_{a.policy}.json")
    rep = dict(created_utc=_now(), ds=a.ds, policy=a.policy, cfg=a.cfg, host=socket.gethostname(),
               gpus=visible_gpus(), smoke_n=a.smoke)
    meta = prepare_sft(ex, a.cfg, wd, smoke_n=a.smoke)
    rep["sft_data"] = meta
    if a.canary_prep and os.path.exists(os.path.join(ckdir, "DONE.json")):
        log(f"canary checkpoint exists: {_read_json(os.path.join(ckdir, 'DONE.json'))['final_ckpt']}")
        return 0
    done = train(wd, ckdir, meta, epochs="1", save_steps=1, drop_optimizer_states=False)
    s1 = done["sessions"][-1]
    sft_dir = os.path.join(ckdir, "sft")
    rep["train_session_1"] = s1
    ts = _read_json(os.path.join(done["final_ckpt"], "trainer_state.json"))
    rep["log_history"] = ts.get("log_history")
    ck = _complete_ckpts(sft_dir)
    if a.canary_prep:
        rep["resume_test"] = "skipped (canary-prep)"
    elif len(ck) >= 2 and not os.path.exists(os.path.join(wd, "resume_test.json")):
        cmd = swift_cmd(os.path.join(wd, "sft_train.jsonl"), sft_dir + "_resume", len(visible_gpus()), "1", 1,
                        ck[0][1])
        rc, wall, cpu = run_logged(cmd, os.path.join(wd, "train_resume.log"))
        r2 = dict(rc=rc, wall_s=wall, resume_from=ck[0][1])
        rck = _complete_ckpts(sft_dir + "_resume")
        if rc == 0 and rck:
            lh = _read_json(os.path.join(rck[-1][1], "trainer_state.json")).get("log_history", [])
            r2["log_history"] = lh
            l_orig = [x.get("loss") for x in ts.get("log_history", []) if x.get("step") == ck[-1][0] and "loss" in x]
            l_res = [x.get("loss") for x in lh if x.get("step") == ck[-1][0] and "loss" in x]
            r2.update(loss_last_step_original=l_orig, loss_last_step_resumed=l_res)
            shutil.rmtree(sft_dir + "_resume")
        _write_json(r2, os.path.join(wd, "resume_test.json"))
        rep["resume_test"] = r2
    else:
        rep["resume_test"] = _read_json(os.path.join(wd, "resume_test.json")) if os.path.exists(
            os.path.join(wd, "resume_test.json")) else "not run (< 2 checkpoints)"
    sizes = {}
    for step, p in _complete_ckpts(sft_dir):
        tot = 0
        for dp, _, fs in os.walk(p):
            tot += sum(os.path.getsize(os.path.join(dp, f)) for f in fs)
        sizes[os.path.basename(p)] = tot / 1e9
    rep["checkpoint_gb"] = sizes
    for step, p in _complete_ckpts(sft_dir):
        for n in sorted(os.listdir(p)):
            if re.fullmatch(r"global_step\d+", n):
                shutil.rmtree(os.path.join(p, n))
        if p != done["final_ckpt"]:
            shutil.rmtree(p)
    lj = os.path.join(sft_dir, "logging.jsonl")
    if os.path.exists(lj):
        with open(lj) as f:
            rep["swift_logging"] = [json.loads(x) for x in f if x.strip()]
    tot_tok = meta["tokens"]["total"]
    rep["sft_tokens_total"] = tot_tok
    rep["sft_tokens_per_s_session_wall"] = tot_tok / s1["wall_s"]
    if a.canary_prep:
        _write_json(rep, rep_p)
        return 0
    q = ex.queries("val")["query_id"].to_numpy(np.int64)
    qids = [int(x) for x in np.sort(np.random.default_rng(SEED).choice(q, size=min(SMOKE_VAL, len(q)), replace=False))]
    df, pm = prepare_prompts(ex, a.ds, "val", a.cfg, qids, wd)
    res, gst = generate(df["prompt"].tolist(), done["final_ckpt"], done["model_fingerprint"], "vllm", wd)
    out = parse_outputs(qids, res, ex, "val")
    _write_parquet(out, os.path.join(wd, "responses_val.parquet"))
    rep["generation"] = dict(gst, prompt_tokens=int(out["n_prompt_tokens"].sum()),
                             output_tokens=int(out["n_output_tokens"].sum()),
                             finish_reasons=out["finish_reason"].value_counts().to_dict(),
                             lenient_valid=float(out["valid_lenient"].mean()),
                             strict_valid=float(out["valid_strict"].mean()),
                             example_texts=out["text"].head(10).tolist())
    rep["gen_prompt_tokens_per_s"] = rep["generation"]["prompt_tokens"] / max(1e-9, gst["gen_wall_s"])
    _write_json(rep, rep_p)
    log(f"smoke report -> {os.path.relpath(rep_p, ROOT)}")
    return 0


def token_report(a, ex: Export) -> dict:
    tok = tokenizer()
    out = dict(ds=a.ds, policy=a.policy, cfg=a.cfg, tokenizer=f"{BASE_MODEL}@{BASE_REVISION}")
    df = PR.train_samples(ex, a.cfg)
    n = n_tokens(tok, [PR.messages(p, x) for p, x in zip(df["prompt"], df["answer"])], False)
    out["sft"] = dict(n_samples=int(len(df)), total=int(n.sum()), mean=float(n.mean()), p95=float(np.percentile(n, 95)),
                      max=int(n.max()), n_over_max_length=int((n > int(SFT_ARGS["max_length"])).sum()),
                      steps_per_epoch=int(np.ceil(len(df) / EFFECTIVE_BATCH)))
    for s in a.splits:
        q = ex.queries(s)["query_id"].to_numpy(np.int64)
        tt = PR.load_target_time(a.ds, s) if PR.CFGS[a.cfg]["target_time"] else None
        p = PR.query_prompts(ex, s, q, a.cfg, tt)
        m = n_tokens(tok, [PR.messages(x) for x in p["prompt"]], True)
        out[s] = dict(n=int(len(p)), total=int(m.sum()), mean=float(m.mean()), p95=float(np.percentile(m, 95)),
                      max=int(m.max()))
    pth = os.path.join(ROOT, "logs", "llm4poi", f"tokens_{a.ds}_{a.policy}_{a.cfg}.json")
    _write_json(out, pth)
    log(f"tokens -> {os.path.relpath(pth, ROOT)}: {json.dumps(out)}")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=(__doc__ or " "), formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", nargs="?", choices=["tokens", "_gen"])
    ap.add_argument("--ds", required=True)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--splits", "--split", dest="splits", nargs="+", default=["val", "test"], choices=["val", "test"])
    ap.add_argument("--cfg", default="default", choices=sorted(PR.CFGS))
    ap.add_argument("--base", default=None, choices=sorted(BASES),
                    help="base model key (scaling study); default $POIBENCH_LLM4POI_BASE or qwen2.5-7b")
    ap.add_argument("--stage", default="all", choices=["all", "prepare", "train", "predict"])
    ap.add_argument("--backend", default="vllm", choices=["vllm", "mock"],
                    help="mock = CPU stand-in; its output never goes to preds/")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", type=int, default=0, help="S1: number of SFT samples (work/ only)")
    ap.add_argument("--save-steps", type=int, default=SAVE_STEPS_DEFAULT)
    ap.add_argument("--canary-prep", action="store_true")
    ap.add_argument("--no-tokens", action="store_true", help="prepare without the tokenizer statistics")
    ap.add_argument("--model")
    ap.add_argument("--model-fp")
    ap.add_argument("--cache-dir")
    ap.add_argument("--inp")
    ap.add_argument("--sampling")
    a = ap.parse_args(argv)
    if a.seed != 0:
        ap.error("seed 0 only (owner rule)")
    if a.base:
        set_base(a.base)
    if a.cmd == "_gen":
        return gen_worker(a)
    ed = export_dir(a.ds, a.policy, ROOT)
    if not os.path.exists(os.path.join(ed, "MANIFEST.json")):
        raise SystemExit(f"{ed} missing: run scripts/export_inputs.py --datasets {a.ds} --policies {a.policy}")
    ex = Export(ed)
    if a.cmd == "tokens":
        token_report(a, ex)
        return 0
    if a.canary_prep:
        wd, _ = run_dirs(a.ds, a.policy, a.cfg, smoke=False)
        prepare_sft(ex, a.cfg, wd, with_tokens=not a.no_tokens)
        if not os.path.exists(default_canary_ckpt()):
            b = argparse.Namespace(**vars(a))
            b.ds, b.policy, b.cfg, b.smoke = CANARY_KEY[0], CANARY_KEY[1], CANARY_KEY[2], 2048
            smoke(b, Export(export_dir(b.ds, b.policy, ROOT)))
        log(f"canary-prep: fixed checkpoint {_read_json(default_canary_ckpt())['final_ckpt']}")
        return 0
    if a.smoke:
        return smoke(a, ex)
    if a.cfg == "tt" and not os.path.exists(os.path.join(ROOT, "exports_ablation", a.ds, "MANIFEST.json")):
        raise SystemExit(f"cfg tt needs exports_ablation/{a.ds}/ (scripts/export_target_time.py --ds {a.ds})")
    wd, ckdir = run_dirs(a.ds, a.policy, a.cfg, smoke=False)
    if a.backend != "vllm":
        wd = wd + "_" + a.backend
    log(f"ds={a.ds} policy={a.policy} cfg={a.cfg} stage={a.stage} splits={a.splits} backend={a.backend} "
        f"gpus={visible_gpus()} ckpt={ckdir} resume_env={os.environ.get('POIBENCH_RESUME')}")
    prompts, pmetas = {}, {}
    if a.stage in ("all", "prepare", "predict"):
        for s in a.splits:
            q = [int(x) for x in ex.queries(s)["query_id"].to_numpy(np.int64)]
            prompts[s], pmetas[s] = prepare_prompts(ex, a.ds, s, a.cfg, q, wd)
    done = None
    if a.stage in ("all", "prepare", "train"):
        meta = prepare_sft(ex, a.cfg, wd, with_tokens=not a.no_tokens)
        if a.stage in ("all", "train") and a.backend == "vllm":
            done = train(wd, ckdir, meta, save_steps=a.save_steps)
    if a.stage in ("all", "predict"):
        if a.backend == "vllm" and done is None:
            dp = os.path.join(ckdir, "DONE.json")
            if not os.path.exists(dp):
                raise SystemExit(f"no finished training at {ckdir}")
            done = _read_json(dp)
        out_base = os.path.join(ROOT, "preds") if a.backend == "vllm" else os.path.join(wd, "mock_preds")
        r = do_predict(a, ex, wd, done, prompts, pmetas, out_base)
        _write_json(r, os.path.join(wd, f"predict_summary_{a.backend}.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
