from __future__ import annotations

import hashlib
import json
import os
import re
import socket
import time
import uuid
from datetime import datetime, timezone

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))

MODEL = "Qwen/Qwen2.5-7B-Instruct"
REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
SAMPLING = {"temperature": 0.0, "max_tokens": 2000}
ENGINE_SEED = 0
CHUNK = 1024
DEFAULT_MODEL_KEY = "qwen2.5-7b"
MODELS = {
    "qwen2.5-7b": (MODEL, REVISION),
    "qwen2.5-3b": ("Qwen/Qwen2.5-3B-Instruct", None),
    "qwen2.5-14b": ("Qwen/Qwen2.5-14B-Instruct", None),
    "qwen2.5-32b": ("Qwen/Qwen2.5-32B-Instruct", None),
    "llama3.1-8b": ("meta-llama/Llama-3.1-8B-Instruct", None),
    "mistral-7b-v0.3": ("mistralai/Mistral-7B-Instruct-v0.3", None),
}


def messages(prompt: str) -> list:
    return [{"role": "user", "content": prompt}]


def resolve_model(model: str = MODEL, revision: str = REVISION) -> dict:
    name = model.split("/")[-1]
    cands = [os.environ.get("DSLLM4POI_MODEL_PATH")]
    if os.environ.get("POIBENCH_MODEL_DIR"):
        cands.append(os.path.join(os.environ["POIBENCH_MODEL_DIR"], name))
    for d in cands:
        if d and os.path.isdir(d):
            mk = os.path.join(d, ".poibench_complete")
            sha = None
            if os.path.exists(mk):
                with open(mk) as f:
                    m = json.load(f)
                if m.get("repo") not in (None, model):
                    raise RuntimeError(f"{d} holds {m.get('repo')}, not {model}")
                sha = m.get("sha")
            elif d != os.environ.get("DSLLM4POI_MODEL_PATH"):
                continue
            return dict(path=d, revision=sha or "unknown-local", local=True, pinned=revision,
                        matches_pin=(revision is None or sha == revision))
    if revision is None:
        raise RuntimeError(f"{model}: no local copy with a .poibench_complete marker under $POIBENCH_MODEL_DIR "
                           f"(scaling-study models are only used from a verified local copy)")
    return dict(path=model, revision=revision, local=False, pinned=revision, matches_pin=True)


class VLLMBackend:
    kind = "vllm"

    def __init__(self, model: str = MODEL, revision: str = REVISION):
        self.model = model
        self.src = resolve_model(model, revision)
        self.revision = self.src["revision"]
        if not self.src["matches_pin"]:
            print(f"WARNING ds_llm4poi: weights at {self.src['path']} are revision {self.revision}, pinned "
                  f"{revision}; the actual revision is recorded in cost.json and in the cache key", flush=True)
        self.id = f"vllm|{model}@{self.revision}"
        self.llm = None
        self.loaded = False

    def load(self):
        from vllm import LLM, SamplingParams
        if self.src["local"]:
            self.llm = LLM(model=self.src["path"], seed=ENGINE_SEED)
        else:
            self.llm = LLM(model=self.model, revision=self.revision, tokenizer_revision=self.revision,
                           seed=ENGINE_SEED)
        self.sp = SamplingParams(temperature=SAMPLING["temperature"], max_tokens=SAMPLING["max_tokens"])
        try:
            self.max_len = int(self.llm.llm_engine.model_config.max_model_len)
        except Exception:
            self.max_len = 32768
        self.tok = self.llm.get_tokenizer()
        self.loaded = True
        return self

    def n_prompt_tokens(self, prompt: str) -> int:
        text = self.tok.apply_chat_template(messages(prompt), tokenize=False, add_generation_prompt=True)
        return len(self.tok(text, add_special_tokens=False)["input_ids"])

    def generate(self, prompts: list) -> list:
        res = [None] * len(prompts)
        run = []
        for k, p in enumerate(prompts):
            if len(p) >= self.max_len - 64:
                n = self.n_prompt_tokens(p)
                if n >= self.max_len:
                    res[k] = dict(text="", n_prompt_tokens=n, n_output_tokens=0,
                                  finish_reason="skipped_prompt_too_long", n_cached_prompt_tokens=0)
                    continue
            run.append(k)
        outs = self.llm.chat([messages(prompts[k]) for k in run], sampling_params=self.sp, use_tqdm=False) if run else []
        for k, o in zip(run, outs):
            c = o.outputs[0]
            res[k] = dict(text=c.text, n_prompt_tokens=len(o.prompt_token_ids or []),
                          n_output_tokens=len(c.token_ids), finish_reason=str(c.finish_reason),
                          n_cached_prompt_tokens=int(getattr(o, "num_cached_tokens", 0) or 0))
        return res


_CTX_ID = re.compile(r"\(\d\d:\d\d [AP]M, \w+, (\d+), ")


def mock_response(prompt: str) -> str:
    h = int(hashlib.sha256(prompt.encode()).hexdigest(), 16)
    ids = list(dict.fromkeys(_CTX_ID.findall(prompt))) or ["0"]
    pid = ids[h % len(ids)]
    mode = (h >> 64) % 5
    if mode == 0:
        return f'```json\n{{"place_id": {pid}, "place_category": "Bar"}}\n```'
    if mode == 1:
        return f'{{"place_id": {pid}, "place_category": "Bar"}}'
    if mode == 2:
        return f'```json\n{{"place_id": "{pid}", "place_category": "Bar"}}\n```'
    if mode == 3:
        return f'```json\n{{"place_id": {pid}}}\n```'
    return "I cannot determine the next place."


class MockBackend:
    kind = "mock"
    id = "mock-v1"
    loaded = False

    def load(self):
        self.loaded = True
        return self

    def generate(self, prompts: list) -> list:
        out = []
        for p in prompts:
            t = mock_response(p)
            out.append(dict(text=t, n_prompt_tokens=len(p) // 4, n_output_tokens=max(1, len(t) // 4),
                            finish_reason="stop", n_cached_prompt_tokens=0))
        return out


BACKENDS = {"vllm": VLLMBackend, "mock": MockBackend}


def cache_key(backend_id: str, prompt: str) -> str:
    s = json.dumps({"backend": backend_id, "sampling": SAMPLING, "messages": messages(prompt)}, sort_keys=True,
                   ensure_ascii=False)
    return hashlib.sha256(s.encode()).hexdigest()


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)[:120]


class ResponseCache:
    COLS = ["key", "text", "n_prompt_tokens", "n_output_tokens", "finish_reason", "n_cached_prompt_tokens",
            "gen_s", "n_gpus", "chunk_size", "job", "created_utc", "host"]

    def __init__(self, backend_id: str, root: str = ROOT, cache_dir: str | None = None):
        self.dir = cache_dir or os.path.join(root, "work", "ds_llm4poi", "llm_cache", _slug(backend_id))
        os.makedirs(self.dir, exist_ok=True)
        self.mem = {}
        self._seq = 0
        self.reload()

    def reload(self):
        for f in sorted(os.listdir(self.dir)):
            if f.startswith("shard_") and f.endswith(".parquet"):
                df = pd.read_parquet(os.path.join(self.dir, f))
                for r in df.to_dict("records"):
                    self.mem.setdefault(r["key"], r)

    def put(self, recs: list) -> str:
        self._seq += 1
        name = (f"shard_{datetime.now(timezone.utc):%Y%m%dT%H%M%S}_{_slug(socket.gethostname())}_{os.getpid()}_"
                f"{self._seq:05d}_{uuid.uuid4().hex[:12]}.parquet")
        p = os.path.join(self.dir, name)
        pd.DataFrame(recs, columns=self.COLS).to_parquet(p + ".tmp", index=False)
        os.replace(p + ".tmp", p)
        for r in recs:
            self.mem.setdefault(r["key"], r)
        return p


def generate_cached(prompts: list, backend, cache: ResponseCache, n_gpus: int, job: str = "", chunk: int = CHUNK,
                    log=print, on_load=None) -> tuple[list, dict]:
    keys = [cache_key(backend.id, p) for p in prompts]
    first = {}
    for k, p in zip(keys, prompts):
        first.setdefault(k, p)
    todo = [k for k in first if k not in cache.mem]
    hits_before = len(first) - len(todo)
    stats = dict(n_prompts=len(prompts), n_unique=len(first), n_cache_hits=hits_before, n_generated=0,
                 gen_wall_s=0.0, loaded=False)
    if todo:
        if not backend.loaded:
            (on_load or (lambda f: f()))(backend.load)
            stats["loaded"] = True
        for s in range(0, len(todo), chunk):
            ks = todo[s:s + chunk]
            t0 = time.perf_counter()
            res = backend.generate([first[k] for k in ks])
            dt = time.perf_counter() - t0
            now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            cache.put([dict(key=k, n_gpus=n_gpus, gen_s=dt / len(ks), chunk_size=len(ks), job=job,
                            created_utc=now, host=socket.gethostname(), **r) for k, r in zip(ks, res)])
            stats["n_generated"] += len(ks)
            stats["gen_wall_s"] += dt
            log(f"  generated {stats['n_generated']}/{len(todo)} new prompts ({dt:.1f} s for {len(ks)})")
    out = [cache.mem[k] for k in keys]
    return out, stats
