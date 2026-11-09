from __future__ import annotations

import base64
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

MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"
ENGINE = {"seed": 0, "max_model_len": 32768, "limit_mm_per_prompt": {"image": 24}}
SAMPLING = {"rank": {"temperature": 0.0, "max_tokens": 256}, "caption": {"temperature": 0.0, "max_tokens": 512}}
CHUNK = 512


def request_key(backend_id: str, kind: str, req: list) -> str:
    norm = [[role, [list(s[:2]) if s[0] == "text" else ["image", s[2]] for s in segs]] for role, segs in req]
    s = json.dumps({"backend": backend_id, "sampling": SAMPLING[kind], "req": norm}, sort_keys=True,
                   ensure_ascii=False)
    return hashlib.sha256(s.encode()).hexdigest()


def n_images(req: list) -> int:
    return sum(1 for _, segs in req for s in segs if s[0] == "image")


def _data_url(path: str) -> str:
    import io
    from PIL import Image
    with Image.open(path) as im:
        buf = io.BytesIO()
        im.convert("RGB").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def to_openai_messages(req: list) -> list:
    out = []
    for role, segs in req:
        if all(s[0] == "text" for s in segs):
            out.append({"role": role, "content": "".join(s[1] for s in segs)})
            continue
        parts = []
        for s in segs:
            if s[0] == "text":
                if s[1]:
                    parts.append({"type": "text", "text": s[1]})
            else:
                parts.append({"type": "image_url", "image_url": {"url": _data_url(s[1])}})
        out.append({"role": role, "content": parts})
    return out


def resolve_model(model: str = MODEL, revision: str = REVISION) -> dict:
    name = model.split("/")[-1]
    cands = [os.environ.get("QWENVL_MODEL_PATH")]
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
            elif d != os.environ.get("QWENVL_MODEL_PATH"):
                continue
            return dict(path=d, revision=sha or "unknown-local", local=True, pinned=revision,
                        matches_pin=(sha == revision))
    return dict(path=model, revision=revision, local=False, pinned=revision, matches_pin=True)


class VLLMBackend:
    kind = "vllm"

    def __init__(self):
        self.src = resolve_model()
        self.revision = self.src["revision"]
        if not self.src["matches_pin"]:
            print(f"WARNING qwen25_vl_ref: weights at {self.src['path']} are revision {self.revision}, pinned "
                  f"{REVISION}", flush=True)
        self.id = f"vllm|{MODEL}@{self.revision}"
        self.loaded = False

    def load(self):
        from vllm import LLM, SamplingParams
        kw = dict(seed=ENGINE["seed"], max_model_len=ENGINE["max_model_len"],
                  limit_mm_per_prompt=ENGINE["limit_mm_per_prompt"])
        if self.src["local"]:
            self.llm = LLM(model=self.src["path"], **kw)
        else:
            self.llm = LLM(model=MODEL, revision=self.revision, tokenizer_revision=self.revision, **kw)
        self.sp = {k: SamplingParams(**v) for k, v in SAMPLING.items()}
        self.loaded = True
        return self

    def generate(self, reqs: list, kind: str) -> list:
        outs = self.llm.chat([to_openai_messages(r) for r in reqs], sampling_params=self.sp[kind], use_tqdm=False)
        res = []
        for o in outs:
            c = o.outputs[0]
            res.append(dict(text=c.text, n_prompt_tokens=len(o.prompt_token_ids or []),
                            n_output_tokens=len(c.token_ids), finish_reason=str(c.finish_reason)))
        return res


_PLACE = re.compile(r"place (\d+)")


class MockBackend:
    kind = "mock"
    id = "mock-v1"
    loaded = False
    revision = None
    src = None

    def load(self):
        self.loaded = True
        return self

    def generate(self, reqs: list, kind: str) -> list:
        out = []
        for r in reqs:
            h = request_key(self.id, kind, r)
            text = "".join(s[1] for _, segs in r for s in segs if s[0] == "text")
            if kind == "caption":
                t = json.dumps({"summary": f"mock summary {h[:8]}", "keywords": [h[8:12], h[12:16]]})
            else:
                ids = list(dict.fromkeys(_PLACE.findall(text.split("Candidate places", 1)[-1])))
                k = int(h, 16)
                ids = sorted(ids, key=lambda x: hashlib.sha256((h + x).encode()).hexdigest())
                t = ("[" + ", ".join(ids[: 5 + k % 16]) + "]") if k % 7 else "I am not sure."
            out.append(dict(text=t, n_prompt_tokens=len(text) // 4 + 51 * n_images(r),
                            n_output_tokens=max(1, len(t) // 4), finish_reason="stop"))
        return out


BACKENDS = {"vllm": VLLMBackend, "mock": MockBackend}
_REGISTRY = {}


def get_backend(kind: str):
    if kind not in _REGISTRY:
        _REGISTRY[kind] = BACKENDS[kind]()
    return _REGISTRY[kind]


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)[:120]


class ResponseCache:
    COLS = ["key", "text", "n_prompt_tokens", "n_output_tokens", "finish_reason", "gen_s", "n_gpus", "chunk_size",
            "job", "created_utc", "host"]

    def __init__(self, backend_id: str, kind: str, cache_dir: str | None = None):
        base = cache_dir or os.environ.get("QWENVL_CACHE_DIR") or os.path.join(ROOT, "work", "qwen25_vl_ref", "cache")
        self.dir = os.path.join(base, _slug(backend_id), kind)
        os.makedirs(self.dir, exist_ok=True)
        self.mem = {}
        self._seq = 0
        for f in sorted(os.listdir(self.dir)):
            if f.startswith("shard_") and f.endswith(".parquet"):
                for r in pd.read_parquet(os.path.join(self.dir, f)).to_dict("records"):
                    self.mem.setdefault(r["key"], r)

    def put(self, recs: list) -> None:
        self._seq += 1
        name = (f"shard_{datetime.now(timezone.utc):%Y%m%dT%H%M%S}_{_slug(socket.gethostname())}_{os.getpid()}_"
                f"{self._seq:05d}_{uuid.uuid4().hex[:12]}.parquet")
        p = os.path.join(self.dir, name)
        pd.DataFrame(recs, columns=self.COLS).to_parquet(p + ".tmp", index=False)
        os.replace(p + ".tmp", p)
        for r in recs:
            self.mem.setdefault(r["key"], r)


def n_pending(reqs: list, kind: str, backend, cache: ResponseCache) -> int:
    return len({request_key(backend.id, kind, r) for r in reqs} - set(cache.mem))


def generate_cached(reqs: list, kind: str, backend, cache: ResponseCache, n_gpus: int, job: str = "",
                    log=print, on_load=None) -> tuple[list, dict]:
    keys = [request_key(backend.id, kind, r) for r in reqs]
    first = {}
    for k, r in zip(keys, reqs):
        first.setdefault(k, r)
    todo = [k for k in first if k not in cache.mem]
    st = dict(n_requests=len(reqs), n_unique=len(first), n_cache_hits=len(first) - len(todo), n_generated=0,
              gen_wall_s=0.0, loaded=False, cached_keys=[k for k in first if k in cache.mem])
    if todo:
        if not backend.loaded:
            (on_load or (lambda f: f()))(backend.load)
            st["loaded"] = True
        for s in range(0, len(todo), CHUNK):
            ks = todo[s:s + CHUNK]
            t0 = time.perf_counter()
            res = backend.generate([first[k] for k in ks], kind)
            dt = time.perf_counter() - t0
            now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            cache.put([dict(key=k, n_gpus=n_gpus, gen_s=dt / len(ks), chunk_size=len(ks), job=job, created_utc=now,
                            host=socket.gethostname(), **r) for k, r in zip(ks, res)])
            st["n_generated"] += len(ks)
            st["gen_wall_s"] += dt
            log(f"  {kind}: generated {st['n_generated']}/{len(todo)} ({dt:.1f} s for {len(ks)})")
    return [cache.mem[k] for k in keys], st
