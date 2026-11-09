from __future__ import annotations

import json
import os
import platform
import resource
import socket
import sys
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone

try:
    import psutil
except Exception:
    psutil = None

REQUIRED_STAGE_FIELDS = ("name", "wall_s", "cpu_core_s", "gpu_s", "peak_mem_gb", "one_time")


def _cpu_seconds() -> float:
    s = resource.getrusage(resource.RUSAGE_SELF)
    c = resource.getrusage(resource.RUSAGE_CHILDREN)
    return s.ru_utime + s.ru_stime + c.ru_utime + c.ru_stime


def _proc_mem(p) -> int:
    try:
        return p.memory_full_info().pss
    except Exception:
        return p.memory_info().rss


def _rss_bytes() -> int:
    if psutil is None:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    p = psutil.Process()
    total = 0
    try:
        total = _proc_mem(p)
        for ch in p.children(recursive=True):
            try:
                total += _proc_mem(ch)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    return total


def _torch_cuda_in_use():
    torch = sys.modules.get("torch")
    if torch is None:
        return None
    try:
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            return torch
    except Exception:
        return None
    return None


def _visible_gpu_indices(n_gpus: int):
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd:
        idx = []
        for tok in cvd.split(","):
            tok = tok.strip()
            if tok.isdigit():
                idx.append(int(tok))
        if idx:
            return idx[:n_gpus]
    return list(range(n_gpus))


class _Sampler(threading.Thread):
    def __init__(self, interval: float, gpu_indices):
        super().__init__(daemon=True)
        self.interval = interval
        self.peak_rss = _rss_bytes()
        self.gpu_busy_s = 0.0
        self.peak_gpu_used_bytes = 0
        self._stop_evt = threading.Event()
        self._nvml_handles = []
        if gpu_indices:
            try:
                import pynvml
                pynvml.nvmlInit()
                self._pynvml = pynvml
                self._nvml_handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in gpu_indices]
            except Exception:
                self._nvml_handles = []

    def run(self):
        last = time.perf_counter()
        while not self._stop_evt.wait(self.interval):
            self.peak_rss = max(self.peak_rss, _rss_bytes())
            now = time.perf_counter()
            if self._nvml_handles:
                try:
                    util = sum(self._pynvml.nvmlDeviceGetUtilizationRates(h).gpu for h in self._nvml_handles)
                    self.gpu_busy_s += util / 100.0 * (now - last)
                    used = sum(self._pynvml.nvmlDeviceGetMemoryInfo(h).used for h in self._nvml_handles)
                    self.peak_gpu_used_bytes = max(self.peak_gpu_used_bytes, int(used))
                except Exception:
                    pass
            last = now

    def stop(self):
        self._stop_evt.set()
        self.join()
        self.peak_rss = max(self.peak_rss, _rss_bytes())


def hardware_info() -> dict:
    info = {
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count_logical": os.cpu_count(),
        "cpu_affinity": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "cpu_model": None,
        "gpu_model": None,
    }
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    info["cpu_model"] = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    torch = _torch_cuda_in_use()
    if torch is not None:
        try:
            info["gpu_model"] = torch.cuda.get_device_name(0)
            info["gpu_count_visible"] = torch.cuda.device_count()
        except Exception:
            pass
    return info


class CostRecorder:

    def __init__(self, path: str | None = None, method: str | None = None, ds: str | None = None,
                 split: str | None = None, sample_interval: float = 0.2, extra: dict | None = None):
        self.path = path
        self.meta = {"method": method, "ds": ds, "split": split}
        if extra:
            self.meta.update(extra)
        self.sample_interval = sample_interval
        self.stages: list[dict] = []

    @contextmanager
    def stage(self, name: str, one_time: bool, n_gpus: int | None = None, n_items: int | None = None,
              notes: str | None = None):
        torch = _torch_cuda_in_use()
        if n_gpus is None:
            n_gpus = torch.cuda.device_count() if torch is not None else 0
        gpu_idx = _visible_gpu_indices(n_gpus) if n_gpus > 0 else []
        if torch is not None and n_gpus > 0:
            for d in range(torch.cuda.device_count()):
                try:
                    torch.cuda.reset_peak_memory_stats(d)
                except Exception:
                    pass
        sampler = _Sampler(self.sample_interval, gpu_idx)
        sampler.start()
        cpu0 = _cpu_seconds()
        t0 = time.perf_counter()
        rec = {"name": name}
        try:
            yield rec
        finally:
            if torch is not None and n_gpus > 0:
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
            wall = time.perf_counter() - t0
            cpu = _cpu_seconds() - cpu0
            sampler.stop()
            out = {
                "name": name,
                "wall_s": round(wall, 6),
                "cpu_core_s": round(cpu, 6),
                "gpu_s": round(wall * n_gpus, 6),
                "peak_mem_gb": round(sampler.peak_rss / 1e9, 6),
                "one_time": bool(one_time),
                "n_gpus": int(n_gpus),
            }
            if gpu_idx:
                out["gpu_busy_s"] = round(sampler.gpu_busy_s, 6)
                if sampler.peak_gpu_used_bytes:
                    out["peak_gpu_mem_nvml_gb"] = round(sampler.peak_gpu_used_bytes / 1e9, 6)
            if torch is not None and n_gpus > 0:
                try:
                    out["peak_gpu_mem_gb"] = round(sum(torch.cuda.max_memory_allocated(d)
                                                       for d in range(torch.cuda.device_count())) / 1e9, 6)
                except Exception:
                    pass
            if n_items is not None:
                out["n_items"] = int(n_items)
            if notes:
                out["notes"] = notes
            for k, v in rec.items():
                if k not in out:
                    out[k] = v
            self.stages.append(out)

    def add_stage(self, name: str, wall_s: float, cpu_core_s: float, gpu_s: float, peak_mem_gb: float,
                  one_time: bool, **extra):
        rec = {"name": name, "wall_s": float(wall_s), "cpu_core_s": float(cpu_core_s), "gpu_s": float(gpu_s),
               "peak_mem_gb": float(peak_mem_gb), "one_time": bool(one_time)}
        rec.update(extra)
        self.stages.append(rec)

    def to_dict(self, n_queries: int | None = None) -> dict:
        d = dict(self.meta)
        d["n_queries"] = n_queries
        d["created_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        d["hardware"] = hardware_info()
        d["stages"] = list(self.stages)
        return d

    def save(self, path: str | None = None, n_queries: int | None = None, keep_measured: tuple | None = None) -> str:
        path = path or self.path
        if path is None:
            raise ValueError("no output path for cost.json")
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        d = self.to_dict(n_queries)
        if keep_measured:
            d = preserve_measured(path, d, keep_measured)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(d, f, indent=2)
        os.replace(tmp, path)
        return path


def load_cost(path: str) -> dict:
    with open(path) as f:
        d = json.load(f)
    for s in d.get("stages", []):
        missing = [k for k in REQUIRED_STAGE_FIELDS if k not in s]
        if missing:
            raise ValueError(f"{path}: stage {s.get('name')} missing {missing}")
    return d


def summarize_cost(d: dict, n_queries: int | None = None) -> dict:
    n = n_queries if n_queries is not None else d.get("n_queries")
    keys = ("wall_s", "cpu_core_s", "gpu_s")
    one = {k: sum(s[k] for s in d["stages"] if s["one_time"]) for k in keys}
    rec = {k: sum(s[k] for s in d["stages"] if not s["one_time"]) for k in keys}
    peak = max([s["peak_mem_gb"] for s in d["stages"]] or [0.0])
    out = {"one_time": one, "recurring": rec, "peak_mem_gb": peak, "n_queries": n}
    if n:
        out["recurring_per_query"] = {k: v / n for k, v in rec.items()}
        out["amortized_per_query_at_n"] = {k: (one[k] + rec[k]) / n for k in keys}
    return out


def cache_hits_of(d: dict):
    e = d.get("execution") or {}
    if e.get("n_cache_hits") is not None:
        return int(e["n_cache_hits"])
    g = d.get("generation") or {}
    for k in ("n_cache_hits", "n_cached"):
        if g.get(k) is not None:
            return int(g[k])
    return None


def _field(d: dict, key: str):
    for part in key.split("."):
        if not isinstance(d, dict):
            return None
        d = d.get(part)
    return d


def measured_cache_hits(d: dict):
    src = d.get("per_query_stages_from")
    if isinstance(src, dict) and src.get("n_cache_hits") is not None:
        return int(src["n_cache_hits"])
    return cache_hits_of(d)


def preserve_measured(path: str, new: dict, match: tuple) -> dict:
    if not os.path.exists(path):
        return new
    try:
        with open(path) as f:
            old = json.load(f)
    except (OSError, ValueError):
        return new
    h_new, h_old = cache_hits_of(new), measured_cache_hits(old)
    if not h_new or h_old is None or h_old >= h_new:
        return new
    for k in match:
        a, b = _field(old, k), _field(new, k)
        if a is None or b is None or a != b:
            return new
    out = dict(new)
    out["stages"] = ([s for s in new["stages"] if s.get("one_time")]
                     + [s for s in old["stages"] if not s.get("one_time")])
    src = old.get("per_query_stages_from")
    if not (isinstance(src, dict) and src.get("n_cache_hits") is not None):
        src = dict(created_utc=old.get("created_utc"), n_cache_hits=h_old, execution=old.get("execution"),
                   why="earlier record of the same work with fewer cache hits (measured cost kept; later executions "
                       "are listed under reruns)")
    out["per_query_stages_from"] = src
    out["reruns"] = list(old.get("reruns") or []) + [dict(new.get("execution") or {}, created_utc=new.get("created_utc"),
                                                          n_cache_hits=h_new)]
    return out


def cached_answers_stage(recs: list, n_items: int | None = None, notes: str | None = None) -> dict | None:
    if not recs:
        return None
    g = sum(float(r.get("gen_s") or 0.0) for r in recs)
    gg = sum(float(r.get("gen_s") or 0.0) * float(r.get("n_gpus") if r.get("n_gpus") is not None else 1)
             for r in recs)
    return dict(name="infer_cached", wall_s=g, cpu_core_s=0.0, gpu_s=gg, peak_mem_gb=0.0, one_time=False,
                n_items=int(n_items if n_items is not None else len(recs)),
                notes=notes or "generation time of answers taken from the response cache, attributed from the "
                               "per-answer generation time stored with them (excludes the producing run's model load)")
