from __future__ import annotations

import json
import os
import subprocess
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COST_EXCLUSIVE = os.path.join(ROOT, "results", "cost_exclusive")


def assert_outside_preds(path: str, root: str = ROOT) -> str:
    p = os.path.realpath(os.path.abspath(path))
    preds = os.path.realpath(os.path.join(root, "preds"))
    if p == preds or p.startswith(preds + os.sep):
        raise RuntimeError(f"cost-only mode refuses to write under preds/: {path}")
    return path


def cost_path(method_at_policy: str, ds: str, split: str, cost_root: str | None = None) -> str:
    p = os.path.join(cost_root or COST_EXCLUSIVE, method_at_policy, ds, f"{split}.cost.json")
    return assert_outside_preds(p)


def _visible_physical_gpus() -> list[int]:
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    return [int(t) for t in cvd.split(",") if t.strip().isdigit()]


def _own_pids() -> set[int]:
    pids = {os.getpid()}
    try:
        import psutil
        pids |= {c.pid for c in psutil.Process().children(recursive=True)}
    except Exception:
        pass
    return pids


class GpuShareMonitor(threading.Thread):

    def __init__(self, interval: float = 10.0, gpus: list[int] | None = None):
        super().__init__(daemon=True)
        self.interval = interval
        self.gpus = _visible_physical_gpus() if gpus is None else list(gpus)
        self.my_job = os.path.basename(os.environ.get("POIBENCH_STATE_DIR", "").rstrip("/")) or None
        self.runner_json = (os.path.join(os.path.dirname(os.environ["POIBENCH_STATE_DIR"].rstrip("/")), "runner.json")
                            if os.environ.get("POIBENCH_STATE_DIR") else None)
        self.n_samples = 0
        self.n_samples_shared = 0
        self.foreign = {}
        self.runner_same_gpu = {}
        self.runner_other_gpu = {}
        self.method = None
        self.error = None
        self.started_utc = self.ended_utc = None
        self._stop_evt = threading.Event()
        self._nvml = None

    def _init_backend(self):
        if not self.gpus:
            return
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml = (pynvml, {g: pynvml.nvmlDeviceGetHandleByIndex(g) for g in self.gpus})
            self.method = "nvml"
            return
        except Exception as e:
            self.error = f"nvml: {type(e).__name__}: {e}"
        try:
            subprocess.run(["nvidia-smi", "-L"], capture_output=True, timeout=30, check=True)
            self.method = "nvidia-smi"
        except Exception as e:
            self.error = (self.error or "") + f"; nvidia-smi: {type(e).__name__}: {e}"

    def _procs(self) -> list[tuple[int, int, int]]:
        out = []
        if self.method == "nvml":
            pynvml, hs = self._nvml
            for g, h in hs.items():
                ps = []
                for fn in ("nvmlDeviceGetComputeRunningProcesses", "nvmlDeviceGetGraphicsRunningProcesses"):
                    try:
                        ps += list(getattr(pynvml, fn)(h))
                    except Exception:
                        pass
                for p in ps:
                    used = getattr(p, "usedGpuMemory", None)
                    out.append((g, int(p.pid), int(used // (1 << 20)) if isinstance(used, int) else -1))
        elif self.method == "nvidia-smi":
            q = subprocess.run(["nvidia-smi", "--query-gpu=index,pci.bus_id", "--format=csv,noheader"],
                               capture_output=True, text=True, timeout=30).stdout
            bus = {b.strip(): int(i) for i, b in (ln.split(",") for ln in q.strip().splitlines())}
            a = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_bus_id,pid,used_memory",
                                "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=30).stdout
            for ln in a.strip().splitlines():
                b, pid, mem = [x.strip() for x in ln.split(",")]
                g = bus.get(b)
                if g in self.gpus:
                    out.append((g, int(pid), int(mem) if mem.isdigit() else -1))
        return out

    def _runner_view(self) -> bool:
        if not self.runner_json:
            return False
        try:
            with open(self.runner_json) as f:
                run = json.load(f).get("running", {})
        except (OSError, ValueError):
            return False
        same = False
        for jid, gp in run.items():
            if jid == self.my_job:
                continue
            hit = bool(set(gp or []) & set(self.gpus))
            same = same or hit
            tgt = self.runner_same_gpu if hit else self.runner_other_gpu
            tgt[jid] = tgt.get(jid, 0) + 1
        return same

    def sample(self):
        shared = False
        if self.method:
            try:
                mine = _own_pids()
                for g, pid, mib in self._procs():
                    if pid in mine:
                        continue
                    shared = True
                    r = self.foreign.setdefault(pid, {"gpu": g, "max_used_mib": mib,
                                                      "first_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
                    r["max_used_mib"] = max(r["max_used_mib"], mib)
            except Exception as e:
                self.error = f"sample: {type(e).__name__}: {e}"
        shared = self._runner_view() or shared
        self.n_samples += 1
        self.n_samples_shared += int(shared)

    def start(self):
        self.started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self._init_backend()
        self.sample()
        super().start()
        return self

    def run(self):
        while not self._stop_evt.wait(self.interval):
            self.sample()

    def stop(self) -> dict:
        self._stop_evt.set()
        if self.is_alive():
            self.join()
        self.sample()
        self.ended_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        checked = bool(self.gpus) and self.method is not None
        return dict(gpus_physical=self.gpus, method=self.method, interval_s=self.interval, n_samples=self.n_samples,
                    n_samples_shared=self.n_samples_shared, foreign_processes=self.foreign,
                    runner_jobs_same_gpu=self.runner_same_gpu, runner_jobs_other_gpu_or_cpu=self.runner_other_gpu,
                    job=self.my_job, started_utc=self.started_utc, ended_utc=self.ended_utc, error=self.error,
                    checked=checked,
                    colocated=(None if not checked else bool(self.foreign or self.runner_same_gpu)))


def gpu_model() -> str | None:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return None


def preds_parity(work_parquet: str, preds_parquet: str) -> dict | None:
    if not (os.path.exists(work_parquet) and os.path.exists(preds_parquet)):
        return None
    import pandas as pd
    a = pd.read_parquet(work_parquet, columns=["query_id", "ranked_venue_ids"]).set_index("query_id")
    b = pd.read_parquet(preds_parquet, columns=["query_id", "ranked_venue_ids"]).set_index("query_id")
    common = a.index.intersection(b.index)
    same = sum(list(a.at[q, "ranked_venue_ids"]) == list(b.at[q, "ranked_venue_ids"]) for q in common)
    return dict(preds_of_record=os.path.relpath(preds_parquet, ROOT), n_common=int(len(common)),
                n_rerun=int(len(a)), n_record=int(len(b)), identical_top50_frac=(same / len(common)) if len(common) else None)


def record(mon: dict | None, **extra) -> dict:
    out = dict(cost_only=True, colocated=None if mon is None else mon["colocated"], colocation_check=mon,
               gpu_model=gpu_model(), cost_of_record_note=(
                   "exclusive re-run of the identical pipeline for the Q3 cost table; predictions of record are "
                   "preds/ (not replaced)"))
    out.update(extra)
    return out
