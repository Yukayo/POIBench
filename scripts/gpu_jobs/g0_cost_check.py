#!/usr/bin/env python
import argparse
import os
import socket
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

from poibench import ROOT
from poibench.cost import CostRecorder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=60.0)
    a = ap.parse_args()
    assert torch.cuda.is_available(), "no CUDA device visible"
    n = torch.cuda.device_count()
    torch.manual_seed(0)
    _ = torch.zeros(1, device="cuda")
    rec = CostRecorder(method="g0_cost_check", ds=None, split=None,
                       extra={"cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                              "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER")})
    with rec.stage("busy_matmul", one_time=False, n_gpus=n):
        xs = [torch.randn(8192, 8192, device=f"cuda:{d}", dtype=torch.bfloat16) for d in range(n)]
        t0 = time.time()
        while time.time() - t0 < a.seconds:
            for x in xs:
                y = x @ x
            torch.cuda.synchronize()
        del y
    with rec.stage("idle_hold", one_time=False, n_gpus=n):
        time.sleep(min(20.0, a.seconds / 3))
    out = os.path.join(ROOT, "results", "gpu_checks", f"g0_{socket.gethostname()}.cost.json")
    rec.save(out)
    busy, idle = rec.stages
    checks = {
        "gpu_s == wall_s * n_gpus": abs(busy["gpu_s"] - busy["wall_s"] * n) < 1e-3,
        "gpu_busy_s present": "gpu_busy_s" in busy,
        "busy stage utilisation > 50%": busy.get("gpu_busy_s", 0) > 0.5 * busy["wall_s"] * n,
        "idle stage utilisation < 20%": idle.get("gpu_busy_s", 1e9) < 0.2 * idle["wall_s"] * n,
        "peak_gpu_mem_gb (torch) > 0.1": busy.get("peak_gpu_mem_gb", 0) > 0.1,
        "peak_gpu_mem_nvml_gb > 0.1": busy.get("peak_gpu_mem_nvml_gb", 0) > 0.1,
        "hardware.gpu_model set": bool(rec.to_dict()["hardware"].get("gpu_model")),
    }
    for k, v in checks.items():
        print(("PASS " if v else "FAIL ") + k)
    print("written:", out)
    sys.exit(0 if all(checks.values()) else 1)


if __name__ == "__main__":
    main()
