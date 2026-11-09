import json
import os
import tempfile
import time
import unittest

from poibench.cost import REQUIRED_STAGE_FIELDS, CostRecorder, load_cost, summarize_cost


class TestCost(unittest.TestCase):
    def test_stages_and_file(self):
        rec = CostRecorder(method="m", ds="toy", split="val", sample_interval=0.02)
        with rec.stage("fit", one_time=True):
            x = 0
            t0 = time.process_time()
            while time.process_time() - t0 < 0.2:
                x += 1
            buf = bytearray(50_000_000)
            time.sleep(0.1)
            del buf
        with rec.stage("predict", one_time=False, n_items=10, notes="x") as r:
            time.sleep(0.05)
            r["api_tokens"] = 123
        rec.add_stage("remote_train", wall_s=10, cpu_core_s=20, gpu_s=20, peak_mem_gb=3, one_time=True, n_gpus=2)
        fit, pred, remote = rec.stages
        for s in rec.stages:
            for k in REQUIRED_STAGE_FIELDS:
                self.assertIn(k, s)
        self.assertGreaterEqual(fit["cpu_core_s"], 0.15)
        self.assertGreaterEqual(fit["wall_s"], 0.3)
        self.assertGreater(fit["peak_mem_gb"], 0.05)
        self.assertEqual(fit["gpu_s"], 0.0)
        self.assertTrue(fit["one_time"])
        self.assertFalse(pred["one_time"])
        self.assertEqual(pred["n_items"], 10)
        self.assertEqual(pred["api_tokens"], 123)
        self.assertGreaterEqual(pred["wall_s"], 0.05)
        with tempfile.TemporaryDirectory() as d:
            p = rec.save(os.path.join(d, "sub", "val.cost.json"), n_queries=10)
            data = load_cost(p)
            self.assertEqual(data["method"], "m")
            self.assertEqual(data["n_queries"], 10)
            self.assertIn("hardware", data)
            s = summarize_cost(data)
            self.assertAlmostEqual(s["one_time"]["gpu_s"], 20.0)
            self.assertAlmostEqual(s["recurring_per_query"]["wall_s"], pred["wall_s"] / 10)
            bad = dict(data)
            bad["stages"] = [{"name": "x"}]
            with open(p, "w") as f:
                json.dump(bad, f)
            with self.assertRaises(ValueError):
                load_cost(p)

    def test_stage_recorded_on_exception(self):
        rec = CostRecorder()
        with self.assertRaises(RuntimeError):
            with rec.stage("boom", one_time=False):
                raise RuntimeError("x")
        self.assertEqual(rec.stages[0]["name"], "boom")

    def test_no_gpu_without_cuda_in_process(self):
        rec = CostRecorder()
        with rec.stage("g", one_time=False):
            time.sleep(0.01)
        self.assertEqual(rec.stages[0]["gpu_s"], 0.0)
        self.assertEqual(rec.stages[0]["n_gpus"], 0)
        self.assertNotIn("gpu_busy_s", rec.stages[0])


if __name__ == "__main__":
    unittest.main()
