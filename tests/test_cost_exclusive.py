import importlib.util
import json
import os
import sys
import tempfile
import unittest

from poibench import exclusive as X
from poibench import exports as E
from tests import synth
from tests.test_exports import write_synth

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GF_DIR = os.path.join(ROOT, "methods", "graph_flashback")


def _load(name, path, extra_path=None):
    if extra_path and extra_path not in sys.path:
        sys.path.insert(0, extra_path)
    sp = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m)
    return m


def _preds_snapshot():
    out = []
    base = os.path.join(ROOT, "preds")
    for d, _, fs in os.walk(base):
        for f in fs:
            p = os.path.join(d, f)
            st = os.stat(p)
            out.append((os.path.relpath(p, base), st.st_size, st.st_mtime_ns))
    return sorted(out)


class TestGuard(unittest.TestCase):
    def test_preds_paths_refused(self):
        for p in (os.path.join(ROOT, "preds"), os.path.join(ROOT, "preds", "getnext@H-100", "nyc", "val.cost.json"),
                  os.path.join(ROOT, "results", "..", "preds", "x.json")):
            with self.assertRaises(RuntimeError):
                X.assert_outside_preds(p)
        with self.assertRaises(RuntimeError):
            X.cost_path("getnext@H-100", "nyc", "val", cost_root=os.path.join(ROOT, "preds"))
        self.assertEqual(X.cost_path("getnext@H-100", "nyc", "val"),
                         os.path.join(ROOT, "results", "cost_exclusive", "getnext@H-100", "nyc", "val.cost.json"))
        X.assert_outside_preds(os.path.join(ROOT, "predsx", "a.json"))

    def test_symlink_into_preds_refused(self):
        with tempfile.TemporaryDirectory() as td:
            link = os.path.join(td, "lnk")
            os.symlink(os.path.join(ROOT, "preds"), link)
            with self.assertRaises(RuntimeError):
                X.assert_outside_preds(os.path.join(link, "getnext@H-100", "nyc", "val.cost.json"))


class TestMonitor(unittest.TestCase):
    def test_no_gpu_is_unchecked(self):
        r = X.GpuShareMonitor(interval=0.05, gpus=[]).start().stop()
        self.assertFalse(r["checked"])
        self.assertIsNone(r["colocated"])

    def _fake(self, procs, running=None):
        with tempfile.TemporaryDirectory() as td:
            sd = os.path.join(td, "X01")
            os.makedirs(sd)
            if running is not None:
                with open(os.path.join(td, "runner.json"), "w") as f:
                    json.dump({"running": running}, f)
            old = os.environ.get("POIBENCH_STATE_DIR")
            os.environ["POIBENCH_STATE_DIR"] = sd
            try:
                m = X.GpuShareMonitor(interval=0.05, gpus=[1])
                m._init_backend = lambda: setattr(m, "method", "fake")
                m._procs = lambda: procs
                return m.start().stop()
            finally:
                if old is None:
                    os.environ.pop("POIBENCH_STATE_DIR")
                else:
                    os.environ["POIBENCH_STATE_DIR"] = old

    def test_exclusive(self):
        r = self._fake([(1, os.getpid(), 900)], running={"X01": [1], "J31": [0], "K1": []})
        self.assertTrue(r["checked"])
        self.assertIs(r["colocated"], False)
        self.assertEqual(set(r["runner_jobs_other_gpu_or_cpu"]), {"J31", "K1"})
        self.assertEqual(r["n_samples_shared"], 0)

    def test_foreign_process_flags_colocation(self):
        r = self._fake([(1, os.getpid(), 900), (1, 999999, 4000)])
        self.assertIs(r["colocated"], True)
        self.assertIn(999999, r["foreign_processes"])
        self.assertEqual(r["n_samples_shared"], r["n_samples"])

    def test_runner_job_on_same_gpu_flags_colocation(self):
        r = self._fake([], running={"X01": [1], "J17": [1]})
        self.assertIs(r["colocated"], True)
        self.assertEqual(r["runner_jobs_same_gpu"]["J17"], r["n_samples"])


class TestGetnextCostOnly(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.G = _load("getnext_run_costonly", os.path.join(ROOT, "methods", "getnext", "run.py"))
        rows, nv = synth.random_rows(n_users=20, seed=3)
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = cls.tmp.name
        write_synth(cls.root, "rnd", synth.build_tables(rows, n_venues=nv), xs=(100,))
        E.export_inputs("rnd", "H-100", cls.root)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_cost_only_writes_only_cost_records(self):
        G = self.G
        ed = E.export_dir("rnd", "H-100", self.root)
        ex = E.Export(ed)
        wd = os.path.join(self.root, "work_cost", "getnext", "rnd", "H-100")
        croot = os.path.join(self.root, "cost_exclusive")
        orig, G.candidates_file_decl = G.candidates_file_decl, lambda ex, split: f"data/rnd/candidates_{split}.parquet"
        self.addCleanup(setattr, G, "candidates_file_decl", orig)
        before = _preds_snapshot()
        out = G.cost_only(ex, ed, wd, "released", ["val", "test"], "cpu", cost_root=croot, epochs=1,
                          log=lambda *a: None)
        self.assertEqual(_preds_snapshot(), before)
        self.assertFalse(os.path.exists(os.path.join(self.root, "preds")))
        for s in ("val", "test"):
            p = os.path.join(croot, "getnext@H-100", "rnd", f"{s}.cost.json")
            self.assertEqual(out[s], p)
            c = json.load(open(p))
            self.assertTrue(c["cost_only"])
            self.assertIsNone(c["colocated"])
            self.assertEqual([st["name"] for st in c["stages"]], ["prepare", "train", f"predict_{s}"])
            self.assertTrue(os.path.exists(os.path.join(ROOT, c["rerun_preds"])))
            self.assertIsNone(c["preds_parity"])
            self.assertEqual(c["epochs"], 1)
        with self.assertRaises(RuntimeError):
            G.cost_only(ex, ed, wd, "released", ["val"], "cpu", cost_root=os.path.join(ROOT, "preds"), epochs=1)
        with self.assertRaises(RuntimeError):
            G.cost_only(ex, ed, os.path.join(ROOT, "preds", "w"), "released", ["val"], "cpu", cost_root=croot,
                        epochs=1)
        self.assertEqual(_preds_snapshot(), before)

    def test_cli_rules(self):
        G = self.G
        for argv in (["train", "--ds", "nyc", "--policy", "H-100", "--cost-only"],
                     ["--ds", "nyc", "--policy", "H-100", "--cost-only", "--epochs", "3"],
                     ["--ds", "nyc", "--policy", "H-100", "--cost-only", "--smoke-users", "0.1"],
                     ["--ds", "nyc", "--policy", "H-100", "--work-dir", "unused_dir"]):
            with self.assertRaises(SystemExit):
                G.main(argv)


class TestGraphFlashbackCostOnly(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tests.test_graph_flashback import _synth_root
        cls.R = _load("gf_run_costonly", os.path.join(GF_DIR, "run.py"), GF_DIR)
        cls.tmp = tempfile.TemporaryDirectory(dir=os.path.join(ROOT, "tmp_tests"))
        cls.root = cls.tmp.name
        _synth_root(cls.root, policies=(("H", 100),))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_cost_only_writes_only_cost_records(self):
        ed = os.path.join(self.root, "exports", "gfsyn", "H-100")
        wr = os.path.join(self.root, "work_cost")
        croot = os.path.join(self.root, "cost_exclusive")
        common = ["--export-dir", ed, "--work-dir", wr, "--cost-only", "--cost-root", croot, "--smoke",
                  "--device", "cpu", "--threads", "2"]
        before = _preds_snapshot()
        self.assertEqual(self.R.main(["prepare"] + common), 0)
        self.assertEqual(self.R.main(["all", "--split", "val", "test"] + common), 0)
        self.assertEqual(_preds_snapshot(), before)
        for s in ("val", "test"):
            c = json.load(open(os.path.join(croot, "graph-flashback@H-100", "gfsyn", f"{s}.cost.json")))
            self.assertTrue(c["cost_only"])
            self.assertIsNone(c["colocated"])
            names = [st["name"] for st in c["stages"]]
            self.assertIn("train", names)
            self.assertEqual(names[-1], f"predict_{s}")
            self.assertTrue(c["rerun_preds"].endswith(f"cost_only_preds/{s}.parquet"))
            self.assertTrue(os.path.exists(os.path.join(ROOT, c["rerun_preds"])))
            self.assertFalse(os.path.exists(os.path.join(ROOT, c["rerun_preds"][:-len(".parquet")] + ".cost.json")))
        checks = [json.loads(ln) for ln in open(os.path.join(wr, "gfsyn", "H-100", "colocation_checks.jsonl"))]
        self.assertEqual([c["cmd"] for c in checks], ["prepare", "all"])
        for bad in (["--cost-root", os.path.join(ROOT, "preds")], ["--work-dir", os.path.join(ROOT, "preds", "w")]):
            argv = ["all", "--split", "val"] + common
            i = argv.index(bad[0])
            argv[i + 1] = bad[1]
            with self.assertRaises(RuntimeError):
                self.R.main(argv)
        self.assertEqual(_preds_snapshot(), before)

    def test_cli_rules(self):
        for argv in (["all", "--ds", "nyc", "--policy", "H-100", "--cost-only", "--out", "x.parquet",
                      "--split", "val"],
                     ["train", "--ds", "nyc", "--policy", "H-100", "--cost-only"],
                     ["all", "--ds", "nyc", "--policy", "H-100", "--cost-only", "--smoke"],
                     ["all", "--ds", "nyc", "--policy", "H-100", "--cost-root", "unused_dir"]):
            with self.assertRaises(SystemExit):
                self.R.main(argv)


class TestCostTablePreference(unittest.TestCase):

    def _rec(self, root, rel, wall, coloc=None, n=10):
        p = os.path.join(root, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        d = dict(n_queries=n, stages=[dict(name="train", wall_s=wall, cpu_core_s=1.0, gpu_s=wall, peak_mem_gb=1.0,
                                           one_time=True),
                                      dict(name="predict", wall_s=1.0, cpu_core_s=1.0, gpu_s=1.0, peak_mem_gb=1.0,
                                           one_time=False)])
        if coloc is not None:
            d["colocated"] = coloc
        with open(p, "w") as f:
            json.dump(d, f)

    def test_preference_and_flags(self):
        from pathlib import Path
        T = _load("make_tables_costtest", os.path.join(ROOT, "paper", "scripts", "make_tables.py"))
        with tempfile.TemporaryDirectory() as td:
            T.ROOT = Path(td)
            for sp in ("val", "test"):
                self._rec(td, f"preds/m@H-100/nyc/{sp}.cost.json", 100.0)
                self._rec(td, f"preds/u@H-100/nyc/{sp}.cost.json", 7.0)
                os.makedirs(os.path.join(td, "results/cost_colocation/m@H-100/nyc"), exist_ok=True)
                with open(os.path.join(td, f"results/cost_colocation/m@H-100/nyc/{sp}.colocation.json"), "w") as f:
                    json.dump({"colocated": True}, f)
            c = T.load_cost("m", "H-100", "nyc")
            self.assertEqual((c["source"], c["colocated"], c["once"]["wall_s"]), ("preds", True, 100.0))
            c = T.load_cost("u", "H-100", "nyc")
            self.assertEqual((c["source"], c["colocated"]), ("preds", None))
            self._rec(td, "results/cost_exclusive/m@H-100/nyc/test.cost.json", 60.0, coloc=False)
            c = T.load_cost("m", "H-100", "nyc")
            self.assertEqual(c["source"], "preds")
            self._rec(td, "results/cost_exclusive/m@H-100/nyc/val.cost.json", 60.0, coloc=False)
            c = T.load_cost("m", "H-100", "nyc")
            self.assertEqual((c["source"], c["colocated"], c["once"]["wall_s"]), ("exclusive", False, 60.0))


if __name__ == "__main__":
    unittest.main()
