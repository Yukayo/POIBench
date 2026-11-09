import os
import tempfile
import unittest

from poibench import canary as C
from poibench import exports as E
from tests import synth
from tests.test_exports import write_synth

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestCanary(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rows, nv = synth.random_rows(n_users=30, seed=5)
        cls.t = synth.build_tables(rows, n_venues=nv)
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = cls.tmp.name
        write_synth(cls.root, "rnd", cls.t, xs=(50, 100))
        for p in ("H-50", "S-50", "H-100"):
            E.export_inputs("rnd", p, cls.root)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _run(self, entry, policy="H-50", split="val"):
        return C.run_canary(entry, "rnd", policy, split, n=40, root=self.root, lint_paths=[])

    def test_clean_builtin_passes(self):
        for pol in ("H-50", "S-50"):
            for split in ("val", "test"):
                rep = self._run("builtin:prev_then_pop", pol, split)
                self.assertEqual(rep["result"], "PASS", (pol, split, rep))
                self.assertTrue(all(rep["exporter_invariance"].values()))
                self.assertGreater(rep["perturbation"]["n_future_replaced"], 0)
        rep = self._run("builtin:prev_then_pop", "H-50")
        self.assertGreater(rep["perturbation"]["n_hidden_replaced"], 0)
        self.assertTrue(os.path.exists(os.path.join(self.root, "logs", "canary",
                                                    "builtin_prev_then_pop__rnd__H-50__val.json")))
        left = [d for d in os.listdir(os.path.join(self.root, "work", "canary"))
                if d.startswith("builtin_prev_then_pop__rnd__H-50__val__")
                and os.path.exists(os.path.join(self.root, "work", "canary", d, "export_perturbed", "MANIFEST.json"))]
        self.assertEqual(left, [])

    def test_leaky_builtin_fails(self):
        rep = self._run("builtin:leaky_sibling")
        self.assertEqual(rep["result"], "FAIL")
        self.assertGreater(rep["n_mismatch"]["ranked_venue_ids"], 0)
        self.assertGreater(rep["s_targets_in_other_contexts"], 0)
        self.assertEqual(rep["s_target_copies_changed"] > 0, True)
        self.assertTrue(all(rep["exporter_invariance"].values()))

    def test_reference_adapter_passes(self):
        entry = os.path.join(ROOT, "methods", "export_hybrid", "run.py") + ":predict"
        for pol in ("H-50", "S-50", "H-100"):
            rep = C.run_canary(entry, "rnd", pol, "val", n=40, root=self.root,
                               lint_paths=[os.path.join(ROOT, "methods", "export_hybrid")])
            self.assertEqual(rep["result"], "PASS", rep)
            self.assertEqual(rep["lint"], [])
            self.assertIn("cand_ranked_venue_ids", rep["n_mismatch"])

    def test_cmd_entry(self):
        py = os.path.join(ROOT, "tests", "_canary_cmd_entry.py")
        entry = f"cmd:{__import__('sys').executable} -I {py} {{export_dir}} {{split}} {{query_ids}} {{out}}"
        rep = C.run_canary(entry, "rnd", "H-50", "val", n=20, root=self.root, lint_paths=[], tag="cmd")
        self.assertEqual(rep["result"], "PASS", rep)

    def test_stale_export_refused(self):
        with self.assertRaises(RuntimeError):
            C.run_canary("builtin:prev_then_pop", "rnd", "S-100", "val", n=5, root=self.root)


class TestLint(unittest.TestCase):
    def test_lint_flags_data_reads(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "run.py")
            with open(p, "w") as f:
                f.write('import pandas as pd\n'
                        'ck = pd.read_parquet(os.path.join(ROOT, "data", ds, "checkins.parquet"))\n'
                        'q = load_queries(ds, "val")\n'
                        'y = q["target_venue_id"]\n'
                        'from poibench import data as D\n'
                        'x = 1  # canary-allow: reads nothing, test line with target_venue_id\n'
                        'e = Export(export_dir)\n'
                        'qi = e.queries("val")  # query_inputs_val.parquet is fine\n')
            hits = C.lint_sources([d])
            self.assertEqual(sorted(h[1] for h in hits), [2, 3, 4, 5])

    def test_reference_adapter_is_clean_and_baselines_are_not(self):
        self.assertEqual(C.lint_sources([os.path.join(ROOT, "methods", "export_hybrid")]), [])
        self.assertTrue(C.lint_sources([os.path.join(ROOT, "scripts", "run_simple.py")]))


if __name__ == "__main__":
    unittest.main()
