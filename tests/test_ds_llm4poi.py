import importlib.util
import os
import random
import tempfile
import unittest

import numpy as np
import pandas as pd

from methods.ds_llm4poi import llm as L
from methods.ds_llm4poi import parse as PA
from methods.ds_llm4poi import prompts as PR
from poibench import canary as C
from poibench import exports as E
from tests import synth
from tests.test_exports import write_synth

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_PY = os.path.join(ROOT, "methods", "ds_llm4poi", "run.py")


def _upstream_module(rel, name):
    d = PR.upstream().path
    sp = importlib.util.spec_from_file_location(name, os.path.join(d, rel))
    m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m)
    return m


def _cfg(num_demo=5, reverse=False):
    return PR.SimpleNamespace(prompting=PR.SimpleNamespace(num_demo=num_demo, random_seed=0, reverse_order=reverse))


class TestFaithfulPieces(unittest.TestCase):
    def test_lcs_equals_authors_calc_lcs(self):
        sim = _upstream_module("src/calc_traj_similarity.py", "dsl_up_sim")
        rng = random.Random(0)
        for _ in range(300):
            a = [rng.randrange(6) for _ in range(rng.randrange(0, 9))]
            b = [rng.randrange(6) for _ in range(rng.randrange(0, 9))]
            want = sim.calc_lcs(pd.DataFrame({"PoiId": a}), pd.DataFrame({"PoiId": b}))
            self.assertEqual(PR.lcs_len(a, b), int(want), (a, b))

    def test_merge_then_authors_selection(self):
        sel = PR.upstream().demo_selection
        traj = [10, 11, 12, 13, 14, 15]
        score = [1, 3, 0, 3, 2, 3]
        merged = PR.merge_similarity(traj, score)
        self.assertEqual(merged, [11, 13, 15, 14, 10, 12])
        cur = pd.DataFrame({"pseudo_session_trajectory_id": [99]})
        hist = pd.DataFrame({"pseudo_session_trajectory_id": traj})
        got = sel.similarity_based_selection(cur, hist, {"99": merged}, _cfg(num_demo=2))
        self.assertEqual(got, [13, 11])
        got = sel.similarity_based_selection(cur, hist, {"99": merged}, _cfg(num_demo=4))
        self.assertEqual(got, [14, 15, 13, 11])
        self.assertEqual(PR.merge_similarity([1, 2, 3], [5.0, 1.0, 3.0], higher_is_more_similar=False), [2, 3, 1])
        got = sel.similarity_based_selection(cur, hist[hist.pseudo_session_trajectory_id != 13], {"99": merged},
                                             _cfg(num_demo=2))
        self.assertEqual(got, [15, 11])

    def test_time_features_equal_authors(self):
        du = PR.upstream().data_utils
        rng = np.random.default_rng(0)
        ts = pd.to_datetime(rng.integers(1_330_000_000, 1_380_000_000, size=3000), unit="s")
        ts = ts.append(pd.DatetimeIndex(["2012-05-01 00:07:00", "2012-05-01 12:59:00", "2012-05-06 23:00:00"]))
        df = du.add_time_features(pd.DataFrame({"UTCTimeOffset": ts}))
        st, dow = PR.time_features(ts)
        self.assertEqual(list(df["start_time"]), list(st))
        self.assertEqual(list(df["day_of_week"]), list(dow))
        self.assertIn("00:07 AM", list(st))
        self.assertIn("00:59 PM", list(st))

    def test_no_target_time_prompt_is_minimal_edit(self):
        P = PR.upstream().prompt_utils
        idx = pd.Index([501, 502, 503])
        ctx = pd.DataFrame({"start_time": ["01:00 PM", "02:00 PM", "03:00 PM"], "day_of_week": ["Monday"] * 3,
                            "PoiId": [7, 8, 9], "PoiCategoryName": ["Bar", "Park", "Gym"]}, index=idx)
        tgt = pd.DataFrame({"start_time": ["09:41 PM"], "day_of_week": ["Tuesday"], "PoiId": [None],
                            "PoiCategoryName": [None]}, index=pd.Index([504]))
        demos = [{"context_stay": "(01:00 PM, Monday, 1, Bar), ", "target_stay": "(02:00 PM, Monday, 2, Park)"}]
        orig = P.create_prompt(demos, P.create_case_dict(pd.concat([ctx, tgt]).astype(object), is_test=True))
        new = P.create_prompt_no_target_time(demos, P.create_case_dict_no_target_time(ctx))
        self.assertIn("(09:41 PM, Tuesday, <next_place_id>, <next_place_category>)", orig)
        self.assertNotIn("09:41 PM", new)
        self.assertNotIn("Tuesday", new)
        self.assertNotIn("temporal information", new)
        item3 = [ln for ln in orig.split("\n") if ln.strip().startswith("3. the temporal information")][0]
        edited = (orig.replace("(09:41 PM, Tuesday, <next_place_id>, <next_place_category>)",
                               "(<next_place_id>, <next_place_category>)")
                  .replace(", while temporal information is provided.", ".").replace(item3 + "\n", ""))
        self.assertEqual(new, edited)
        self.assertIn("<context_current>: (01:00 PM, Monday, 7, Bar), (02:00 PM, Monday, 8, Park), "
                      "(03:00 PM, Monday, 9, Gym), \n", new)


class TestParse(unittest.TestCase):
    CASES = [
        ('```json\n{"place_id": 12, "place_category": "Bar"}\n```', 12, 12),
        ('{"place_id": 12, "place_category": "Bar"}', None, 12),
        ('```json\n{"place_id": "12", "place_category": "Bar"}\n```', None, 12),
        ('```json\n{"place_id": 12}\n```', None, 12),
        ('```json\n{"place_id": 12.0, "place_category": "Bar"}\n```', 12, 12),
        ('```\n{"place_id": 5, "place_category": "Bar"}\n```', None, 5),
        ('The answer is {"place_id": 33, "place_category": "Park"} because ...', None, 33),
        ('"place_id": 41, "place_category": "Pa', None, 41),
        ('I cannot determine the next place.', None, None),
        ('```json\n{"place_id": true, "place_category": "Bar"}\n```', None, None),
        ('```json\n[1, 2]\n```', None, None),
        ('', None, None),
    ]

    def test_cases(self):
        for text, s, l in self.CASES:
            self.assertEqual(PA.parse_strict(text), s, text)
            self.assertEqual(PA.parse_lenient(text), l, text)

    def test_ranked_list(self):
        self.assertEqual(PA.ranked_list(3, [5, 3, 4], 10, 50), ([3, 5, 4], True))
        self.assertEqual(PA.ranked_list(None, [5, 3, 4], 10, 50), ([5, 3, 4], False))
        self.assertEqual(PA.ranked_list(99, [5, 3, 4], 10, 50), ([5, 3, 4], False))
        self.assertEqual(PA.ranked_list(7, list(range(60)), 100, 50)[0][:3], [7, 0, 1])
        self.assertEqual(len(PA.ranked_list(7, list(range(60)), 100, 50)[0]), 50)

    def test_mock_is_deterministic_and_covers_formats(self):
        ps = [f"x\n<context_current>: (01:00 PM, Monday, {i}, Bar), \n<target_current>: (<next_place_id>, "
              f"<next_place_category>)" for i in range(200)]
        a = [L.mock_response(p) for p in ps]
        self.assertEqual(a, [L.mock_response(p) for p in ps])
        self.assertTrue(any(PA.parse_strict(t) is not None for t in a))
        self.assertTrue(any(PA.parse_strict(t) is None and PA.parse_lenient(t) is not None for t in a))
        self.assertTrue(any(PA.parse_lenient(t) is None for t in a))
        self.assertNotEqual(L.cache_key("mock-v1", ps[0]), L.cache_key("mock-v1", ps[1]))


class TestResumableCache(unittest.TestCase):
    def test_crash_then_resume_generates_only_the_rest(self):
        class Flaky(L.MockBackend):
            calls = 0

            def generate(self, prompts):
                Flaky.calls += 1
                if Flaky.calls == 2:
                    raise RuntimeError("simulated preemption")
                return super().generate(prompts)
        with tempfile.TemporaryDirectory() as d:
            ps = [f"p{i} (01:00 PM, Monday, {i}, Bar), " for i in range(25)] + ["p0 (01:00 PM, Monday, 0, Bar), "]
            be = Flaky()
            with self.assertRaises(RuntimeError):
                L.generate_cached(ps, be, L.ResponseCache(be.id, cache_dir=d), 0, chunk=10, log=lambda *a: None)
            cache = L.ResponseCache(be.id, cache_dir=d)
            self.assertEqual(len(cache.mem), 10)
            res, st = L.generate_cached(ps, L.MockBackend(), cache, 0, chunk=10, log=lambda *a: None)
            self.assertEqual((st["n_unique"], st["n_cache_hits"], st["n_generated"]), (25, 10, 15))
            self.assertEqual([r["text"] for r in res], [L.mock_response(p) for p in ps])
            self.assertEqual(len([f for f in os.listdir(d) if f.endswith(".parquet")]), 3)


class TestOnSyntheticExports(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rows, nv = synth.random_rows(n_users=30, seed=5)
        cls.t = synth.build_tables(rows, n_venues=nv)
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = cls.tmp.name
        write_synth(cls.root, "rnd", cls.t, xs=(50, 100))
        for p in ("H-50", "S-50", "H-100"):
            E.export_inputs("rnd", p, cls.root)
        cls.old_env = {k: os.environ.get(k) for k in ("DSLLM4POI_BACKEND", "DSLLM4POI_CACHE_DIR", "DSLLM4POI_CFG")}
        os.environ["DSLLM4POI_BACKEND"] = "mock"
        os.environ["DSLLM4POI_CACHE_DIR"] = os.path.join(cls.root, "llm_cache")

    @classmethod
    def tearDownClass(cls):
        for k, v in cls.old_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        cls.tmp.cleanup()

    def _ex(self, p):
        return E.Export(E.export_dir("rnd", p, self.root))

    def test_bank_and_prompt_content(self):
        ck = self.t["checkins"].sort_values("row_id").reset_index(drop=True)
        for pol in ("H-50", "S-50", "H-100"):
            ex = self._ex(pol)
            train_q = set(ex.queries("train")["query_id"].astype(int))
            vis = set(ex.rows["row_id"].astype(int))
            for cfg in ("lcs", "random"):
                df = PR.build_prompts(ex.path, "val", cfg)
                self.assertEqual(len(df), len(ex.queries("val")))
                b = PR.PromptBuilder(ex.path, "val", cfg)
                for k, r in enumerate(df.itertuples()):
                    self.assertTrue(r.prompt.endswith("<target_current>: (<next_place_id>, <next_place_category>)"))
                    self.assertLessEqual(r.n_demos, PR.NUM_DEMO)
                    self.assertEqual(r.n_demos, min(PR.NUM_DEMO, r.n_bank))
                    bank = b._bank(int(b.qpos[r.query_id]))
                    for t in r.demo_traj_ids:
                        rows = bank[bank.pseudo_session_trajectory_id == t].index.to_numpy()
                        self.assertTrue(len(rows) >= 2 and int(rows[-1]) in train_q and set(rows) <= vis)
                        self.assertTrue((rows < r.query_id).all())
                    prev = int(ck.loc[r.query_id - 1, "venue_id"])
                    tail = r.prompt.split("<context_current>:")[1]
                    self.assertIn(f", {prev}, ", tail.split("<target_current>")[0].strip().split("), (")[-1] + " ")
            if pol == "S-50":
                full = PR.build_prompts(self._ex("H-100").path, "val", "lcs")
                self.assertLessEqual(df["n_bank"].sum(), full["n_bank"].sum())

    def test_parallel_equals_serial(self):
        ex = self._ex("H-100")
        a = PR.build_prompts(ex.path, "val", "lcs", n_jobs=1)
        b = PR.build_prompts(ex.path, "val", "lcs", n_jobs=3)
        self.assertEqual(list(a["prompt_sha256"]), list(b["prompt_sha256"]))

    def test_canary_passes_with_mock(self):
        for cfg in ("lcs", "random"):
            os.environ["DSLLM4POI_CFG"] = cfg
            for pol in ("H-50", "S-50"):
                rep = C.run_canary(f"{RUN_PY}:predict", "rnd", pol, "val", n=40, root=self.root,
                                   lint_paths=[os.path.dirname(RUN_PY)], tag=f"ds_llm4poi_{cfg}")
                self.assertEqual(rep["result"], "PASS", (cfg, pol, rep["n_mismatch"], rep["lint"]))
                self.assertGreater(rep["perturbation"]["n_future_replaced"], 0)
        os.environ["DSLLM4POI_CFG"] = "lcs-tt"
        mod = C.load_entry(f"{RUN_PY}:predict", self.root)
        with self.assertRaises(ValueError):
            mod(E.export_dir("rnd", "H-50", self.root), "val", [1], None)
        os.environ["DSLLM4POI_CFG"] = "lcs"

    def test_canary_catches_a_leaky_prompt(self):
        orig_init, orig_build = PR.PromptBuilder.__init__, PR.PromptBuilder.build

        def init(self, export_dir, *a, **k):
            orig_init(self, export_dir, *a, **k)
            ctx = E.Export(export_dir).context(self.split)
            self._leak = dict(zip(ctx["row_id"].astype(int), ctx["venue_id"].astype(int)))

        def build(self, i):
            rec = orig_build(self, i)
            rec["prompt"] += f" hint={self._leak.get(rec['query_id'], -1)}"
            return rec
        os.environ["DSLLM4POI_CFG"] = "lcs"
        PR.PromptBuilder.__init__, PR.PromptBuilder.build = init, build
        try:
            rep = C.run_canary(f"{RUN_PY}:predict", "rnd", "H-50", "val", n=40, root=self.root, lint_paths=[],
                               tag="ds_llm4poi_leaky")
        finally:
            PR.PromptBuilder.__init__, PR.PromptBuilder.build = orig_init, orig_build
        self.assertEqual(rep["result"], "FAIL")
        self.assertGreater(rep["n_mismatch"]["ranked_venue_ids"], 0)

if __name__ == "__main__":
    unittest.main()
