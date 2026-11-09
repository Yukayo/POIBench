import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from poibench import canary as CN
from poibench import data as D
from poibench import evaluate as EV
from poibench import exports as E
from poibench.cost import CostRecorder, preserve_measured, summarize_cost
from tests import synth
from tests.test_exports import write_synth

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class _Ex:
    manifest = {"ds": "audit", "files": {"x": "1"}, "source_dir": "data/audit",
                "source_sha256": {"candidates_val.parquet": "x"}}
    venues = pd.DataFrame({"venue_id": list(range(10))})

    def candidates(self, split):
        return pd.DataFrame({"query_id": [1, 2], "cand_venue_ids": [list(range(10)), list(range(10))]})


def _stages(d, name):
    return [s for s in d["stages"] if s["name"] == name]


class TestGALLM(unittest.TestCase):
    def setUp(self):
        from methods.gallm import run as ga
        self.ga = ga
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        self.p = [patch.object(ga, "ROOT", self.root), patch.object(ga, "Export", return_value=_Ex()),
                  patch.object(ga, "ensure_src", return_value=self.root),
                  patch.object(ga, "_export_dir", return_value=self.root),
                  patch.object(ga, "prepare", side_effect=self._prepare)]
        for x in self.p:
            x.start()

    def tearDown(self):
        for x in self.p:
            x.stop()
        self.tmp.cleanup()

    def _prepare(self, ex, ds, cfg, wd, splits, smoke_n=0):
        os.makedirs(wd, exist_ok=True)
        for s in ("val", "test"):
            self.ga._write_parquet(pd.DataFrame({"query_id": [1, 2], "prompt": ["p one", "p two"],
                                                 "prompt_sha256": ["a", "b"]}), os.path.join(wd, f"prompts_{s}.parquet"))
        return {"files": {"stage1_train.json": {"n": 1}, "stage2_train.json": {"n": 1}},
                "stage": dict(wall_s=0.0, cpu_core_s=0.0, gpu_s=0.0, peak_mem_gb=0.0, one_time=True)}

    def _main(self):
        self.ga.main(["--ds", "audit", "--policy", "H-100", "--splits", "val", "--backend", "mock"])
        wd = os.path.join(self.root, "work", "gallm", "audit", "H-100", "default_mock")
        return wd, os.path.join(wd, "preds", "ga-llm@H-100", "audit", "val")

    def test_mock_never_writes_preds(self):
        official = os.path.join(self.root, "preds", "ga-llm@H-100", "audit", "val.parquet")
        self.ga._write_parquet(pd.DataFrame({"query_id": [1], "ranked_venue_ids": [[9, 8, 7]]}), official)
        wd, out = self._main()
        self.assertEqual(pd.read_parquet(official).iloc[0]["ranked_venue_ids"].tolist(), [9, 8, 7])
        self.assertFalse(os.path.exists(official.replace(".parquet", ".cost.json")))
        self.assertTrue(os.path.exists(out + ".parquet"))
        c = json.load(open(out + ".cost.json"))
        self.assertEqual(c["backend"], "mock")
        self.assertEqual(c["generation"]["model_fingerprint"], "mock-v1")
        self.assertFalse(os.path.exists(os.path.join(self.root, "work", "gallm", "audit", "H-100", "default")))
        self.assertTrue(all(f.endswith(("_mock_in.parquet", ".jsonl", ".log")) or "_mock" in f
                            for f in os.listdir(os.path.join(wd, "gen"))))

    def test_cache_hit_rerun_keeps_cost(self):
        _, out = self._main()
        first = json.load(open(out + ".cost.json"))
        self.assertEqual(first["execution"]["n_cache_hits"], 0)
        self.assertGreater(_stages(first, "infer")[0]["wall_s"], 0)
        _, out = self._main()
        second = json.load(open(out + ".cost.json"))
        self.assertEqual(second["execution"]["n_cache_hits"], 2)
        self.assertEqual(_stages(second, "infer"), _stages(first, "infer"))
        self.assertEqual(second["reruns"][-1]["n_cache_hits"], 2)
        os.remove(out + ".cost.json")
        _, out = self._main()
        third = json.load(open(out + ".cost.json"))
        ic = _stages(third, "infer_cached")
        self.assertEqual(len(ic), 1)
        self.assertEqual(ic[0]["n_items"], 2)
        ledger = [json.load(open(os.path.join(dp, f))) for dp, _, fs in os.walk(os.path.join(self.root, "work"))
                  for f in fs if os.path.basename(dp) == "runs"]
        self.assertEqual(len(ledger), 1)
        self.assertAlmostEqual(ic[0]["wall_s"], ledger[0]["wall_s"], places=6)
        self.assertGreater(summarize_cost(third, 2)["recurring"]["wall_s"], 0.5 * ledger[0]["wall_s"])

    def test_legacy_cache_entry_and_resumed_answers(self):
        ga = self.ga
        mfp, maxnew = "mock-v1", max(ga.GEN["max_new_tokens"], len(str(9)))
        prompts = pd.DataFrame({"query_id": [1, 2, 3], "prompt": ["a", "b", "c"]})
        cache = ga.ResponseCache(mfp)
        cache.add({ga._cache_key(mfp, 1, "a", maxnew): dict(query_id=1, text="1", n_prompt_tokens=2, gen_s=42.0)})
        wd = os.path.join(self.root, "w")
        os.makedirs(os.path.join(wd, "gen"))
        ga._write_parquet(prompts.iloc[1:].assign(key=[ga._cache_key(mfp, q, p, maxnew) for q, p in
                                                       zip(prompts["query_id"][1:], prompts["prompt"][1:])]),
                          os.path.join(wd, "gen", "t_mock_in.parquet"))
        with open(os.path.join(wd, "gen", "t_mock_out0.jsonl"), "w") as f:
            f.write(json.dumps(dict(query_id=2, text="2", n_prompt_tokens=2, skipped=False, gen_s=7.0)) + "\n")
        info = {}
        res, gst = ga.run_infer(prompts, wd, None, None, "audit", 10, "mock", "t", info=info)
        self.assertEqual(len(info["cached_keys"]), 1)
        self.assertEqual((gst["n_generated"], gst["n_resumed"], gst["resumed_gen_s"]), (2, 1, 7.0))
        cc = ga.cached_answer_costs(["val"] * 3, info, [])
        self.assertAlmostEqual(cc["val"]["wall_s"], 42.0)
        self.assertIn("1 from their stored generation time", cc["val"]["notes"])
        info2 = {}
        ga.run_infer(prompts, wd, None, None, "audit", 10, "mock", "t", info=info2)
        cc2 = ga.cached_answer_costs(["val"] * 3, info2, [])
        lg = ga.ResponseCache(mfp).ledger(info["run"])
        self.assertAlmostEqual(cc2["val"]["wall_s"], 42.0 + lg["wall_s"] + lg["resumed_gen_s"], places=6)
        cc3 = ga.cached_answer_costs(["val"] * 3, info2, [dict(shard=0, nshards=1)])
        self.assertEqual(cc3, {})

    def test_predict_cfg_variable(self):
        with patch.dict(os.environ, {"GALLM_CFG": "nope", "GALLM_BACKEND": "mock"}):
            with self.assertRaises(ValueError):
                self.ga.predict(self.root, "val", [1])
        with patch.dict(os.environ, {"GALLM_CFG": "tt", "GALLM_BACKEND": "hf"}), \
                patch.object(self.ga.PR, "query_prompts", return_value=pd.DataFrame()), \
                patch.object(self.ga.PR, "load_target_time", return_value=None):
            os.environ.pop("GALLM_STAGE2", None)
            with self.assertRaises(ValueError):
                self.ga.predict(self.root, "val", [1])
        seen = []

        def cap(ex, split, qids, cfg, tt=None):
            seen.append((cfg, tt))
            raise StopIteration

        with patch.dict(os.environ, {"GALLM_CFG": "tt", "GALLM_BACKEND": "mock"}), \
                patch.object(self.ga.PR, "query_prompts", side_effect=cap), \
                patch.object(self.ga.PR, "load_target_time", return_value="TT"):
            with self.assertRaises(StopIteration):
                self.ga.predict(self.root, "val", [1])
        self.assertEqual(seen, [("tt", "TT")])


class TestLLM4POI(unittest.TestCase):
    def setUp(self):
        from methods.llm4poi import run as ll
        self.ll = ll
        self.tmp = tempfile.TemporaryDirectory()
        self.p = patch.dict(os.environ, {"LLM4POI_CACHE_DIR": self.tmp.name})
        self.p.start()

    def tearDown(self):
        self.p.stop()
        self.tmp.cleanup()

    def _do_predict(self, out):
        ll, t = self.ll, self.tmp.name
        a = SimpleNamespace(backend="vllm", cfg="default", ds="audit", policy="H-100", splits=["val"])
        done = dict(final_ckpt=t, model_fingerprint="fp", train_wall_s=1.0, train_cpu_core_s=1.0, train_gpu_s=1.0,
                    sessions=[{"n_gpus": 1}], global_step=1)
        prompts = {"val": pd.DataFrame({"query_id": [1, 2], "prompt": ["pa", "pb"], "prompt_sha256": ["a", "b"]})}
        pmeta = {"val": {"stage": dict(wall_s=0.0, cpu_core_s=0.0, gpu_s=0.0, peak_mem_gb=0.0, one_time=False),
                         "stats": {}}}
        with patch.object(ll, "base_model", return_value={"revision": "x"}), \
                patch.object(ll, "effective_sampling", return_value=dict(ll.SAMPLING)):
            ll.do_predict(a, _Ex(), t, done, prompts, pmeta, out)
        return json.load(open(os.path.join(out, "llm4poi@H-100", "audit", "val.cost.json")))

    def _put(self, prompt, gen_s):
        ll = self.ll
        ll.ResponseCache("fp").put([dict(key=ll.cache_key("fp", prompt, dict(ll.SAMPLING)), text="1", n_prompt_tokens=2,
                                         n_output_tokens=1, finish_reason="stop", gen_s=gen_s)])

    def test_cached_answers_keep_their_cost(self):
        self._put("pa", 42.0)
        self._put("pb", 8.0)
        c = self._do_predict(os.path.join(self.tmp.name, "preds"))
        self.assertEqual(c["execution"]["n_cache_hits"], 2)
        self.assertEqual(c["execution"]["n_generated"], 0)
        ic = _stages(c, "infer_cached")
        self.assertEqual(len(ic), 1)
        self.assertAlmostEqual(ic[0]["wall_s"], 50.0)
        self.assertAlmostEqual(ic[0]["gpu_s"], 50.0)
        self.assertGreaterEqual(summarize_cost(c, 2)["recurring"]["gpu_s"], 50.0)

    def test_rerun_never_replaces_measured_cost(self):
        self._put("pa", 42.0)
        self._put("pb", 8.0)
        out = os.path.join(self.tmp.name, "preds")
        path = os.path.join(out, "llm4poi@H-100", "audit", "val.cost.json")
        c = self._do_predict(out)
        c["execution"]["n_cache_hits"] = 0
        c["stages"] = [s for s in c["stages"] if s["one_time"]] + [dict(name="infer", wall_s=100.0, cpu_core_s=1.0,
                                                                        gpu_s=200.0, peak_mem_gb=0.0, one_time=False)]
        json.dump(c, open(path, "w"))
        c2 = self._do_predict(out)
        self.assertEqual([s["wall_s"] for s in c2["stages"] if not s["one_time"]], [100.0])
        self.assertEqual(c2["per_query_stages_from"]["n_cache_hits"], 0)
        self.assertEqual(c2["reruns"][-1]["n_cache_hits"], 2)

    def test_predict_cfg_variable(self):
        with patch.dict(os.environ, {"LLM4POI_CFG": "tt", "LLM4POI_BACKEND": "vllm"}), \
                patch.object(self.ll, "Export", return_value=_Ex()), \
                patch.object(self.ll.PR, "query_prompts", return_value=pd.DataFrame()), \
                patch.object(self.ll.PR, "load_target_time", return_value=None):
            os.environ.pop("LLM4POI_CKPT", None)
            with self.assertRaises(ValueError):
                self.ll.predict("x", "val", [1])
        with patch.dict(os.environ, {"LLM4POI_CFG": "nope"}):
            with self.assertRaises(ValueError):
                self.ll.predict("x", "val", [1])


class TestPreserveMeasured(unittest.TestCase):
    def test_rules(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "c.json")
            old = dict(model_fingerprint="m", n_queries=2, execution=dict(n_cache_hits=0), created_utc="t0",
                       stages=[dict(name="train", one_time=True, wall_s=5), dict(name="infer", one_time=False, wall_s=9)])
            json.dump(old, open(p, "w"))
            new = dict(old, execution=dict(n_cache_hits=2), created_utc="t1",
                       stages=[dict(name="train", one_time=True, wall_s=6), dict(name="infer", one_time=False, wall_s=0)])
            kept = preserve_measured(p, new, ("model_fingerprint", "n_queries"))
            self.assertEqual([s["wall_s"] for s in kept["stages"]], [6, 9])
            self.assertIs(preserve_measured(p, dict(new, model_fingerprint="other"), ("model_fingerprint",)).get(
                "per_query_stages_from"), None)
            self.assertIs(preserve_measured(p, dict(new, execution=dict(n_cache_hits=0)), ("model_fingerprint",))
                          .get("per_query_stages_from"), None)


class TestEvaluatorRefusesMock(unittest.TestCase):
    def test_refuses(self):
        with tempfile.TemporaryDirectory() as root:
            t = synth.hand_tables()
            write_synth(root, "toy", t)
            q = t["queries"]["val"]
            preds = pd.DataFrame({"query_id": q["query_id"], "ranked_venue_ids": [[0, 1, 2]] * len(q)})
            for m, rec in (("real@H-100", {"backend": "vllm|x"}), ("mocky@H-100", {"backend": "mock-v1"}),
                           ("gamock@H-100", {"generation": {"model_fingerprint": "mock-v1"}})):
                d = os.path.join(root, "preds", m, "toy")
                os.makedirs(d)
                preds.to_parquet(os.path.join(d, "val.parquet"), index=False)
                json.dump(rec, open(os.path.join(d, "val.cost.json"), "w"))
            self.assertEqual(EV.mock_artifacts(root, "toy", "val", ["real@H-100", "mocky@H-100", "gamock@H-100"]),
                             ["mocky@H-100", "gamock@H-100"])
            res = EV.evaluate_files(root, "toy", "val", ["real@H-100"], B=0, pairs="none", e2e_only=True)
            self.assertGreater(len(res["results"]), 0)
            for m in ("mocky@H-100", "gamock@H-100"):
                with self.assertRaisesRegex(ValueError, "mock"):
                    EV.evaluate_files(root, "toy", "val", ["real@H-100", m], B=0, pairs="none", e2e_only=True)


class TestCanaryConfig(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rows, nv = synth.random_rows(n_users=20, seed=3)
        cls.t = synth.build_tables(rows, n_venues=nv)
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = cls.tmp.name
        write_synth(cls.root, "rnd", cls.t, xs=(100,))
        E.export_inputs("rnd", "H-100", cls.root)
        cls.entry = os.path.join(ROOT, "tests", "_canary_cfg_entry.py") + ":predict"

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _run(self, env, expect):
        with patch.dict(os.environ, env):
            return CN.run_canary(self.entry, "rnd", "H-100", "val", n=20, root=self.root, write_report=False,
                                 lint_paths=[], expect_config=expect)

    def test_expected_config_used(self):
        rep = self._run({"CFG_ENTRY_CFG": "random"}, {"cfg": "random"})
        self.assertEqual((rep["result"], rep["config_check"]), ("PASS", "PASS"))
        self.assertEqual(rep["adapter_config"], {"cfg": "random"})

    def test_default_config_fails_variant_gate(self):
        os.environ.pop("CFG_ENTRY_CFG", None)
        rep = self._run({}, {"cfg": "random"})
        self.assertEqual(rep["result"], "FAIL")
        self.assertTrue(rep["config_check"].startswith("FAIL"))

    def test_unreported_config_fails(self):
        rep = self._run({"CFG_ENTRY_CFG": "random", "CFG_ENTRY_SILENT": "1"}, {"cfg": "random"})
        self.assertEqual(rep["result"], "FAIL")
        self.assertIn("does not report", rep["config_check"])
        rep = self._run({"CFG_ENTRY_SILENT": "1"}, None)
        self.assertEqual((rep["result"], rep["config_check"]), ("PASS", "not requested"))

    def test_gate_variable_table(self):
        sh = os.path.join(ROOT, "scripts", "gpu_jobs", "canary_gate_variant.sh")
        r = subprocess.run(["bash", sh, "mmpoi", "x"], env=dict(os.environ, GATE_PY=sys.executable),
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 2)
        self.assertIn("refusing", r.stderr)
        src = open(sh).read()
        for m, var in (("ds_llm4poi", "DSLLM4POI_CFG"), ("llm4poi", "LLM4POI_CFG"), ("gallm", "GALLM_CFG"),
                       ("getnext", "GETNEXT_CFG"), ("gnpr_sid", "GNPR_SID_CFG")):
            self.assertIn(f"{m}) var={var} ;;", src)
        self.assertIn("expect_config={'cfg': '$cfg'}", src)

    def test_ds_llm4poi_reports_random(self):
        from methods.ds_llm4poi import prompts as DP
        try:
            DP.upstream()
        except Exception as ex:
            self.skipTest(f"DS-LLM4POI upstream copy not available: {ex}")
        from methods.ds_llm4poi import run as ds
        ex_dir = E.export_dir("rnd", "H-100", self.root)
        qids = E.Export(ex_dir).queries("val")["query_id"].tolist()[:5]
        with tempfile.TemporaryDirectory() as cache, \
                patch.dict(os.environ, {"DSLLM4POI_CFG": "random", "DSLLM4POI_BACKEND": "mock",
                                        "DSLLM4POI_CACHE_DIR": cache, "DSLLM4POI_NJOBS": "1"}):
            out = ds.predict(ex_dir, "val", qids)
        self.assertEqual(out.attrs["adapter_config"]["cfg"], "random")
        self.assertEqual(out.attrs["adapter_config"]["demo_selection"], "random")
        with patch.dict(os.environ, {"DSLLM4POI_CFG": "bogus"}):
            with self.assertRaises(ValueError):
                ds.predict(ex_dir, "val", qids)


LOAD_S = 1.0


def _no_overlap(test, rec, elapsed):
    names = [s["name"] for s in rec.stages]
    test.assertEqual(names[:2], ["model_load", "infer"])
    load, infer = rec.stages[0], rec.stages[1]
    test.assertGreaterEqual(load["wall_s"], LOAD_S)
    test.assertLess(infer["wall_s"], 0.8 * LOAD_S)
    total = summarize_cost(rec.to_dict(), 1)
    test.assertLessEqual(total["one_time"]["wall_s"] + total["recurring"]["wall_s"], elapsed + 0.01)


class TestModelLoadOutsideInfer(unittest.TestCase):
    def test_ds_llm4poi(self):
        from methods.ds_llm4poi import llm as L
        from methods.ds_llm4poi import run as ds

        class Slow(L.MockBackend):
            def load(self):
                time.sleep(LOAD_S)
                return super().load()

        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ, {"DSLLM4POI_CACHE_DIR": d}), \
                patch.object(ds, "get_backend", return_value=Slow()):
            rec = CostRecorder(sample_interval=0.01)
            t0 = time.perf_counter()
            res, gst, _ = ds.run_generation(pd.DataFrame({"prompt": ["x", "y"]}), "mock", "t", rec)
            _no_overlap(self, rec, time.perf_counter() - t0)
            self.assertTrue(gst["loaded"])
            self.assertEqual(len(res), 2)

    def test_qwen25_vl_ref(self):
        from methods.qwen25_vl_ref import run as R
        from methods.qwen25_vl_ref import vlm as V

        class Slow(V.MockBackend):
            def load(self):
                time.sleep(LOAD_S)
                return super().load()

        reqs = [[("user", [("text", f"rank {i}")])] for i in range(3)]
        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ, {"QWENVL_CACHE_DIR": d}), \
                patch.object(V, "get_backend", return_value=Slow()):
            rec = CostRecorder(sample_interval=0.01)
            t0 = time.perf_counter()
            res, _ = R._generate(reqs, "mock", "t", rec)
            _no_overlap(self, rec, time.perf_counter() - t0)
            self.assertEqual(len(res), 3)

    def test_gnpr_sid(self):
        from methods.gnpr_sid import run as G
        cands = list(range(10))

        class Ex:
            manifest = {"ds": "audit"}

            def candidates(self, split):
                return pd.DataFrame({"query_id": [1, 2], "cand_venue_ids": [cands, cands]})

        class Gen:
            def run(self, prompts, cand_lists, log=None):
                return [([(c[0], -1.0)], [0.0] * len(c)) for c in cand_lists]

        def fake_get_generator(adapter_dir, pid2code, rec):
            with rec.stage("model_load", one_time=True, n_gpus=0):
                time.sleep(LOAD_S)
            return Gen()

        prompts = pd.DataFrame({"query_id": [1, 2], "prompt": ["a", "b"], "n_items": [1, 1], "n_dropped_no_sid": [0, 0]})
        tr = dict(pid2code={}, meta=dict(sft_samples_sha256="s", sid_stats={}), adapter_dir="x")
        with tempfile.TemporaryDirectory() as wd, patch.object(G, "get_generator", side_effect=fake_get_generator), \
                patch.object(G.D, "eval_prompts", return_value=prompts), patch.object(G, "L_infer_batch", return_value=1):
            rec = CostRecorder(sample_interval=0.01)
            t0 = time.perf_counter()
            out, _ = G.predict_split(Ex(), "val", [1, 2], tr, wd, rec, None)
            _no_overlap(self, rec, time.perf_counter() - t0)
            self.assertEqual(len(out), 2)
            rec2 = CostRecorder(sample_interval=0.01)
            G.predict_split(Ex(), "val", [1, 2], tr, wd, rec2, None)
            self.assertEqual([s["name"] for s in rec2.stages], ["infer", "infer_cached"])


class TestRawRoot(unittest.TestCase):
    def test_every_raw_file_under_explicit_root(self):
        for ds in D.DATASETS:
            for key in D.RAW_FILES[ds]:
                self.assertFalse(os.path.isabs(key), key)
                p = D.raw_path(key, "/ext/raw")
                self.assertTrue(p.startswith("/ext/raw/"), p)
                self.assertNotIn("/data_raw/", p.replace("/ext/raw/", "/"))

    def test_default_layout_unchanged(self):
        if os.environ.get("POIBENCH_RAW_ROOT"):
            self.skipTest("POIBENCH_RAW_ROOT is set")
        old_abs = {"ca": [D.GOWALLA_SNAP, D.GOWALLA_SPOTS,
                          os.path.join(D.ROOT, "data_raw", "llm4poi_release", "ca", "us_state_polygon_json.json")],
                   "sin": [os.path.join(D.ROOT, "data_raw", "foursquare_tist2015", "extracted", f)
                           for f in ("dataset_TIST2015_Checkins.txt", "dataset_TIST2015_POIs.txt")]
                   + [os.path.join(D.ROOT, "data_raw", "foursquare_tist2015", "dataset_TIST2015.zip")]}
        for ds, files in old_abs.items():
            got = [D.raw_path(k) for k in D.RAW_FILES[ds]]
            want = [f if os.path.isabs(f) else os.path.join(D.RAW_ROOT, f) for f in files]
            self.assertEqual(got, want)
            keys = [os.path.relpath(f, D.ROOT) if os.path.isabs(f) else f for f in files]
            self.assertEqual(D.RAW_FILES[ds], keys)
        self.assertEqual(D.raw_path(D.RAW_FILES["nyc"][0]), os.path.join(D.RAW_ROOT, D.RAW_FILES["nyc"][0]))
        self.assertEqual(D.CA_POLYGONS, os.path.join(D.ROOT, "data_raw", "llm4poi_release", "ca",
                                                     "us_state_polygon_json.json"))

    def test_meta_json_keys_match(self):
        src = {"ca": ("polygons", D.CA_POLYGONS_KEY), "sin": ("dir", D.TIST_KEY), "gowalla": ("tz_table", D.GOWALLA_TZ_KEY),
               "foursquare_global": ("dir", D.WWW2019_KEY)}
        n = 0
        for ds in D.DATASETS:
            mp = os.path.join(D.ROOT, "data", ds, "meta.json")
            if not os.path.exists(mp):
                continue
            meta = json.load(open(mp))
            self.assertEqual(list(meta["raw_files"]), D.RAW_FILES[ds])
            if ds in src:
                self.assertEqual(meta["source"][src[ds][0]], src[ds][1])
            n += 1
        if not n:
            self.skipTest("no built dataset under data/")

    def test_loaders_use_raw_root(self):
        seen = []

        def fake_csv(path, *a, **k):
            seen.append(path)
            raise StopIteration

        with patch.object(D.pd, "read_csv", side_effect=fake_csv):
            for ds in ("sin", "foursquare_global", "ca"):
                with self.assertRaises(StopIteration):
                    D.load_raw(ds, "/ext/raw")
        self.assertTrue(seen and all(str(p).startswith("/ext/raw/") for p in seen), seen)


if __name__ == "__main__":
    unittest.main()
