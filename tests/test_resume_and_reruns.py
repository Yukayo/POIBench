import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from poibench.cost import preserve_measured, summarize_cost

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def mock_text(prompt: str) -> str:
    return str(int(hashlib.sha256(prompt.encode()).hexdigest()[:8], 16) % 997)


class _Ex:
    manifest = {"ds": "audit", "files": {"x": "1"}, "source_dir": "data/audit",
                "source_sha256": {"candidates_val.parquet": "x"}}
    venues = pd.DataFrame({"venue_id": list(range(10))})

    def candidates(self, split):
        return pd.DataFrame({"query_id": [1, 2], "cand_venue_ids": [list(range(10)), list(range(10))]})


class _GA(unittest.TestCase):
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


class TestResumeKey(_GA):
    def _infer(self, wd, prompt, tag="main_val"):
        info = {}
        res, st = self.ga.run_infer(pd.DataFrame({"query_id": [1], "prompt": [prompt]}), wd, None, None, "audit", 1000,
                                    "mock", tag, info=info)
        return res[0]["text"], st, info

    def test_reviewer_repro_changed_prompt_is_regenerated(self):
        wd = os.path.join(self.root, "workdir")
        t1, st1, i1 = self._infer(wd, "first prompt")
        self.assertEqual(t1, mock_text("first prompt"))
        t2, st2, i2 = self._infer(wd, "changed prompt")
        self.assertEqual(st2["n_cached"], 0)
        self.assertEqual(st2["n_resumed"], 0)
        self.assertEqual(t2, mock_text("changed prompt"))
        self.assertNotEqual(t2, t1)
        cache = self.ga.ResponseCache("mock-v1")
        self.assertEqual(cache.d[i2["keys"][0]]["text"], mock_text("changed prompt"))
        self.assertEqual(cache.d[i1["keys"][0]]["text"], mock_text("first prompt"))
        self.assertNotIn("key", cache.d[i2["keys"][0]])

    def test_same_request_still_resumed(self):
        ga, wd = self.ga, os.path.join(self.root, "w")
        os.makedirs(os.path.join(wd, "gen"))
        key = ga._cache_key("mock-v1", 1, "p", max(ga.GEN["max_new_tokens"], 3))
        with open(os.path.join(wd, "gen", "t_mock_out0.jsonl"), "w") as f:
            f.write(json.dumps(dict(query_id=1, text="SENTINEL", n_prompt_tokens=1, skipped=False, gen_s=3.0, key=key))
                    + "\n")
        text, st, _ = self._infer(wd, "p", tag="t")
        self.assertEqual((text, st["n_resumed"], st["resumed_gen_s"]), ("SENTINEL", 1, 3.0))

    def test_other_model_key_is_regenerated(self):
        ga, wd = self.ga, os.path.join(self.root, "w")
        os.makedirs(os.path.join(wd, "gen"))
        other = ga._cache_key("other-model", 1, "p", max(ga.GEN["max_new_tokens"], 3))
        with open(os.path.join(wd, "gen", "t_mock_out0.jsonl"), "w") as f:
            f.write(json.dumps(dict(query_id=1, text="STALE", n_prompt_tokens=1, skipped=False, gen_s=3.0, key=other))
                    + "\n")
        text, st, _ = self._infer(wd, "p", tag="t")
        self.assertEqual((text, st["n_resumed"]), (mock_text("p"), 0))

    def _legacy(self, wd, in_prompt, in_key=None):
        os.makedirs(os.path.join(wd, "gen"), exist_ok=True)
        inp = {"query_id": [1], "prompt": [in_prompt]}
        if in_key is not None:
            inp["key"] = [in_key]
        self.ga._write_parquet(pd.DataFrame(inp), os.path.join(wd, "gen", "t_mock_in.parquet"))
        with open(os.path.join(wd, "gen", "t_mock_out0.jsonl"), "w") as f:
            f.write(json.dumps(dict(kind="meta", load_s=0.0, trainable_keys_matched=None, peak_gpu_mem_gb=None)) + "\n")
            f.write(json.dumps(dict(query_id=1, text="LEGACY", n_prompt_tokens=1, skipped=False, gen_s=2.0)) + "\n")

    def test_legacy_record_same_prompt_unkeyed_input_regenerated(self):
        wd = os.path.join(self.root, "w")
        self._legacy(wd, "p")
        text, st, _ = self._infer(wd, "p", tag="t")
        self.assertEqual((text, st["n_resumed"]), (mock_text("p"), 0))

    def test_legacy_record_same_key_in_input_resumed(self):
        ga, wd = self.ga, os.path.join(self.root, "w")
        self._legacy(wd, "p", in_key=ga._cache_key("mock-v1", 1, "p", max(ga.GEN["max_new_tokens"], 3)))
        text, st, _ = self._infer(wd, "p", tag="t")
        self.assertEqual((text, st["n_resumed"], st["resumed_gen_s"]), ("LEGACY", 1, 2.0))

    def test_legacy_record_other_key_in_input_regenerated(self):
        ga, wd = self.ga, os.path.join(self.root, "w")
        self._legacy(wd, "p", in_key=ga._cache_key("other-model", 1, "p", max(ga.GEN["max_new_tokens"], 3)))
        text, st, _ = self._infer(wd, "p", tag="t")
        self.assertEqual((text, st["n_resumed"]), (mock_text("p"), 0))

    def test_legacy_record_changed_prompt_regenerated(self):
        wd = os.path.join(self.root, "w")
        self._legacy(wd, "old prompt")
        text, st, _ = self._infer(wd, "new prompt", tag="t")
        self.assertEqual((text, st["n_resumed"]), (mock_text("new prompt"), 0))

    def test_legacy_record_without_input_file_regenerated(self):
        wd = os.path.join(self.root, "w")
        self._legacy(wd, "p")
        os.remove(os.path.join(wd, "gen", "t_mock_in.parquet"))
        text, st, _ = self._infer(wd, "p", tag="t")
        self.assertEqual((text, st["n_resumed"]), (mock_text("p"), 0))

    def test_legacy_hf_checks_weights_and_mock_answers(self):
        ga, wd = self.ga, os.path.join(self.root, "w")
        os.makedirs(os.path.join(wd, "gen"))
        s2 = os.path.join(self.root, "stage2")
        os.makedirs(s2)
        ga._write_parquet(pd.DataFrame({"query_id": [1, 2], "prompt": ["a", "b"], "key": ["k1", "k2"]}),
                          os.path.join(wd, "gen", "t_in.parquet"))
        out = os.path.join(wd, "gen", "t_out0.jsonl")
        todo = pd.DataFrame({"query_id": [1, 2], "prompt": ["a", "b"], "key": ["k1", "k2"]})

        def write():
            with open(out, "w") as f:
                f.write(json.dumps(dict(query_id=1, text="5", n_prompt_tokens=1, skipped=False, gen_s=0.7)) + "\n")
                f.write(json.dumps(dict(query_id=2, text="9", n_prompt_tokens=1, skipped=False, gen_s=0.0)) + "\n")

        open(os.path.join(s2, "trainable_params.bin"), "w").close()
        os.utime(os.path.join(s2, "trainable_params.bin"), (time.time() - 100, time.time() - 100))
        write()
        kept = ga.vet_resume(wd, "t", todo, 1, "hf", s2)
        self.assertEqual(kept, {1: 0.7})
        self.assertEqual([json.loads(x).get("key") for x in open(out)], ["k1"])
        write()
        os.utime(out, (time.time() - 1000, time.time() - 1000))
        self.assertEqual(ga.vet_resume(wd, "t", todo, 1, "hf", s2), {})
        self.assertEqual(open(out).read(), "")

    def test_infer_worker_standalone(self):
        wd = os.path.join(self.root, "w")
        os.makedirs(wd)
        inp, out = os.path.join(wd, "in.parquet"), os.path.join(wd, "out.jsonl")
        stale = json.dumps(dict(query_id=1, text="STALE", n_prompt_tokens=1, skipped=False, gen_s=0.0, key="old")) + "\n"

        def run(df):
            df.to_parquet(inp, index=False)
            subprocess.run([sys.executable, "-I", os.path.join(ROOT, "methods", "gallm", "infer.py"), "--src", wd,
                            "--inp", inp, "--out", out, "--backend", "mock"], check=True)
            return [json.loads(x) for x in open(out) if '"kind"' not in x]

        open(out, "w").write(stale)
        recs = run(pd.DataFrame({"query_id": [1], "prompt": ["p"], "key": ["new"]}))
        self.assertEqual([(r["text"], r["key"]) for r in recs], [(mock_text("p"), "new")])
        open(out, "w").write(stale)
        recs = run(pd.DataFrame({"query_id": [1], "prompt": ["p"]}))
        self.assertEqual([r["text"] for r in recs], ["STALE"])


class TestLegacyLineage(unittest.TestCase):

    def setUp(self):
        from methods.gallm import run as ga
        self.ga = ga
        self.tmp = tempfile.TemporaryDirectory()
        r = self.tmp.name
        self.work, self.stage1, self.stage2 = (os.path.join(r, "work"), os.path.join(r, "checkpoint", "stage1"),
                                               os.path.join(r, "checkpoint", "stage2"))
        for d in (os.path.join(self.work, "gen"), self.stage1, self.stage2):
            os.makedirs(d)
        adapter = os.path.join(self.stage2, "adapter_model.safetensors")
        open(adapter, "wb").write(b"stage2 adapter fixture")
        os.utime(adapter, (time.time() - 120, time.time() - 120))
        self.done = os.path.join(self.stage1, "DONE.json")
        self.out = os.path.join(self.work, "gen", "main_val_out0.jsonl")

    def tearDown(self):
        self.tmp.cleanup()

    def _attempt(self, encoder, input_key: bool):
        ga = self.ga
        json.dump({"encoder_sha256": encoder}, open(self.done, "w"))
        key = ga._cache_key(ga.model_fingerprint(self.stage2), 1, "same prompt", 4)
        inp = {"query_id": [1], "prompt": ["same prompt"]}
        if input_key:
            inp["key"] = [key]
        pd.DataFrame(inp).to_parquet(os.path.join(self.work, "gen", "main_val_in.parquet"), index=False)
        open(self.out, "w").write(json.dumps({"query_id": 1, "text": "OLD_MODEL_ANSWER", "n_prompt_tokens": 2,
                                              "skipped": False, "gen_s": 0.7}) + "\n")
        return key

    def _vet(self):
        ga = self.ga
        key = ga._cache_key(ga.model_fingerprint(self.stage2), 1, "same prompt", 4)
        todo = pd.DataFrame({"query_id": [1], "prompt": ["same prompt"], "key": [key]})
        return ga.vet_resume(self.work, "main_val", todo, 1, "hf", self.stage2), todo, key

    def test_reviewer_repro_stage1_change_regenerated(self):
        old_key = self._attempt("old-stage1-encoder", input_key=False)
        json.dump({"encoder_sha256": "new-stage1-encoder"}, open(self.done, "w"))
        kept, todo, new_key = self._vet()
        self.assertNotEqual(old_key, new_key)
        self.assertEqual(kept, {})
        self.assertEqual([x for x in open(self.out) if '"kind"' not in x], [])
        todo.to_parquet(os.path.join(self.work, "gen", "main_val_in.parquet"), index=False)
        subprocess.run([sys.executable, "-I", os.path.join(ROOT, "methods", "gallm", "infer.py"), "--src", self.work,
                        "--inp", os.path.join(self.work, "gen", "main_val_in.parquet"), "--out", self.out,
                        "--backend", "mock"], check=True)
        final = [json.loads(x) for x in open(self.out) if '"kind"' not in x]
        self.assertEqual([(r["text"], r["key"]) for r in final], [(mock_text("same prompt"), new_key)])

    def test_unkeyed_input_same_lineage_still_regenerated(self):
        self._attempt("enc", input_key=False)
        self.assertEqual(self._vet()[0], {})

    def test_keyed_input_stage1_change_regenerated(self):
        self._attempt("old-stage1-encoder", input_key=True)
        json.dump({"encoder_sha256": "new-stage1-encoder"}, open(self.done, "w"))
        self.assertEqual(self._vet()[0], {})

    def test_keyed_input_stage2_change_regenerated(self):
        self._attempt("enc", input_key=True)
        open(os.path.join(self.stage2, "trainable_params.bin"), "wb").write(b"new params")
        os.utime(os.path.join(self.stage2, "trainable_params.bin"), (time.time() - 60, time.time() - 60))
        self.assertEqual(self._vet()[0], {})

    def test_keyed_input_same_lineage_resumed_and_stamped(self):
        self._attempt("enc", input_key=True)
        kept, _, key = self._vet()
        self.assertEqual(kept, {1: 0.7})
        self.assertEqual([json.loads(x)["key"] for x in open(self.out)], [key])


class TestShardRerun(_GA):
    ARGS = ["--ds", "audit", "--policy", "H-100", "--splits", "val", "--backend", "mock"]

    def _paths(self):
        wd = os.path.join(self.root, "work", "gallm", "audit", "H-100", "default_mock")
        return (os.path.join(wd, "gen", "shard_0_of_1_val.json"),
                os.path.join(wd, "preds", "ga-llm@H-100", "audit", "val.cost.json"))

    def test_reviewer_repro_rerun_keeps_shard_cost(self):
        ga = self.ga
        shard_path, cost_path = self._paths()
        ga.main(self.ARGS + ["--infer-shard", "0/1"])
        first = json.load(open(shard_path))
        self.assertGreater(first["wall_s"], 0)
        self.assertEqual(first["cached_positions"], [])
        for i in (1, 2, 3):
            ga.main(self.ARGS + ["--infer-shard", "0/1"])
            again = json.load(open(shard_path))
            self.assertEqual({k: again[k] for k in ("wall_s", "cpu_core_s", "gpu_s", "n_generated", "n_cached",
                                                    "finished", "ledger_run")},
                             {k: first[k] for k in ("wall_s", "cpu_core_s", "gpu_s", "n_generated", "n_cached",
                                                    "finished", "ledger_run")})
            self.assertEqual(len(again["reruns"]), i)
            self.assertEqual(again["reruns"][-1]["n_cached"], 2)
            self.assertEqual(again["reruns"][-1]["wall_s"], 0.0)
        ga.main(self.ARGS)
        assembled = json.load(open(cost_path))
        rec = summarize_cost(assembled, 2)["recurring"]
        self.assertAlmostEqual(rec["wall_s"], first["wall_s"], places=6)
        self.assertAlmostEqual(rec["cpu_core_s"], first["cpu_core_s"], places=6)
        self.assertEqual([s["name"] for s in assembled["stages"] if not s["one_time"]], ["infer"])

    def test_legacy_zeroed_shard_record_costed_from_ledger(self):
        ga = self.ga
        shard_path, cost_path = self._paths()
        ga.main(self.ARGS + ["--infer-shard", "0/1"])
        first = json.load(open(shard_path))
        legacy = {k: v for k, v in first.items() if k not in ("cached_positions", "keys_sha256", "ledger_run")}
        legacy.update(wall_s=0.0, cpu_core_s=0.0, gpu_s=0.0, n_generated=0, n_cached=2, n_resumed=0)
        json.dump(legacy, open(shard_path, "w"))
        ga.main(self.ARGS)
        assembled = json.load(open(cost_path))
        ic = [s for s in assembled["stages"] if s["name"] == "infer_cached"]
        self.assertEqual(len(ic), 1)
        self.assertAlmostEqual(ic[0]["wall_s"], first["wall_s"], places=6)
        self.assertGreater(summarize_cost(assembled, 2)["recurring"]["wall_s"], 0)

    def test_shard_of_other_requests_ignored(self):
        ga = self.ga
        shard_path, cost_path = self._paths()
        ga.main(self.ARGS + ["--infer-shard", "0/1"])
        rec = json.load(open(shard_path))
        rec["keys_sha256"] = "0" * 64
        json.dump(rec, open(shard_path, "w"))
        ga.main(self.ARGS)
        assembled = json.load(open(cost_path))
        self.assertNotIn("shards", assembled["generation"])
        ic = [s for s in assembled["stages"] if s["name"] == "infer_cached"]
        self.assertEqual(ic[0]["n_items"], 2)

    def test_keep_shard_measurement_rules(self):
        ga = self.ga
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "s.json")
            old = dict(shard=0, nshards=2, n=5, model_fingerprint="m", keys_sha256="h", n_cached=0, wall_s=9.0)
            json.dump(old, open(p, "w"))
            new = dict(old, n_cached=5, wall_s=0.0)
            self.assertEqual(ga.keep_shard_measurement(p, new)["wall_s"], 9.0)
            self.assertEqual(ga.keep_shard_measurement(p, dict(new, keys_sha256="x"))["wall_s"], 0.0)
            self.assertEqual(ga.keep_shard_measurement(p, dict(new, model_fingerprint="o"))["wall_s"], 0.0)
            self.assertEqual(ga.keep_shard_measurement(p, dict(new, n_cached=0, wall_s=8.0))["wall_s"], 8.0)


class TestPreserveIdempotent(unittest.TestCase):
    def test_reviewer_repro_llm4poi_consecutive_reruns(self):
        from methods.llm4poi import run as ll
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {"LLM4POI_CACHE_DIR": temp}):
            fp, prompt = "audit-weights", "audit prompt"
            ll.ResponseCache(fp).put([dict(key=ll.cache_key(fp, prompt), text="1", n_prompt_tokens=2,
                                           n_output_tokens=1, finish_reason="stop", gen_s=42.0)])
            a = SimpleNamespace(backend="vllm", cfg="default", ds="audit", policy="H-100", splits=["val"])
            done = dict(final_ckpt=temp, model_fingerprint=fp, train_wall_s=1.0, train_cpu_core_s=1.0, train_gpu_s=1.0,
                        sessions=[{"n_gpus": 1}], global_step=1)
            prompts = {"val": pd.DataFrame({"query_id": [1], "prompt": [prompt], "prompt_sha256": ["audit"]})}
            pmeta = {"val": {"stage": dict(wall_s=0.0, cpu_core_s=0.0, gpu_s=0.0, peak_mem_gb=0.0, one_time=False),
                             "stats": {}}}
            out = os.path.join(temp, "preds")
            cost_path = os.path.join(out, "llm4poi@H-100", "audit", "val.cost.json")

            class Ex(_Ex):
                manifest = dict(_Ex.manifest, source_sha256={"candidates_val.parquet": "audit"})

                def candidates(self, split):
                    return pd.DataFrame({"query_id": [1], "cand_venue_ids": [list(range(10))]})

            def rerun():
                with patch.object(ll, "base_model", return_value={"revision": "audit"}):
                    ll.do_predict(a, Ex(), temp, done, prompts, pmeta, out)
                return json.load(open(cost_path))

            first = rerun()
            first["execution"]["n_cache_hits"] = 0
            first["execution"]["n_generated"] = 1
            first["created_utc"] = "2026-01-01T00:00:00Z"
            first["stages"] = [s for s in first["stages"] if s["one_time"]] + [dict(
                name="infer", wall_s=100.0, gpu_s=200.0, cpu_core_s=25.0, peak_mem_gb=12.0, one_time=False)]
            json.dump(first, open(cost_path, "w"))
            for i in (1, 2, 3, 4):
                cur = rerun()
                rec = summarize_cost(cur, 1)["recurring"]
                self.assertEqual((rec["wall_s"], rec["cpu_core_s"], rec["gpu_s"]), (100.0, 25.0, 200.0))
                self.assertEqual(cur["per_query_stages_from"]["created_utc"], "2026-01-01T00:00:00Z")
                self.assertEqual(cur["per_query_stages_from"]["n_cache_hits"], 0)
                self.assertEqual(len(cur["reruns"]), i)
                self.assertEqual(cur["execution"]["n_cache_hits"], 1)
                self.assertEqual(cur["execution"]["n_generated"], 0)

    def test_rules(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "c.json")
            m = ("model_fingerprint",)
            json.dump(dict(model_fingerprint="m", execution=dict(n_cache_hits=0), created_utc="t0",
                           stages=[dict(name="infer", one_time=False, wall_s=9)]), open(p, "w"))
            new = dict(model_fingerprint="m", execution=dict(n_cache_hits=2), created_utc="t1",
                       stages=[dict(name="infer", one_time=False, wall_s=0)])
            for i in range(3):
                kept = preserve_measured(p, dict(new, created_utc=f"t{i + 1}"), m)
                json.dump(kept, open(p, "w"))
                self.assertEqual([s["wall_s"] for s in kept["stages"]], [9])
                self.assertEqual(kept["per_query_stages_from"]["created_utc"], "t0")
                self.assertEqual([r["created_utc"] for r in kept["reruns"]], [f"t{j + 1}" for j in range(i + 1)])
            self.assertNotIn("per_query_stages_from", preserve_measured(p, dict(new, execution=dict(n_cache_hits=0)), m))
            self.assertNotIn("per_query_stages_from", preserve_measured(p, dict(new, model_fingerprint="o"), m))
            self.assertEqual(preserve_measured(p, dict(new, execution=dict(n_cache_hits=1)), m)["stages"][0]["wall_s"], 9)


if __name__ == "__main__":
    unittest.main()
