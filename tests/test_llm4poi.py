import importlib.util
import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

from methods.llm4poi import prompts as PR
from poibench import canary as C
from poibench import exports as E
from tests import synth
from tests.test_exports import write_synth

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_PY = os.path.join(ROOT, "methods", "llm4poi", "run.py")


def _load(path, name):
    sp = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(sp)
    sys.modules[name] = m
    sp.loader.exec_module(m)
    return m


def _upstream():
    return _load(os.path.join(PR.UPSTREAM_DIR, PR.UPSTREAM_FILE), "llm4poi_upstream_convert")


RUN = _load(RUN_PY, "llm4poi_run_under_test")


class TestTemplate(unittest.TestCase):
    def test_template_is_upstream_minus_clause(self):
        up = PR.upstream_pieces()["template"]
        self.assertIn("Given the data, at {target_time}, which POI will user {user_id} visit?", up)
        self.assertEqual(PR.template("tt"), up)
        self.assertEqual(PR.template("default"), up.replace("at {target_time}, ", ""))
        self.assertNotIn("target_time", PR.template("default"))
        self.assertIn("Given the data, which POI will user {user_id} visit?\n", PR.template("default"))

    def test_format_entries_matches_upstream(self):
        up = _upstream()
        e = [dict(time_str="2012-04-01 10:00:00", poiid="5", category="Bar"),
             dict(time_str="2012-04-01 11:00:00", poiid="7", category="Gym")]
        for entries in ([], e):
            lines = [f"{x['time_str']} | {x['poiid']} | {x['category']}" for x in entries]
            self.assertEqual(PR.format_entries(lines, "H:"), up.format_entries(entries, "H:"))

    def test_parsers(self):
        self.assertEqual(RUN.parse_strict("123"), 123)
        self.assertEqual(RUN.parse_strict(" 123 \n"), 123)
        self.assertEqual(RUN.parse_strict("123<|endoftext|>"), 123)
        self.assertIsNone(RUN.parse_strict("POI 123"))
        self.assertIsNone(RUN.parse_strict("0123"))
        self.assertIsNone(RUN.parse_strict(""))
        self.assertEqual(RUN.parse_lenient("POI 123, maybe 7"), 123)
        self.assertIsNone(RUN.parse_lenient("unknown"))
        lst, ok = RUN.ranked_list(5, [3, 5, 9], 10)
        self.assertEqual((lst, ok), ([5, 3, 9], True))
        lst, ok = RUN.ranked_list(99, [3, 5, 9], 10)
        self.assertEqual((lst, ok), ([3, 5, 9], False))

    def test_swift_cmd_keeps_authors_flags(self):
        cmd = RUN.swift_cmd("d.jsonl", "out", 2, "3", 20, None)
        a = dict(zip(cmd[::1], cmd[1::1]))
        self.assertEqual(int(a["--per_device_train_batch_size"]) * int(a["--gradient_accumulation_steps"]) * 2, 1024)
        for k, v in (("--learning_rate", "1e-5"), ("--num_train_epochs", "3"), ("--max_length", "12024"),
                     ("--warmup_ratio", "0.05"), ("--train_type", "full"), ("--deepspeed", "zero3"),
                     ("--seed", "0"), ("--data_seed", "0"), ("--save_only_model", "false"),
                     ("--use_liger_kernel", "true"), ("--attn_impl", "flash_attn"), ("--packing", "false")):
            self.assertEqual(a[k], v, k)
        self.assertIn("--nproc_per_node", cmd)
        self.assertNotIn("--resume_from_checkpoint", cmd)
        self.assertIn("--resume_from_checkpoint", RUN.swift_cmd("d", "o", 2, "3", 20, "o/checkpoint-5"))
        with self.assertRaises(ValueError):
            RUN.swift_cmd("d", "o", 3, "3", 20, None)


class TestOnSyntheticExports(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rows, nv = synth.random_rows(n_users=30, seed=7)
        cls.t = synth.build_tables(rows, n_venues=nv)
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = cls.tmp.name
        write_synth(cls.root, "rnd", cls.t, xs=(50, 100))
        for p in ("H-50", "S-50", "H-100"):
            E.export_inputs("rnd", p, cls.root)
        cls.old = {k: os.environ.get(k) for k in ("LLM4POI_BACKEND", "LLM4POI_CACHE_DIR")}
        os.environ["LLM4POI_BACKEND"] = "mock"
        os.environ["LLM4POI_CACHE_DIR"] = os.path.join(cls.root, "llm_cache")

    @classmethod
    def tearDownClass(cls):
        for k, v in cls.old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        cls.tmp.cleanup()

    def _ex(self, p):
        return E.Export(E.export_dir("rnd", p, self.root))

    def test_train_samples_equal_upstream_build_samples(self):
        up = _upstream()
        ex = self._ex("H-100")
        ours = PR.train_samples(ex, "tt")
        r = ex.rows
        df = pd.DataFrame({"UserId": r.user_id, "pseudo_session_trajectory_id": r.traj_id, "UTCTimeOffset": r.t_local,
                           "PoiId": r.venue_id, "PoiCategoryName": r.category})
        df.sort_values(["UserId", "UTCTimeOffset", "pseudo_session_trajectory_id"], inplace=True)
        df["category_name"] = df["PoiCategoryName"].apply(up.normalize_category)
        s = up.build_samples(df, up.build_history(df), "rnd", "train", 50)
        u = {x["trajectory_id"]: x for x in s}
        self.assertEqual(set(u), set(ours["traj_id"]))
        for row in ours.itertuples():
            self.assertEqual(row.prompt, u[row.traj_id]["user_prompt"])
            self.assertEqual(row.answer, u[row.traj_id]["assistant_prompt"])
        dflt = PR.train_samples(ex, "default")
        self.assertEqual(list(dflt["prompt"]), [p.split("Given the data, at ")[0] + "Given the data, which POI will user "
                                                + p.split("which POI will user ")[1] for p in ours["prompt"]])

    def test_policy_samples(self):
        tl = {p: self._ex(p).train_labels() for p in ("H-50", "S-50", "H-100")}
        s = {p: PR.train_samples(self._ex(p)) for p in ("H-50", "S-50", "H-100")}
        self.assertEqual(set(tl["H-50"]["query_id"]), set(tl["S-50"]["query_id"]))
        self.assertEqual(list(s["H-50"]["query_id"]), list(s["S-50"]["query_id"]))
        self.assertTrue(set(s["H-50"]["query_id"]) <= set(tl["H-50"]["query_id"]))
        m = s["S-50"].merge(s["H-50"], on="query_id", suffixes=("_s", "_h"))
        self.assertTrue((m["n_hist_s"] >= m["n_hist_h"]).all() and (m["n_cur_s"] >= m["n_cur_h"]).all())
        self.assertTrue((m["n_hist_s"] > m["n_hist_h"]).any())
        h100 = s["H-100"].set_index("query_id")["prompt"]
        both = [q for q in s["S-50"]["query_id"] if q in h100.index]
        self.assertTrue(both)
        for q in both:
            self.assertEqual(s["S-50"].set_index("query_id").at[q, "prompt"], h100.at[q])
        ex = self._ex("H-50")
        vis_lines = set(PR.entry_lines(ex.rows))
        for p in s["H-50"]["prompt"]:
            body = p.split("Entries:\n")[1].split("</history>")[0] + p.split("poi category):\n")[-1].split("</current>")[0]
            for line in body.strip().split("\n"):
                if line and line != "None.":
                    self.assertIn(line, vis_lines)

    def test_query_prompts_use_own_prefix_only(self):
        ex = self._ex("H-50")
        q = ex.queries("val")
        df = PR.query_prompts(ex, "val", q["query_id"].tolist())
        ck = self.t["checkins"].sort_values("row_id").reset_index(drop=True)
        for r in df.itertuples():
            cur = r.prompt.split("poi category):\n")[-1].split("</current>")[0].strip().split("\n")
            prev = ck.loc[r.query_id - 1]
            self.assertTrue(cur[-1].split(" | ")[1] == str(int(prev.venue_id)))
            self.assertLessEqual(r.n_hist, PR.HISTORY_LIMIT)
            self.assertNotIn("target_time", r.prompt)
            self.assertEqual(len(cur), r.n_cur)

    def test_mock_predict_and_cache(self):
        ex_dir = E.export_dir("rnd", "H-50", self.root)
        qids = self._ex("H-50").queries("val")["query_id"].tolist()[:30]
        a = RUN.predict(ex_dir, "val", qids, os.path.join(self.root, "w1"))
        b = RUN.predict(ex_dir, "val", qids, os.path.join(self.root, "w2"))
        self.assertEqual([list(x) for x in a["ranked_venue_ids"]], [list(x) for x in b["ranked_venue_ids"]])
        self.assertTrue(all(len(x) >= 10 for x in a["ranked_venue_ids"]))

    def test_canary_passes_with_mock(self):
        for pol in ("H-50", "S-50"):
            rep = C.run_canary(f"{RUN_PY}:predict", "rnd", pol, "val", n=40, root=self.root,
                               lint_paths=[os.path.dirname(RUN_PY)], tag="llm4poi_test")
            self.assertEqual(rep["result"], "PASS", (pol, rep["n_mismatch"], rep["lint"]))
            self.assertGreater(rep["perturbation"]["n_future_replaced"], 0)

    def test_canary_catches_a_leaky_prompt(self):
        orig = PR.query_prompts

        def leaky(ex, split, query_ids, cfg="default", target_time=None):
            df = orig(ex, split, query_ids, cfg, target_time)
            ctx = ex.context(split)
            leak = dict(zip(ctx["row_id"].astype(int), ctx["venue_id"].astype(int)))
            df["prompt"] = [p + f" | {leak.get(int(q), -1)} |" for p, q in zip(df["prompt"], df["query_id"])]
            return df
        PR.query_prompts = leaky
        try:
            rep = C.run_canary(f"{RUN_PY}:predict", "rnd", "H-50", "val", n=40, root=self.root, lint_paths=[],
                               tag="llm4poi_leaky")
        finally:
            PR.query_prompts = orig
        self.assertEqual(rep["result"], "FAIL")
        self.assertGreater(rep["n_mismatch"]["ranked_venue_ids"], 0)

    def test_sft_data_invariant_under_canary_perturbation(self):
        ck, venues, queries, pol, cands, meta = E.load_sources("rnd", "H-50", self.root)
        S = C.choose_queries(queries["val"], 40, 0)
        ck2, q2, _ = C.perturb_sources(ck, venues, queries, pol, S, 0)
        pexp = os.path.join(self.root, "pert_H-50")
        E.write_export(E.build_export(ck2, venues, q2, pol, cands), pexp, dict(ds="rnd", policy="H-50"))
        a = PR.train_samples(self._ex("H-50"))
        b = PR.train_samples(E.Export(pexp))
        self.assertEqual(list(a["prompt"]), list(b["prompt"]))
        self.assertEqual(list(a["answer"]), list(b["answer"]))


if __name__ == "__main__":
    unittest.main()
