import math
import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from poibench import evaluate as ev
from poibench.hashing import sha256_file, tree_fingerprint
from tests import synth


def _git_commit(root):
    import subprocess
    g = ["git", "-c", f"safe.directory={root}", "-c", "user.name=t", "-c", "user.email=t@t", "-C", root]
    subprocess.run(g + ["init", "-q"], check=True)
    with open(os.path.join(root, "README"), "w") as f:
        f.write("x\n")
    subprocess.run(g + ["add", "README"], check=True)
    subprocess.run(g + ["commit", "-q", "-m", "t"], check=True)
    return subprocess.run(g + ["rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()


def _queries(rows):
    return pd.DataFrame([{"query_id": q, "user_id": u, "traj_id": "t", "target_row_id": i, "target_venue_id": t,
                          "target_t_local": pd.Timestamp("2012-01-01"), "prev_row_id": i + 1000,
                          "in_user_history": h, "user_hist_len": L}
                         for i, (q, u, t, h, L) in enumerate(rows)])


class TestMetrics(unittest.TestCase):
    def test_metric_values(self):
        m = ev.metrics_from_ranks(np.array([1, 3, np.inf]))
        acc1, acc5, acc10, ndcg, mrr, ndcg5, map20 = m.mean(axis=0)
        self.assertAlmostEqual(acc1, 1 / 3)
        self.assertAlmostEqual(acc5, 2 / 3)
        self.assertAlmostEqual(acc10, 2 / 3)
        self.assertAlmostEqual(ndcg, (1 + 0.5) / 3)
        self.assertAlmostEqual(mrr, (1 + 1 / 3) / 3)
        self.assertAlmostEqual(ndcg5, (1 + 0.5) / 3)
        self.assertAlmostEqual(map20, (1 + 1 / 3) / 3)

    def test_new_metric_boundaries(self):
        m = ev.metrics_from_ranks(np.array([5, 6, 10, 11]))
        self.assertAlmostEqual(m[0, 5], 1 / math.log2(6))
        self.assertEqual(m[1, 5], 0.0)
        self.assertAlmostEqual(m[2, 6], 1 / 10)
        self.assertAlmostEqual(m[3, 6], 1 / 11)
        m2 = ev.metrics_from_ranks(np.array([20, 21]))
        self.assertAlmostEqual(m2[0, 6], 1 / 20)
        self.assertEqual(m2[1, 6], 0.0)
        self.assertAlmostEqual(m2[1, 4], 1 / 21)

    def test_rank_boundaries(self):
        m = ev.metrics_from_ranks(np.array([5, 6, 10, 11]))
        np.testing.assert_array_equal(m[:, 1], [1, 0, 0, 0])
        np.testing.assert_array_equal(m[:, 2], [1, 1, 1, 0])
        self.assertAlmostEqual(m[2, 3], 1 / math.log2(11))
        self.assertEqual(m[3, 3], 0.0)
        self.assertAlmostEqual(m[3, 4], 1 / 11)

    def test_weighted_terciles(self):
        q1, q2 = ev.weighted_terciles(np.array([1, 2, 3, 6]), np.array([1, 2, 3, 6]))
        self.assertEqual((q1, q2), (3.0, 6.0))


class TestEvaluate(unittest.TestCase):
    def setUp(self):
        self.q = _queries([("a", "u1", "X", True, 1), ("b", "u1", "Y", False, 5),
                           ("c", "u2", "Z", False, 9), ("d", "u3", "W", True, 20)])
        self.cands = pd.DataFrame({"query_id": ["a", "b", "c", "d"],
                                   "cand_venue_ids": [["P", "X", "Q"], ["Y", "P"], ["P", "Q"], ["W", "X"]]})
        self.venues = pd.DataFrame({"venue_id": list("XYZWPQ"), "n_train_visits": [10, 5, 0, 2, 1, 1]})

    def test_e2e_and_cand_modes(self):
        preds = pd.DataFrame({"query_id": ["a", "b", "c", "d"],
                              "ranked_venue_ids": [["Q", "P", "X"], ["Y"], ["Z"], [str(i) for i in range(10)] + ["W"]]})
        res = ev.evaluate({"m": preds}, self.q, self.cands, self.venues, B=50, split="val")
        r = pd.DataFrame(res["results"]).set_index(["mode", "stratum", "metric"])["value"]
        self.assertAlmostEqual(r[("e2e", "all", "acc@1")], 0.5)
        self.assertAlmostEqual(r[("e2e", "all", "mrr")], (1 / 3 + 1 + 1 + 1 / 11) / 4)
        self.assertAlmostEqual(r[("e2e", "all", "acc@10")], 0.75)
        self.assertAlmostEqual(r[("cand", "all", "acc@1")], 0.5)
        self.assertAlmostEqual(r[("cand", "all", "mrr")], (1 / 3 + 1 + 0 + 1) / 4)
        self.assertAlmostEqual(res["strata_info"]["cand_recall"], 0.75)
        d = res["diagnostics"]["m"]
        self.assertEqual(d["n_short_lists"], 3)
        self.assertAlmostEqual(d["cand_fill_rate"], (0 + 0.5 + 1 + 0.5) / 4)

    def test_cand_ranked_column_and_int_ids(self):
        q = self.q.copy()
        q["target_venue_id"] = [1, 2, 3, 4]
        cands = pd.DataFrame({"query_id": list("abcd"), "cand_venue_ids": [[5, 1], [2], [5], [4]]})
        preds = pd.DataFrame({"query_id": list("abcd"),
                              "ranked_venue_ids": [["5", "1"], [2], [3], [4]],
                              "cand_ranked_venue_ids": [[1, 5], None, [5], [4]]})
        res = ev.evaluate({"m": preds}, q, cands, None, B=0, split="val")
        r = pd.DataFrame(res["results"]).set_index(["mode", "stratum", "metric"])["value"]
        self.assertAlmostEqual(r[("e2e", "all", "acc@1")], 0.75)
        self.assertAlmostEqual(r[("cand", "all", "acc@1")], 0.75)

    def test_strata(self):
        preds = pd.DataFrame({"query_id": list("abcd"), "ranked_venue_ids": [["X"], ["Y"], ["Q"], ["Q"]]})
        res = ev.evaluate({"m": preds}, self.q, self.cands, self.venues, B=0, split="val")
        sz = res["strata_sizes"]
        self.assertEqual(sz["hist=repeat"], 2)
        self.assertEqual(sz["pop=cold"], 1)
        self.assertEqual(sz["cand=hit"], 3)
        self.assertEqual(sz["hlen=short"] + sz["hlen=mid"] + sz["hlen=long"], 4)
        self.assertEqual(sz["pop=cold"] + sz["pop=tail"] + sz["pop=mid"] + sz["pop=head"], 4)
        r = pd.DataFrame(res["results"]).set_index(["mode", "stratum", "metric"])["value"]
        self.assertAlmostEqual(r[("e2e", "hist=repeat", "acc@1")], 0.5)
        self.assertAlmostEqual(r[("e2e", "hist=explore", "acc@1")], 0.5)

    def test_missing_and_extra_predictions(self):
        preds = pd.DataFrame({"query_id": ["a"], "ranked_venue_ids": [["X"]]})
        with self.assertRaises(ValueError):
            ev.evaluate({"m": preds}, self.q, self.cands, self.venues, B=0, split="val")
        res = ev.evaluate({"m": preds}, self.q, self.cands, self.venues, B=0, allow_missing=True, split="val")
        r = pd.DataFrame(res["results"]).set_index(["mode", "stratum", "metric"])["value"]
        self.assertAlmostEqual(r[("e2e", "all", "acc@1")], 0.25)
        self.assertAlmostEqual(r[("cand", "all", "acc@1")], 0.25)
        extra = pd.DataFrame({"query_id": ["zz"], "ranked_venue_ids": [["X"]]})
        with self.assertRaises(ValueError):
            ev.evaluate({"m": extra}, self.q, self.cands, self.venues, B=0, allow_missing=True, split="val")

    def test_duplicates_in_list_are_removed(self):
        preds = pd.DataFrame({"query_id": list("abcd"),
                              "ranked_venue_ids": [["Q", "Q", "X"], ["Y"], ["Z"], ["W"]]})
        res = ev.evaluate({"m": preds}, self.q, None, None, B=0, split="val")
        r = pd.DataFrame(res["results"]).set_index(["mode", "stratum", "metric"])["value"]
        self.assertAlmostEqual(r[("e2e", "all", "acc@5")], 1.0)
        self.assertAlmostEqual(r[("e2e", "all", "mrr")], (0.5 + 1 + 1 + 1) / 4)
        self.assertEqual(res["diagnostics"]["m"]["n_lists_with_duplicates"], 1)


class TestBootstrap(unittest.TestCase):
    def setUp(self):
        t = synth.build_tables(*synth.random_rows(seed=1)[:1])
        self.q = t["queries"]["val"]
        rng = np.random.default_rng(0)
        venues = sorted(t["checkins"].venue_id.unique())
        lists = []
        for tgt in self.q.target_venue_id:
            lst = list(rng.choice(venues, size=10, replace=False))
            if rng.random() < 0.3 and tgt not in lst:
                lst[0] = tgt
            lists.append(lst)
        self.p1 = pd.DataFrame({"query_id": self.q.query_id, "ranked_venue_ids": lists})
        self.p2 = pd.DataFrame({"query_id": self.q.query_id, "ranked_venue_ids": [[t] for t in self.q.target_venue_id]})

    def test_weights(self):
        W = ev.bootstrap_weights(7, B=20, seed=0)
        self.assertEqual(W.shape, (20, 7))
        np.testing.assert_array_equal(W.sum(axis=1), 7)
        np.testing.assert_array_equal(W, ev.bootstrap_weights(7, B=20, seed=0))

    def test_ci_and_pairs(self):
        res = ev.evaluate({"m1": self.p1, "same": self.p1.copy(), "oracle": self.p2}, self.q, None, None, B=200, split="val")
        rows = pd.DataFrame(res["results"])
        for _, r in rows[rows.stratum == "all"].iterrows():
            self.assertLessEqual(r.ci_lo, r.value + 1e-12)
            self.assertGreaterEqual(r.ci_hi, r.value - 1e-12)
        pairs = pd.DataFrame(res["pairs"])
        same = pairs[(pairs.a == "m1") & (pairs.b == "same")]
        self.assertTrue(np.allclose(same[["diff", "ci_lo", "ci_hi"]].to_numpy(), 0))
        orc = pairs[(pairs.a == "m1") & (pairs.b == "oracle") & (pairs.stratum == "all") & (pairs.metric == "acc@1")].iloc[0]
        self.assertLess(orc["diff"], 0)
        self.assertLess(orc["ci_hi"], 0)
        self.assertLess(orc["p_boot"], 0.01)
        res2 = ev.evaluate({"m1": self.p1, "same": self.p1.copy(), "oracle": self.p2}, self.q, None, None, B=200, split="val")
        self.assertEqual(res["results"], res2["results"])

    def test_ref_pairs(self):
        res = ev.evaluate({"m1": self.p1, "oracle": self.p2}, self.q, None, None, B=10, pairs="ref", ref="oracle", split="val")
        self.assertEqual({(p["a"], p["b"]) for p in res["pairs"]}, {("m1", "oracle")})


class TestFiles(unittest.TestCase):
    def test_test_split_guard_and_files(self):
        t = synth.hand_tables()
        with tempfile.TemporaryDirectory() as root:
            synth.write_dataset(root, "toy", t)
            for split in ("val", "test"):
                q = t["queries"][split]
                d = os.path.join(root, "preds", "oracle", "toy")
                os.makedirs(d, exist_ok=True)
                pd.DataFrame({"query_id": q.query_id, "ranked_venue_ids": [[v] for v in q.target_venue_id]}
                             ).to_parquet(os.path.join(d, f"{split}.parquet"))
            res = ev.evaluate_files(root, "toy", "val", ["oracle"], out_prefix=os.path.join(root, "res", "x"), B=10)
            self.assertTrue(os.path.exists(os.path.join(root, "res", "x.csv")))
            self.assertEqual(res["config"]["split"], "val")
            self.assertEqual((res["strata_sizes"]["pos=last"], res["strata_sizes"]["pos=inner"]), (1, 1))
            r = pd.DataFrame(res["results"]).set_index(["mode", "stratum", "metric"])["n_queries"]
            self.assertEqual(r[("e2e", "pos=last", "acc@1")], 1)
            with self.assertRaises(PermissionError):
                ev.evaluate_files(root, "toy", "test", ["oracle"], B=10)
            commit = _git_commit(root)
            data_fp = tree_fingerprint(os.path.join(root, "data"))
            eval_fp = sha256_file(ev.__file__)
            good = f"Status: FROZEN v1.0 2026-11-01 data={data_fp} eval={eval_fp} git={commit}"
            for head in ("Status: DRAFT\n", "**Status: FROZEN** (2026-11-01)\n",
                         "Status: FROZEN means nothing yet\n",
                         "- " + good + "\n",
                         "> " + good + "\n",
                         "Status: DRAFT v0\n" + good + "\n",
                         good.replace(data_fp, "0" * 64) + "\n",
                         good.replace(eval_fp, "0" * 64) + "\n",
                         good.replace(commit, "deadbeef" * 5) + "\n",
                         good + " (planned)\n",
                         "\n" * 12 + good + "\n"):
                with open(os.path.join(root, "PROTOCOL.md"), "w") as f:
                    f.write("# Protocol\n\n" + head)
                with self.assertRaises(PermissionError, msg=head):
                    ev.evaluate_files(root, "toy", "test", ["oracle"], B=10)
                q = t["queries"]["test"]
                with self.assertRaises(PermissionError, msg=head):
                    ev.evaluate({"o": pd.DataFrame({"query_id": q.query_id, "ranked_venue_ids": [[0]] * len(q)})},
                                q, None, None, B=0, split="test", root=root)
            with open(os.path.join(root, "PROTOCOL.md"), "w") as f:
                f.write("# Protocol\n\n" + good + "\n")
            res = ev.evaluate_files(root, "toy", "test", ["oracle"], B=10)
            r = pd.DataFrame(res["results"])
            self.assertTrue(np.allclose(r[(r.stratum == "all") & (r["mode"] == "e2e")].value, 1.0))
            with open(os.path.join(root, "data", "toy", "extra.json"), "w") as f:
                f.write("{}")
            with self.assertRaises(PermissionError):
                ev.evaluate_files(root, "toy", "test", ["oracle"], B=10)

    def test_evaluate_requires_split_and_candidate_recall_lock(self):
        t = synth.hand_tables()
        q = t["queries"]["val"]
        preds = pd.DataFrame({"query_id": q.query_id, "ranked_venue_ids": [[v] for v in q.target_venue_id]})
        with self.assertRaises(TypeError):
            ev.evaluate({"m": preds}, q, None, None, B=0)
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(PermissionError):
                ev.evaluate({"m": preds}, t["queries"]["test"], None, None, B=0, split="test", root=root)
            from poibench.data import candidate_recall
            split = t["checkins"]["split"].to_numpy()
            r = candidate_recall(q, t["candidates"]["val"], row_split=split, root=root)
            self.assertIn("recall@20", r)
            with self.assertRaises(PermissionError):
                candidate_recall(t["queries"]["test"], t["candidates"]["test"], row_split=split, root=root)
            mixed = pd.concat([q, t["queries"]["test"]])
            with self.assertRaises(PermissionError):
                candidate_recall(mixed, pd.concat([t["candidates"]["val"], t["candidates"]["test"]]),
                                 row_split=split, root=root)

    def test_val_feedback(self):
        t = synth.hand_tables()
        with tempfile.TemporaryDirectory() as root:
            synth.write_dataset(root, "toy", t)
            q = t["queries"]["val"]
            preds = pd.DataFrame({"query_id": q.query_id, "ranked_venue_ids": [[9, v] for v in q.target_venue_id]})
            fb = ev.val_feedback("toy", preds, root=root)
            self.assertEqual((fb["acc@1"], fb["acc@5"], fb["n_queries"], fb["n_missing"]), (0.0, 1.0, len(q), 0))
            self.assertAlmostEqual(fb["mrr"], 0.5)

    def test_candidate_file_mismatch_guard(self):
        import json
        t = synth.hand_tables()
        with tempfile.TemporaryDirectory() as root:
            synth.write_dataset(root, "toy", t)
            q = t["queries"]["val"]
            d = os.path.join(root, "preds", "m", "toy")
            os.makedirs(d, exist_ok=True)
            pd.DataFrame({"query_id": q.query_id, "ranked_venue_ids": [[v] for v in q.target_venue_id]}
                         ).to_parquet(os.path.join(d, "val.parquet"))
            with open(os.path.join(d, "val.cost.json"), "w") as f:
                json.dump({"stages": [], "candidates_file": "data/toy/evidence/OTHER.candidates_val.parquet"}, f)
            with self.assertRaises(ValueError):
                ev.evaluate_files(root, "toy", "val", ["m"], B=10)
            res = ev.evaluate_files(root, "toy", "val", ["m"], B=10, e2e_only=True)
            self.assertEqual(set(pd.DataFrame(res["results"])["mode"]), {"e2e"})
            with open(os.path.join(d, "val.cost.json"), "w") as f:
                json.dump({"stages": [], "candidates_file": "data/toy/candidates_val.parquet"}, f)
            res = ev.evaluate_files(root, "toy", "val", ["m"], B=10)
            self.assertEqual(set(pd.DataFrame(res["results"])["mode"]), {"e2e", "cand"})
            p = pd.read_parquet(os.path.join(d, "val.parquet"))
            p["cand_ranked_venue_ids"] = p["ranked_venue_ids"]
            p.to_parquet(os.path.join(d, "val.parquet"))
            res = ev.evaluate_files(root, "toy", "val", ["m"], B=10)
            with open(os.path.join(d, "val.cost.json"), "w") as f:
                json.dump({"stages": []}, f)
            with self.assertRaises(ValueError):
                ev.evaluate_files(root, "toy", "val", ["m"], B=10)
            os.remove(os.path.join(d, "val.cost.json"))
            with self.assertRaises(ValueError):
                ev.evaluate_files(root, "toy", "val", ["m"], B=10)
            res = ev.evaluate_files(root, "toy", "val", ["m"], B=10, e2e_only=True)
            self.assertEqual(set(pd.DataFrame(res["results"])["mode"]), {"e2e"})


if __name__ == "__main__":
    unittest.main()
