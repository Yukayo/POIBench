import json
import os
import tempfile
import unittest
from collections import Counter

import numpy as np
import pandas as pd

from poibench.baselines_simple import METHODS, SimpleBaselines
from tests import synth
from tests.synth import NAME, VID


def _names(lst):
    return "".join(NAME.get(int(v), f"<{int(v)}>") for v in lst)


def _pred(model, t, split, method, queries=None, **kw):
    q = t["queries"][split] if queries is None else queries
    p = model.predict(method, q, t["candidates"][split], topk=50, **kw)
    return {int(qid): _names(lst) for qid, lst in zip(p.query_id, p.ranked_venue_ids)}, p


class TestHandBaselines(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.t = synth.hand_tables()
        cls.full = SimpleBaselines(cls.t["checkins"], cls.t["venues"], policy=None)
        cls.h50 = SimpleBaselines(cls.t["checkins"], cls.t["venues"], policy=synth.h_policy(cls.t, 50, "H"))
        cls.s50 = SimpleBaselines(cls.t["checkins"], cls.t["venues"], policy=synth.h_policy(cls.t, 50, "S"))

    def test_popularity(self):
        self.assertEqual(self.full.pop.tolist(), [5, 3, 3, 1, 0, 0])
        r, _ = _pred(self.full, self.t, "test", "pop")
        self.assertEqual(set(r.values()), {"ABCDEF"})
        self.assertEqual(self.h50.pop.tolist(), [3, 1, 2, 1, 0, 0])
        r, _ = _pred(self.h50, self.t, "test", "pop")
        self.assertEqual(set(r.values()), {"ACBDEF"})

    def test_repeat_full(self):
        r, p = _pred(self.full, self.t, "test", "repeat")
        self.assertEqual(r[9], "CBADEF")
        self.assertEqual(r[10], "CBAEDF")
        self.assertEqual(r[19], "BCADEF")
        self.assertEqual(dict(zip(p.query_id, p.n_visible_own)), {9: 6, 10: 7, 19: 3})

    def test_repeat_history_budget(self):
        r, p = _pred(self.h50, self.t, "test", "repeat")
        self.assertEqual(r[9], "CBADEF")
        self.assertEqual(r[19], "BCADEF")
        self.assertEqual(dict(zip(p.query_id, p.n_visible_own))[9], 4)
        self.assertEqual(dict(zip(p.query_id, p.n_visible_own))[19], 2)

    def test_markov_and_hybrid(self):
        r, _ = _pred(self.full, self.t, "test", "markov")
        self.assertEqual((r[9], r[10], r[19]), ("ABCDEF", "ABCDEF", "CABDEF"))
        r, _ = _pred(self.h50, self.t, "test", "markov")
        self.assertEqual((r[9], r[19]), ("ACBDEF", "ACBDEF"))
        r, _ = _pred(self.full, self.t, "test", "hybrid")
        self.assertEqual(r[9], "ACBDEF")
        self.assertEqual(r[19], "CBADEF")

    def test_sample_budget_equals_full(self):
        for m in METHODS:
            a, _ = _pred(self.full, self.t, "test", m)
            b, _ = _pred(self.s50, self.t, "test", m)
            self.assertEqual(a, b, m)

    def test_train_queries_leave_one_out(self):
        q = self.t["queries"]["train"]
        r, _ = _pred(self.full, self.t, "train", "pop")
        self.assertEqual(r[4], "ACBDEF")
        r, _ = _pred(self.full, self.t, "train", "markov")
        self.assertEqual(r[4], "CBADEF")
        r, _ = _pred(self.full, self.t, "train", "repeat")
        self.assertEqual(r[4], "ACBDEF")
        self.assertEqual(sorted(self.h50.allowed_queries(q).query_id.tolist()), [4, 15])
        with self.assertRaises(ValueError):
            self.h50.predict("pop", q)
        self.h50.predict("pop", self.h50.allowed_queries(q))

    def test_candidate_rerank_consistent_with_e2e(self):
        cands = self.t["candidates"]["test"].copy()
        sub = [VID["F"], VID["B"], VID["A"], VID["B"]]
        cands["cand_venue_ids"] = [sub] * len(cands)
        for model in (self.full, self.h50):
            for meth in METHODS:
                p = model.predict(meth, self.t["queries"]["test"], cands)
                for e2e, c in zip(p.ranked_venue_ids, p.cand_ranked_venue_ids):
                    self.assertEqual(sorted(c), sorted({VID["A"], VID["B"], VID["F"]}))
                    self.assertEqual(c, [v for v in e2e if v in set(sub)])

    def test_input_validation(self):
        q = self.t["queries"]["test"].copy()
        q.loc[0, "prev_row_id"] = q.loc[0, "target_row_id"]
        with self.assertRaises(ValueError):
            self.full.predict("repeat", q)
        bad = synth.h_policy(self.t, 50)
        bad["user_visible_from_row_id"]["99"] = 0
        with self.assertRaises(ValueError):
            SimpleBaselines(self.t["checkins"], self.t["venues"], policy=bad)


class TestRandomLeakage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rows, nv = synth.random_rows(n_users=80, n_venues=80, n_trajs_per_user=14, max_len=7, seed=3)
        cls.t = synth.build_tables(rows, n_venues=nv)
        cls.model = SimpleBaselines(cls.t["checkins"], cls.t["venues"], policy=None)

    def _bruteforce_visible(self, ck, row):
        tgt = ck.loc[row]
        u = ck[ck.user_id == tgt.user_id]
        traj = u[u.traj_id == tgt.traj_id]
        prefix = traj[traj.t_utc < tgt.t_utc]
        if tgt.split == "train":
            hist = u[u.t_utc < traj.t_utc.min()]
        else:
            hist = u[(u.split == "train") & (u.t_utc < traj.t_utc.min())]
        return pd.concat([hist, prefix]).sort_values("t_utc").venue_id.tolist()

    def test_repeat_matches_bruteforce(self):
        ck = self.t["checkins"].set_index("row_id")
        for split in ("test", "train"):
            q = self.t["queries"][split].iloc[:400]
            r, p = _pred(self.model, self.t, split, "repeat", queries=q)
            raw = dict(zip(p.query_id, p.ranked_venue_ids))
            for row in q.target_row_id:
                vis = self._bruteforce_visible(ck, int(row))
                cnt = Counter(vis)
                last = {v: i for i, v in enumerate(vis)}
                exp = sorted(cnt, key=lambda v: (-cnt[v], -last[v]))
                self.assertEqual(raw[row][:len(exp)], exp)

    def test_markov_counts_use_visible_train_only(self):
        ck = self.t["checkins"]
        tr = ck[ck.split == "train"].sort_values("row_id")
        pairs = Counter()
        for _, g in tr.groupby("traj_id"):
            v = g.venue_id.tolist()
            pairs.update(zip(v[:-1], v[1:]))
        m = self.model
        self.assertEqual(sum(int(c.sum()) for _, c in m.markov.values()), sum(pairs.values()))
        for (a, b), c in pairs.items():
            sv, sc = m.markov[a]
            self.assertEqual(int(sc[list(sv).index(b)]), c)

    def test_parallel_equals_serial(self):
        q = self.t["queries"]["train"]
        self.assertGreaterEqual(len(q), 2000)
        c = self.t["candidates"]["train"]
        for meth in ("repeat", "hybrid"):
            a = self.model.predict(meth, q, c, n_jobs=1)
            b = self.model.predict(meth, q, c, n_jobs=3)
            pd.testing.assert_frame_equal(a, b)


class TestRunSimpleScript(unittest.TestCase):
    def test_end_to_end(self):
        import importlib.util
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        spec = importlib.util.spec_from_file_location("run_simple", os.path.join(here, "scripts", "run_simple.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        t = synth.hand_tables()
        with tempfile.TemporaryDirectory() as root:
            h50 = synth.h_policy(t, 50)
            h50["candidates"] = {s: f"evidence/H-50.candidates_{s}.parquet" for s in ("train", "val", "test")}
            pols = {"H-100": synth.h_policy(t, 100), "H-50": h50}
            d = synth.write_dataset(root, "toy", t, policies=pols)
            for s in ("train", "val", "test"):
                c = t["candidates"][s].copy()
                c["cand_venue_ids"] = [[VID["F"], VID["E"]]] * len(c)
                c.to_parquet(os.path.join(d, "evidence", f"H-50.candidates_{s}.parquet"))
            mod.main(["toy", "val", "H-100", "--root", root, "--n-jobs", "1"])
            mod.main(["toy", "test", "H-50", "--root", root, "--methods", "repeat"])
            mod.main(["toy", "train", "H-50", "--root", root, "--methods", "pop"])
            for meth in METHODS:
                p = pd.read_parquet(os.path.join(root, "preds", f"{meth}@H-100", "toy", "val.parquet"))
                self.assertEqual(len(p), len(t["queries"]["val"]))
                with open(os.path.join(root, "preds", f"{meth}@H-100", "toy", "val.cost.json")) as f:
                    cost = json.load(f)
                self.assertEqual([s["name"] for s in cost["stages"]], ["load_data", "fit_counts", "predict"])
                self.assertEqual([s["one_time"] for s in cost["stages"]], [True, True, False])
                self.assertEqual(cost["candidates_file"], os.path.join("data", "toy", "candidates_val.parquet"))
            p = pd.read_parquet(os.path.join(root, "preds", "repeat@H-50", "toy", "test.parquet"))
            self.assertTrue(all(sorted(c) == [VID["E"], VID["F"]] for c in p.cand_ranked_venue_ids))
            p = pd.read_parquet(os.path.join(root, "preds", "pop@H-50", "toy", "train.parquet"))
            self.assertEqual(sorted(p.query_id.tolist()), [4, 15])
            from poibench.evaluate import evaluate_files
            res = evaluate_files(root, "toy", "val", [f"{m}@H-100" for m in METHODS], B=20, policy="H-100")
            self.assertEqual(res["n_queries"], len(t["queries"]["val"]))
            res = evaluate_files(root, "toy", "train", ["pop@H-50"], B=20, policy="H-50")
            self.assertEqual(res["n_queries"], 2)
            self.assertEqual(res["config"]["candidates_file"], os.path.join("data", "toy", "evidence",
                                                                            "H-50.candidates_train.parquet"))


class TestHybridTT(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.t = synth.hand_tables()
        cls.model = SimpleBaselines(cls.t["checkins"], cls.t["venues"], policy=None)
        q = cls.t["queries"]["test"]
        cls.tt = pd.DataFrame({"query_id": q["query_id"].to_numpy(), "target_t_local": q["target_t_local"].to_numpy()})

    def _lists(self, method, **kw):
        p = self.model.predict(method, self.t["queries"]["test"], self.t["candidates"]["test"], topk=50, **kw)
        return p.ranked_venue_ids.tolist(), p.cand_ranked_venue_ids.tolist()

    def test_half_day_window_is_hybrid(self):
        import poibench.baselines_simple as bs
        old = bs.TT_WINDOW_H
        try:
            bs.TT_WINDOW_H = 12.0
            self.assertEqual(self._lists("hybrid-tt", target_time=self.tt), self._lists("hybrid"))
        finally:
            bs.TT_WINDOW_H = old

    def test_empty_window_falls_back_to_all_time_frequency(self):
        import poibench.baselines_simple as bs
        old = bs.TT_WINDOW_H
        try:
            bs.TT_WINDOW_H = 0.0
            tt = self.tt.assign(target_t_local=pd.Timestamp("2000-01-01 00:00:07"))
            self.assertEqual(self._lists("hybrid-tt", target_time=tt), self._lists("hybrid"))
        finally:
            bs.TT_WINDOW_H = old

    def test_window_changes_the_order(self):
        import poibench.baselines_simple as bs
        old = bs.TT_WINDOW_H
        try:
            bs.TT_WINDOW_H = 0.01
            ck = self.t["checkins"].sort_values("row_id")
            hour = ck["t_local"].dt.hour + ck["t_local"].dt.minute / 60.0
            tt = self.tt.assign(target_t_local=pd.Timestamp("2000-01-01") + pd.to_timedelta(float(hour.iloc[2]), unit="h"))
            self.assertNotEqual(self._lists("hybrid-tt", target_time=tt)[0], self._lists("hybrid")[0])
        finally:
            bs.TT_WINDOW_H = old

    def test_needs_target_time_and_is_not_a_default_method(self):
        self.assertNotIn("hybrid-tt", METHODS)
        with self.assertRaises(ValueError):
            self.model.predict("hybrid-tt", self.t["queries"]["test"], self.t["candidates"]["test"])
        with self.assertRaises(ValueError):
            self.model.predict("hybrid-tt", self.t["queries"]["test"], self.t["candidates"]["test"],
                               target_time=self.tt.iloc[:1])


if __name__ == "__main__":
    unittest.main()
