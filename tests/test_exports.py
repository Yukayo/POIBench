import json
import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from poibench import exports as E
from poibench.baselines_simple import SimpleBaselines
from poibench.data import Arrays
from tests import synth

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def write_synth(root, ds, tables, xs=(50, 100)):
    pols = {}
    for x in xs:
        for kind in ("H", "S"):
            pols[f"{kind}-{x}"] = synth.h_policy(tables, x, kind)
    d = synth.write_dataset(root, ds, tables, policies=pols)
    tv = tables["checkins"].loc[tables["checkins"].split == "val", "t_utc"].min()
    with open(os.path.join(d, "meta.json"), "w") as f:
        json.dump({"t_val_start_utc": str(tv)}, f)
    return d, pols


def _load_hybrid():
    import importlib.util
    sp = importlib.util.spec_from_file_location("export_hybrid", os.path.join(ROOT, "methods", "export_hybrid", "run.py"))
    m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m)
    return m


class TestHandExport(unittest.TestCase):
    def setUp(self):
        self.t = synth.hand_tables()
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        write_synth(self.root, "toy", self.t)

    def tearDown(self):
        self.tmp.cleanup()

    def test_h100_content(self):
        man = E.export_inputs("toy", "H-100", self.root)
        ex = E.Export(E.export_dir("toy", "H-100", self.root))
        self.assertEqual(man["export_version"], E.EXPORT_VERSION)
        self.assertEqual(ex.rows["row_id"].tolist(), [0, 1, 2, 3, 4, 11, 12, 13, 14, 15, 16, 17])
        self.assertEqual(list(ex.rows.columns), E.ROW_COLS)
        for name in os.listdir(ex.path):
            if name.endswith(".parquet"):
                cols = pd.read_parquet(os.path.join(ex.path, name)).columns
                self.assertFalse(set(cols) & set(E.FORBIDDEN_COLS), name)
        q = ex.queries("val").set_index("query_id")
        self.assertEqual(sorted(q.index), [6, 7])
        self.assertEqual(tuple(q.loc[7, ["hist_lo", "hist_hi", "prefix_lo", "prefix_hi"]]), (0, 5, 5, 7))
        self.assertEqual(q.loc[7, "prev_venue_id"], synth.VID["A"])
        ctx = ex.context("val")
        self.assertEqual(ctx[ctx.query_id == 7]["row_id"].tolist(), [5, 6])
        self.assertEqual(ctx[ctx.query_id == 6]["row_id"].tolist(), [5])
        qi = ex.query_inputs("val")
        i = int(np.flatnonzero(qi.table.query_id.to_numpy() == 7)[0])
        self.assertEqual(qi.own(i).tolist(), [synth.VID[c] for c in "ABCAB" + "EA"])
        ven = ex.venues.set_index("venue_id")
        exp = np.bincount(self.t["checkins"].venue_id[self.t["checkins"].split == "train"], minlength=6)
        self.assertEqual(ven["n_visible_train_visits"].tolist(), exp.tolist())
        self.assertNotIn("n_train_visits", ven.columns)
        lab = ex.train_labels()
        self.assertEqual(set(lab.columns), set(E.LABEL_COLS))
        self.assertEqual(lab.query_id.tolist(), ex.queries("train").query_id.tolist())

    def test_h50_hides_history_and_counts(self):
        E.export_inputs("toy", "H-50", self.root)
        ex = E.Export(E.export_dir("toy", "H-50", self.root))
        self.assertEqual(ex.rows["row_id"].tolist(), [2, 3, 4, 13, 14, 15, 17])
        self.assertEqual(ex.queries("train")["query_id"].tolist(), [4, 15])
        self.assertEqual(ex.train_labels()["label_venue_id"].tolist(), [synth.VID["B"], synth.VID["A"]])
        exp = np.bincount(self.t["checkins"].venue_id.to_numpy()[[2, 3, 4, 13, 14, 15, 17]], minlength=6)
        self.assertEqual(ex.venues.sort_values("venue_id")["n_visible_train_visits"].tolist(), exp.tolist())
        q = ex.queries("test").set_index("query_id")
        self.assertEqual(tuple(q.loc[19, ["hist_lo", "hist_hi", "prefix_lo", "prefix_hi"]]), (17, 18, 18, 19))
        pol = json.load(open(os.path.join(ex.path, "policy_public.json")))
        self.assertNotIn("user_visible_from_row_id", pol)
        self.assertNotIn("train_query_ids", pol)
        E.export_inputs("toy", "S-50", self.root)
        s = E.Export(E.export_dir("toy", "S-50", self.root))
        self.assertEqual(len(s.rows), 12)
        self.assertEqual(s.queries("train")["query_id"].tolist(), [4, 15])

    def test_resumable_and_stale_detection(self):
        m1 = E.export_inputs("toy", "H-100", self.root)
        self.assertFalse(m1.get("skipped", False))
        m2 = E.export_inputs("toy", "H-100", self.root)
        self.assertTrue(m2["skipped"])
        self.assertTrue(E.export_is_current("toy", "H-100", self.root))
        ck = self.t["checkins"].copy()
        ck.loc[0, "venue_id"] = synth.VID["F"]
        ck.to_parquet(os.path.join(self.root, "data", "toy", "checkins.parquet"), index=False)
        self.assertFalse(E.export_is_current("toy", "H-100", self.root))
        m3 = E.export_inputs("toy", "H-100", self.root)
        self.assertFalse(m3.get("skipped", False))
        for fn, h in m3["files"].items():
            from poibench.hashing import sha256_file
            self.assertEqual(sha256_file(os.path.join(E.export_dir("toy", "H-100", self.root), fn)), h)

    def test_check_export_catches_violations(self):
        t = self.t
        pol = synth.h_policy(t, 50)
        A = Arrays.from_checkins(t["checkins"])
        good = E.build_export(t["checkins"], t["venues"], t["queries"], pol, t["candidates"])
        rows = np.arange(len(A.user))
        vis = A.is_train & (rows >= np.array([int(v) for v in pol["user_visible_from_row_id"].values()])[A.user])
        bad = dict(good)
        bad["visible_rows"] = pd.concat([good["visible_rows"], t["checkins"].iloc[[5]][E.ROW_COLS]])
        with self.assertRaises(AssertionError):
            E.check_export(bad, A, vis)
        bad = dict(good)
        c = good["context_val"].copy()
        c.loc[c.index[0], "row_id"] = int(c.loc[c.index[0], "query_id"])
        bad["context_val"] = c
        with self.assertRaises(AssertionError):
            E.check_export(bad, A, vis)
        bad = dict(good)
        bad["query_inputs_val"] = good["query_inputs_val"].assign(target_venue_id=0)
        with self.assertRaises(AssertionError):
            E.check_export(bad, A, vis)
        bad = dict(good)
        bad["venues_public"] = good["venues_public"].assign(
            n_visible_train_visits=np.bincount(t["checkins"].venue_id, minlength=6))
        with self.assertRaises(AssertionError):
            E.check_export(bad, A, vis)


class TestRandomExportMatchesBaselines(unittest.TestCase):
    def test_export_hybrid_equals_simple_hybrid(self):
        rows, nv = synth.random_rows(n_users=30, seed=3)
        t = synth.build_tables(rows, n_venues=nv)
        hyb = _load_hybrid()
        with tempfile.TemporaryDirectory() as root:
            _, pols = write_synth(root, "rnd", t, xs=(10, 50, 100))
            for name in ("H-10", "H-50", "H-100", "S-10"):
                E.export_inputs("rnd", name, root)
                sb = SimpleBaselines(t["checkins"], t["venues"], policy=pols[name])
                for split in ("val", "test"):
                    q = t["queries"][split]
                    ref = sb.predict("hybrid", q, t["candidates"][split], topk=50)
                    got = hyb.predict(E.export_dir("rnd", name, root), split, q["query_id"].tolist())
                    self.assertEqual([list(x) for x in ref.ranked_venue_ids], [list(x) for x in got.ranked_venue_ids],
                                     (name, split))
                    self.assertEqual([list(x) for x in ref.cand_ranked_venue_ids],
                                     [list(x) for x in got.cand_ranked_venue_ids], (name, split))


class TestBuiltExports(unittest.TestCase):

    def test_built_exports(self):
        top = os.path.join(ROOT, "exports")
        dirs = [os.path.join(top, d, p) for d in (sorted(os.listdir(top)) if os.path.isdir(top) else [])
                if os.path.isdir(os.path.join(top, d)) for p in E.POLICIES]
        dirs = [d for d in dirs if os.path.exists(os.path.join(d, "MANIFEST.json"))]
        if not dirs:
            self.skipTest("no exports built")
        from poibench.hashing import sha256_file
        for d in dirs:
            ex = E.Export(d)
            ds, pol = ex.manifest["ds"], ex.manifest["policy"]
            self.assertTrue(E.export_is_current(ds, pol, ROOT), d)
            for fn, h in ex.manifest["files"].items():
                self.assertEqual(sha256_file(os.path.join(d, fn)), h, (d, fn))
                if fn.endswith(".parquet"):
                    import pyarrow.parquet as pq
                    cols = pq.read_schema(os.path.join(d, fn)).names
                    self.assertFalse(set(cols) & set(E.FORBIDDEN_COLS), (d, fn))
            for split in ("val", "test"):
                q, c = ex.queries(split), ex.context(split)
                self.assertTrue((c.row_id.to_numpy() < c.query_id.to_numpy()).all())
                self.assertEqual(len(c), int((q.prefix_hi - q.prefix_lo).sum()))
                self.assertTrue((q.hist_hi.le(q.prefix_lo) | q.hist_hi.eq(q.hist_lo)).all())
            rid = ex.rows.row_id.to_numpy()
            self.assertTrue((np.diff(rid) > 0).all())


if __name__ == "__main__":
    unittest.main()
