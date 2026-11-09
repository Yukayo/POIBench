import importlib.util
import json
import os
import sys
import tempfile
import unittest

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import pandas as pd

from poibench import exports as E
from tests import synth

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MDIR = os.path.join(ROOT, "methods", "graph_flashback")


def _load(name, path):
    if MDIR not in sys.path:
        sys.path.insert(0, MDIR)
    sp = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m)
    return m


def _synth_root(root, ds="gfsyn", policies=(("H", 100), ("H", 50), ("S", 50))):
    rows, nv = synth.random_rows(n_users=12, n_venues=40, n_trajs_per_user=20, max_len=5, seed=7)
    t = synth.build_tables(rows, n_venues=nv)
    pols = {f"{k}-{x}": synth.h_policy(t, x, k) for k, x in policies}
    d = synth.write_dataset(root, ds, t, policies=pols)
    tv = t["checkins"].loc[t["checkins"].split == "val", "t_utc"].min()
    with open(os.path.join(d, "meta.json"), "w") as f:
        json.dump({"t_val_start_utc": str(tv)}, f)
    for p in pols:
        E.export_inputs(ds, p, root=root)
    return t


class TestGraphFlashbackAdapter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pl = _load("gf_pipeline_test", os.path.join(MDIR, "pipeline.py"))
        cls.tmp = tempfile.TemporaryDirectory(dir=os.path.join(ROOT, "tmp_tests"))
        cls.root = cls.tmp.name
        cls.tables = _synth_root(cls.root)
        cls.cfg = cls.pl.make_config(smoke=True, kge_epochs=1, epochs=2)
        cls.work = os.path.join(cls.root, "work")
        cls.ex = cls.pl.Export(os.path.join(cls.root, "exports", "gfsyn", "H-100"))
        cls.pdir = cls.pl.prepare(cls.ex, cls.cfg, cls.work, "cpu")
        cls.tdir = cls.pl.train(cls.ex, cls.cfg, cls.work, cls.pdir, "cpu")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_patches_applied(self):
        m = self.pl.authors_modules()
        self.assertEqual(m.dataset.TRAIN_FRACTION, 1.0)
        with open(os.path.join(m.src["Graph-Flashback-KGE"], "jTransUP", "models", "knowledge_representation.py")) as f:
            self.assertIn('getattr(FLAGS, "kge_epochs", 100)', f.read())
        self.assertEqual(m.kge_flag_defaults["embedding_size"], 100)
        self.assertEqual(m.kge_flag_defaults["learning_rate"], 0.001)

    def test_prepare_outputs(self):
        with open(os.path.join(self.pdir, "DONE.json")) as f:
            info = json.load(f)
        idx = self.pl.Index.load(os.path.join(self.pdir, "index.npz"))
        self.assertEqual(info["kge"]["entity_total"], idx.U + idx.L)
        self.assertEqual(info["graphs"]["loc_graph_shape"], [idx.L, idx.L])
        self.assertEqual(info["graphs"]["user_loc_graph_shape"], [idx.U, idx.L])
        self.assertEqual(set(info["triplets_by_relation"]), {"0", "1", "2"})
        rows = self.ex.rows
        n_pairs = len(rows[["user_id", "venue_id"]].drop_duplicates())
        self.assertEqual(info["triplets_by_relation"]["0"], n_pairs)
        cnt = rows.groupby("user_id").size()
        self.assertTrue((cnt.loc[idx.train_users] >= self.cfg["min_checkins"]).all())
        self.assertTrue((cnt.reindex(idx.other_users).fillna(0) < self.cfg["min_checkins"]).all()
                        or self.cfg["smoke"])

    def test_loss_mask_counts(self):
        with open(os.path.join(self.tdir, "train_meta.json")) as f:
            meta = json.load(f)
        d = meta["data"]
        idx = self.pl.Index.load(os.path.join(self.pdir, "index.npz"))
        rows = self.ex.rows.sort_values("row_id")
        trainq = set(self.ex.queries("train")["query_id"].tolist())
        L = self.cfg["sequence_length"]
        n_pos = n_lab = 0
        for uid in idx.train_users:
            r = rows.loc[rows.user_id == uid, "row_id"].to_numpy()
            k = ((len(r) - 1) // L) * L
            n_pos += k
            n_lab += sum(int(x) in trainq for x in r[1:1 + k])
        self.assertEqual(d["n_window_positions"], n_pos)
        self.assertEqual(d["n_loss_positions"], n_lab)
        self.assertLess(n_lab, n_pos)

    def test_padding_is_causal(self):
        import torch
        b, _, _ = self.pl.load_bundle(self.tdir, "cpu")
        qw = self.pl.QueryWindows(self.ex, "val", b.idx, self.ex.queries("val")["query_id"].tolist()[:8], 20)
        k = int(np.argmax(qw.ok & (qw.length < 20)))
        ln = int(qw.length[k])
        outs = []
        for fill in (0, 1):
            x, t, s = qw.x[k].copy(), qw.t[k].copy(), qw.s[k].copy()
            if fill:
                x[ln:] = 3
                t[ln:] = t[ln - 1] + 1e6
                s[ln:] = 0.0
            X = torch.as_tensor(np.stack([x, x]).T.copy())
            T = torch.as_tensor(np.stack([t, t]).T.copy(), dtype=torch.float32)
            S = torch.as_tensor(np.stack([s, s]).transpose(1, 0, 2).copy(), dtype=torch.float32)
            au = torch.as_tensor([qw.user[k]] * 2).view(1, -1)
            with torch.no_grad():
                o, _ = b.fb_trainer.evaluate(X, T, T, S, None, None, None, b.h0_strategy.on_init(2, "cpu"), au)
            outs.append(o[0, ln - 1].numpy())
        np.testing.assert_allclose(outs[0], outs[1], rtol=0, atol=0)

    def test_predictions_and_determinism(self):
        b, _, _ = self.pl.load_bundle(self.tdir, "cpu")
        q = self.ex.queries("val")["query_id"].tolist()
        p1 = self.pl.predict_windows(b, self.ex, "val", q)
        p2 = self.pl.predict_windows(b, self.ex, "val", list(reversed(q)))
        self.assertEqual(len(p1), len(q))
        cat = set(b.idx.cat_venues.tolist())
        for lst in p1["ranked_venue_ids"]:
            self.assertEqual(len(lst), min(50, len(cat)))
            self.assertEqual(len(set(lst)), len(lst))
            self.assertTrue(set(lst) <= cat)
        m2 = p2.set_index("query_id")
        for qid, a, c in zip(p1.query_id, p1.ranked_venue_ids, p1.cand_ranked_venue_ids):
            self.assertEqual(list(a), list(m2.at[qid, "ranked_venue_ids"]))
            self.assertEqual(list(c), list(m2.at[qid, "cand_ranked_venue_ids"]))
        cands = dict(zip(self.ex.candidates("val").query_id, self.ex.candidates("val").cand_venue_ids))
        for qid, c in zip(p1.query_id, p1.cand_ranked_venue_ids):
            self.assertEqual(sorted(c), sorted(int(v) for v in cands[qid]))

    def test_full_pipeline_retrain_is_deterministic(self):
        work2 = os.path.join(self.root, "work2")
        pdir = self.pl.prepare(self.ex, self.cfg, work2, "cpu")
        tdir = self.pl.train(self.ex, self.cfg, work2, pdir, "cpu")
        b1, _, _ = self.pl.load_bundle(self.tdir, "cpu")
        b2, _, _ = self.pl.load_bundle(tdir, "cpu")
        for (k, v1), (_, v2) in zip(b1.fb_trainer.model.state_dict().items(), b2.fb_trainer.model.state_dict().items()):
            self.assertTrue(bool((v1 == v2).all()), k)

    def test_h_policy_reads_only_visible_rows(self):
        ex = self.pl.Export(os.path.join(self.root, "exports", "gfsyn", "H-50"))
        idx = self.pl.Index.build(ex, self.cfg)
        n_vis = ex.rows.groupby("user_id").size()
        self.assertTrue((n_vis.loc[idx.train_users] >= self.cfg["min_checkins"]).all())
        self.assertEqual(len(ex.rows), int(n_vis.sum()))
        ex_s = self.pl.Export(os.path.join(self.root, "exports", "gfsyn", "S-50"))
        self.assertEqual(self.pl.prepare_key(ex_s, self.cfg), self.pl.prepare_key(self.ex, self.cfg))
        self.assertNotEqual(self.pl.train_key(ex_s, self.cfg), self.pl.train_key(self.ex, self.cfg))
        self.assertNotEqual(self.pl.prepare_key(ex, self.cfg), self.pl.prepare_key(self.ex, self.cfg))


if __name__ == "__main__":
    unittest.main()
