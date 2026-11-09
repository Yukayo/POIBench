import importlib.util
import os
import types
import unittest

import numpy as np
import torch

from poibench import canary as C
from poibench import exports as E
from tests import synth
from tests.test_exports import write_synth

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN = os.path.join(ROOT, "methods", "getnext", "run.py")


def _load():
    sp = importlib.util.spec_from_file_location("getnext_run_test", RUN)
    m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m)
    return m


G = _load()

SPEC_DEFAULTS = dict(short_traj_thres=2, time_units=48, time_feature="norm_in_day_time", poi_embed_dim=128,
                     user_embed_dim=128, gcn_dropout=0.3, gcn_nhid=[32, 64], transformer_nhid=1024,
                     transformer_nlayers=2, transformer_nhead=2, transformer_dropout=0.3, time_embed_dim=32,
                     cat_embed_dim=32, time_loss_weight=10, node_attn_nhid=128, batch=20, epochs=200, lr=1e-3,
                     lr_scheduler_factor=0.1, weight_decay=5e-4)


def toy_model(cfg="released", n_nodes=30, n_cats=4, n_users=5, seed=0, device="cpu"):
    rng = np.random.default_rng(seed)
    X = np.zeros((n_nodes, 3 + n_cats), dtype=np.float32)
    node_cat = rng.integers(0, n_cats, n_nodes)
    X[:, 0] = rng.integers(0, 20, n_nodes)
    X[np.arange(n_nodes), 1 + node_cat] = 1
    X[:, -2:] = rng.normal(size=(n_nodes, 2))
    raw = (rng.random((n_nodes, n_nodes)) < 0.1).astype(float) * rng.integers(1, 5, (n_nodes, n_nodes))
    A = np.asarray(G.upstream("utils").calculate_laplacian_matrix(raw, "hat_rw_normd_lap_mat"))
    perm = rng.permutation(n_nodes) if cfg in ("released", "fixed") else np.arange(n_nodes)
    node_venue = np.arange(n_nodes) + 100
    meta = dict(X=X, A_dense=A, node_venue=node_venue, poi_venue=node_venue[perm], poi_cat=node_cat[perm],
                cat_names=[f"c{i}" for i in range(n_cats)], user_ids=np.arange(n_users),
                node_checkin_cnt=X[:, 0].astype(np.int64), n_venues_all=200)
    torch.manual_seed(seed)
    m = G.GETNext(meta, G.upstream_defaults(), cfg, device)
    m.train(False)
    return m


class TestUpstream(unittest.TestCase):
    def test_pinned_files_and_defaults(self):
        G.check_upstream()
        self.assertEqual(G.upstream_defaults(), SPEC_DEFAULTS)

    def test_time_feature_matches_release(self):
        import pandas as pd
        t = pd.to_datetime(["2012-04-03 14:00:09", "2012-04-03 04:35:00", "2012-04-03 04:12:00"])
        np.testing.assert_allclose(G.norm_in_day_time(t), [0.5833333333333334, 0.1875, 0.16666666666666666])


class TestParity(unittest.TestCase):

    def setUp(self):
        self.m = toy_model()
        self.m.train(False)

    def _ns(self, **kw):
        m = self.m
        ns = dict(torch=torch, args=types.SimpleNamespace(device="cpu"),
                  poi_idx2cat_idx_dict={p: int(c) for p, c in enumerate(m.meta["poi_cat"])},
                  user_id2idx_dict={str(u): int(u) for u in m.meta["user_ids"]},
                  user_embed_model=m.user_embed_model, time_embed_model=m.time_embed_model,
                  cat_embed_model=m.cat_embed_model, embed_fuse_model1=m.embed_fuse_model1,
                  embed_fuse_model2=m.embed_fuse_model2, node_attn_model=m.node_attn_model, X=m.X, A=m.A)
        ns.update(kw)
        return ns

    def test_input_traj_to_embeddings(self):
        ns = self._ns()
        exec(G.upstream_train_function_source("input_traj_to_embeddings"), ns)
        rng = np.random.default_rng(1)
        with torch.no_grad():
            pe = self.m.poi_embed_model(self.m.X, self.m.A)
            for u in range(3):
                pois = rng.integers(0, 30, 7)
                ts = rng.integers(0, 48, 7) / 48
                ref = torch.stack(ns["input_traj_to_embeddings"]((f"{u}_9", list(zip(pois.tolist(), ts.tolist())),
                                                                   None), pe))
                got = self.m.embed(torch.full((7,), u), torch.as_tensor(pois), torch.as_tensor(ts, dtype=torch.float),
                                   pe)
                torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-5)

    def test_adjust_pred_prob_by_graph(self):
        rng = np.random.default_rng(2)
        seqs = [rng.integers(0, 30, n).tolist() for n in (5, 2, 7)]
        y = torch.randn(3, 7, 30)
        ns = self._ns(batch_seq_lens=[len(s) for s in seqs], batch_input_seqs=seqs)
        exec(G.upstream_train_function_source("adjust_pred_prob_by_graph"), ns)
        with torch.no_grad():
            ref = ns["adjust_pred_prob_by_graph"](y)
            idx = torch.full((3, 7), -1, dtype=torch.long)
            for i, s in enumerate(seqs):
                idx[i, :len(s)] = torch.as_tensor(s)
            got = G.GETNext.adjust(y, idx, self.m.node_attn_model(self.m.X, self.m.A))
        torch.testing.assert_close(got, ref)


def _qs(user, pois, ts):
    return dict(query_id=list(range(len(pois))), user=list(user), poi=[np.asarray(p) for p in pois],
                t=[np.asarray(t, dtype=float) for t in ts], n_dropped=0)


class TestInference(unittest.TestCase):
    def _full_batch_released(self, m, user, p, t):
        k, bsz = len(p), m.hp["batch"]
        c = (k - 1) // bsz
        batch = [dict(user=user, poi=np.asarray(p[:j]), t=np.asarray(t[:j]), y_poi=np.zeros(j), y_time=np.zeros(j),
                      y_cat=np.zeros(j)) for j in range(c * bsz + 1, k + 1)]
        with torch.no_grad():
            pe = m.poi_embed_model(m.X, m.A)
            x, in_idx, *_ = G.collate(m, batch, pe)
            yp, _, _ = m.seq_forward(x)
            adj = G.GETNext.adjust(yp, in_idx, m.node_attn_model(m.X, m.A))
        return adj[len(batch) - 1, k - 1]

    def test_released_column_shortcut_equals_full_batch(self):
        m = toy_model("released")
        rng = np.random.default_rng(3)
        for k in (1, 2, 5, 20, 21, 23, 45):
            p = rng.integers(0, 30, k)
            t = rng.integers(0, 48, k) / 48
            ref = self._full_batch_released(m, 2, p, t)
            got = G.predict_scores(m, _qs([2], [p], [t]))[0]
            torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-4)

    def test_released_quirk_attends_over_batch(self):
        m = toy_model("released")
        with torch.no_grad():
            x = torch.randn(2, 4, 320)
            y1, _, _ = m.seq_forward(x)
            x2 = x.clone()
            x2[0, 2] += 1.0
            y2, _, _ = m.seq_forward(x2)
            self.assertFalse(torch.allclose(y1[1, 2], y2[1, 2]))
            torch.testing.assert_close(y1[1, 1], y2[1, 1])
            x3 = x.clone()
            x3[1, 2] += 1.0
            y3, _, _ = m.seq_forward(x3)
            torch.testing.assert_close(y1[0], y3[0])

    def test_fixed_is_causal_and_batch_invariant(self):
        m = toy_model("fixed")
        rng = np.random.default_rng(4)
        ps = [rng.integers(0, 30, n) for n in (1, 3, 8, 12)]
        ts = [rng.integers(0, 48, len(p)) / 48 for p in ps]
        together = G.predict_scores(m, _qs([0, 1, 2, 3], ps, ts))
        for i in range(4):
            alone = G.predict_scores(m, _qs([i], [ps[i]], [ts[i]]))[0]
            torch.testing.assert_close(together[i], alone, rtol=1e-4, atol=1e-4)
        with torch.no_grad():
            pe = m.poi_embed_model(m.X, m.A)
            mk = lambda p, t: [dict(user=0, poi=p, t=t, y_poi=p, y_time=t, y_cat=p)]
            x1, *_ = G.collate(m, mk(ps[2], ts[2]), pe)
            p2 = ps[2].copy()
            p2[-1] = (p2[-1] + 1) % 30
            x2, *_ = G.collate(m, mk(p2, ts[2]), pe)
            y1, _, _ = m.seq_forward(x1)
            y2, _, _ = m.seq_forward(x2)
            torch.testing.assert_close(y1[0, :-1], y2[0, :-1], rtol=1e-4, atol=1e-4)

    def test_released_unusable_queries_fall_back(self):
        m = toy_model("released")
        qs = _qs([0, -1, 1], [np.array([3, 4]), np.array([5]), np.array([], dtype=int)],
                 [np.array([.1, .2]), np.array([.3]), np.array([])])
        sc = G.predict_scores(m, qs)
        self.assertIsNotNone(sc[0])
        self.assertIsNone(sc[1])
        self.assertIsNone(sc[2])
        df, n_fb = G.rank_lists(m, qs, sc, {0: [101, 102, 101], 1: [103], 2: [104, 105]})
        self.assertEqual(n_fb, 2)
        self.assertEqual(sorted(df.cand_ranked_venue_ids[0]), [101, 102])
        self.assertEqual(len(df.ranked_venue_ids[0]), 30)


def _explicit_softmax_seq_model(sm, src, mask):
    import math
    x = src * math.sqrt(sm.embed_size) + sm.pos_encoder.pe[:src.size(0)]
    for layer in sm.transformer_encoder.layers:
        sa = layer.self_attn
        S, N, E = x.shape
        h = sa.num_heads
        d = E // h
        q, k, v = (x @ sa.in_proj_weight.T + sa.in_proj_bias).split(E, dim=-1)
        q, k, v = (t.reshape(S, N * h, d).transpose(0, 1) for t in (q, k, v))
        att = torch.softmax((q * d ** -0.5) @ k.transpose(1, 2) + mask, dim=-1)
        o = (att @ v).transpose(0, 1).reshape(S, N, E) @ sa.out_proj.weight.T + sa.out_proj.bias
        x = layer.norm1(x + o)
        x = layer.norm2(x + layer.linear2(torch.relu(layer.linear1(x))))
    return sm.decoder_poi(x), sm.decoder_time(x), sm.decoder_cat(x)


class TestAttentionNumerics(unittest.TestCase):

    def test_attention_runs_on_math_backend_only(self):
        seen = []

        def hook(mod, args, kwargs):
            b = torch.backends.cuda
            seen.append((b.math_sdp_enabled(), b.mem_efficient_sdp_enabled(), b.flash_sdp_enabled(),
                         b.cudnn_sdp_enabled()))
        for cfg in ("released", "fixed"):
            m = toy_model(cfg)
            hs = [lay.self_attn.register_forward_pre_hook(hook, with_kwargs=True)
                  for lay in m.seq_model.transformer_encoder.layers]
            m.train(True)
            m.seq_forward(torch.randn(3, 4, 320))
            m.train(False)
            G.predict_scores(m, _qs([0, 1], [np.array([1, 2]), np.array([3])], [np.array([.1, .2]), np.array([.3])]))
            for hd in hs:
                hd.remove()
        self.assertGreater(len(seen), 0)
        self.assertEqual(set(seen), {(True, False, False, False)})

    def _check_grads(self, device):
        for cfg in ("released", "fixed"):
            m = toy_model(cfg, device=device)
            torch.manual_seed(1)
            v = torch.randn(320)
            for scale in (1.0, 1000.0):
                x0 = ((v + 0.1 * torch.randn(20, 30, 320)) * scale).to(device)
                x = x0.clone().requires_grad_(True)
                yp, yt, yc = m.seq_forward(x)
                (yp.sum() + yt.sum() + yc.sum()).backward()
                ref_sm = __import__("copy").deepcopy(m.seq_model).double()
                xr = x0.double().clone().requires_grad_(True)
                if cfg == "released":
                    src, S = xr, xr.shape[0]
                else:
                    src, S = xr.transpose(0, 1), xr.shape[1]
                mask = ref_sm.generate_square_subsequent_mask(S).to(device).double()
                rp, rt, rc = _explicit_softmax_seq_model(ref_sm, src, mask)
                (rp.sum() + rt.sum() + rc.sum()).backward()
                self.assertTrue(torch.isfinite(x.grad).all(), (cfg, scale))
                g, r = x.grad.double(), xr.grad
                tol = 1e-3 * float(r.abs().max())
                torch.testing.assert_close(g, r, rtol=1e-3, atol=tol, msg=f"{cfg} scale {scale} {device}")

    def test_gradients_match_explicit_softmax_cpu(self):
        self._check_grads("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA only (the fused kernels are the CUDA default)")
    def test_gradients_match_explicit_softmax_cuda(self):
        self._check_grads("cuda")


class TestEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import tempfile
        rows, nv = synth.random_rows(n_users=30, seed=5)
        cls.t = synth.build_tables(rows, n_venues=nv)
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = cls.tmp.name
        write_synth(cls.root, "rnd", cls.t, xs=(50, 100))
        for p in ("H-50", "S-50", "H-100"):
            E.export_inputs("rnd", p, cls.root)
        cls.ck = {}
        for p, cfg in (("H-50", "released"), ("H-50", "fixed"), ("H-100", "released")):
            ed = E.export_dir("rnd", p, cls.root)
            ex = E.Export(ed)
            wd = os.path.join(cls.root, "work", "getnext", "rnd", p)
            d = G.train(ex, ed, wd, cfg, "cpu", epochs=2, user_frac=None, out_sub="train", log=lambda *a: None)
            cls.ck[(p, cfg)] = (os.path.join(wd, cfg, "train", "best.pt"), d)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_training_samples_follow_policy(self):
        for p in ("H-50", "S-50", "H-100"):
            ed = E.export_dir("rnd", p, self.root)
            ex = E.Export(ed)
            wd = os.path.join(self.root, "work", "getnext", "rnd", p)
            G.prepare(ex, wd)
            meta = G.build_meta(ex, os.path.join(wd, "prepare"), "released")
            seqs, st = G.train_sequences(ex, meta, G.upstream_defaults())
            used = sum(int((s["y_poi"] >= 0).sum()) for s in seqs)
            self.assertEqual(used, st["n_labels_used"])
            self.assertLessEqual(used, len(ex.queries("train")))
            if p.startswith("H"):
                self.assertEqual(st["n_labels_masked"], 0)
            else:
                self.assertGreater(st["n_labels_masked"], 0)
            tq = set(ex.queries("train")["query_id"].tolist())
            rid = ex.rows["row_id"].to_numpy()
            for s in seqs[:20]:
                start = int(np.searchsorted(ex.rows["traj_id"].to_numpy(), s["traj_id"]))
                for j, y in enumerate(s["y_poi"]):
                    self.assertEqual(y >= 0, int(rid[start + j + 1]) in tq)

    def test_graph_is_visible_population_only(self):
        ex = E.Export(E.export_dir("rnd", "H-50", self.root))
        pdir = os.path.join(self.root, "work", "getnext", "rnd", "H-50", "prepare")
        import pandas as pd
        nodes = pd.read_csv(os.path.join(pdir, "graph_X.csv"), dtype={"node_name/poi_id": str})
        cnt = dict(zip(nodes["node_name/poi_id"].astype(int), nodes["checkin_cnt"]))
        vis = ex.rows["venue_id"].value_counts()
        for v, c in cnt.items():
            self.assertEqual(c, int(vis.get(v, 0)))
        self.assertEqual(set(cnt), set(ex.venues.loc[ex.venues.in_catalog, "venue_id"].tolist()))

    def test_predict_outputs(self):
        for (p, cfg), (ck, d) in self.ck.items():
            ed = E.export_dir("rnd", p, self.root)
            ex = E.Export(ed)
            q = ex.queries("val")["query_id"].tolist()
            df = G.predict(ed, "val", q, ckpt=ck)
            self.assertEqual(df["query_id"].tolist(), q)
            cands = dict(zip(ex.candidates("val").query_id, ex.candidates("val").cand_venue_ids))
            for qid, top, cr in zip(df.query_id, df.ranked_venue_ids, df.cand_ranked_venue_ids):
                self.assertEqual(len(top), len(set(top)))
                self.assertTrue(1 <= len(top) <= 50)
                self.assertEqual(sorted(cr), sorted(set(int(x) for x in cands[qid])))
            again = G.predict(ed, "val", q[::-1], ckpt=ck).set_index("query_id").loc[q]
            self.assertEqual([list(x) for x in again.ranked_venue_ids], [list(x) for x in df.ranked_venue_ids])
            self.assertEqual(d["epochs"], 2)
            self.assertIsNotNone(d["best_epoch"])

    def test_normfeat_changes_only_the_count_feature(self):
        p = "H-50"
        ed = E.export_dir("rnd", p, self.root)
        ex = E.Export(ed)
        wd = os.path.join(self.root, "work", "getnext", "rnd", p)
        G.prepare(ex, wd)
        mr = G.build_meta(ex, os.path.join(wd, "prepare"), "released")
        mn = G.build_meta(ex, os.path.join(wd, "prepare"), "normfeat")
        self.assertEqual(G.method_name("normfeat"), "getnext-normfeat")
        np.testing.assert_array_equal(mr["X"][:, 1:], mn["X"][:, 1:])
        np.testing.assert_allclose(mn["X"][:, 0], np.log1p(mr["X"][:, 0].astype(np.float64)), rtol=1e-6)
        self.assertGreater(mr["X"][:, 0].max(), mn["X"][:, 0].max())
        for k in ("A_dense", "node_venue", "poi_venue", "poi_cat", "user_ids", "node_checkin_cnt"):
            np.testing.assert_array_equal(np.asarray(mr[k]), np.asarray(mn[k]), err_msg=k)
        hp = G.upstream_defaults()
        self.assertFalse(G.GETNext(mn, hp, "normfeat", "cpu").fix_mask)

    def test_zfeat_is_the_authors_zscore_of_every_column(self):
        p = "H-50"
        ed = E.export_dir("rnd", p, self.root)
        ex = E.Export(ed)
        wd = os.path.join(self.root, "work", "getnext", "rnd", p)
        G.prepare(ex, wd)
        mr = G.build_meta(ex, os.path.join(wd, "prepare"), "released")
        mz = G.build_meta(ex, os.path.join(wd, "prepare"), "zfeat")
        self.assertEqual(G.method_name("zfeat"), "getnext-zfeat")
        Xr = mr["X"].astype(np.float64)
        want, mu, sd = G.upstream("utils").get_normalized_features(Xr)
        np.testing.assert_allclose(mz["X"], want.astype(np.float32), rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(mz["X"].mean(0), 0.0, atol=1e-5)
        np.testing.assert_allclose(mz["X"].std(0), 1.0, atol=1e-4)
        for k in ("A_dense", "node_venue", "poi_venue", "poi_cat", "user_ids", "node_checkin_cnt", "cat_names"):
            np.testing.assert_array_equal(np.asarray(mr[k]), np.asarray(mz[k]), err_msg=k)
        hp = G.upstream_defaults()
        self.assertFalse(G.GETNext(mz, hp, "zfeat", "cpu").fix_mask)
        self.assertEqual(G.upstream_defaults(), hp)

    def test_resume_is_noop_when_done(self):
        p, cfg = "H-100", "released"
        ed = E.export_dir("rnd", p, self.root)
        wd = os.path.join(self.root, "work", "getnext", "rnd", p)
        before = os.path.getmtime(self.ck[(p, cfg)][0])
        d = G.train(E.Export(ed), ed, wd, cfg, "cpu", epochs=2, user_frac=None, out_sub="train", log=lambda *a: None)
        self.assertEqual(os.path.getmtime(self.ck[(p, cfg)][0]), before)
        self.assertEqual(d["best_epoch"], self.ck[(p, cfg)][1]["best_epoch"])

    def _resume_after_interrupt(self, device):
        import shutil
        p, cfg = "H-50", "released"
        ed = E.export_dir("rnd", p, self.root)
        wd = os.path.join(self.root, "work", "getnext_resume_" + device, "rnd", p)
        G.train(E.Export(ed), ed, wd, cfg, device, epochs=2, user_frac=None, out_sub="train", log=lambda *a: None)
        od = os.path.join(wd, cfg, "train")
        ck = torch.load(os.path.join(od, "last.pt"), map_location="cpu", weights_only=False)
        ck["train_state"]["epoch"] = 0
        ck["train_state"]["history"] = ck["train_state"]["history"][:1]
        torch.save(ck, os.path.join(od, "last.pt"))
        os.remove(os.path.join(od, "DONE.json"))
        msgs = []
        d = G.train(E.Export(ed), ed, wd, cfg, device, epochs=2, user_frac=None, out_sub="train", log=msgs.append)
        self.assertTrue(any("resume from" in m for m in msgs), msgs)
        self.assertEqual(d["epochs"], 2)
        shutil.rmtree(os.path.dirname(os.path.dirname(wd)), ignore_errors=True)

    def test_resume_after_interrupt_cpu(self):
        self._resume_after_interrupt("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA only (map_location=cuda put RNG states on the GPU)")
    def test_resume_after_interrupt_cuda(self):
        self._resume_after_interrupt("cuda")

    def test_canary_passes(self):
        for cfg in ("released", "fixed"):
            os.environ["GETNEXT_CKPT"] = self.ck[("H-50", cfg)][0]
            try:
                rep = C.run_canary(f"{RUN}:predict", "rnd", "H-50", "val", n=40, root=self.root,
                                   lint_paths=[os.path.dirname(RUN)], tag=f"getnext-{cfg}")
            finally:
                os.environ.pop("GETNEXT_CKPT", None)
            self.assertEqual(rep["result"], "PASS", (cfg, rep["n_mismatch"], rep["lint"], rep["exporter_invariance"]))
            self.assertGreater(rep["perturbation"]["n_rows_replaced"], 0)


if __name__ == "__main__":
    unittest.main()
