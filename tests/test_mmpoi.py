import ast
import importlib.util
import os
import tempfile
import types
import unittest

import numpy as np
import pandas as pd
import torch

from poibench import canary as C
from poibench import exports as E
from tests import synth
from tests.test_exports import write_synth

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN = os.path.join(ROOT, "methods", "mmpoi", "run.py")


def _load(path, name):
    sp = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m)
    return m


M = _load(RUN, "mmpoi_run_test")
TXT = _load(os.path.join(ROOT, "scripts", "mmpoi_texts.py"), "mmpoi_texts_test")

SPEC_DEFAULTS = dict(knn_k=5, A_plus_weights=[0.4, 0.6], gcn_dropout=0.3, transformer_dropout=0.1, batch=256, lr=1e-4,
                     use_A_plus=True, loss_geo_enabled=True, loss_cat_enabled=False, cat_encoder="OneHot",
                     loss_time_enabled=False, reg_enabled=False, multimodal_enabled=True, cat_loss_type="id_loss",
                     fusion_method="fusion", use_time=True, use_cat=True, cat_type="SentenceTransformer",
                     NLP_embedding_dim=768, cat_freeze=True, adjust_enabled=False, geo_graph_enabled=True,
                     multi_loss_weight=True, task_weight_negative=True, multimodal_graph_weights=[1, 1, 1, 1],
                     short_traj_thres=2, time_units=48, time_feature="norm_in_day_time", poi_embed_dim=128,
                     geo_embed_dim=128, multimodal_embed_dim=128, user_embed_dim=128, gcn_nhid=[32, 64],
                     transformer_nhid=1024, transformer_nlayers=2, transformer_nhead=2, time_embed_dim=128,
                     cat_embed_dim=32, epochs=200, lr_scheduler_factor=0.1, weight_decay=5e-4)


def toy_meta(n_nodes=30, n_cats=4, n_geo=7, n_users=5, seed=0, d=768):
    rng = np.random.default_rng(seed)
    U = M.upstream("utils")
    node_cat = rng.integers(0, n_cats, n_nodes)
    X = np.zeros((n_nodes, n_cats), dtype=np.float32)
    X[np.arange(n_nodes), node_cat] = 1
    raw = (rng.random((n_nodes, n_nodes)) < 0.1).astype(float) * rng.integers(1, 5, (n_nodes, n_nodes))
    graw = (rng.random((n_geo, n_geo)) < 0.3).astype(float)
    emb = rng.normal(size=(3, n_nodes, d)).astype(np.float32)
    return dict(X=X, A_dense=np.asarray(U.calculate_laplacian_matrix(raw, "hat_rw_normd_lap_mat")),
                X_geo=np.eye(n_geo, dtype=np.float32),
                A_geo_dense=np.asarray(U.calculate_laplacian_matrix(graw, "hat_rw_normd_lap_mat")),
                X_image=emb[0], X_meta=emb[1], X_review=emb[2], node_venue=np.arange(n_nodes) + 100,
                poi_venue=np.arange(n_nodes) + 100, poi_cat=node_cat, num_cats=n_nodes,
                poi_geo=rng.integers(0, n_geo, n_nodes), user_ids=np.arange(n_users),
                node_checkin_cnt=rng.integers(0, 9, n_nodes), n_venues_all=200, variant="image")


def toy_model(seed=0, **kw):
    torch.manual_seed(seed)
    hp = M.upstream_defaults()
    m = M.MMPOI(toy_meta(seed=seed, **kw), hp, "cpu")
    m.train(False)
    return m


class TestUpstream(unittest.TestCase):
    def test_pinned_files_and_defaults(self):
        M.check_upstream()
        hp = M.upstream_defaults()
        self.assertEqual(hp, SPEC_DEFAULTS)
        M.check_default_code_path(hp)

    def test_no_eval_on_data(self):
        for p in (RUN, os.path.join(ROOT, "scripts", "mmpoi_texts.py")):
            tree = ast.parse(open(p).read())
            calls = {n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
            self.assertFalse(calls & {"eval", "exec"}, p)
            attrs = {(getattr(n.func.value, "id", None), n.func.attr) for n in ast.walk(tree)
                     if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
            self.assertNotIn(("pickle", "load"), attrs, p)

    def test_texts_parser_rejects_code(self):
        ok, nk = TXT.parse_review_summary(b'{"3_ABC": ["a", "b"]}\n{"1_abc": ["c"]}\n\n{"2_def": []}\n')
        self.assertEqual(ok, {"abc": ["c", "a", "b"], "def": []})
        self.assertEqual(nk, {"abc": 2, "def": 1})
        for bad in (b"{'1_abc': __import__('os').getcwd()}\n", b'{"1_abc": "not a list"}\n'):
            with self.assertRaises(ValueError):
                TXT.parse_review_summary(bad)

    def test_geohash(self):
        bg = M.upstream("build_graph")
        self.assertEqual(bg.geohash_encode(57.64911, 10.40744, precision=11), "u4pruydqqvj")
        fb = M._geohash_fallback()
        rng = np.random.default_rng(0)
        pts = np.c_[rng.uniform(40.4, 41.0, 500), rng.uniform(-74.3, -73.6, 500)]
        try:
            import geohash as real
        except ImportError:
            real = None
        for la, lo in pts:
            g = fb.encode(la, lo, 6)
            self.assertEqual(len(g), 6)
            if real is not None and not getattr(real, "__poibench_fallback__", False):
                self.assertEqual(g, real.encode(la, lo, 6))


class TestParity(unittest.TestCase):

    def test_input_traj_to_embeddings_with_geo(self):
        m = toy_model()
        ns = dict(torch=torch, args=types.SimpleNamespace(device="cpu", use_time=True), dataset_name="NYC",
                  poi_idx2geo_idx_dict={p: int(g) for p, g in enumerate(m.meta["poi_geo"])},
                  user_id2idx_dict={str(u): int(u) for u in m.meta["user_ids"]},
                  user_embed_model=m.user_embed_model, time_embed_model=m.time_embed_model,
                  embed_fuse_multimodal_model=m.embed_fuse_multimodal_model, embed_fuse_model1=m.embed_fuse_model1,
                  embed_fuse_model2=m.embed_fuse_model2)
        exec(M.upstream_train_function_source("input_traj_to_embeddings_with_geo"), ns)
        rng = np.random.default_rng(1)
        with torch.no_grad():
            Em = m.graph_embeddings()
            for u in range(3):
                pois = rng.integers(0, 30, 7)
                ts = rng.integers(0, 48, 7) / 48
                ref = torch.stack(ns["input_traj_to_embeddings_with_geo"](
                    (f"{u}_9", list(zip(pois.tolist(), ts.tolist())), None), Em["poi"], Em["geo"], Em["image"],
                    Em["meta"], Em["review"], Em["review"]))
                got = m.embed(torch.full((7,), u), torch.as_tensor(pois), torch.as_tensor(ts, dtype=torch.float), Em)
                torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-5)
        self.assertEqual(got.shape[-1], 896)

    def test_task_weight_precision_is_stale_as_released(self):
        w_up = torch.nn.Parameter(torch.tensor(0.5))
        prec_up = torch.exp(-w_up)
        w_up.data.fill_(1.7)
        loss_poi = torch.tensor(3.0)
        (prec_up * loss_poi + w_up).backward(retain_graph=True)
        m = toy_model()
        w = m.task_weights[0]
        with torch.no_grad():
            w.fill_(1.7)
        p = m.precisions()[0]
        (p * loss_poi + w).backward()
        self.assertAlmostEqual(float(p.detach()), float(prec_up.detach()), places=6)
        self.assertAlmostEqual(float(w.grad), float(w_up.grad), places=6)

    def test_seeded_init_order(self):
        a, b = toy_model(seed=3), toy_model(seed=3)
        for x, y in zip(a.optim_parameters(), b.optim_parameters()):
            torch.testing.assert_close(x, y)
        self.assertEqual(len(a.optim_parameters()) + len(list(a.cat_embed_model.parameters())),
                         sum(len(list(mm.parameters())) for mm in a.modules()) + 4)


class TestInference(unittest.TestCase):
    def test_column_shortcut_equals_full_prefix_batch(self):
        m = toy_model()
        rng = np.random.default_rng(4)
        for k in (1, 6, 300):
            pois = rng.integers(0, 30, k)
            ts = rng.integers(0, 48, k) / 48.0
            qs = dict(query_id=[7], user=[2], poi=[pois], t=[ts])
            got = M.predict_scores(m, qs)[0]
            with torch.no_grad():
                Em = m.graph_embeddings()
                b0 = ((k - 1) // 256) * 256
                seqs = [dict(user=2, poi=pois[:j], t=ts[:j], y_poi=np.zeros(j), y_time=np.zeros(j),
                             y_cat=np.zeros(j), y_geo=np.zeros(j)) for j in range(b0 + 1, k + 1)]
                x = M.collate(m, seqs, Em)[0]
                ref = m.seq_forward(x)[0][len(seqs) - 1, k - 1]
            torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-4)

    def test_unusable_queries_fall_back(self):
        m = toy_model()
        qs = dict(query_id=[1, 2], user=[-1, 0], poi=[np.array([3]), np.array([], dtype=np.int64)],
                  t=[np.array([0.1]), np.array([])])
        sc = M.predict_scores(m, qs)
        self.assertEqual(sc, [None, None])
        df, n_fb = M.rank_lists(m, qs, sc, None)
        self.assertEqual(n_fb, 2)
        self.assertEqual(len(df.ranked_venue_ids[0]), 30)


class FakeEncoder:
    revision, backend = "test", "fake"

    def encode(self, texts, batch=64):
        out = np.zeros((len(texts), 768), dtype=np.float32)
        for i, t in enumerate(texts):
            r = np.random.default_rng(abs(hash(t)) % (2 ** 32)).normal(size=768)
            out[i] = r / np.linalg.norm(r)
        return out


def write_mm(root, ds, venues):
    import hashlib
    import io
    import json
    import tarfile
    import zipfile
    from PIL import Image
    d = os.path.join(root, "exports_mm", ds)
    os.makedirs(d, exist_ok=True)
    recs, trec, members = [], [], []
    for v in venues.venue_id.astype(int):
        for j in range(1, (v % 3) + 1):
            buf = io.BytesIO()
            Image.new("RGB", (8, 8), ((v * 7 + j) % 255, 0, 0) if not (v == 4 and j == 1) else (1, 2, 3)).save(
                buf, format="PNG")
            name = f"downloaded_multimodal_data/gmap_{v}_{v:024x}_{j}.png"
            members.append((name, buf.getvalue()))
            recs.append(dict(venue_id=v, j=j, member=name, sha256=hashlib.sha256(buf.getvalue()).hexdigest(),
                             nbytes=len(buf.getvalue())))
        tips = [f"tip {v} {j}" for j in range(v % 4)]
        trec.append(dict(venue_id=v, has_tips=bool(tips), n_tips=len(tips), tips=tips, n_source_keys=1))
    tb = io.BytesIO()
    with tarfile.open(fileobj=tb, mode="w:gz") as tf:
        for name, b in members[::-1]:
            ti = tarfile.TarInfo(name)
            ti.size = len(b)
            tf.addfile(ti, io.BytesIO(b))
    os.makedirs(os.path.join(root, "data_raw", "mmpoi", "archives"), exist_ok=True)
    arch = os.path.join(root, "data_raw", "mmpoi", "archives", "SYN.zip")
    with zipfile.ZipFile(arch, "w") as z:
        z.writestr("NYC/image.tgz", tb.getvalue())
    pd.DataFrame(recs).to_parquet(os.path.join(d, "venue_photos_all.parquet"), index=False)
    with open(os.path.join(d, "MANIFEST_photos_all.json"), "w") as f:
        json.dump(dict(archive="data_raw/mmpoi/archives/SYN.zip", archive_sha256=M._sha256(arch),
                       tar_member="NYC/image.tgz", rule="all"), f)
    pd.DataFrame(trec).to_parquet(os.path.join(d, "venue_texts.parquet"), index=False)
    return pd.DataFrame(recs)


class TestEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rows, nv = synth.random_rows(n_users=30, seed=5)
        cls.t = synth.build_tables(rows, n_venues=nv)
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = cls.tmp.name
        write_synth(cls.root, "rnd", cls.t, xs=(50, 100))
        for p in ("H-50", "S-50", "H-100"):
            E.export_inputs("rnd", p, cls.root)
        cls.photos = write_mm(cls.root, "rnd", E.Export(E.export_dir("rnd", "H-100", cls.root)).venues)
        cls.caps = []

        def captioner(blobs):
            import hashlib
            cls.caps.append(len(blobs))
            return [f"a photo {hashlib.sha256(b).hexdigest()[:8]}" for b in blobs]
        cls.emb = M.embed("rnd", "cpu", encoder=FakeEncoder(), captioner=captioner, root=cls.root,
                          log=lambda *a: None)
        cls.ck = {}
        for p, var in (("H-50", "image"), ("H-50", "noimage"), ("H-100", "image")):
            ed = E.export_dir("rnd", p, cls.root)
            ex = E.Export(ed)
            wd = os.path.join(cls.root, "work", "mmpoi", "rnd", p)
            d = M.train(ex, ed, wd, var, "cpu", epochs=2, user_frac=None, out_sub="train", emb_root=cls.root,
                        log=lambda *a: None)
            cls.ck[(p, var)] = (os.path.join(wd, var, "train", "best.pt"), d)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_embeddings(self):
        z, man = M.load_embeddings("rnd", E.Export(E.export_dir("rnd", "H-50", self.root)).venues, self.root)
        self.assertTrue(np.allclose(np.linalg.norm(z["image"][z["has_image"]], axis=1), 1, atol=1e-5))
        self.assertTrue((z["image"][~z["has_image"]] == 0).all())
        self.assertTrue((z["review"][~z["has_tips"]] == 0).all())
        self.assertTrue(np.allclose(np.linalg.norm(z["meta"], axis=1), 1, atol=1e-5))
        enc = FakeEncoder()
        v = int(z["venue_id"][z["has_tips"]][0])
        tips = [f"tip {v} {j}" for j in range(v % 4)]
        ref = enc.encode(tips).mean(0)
        np.testing.assert_allclose(z["review"][list(z["venue_id"]).index(v)], ref / np.linalg.norm(ref), atol=1e-6)
        self.assertEqual(self.caps, [len(self.photos)])
        self.assertEqual(int(z["has_image"].sum()), self.photos.venue_id.nunique())
        v = int(self.photos.venue_id[self.photos.venue_id.duplicated()].iloc[0])
        caps = [f"a photo {h[:8]}" for h in self.photos.loc[self.photos.venue_id == v, "sha256"]]
        ref = enc.encode(caps).mean(0)
        np.testing.assert_allclose(z["image"][list(z["venue_id"]).index(v)], ref / np.linalg.norm(ref), atol=1e-6)
        again = M.embed("rnd", "cpu", encoder=FakeEncoder(), captioner=None, root=self.root, log=lambda *a: None)
        self.assertEqual(again["embeddings_sha256"], man["embeddings_sha256"])

    def test_noimage_variant_only_removes_image(self):
        ex = E.Export(E.export_dir("rnd", "H-50", self.root))
        pdir = os.path.join(self.root, "work", "mmpoi", "rnd", "H-50", "prepare")
        z, _ = M.load_embeddings("rnd", ex.venues, self.root)
        a, b = M.build_meta(ex, pdir, "image", z), M.build_meta(ex, pdir, "noimage", z)
        self.assertTrue((b["X_image"] == 0).all())
        self.assertGreater(np.abs(a["X_image"]).sum(), 0)
        for k in ("X", "A_dense", "X_geo", "A_geo_dense", "X_meta", "X_review", "poi_venue", "poi_geo", "poi_cat"):
            np.testing.assert_array_equal(np.asarray(a[k]), np.asarray(b[k]), err_msg=k)
        self.assertEqual(M.method_name("noimage"), "mmpoi-noimage")

    def test_training_samples_follow_policy(self):
        z, _ = M.load_embeddings("rnd", E.Export(E.export_dir("rnd", "H-50", self.root)).venues, self.root)
        for p in ("H-50", "S-50", "H-100"):
            ed = E.export_dir("rnd", p, self.root)
            ex = E.Export(ed)
            wd = os.path.join(self.root, "work", "mmpoi", "rnd", p)
            M.prepare(ex, wd)
            meta = M.build_meta(ex, os.path.join(wd, "prepare"), "image", z)
            seqs, st = M.train_sequences(ex, meta, M.upstream_defaults())
            used = sum(int((s["y_poi"] >= 0).sum()) for s in seqs)
            self.assertEqual(used, st["n_labels_used"])
            self.assertLessEqual(used, len(ex.queries("train")))
            self.assertEqual([s["key"] for s in seqs], sorted(s["key"] for s in seqs))
            if p.startswith("H"):
                self.assertEqual(st["n_labels_masked"], 0)
            else:
                self.assertGreater(st["n_labels_masked"], 0)
            for s in seqs:
                ok = s["y_poi"] >= 0
                np.testing.assert_array_equal(s["y_geo"][ok], meta["poi_geo"][s["y_poi"][ok]])
                self.assertTrue((s["y_geo"][~ok] == -1).all())

    def test_graphs_are_visible_population_and_catalog(self):
        ex = E.Export(E.export_dir("rnd", "H-50", self.root))
        pdir = os.path.join(self.root, "work", "mmpoi", "rnd", "H-50", "prepare")
        nodes = pd.read_csv(os.path.join(pdir, "graph_X.csv"), dtype={"node_name/poi_id": str, "geohash": str})
        geo = pd.read_csv(os.path.join(pdir, "geo_graph_X.csv"), dtype={"geohash": str})
        cnt = dict(zip(nodes["node_name/poi_id"].astype(int), nodes["checkin_cnt"]))
        vis = ex.rows["venue_id"].value_counts()
        for v, c in cnt.items():
            self.assertEqual(c, int(vis.get(v, 0)))
        self.assertEqual(set(cnt), set(ex.venues.loc[ex.venues.in_catalog, "venue_id"].tolist()))
        self.assertTrue(set(nodes["geohash"]) <= set(geo["geohash"]))
        self.assertEqual(int(geo["checkin_cnt"].sum()), len(ex.rows))

    def test_predict_outputs(self):
        for (p, var), (ck, d) in self.ck.items():
            ed = E.export_dir("rnd", p, self.root)
            ex = E.Export(ed)
            q = ex.queries("val")["query_id"].tolist()
            df = M.predict(ed, "val", q, ckpt=ck)
            self.assertEqual(df["query_id"].tolist(), q)
            cands = dict(zip(ex.candidates("val").query_id, ex.candidates("val").cand_venue_ids))
            for qid, top, cr in zip(df.query_id, df.ranked_venue_ids, df.cand_ranked_venue_ids):
                self.assertEqual(len(top), len(set(top)))
                self.assertTrue(1 <= len(top) <= 50)
                self.assertEqual(sorted(cr), sorted(set(int(x) for x in cands[qid])))
            again = M.predict(ed, "val", q[::-1], ckpt=ck).set_index("query_id").loc[q]
            self.assertEqual([list(x) for x in again.ranked_venue_ids], [list(x) for x in df.ranked_venue_ids])
            self.assertEqual(d["epochs"], 2)
            self.assertIsNotNone(d["best_epoch"])

    def test_write_preds_cost(self):
        import json
        p, var = "H-50", "image"
        ed = E.export_dir("rnd", p, self.root)
        wd = os.path.join(self.root, "work", "mmpoi", "rnd", p)
        real = M.candidates_file_decl
        M.candidates_file_decl = lambda ex, split: f"data/rnd/evidence/{p}.candidates_{split}.parquet"
        try:
            pqs = [M.write_preds(E.Export(ed), ed, wd, v, "val", "cpu", out_root=self.root, emb_root=self.root,
                                 log=lambda *a: None) for v in ("image", "noimage")]
        finally:
            M.candidates_file_decl = real
        for v, pq in zip(("image", "noimage"), pqs):
            c = json.load(open(pq.replace(".parquet", ".cost.json")))
            names = [s["name"] for s in c["stages"]]
            self.assertIn("embed_text_meta_review", names)
            self.assertEqual("embed_caption_images" in names, v == "image")
            self.assertIn("train", names)
            self.assertIn("predict_val", names)
            self.assertTrue(c["candidates_file"].endswith("candidates_val.parquet"))

    def test_resume_is_noop_when_done(self):
        p, var = "H-100", "image"
        ed = E.export_dir("rnd", p, self.root)
        wd = os.path.join(self.root, "work", "mmpoi", "rnd", p)
        before = os.path.getmtime(self.ck[(p, var)][0])
        d = M.train(E.Export(ed), ed, wd, var, "cpu", epochs=2, user_frac=None, out_sub="train", emb_root=self.root,
                    log=lambda *a: None)
        self.assertEqual(os.path.getmtime(self.ck[(p, var)][0]), before)
        self.assertEqual(d["best_epoch"], self.ck[(p, var)][1]["best_epoch"])

    def test_resume_after_interrupt(self):
        import shutil
        p, var = "H-50", "image"
        ed = E.export_dir("rnd", p, self.root)
        wd = os.path.join(self.root, "work", "mmpoi_resume", "rnd", p)
        full = M.train(E.Export(ed), ed, wd, var, "cpu", epochs=2, user_frac=None, out_sub="train",
                       emb_root=self.root, log=lambda *a: None)
        od = os.path.join(wd, var, "train")
        ck = torch.load(os.path.join(od, "last.pt"), map_location="cpu", weights_only=False)
        ck["train_state"]["epoch"] = 0
        ck["train_state"]["history"] = ck["train_state"]["history"][:1]
        torch.save(ck, os.path.join(od, "last.pt"))
        os.remove(os.path.join(od, "DONE.json"))
        msgs = []
        d = M.train(E.Export(ed), ed, wd, var, "cpu", epochs=2, user_frac=None, out_sub="train", emb_root=self.root,
                    log=msgs.append)
        self.assertTrue(any("resume from" in m for m in msgs), msgs)
        self.assertEqual(d["epochs_run"], 2)
        self.assertEqual(full["epochs_run"], 2)
        shutil.rmtree(os.path.join(self.root, "work", "mmpoi_resume"), ignore_errors=True)

    def test_canary_passes(self):
        for var in ("image", "noimage"):
            os.environ["MMPOI_CKPT"] = self.ck[("H-50", var)][0]
            try:
                rep = C.run_canary(f"{RUN}:predict", "rnd", "H-50", "val", n=40, root=self.root,
                                   lint_paths=[os.path.dirname(RUN)], tag=f"mmpoi-{var}")
            finally:
                os.environ.pop("MMPOI_CKPT", None)
            self.assertEqual(rep["result"], "PASS", (var, rep["n_mismatch"], rep["lint"], rep["exporter_invariance"]))
            self.assertGreater(rep["perturbation"]["n_rows_replaced"], 0)


if __name__ == "__main__":
    unittest.main()
