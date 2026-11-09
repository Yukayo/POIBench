import base64
import io
import os
import re
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from PIL import Image

from methods.qwen25_vl_ref import run as R
from methods.qwen25_vl_ref import vlm as V
from poibench.exports import Export, export_dir


class TestParsers(unittest.TestCase):
    def test_ranking(self):
        cand = [5, 7, 9, 11]
        self.assertEqual(R.parse_ranking("[9, 5]", cand), ([9, 5, 7, 11], 2, True))
        self.assertEqual(R.parse_ranking("```json\n[11, 11, 3, 7, 5, 9]\n```", cand), ([11, 7, 5, 9], 4, True))
        self.assertEqual(R.parse_ranking("no idea", cand), ([5, 7, 9, 11], 0, False))
        self.assertEqual(R.parse_ranking("[1, 2] then [7]", cand), ([7, 5, 9, 11], 1, True))

    def test_caption(self):
        c, ok = R.parse_caption('```json\n{"summary": "A cozy bar.", "keywords": ["bar", "wood", "", "dim"]}\n```')
        self.assertTrue(ok)
        self.assertEqual(c, "A cozy bar. Keywords: bar, wood, dim.")
        c, ok = R.parse_caption('{"summary": "Truncated cafe", "keywords": ["a"')
        self.assertFalse(ok)
        self.assertEqual(c, "Truncated cafe")


NYC_EXPORT = os.path.join(export_dir("nyc", "H-100", ROOT), "MANIFEST.json")
NYC_IMAGES = os.path.join(R.mm_dir("nyc"), "venue_images.parquet")


@unittest.skipUnless(os.path.exists(NYC_EXPORT) and os.path.exists(NYC_IMAGES),
                     "integration test: needs exports/nyc/H-100 (scripts/export_inputs.py) and exports_mm/nyc "
                     "(multimodal assets, not redistributed)")
class TestRequests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ex = Export(export_dir("nyc", "H-100", ROOT))
        cls.q = [int(x) for x in cls.ex.queries("val")["query_id"].iloc[:40]]
        cls.imgs = R.load_images("nyc")
        cls.caps = {v: f"caption of {v}" for v in cls.imgs}

    def _text(self, it):
        return "".join(s[1] for _, segs in it["req"] for s in segs if s[0] == "text")

    def test_arms_paired(self):
        A = R.build_requests(self.ex, "val", self.q, "A", self.imgs, None)
        B = R.build_requests(self.ex, "val", self.q, "B", self.imgs, self.caps)
        C = R.build_requests(self.ex, "val", self.q, "C", self.imgs, None)
        for a, b, c in zip(A, B, C):
            self.assertEqual(a["cand"], b["cand"])
            self.assertEqual(a["cand"], c["cand"])
            self.assertEqual(b["n_captions"], c["n_images"])
            self.assertLessEqual(c["n_images"], R.N_CAND + R.N_HIST_PHOTO)
            self.assertEqual(V.n_images(c["req"]), c["n_images"])
            tb = re.sub(r"\n   photo description: caption of \d+", "", self._text(b))
            tc = self._text(c).replace("\n   photo: ", "")
            self.assertEqual(self._text(a), tb)
            self.assertEqual(self._text(a), tc)
            capv = [int(x) for x in re.findall(r"caption of (\d+)", self._text(b))]
            imgv = [int(os.path.basename(s[1])[:-4]) for _, segs in c["req"] for s in segs if s[0] == "image"]
            self.assertEqual(capv, imgv)
            self.assertNotIn("target", self._text(a).lower())

    def test_openai_messages(self):
        it = R.build_requests(self.ex, "val", self.q[:3], "C", self.imgs, None)[0]
        msgs = V.to_openai_messages(it["req"])
        parts = msgs[0]["content"]
        urls = [p["image_url"]["url"] for p in parts if p["type"] == "image_url"]
        self.assertEqual(len(urls), it["n_images"])
        im = Image.open(io.BytesIO(base64.b64decode(urls[0].split(",", 1)[1])))
        self.assertEqual((im.format, im.mode), ("PNG", "RGB"))
        a = R.build_requests(self.ex, "val", self.q[:1], "A", self.imgs, None)[0]
        self.assertIsInstance(V.to_openai_messages(a["req"])[0]["content"], str)

    def test_key_uses_image_hash(self):
        it = R.build_requests(self.ex, "val", self.q[:1], "C", self.imgs, None)[0]
        k1 = V.request_key("x", "rank", it["req"])
        req2 = [(r, [s if s[0] == "text" else ("image", "/elsewhere.png", s[2]) for s in segs]) for r, segs in it["req"]]
        self.assertEqual(k1, V.request_key("x", "rank", req2))
        req3 = [(r, [s if s[0] == "text" else ("image", s[1], "0" * 64) for s in segs]) for r, segs in it["req"]]
        self.assertNotEqual(k1, V.request_key("x", "rank", req3))


class TestCostOnly(unittest.TestCase):

    @unittest.skipUnless(os.path.exists(NYC_EXPORT) and os.path.exists(NYC_IMAGES),
                         "integration test: needs exports/nyc/H-100 and exports_mm/nyc (arm C reads the photos)")
    def test_cost_only_mock(self):
        import argparse
        import json
        import shutil
        tmp = os.path.join(ROOT, "work", "tmp_cost_only_test")
        shutil.rmtree(tmp, ignore_errors=True)
        os.makedirs(tmp)
        try:
            ex = Export(export_dir("nyc", "H-100", ROOT))
            a = argparse.Namespace(ds="nyc", policy="H-100", arm="C", split=["val", "test"], device="cpu",
                                   backend="mock", seed=0)
            preds = os.path.join(ROOT, "preds", "qwen2.5-vl-ref-C@H-100", "nyc")
            before = sorted((f, os.path.getmtime(os.path.join(preds, f))) for f in os.listdir(preds)) \
                if os.path.isdir(preds) else None
            old = os.environ.get("QWENVL_CACHE_DIR")
            cjs = R.run_cost_only(a, ex, None, cost_root=os.path.join(tmp, "cost"), wdir=os.path.join(tmp, "w"), limit=25,
                                  log=lambda *x: None)
            self.assertEqual(os.environ.get("QWENVL_CACHE_DIR"), old)
            for s in ("val", "test"):
                self.assertTrue(cjs[s].startswith(os.path.join(tmp, "cost")))
                c = json.load(open(cjs[s]))
                self.assertTrue(c["cost_only"])
                self.assertEqual(c["n_queries"], 25)
                infer = [st for st in c["stages"] if st["name"] == "infer"][0]
                self.assertEqual(infer["n_cache_hits"], 0)
                self.assertGreater(infer["n_generated"], 0)
                self.assertIn("preds_parity", c)
            after = sorted((f, os.path.getmtime(os.path.join(preds, f))) for f in os.listdir(preds)) \
                if os.path.isdir(preds) else None
            self.assertEqual(before, after)
            self.assertTrue(os.path.isdir(os.path.join(tmp, "w", "cache")))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_refuses_preds(self):
        from poibench import exclusive as X
        with self.assertRaises(RuntimeError):
            X.assert_outside_preds(os.path.join(ROOT, "preds", "x"))


if __name__ == "__main__":
    unittest.main()
