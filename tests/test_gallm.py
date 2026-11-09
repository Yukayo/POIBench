import os
import sys

import numpy as np
import pandas as pd
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "methods"))
from gallm import prompts as PR
from poibench.exports import Export, export_dir

EX = os.path.join(export_dir("nyc", "H-100"), "MANIFEST.json")
needs_export = unittest.skipUnless(os.path.exists(EX), "exports/nyc/H-100 missing")


def _rows(n, user=7, t0="2012-04-01 10:00:00", venue0=10):
    t = pd.date_range(t0, periods=n, freq="h")
    return pd.DataFrame({"venue_id": np.arange(venue0, venue0 + n), "lat": 40.0 + np.arange(n) / 100,
                         "lon": -74.0 + np.arange(n) / 100, "category": ["Bar"] * n, "t_utc": t, "t_local": t,
                         "user_id": user})


class TestGALLM(unittest.TestCase):
    def test_upstream_text_and_target_time_removal(self):
        hist, cur = _rows(3), _rows(2, t0="2012-05-01 10:00:00", venue0=50)
        cat = {"Bar": 0}
        tgt = dict(venue_id=99, lat=1.0, lon=2.0, t_local_str=PR.SENT, t_epoch=2_000_000_000)
        q, a = PR._one("llm", cur, hist, 7, 1, tgt, 100, cat)
        assert q.count(f"At {PR.SENT}, ") == 1 and a == f"<answer>: At {PR.SENT}, user 7 will visit POI id 99."
        q2, a2 = PR._strip_tt(q, a, 7)
        assert PR.SENT not in q2 and a2 == "<answer>: user 7 will visit POI id 99."
        assert "There is also earlier, history trajectory data for user 7:" in q2
        assert "with the coordinates token <GPS 50>. It is a Bar with category ID 0." in q2
        assert q2.endswith("Given the data, Which POI id will user 7 visit? Note that POI id is an integer in the range "
                           "from 0 to 100.")
        assert "POI id 99," not in q2 and "<GPS 99>" not in q2
        p = PR.inference_prompt(q2, "default")
        assert p == q2 + "<answer>: user 7 will visit POI id " and (q2 + a2).startswith(p)


    def test_tt_prefix_is_upstream_rule(self):
        hist, cur = _rows(2), _rows(2, t0="2012-05-01 10:00:00", venue0=50)
        tgt = dict(venue_id=-1, lat=0.0, lon=0.0, t_local_str="2012-05-01 13:00:00", t_epoch=2_000_000_000)
        q, _ = PR._one("llm", cur, hist, 7, 1, tgt, 100, {"Bar": 0})
        p = PR.inference_prompt(q, "tt")
        assert p.endswith("<answer>:At 2012-05-01 13:00:00,  user 7 will visit POI id ")


    def test_stage1_text(self):
        hist, cur = _rows(2), _rows(2, t0="2012-05-01 10:00:00", venue0=50)
        tgt = dict(venue_id=99, lat=1.5, lon=2.5, t_local_str=PR.SENT, t_epoch=2_000_000_000)
        q, a = PR._one("stage1", cur, hist, 7, 1, tgt, 100, {"Bar": 0})
        q, a = PR._strip_tt(q, a, 7)
        assert a == "<answer>: User 7 will visit a POI at coordinates (1.5, 2.5)."
        assert q.endswith("user 7 will visit a POI with the coordinates token <GPS 51>. What are its corresponding "
                          "latitude and longitude coordinates?")


    def test_long_trajectory_crash_fix(self):
        hist, cur = _rows(5), _rows(230, t0="2012-05-01 10:00:00", venue0=50)
        tgt = dict(venue_id=1, lat=0.0, lon=0.0, t_local_str=PR.SENT, t_epoch=2_000_000_000)
        q, _ = PR._one("llm", cur, hist, 7, 1, tgt, 1000, {"Bar": 0})
        assert "earlier, history" not in q


    @needs_export
    def test_query_prompts_use_own_context_only(self):
        ex = Export(export_dir("nyc", "H-100"))
        q = ex.queries("test")["query_id"].to_numpy()[:20]
        p = PR.query_prompts(ex, "test", q)
        assert len(p) == 20 and p["prompt"].str.endswith("will visit POI id ").all()
        assert not p["prompt"].str.contains(PR.SENT).any()


    def test_vec_encoder_equals_loop(self):
        try:
            import torch
            import nltk
        except ImportError:
            self.skipTest("torch/nltk not available")
        src = [d for d in sorted(os.listdir(os.path.join(ROOT, "work", "gallm"))) if d.startswith("src_") and os.path.exists(os.path.join(ROOT, "work", "gallm", d, "poibench_vec_geo.py"))] \
            if os.path.isdir(os.path.join(ROOT, "work", "gallm")) else []
        if not src:
            self.skipTest("no GA-LLM working copy")
        sys.path.insert(0, os.path.join(ROOT, "work", "gallm", src[-1]))
        from geo_model.fusion_geo_encoder import FusionGeoEncoder4096 as F
        from poibench_vec_geo import vec_fusion
        torch.manual_seed(0)
        e = F(gps_embed_dim=128, quadkey_length=25, n=6, vocab_size=4096, fourier_output_dim=128, final_dim=4096).eval()
        rng = np.random.default_rng(0)
        pts = [(40.7 + float(a), -74.0 + float(b)) for a, b in rng.random((50, 2)) * 0.2]
        with torch.no_grad():
            a = torch.stack([e(latitude=x, longitude=y).squeeze(0) for x, y in pts])
            b = vec_fusion(e, pts)
        assert float((a - b).abs().max()) <= 1e-5


if __name__ == "__main__":
    unittest.main()
