import ast
import json
import os
import unittest

import numpy as np
import pandas as pd

from methods.gnpr_sid import data as D
from methods.gnpr_sid import upstream as U
from poibench import ROOT
from poibench.canary import lint_sources
from poibench.exports import Export

EXP = os.path.join(ROOT, "exports", "nyc")


def authors_trajectory_sequences(ex: Export) -> pd.DataFrame:
    r = ex.rows
    df = pd.DataFrame({"UserId": r["user_id"], "PoiId": r["venue_id"], "UTCTimeOffset": r["t_local"],
                       "pseudo_session_trajectory_id": r["traj_id"], "SplitTag": "train", "_row": r["row_id"]})
    df = df.sort_values(["UserId", "_row"]).reset_index(drop=True)
    results = []
    for user_id, user_df in df.groupby("UserId", sort=False):
        user_df = user_df.reset_index(drop=True)
        for (traj_id, split_tag), traj_df in user_df.groupby(["pseudo_session_trajectory_id", "SplitTag"], sort=False):
            traj_df = traj_df.reset_index(drop=True)
            start_row = traj_df["_row"].iloc[0]
            history_df = user_df[user_df["_row"] < start_row].copy()
            merged_df = pd.concat([history_df, traj_df], axis=0).sort_values("_row")
            if len(merged_df) > 50:
                merged_df = merged_df.iloc[-50:].reset_index(drop=True)
            else:
                merged_df = merged_df.reset_index(drop=True)
            if split_tag == "train" and len(merged_df) < 20:
                continue
            results.append({"UserId": user_id, "sequence_PoiId": merged_df["PoiId"].tolist(),
                            "target_row": int(merged_df["_row"].iloc[-1])})
    return pd.DataFrame(results)


@unittest.skipUnless(os.path.exists(os.path.join(EXP, "H-10", "MANIFEST.json")), "nyc exports not built")
class TestInputsAgainstAuthors(unittest.TestCase):
    def check_policy(self, pol):
        ex = Export(os.path.join(EXP, pol))
        ours, st = D.sft_samples(ex, None)
        ref = authors_trajectory_sequences(ex)
        trainq = set(ex.queries("train")["query_id"].astype(int))
        ref = ref[[len(s) >= 2 for s in ref["sequence_PoiId"]] & ref["target_row"].isin(trainq).to_numpy()]
        ref = ref.groupby("UserId", group_keys=False).tail(D.KEEP_LAST_K).reset_index(drop=True)
        self.assertEqual(len(ours), len(ref))
        for s, (_, r) in zip(ours, ref.iterrows()):
            seq = [int(x) for x in r["sequence_PoiId"]]
            self.assertEqual(s["output"], str(seq[-1]))
            body = s["input"].split("checkin history: ", 1)[1].split(".\nNext, ", 1)[0]
            self.assertEqual([int(p.split(" visited ")[1]) for p in body.split(", ")], seq[:-1])
            self.assertTrue(s["input"].startswith(f"User_{int(r['UserId'])} checkin history: "))

    def test_sft_samples_h10(self):
        self.check_policy("H-10")

    def test_sft_samples_s10(self):
        self.check_policy("S-10")

    def test_poi_info_hours(self):
        ex = Export(os.path.join(EXP, "H-10"))
        pi = D.poi_info(ex)
        cat = ex.venues[ex.venues["in_catalog"]]
        self.assertEqual(pi["pid"].tolist(), sorted(cat["venue_id"].astype(int).tolist()))
        r = ex.rows
        h = pd.to_datetime(r["t_local"]).dt.hour
        for pid in pi["pid"].to_numpy()[:200]:
            want = h[r["venue_id"] == pid].value_counts().to_dict()
            got = ast.literal_eval(pi.loc[pi["pid"] == pid, "visit_time_and_count"].iloc[0])
            self.assertEqual(got, {int(k): int(v) for k, v in want.items()})
        self.assertIn("{}", set(pi["visit_time_and_count"]))

    def test_eval_prompt_window_and_no_target_time(self):
        ex = Export(os.path.join(EXP, "H-100"))
        qi = ex.query_inputs("val")
        qids = qi.table["query_id"].to_numpy()[:50]
        p2c = {int(v): f"<a_{int(v) % 64}><b_{int(v) // 64 % 64}><c_{int(v) % 7}>" for v in ex.venues["venue_id"]}
        pr = D.eval_prompts(ex, "val", qids, p2c)
        for i, (q, prompt) in enumerate(zip(pr["query_id"], pr["prompt"])):
            own = qi.own(i, "venue_id")
            exp = [p2c[int(v)] for v in own][-(D.HIST_WINDOW - 1):]
            self.assertTrue(prompt.endswith("Next, user_%d is likely to visit:\n\n### Response:\n"
                                            % int(qi.table["user_id"].iloc[i])))
            got = [x.split(" visited ")[1] for x in prompt.split("checkin history: ", 1)[1]
                   .split(".\nNext, ", 1)[0].split(", ")]
            self.assertEqual(got, exp)
            self.assertNotIn("When ", prompt)
            self.assertIn(D.INSTRUCTION, prompt)


@unittest.skipUnless(os.path.exists(os.path.join(EXP, "H-10", "MANIFEST.json")), "nyc exports not built")
class TestFullPipelineGuard(unittest.TestCase):
    def test_predict_refuses_other_training_inputs(self):
        import tempfile
        from methods.gnpr_sid import run as R
        ex10 = Export(os.path.join(EXP, "H-10"))
        fp, _, _ = R.inputs_fingerprint(ex10)
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "train_meta.json"), "w") as f:
                json.dump({"fingerprint": fp}, f)
            old = os.environ.get("GNPR_SID_CKPT")
            os.environ["GNPR_SID_CKPT"] = d
            try:
                with self.assertRaises(RuntimeError):
                    R.predict(os.path.join(EXP, "H-100"), "val", [], None)
            finally:
                if old is None:
                    os.environ.pop("GNPR_SID_CKPT")
                else:
                    os.environ["GNPR_SID_CKPT"] = old


class TestTextAndEdits(unittest.TestCase):
    def test_encode_item_matches_bundled_data(self):
        self.assertEqual(D.encode_item([60, 47, 58, 0]), "<a_60><b_47><c_58><d_0>")
        with open(os.path.join(U.CLONE, "V1", "datasets", "nyc", "llm_train.json")) as f:
            s = json.load(f)[0]
        self.assertEqual(s["instruction"], D.INSTRUCTION_UPSTREAM)
        self.assertIn("\nWhen ", s["input"])

    def test_instruction_is_minimal_edit(self):
        a, b = D.INSTRUCTION_UPSTREAM, D.INSTRUCTION
        self.assertEqual(a.replace("at the specified time.", "next."), b)

    def test_prompt_equals_format_fn_prefix(self):
        txt = U.edited_sft_text()
        self.assertIn('f"### Response:\\n{resp.strip()}<|eot_id|>"', txt)
        full = (f"### Instruction:\n{D.INSTRUCTION}\n\n### Input:\nX\n\n### Response:\n<a_1><|eot_id|>")
        self.assertTrue(full.startswith(D.prompt_text("X")))

    def test_sft_edits_apply_once_and_keep_hyperparameters(self):
        txt = U.edited_sft_text()
        for s in ('r=16,', 'lora_alpha=32,', 'lora_dropout=0.05,', 'warmup_steps=100,', 'bf16=True,',
                  'response_template = "### Response:\\n"', 'max_seq_length=cutoff_len,',
                  'target_modules=["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"],',
                  'eval_strategy="no",', 'report_to="none",', 'seed=seed,'):
            self.assertIn(s, txt)
        self.assertNotIn("import wandb", txt)
        self.assertNotIn('eval_steps=50', txt)

    def test_canary_lint_clean(self):
        self.assertEqual(lint_sources([os.path.join(ROOT, "methods", "gnpr_sid")]), [])


@unittest.skipUnless(os.path.exists(os.path.join(EXP, "H-100", "MANIFEST.json")), "nyc exports not built")
class TestTargetTimeCfg(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.ex = Export(os.path.join(EXP, "H-100"))
        cls.p2c = {int(v): f"<a_{int(v) % 64}><b_{int(v) // 64 % 64}><c_{int(v) % 7}>" for v in cls.ex.venues["venue_id"]}

    def test_default_training_inputs_unchanged(self):
        from methods.gnpr_sid import run as R
        tm_p = os.path.join(ROOT, "work", "gnpr_sid", "runs", "nyc", "H-100", "train_meta.json")
        if not os.path.exists(tm_p):
            self.skipTest("no finished nyc H-100 run")
        with open(tm_p) as f:
            tm = json.load(f)
        fp, _, _ = R.inputs_fingerprint(self.ex)
        self.assertEqual(fp["sft_samples_venue_sha256"], tm["fingerprint"]["sft_samples_venue_sha256"])
        self.assertEqual(fp["poi_info_sha256"], tm["fingerprint"]["poi_info_sha256"])
        self.assertEqual(R.method_name(), "gnpr-sid")
        self.assertEqual(R.method_name("tt"), "gnpr-sid-tt")
        self.assertEqual(R.work_dir_for("runs", "nyc", "H-100"),
                         os.path.join(ROOT, "work", "gnpr_sid", "runs", "nyc", "H-100"))
        self.assertNotEqual(R.work_dir_for("runs", "nyc", "H-100", "tt"), R.work_dir_for("runs", "nyc", "H-100"))

    def test_sft_samples_tt(self):
        d, _ = D.sft_samples(self.ex, self.p2c)
        t, st = D.sft_samples(self.ex, self.p2c, "tt")
        self.assertEqual(len(d), len(t))
        self.assertGreater(len(t), 100)
        rows = self.ex.rows.sort_values("row_id")
        tstr = set(D._fmt_times(rows["t_local"]))
        for a, b in zip(d, t):
            self.assertEqual(a["output"], b["output"])
            self.assertEqual(a["instruction"], D.INSTRUCTION)
            self.assertEqual(b["instruction"], D.INSTRUCTION_UPSTREAM)
            ha, hb = a["input"].split(".\nNext, ")[0], b["input"].split(".\nWhen ")[0]
            self.assertEqual(ha, hb)
            tail = b["input"].split(".\nWhen ", 1)[1]
            tt, rest = tail[:19], tail[19:]
            self.assertIn(tt, tstr)
            self.assertRegex(rest, r"^ user_\d+ is likely to visit:$")

    def test_eval_prompts_tt(self):
        qi = self.ex.query_inputs("val")
        qids = qi.table["query_id"].to_numpy()[:50]
        tt = D.load_target_time("nyc", "val")
        with self.assertRaises(ValueError):
            D.eval_prompts(self.ex, "val", qids, self.p2c, "tt")
        a = D.eval_prompts(self.ex, "val", qids, self.p2c)
        b = D.eval_prompts(self.ex, "val", qids, self.p2c, "tt", tt)
        for i, (q, pa, pb) in enumerate(zip(a["query_id"], a["prompt"], b["prompt"])):
            u = int(qi.table["user_id"].iloc[i])
            want = D._fmt_times([tt[int(q)]])[0]
            self.assertTrue(pb.endswith(f"When {want} user_{u} is likely to visit:\n\n### Response:\n"))
            self.assertIn(D.INSTRUCTION_UPSTREAM, pb)
            self.assertEqual(pa.replace(D.INSTRUCTION, D.INSTRUCTION_UPSTREAM)
                             .replace(f"Next, user_{u} is likely", f"When {want} user_{u} is likely"), pb)

    def test_input_text_matches_authors_template(self):
        with open(os.path.join(U.CLONE, "V1", "datasets", "nyc", "llm_train.json")) as f:
            s = json.load(f)[0]
        hist, tail = s["input"].split(".\nWhen ", 1)
        uid = int(hist.split("User_", 1)[1].split(" ", 1)[0])
        items = hist.split("checkin history: ", 1)[1].split(", ")
        times = [x.split(" visited ")[0] for x in items]
        codes = [x.split(" visited ")[1] for x in items]
        self.assertEqual(D.input_text(uid, times, codes, tail[:19]), s["input"])

    def test_predict_refuses_cfg_mismatch(self):
        import tempfile
        from methods.gnpr_sid import run as R
        fp, _, _ = R.inputs_fingerprint(self.ex)
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "train_meta.json"), "w") as f:
                json.dump({"fingerprint": fp, "cfg": "default"}, f)
            old = {k: os.environ.get(k) for k in ("GNPR_SID_CKPT", "GNPR_SID_CFG")}
            os.environ.update(GNPR_SID_CKPT=d, GNPR_SID_CFG="tt")
            try:
                with self.assertRaises(RuntimeError):
                    R.predict(os.path.join(EXP, "H-100"), "val", [], None)
            finally:
                for k, v in old.items():
                    if v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = v


if __name__ == "__main__":
    unittest.main()
