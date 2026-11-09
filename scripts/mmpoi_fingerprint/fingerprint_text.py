import json
import os
import sys
import zipfile

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MP = sys.argv[1]
tok = AutoTokenizer.from_pretrained(MP)
mod = AutoModel.from_pretrained(MP).eval()


def emb(texts, maxlen=384):
    out = []
    for t in texts:
        b = tok([t], padding=True, truncation=True, max_length=maxlen, return_tensors="pt")
        with torch.no_grad():
            h = mod(**b).last_hidden_state
        m = b["attention_mask"].unsqueeze(-1).float()
        v = (h * m).sum(1) / m.sum(1)
        v = torch.nn.functional.normalize(v, dim=-1)[0].numpy()
        out.append(v)
    return np.stack(out)


def load(z, n):
    d = {}
    for ln in z.read(n).split(b"\n"):
        if ln.strip():
            d.update(json.loads(ln))
    return d


zp = zipfile.ZipFile(f"{ROOT}/data_raw/mmpoi/processed_dl/NYC_processed.zip")
zr = zipfile.ZipFile(f"{ROOT}/data_raw/mmpoi/archives/NYC_WWW2024.zip")
meta_e = load(zp, "NYC/modal_meta_embedding.json")
rev_e = load(zp, "NYC/modal_review_embedding.json")
meta_lines = [ln.decode() for ln in zp.read("NYC/meta-NYC.json").split(b"\n") if ln.strip()]
rs = load(zr, "NYC/review_summary.json")
rs_by = {}
for k, v in rs.items():
    rs_by.setdefault(k.split("_", 1)[1], []).append((int(k.split("_", 1)[0]), v))

for ln in meta_lines[:4]:
    d = json.loads(ln)
    vid = d["gmap_id"]
    tgt = np.asarray(meta_e[vid], dtype=float).reshape(-1)
    nn = {k: v for k, v in d.items() if v != "null"}
    cands = {
        "line": ln, "json.dumps": json.dumps(d), "str(dict)": str(d), "category": d["category"],
        "values": " ".join(str(v) for v in d.values()), "nonnull str": str(nn),
        "nonnull values": " ".join(str(v) for v in nn.values()),
        "nonnull k: v": ", ".join(f"{k}: {v}" for k, v in nn.items()),
        "cat+lat+lon": f"{d['category']} {d['latitude']} {d['longitude']}",
    }
    E = emb(list(cands.values()))
    print("META", vid, {k: round(float(E[i] @ tgt), 4) for i, k in enumerate(cands)})

for vid in list(rev_e)[:6]:
    tgt = np.asarray(rev_e[vid], dtype=float)
    if np.linalg.norm(tgt) == 0 or vid not in rs_by:
        continue
    tips = [t for _, v in sorted(rs_by[vid]) for t in (v if isinstance(v, list) else [v])]
    cands = {"join space": " ".join(tips), "join nl": "\n".join(tips), "str(list)": str(tips),
             "first": tips[0], "json": json.dumps(tips), "join ''": "".join(tips)}
    E = emb(list(cands.values()))
    sc = {k: round(float(E[i] @ tgt), 4) for i, k in enumerate(cands)}
    T = emb(tips)
    mv = T.mean(0)
    sc["mean tips"] = round(float(mv @ tgt / np.linalg.norm(mv)), 4)
    sc["max single tip"] = round(float((T @ tgt).max()), 4)
    print("REVIEW", vid, len(tips), sorted(k for k, _ in rs_by[vid]), sc)
