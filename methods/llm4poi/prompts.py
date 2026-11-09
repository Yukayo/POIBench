from __future__ import annotations

import ast
import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
UPSTREAM_DIR = os.path.join(ROOT, "third_party", "LLM4POI")
UPSTREAM_COMMIT = "83a44f4eece089828f58bae258b4692592eb3213"
UPSTREAM_FILE = "v2/convert_prompt_llm4poi.py"
UPSTREAM_SHA256 = "9455b1656d08317fd60f74abb6a45e55fba70e764e8dee7b7d6ad093d7875a98"

HISTORY_LIMIT = 50
TIME_FMT = "%Y-%m-%d %H:%M:%S"
TT_CLAUSE = "at {target_time}, "
SYSTEM = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
DATASET_NAME = {"nyc": "NYC", "tky": "TKY", "istanbul": "Istanbul", "nyc_trainfilter": "NYC",
                "tky_trainfilter": "TKY"}
CFGS = {
    "default": dict(target_time=False, method="llm4poi"),
    "tt": dict(target_time=True, method="llm4poi-tt"),
}


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


_UP = {}


def upstream_pieces() -> dict:
    if _UP:
        return _UP
    p = os.path.join(UPSTREAM_DIR, UPSTREAM_FILE)
    got = _sha256(p)
    if got != UPSTREAM_SHA256:
        raise RuntimeError(f"{p} sha256 {got} != pinned {UPSTREAM_SHA256} (LLM4POI {UPSTREAM_COMMIT})")
    with open(p) as f:
        tree = ast.parse(f.read())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "build_samples")
    consts = {}
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                consts[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                pass
    calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "format_entries"]
    headers = []
    for c in calls:
        for a in list(c.args[1:]) + [k.value for k in c.keywords if k.arg == "header"]:
            headers.append(ast.literal_eval(a))
    _UP.update(template=consts["user_template"], history_header=consts["history_header"],
               history_entries_header=headers[0], current_header=headers[1])
    return _UP


def template(cfg: str) -> str:
    t = upstream_pieces()["template"]
    if CFGS[cfg]["target_time"]:
        return t
    if t.count(TT_CLAUSE) != 1:
        raise RuntimeError("upstream template does not contain the target-time clause exactly once")
    return t.replace(TT_CLAUSE, "")


def format_entries(lines: list, header: str) -> str:
    if not lines:
        return header + "\nNone.\n"
    return "\n".join([header] + list(lines)) + "\n"


def normalize_category(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "nan"
    return str(v).strip()


def entry_lines(df: pd.DataFrame) -> np.ndarray:
    t = pd.to_datetime(df["t_local"]).dt.strftime(TIME_FMT).to_numpy(dtype=object)
    v = df["venue_id"].astype(np.int64).astype(str).to_numpy(dtype=object)
    c = np.array([normalize_category(x) for x in df["category"].to_numpy()], dtype=object)
    return t + " | " + v + " | " + c


def render(cfg: str, ds: str, user_id: int, hist_lines, cur_lines, target_time: str | None = None) -> str:
    up = upstream_pieces()
    kw = dict(dataset=DATASET_NAME.get(ds, ds), history_section_header=up["history_header"],
              history_section=format_entries(hist_lines, up["history_entries_header"]),
              current_section=format_entries(cur_lines, up["current_header"]), user_id=int(user_id))
    if CFGS[cfg]["target_time"]:
        if target_time is None:
            raise ValueError("cfg tt needs the target time")
        kw["target_time"] = target_time
    return template(cfg).format(**kw)


def _ds_of(ex) -> str:
    return ex.manifest.get("ds") or os.path.basename(os.path.dirname(ex.path.rstrip("/")))


def train_samples(ex, cfg: str = "default", query_ids=None) -> pd.DataFrame:
    ds = _ds_of(ex)
    qi = ex.query_inputs("train")
    q = qi.table
    lab = ex.train_labels().set_index("query_id")
    keep = np.ones(len(q), dtype=bool)
    if query_ids is not None:
        keep = q["query_id"].isin(set(int(x) for x in query_ids)).to_numpy()
    sub = q.loc[keep, ["query_id", "traj_id"]]
    last = sub.groupby("traj_id", sort=True)["query_id"].max()
    pos = np.flatnonzero(q["query_id"].isin(set(last.to_numpy().tolist())).to_numpy())
    lines = entry_lines(ex.rows)
    out = []
    for i in pos:
        a, b = qi.hist_pos[i]
        c, d = qi.prefix_pos[i]
        h = list(lines[max(a, b - HISTORY_LIMIT):b])
        cur = list(lines[c:d])
        qid = int(q["query_id"].iat[i])
        tt = None
        if CFGS[cfg]["target_time"]:
            tt = pd.Timestamp(lab.at[qid, "label_t_local"]).strftime(TIME_FMT)
        out.append(dict(query_id=qid, user_id=int(q["user_id"].iat[i]), traj_id=int(q["traj_id"].iat[i]),
                        prompt=render(cfg, ds, q["user_id"].iat[i], h, cur, tt),
                        answer=str(int(lab.at[qid, "label_venue_id"])), n_hist=len(h), n_cur=len(cur)))
    return pd.DataFrame(out)


def query_prompts(ex, split: str, query_ids, cfg: str = "default", target_time: dict | None = None) -> pd.DataFrame:
    if split == "train":
        raise ValueError("query prompts are for val/test")
    ds = _ds_of(ex)
    qi = ex.query_inputs(split)
    q = qi.table
    idx = pd.Index(q["query_id"].to_numpy(np.int64))
    want = np.asarray([int(x) for x in query_ids], dtype=np.int64)
    pos = idx.get_indexer(want)
    if (pos < 0).any():
        raise KeyError(f"{int((pos < 0).sum())} query ids not in query_inputs_{split}")
    hl = entry_lines(ex.rows)
    ctx = qi.prefix_source
    cl = entry_lines(ctx)
    cq = ctx["query_id"].to_numpy(np.int64)
    out = []
    for i, qid in zip(pos, want):
        a, b = qi.hist_pos[i]
        c, d = qi.prefix_pos[i]
        if not (cq[c:d] == qid).all():
            raise AssertionError("context slice of another query")
        h = list(hl[max(a, b - HISTORY_LIMIT):b])
        cur = list(cl[c:d])
        tt = None
        if CFGS[cfg]["target_time"]:
            tt = pd.Timestamp(target_time[int(qid)]).strftime(TIME_FMT)
        p = render(cfg, ds, q["user_id"].iat[i], h, cur, tt)
        out.append(dict(query_id=int(qid), user_id=int(q["user_id"].iat[i]), prompt=p,
                        prompt_sha256=hashlib.sha256(p.encode()).hexdigest(), n_hist=len(h), n_cur=len(cur)))
    return pd.DataFrame(out)


def load_target_time(ds: str, split: str) -> dict:
    p = os.path.join(ROOT, "exports_ablation", ds, f"target_time_{split}.parquet")
    t = pd.read_parquet(p)
    return dict(zip(t["query_id"].astype(np.int64), t.iloc[:, 1]))   # canary-allow: labelled tt ablation only


def code_fingerprint() -> str:
    h = hashlib.sha256()
    with open(os.path.abspath(__file__), "rb") as f:
        h.update(f.read())
    h.update(UPSTREAM_SHA256.encode())
    return h.hexdigest()[:16]


def messages(prompt: str, answer: str | None = None) -> list:
    m = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
    if answer is not None:
        m.append({"role": "assistant", "content": answer})
    return m
