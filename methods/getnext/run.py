#!/usr/bin/env python
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import math
import os
import random
import shutil
import sys
import time
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
warnings.filterwarnings("ignore", message="enable_nested_tensor is True")

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.utils.rnn import pad_sequence

from poibench.cost import CostRecorder
from poibench.exports import Export, export_dir

ADAPTER_VERSION = "g2"
ATTN_BACKENDS = [SDPBackend.MATH]
UPSTREAM = os.path.join(ROOT, "third_party", "GETNext")
PINNED_COMMIT = "6a79081aec6951ebec31bff6400d6ba0e40a613e"
UPSTREAM_SHA256 = {
    "model.py": "3e439604bdaa2b2e770ae67aeace089264d22b4226397b39b68c0ada8eb61976",
    "utils.py": "04369655005b3242d1064d5f27bbf1935b016295029b2088b406a33138b4151b",
    "build_graph.py": "97ed58a1e987d8e1f60a7dbcfc145bcb938f73e8101854cdb6d018803d5c6c83",
    "train.py": "eec2bfb0b8edac4bcce643f883f3fb6905aec88c750681ff9c24c5e31d92aaa5",
    "param_parser.py": "79f9d6c29c9f4c8cbe9d616805b5c466dd2b0766eede034a73c712cce1b92caf",
}
CFGS = {"released": "getnext", "fixed": "getnext-fixed", "aligned": "getnext-aligned",
        "fixed-aligned": "getnext-fixed-aligned", "normfeat": "getnext-normfeat",
        "zfeat": "getnext-zfeat"}
SEED = 0
TOPK = 50
SMOKE_USERS = 0.05
SMOKE_EPOCHS = 1
PRED_CHUNK = 256
HP_KEYS = ("short_traj_thres", "time_units", "time_feature", "poi_embed_dim", "user_embed_dim", "gcn_dropout",
           "gcn_nhid", "transformer_nhid", "transformer_nlayers", "transformer_nhead", "transformer_dropout",
           "time_embed_dim", "cat_embed_dim", "time_loss_weight", "node_attn_nhid", "batch", "epochs", "lr",
           "lr_scheduler_factor", "weight_decay")


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def check_upstream() -> None:
    for fn, want in UPSTREAM_SHA256.items():
        got = _sha256(os.path.join(UPSTREAM, fn))
        if got != want:
            raise RuntimeError(f"third_party/GETNext/{fn} sha256 {got} != pinned {want} (commit {PINNED_COMMIT}); "
                               f"run methods/getnext/patches/fetch_upstream.sh")


_MODS = {}


def upstream(name: str):
    if name not in _MODS:
        check_upstream()
        sp = importlib.util.spec_from_file_location(f"getnext_upstream_{name}", os.path.join(UPSTREAM, name + ".py"))
        m = importlib.util.module_from_spec(sp)
        sp.loader.exec_module(m)
        _MODS[name] = m
    return _MODS[name]


def upstream_defaults() -> dict:
    pp = upstream("param_parser")
    old = sys.argv
    try:
        sys.argv = ["getnext"]
        a = pp.parameter_parser()
    finally:
        sys.argv = old
    return {k: getattr(a, k) for k in HP_KEYS}


def upstream_train_function_source(name: str) -> str:
    with open(os.path.join(UPSTREAM, "train.py")) as f:
        src = f.read()
    tree = ast.parse(src)
    train_fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "train")
    fn = next(n for n in train_fn.body if isinstance(n, ast.FunctionDef) and n.name == name)
    import textwrap
    return textwrap.dedent(ast.get_source_segment(src, fn, padded=True))


def seed_all(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    upstream("utils").init_torch_seeds(seed)
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        torch.cuda.manual_seed_all(seed)


def norm_in_day_time(t_local) -> np.ndarray:
    t = pd.DatetimeIndex(t_local)
    return (np.floor((t.hour.to_numpy() * 60 + t.minute.to_numpy()) / 30.0) / 48.0).astype(np.float64)


def resolve_device(dev: str) -> str:
    if dev == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return dev


def root_of_export(ed: str) -> str:
    p = os.path.abspath(ed)
    parts = p.split(os.sep)
    if len(parts) >= 3 and parts[-3] == "exports":
        return os.sep.join(parts[:-3]) or os.sep
    return ROOT


def method_name(cfg: str) -> str:
    return CFGS[cfg]


def work_dir(ds: str, policy: str, root: str = ROOT) -> str:
    return os.path.join(root, "work", "getnext", ds, policy)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _json_dump(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f, indent=1, default=str)
    os.replace(path + ".tmp", path)


def _torch_save(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(obj, path + ".tmp")
    os.replace(path + ".tmp", path)


def candidates_file_decl(ex: Export, split: str) -> str:
    src = ex.manifest.get("source_sha256", {})
    keys = [k for k in src if k.endswith(f"candidates_{split}.parquet")]
    if len(keys) != 1:
        raise RuntimeError(f"cannot identify the {split} candidate file in the export MANIFEST: {keys}")
    return f"{ex.manifest['source_dir']}/{keys[0]}"


def to_getnext_frame(ex: Export) -> pd.DataFrame:
    r = ex.rows.sort_values(["t_utc", "row_id"], kind="mergesort")
    cats = sorted(ex.venues.loc[ex.venues["in_catalog"], "category"].astype(str).unique())
    code = {c: i for i, c in enumerate(cats)}
    tl = pd.DatetimeIndex(r["t_local"])
    off = ((r["t_local"] - r["t_utc"]).dt.total_seconds() // 60).astype(np.int64)
    cat = r["category"].astype(str)
    return pd.DataFrame({
        "user_id": r["user_id"].to_numpy(np.int64),
        "POI_id": r["venue_id"].astype(np.int64).astype(str).to_numpy(),
        "POI_catid": cat.to_numpy(),
        "POI_catid_code": cat.map(code).to_numpy(np.int64),
        "POI_catname": cat.to_numpy(),
        "latitude": r["lat"].to_numpy(), "longitude": r["lon"].to_numpy(),
        "timezone": off.to_numpy(),
        "UTC_time": r["t_utc"].astype(str).to_numpy(), "local_time": r["t_local"].astype(str).to_numpy(),
        "day_of_week": tl.dayofweek.to_numpy(),
        "norm_in_day_time": norm_in_day_time(r["t_local"]),
        "trajectory_id": (r["user_id"].astype(str) + "_" + r["traj_id"].astype(str)).to_numpy(),
        "norm_day_shift": np.nan, "norm_relative_time": np.nan,
        "row_id": r["row_id"].to_numpy(np.int64),
    })


def build_graph(ex: Export, gdf: pd.DataFrame) -> dict:
    import scipy.sparse as sp
    bg = upstream("build_graph")
    G = bg.build_global_POI_checkin_graph(gdf.drop(columns=["row_id"]))
    ven = ex.venues.sort_values("venue_id")
    catalog = ven.loc[ven["in_catalog"], "venue_id"].to_numpy(np.int64)
    present = set(int(n) for n in G.nodes())
    if not present <= set(catalog.tolist()):
        raise RuntimeError("visible rows contain non-catalog venues")
    vt = ven.set_index("venue_id")
    code = dict(zip(gdf["POI_catid"], gdf["POI_catid_code"]))
    cats = sorted(ven.loc[ven["in_catalog"], "category"].astype(str).unique())
    code = {c: i for i, c in enumerate(cats)} | code
    n_added = 0
    for v in catalog:
        if int(v) not in present:
            c = str(vt.at[v, "category"])
            G.add_node(str(int(v)), checkin_cnt=0, poi_catid=c, poi_catid_code=code[c], poi_catname=c,
                       latitude=float(vt.at[v, "lat"]), longitude=float(vt.at[v, "lon"]))
            n_added += 1
    import networkx as nx
    nodelist = list(G.nodes())
    A = sp.csr_matrix(nx.adjacency_matrix(G, nodelist=nodelist), dtype=np.float64)
    nd = [G.nodes[n] for n in nodelist]
    nodes = pd.DataFrame({"node_name/poi_id": nodelist,
                          "checkin_cnt": [d["checkin_cnt"] for d in nd],
                          "poi_catid": [d["poi_catid"] for d in nd],
                          "poi_catid_code": [d["poi_catid_code"] for d in nd],
                          "poi_catname": [d["poi_catname"] for d in nd],
                          "latitude": [d["latitude"] for d in nd], "longitude": [d["longitude"] for d in nd]})
    return dict(nodes=nodes, A=A, n_isolated_added=n_added, n_edges=int(A.nnz))


def prepare(ex: Export, wdir: str, force: bool = False) -> dict:
    import scipy.sparse as sp
    pdir = os.path.join(wdir, "prepare")
    meta_p = os.path.join(pdir, "prepare.json")
    msha = ex.manifest.get("files", {})
    if not force and os.path.exists(meta_p):
        with open(meta_p) as f:
            meta = json.load(f)
        if meta.get("export_files_sha256") == msha and meta.get("adapter_version") == ADAPTER_VERSION:
            return meta
    os.makedirs(pdir, exist_ok=True)
    rec = CostRecorder(None)
    with rec.stage("prepare", one_time=True, n_gpus=0, notes="export -> GETNext csv + transition graph (CPU)"):
        gdf = to_getnext_frame(ex)
        gdf.drop(columns=["row_id"]).to_csv(os.path.join(pdir, "getnext_train.csv.tmp"), index=False)
        os.replace(os.path.join(pdir, "getnext_train.csv.tmp"), os.path.join(pdir, "getnext_train.csv"))
        g = build_graph(ex, gdf)
        g["nodes"].to_csv(os.path.join(pdir, "graph_X.csv.tmp"), index=False)
        os.replace(os.path.join(pdir, "graph_X.csv.tmp"), os.path.join(pdir, "graph_X.csv"))
        sp.save_npz(os.path.join(pdir, "graph_A.tmp.npz"), g["A"])
        os.replace(os.path.join(pdir, "graph_A.tmp.npz"), os.path.join(pdir, "graph_A.npz"))
    meta = dict(adapter_version=ADAPTER_VERSION, created_utc=_now(), ds=ex.manifest.get("ds"),
                policy=ex.manifest.get("policy"), export_files_sha256=msha, n_rows=int(len(gdf)),
                n_nodes=int(len(g["nodes"])), n_isolated_nodes_added=g["n_isolated_added"], n_edges=g["n_edges"],
                n_trajectories=int(gdf["trajectory_id"].nunique()), cost_stage=rec.stages[0])
    _json_dump(meta, meta_p)
    return meta


class GETNext:

    def __init__(self, meta: dict, hp: dict, cfg: str, device: str):
        M = upstream("model")
        self.meta, self.hp, self.cfg, self.device = meta, hp, cfg, device
        self.fix_mask = cfg in ("fixed", "fixed-aligned")
        X = torch.as_tensor(np.asarray(meta["X"], dtype=np.float32))
        self.X = X.to(device=device, dtype=torch.float)
        self.A = torch.as_tensor(np.asarray(meta["A_dense"], dtype=np.float64)).to(device=device, dtype=torch.float)
        n_pois, n_cats, n_users = len(meta["poi_venue"]), len(meta["cat_names"]), len(meta["user_ids"])
        self.poi_embed_model = M.GCN(ninput=X.shape[1], nhid=hp["gcn_nhid"], noutput=hp["poi_embed_dim"],
                                     dropout=hp["gcn_dropout"])
        self.node_attn_model = M.NodeAttnMap(in_features=X.shape[1], nhid=hp["node_attn_nhid"], use_mask=False)
        self.user_embed_model = M.UserEmbeddings(n_users, hp["user_embed_dim"])
        self.time_embed_model = M.Time2Vec('sin', out_dim=hp["time_embed_dim"])
        self.cat_embed_model = M.CategoryEmbeddings(n_cats, hp["cat_embed_dim"])
        self.embed_fuse_model1 = M.FuseEmbeddings(hp["user_embed_dim"], hp["poi_embed_dim"])
        self.embed_fuse_model2 = M.FuseEmbeddings(hp["time_embed_dim"], hp["cat_embed_dim"])
        seq_in = hp["poi_embed_dim"] + hp["user_embed_dim"] + hp["time_embed_dim"] + hp["cat_embed_dim"]
        self.seq_model = M.TransformerModel(n_pois, n_cats, seq_in, hp["transformer_nhead"], hp["transformer_nhid"],
                                            hp["transformer_nlayers"], dropout=hp["transformer_dropout"])
        for m in self.modules():
            m.to(device=device)
        self.poi_cat = torch.as_tensor(np.asarray(meta["poi_cat"], dtype=np.int64), device=device)

    def modules(self):
        return [self.poi_embed_model, self.node_attn_model, self.user_embed_model, self.time_embed_model,
                self.cat_embed_model, self.embed_fuse_model1, self.embed_fuse_model2, self.seq_model]

    MOD_NAMES = ("poi_embed", "node_attn", "user_embed", "time_embed", "cat_embed", "embed_fuse1", "embed_fuse2",
                 "seq_model")

    def parameters(self):
        ps = []
        for m in self.modules():
            ps += list(m.parameters())
        return ps

    def train(self, mode=True):
        for m in self.modules():
            m.train(mode)

    def state_dict(self):
        return {n: m.state_dict() for n, m in zip(self.MOD_NAMES, self.modules())}

    def load_state_dict(self, sd):
        pe = sd["seq_model"].get("pos_encoder.pe")
        if pe is not None:
            self.ensure_pe(pe.shape[0])
        for n, m in zip(self.MOD_NAMES, self.modules()):
            m.load_state_dict(sd[n])

    def ensure_pe(self, n: int) -> None:
        pe = self.seq_model.pos_encoder.pe
        if pe.shape[0] >= n:
            return
        d = pe.shape[-1]
        new = torch.zeros(n, d)
        position = torch.arange(0, n, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        new[:, 0::2] = torch.sin(position * div_term)
        new[:, 1::2] = torch.cos(position * div_term)
        self.seq_model.pos_encoder.pe = new.unsqueeze(0).transpose(0, 1).to(pe.device)

    def embed(self, user_idx, poi_idx, times, poi_embeddings):
        user_embedding = self.user_embed_model(user_idx)
        poi_embedding = poi_embeddings[poi_idx]
        time_embedding = self.time_embed_model(times.view(-1, 1))
        cat_embedding = self.cat_embed_model(self.poi_cat[poi_idx])
        f1, f2 = self.embed_fuse_model1, self.embed_fuse_model2
        fused1 = f1.leaky_relu(f1.fuse_embed(torch.cat((user_embedding, poi_embedding), -1)))
        fused2 = f2.leaky_relu(f2.fuse_embed(torch.cat((time_embedding, cat_embedding), -1)))
        return torch.cat((fused1, fused2), dim=-1)

    def seq_forward(self, x):
        sm = self.seq_model
        with sdpa_kernel(ATTN_BACKENDS):
            if not self.fix_mask:
                src_mask = sm.generate_square_subsequent_mask(x.shape[0]).to(x.device)
                return sm(x, src_mask)
            self.ensure_pe(x.shape[1])
            src_mask = sm.generate_square_subsequent_mask(x.shape[1]).to(x.device)
            yp, yt, yc = sm(x.transpose(0, 1), src_mask)
            return yp.transpose(0, 1), yt.transpose(0, 1), yc.transpose(0, 1)

    @staticmethod
    def adjust(y_pred_poi, in_idx, attn_map):
        valid = in_idx >= 0
        rows = attn_map[in_idx.clamp(min=0)]
        return torch.where(valid.unsqueeze(-1), rows + y_pred_poi, torch.zeros((), dtype=y_pred_poi.dtype,
                                                                                device=y_pred_poi.device))


def build_meta(ex: Export, pdir: str, cfg: str) -> dict:
    import scipy.sparse as sp
    from sklearn.preprocessing import OneHotEncoder
    U = upstream("utils")
    nodes = pd.read_csv(os.path.join(pdir, "graph_X.csv"), dtype={"node_name/poi_id": str, "poi_catid": str})
    raw_A = sp.load_npz(os.path.join(pdir, "graph_A.npz")).toarray()
    enc = OneHotEncoder()
    cat_list = nodes["poi_catid"].tolist()
    enc.fit([[c] for c in cat_list])
    oh = enc.transform([[c] for c in cat_list]).toarray()
    n_cats = oh.shape[-1]
    X = np.zeros((len(nodes), 3 + n_cats), dtype=np.float32)
    X[:, 0] = nodes["checkin_cnt"].to_numpy()
    if cfg == "normfeat":
        X[:, 0] = np.log1p(nodes["checkin_cnt"].to_numpy(np.float64)).astype(np.float32)
    X[:, 1:n_cats + 1] = oh
    X[:, n_cats + 1:] = nodes[["latitude", "longitude"]].to_numpy()
    if cfg == "zfeat":
        Xf = X.astype(np.float64)
        if (Xf.std(axis=0) == 0).any():
            raise RuntimeError("zfeat: constant feature column (zero std)")
        Xf, _, _ = U.get_normalized_features(Xf)
        X = Xf.astype(np.float32)
    A_hat = np.asarray(U.calculate_laplacian_matrix(raw_A, mat_type='hat_rw_normd_lap_mat'))
    node_venue = nodes["node_name/poi_id"].astype(np.int64).to_numpy()
    n = len(node_venue)
    if cfg in ("aligned", "fixed-aligned"):
        perm = np.arange(n)
    else:
        perm = np.random.default_rng(SEED).permutation(n)
    poi_venue = node_venue[perm]
    cat_names = sorted(set(cat_list))
    cmap = {c: i for i, c in enumerate(cat_names)}
    node_cat = np.array([cmap[c] for c in cat_list], dtype=np.int64)
    poi_cat = node_cat[perm]
    user_ids = np.sort(ex.rows["user_id"].unique()).astype(np.int64)
    return dict(X=X, A_dense=A_hat, node_venue=node_venue, poi_venue=poi_venue, poi_cat=poi_cat,
                cat_names=cat_names, user_ids=user_ids, node_checkin_cnt=nodes["checkin_cnt"].to_numpy(np.int64),
                n_venues_all=int(len(ex.venues)))


def meta_for_ckpt(meta: dict) -> dict:
    import scipy.sparse as sp
    m = {k: v for k, v in meta.items() if k != "A_dense"}
    A = sp.csr_matrix(meta["A_dense"])
    m["A_sparse"] = dict(data=A.data, indices=A.indices, indptr=A.indptr, shape=A.shape)
    return m


def meta_from_ckpt(m: dict) -> dict:
    import scipy.sparse as sp
    m = dict(m)
    a = m.pop("A_sparse")
    m["A_dense"] = sp.csr_matrix((a["data"], a["indices"], a["indptr"]), shape=tuple(a["shape"])).toarray()
    return m


def train_sequences(ex: Export, meta: dict, hp: dict, users=None) -> tuple[list, dict]:
    rows = ex.rows
    lab = ex.train_labels().set_index("query_id")
    venue2poi = np.full(meta["n_venues_all"], -1, dtype=np.int64)
    venue2poi[meta["poi_venue"]] = np.arange(len(meta["poi_venue"]))
    uid2idx = {int(u): i for i, u in enumerate(meta["user_ids"])}
    rid = rows["row_id"].to_numpy(np.int64)
    traj = rows["traj_id"].to_numpy(np.int64)
    user = rows["user_id"].to_numpy(np.int64)
    poi = venue2poi[rows["venue_id"].to_numpy(np.int64)]
    if (poi < 0).any():
        raise RuntimeError("a visible row's venue is not a graph node")
    tfeat = norm_in_day_time(rows["t_local"])
    allowed = np.zeros(len(rows), dtype=bool)
    lab_rows = lab["label_row_id"].to_numpy(np.int64)
    pos = np.searchsorted(rid, lab_rows)
    if not (rid[pos] == lab_rows).all():
        raise RuntimeError("train_labels outside visible_rows")
    if not (rows["venue_id"].to_numpy()[pos] == lab["label_venue_id"].to_numpy()).all():
        raise RuntimeError("train_labels disagree with visible_rows")
    allowed[pos] = True
    bounds = np.r_[0, np.flatnonzero((np.diff(traj) != 0) | (np.diff(rid) != 1)) + 1, len(rows)]
    seqs = []
    stats = dict(n_traj=0, n_short=0, n_no_label=0, n_user_skipped=0, n_labels_used=0, n_labels_masked=0)
    uset = None if users is None else set(int(u) for u in users)
    for s, e in zip(bounds[:-1], bounds[1:]):
        stats["n_traj"] += 1
        if uset is not None and int(user[s]) not in uset:
            stats["n_user_skipped"] += 1
            continue
        if (e - s - 1) < hp["short_traj_thres"]:
            stats["n_short"] += 1
            continue
        ok = allowed[s + 1:e]
        if not ok.any():
            stats["n_no_label"] += 1
            continue
        lp = np.where(ok, poi[s + 1:e], -1)
        lt = np.where(ok, tfeat[s + 1:e], -1.0)
        lc = np.where(ok, meta["poi_cat"][np.maximum(poi[s + 1:e], 0)], -1)
        stats["n_labels_used"] += int(ok.sum())
        stats["n_labels_masked"] += int((~ok).sum())
        seqs.append(dict(traj_id=int(traj[s]), user=uid2idx[int(user[s])], poi=poi[s:e - 1], t=tfeat[s:e - 1],
                         y_poi=lp, y_time=lt, y_cat=lc))
    stats["n_sequences"] = len(seqs)
    return seqs, stats


def collate(model: GETNext, seqs: list, poi_embeddings):
    dev = model.device
    lens = [len(s["poi"]) for s in seqs]
    B, L = len(seqs), max(lens)
    users = torch.as_tensor(np.concatenate([np.full(n, s["user"]) for s, n in zip(seqs, lens)]), device=dev)
    pois = torch.as_tensor(np.concatenate([s["poi"] for s in seqs]), device=dev)
    ts = torch.as_tensor(np.concatenate([s["t"] for s in seqs]), dtype=torch.float, device=dev)
    e = model.embed(users, pois, ts, poi_embeddings)
    valid = torch.zeros(B, L, dtype=torch.bool, device=dev)
    for i, n in enumerate(lens):
        valid[i, :n] = True
    x = torch.full((B, L, e.shape[-1]), -1.0, device=dev, dtype=e.dtype)
    x[valid] = e
    in_idx = torch.full((B, L), -1, dtype=torch.long, device=dev)
    in_idx[valid] = pois

    def pad(key, dtype):
        return pad_sequence([torch.as_tensor(np.asarray(s[key])) for s in seqs], batch_first=True,
                            padding_value=-1).to(device=dev, dtype=dtype)
    return x, in_idx, pad("y_poi", torch.long), pad("y_time", torch.float), pad("y_cat", torch.long)


def query_sequences(ex: Export, split: str, query_ids, meta: dict) -> dict:
    qi = ex.query_inputs(split)
    tab = qi.table
    pos = pd.Series(np.arange(len(tab)), index=tab["query_id"].to_numpy())
    venue2poi = np.full(meta["n_venues_all"], -1, dtype=np.int64)
    venue2poi[meta["poi_venue"]] = np.arange(len(meta["poi_venue"]))
    uid2idx = {int(u): i for i, u in enumerate(meta["user_ids"])}
    cv = qi.prefix_source["venue_id"].to_numpy(np.int64)
    ct = norm_in_day_time(qi.prefix_source["t_local"])
    users = tab["user_id"].to_numpy(np.int64)
    out = dict(query_id=[], user=[], poi=[], t=[], n_dropped=0)
    for q in query_ids:
        i = int(pos[int(q)])
        c, d = qi.prefix_pos[i]
        p = venue2poi[cv[c:d]]
        keep = p >= 0
        out["n_dropped"] += int((~keep).sum())
        out["query_id"].append(int(q))
        out["user"].append(uid2idx.get(int(users[i]), -1))
        out["poi"].append(p[keep])
        out["t"].append(ct[c:d][keep])
    return out


@torch.no_grad()
def predict_scores(model: GETNext, qs: dict, chunk: int = PRED_CHUNK):
    model.train(False)
    n = len(qs["query_id"])
    poi_embeddings = model.poi_embed_model(model.X, model.A)
    attn_map = model.node_attn_model(model.X, model.A)
    usable = [i for i in range(n) if qs["user"][i] >= 0 and len(qs["poi"][i]) > 0]
    scores = [None] * n
    bsz = model.hp["batch"]
    if not model.fix_mask:
        groups = {}
        for i in usable:
            groups.setdefault((len(qs["poi"][i]) - 1) % bsz, []).append(i)
        for s, idx in sorted(groups.items()):
            for a in range(0, len(idx), chunk):
                ii = idx[a:a + chunk]
                last_poi = torch.as_tensor([int(qs["poi"][i][-1]) for i in ii], device=model.device)
                e = model.embed(torch.as_tensor([qs["user"][i] for i in ii], device=model.device), last_poi,
                                torch.as_tensor([float(qs["t"][i][-1]) for i in ii], dtype=torch.float,
                                                device=model.device), poi_embeddings)
                x = torch.full((s + 1, len(ii), e.shape[-1]), -1.0, device=model.device)
                x[s] = e
                yp, _, _ = model.seq_forward(x)
                adj = attn_map[last_poi] + yp[s]
                for j, i in enumerate(ii):
                    scores[i] = adj[j]
    else:
        order = sorted(usable, key=lambda i: len(qs["poi"][i]))
        for a in range(0, len(order), chunk):
            ii = order[a:a + chunk]
            seqs = [dict(user=qs["user"][i], poi=qs["poi"][i], t=qs["t"][i]) for i in ii]
            lens = [len(s["poi"]) for s in seqs]
            dev = model.device
            users = torch.as_tensor(np.concatenate([np.full(k, s["user"]) for s, k in zip(seqs, lens)]), device=dev)
            pois = torch.as_tensor(np.concatenate([s["poi"] for s in seqs]), device=dev)
            ts = torch.as_tensor(np.concatenate([s["t"] for s in seqs]), dtype=torch.float, device=dev)
            e = model.embed(users, pois, ts, poi_embeddings)
            x = pad_sequence(list(torch.split(e, lens)), batch_first=True, padding_value=-1)
            yp, _, _ = model.seq_forward(x)
            last = torch.as_tensor([k - 1 for k in lens], device=dev)
            ar = torch.arange(len(ii), device=dev)
            lp = torch.as_tensor([int(s["poi"][-1]) for s in seqs], device=dev)
            adj = attn_map[lp] + yp[ar, last]
            for j, i in enumerate(ii):
                scores[i] = adj[j]
    return scores


def rank_lists(model: GETNext, qs: dict, scores, cands: dict | None):
    poi_venue = np.asarray(model.meta["poi_venue"], dtype=np.int64)
    node_venue = np.asarray(model.meta["node_venue"], dtype=np.int64)
    cnt = np.asarray(model.meta["node_checkin_cnt"], dtype=np.int64)
    pop = node_venue[np.lexsort((node_venue, -cnt))][:TOPK].tolist()
    nva = model.meta["n_venues_all"]
    v2p = np.full(nva, -1, dtype=np.int64)
    v2p[poi_venue] = np.arange(len(poi_venue))
    top, cand, n_fb = [], [], 0
    for i, q in enumerate(qs["query_id"]):
        c = None
        if cands is not None:
            c0 = np.asarray(cands[q], dtype=np.int64)
            _, first = np.unique(c0, return_index=True)
            c = c0[np.sort(first)]
        if scores[i] is None:
            n_fb += 1
            top.append(pop)
            if c is not None:
                cand.append(c.tolist())
            continue
        s = scores[i].detach().float().cpu().numpy()
        k = min(len(s), 4 * TOPK)
        part = np.argpartition(-s, k - 1)[:k] if k < len(s) else np.arange(len(s))
        thr = s[part].min()
        part = np.flatnonzero(s >= thr)
        o = part[np.lexsort((poi_venue[part], -s[part]))][:TOPK]
        top.append(poi_venue[o].tolist())
        if c is not None:
            p = v2p[c]
            sc = np.where(p >= 0, s[np.maximum(p, 0)], -np.inf)
            cand.append(c[np.lexsort((np.arange(len(c)), -sc))].tolist())
    df = pd.DataFrame({"query_id": [int(q) for q in qs["query_id"]], "ranked_venue_ids": top})
    if cands is not None:
        df["cand_ranked_venue_ids"] = cand
    return df, n_fb


def load_model(ckpt_path: str, device: str = "cpu") -> GETNext:
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = GETNext(meta_from_ckpt(ck["meta"]), ck["hp"], ck["cfg"], device)
    model.load_state_dict(ck["state"])
    model.train(False)
    model.ckpt_info = {k: ck[k] for k in ("epoch", "cfg", "ds", "policy", "monitor_score") if k in ck}
    return model


def default_ckpt(ds: str, policy: str, cfg: str) -> str:
    return os.path.join(work_dir(ds, policy), cfg, "smoke", "best.pt")


def predict(export_dir, split, query_ids, work_dir=None, ckpt=None, device=None, cfg=None, with_cand=True):
    ex = Export(export_dir)
    cfg = cfg or os.environ.get("GETNEXT_CFG", "released")
    ckpt = ckpt or os.environ.get("GETNEXT_CKPT") or default_ckpt(ex.manifest["ds"], ex.manifest["policy"], cfg)
    model = ckpt if isinstance(ckpt, GETNext) else load_model(ckpt, device or os.environ.get("GETNEXT_DEVICE", "cpu"))
    qs = query_sequences(ex, split, query_ids, model.meta)
    scores = predict_scores(model, qs)
    cands = None
    if with_cand:
        cd = ex.candidates(split)
        cands = dict(zip(cd["query_id"].to_numpy(np.int64).tolist(), cd["cand_venue_ids"]))
    df, n_fb = rank_lists(model, qs, scores, cands)
    df.attrs.update(n_fallback=n_fb, n_context_rows_dropped=qs["n_dropped"])
    info = getattr(model, "ckpt_info", {}) or {}
    df.attrs["adapter_config"] = dict(cfg=info.get("cfg", cfg), requested_cfg=cfg,
                                      checkpoint=ckpt if isinstance(ckpt, str) else None)
    return df


def train(ex: Export, ed: str, wdir: str, cfg: str, device: str, epochs: int | None = None,
          user_frac: float | None = None, out_sub: str = "train", log=print) -> dict:
    from poibench.evaluate import val_feedback
    hp = upstream_defaults()
    if epochs is not None:
        hp["epochs"] = int(epochs)
    odir = os.path.join(wdir, cfg, out_sub)
    os.makedirs(odir, exist_ok=True)
    done_p = os.path.join(odir, "DONE.json")
    run_key = dict(adapter_version=ADAPTER_VERSION, cfg=cfg, hp=hp, user_frac=user_frac, seed=SEED,
                   export_files_sha256=ex.manifest.get("files", {}))
    if os.path.exists(done_p):
        with open(done_p) as f:
            d = json.load(f)
        if d.get("run_key") == json.loads(json.dumps(run_key, default=str)):
            log(f"[getnext] {odir}: training already done (best epoch {d.get('best_epoch')})")
            return d
    pdir = os.path.join(wdir, "prepare")
    pmeta = prepare(ex, wdir)
    seed_all(SEED)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    meta = build_meta(ex, pdir, cfg)
    users = None
    if user_frac is not None:
        allu = np.sort(ex.rows["user_id"].unique())
        users = np.sort(np.random.default_rng(SEED).choice(allu, size=max(1, math.ceil(user_frac * len(allu))),
                                                           replace=False))
    seqs, sstats = train_sequences(ex, meta, hp, users)
    seed_all(SEED)
    model = GETNext(meta, hp, cfg, device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    criterion_poi = torch.nn.CrossEntropyLoss(ignore_index=-1)
    criterion_cat = torch.nn.CrossEntropyLoss(ignore_index=-1)
    criterion_time = upstream("utils").maksed_mse_loss
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=hp["lr_scheduler_factor"])
    gen = torch.Generator().manual_seed(SEED)

    val_q = ex.queries("val")
    if users is not None:
        val_q = val_q[val_q["user_id"].isin(set(int(u) for u in users))]
    val_ids = val_q["query_id"].to_numpy(np.int64).tolist()
    vqs = query_sequences(ex, "val", val_ids, meta)
    ds = ex.manifest["ds"]
    vroot = root_of_export(ed)

    state = dict(epoch=-1, best_score=-np.inf, best_epoch=None, history=[], sessions=[])
    last_p, best_p = os.path.join(odir, "last.pt"), os.path.join(odir, "best.pt")
    if os.path.exists(last_p):
        ck = torch.load(last_p, map_location="cpu", weights_only=False)
        if ck.get("run_key") == json.loads(json.dumps(run_key, default=str)):
            model.load_state_dict(ck["state"])
            optimizer.load_state_dict(ck["optimizer"])
            lr_scheduler.load_state_dict(ck["scheduler"])
            gen.set_state(ck["gen_state"])
            torch.set_rng_state(ck["torch_rng"])
            if device.startswith("cuda") and ck.get("cuda_rng") is not None:
                torch.cuda.set_rng_state_all(ck["cuda_rng"])
            np.random.set_state(ck["np_rng"])
            random.setstate(ck["py_rng"])
            state = ck["train_state"]
            log(f"[getnext] resume from {last_p}: epoch {state['epoch']} done, best {state['best_epoch']}")
        else:
            log(f"[getnext] {last_p} belongs to another configuration; starting fresh")
    ckpt_common = dict(meta=meta_for_ckpt(meta), hp=hp, cfg=cfg, ds=ds, policy=ex.manifest["policy"],
                       adapter_version=ADAPTER_VERSION, upstream_commit=PINNED_COMMIT)
    rec = CostRecorder(None)
    ngpu = 1 if device.startswith("cuda") else 0
    log(f"[getnext] {ds}/{ex.manifest['policy']}/{cfg}: {sstats['n_sequences']} train sequences, "
        f"{len(meta['poi_venue'])} POIs, {len(meta['user_ids'])} users, {len(val_ids)} val queries, device {device}")
    with rec.stage("train", one_time=True, n_gpus=ngpu, notes="incl. per-epoch val inference for selection") as srec:
        for epoch in range(state["epoch"] + 1, hp["epochs"]):
            t0 = time.time()
            model.train(True)
            perm = torch.randperm(len(seqs), generator=gen).tolist()
            losses = []
            for b in range(0, len(perm), hp["batch"]):
                batch = [seqs[i] for i in perm[b:b + hp["batch"]]]
                poi_embeddings = model.poi_embed_model(model.X, model.A)
                x, in_idx, y_poi, y_time, y_cat = collate(model, batch, poi_embeddings)
                y_pred_poi, y_pred_time, y_pred_cat = model.seq_forward(x)
                attn_map = model.node_attn_model(model.X, model.A)
                y_pred_poi_adjusted = GETNext.adjust(y_pred_poi, in_idx, attn_map)
                loss_poi = criterion_poi(y_pred_poi_adjusted.transpose(1, 2), y_poi)
                loss_time = criterion_time(y_pred_time.squeeze(-1), y_time)
                loss_cat = criterion_cat(y_pred_cat.transpose(1, 2), y_cat)
                loss = loss_poi + loss_time * hp["time_loss_weight"] + loss_cat
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
                losses.append([float(loss.detach()), float(loss_poi.detach()), float(loss_time.detach()),
                               float(loss_cat.detach())])
            scores = predict_scores(model, vqs)
            vdf, _ = rank_lists(model, vqs, scores, None)
            fb = val_feedback(ds, vdf, ks=(1, 5, 10, 20), root=vroot, query_ids=val_ids)
            monitor_score = float(fb["acc@1"] * 4 + fb["acc@20"])
            lr_scheduler.step(monitor_score)
            ml = np.mean(losses, axis=0) if losses else [float("nan")] * 4
            h = dict(epoch=epoch, train_loss=float(ml[0]), train_poi_loss=float(ml[1]), train_time_loss=float(ml[2]),
                     train_cat_loss=float(ml[3]), val=fb, monitor_score=monitor_score,
                     lr=float(optimizer.param_groups[0]["lr"]), wall_s=round(time.time() - t0, 2))
            state["history"].append(h)
            state["epoch"] = epoch
            if monitor_score >= state["best_score"]:
                state["best_score"], state["best_epoch"] = monitor_score, epoch
                _torch_save(dict(ckpt_common, state=model.state_dict(), epoch=epoch, monitor_score=monitor_score),
                            best_p)
            _torch_save(dict(run_key=json.loads(json.dumps(run_key, default=str)), state=model.state_dict(),
                             optimizer=optimizer.state_dict(), scheduler=lr_scheduler.state_dict(),
                             gen_state=gen.get_state(), torch_rng=torch.get_rng_state(),
                             cuda_rng=torch.cuda.get_rng_state_all() if device.startswith("cuda") else None,
                             np_rng=np.random.get_state(), py_rng=random.getstate(), train_state=state), last_p)
            with open(os.path.join(odir, "log.jsonl"), "a") as f:
                f.write(json.dumps(h) + "\n")
            log(f"[getnext] epoch {epoch}: loss {h['train_loss']:.4f} val acc@1 {fb['acc@1']:.4f} "
                f"acc@20 {fb['acc@20']:.4f} score {monitor_score:.4f} lr {h['lr']:.1e} ({h['wall_s']} s)")
        srec["epochs_run_this_session"] = len(state["history"])
    sess = rec.stages[0]
    state["sessions"].append(sess)
    out = dict(run_key=json.loads(json.dumps(run_key, default=str)), finished_utc=_now(), best_epoch=state["best_epoch"],
               best_monitor_score=state["best_score"], epochs=hp["epochs"], seq_stats=sstats,
               n_val_queries_for_selection=len(val_ids), sessions=state["sessions"], prepare=pmeta["cost_stage"],
               users_subsample=None if users is None else [int(u) for u in users])
    if os.path.exists(last_p):
        ck = torch.load(last_p, map_location="cpu", weights_only=False)
        ck["train_state"] = state
        _torch_save(ck, last_p)
    _json_dump(out, done_p)
    return out


def write_preds(ex: Export, ed: str, wdir: str, cfg: str, split: str, device: str, force: bool = False,
                out_root: str = ROOT, log=print) -> str:
    policy, ds = ex.manifest["policy"], ex.manifest["ds"]
    mname = f"{method_name(cfg)}@{policy}"
    odir = os.path.join(out_root, "preds", mname, ds)
    pq, cj = os.path.join(odir, f"{split}.parquet"), os.path.join(odir, f"{split}.cost.json")
    best_p = os.path.join(wdir, cfg, "train", "best.pt")
    if not force and os.path.exists(pq) and os.path.exists(cj):
        with open(cj) as f:
            c = json.load(f)
        if c.get("checkpoint_sha256") == _sha256(best_p):
            log(f"[getnext] {pq} exists for this checkpoint; skipped")
            return pq
    return _predict_to(ex, ed, wdir, cfg, split, device, pq, cj, log=log)


def _predict_to(ex: Export, ed: str, wdir: str, cfg: str, split: str, device: str, pq: str, cj: str,
                log=print) -> str:
    policy, ds = ex.manifest["policy"], ex.manifest["ds"]
    mname = f"{method_name(cfg)}@{policy}"
    with open(os.path.join(wdir, cfg, "train", "DONE.json")) as f:
        done = json.load(f)
    best_p = os.path.join(wdir, cfg, "train", "best.pt")
    qids = ex.queries(split)["query_id"].to_numpy(np.int64).tolist()
    rec = CostRecorder(cj, method=mname, ds=ds, split=split,
                       extra=dict(adapter="methods/getnext/run.py", adapter_version=ADAPTER_VERSION, cfg=cfg,
                                  evidence_policy=policy, topk=TOPK, seed=SEED, upstream_commit=PINNED_COMMIT,
                                  candidates_file=candidates_file_decl(ex, split),
                                  checkpoint_sha256=_sha256(best_p), best_epoch=done["best_epoch"],
                                  best_val_monitor_score=done["best_monitor_score"], epochs=done["epochs"],
                                  train_sequences=done["seq_stats"]))
    rec.add_stage(**done["prepare"])
    ss = done["sessions"]
    tr = dict(name="train", wall_s=sum(s["wall_s"] for s in ss), cpu_core_s=sum(s["cpu_core_s"] for s in ss),
              gpu_s=sum(s["gpu_s"] for s in ss), peak_mem_gb=max(s["peak_mem_gb"] for s in ss), one_time=True,
              n_gpus=max(s.get("n_gpus", 0) for s in ss), n_sessions=len(ss),
              notes="sum over training sessions (resume); incl. per-epoch val inference for checkpoint selection")
    for k in ("gpu_busy_s",):
        if any(k in s for s in ss):
            tr[k] = sum(s.get(k, 0.0) for s in ss)
    for k in ("peak_gpu_mem_gb", "peak_gpu_mem_nvml_gb"):
        if any(k in s for s in ss):
            tr[k] = max(s.get(k, 0.0) for s in ss)
    rec.add_stage(**{k: v for k, v in tr.items()})
    ngpu = 1 if device.startswith("cuda") else 0
    with rec.stage(f"predict_{split}", one_time=False, n_gpus=ngpu, n_items=len(qids)) as srec:
        model = load_model(best_p, device)
        df = predict(ed, split, qids, ckpt=model)
        srec["n_fallback_popularity"] = int(df.attrs["n_fallback"])
        srec["n_context_rows_not_in_graph"] = int(df.attrs["n_context_rows_dropped"])
    os.makedirs(os.path.dirname(os.path.abspath(pq)), exist_ok=True)
    df.attrs = {}
    df.to_parquet(pq + ".tmp", index=False)
    os.replace(pq + ".tmp", pq)
    rec.save(n_queries=len(qids))
    log(f"[getnext] wrote {pq} ({len(df)} queries, {rec.stages[-1]['n_fallback_popularity']} popularity fallbacks)")
    return pq


def cost_work_dir(ds: str, policy: str) -> str:
    return os.path.join(ROOT, "work", "cost_exclusive", "getnext", ds, policy)


def cost_only(ex: Export, ed: str, wdir: str, cfg: str, splits, device: str, cost_root: str | None = None,
              epochs: int | None = None, log=print) -> dict:
    from poibench import exclusive as X
    policy, ds = ex.manifest["policy"], ex.manifest["ds"]
    mname = f"{method_name(cfg)}@{policy}"
    X.assert_outside_preds(wdir)
    cjs = {s: X.cost_path(mname, ds, s, cost_root) for s in splits}
    pqs = {s: X.assert_outside_preds(os.path.join(wdir, cfg, "cost_only_preds", f"{s}.parquet")) for s in splits}
    mon = X.GpuShareMonitor().start() if device.startswith("cuda") else None
    try:
        train(ex, ed, wdir, cfg, device, epochs=epochs, user_frac=None, out_sub="train", log=log)
        for s in splits:
            _predict_to(ex, ed, wdir, cfg, s, device, pqs[s], cjs[s], log=log)
    finally:
        chk = mon.stop() if mon is not None else None
        os.makedirs(os.path.join(wdir, cfg), exist_ok=True)
        with open(os.path.join(wdir, cfg, "colocation_checks.jsonl"), "a") as f:
            f.write(json.dumps(dict(attempt=os.environ.get("POIBENCH_ATTEMPT"), device=device, check=chk)) + "\n")
    with open(os.path.join(wdir, cfg, "colocation_checks.jsonl")) as f:
        checks = [json.loads(ln) for ln in f if ln.strip()]
    flags = [c["check"]["colocated"] if c["check"] else None for c in checks]
    colocated = True if any(v is True for v in flags) else (False if flags and all(v is False for v in flags)
                                                           else None)
    for s in splits:
        with open(cjs[s]) as f:
            c = json.load(f)
        c.update(X.record(chk, colocated=colocated, colocation_checks_all_attempts=checks,
                          work_dir=os.path.relpath(wdir, ROOT), rerun_preds=os.path.relpath(pqs[s], ROOT),
                          preds_parity=X.preds_parity(pqs[s], os.path.join(ROOT, "preds", mname, ds, f"{s}.parquet")),
                          rerun_of="main-table job of the same method/policy/ds (preds/ of record)"))
        _json_dump(c, cjs[s])
        log(f"[getnext] cost-only: wrote {cjs[s]} (colocated={colocated}, gpu={c.get('gpu_model')})")
    return {s: cjs[s] for s in splits}


def smoke_predict_val(ex: Export, ed: str, wdir: str, cfg: str, device: str, users, log=print) -> dict:
    from poibench.evaluate import val_feedback
    sdir = os.path.join(wdir, cfg, "smoke")
    q = ex.queries("val")
    qids = q.loc[q["user_id"].isin(set(users)), "query_id"].to_numpy(np.int64).tolist()
    rec = CostRecorder(os.path.join(sdir, "val.cost.json"), method=f"{method_name(cfg)}@{ex.manifest['policy']}-smoke",
                       ds=ex.manifest["ds"], split="val", extra=dict(smoke=True, cfg=cfg))
    with rec.stage("predict_val", one_time=False, n_gpus=0, n_items=len(qids)) as srec:
        df = predict(ed, "val", qids, ckpt=os.path.join(sdir, "best.pt"), device=device)
        srec["n_fallback_popularity"] = int(df.attrs["n_fallback"])
    df.attrs = {}
    df.to_parquet(os.path.join(sdir, "val_preds.parquet"), index=False)
    rec.save(n_queries=len(qids))
    fb = val_feedback(ex.manifest["ds"], df, ks=(1, 5, 10, 20), root=root_of_export(ed), query_ids=qids)
    _json_dump(fb, os.path.join(sdir, "val_feedback.json"))
    log(f"[getnext] smoke val ({len(qids)} queries of {len(users)} users): {fb}")
    return fb


def main(argv=None):
    ap = argparse.ArgumentParser(description="GETNext adapter (POIBench exports)")
    ap.add_argument("cmd", nargs="?", default="all", choices=["prepare", "train", "predict", "all"])
    ap.add_argument("--ds", required=True)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--splits", "--split", nargs="+", default=["val", "test"], choices=["val", "test"])
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:N")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cfg", default="released", choices=list(CFGS))
    ap.add_argument("--smoke-users", type=float, default=None,
                    help="smoke run: train on this fraction of users (seed 0); outputs under work/ only")
    ap.add_argument("--epochs", type=int, default=None, help="only with --smoke-users (default: upstream 200)")
    ap.add_argument("--canary-prep", action="store_true",
                    help="train the smoke checkpoint predict() loads for the canary (1 epoch, 5%% users)")
    ap.add_argument("--force", action="store_true", help="redo prepare/predict even if outputs exist")
    ap.add_argument("--cost-only", action="store_true",
                    help="exclusive cost re-run: identical pipeline into --work-dir, writes only "
                         "results/cost_exclusive/<method>@<policy>/<ds>/<split>.cost.json (never preds/)")
    ap.add_argument("--work-dir", help="with --cost-only: work dir (default work/cost_exclusive/getnext/<ds>/<policy>)")
    ap.add_argument("--cost-root", help="with --cost-only: root of the cost records (default results/cost_exclusive)")
    a = ap.parse_args(argv)
    if a.seed != SEED:
        ap.error("owner rule: seed 0 only")
    if a.cost_only:
        if a.cmd != "all" or a.canary_prep or a.smoke_users is not None or a.force:
            ap.error("--cost-only runs the whole pipeline (cmd all) and excludes --canary-prep/--smoke-users/--force")
        if a.epochs is not None and not a.cost_root:
            ap.error("--epochs with --cost-only only for tests, with an explicit --cost-root")
    elif a.work_dir or a.cost_root:
        ap.error("--work-dir/--cost-root only with --cost-only")
    elif a.epochs is not None and a.smoke_users is None:
        ap.error("--epochs overrides the upstream default and is allowed only in smoke runs (--smoke-users)")
    device = resolve_device(a.device)
    if a.canary_prep:
        device = "cpu" if a.device == "auto" else device
    ed = export_dir(a.ds, a.policy)
    ex = Export(ed)
    wdir = work_dir(a.ds, a.policy)
    t0 = time.time()
    if a.cost_only:
        cost_only(ex, ed, a.work_dir or cost_work_dir(a.ds, a.policy), a.cfg, a.splits, device,
                  cost_root=a.cost_root, epochs=a.epochs)
        print(f"[getnext] cost-only {a.ds}/{a.policy}/{a.cfg} finished in {time.time() - t0:.0f} s")
        return 0
    if a.canary_prep or a.smoke_users is not None:
        frac = SMOKE_USERS if a.smoke_users is None else a.smoke_users
        ep = SMOKE_EPOCHS if a.epochs is None else a.epochs
        d = train(ex, ed, wdir, a.cfg, device, epochs=ep, user_frac=frac, out_sub="smoke")
        if a.canary_prep:
            print(f"[getnext] canary-prep done: {os.path.join(wdir, a.cfg, 'smoke', 'best.pt')}")
        elif "val" in a.splits:
            smoke_predict_val(ex, ed, wdir, a.cfg, device, d["users_subsample"])
        return 0
    if a.cmd in ("prepare", "all"):
        m = prepare(ex, wdir, force=a.force)
        print(f"[getnext] prepare: {m['n_nodes']} nodes, {m['n_edges']} edges, {m['n_rows']} rows")
    if a.cmd in ("train", "all"):
        train(ex, ed, wdir, a.cfg, device)
    if a.cmd in ("predict", "all"):
        for s in a.splits:
            write_preds(ex, ed, wdir, a.cfg, s, device, force=a.force)
    print(f"[getnext] {a.cmd} {a.ds}/{a.policy}/{a.cfg} finished in {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
