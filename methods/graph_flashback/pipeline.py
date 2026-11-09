from __future__ import annotations

import copy
import hashlib
import json
import logging
import math
import os
import pickle
import random
import shutil
import sys
import time
import types
import warnings

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from poibench.cost import CostRecorder
from poibench.exports import Export

import apply_patches

METHOD = "graph-flashback"
TOPK = 50
ADAPTER_VERSION = "gf1"

DEFAULTS = dict(
    hidden_dim=10, weight_decay=0.0, lr=0.01, epochs=100, rnn="rnn", sequence_length=20, batch_size=512,
    lambda_t=0.1, lambda_s=100.0, lambda_loc=1.0, lambda_user=1.0, use_weight=False, use_graph_user=False,
    use_spatial_graph=False, validate_epoch=5,
    optimizer="Adam", milestones=[20, 40, 60, 80], gamma=0.2,
    min_checkins=21,
    loc_threshold=20, user_loc_threshold=100, kge_model="transe", spatial_scheme=2,
    kge_batch_size=128, kge_optimizer="Adam", kge_epochs=100,
    loss_positions="train_queries", selection="best_val_acc1", seed=0,
    smoke=False, smoke_users=0, smoke_val_queries=0,
)
SMOKE = dict(epochs=2, kge_epochs=1, smoke=True, smoke_users=100, smoke_val_queries=500, selection="last")
PREPARE_KEYS = ("min_checkins", "loc_threshold", "user_loc_threshold", "kge_model", "spatial_scheme",
                "kge_batch_size", "kge_optimizer", "kge_epochs", "seed", "smoke", "smoke_users")
TRAIN_KEYS = tuple(k for k in DEFAULTS if k not in ("smoke_val_queries",))


def make_config(smoke: bool = False, **over) -> dict:
    cfg = dict(DEFAULTS)
    if smoke:
        cfg.update(SMOKE)
    cfg.update({k: v for k, v in over.items() if v is not None})
    if cfg["seed"] != 0:
        raise ValueError("owner rule: seed 0 only")
    return cfg


def log(msg: str) -> None:
    print(f"[graph_flashback {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:12]


def seed_all(seed: int = 0) -> None:
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def export_info(ex: Export) -> dict:
    m = ex.manifest
    return dict(ds=m.get("ds"), policy=m.get("policy") or ex.policy.get("policy"),
                perturbed=bool(m.get("perturbed_for_canary")), files=m.get("files", {}))


def candidates_file_decl(ex: Export, split: str) -> str | None:
    m = ex.manifest
    keys = [k for k in (m.get("source_sha256") or {}) if k.endswith(f"candidates_{split}.parquet")]
    src = m.get("source_dir")
    if not keys or not src:
        return None
    return f"{src}/{keys[0]}"


_MODS = None


def authors_modules():
    global _MODS
    if _MODS is not None:
        return _MODS
    paths = apply_patches.build_src(ROOT, verbose=False)
    gf, kge = paths["Graph-Flashback"], paths["Graph-Flashback-KGE"]
    import gflags_shim
    sys.modules["gflags"] = gflags_shim
    for p in (kge, os.path.join(gf, "KGE"), gf):
        if p not in sys.path:
            sys.path.insert(0, p)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    import importlib
    m = types.SimpleNamespace(src=paths)
    for name in ("network", "trainer", "dataset", "dataloader", "utils", "generate_triplet",
                 "construct_loc_loc_graph", "construct_user_loc_graph"):
        setattr(m, name.replace(".", "_"), importlib.import_module(name))
    m.kr = importlib.import_module("jTransUP.models.knowledge_representation")
    m.base = importlib.import_module("jTransUP.models.base")
    m.ltd = importlib.import_module("jTransUP.data.load_triple_data")
    m.kmisc = importlib.import_module("jTransUP.utils.misc")
    m.ktrainer = importlib.import_module("jTransUP.utils.trainer")
    gflags_shim.FLAGS.__dict__.clear()
    m.base.get_flags()
    m.kge_flag_defaults = gflags_shim.defaults()
    m.dataset.TRAIN_FRACTION = 1.0
    m.relation_file = os.path.join(gf, "KGE", "dataset", "gowalla", "relation2id.txt")
    _MODS = m
    return m


class Index:

    def __init__(self, cat_venues, venue_lat, venue_lon, train_users, other_users, triplet_users, n_venue_ids,
                 max_user_id):
        self.cat_venues = np.asarray(cat_venues, dtype=np.int64)
        self.venue_lat = np.asarray(venue_lat, dtype=np.float64)
        self.venue_lon = np.asarray(venue_lon, dtype=np.float64)
        self.train_users = np.asarray(train_users, dtype=np.int64)
        self.other_users = np.asarray(other_users, dtype=np.int64)
        self.triplet_users = np.asarray(triplet_users, dtype=np.int64)
        self.n_venue_ids = int(n_venue_ids)
        self.max_user_id = int(max_user_id)
        self.pidx_of = np.full(self.n_venue_ids, -1, dtype=np.int64)
        self.pidx_of[self.cat_venues] = np.arange(len(self.cat_venues))
        self.users = np.concatenate([self.train_users, self.other_users])
        self.uidx_of = np.full(self.max_user_id + 1, -1, dtype=np.int64)
        self.uidx_of[self.users] = np.arange(len(self.users))

    @property
    def L(self) -> int:
        return len(self.cat_venues)

    @property
    def U(self) -> int:
        return len(self.users)

    @property
    def n_train(self) -> int:
        return len(self.train_users)

    def uidx(self, user_id: int) -> int:
        return int(self.uidx_of[user_id]) if 0 <= user_id <= self.max_user_id else -1

    @classmethod
    def build(cls, ex: Export, cfg: dict) -> "Index":
        ven = ex.venues.sort_values("venue_id")
        cat = ven[ven["in_catalog"].to_numpy(bool)]
        rows = ex.rows
        cnt = rows.groupby("user_id").size()
        all_users = np.sort(cnt.index.to_numpy(np.int64))
        if cfg["smoke"]:
            rng = np.random.default_rng(cfg["seed"])
            sub = np.sort(rng.choice(all_users, size=min(cfg["smoke_users"], len(all_users)), replace=False))
        else:
            sub = all_users
        n_of = cnt.reindex(sub).to_numpy()
        train_users = sub[n_of >= cfg["min_checkins"]]
        other = np.setdiff1d(all_users, train_users)
        return cls(cat["venue_id"].to_numpy(), cat["lat"].to_numpy(), cat["lon"].to_numpy(), train_users, other,
                   sub, len(ven), int(all_users.max()))

    def save(self, path: str) -> None:
        np.savez(path, cat_venues=self.cat_venues, venue_lat=self.venue_lat, venue_lon=self.venue_lon,
                 train_users=self.train_users, other_users=self.other_users, triplet_users=self.triplet_users,
                 n_venue_ids=self.n_venue_ids, max_user_id=self.max_user_id)

    @classmethod
    def load(cls, path: str) -> "Index":
        z = np.load(path)
        return cls(z["cat_venues"], z["venue_lat"], z["venue_lon"], z["train_users"], z["other_users"],
                   z["triplet_users"], int(z["n_venue_ids"]), int(z["max_user_id"]))


def prepare_key(ex: Export, cfg: dict) -> str:
    f = export_info(ex)["files"]
    return _hash(dict(v=ADAPTER_VERSION, src=apply_patches.src_key(), cfg={k: cfg[k] for k in PREPARE_KEYS},
                      rows=f.get("visible_rows.parquet"), venues=f.get("venues_public.parquet")))


def prepare_dir(ex: Export, cfg: dict, work_root: str) -> str:
    return os.path.join(work_root, export_info(ex)["ds"] or "unknown", f"prepare_{prepare_key(ex, cfg)}")


def _write_triplet_input(ex: Export, idx: Index, path: str) -> int:
    rows = ex.rows.sort_values("row_id")
    rows = rows[rows["user_id"].isin(set(idx.triplet_users.tolist()))]
    n = 0
    with open(path, "w") as f:
        for uid, g in rows.groupby("user_id", sort=True):
            p = idx.pidx_of[g["venue_id"].to_numpy(np.int64)]
            if (p < 0).any():
                raise AssertionError("visible train row at a venue outside the catalog")
            f.write(" ".join([str(idx.uidx(int(uid)))] + [str(int(x)) for x in p]) + " \n")
            n += len(p)
    return n


def prepare(ex: Export, cfg: dict, work_root: str, device: str = "cpu") -> str:
    import torch
    import torch.nn as nn
    pdir = prepare_dir(ex, cfg, work_root)
    done = os.path.join(pdir, "DONE.json")
    if os.path.exists(done):
        log(f"prepare: reusing {os.path.relpath(pdir, ROOT)}")
        return pdir
    m = authors_modules()
    tmp = f"{pdir}.tmp_{os.getpid()}"
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    kg = os.path.join(tmp, "kg")
    os.makedirs(kg)
    rec = CostRecorder()
    info = dict(prepare_key=os.path.basename(pdir), config={k: cfg[k] for k in PREPARE_KEYS},
                export=export_info(ex), src=apply_patches.src_key())
    with rec.stage("kge_triplets", one_time=True, n_gpus=0) as st:
        idx = Index.build(ex, cfg)
        idx.save(os.path.join(tmp, "index.npz"))
        gt = m.generate_triplet
        gt.users_count = idx.U
        gt.user2id = {}
        gt.poi2id = {int(v): int(p) for p, v in enumerate(idx.cat_venues)}
        gt.poi2gps = {int(p): (float(a), float(b)) for p, (a, b) in enumerate(zip(idx.venue_lat, idx.venue_lon))}
        train_file = os.path.join(tmp, "triplet_input_train.txt")
        n_rows = _write_triplet_input(ex, idx, train_file)
        friend = os.path.join(tmp, "friend_empty.txt")
        open(friend, "w").close()
        gt.generate_entity_file(os.path.join(kg, "e_map.txt"))
        raw = os.path.join(tmp, "train_triplets_raw.txt")
        gt.generate_train_test_triplets(train_file, raw, friend)
        with open(raw) as f:
            lines = list(dict.fromkeys(ln.rstrip("\n") for ln in f if ln.strip()))
        with open(os.path.join(kg, "train.txt"), "w") as f:
            f.write("\n".join(lines) + "\n")
        os.remove(raw)
        shutil.copyfile(m.relation_file, os.path.join(kg, "r_map.txt"))
        rel_counts = pd.Series([ln.split("\t")[2] for ln in lines]).value_counts().sort_index().to_dict()
        st["n_items"] = len(lines)
        info.update(n_users=idx.U, n_train_users=idx.n_train, n_triplet_users=int(len(idx.triplet_users)),
                    n_venues=idx.L, n_rows_in_triplets=n_rows, n_triplets=len(lines),
                    triplets_by_relation={str(k): int(v) for k, v in rel_counts.items()})
        log(f"prepare: {idx.U} users ({idx.n_train} with >= {cfg['min_checkins']} visible check-ins), {idx.L} "
            f"catalog venues, {len(lines)} triplets {info['triplets_by_relation']}")
    ngpu = 1 if device.startswith("cuda") else 0
    with rec.stage("kge_transe", one_time=True, n_gpus=ngpu) as st:
        seed_all(cfg["seed"])
        use_cuda = device.startswith("cuda")
        if use_cuda and not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is not available")
        for mod in (m.kmisc, m.ktrainer, m.kr):
            if hasattr(mod, "USE_CUDA"):
                mod.USE_CUDA = use_cuda
        FL = types.SimpleNamespace(**m.kge_flag_defaults)
        FL.model_type = cfg["kge_model"]
        FL.batch_size = cfg["kge_batch_size"]
        FL.optimizer_type = cfg["kge_optimizer"]
        FL.kge_epochs = cfg["kge_epochs"]
        FL.seed = cfg["seed"]
        FL.kg_test_files = None
        FL.experiment_name = "transe"
        FL.log_path = FL.ckpt_path = tmp
        FL.data_path = tmp
        FL.has_visualization = False
        logger = logging.getLogger(f"kge_{os.getpid()}")
        logger.handlers.clear()
        logger.setLevel(logging.INFO)
        logger.propagate = False
        fh = logging.FileHandler(os.path.join(tmp, "kge.log"))
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(fh)
        train_dataset, eval_datasets, e_map, r_map = m.ltd.load_data(kg, [], FL.batch_size, logger=logger,
                                                                     negtive_samples=FL.negtive_samples)
        entity_total = max(len(e_map), max(e_map.values()))
        relation_total = max(len(r_map), max(r_map.values()))
        if entity_total != idx.U + idx.L:
            raise AssertionError(f"entity_total {entity_total} != users + venues {idx.U + idx.L}")
        model = m.base.init_model(FL, 0, 0, entity_total, relation_total, logger)
        epoch_length = math.ceil(train_dataset[1] / FL.batch_size)
        trainer = m.ktrainer.ModelTrainer(model, logger, epoch_length, FL)
        log(f"prepare: TransE {FL.kge_epochs} epochs x {epoch_length} steps (batch {FL.batch_size}, "
            f"{'cuda' if use_cuda else 'cpu'})")
        m.kr.train_loop(FL, model, trainer, train_dataset, eval_datasets, entity_total, relation_total, logger,
                        vis=None, is_report=False)
        fh.close()
        logger.removeHandler(fh)
        ckpt = trainer.checkpoint_path + "_final"
        st["n_items"] = int(epoch_length * FL.kge_epochs)
        info["kge"] = dict(flags={k: v for k, v in vars(FL).items() if k not in ("log_path", "ckpt_path", "data_path")},
                           entity_total=int(entity_total), relation_total=int(relation_total),
                           steps=int(epoch_length * FL.kge_epochs), checkpoint=os.path.basename(ckpt))
    with rec.stage("graphs", one_time=True, n_gpus=ngpu) as st:
        dev = torch.device(device)
        sd = torch.load(ckpt, map_location="cpu", weights_only=False)["model_state_dict"]
        ent, rel = sd["ent_embeddings.weight"], sd["rel_embeddings.weight"]
        U = idx.U
        m.construct_loc_loc_graph.device = dev
        m.construct_user_loc_graph.device = dev
        user_encoder = nn.Embedding.from_pretrained(ent[:U]).to(dev)
        loc_encoder = nn.Embedding.from_pretrained(ent[U:]).to(dev)
        rel_encoder = nn.Embedding.from_pretrained(rel).to(dev)
        interact_pref = rel_encoder(torch.LongTensor([0]).to(dev))
        temporal_pref = rel_encoder(torch.LongTensor([1]).to(dev))
        common = dict(loc_count=idx.L, user_count=U, L1_flag=True, model_type=cfg["kge_model"])
        loc_file = os.path.join(tmp, f"loc_temporal_{cfg['loc_threshold']}.pkl")
        m.construct_loc_loc_graph.construct_transition_graph(
            types.SimpleNamespace(loc_graph=True, threshold=cfg["loc_threshold"], **common), loc_file, loc_encoder,
            temporal_pref)
        ul_file = os.path.join(tmp, f"user-loc_{cfg['user_loc_threshold']}.pkl")
        m.construct_user_loc_graph.construct_transition_graph(
            types.SimpleNamespace(loc_graph=False, threshold=cfg["user_loc_threshold"], **common), ul_file,
            user_encoder, loc_encoder, interact_pref)
        with open(loc_file, "rb") as f:
            g1 = pickle.load(f)
        with open(ul_file, "rb") as f:
            g2 = pickle.load(f)
        info["graphs"] = dict(loc_graph=os.path.basename(loc_file), loc_graph_shape=list(g1.shape),
                              loc_graph_nnz=int(g1.nnz), user_loc_graph=os.path.basename(ul_file),
                              user_loc_graph_shape=list(g2.shape), user_loc_graph_nnz=int(g2.nnz))
        st["n_items"] = int(g1.shape[0] + g2.shape[0])
    info["cost_stages"] = rec.stages
    with open(os.path.join(tmp, "DONE.json"), "w") as f:
        json.dump(info, f, indent=1, default=str)
    try:
        os.replace(tmp, pdir)
    except OSError:
        if not os.path.exists(done):
            raise
        shutil.rmtree(tmp, ignore_errors=True)
    log(f"prepare: done -> {os.path.relpath(pdir, ROOT)}")
    return pdir


def _seconds(ts) -> np.ndarray:
    a = np.asarray(ts, dtype="datetime64[s]").astype(np.int64)
    return a.astype(np.float64)


class QueryWindows:

    def __init__(self, ex: Export, split: str, idx: Index, query_ids, seq_len: int):
        qi = ex.query_inputs(split)
        tab = qi.table
        pos = pd.Series(np.arange(len(tab)), index=tab["query_id"].to_numpy(np.int64))
        rv = qi.rows["venue_id"].to_numpy(np.int64)
        rt = _seconds(qi.rows["t_utc"].to_numpy())
        ra, rb = qi.rows["lat"].to_numpy(np.float64), qi.rows["lon"].to_numpy(np.float64)
        src = qi.prefix_source
        sv = src["venue_id"].to_numpy(np.int64)
        stt = _seconds(src["t_utc"].to_numpy())
        sa, sb = src["lat"].to_numpy(np.float64), src["lon"].to_numpy(np.float64)
        users = tab["user_id"].to_numpy(np.int64)
        self.query_ids = np.asarray([int(q) for q in query_ids], dtype=np.int64)
        n = len(self.query_ids)
        self.x = np.zeros((n, seq_len), dtype=np.int64)
        self.t = np.zeros((n, seq_len), dtype=np.float64)
        self.s = np.zeros((n, seq_len, 2), dtype=np.float64)
        self.length = np.zeros(n, dtype=np.int64)
        self.user = np.full(n, -1, dtype=np.int64)
        self.n_dropped_noncatalog = 0
        for k, qid in enumerate(self.query_ids):
            i = int(pos[qid])
            a, b = qi.hist_pos[i]
            c, d = qi.prefix_pos[i]
            v = np.concatenate([rv[a:b], sv[c:d]])
            p = idx.pidx_of[v]
            keep = np.flatnonzero(p >= 0)
            self.n_dropped_noncatalog += int(len(v) - len(keep))
            keep = keep[-seq_len:]
            ln = len(keep)
            self.length[k] = ln
            self.user[k] = idx.uidx(int(users[i]))
            if ln == 0:
                continue
            tt = np.concatenate([rt[a:b], stt[c:d]])[keep]
            la = np.concatenate([ra[a:b], sa[c:d]])[keep]
            lo = np.concatenate([rb[a:b], sb[c:d]])[keep]
            self.x[k, :ln], self.t[k, :ln] = p[keep], tt
            self.s[k, :ln, 0], self.s[k, :ln, 1] = la, lo
            self.x[k, ln:], self.t[k, ln:], self.s[k, ln:] = self.x[k, ln - 1], self.t[k, ln - 1], self.s[k, ln - 1]
        self.ok = (self.length > 0) & (self.user >= 0)


class Bundle:
    def __init__(self, fb_trainer, h0_strategy, idx: Index, cfg: dict, device):
        self.fb_trainer, self.h0_strategy, self.idx, self.cfg, self.device = fb_trainer, h0_strategy, idx, cfg, device


def build_bundle(pdir: str, idx: Index, cfg: dict, device: str) -> Bundle:
    import torch
    from scipy.sparse import coo_matrix, csr_matrix
    m = authors_modules()
    with open(os.path.join(pdir, f"loc_temporal_{cfg['loc_threshold']}.pkl"), "rb") as f:
        transition_graph = coo_matrix(pickle.load(f))
    with open(os.path.join(pdir, f"user-loc_{cfg['user_loc_threshold']}.pkl"), "rb") as f:
        interact_graph = csr_matrix(pickle.load(f))
    seed_all(cfg["seed"])
    fbt = m.trainer.FlashbackTrainer(cfg["lambda_t"], cfg["lambda_s"], cfg["lambda_loc"], cfg["lambda_user"],
                                     cfg["use_weight"], transition_graph, None, None, cfg["use_graph_user"],
                                     cfg["use_spatial_graph"], interact_graph)
    rf = m.network.RnnFactory(cfg["rnn"])
    h0s = m.network.create_h0_strategy(cfg["hidden_dim"], rf.is_lstm())
    dev = torch.device(device)
    fbt.prepare(idx.L, idx.U, cfg["hidden_dim"], rf, dev)
    return Bundle(fbt, h0s, idx, cfg, dev)


def _pop_order(ex: Export, idx: Index) -> np.ndarray:
    ven = ex.venues.sort_values("venue_id")
    ven = ven[ven["in_catalog"].to_numpy(bool)]
    o = np.lexsort((ven["venue_id"].to_numpy(), -ven["n_visible_train_visits"].to_numpy()))
    return ven["venue_id"].to_numpy(np.int64)[o]


def predict_windows(b: Bundle, ex: Export, split: str, query_ids, chunk: int = 256) -> pd.DataFrame:
    import torch
    idx, cfg = b.idx, b.cfg
    qw = QueryWindows(ex, split, idx, query_ids, cfg["sequence_length"])
    cands = ex.candidates(split)
    cmap = dict(zip(cands["query_id"].to_numpy(np.int64), cands["cand_venue_ids"]))
    pop = _pop_order(ex, idx)
    n = len(qw.query_ids)
    top = [None] * n
    cand = [None] * n
    ok = np.flatnonzero(qw.ok)
    with torch.no_grad():
        for s0 in range(0, len(ok), chunk):
            sel = ok[s0:s0 + chunk]
            B = len(sel)
            selp = np.concatenate([sel, sel[:1]]) if B == 1 else sel
            x = torch.as_tensor(qw.x[selp].T.copy(), dtype=torch.long, device=b.device)
            t = torch.as_tensor(qw.t[selp].T.copy(), dtype=torch.float32, device=b.device)
            s = torch.as_tensor(qw.s[selp].transpose(1, 0, 2).copy(), dtype=torch.float32, device=b.device)
            au = torch.as_tensor(qw.user[selp], dtype=torch.long, device=b.device).view(1, -1)
            h = b.h0_strategy.on_init(len(selp), b.device)
            out_t, _ = b.fb_trainer.evaluate(x, t, t, s, None, None, None, h, au)
            lens = torch.as_tensor(qw.length[selp] - 1, device=b.device)
            logits = out_t[torch.arange(len(selp), device=b.device), lens][:B]
            order = torch.sort(logits, dim=1, descending=True, stable=True).indices[:, :TOPK].cpu().numpy()
            lg = logits.cpu().numpy()
            for r, k in enumerate(sel):
                top[k] = idx.cat_venues[order[r]].tolist()
                c = np.asarray(cmap[int(qw.query_ids[k])], dtype=np.int64)
                p = idx.pidx_of[c]
                sc = np.where(p >= 0, lg[r][np.maximum(p, 0)], -np.inf)
                cand[k] = c[np.argsort(-sc, kind="stable")].tolist()
    nfb = 0
    for k in range(n):
        if top[k] is None:
            nfb += 1
            top[k] = pop[:TOPK].tolist()
            cand[k] = [int(v) for v in cmap[int(qw.query_ids[k])]]
    df = pd.DataFrame({"query_id": qw.query_ids, "ranked_venue_ids": top, "cand_ranked_venue_ids": cand,
                       "n_window": qw.length, "fallback": ~qw.ok})
    df.attrs["n_fallback"] = nfb
    df.attrs["n_dropped_noncatalog"] = qw.n_dropped_noncatalog
    return df


def train_key(ex: Export, cfg: dict) -> str:
    f = export_info(ex)["files"]
    return _hash(dict(v=ADAPTER_VERSION, prep=prepare_key(ex, cfg), cfg={k: cfg[k] for k in TRAIN_KEYS},
                      trainq=f.get("query_inputs_train.parquet"), src=apply_patches.src_key()))


def train_dir(ex: Export, cfg: dict, work_root: str) -> str:
    e = export_info(ex)
    return os.path.join(work_root, e["ds"] or "unknown", e["policy"] or "unknown", f"train_{train_key(ex, cfg)}")


def _write_checkins_file(ex: Export, idx: Index, path: str) -> dict:
    rows = ex.rows
    rows = rows[rows["user_id"].isin(set(idx.train_users.tolist()))].sort_values("row_id")
    groups = dict(tuple(rows.groupby("user_id", sort=True)))
    chron = {}
    with open(path, "w") as f:
        for uid in idx.train_users:
            g = groups[int(uid)]
            u = idx.uidx(int(uid))
            chron[u] = g["row_id"].to_numpy(np.int64)
            ts = pd.to_datetime(g["t_utc"]).dt.strftime("%Y-%m-%dT%H:%M:%SZ").to_numpy()
            la, lo, v = g["lat"].to_numpy(), g["lon"].to_numpy(), g["venue_id"].to_numpy()
            for j in range(len(g) - 1, -1, -1):
                f.write(f"{u}\t{ts[j]}\t{float(la[j])!r}\t{float(lo[j])!r}\t{int(v[j])}\n")
    return chron


def _val_acc1(b: Bundle, ex: Export, ds: str, qids) -> dict:
    from poibench.evaluate import val_feedback
    p = predict_windows(b, ex, "val", qids)
    return val_feedback(ds, p[["query_id", "ranked_venue_ids"]], query_ids=qids)


def train(ex: Export, cfg: dict, work_root: str, pdir: str, device: str = "cpu") -> str:
    import torch
    from torch.utils.data import DataLoader
    tdir = train_dir(ex, cfg, work_root)
    model_path = os.path.join(tdir, "model.pt")
    if os.path.exists(model_path):
        log(f"train: reusing {os.path.relpath(tdir, ROOT)}")
        return tdir
    os.makedirs(tdir, exist_ok=True)
    m = authors_modules()
    e = export_info(ex)
    idx = Index.load(os.path.join(pdir, "index.npz"))
    rec = CostRecorder()
    with rec.stage("train_data", one_time=True, n_gpus=0) as st:
        fpath = os.path.join(tdir, f"checkins-4sq-{e['ds']}.txt")
        chron = _write_checkins_file(ex, idx, fpath)
        loader = m.dataloader.PoiDataloader(0, cfg["min_checkins"])
        loader.user2id = {u: u for u in range(idx.n_train)}
        loader.poi2id = {int(v): int(p) for p, v in enumerate(idx.cat_venues)}
        loader.poi2gps = {int(p): (float(a), float(b)) for p, (a, b) in enumerate(zip(idx.venue_lat, idx.venue_lon))}
        loader.read_pois(fpath)
        if loader.users != list(range(idx.n_train)) or len(loader.poi2id) != idx.L:
            raise AssertionError("authors' loader did not reproduce the user / POI index")
        venue_of_row = ex.rows.set_index("row_id")["venue_id"]
        for u in range(idx.n_train):
            exp = idx.pidx_of[venue_of_row.loc[chron[u]].to_numpy(np.int64)]
            if list(loader.locs[u]) != exp.tolist():
                raise AssertionError(f"user {u}: loader sequence differs from the visible rows")
            if len(loader.locs[u]) < cfg["min_checkins"]:
                raise AssertionError("user below min_checkins in the training file")
        n_act = idx.n_train
        if n_act < 3:
            raise RuntimeError(f"only {n_act} users with >= {cfg['min_checkins']} visible check-ins: cannot train")
        batch = cfg["batch_size"] if n_act > cfg["batch_size"] else n_act - 1
        dataset = loader.create_dataset(cfg["sequence_length"], batch, m.dataset.Split.TRAIN)
        trainq = set(ex.queries("train")["query_id"].astype(np.int64).tolist())
        L = cfg["sequence_length"]
        n_pos = n_lab = 0
        for u in range(n_act):
            rows_u = chron[u]
            for j, seq in enumerate(dataset.sequences_labels[u]):
                for k in range(len(seq)):
                    n_pos += 1
                    lab_row = int(rows_u[1 + j * L + k])
                    if cfg["loss_positions"] == "train_queries" and lab_row not in trainq:
                        seq[k] = -100
                    else:
                        n_lab += 1
        st["n_items"] = n_lab
        data_info = dict(n_train_users=n_act, n_users=idx.U, batch_size=batch, n_window_positions=n_pos,
                         n_loss_positions=n_lab, n_train_queries=len(trainq), n_batches_per_epoch=len(dataset),
                         n_checkins_in_file=int(sum(len(v) for v in chron.values())))
        log(f"train: {data_info}")
    state_path = os.path.join(tdir, "state.pt")
    with rec.stage("train", one_time=True, n_gpus=1 if device.startswith("cuda") else 0) as st:
        b = build_bundle(pdir, idx, cfg, device)
        fbt, h0s, dev = b.fb_trainer, b.h0_strategy, b.device
        optimizer = torch.optim.Adam(fbt.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg["milestones"], gamma=cfg["gamma"])
        dl = DataLoader(dataset, batch_size=1, shuffle=False)
        seed_all(cfg["seed"])
        start, best, history = 0, None, []
        select = cfg["selection"] == "best_val_acc1" and not e["perturbed"]
        val_q = ex.queries("val")["query_id"].to_numpy(np.int64)
        if os.path.exists(state_path):
            S = torch.load(state_path, map_location="cpu", weights_only=False)
            fbt.model.load_state_dict(S["model"])
            optimizer.load_state_dict(S["optimizer"])
            scheduler.load_state_dict(S["scheduler"])
            h0s.h0 = S["h0"]
            dataset.user_permutation = S["user_permutation"]
            random.setstate(S["py_random"])
            np.random.set_state(S["np_random"])
            torch.set_rng_state(S["torch_random"])
            start, best, history = S["epoch"], S["best"], S["history"]
            log(f"train: resumed after epoch {start}")
        n_steps = n_skipped = 0
        for ep in range(start, cfg["epochs"]):
            t0 = time.time()
            h = h0s.on_init(batch, dev)
            dataset.shuffle_users()
            losses = []
            for i, (x, t, t_slot, s, y, y_t, y_t_slot, y_s, reset_h, active_users) in enumerate(dl):
                for j, reset in enumerate(reset_h):
                    if reset:
                        h[0, j] = h0s.on_reset(active_users[0][j])
                x = x.squeeze().to(dev)
                t = t.squeeze().to(dev)
                t_slot = t_slot.squeeze().to(dev)
                s = s.squeeze().to(dev)
                y = y.squeeze().to(dev)
                y_t = y_t.squeeze().to(dev)
                y_t_slot = y_t_slot.squeeze().to(dev)
                y_s = y_s.squeeze().to(dev)
                active_users = active_users.to(dev)
                if not bool((y != -100).any()):
                    n_skipped += 1
                    continue
                optimizer.zero_grad()
                loss = fbt.loss(x, t, t_slot, s, y, y_t, y_t_slot, y_s, h, active_users)
                loss.backward(retain_graph=True)
                losses.append(loss.item())
                optimizer.step()
                n_steps += 1
            scheduler.step()
            rec_ep = dict(epoch=ep + 1, loss=float(np.mean(losses)) if losses else None,
                          lr=scheduler.get_last_lr()[0], seconds=round(time.time() - t0, 1))
            if select and (ep + 1) % cfg["validate_epoch"] == 0:
                fb = _val_acc1(b, ex, e["ds"], val_q)
                rec_ep["val"] = {k: fb[k] for k in ("acc@1", "acc@5", "acc@10", "mrr")}
                if best is None or fb["acc@1"] > best["acc@1"]:
                    best = {"epoch": ep + 1, "acc@1": fb["acc@1"],
                            "state": copy.deepcopy(fbt.model.state_dict())}
                torch.save(dict(model=fbt.model.state_dict(), optimizer=optimizer.state_dict(),
                                scheduler=scheduler.state_dict(), h0=h0s.h0, user_permutation=dataset.user_permutation,
                                py_random=random.getstate(), np_random=np.random.get_state(),
                                torch_random=torch.get_rng_state(), epoch=ep + 1, best=best,
                                history=history + [rec_ep]), state_path + ".tmp")
                os.replace(state_path + ".tmp", state_path)
            history.append(rec_ep)
            log(f"train: epoch {rec_ep}")
        st["n_items"] = n_steps
        st["notes"] = f"{n_skipped} batches without train-query labels skipped"
        if select and best is not None:
            fbt.model.load_state_dict(best["state"])
            chosen = best["epoch"]
        else:
            chosen = cfg["epochs"]
        meta = dict(config=cfg, data=data_info, chosen_epoch=chosen,
                    selection=cfg["selection"] if select else "last", history=history,
                    prepare_dir=os.path.relpath(pdir, ROOT), export=e, src=apply_patches.src_key(),
                    n_steps=n_steps, n_skipped_batches=n_skipped, cost_stages=rec.stages)
        torch.save(dict(model=fbt.model.state_dict(), h0=h0s.h0, meta=meta), model_path + ".tmp")
        os.replace(model_path + ".tmp", model_path)
    meta["cost_stages"] = rec.stages
    with open(os.path.join(tdir, "train_meta.json"), "w") as f:
        json.dump(meta, f, indent=1, default=str)
    if os.path.exists(state_path):
        os.remove(state_path)
    log(f"train: done (epoch {chosen}) -> {os.path.relpath(tdir, ROOT)}")
    return tdir


def load_bundle(tdir: str, device: str = "cpu") -> tuple:
    import torch
    S = torch.load(os.path.join(tdir, "model.pt"), map_location="cpu", weights_only=False)
    meta = S["meta"]
    pdir = os.path.join(ROOT, meta["prepare_dir"])
    idx = Index.load(os.path.join(pdir, "index.npz"))
    b = build_bundle(pdir, idx, meta["config"], device)
    b.fb_trainer.model.load_state_dict(S["model"])
    b.h0_strategy.h0 = S["h0"]
    return b, meta, pdir
