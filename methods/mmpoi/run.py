#!/usr/bin/env python
from __future__ import annotations
import os as _pb_os
def _pb_root():
    p = _pb_os.path.dirname(_pb_os.path.abspath(__file__))
    while not _pb_os.path.isfile(_pb_os.path.join(p, "poibench", "__init__.py")) and _pb_os.path.dirname(p) != p:
        p = _pb_os.path.dirname(p)
    return p
_PB_ROOT = _pb_os.environ.get("POIBENCH_ROOT") or _pb_root()
_PB_CACHE = _pb_os.environ.get("POIBENCH_CACHE") or _pb_os.path.join(_PB_ROOT, "work", "cache")

import argparse
import hashlib
import importlib.util
import json
import math
import os
import random
import shutil
import sys
import time
import types
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

ADAPTER_VERSION = "m1"
ATTN_BACKENDS = [SDPBackend.MATH]
UPSTREAM = os.path.join(ROOT, "third_party", "MMPOI")
PINNED_COMMIT = "32f9c76e8768af3b268f0c95679bb7e0c739f766"
UPSTREAM_SHA256 = {
    "model.py": "9a4c9fff93af1968dd4e8d833b5a278de8de8a196af200e25b8440aaa463c8f9",
    "utils.py": "57fbc5ad1c6ba7e8a16f66ea8d25b6349f1ad717b2e13a1fc8ddffe492373a02",
    "build_graph.py": "3a2e4a90e84a00f17554146e3f261f84003bf5256646df95ccb83fe7d1e66dac",
    "train.py": "aed495b5fd2c6a4109c87c287750dc68b175d1ad50bbbb743997c72c099b2130",
    "param_parser.py": "ef22f58063a0ba795be2dabb37b51fa2d5b7aa808ce14f2ef3cb1b305157e9de",
}
VARIANTS = {"image": "mmpoi", "noimage": "mmpoi-noimage"}
SEED = 0
TOPK = 50
SMOKE_USERS = 0.05
SMOKE_EPOCHS = 1
PRED_CHUNK = 256
TASK_WEIGHT_INIT = 0.5
EARLY_STOP_PATIENCE = 5
HP_KEYS = ("knn_k", "A_plus_weights", "gcn_dropout", "transformer_dropout", "batch", "lr", "use_A_plus",
           "loss_geo_enabled", "loss_cat_enabled", "cat_encoder", "loss_time_enabled", "reg_enabled",
           "multimodal_enabled", "cat_loss_type", "fusion_method", "use_time", "use_cat", "cat_type",
           "NLP_embedding_dim", "cat_freeze", "adjust_enabled", "geo_graph_enabled", "multi_loss_weight",
           "task_weight_negative", "multimodal_graph_weights", "short_traj_thres", "time_units", "time_feature",
           "poi_embed_dim", "geo_embed_dim", "multimodal_embed_dim", "user_embed_dim", "gcn_nhid",
           "transformer_nhid", "transformer_nlayers", "transformer_nhead", "time_embed_dim", "cat_embed_dim",
           "epochs", "lr_scheduler_factor", "weight_decay")

TEXT_ENCODER = dict(repo="sentence-transformers/all-mpnet-base-v2",
                    revision="e8c3b32edf5434bc2275fc9bab85f82640a19130", max_seq_length=384,
                    files=["*.json", "*.txt", "model.safetensors", "1_Pooling/*"])
CAPTIONER = dict(repo="Salesforce/blip2-opt-2.7b", revision="59a1ef6c1e5117b3f65523d1c6066825bcf315e3",
                 dtype="float32", batch=32, generate_kwargs={}, files=["*.json", "*.safetensors", "*.txt"])
EMBED_VERSION = "e2"
IMG_CACHE = os.environ.get("POIBENCH_MMPOI_IMG_CACHE", f"{_PB_CACHE}/mmpoi_imgs")


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
            raise RuntimeError(f"third_party/MMPOI/{fn} sha256 {got} != pinned {want} (commit {PINNED_COMMIT})")


def _geohash_fallback():
    m = types.ModuleType("geohash")
    base32 = "0123456789bcdefghjkmnpqrstuvwxyz"

    def encode(latitude, longitude, precision=12):
        lat, lon = [-90.0, 90.0], [-180.0, 180.0]
        out, bit, ch, even = [], 0, 0, True
        while len(out) < precision:
            rng, val = (lon, longitude) if even else (lat, latitude)
            mid = (rng[0] + rng[1]) / 2
            if val > mid:
                ch |= 1 << (4 - bit)
                rng[0] = mid
            else:
                rng[1] = mid
            even = not even
            bit += 1
            if bit == 5:
                out.append(base32[ch])
                bit, ch = 0, 0
        return "".join(out)
    m.encode = encode
    m.__poibench_fallback__ = True
    return m


_MODS = {}


def upstream(name: str):
    if name not in _MODS:
        check_upstream()
        if name != "param_parser":
            sys.modules.setdefault("param_parser", upstream("param_parser"))
        if name == "build_graph":
            try:
                import geohash
            except ImportError:
                sys.modules["geohash"] = _geohash_fallback()
        sp = importlib.util.spec_from_file_location(f"mmpoi_upstream_{name}", os.path.join(UPSTREAM, name + ".py"))
        m = importlib.util.module_from_spec(sp)
        old = sys.argv
        try:
            sys.argv = ["mmpoi"]
            sp.loader.exec_module(m)
        finally:
            sys.argv = old
        _MODS[name] = m
    return _MODS[name]


def upstream_defaults() -> dict:
    pp = upstream("param_parser")
    old = sys.argv
    try:
        sys.argv = ["mmpoi"]
        a = pp.parameter_parser()
    finally:
        sys.argv = old
    return {k: getattr(a, k) for k in HP_KEYS}


def upstream_train_function_source(name: str) -> str:
    import ast
    import textwrap
    with open(os.path.join(UPSTREAM, "train.py")) as f:
        src = f.read()
    tree = ast.parse(src)
    train_fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "train")
    fn = next(n for n in train_fn.body if isinstance(n, ast.FunctionDef) and n.name == name)
    return textwrap.dedent(ast.get_source_segment(src, fn, padded=True))


def check_default_code_path(hp: dict) -> None:
    want = dict(multimodal_enabled=True, geo_graph_enabled=True, use_time=True, use_cat=True, fusion_method="fusion",
                cat_encoder="OneHot", cat_type="SentenceTransformer", cat_freeze=True, cat_loss_type="id_loss",
                adjust_enabled=False, multi_loss_weight=True, task_weight_negative=True, reg_enabled=False,
                loss_time_enabled=False, loss_cat_enabled=False, loss_geo_enabled=True, use_A_plus=True)
    bad = {k: (hp[k], v) for k, v in want.items() if hp[k] != v}
    if bad:
        raise RuntimeError(f"upstream defaults changed; mirrored code path does not apply: {bad}")


def seed_all(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


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


def method_name(variant: str) -> str:
    return VARIANTS[variant]


def work_dir(ds: str, policy: str, root: str = ROOT) -> str:
    return os.path.join(root, "work", "mmpoi", ds, policy)


def embed_dir(ds: str, root: str = ROOT) -> str:
    return os.path.join(root, "work", "mmpoi", "embed", ds)


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


def catalog_hash(venues: pd.DataFrame) -> str:
    v = venues.sort_values("venue_id")
    s = "\n".join(f"{int(i)}\t{c}" for i, c in zip(v["venue_id"], v["category"].astype(str)))
    return hashlib.sha256(s.encode()).hexdigest()


def _hub_snapshot(repo: str, revision: str | None, files=None) -> tuple[str, str]:
    from huggingface_hub import snapshot_download
    p = snapshot_download(repo, revision=revision, allow_patterns=files)
    return p, os.path.basename(os.path.normpath(p))


class TextEncoder:

    def __init__(self, device: str, path: str | None = None):
        p, rev = (path, "local") if path else _hub_snapshot(TEXT_ENCODER["repo"], TEXT_ENCODER["revision"],
                                                                    TEXT_ENCODER["files"])
        self.revision, self.device = rev, device
        try:
            from sentence_transformers import SentenceTransformer
            self.st = SentenceTransformer(p, device=device)
            self.st.max_seq_length = TEXT_ENCODER["max_seq_length"]
            self.backend = "sentence-transformers"
        except ImportError:
            from transformers import AutoModel, AutoTokenizer
            self.st = None
            self.tok = AutoTokenizer.from_pretrained(p)
            self.mod = AutoModel.from_pretrained(p).to(device).eval()
            self.backend = "transformers-meanpool"

    @torch.no_grad()
    def encode(self, texts: list, batch: int = 64) -> np.ndarray:
        if not texts:
            return np.zeros((0, 768), dtype=np.float32)
        if self.st is not None:
            return np.asarray(self.st.encode(list(texts), batch_size=batch, convert_to_numpy=True,
                                             normalize_embeddings=False, show_progress_bar=False), dtype=np.float32)
        out = []
        for a in range(0, len(texts), batch):
            b = self.tok(list(texts[a:a + batch]), padding=True, truncation=True,
                         max_length=TEXT_ENCODER["max_seq_length"], return_tensors="pt").to(self.device)
            h = self.mod(**b).last_hidden_state
            m = b["attention_mask"].unsqueeze(-1).float()
            v = (h * m).sum(1) / m.sum(1)
            out.append(torch.nn.functional.normalize(v, dim=-1).float().cpu().numpy())
        return np.concatenate(out).astype(np.float32)


def stage_archive(root: str, man: dict, log=print) -> str:
    src = os.path.join(root, man["archive"])
    if not os.path.isdir(os.path.dirname(IMG_CACHE)):
        return src
    os.makedirs(IMG_CACHE, exist_ok=True)
    dst = os.path.join(IMG_CACHE, os.path.basename(src))
    if not (os.path.exists(dst) and _sha256(dst) == man["archive_sha256"]):
        shutil.copyfile(src, dst + ".tmp")
        if _sha256(dst + ".tmp") != man["archive_sha256"]:
            raise RuntimeError(f"{src}: sha256 mismatch after copy")
        os.replace(dst + ".tmp", dst)
    log(f"[mmpoi] archive staged: {dst}")
    return dst


def read_photo_blobs(archive: str, man: dict, photos: pd.DataFrame) -> list:
    import tarfile
    import zipfile
    if _sha256(archive) != man["archive_sha256"]:
        raise RuntimeError(f"{archive}: sha256 mismatch")
    want = dict(zip(photos["member"], photos["sha256"]))
    blobs = {}
    with zipfile.ZipFile(archive) as z, z.open(man["tar_member"]) as tz, tarfile.open(fileobj=tz, mode="r|gz") as tf:
        for m in tf:
            if m.isreg() and m.name in want:
                b = tf.extractfile(m).read()
                if hashlib.sha256(b).hexdigest() != want[m.name]:
                    raise RuntimeError(f"{m.name}: sha256 mismatch")
                blobs[m.name] = b
    missing = set(want) - set(blobs)
    if missing:
        raise RuntimeError(f"{len(missing)} listed photos not in the archive, e.g. {sorted(missing)[:3]}")
    return [blobs[x] for x in photos["member"]]


def caption_images(blobs: list, device: str, log=print) -> tuple[list, str]:
    import io
    from PIL import Image
    from transformers import Blip2ForConditionalGeneration, Blip2Processor
    p, rev = _hub_snapshot(CAPTIONER["repo"], CAPTIONER["revision"], CAPTIONER["files"])
    dtype = getattr(torch, CAPTIONER["dtype"]) if device.startswith("cuda") else torch.float32
    proc = Blip2Processor.from_pretrained(p)
    model = Blip2ForConditionalGeneration.from_pretrained(p, dtype=dtype).to(device).eval()
    out = []
    bs = CAPTIONER["batch"]
    for a in range(0, len(blobs), bs):
        ims = [Image.open(io.BytesIO(x)).convert("RGB") for x in blobs[a:a + bs]]
        b = proc(images=ims, return_tensors="pt").to(device)
        b["pixel_values"] = b["pixel_values"].to(dtype)
        with torch.no_grad():
            g = model.generate(**b, **CAPTIONER["generate_kwargs"])
        out += [s.strip() for s in proc.batch_decode(g, skip_special_tokens=True)]
        if (a // bs) % 100 == 0:
            log(f"[mmpoi] captions {a + len(ims)}/{len(blobs)}")
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return out, rev


def mean_by_venue(vid: np.ndarray, owner: np.ndarray, E: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pos = {int(v): i for i, v in enumerate(vid)}
    out = np.zeros((len(vid), E.shape[1] if E.ndim == 2 else 768), dtype=np.float64)
    cnt = np.zeros(len(vid), dtype=np.int64)
    for o, e in zip(owner, E):
        out[pos[int(o)]] += e
        cnt[pos[int(o)]] += 1
    has = cnt > 0
    out[has] /= cnt[has, None]
    out[has] /= np.maximum(np.linalg.norm(out[has], axis=1, keepdims=True), 1e-12)
    return out.astype(np.float32), has


def embed(ds: str, device: str, force: bool = False, encoder=None, captioner=None, root: str = ROOT,
          log=print) -> dict:
    edir = embed_dir(ds, root)
    man_p = os.path.join(edir, "embed.json")
    mm = os.path.join(root, "exports_mm", ds)
    with open(os.path.join(mm, "MANIFEST_photos_all.json")) as f:
        pman = json.load(f)
    ph = pd.read_parquet(os.path.join(mm, "venue_photos_all.parquet"))
    vt = pd.read_parquet(os.path.join(mm, "venue_texts.parquet"))
    venues = Export(export_dir(ds, "H-100", root)).venues
    inputs = dict(venue_photos_all_sha256=_sha256(os.path.join(mm, "venue_photos_all.parquet")),
                  photo_archive_sha256=pman["archive_sha256"],
                  venue_texts_sha256=_sha256(os.path.join(mm, "venue_texts.parquet")),
                  catalog_sha256=catalog_hash(venues))
    if not force and os.path.exists(man_p):
        with open(man_p) as f:
            m = json.load(f)
        if m.get("inputs") == inputs and m.get("embed_version") == EMBED_VERSION:
            log(f"[mmpoi] embeddings exist: {edir}")
            return m
    os.makedirs(edir, exist_ok=True)
    vid = venues.sort_values("venue_id")["venue_id"].to_numpy(np.int64)
    if not (set(ph.venue_id.astype(int)) <= set(vid.tolist())
            and np.array_equal(np.sort(vt.venue_id.to_numpy(np.int64)), vid)):
        raise RuntimeError("exports_mm venue ids differ from the export's venues_public")
    vt = vt.set_index("venue_id").loc[vid]
    cat = venues.set_index("venue_id").loc[vid, "category"].astype(str).tolist()
    rec = CostRecorder(None)
    ngpu = 1 if device.startswith("cuda") else 0
    with rec.stage("embed_caption_images", one_time=True, n_gpus=ngpu, n_items=int(len(ph)),
                   notes=f"{CAPTIONER['repo']} captions of all {len(ph)} photos of {ph.venue_id.nunique()} venues "
                         f"(image variant only; incl. staging the photo archive)"):
        arch = stage_archive(root, pman, log=log)
        blobs = read_photo_blobs(arch, pman, ph)
        caps, cap_rev = (captioner(blobs), "test") if captioner is not None else caption_images(blobs, device, log)
        del blobs
    enc = encoder if encoder is not None else TextEncoder(device)
    with rec.stage("embed_text_captions", one_time=True, n_gpus=ngpu, n_items=len(caps),
                   notes=f"{TEXT_ENCODER['repo']} on the captions, normalised mean per venue (image variant only)"):
        E_img, has_img = mean_by_venue(vid, ph["venue_id"].to_numpy(np.int64), enc.encode(caps))
    has_tips = vt["has_tips"].to_numpy(bool)
    with rec.stage("embed_text_meta_review", one_time=True, n_gpus=ngpu,
                   n_items=int(len(vid) + vt["n_tips"].sum()),
                   notes=f"{TEXT_ENCODER['repo']} on categories and tips (both variants)"):
        ucat = sorted(set(cat))
        cmap = dict(zip(ucat, enc.encode(ucat)))
        E_meta = np.stack([cmap[c] for c in cat]).astype(np.float32)
        tips = [t for ts in vt["tips"] for t in list(ts)]
        owner = np.repeat(vid, vt["n_tips"].to_numpy(np.int64))
        E_rev, has_rev = mean_by_venue(vid, owner, enc.encode(tips))
        assert np.array_equal(has_rev, has_tips)
    np.savez(os.path.join(edir, "embeddings.tmp.npz"), venue_id=vid, image=E_img, meta=E_meta, review=E_rev,
             has_image=has_img, has_tips=has_tips)
    os.replace(os.path.join(edir, "embeddings.tmp.npz"), os.path.join(edir, "embeddings.npz"))
    pd.DataFrame({"venue_id": ph["venue_id"].to_numpy(np.int64), "member": ph["member"].to_numpy(),
                  "caption": caps}).to_parquet(os.path.join(edir, "captions.parquet"), index=False)
    m = dict(embed_version=EMBED_VERSION, created_utc=_now(), ds=ds, inputs=inputs,
             text_encoder=dict(TEXT_ENCODER, resolved_revision=enc.revision, backend=enc.backend),
             captioner=dict(CAPTIONER, resolved_revision=cap_rev, device=device),
             photos=dict(rule=pman["rule"], n_photos=int(len(ph)), table="exports_mm/%s/venue_photos_all.parquet" % ds),
             n_venues=int(len(vid)), n_with_image=int(has_img.sum()), n_with_tips=int(has_tips.sum()),
             n_tips=int(vt["n_tips"].sum()), embeddings_sha256=_sha256(os.path.join(edir, "embeddings.npz")),
             cost_stages=rec.stages)
    _json_dump(m, man_p)
    log(f"[mmpoi] embeddings written: {edir} ({m['n_with_image']} venues with photos, {len(ph)} photos, "
        f"{m['n_with_tips']} venues with tips)")
    return m


def load_embeddings(ds: str, venues: pd.DataFrame, root: str = ROOT) -> tuple[dict, dict]:
    edir = embed_dir(ds, root)
    with open(os.path.join(edir, "embed.json")) as f:
        m = json.load(f)
    if m["inputs"]["catalog_sha256"] != catalog_hash(venues):
        raise RuntimeError("embeddings were computed for another venue catalog")
    if _sha256(os.path.join(edir, "embeddings.npz")) != m["embeddings_sha256"]:
        raise RuntimeError("embeddings.npz does not match embed.json")
    z = np.load(os.path.join(edir, "embeddings.npz"))
    return {k: z[k] for k in z.files}, m


def to_mmpoi_frame(ex: Export) -> pd.DataFrame:
    r = ex.rows.sort_values(["t_utc", "row_id"], kind="mergesort")
    cats = sorted(ex.venues.loc[ex.venues["in_catalog"], "category"].astype(str).unique())
    code = {c: i for i, c in enumerate(cats)}
    tl = pd.DatetimeIndex(r["t_local"])
    cat = r["category"].astype(str)
    return pd.DataFrame({
        "user_id": r["user_id"].to_numpy(np.int64),
        "POI_id": r["venue_id"].astype(np.int64).astype(str).to_numpy(),
        "POI_catid": cat.to_numpy(),
        "POI_catid_code": cat.map(code).to_numpy(np.int64),
        "POI_catname": cat.to_numpy(),
        "latitude": r["lat"].to_numpy(), "longitude": r["lon"].to_numpy(),
        "day_of_week": tl.dayofweek.to_numpy(),
        "norm_in_day_time": norm_in_day_time(r["t_local"]),
        "trajectory_id": (r["user_id"].astype(str) + "_" + r["traj_id"].astype(str)).to_numpy(),
        "row_id": r["row_id"].to_numpy(np.int64),
    })


def build_graphs(ex: Export, gdf: pd.DataFrame) -> dict:
    import networkx as nx
    import scipy.sparse as sp
    bg = upstream("build_graph")
    df = gdf.drop(columns=["row_id"]).copy()
    G = bg.build_global_POI_checkin_graph(df)
    df = bg.geohash_add2_df(df)
    Gg = bg.build_global_POI_geo_graph(df)
    ven = ex.venues.sort_values("venue_id")
    catalog = ven.loc[ven["in_catalog"], "venue_id"].to_numpy(np.int64)
    present = set(int(n) for n in G.nodes())
    if not present <= set(catalog.tolist()):
        raise RuntimeError("visible rows contain non-catalog venues")
    vt = ven.set_index("venue_id")
    cats = sorted(ven.loc[ven["in_catalog"], "category"].astype(str).unique())
    code = {c: i for i, c in enumerate(cats)}
    n_added = 0
    for v in catalog:
        if int(v) not in present:
            c = str(vt.at[v, "category"])
            G.add_node(str(int(v)), checkin_cnt=0, poi_catid=c, poi_catid_code=code[c], poi_catname=c,
                       latitude=float(vt.at[v, "lat"]), longitude=float(vt.at[v, "lon"]))
            n_added += 1
    nodelist = list(G.nodes())
    nd = [G.nodes[n] for n in nodelist]
    nodes = pd.DataFrame({"node_name/poi_id": nodelist,
                          "checkin_cnt": [d["checkin_cnt"] for d in nd],
                          "poi_catid": [d["poi_catid"] for d in nd],
                          "poi_catid_code": [d["poi_catid_code"] for d in nd],
                          "poi_catname": [d["poi_catname"] for d in nd],
                          "latitude": [d["latitude"] for d in nd], "longitude": [d["longitude"] for d in nd]})
    nodes["geohash"] = [bg.geohash_encode(la, lo) for la, lo in zip(nodes["latitude"], nodes["longitude"])]
    n_geo_added = 0
    for g in sorted(set(nodes["geohash"]) - set(Gg.nodes())):
        Gg.add_node(g, checkin_cnt=0)
        n_geo_added += 1
    A = sp.csr_matrix(nx.adjacency_matrix(G, nodelist=nodelist), dtype=np.float64)
    glist = list(Gg.nodes())
    A_geo = sp.csr_matrix(nx.adjacency_matrix(Gg, nodelist=glist), dtype=np.float64)
    geo_nodes = pd.DataFrame({"geohash": glist, "checkin_cnt": [Gg.nodes[g]["checkin_cnt"] for g in glist]})
    return dict(nodes=nodes, A=A, geo_nodes=geo_nodes, A_geo=A_geo, n_isolated_added=n_added,
                n_geo_isolated_added=n_geo_added, n_edges=int(A.nnz), n_geo_edges=int(A_geo.nnz))


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
    with rec.stage("prepare", one_time=True, n_gpus=0, notes="export -> MMPOI csv + POI and geo transition graphs"):
        gdf = to_mmpoi_frame(ex)
        gdf.drop(columns=["row_id"]).to_csv(os.path.join(pdir, "mmpoi_train.csv.tmp"), index=False)
        os.replace(os.path.join(pdir, "mmpoi_train.csv.tmp"), os.path.join(pdir, "mmpoi_train.csv"))
        g = build_graphs(ex, gdf)
        for name, df in (("graph_X.csv", g["nodes"]), ("geo_graph_X.csv", g["geo_nodes"])):
            df.to_csv(os.path.join(pdir, name + ".tmp"), index=False)
            os.replace(os.path.join(pdir, name + ".tmp"), os.path.join(pdir, name))
        for name, A in (("graph_A", g["A"]), ("geo_graph_A", g["A_geo"])):
            sp.save_npz(os.path.join(pdir, name + ".tmp.npz"), A)
            os.replace(os.path.join(pdir, name + ".tmp.npz"), os.path.join(pdir, name + ".npz"))
    meta = dict(adapter_version=ADAPTER_VERSION, created_utc=_now(), ds=ex.manifest.get("ds"),
                policy=ex.manifest.get("policy"), export_files_sha256=msha, n_rows=int(len(gdf)),
                n_nodes=int(len(g["nodes"])), n_isolated_nodes_added=g["n_isolated_added"], n_edges=g["n_edges"],
                n_geo_nodes=int(len(g["geo_nodes"])), n_geo_isolated_added=g["n_geo_isolated_added"],
                n_geo_edges=g["n_geo_edges"], n_trajectories=int(gdf["trajectory_id"].nunique()),
                cost_stage=rec.stages[0])
    _json_dump(meta, meta_p)
    return meta


def build_meta(ex: Export, pdir: str, variant: str, emb: dict) -> dict:
    import scipy.sparse as sp
    from sklearn.preprocessing import OneHotEncoder
    U = upstream("utils")
    nodes = pd.read_csv(os.path.join(pdir, "graph_X.csv"), dtype={"node_name/poi_id": str, "poi_catid": str,
                                                                    "geohash": str})
    geo_nodes = pd.read_csv(os.path.join(pdir, "geo_graph_X.csv"), dtype={"geohash": str})
    raw_A = sp.load_npz(os.path.join(pdir, "graph_A.npz")).toarray()
    geo_raw_A = sp.load_npz(os.path.join(pdir, "geo_graph_A.npz")).toarray()
    cat_list = ["unknown" if str(c).lower() == "nan" else c for c in nodes["poi_catid"].tolist()]
    enc = OneHotEncoder()
    enc.fit([[c] for c in cat_list])
    X = enc.transform([[c] for c in cat_list]).toarray().astype(np.float32)
    geo_list = geo_nodes["geohash"].tolist()
    genc = OneHotEncoder()
    genc.fit([[g] for g in geo_list])
    X_geo = genc.transform([[g] for g in geo_list]).toarray().astype(np.float32)
    A = np.asarray(U.calculate_laplacian_matrix(raw_A, mat_type='hat_rw_normd_lap_mat'))
    A_geo = np.asarray(U.calculate_laplacian_matrix(geo_raw_A, mat_type='hat_rw_normd_lap_mat'))
    node_venue = nodes["node_name/poi_id"].astype(np.int64).to_numpy()
    cat_id2idx = {}
    for i, c in enumerate(cat_list):
        cat_id2idx[c] = i
    poi_cat = np.array([cat_id2idx[c] for c in cat_list], dtype=np.int64)
    geo_id2idx = {g: i for i, g in enumerate(geo_list)}
    poi_geo = np.array([geo_id2idx[g] for g in nodes["geohash"]], dtype=np.int64)
    pos = pd.Series(np.arange(len(emb["venue_id"])), index=emb["venue_id"])
    rows = pos.loc[node_venue].to_numpy()
    X_image = emb["image"][rows].astype(np.float32)
    if variant == "noimage":
        X_image = np.zeros_like(X_image)
    user_ids = np.sort(ex.rows["user_id"].unique()).astype(np.int64)
    return dict(X=X, A_dense=A, X_geo=X_geo, A_geo_dense=A_geo, X_image=X_image,
                X_meta=emb["meta"][rows].astype(np.float32), X_review=emb["review"][rows].astype(np.float32),
                node_venue=node_venue, poi_venue=node_venue.copy(), poi_cat=poi_cat, num_cats=len(cat_list),
                poi_geo=poi_geo, user_ids=user_ids, node_checkin_cnt=nodes["checkin_cnt"].to_numpy(np.int64),
                n_venues_all=int(len(ex.venues)), variant=variant,
                node_has_image=emb["has_image"][rows] & (variant == "image"), node_has_tips=emb["has_tips"][rows])


def meta_for_ckpt(meta: dict) -> dict:
    import scipy.sparse as sp
    m = {k: v for k, v in meta.items() if k not in ("A_dense", "A_geo_dense")}
    for k in ("A_dense", "A_geo_dense"):
        A = sp.csr_matrix(meta[k])
        m[k.replace("_dense", "_sparse")] = dict(data=A.data, indices=A.indices, indptr=A.indptr, shape=A.shape)
    return m


def meta_from_ckpt(m: dict) -> dict:
    import scipy.sparse as sp
    m = dict(m)
    for k in ("A_sparse", "A_geo_sparse"):
        a = m.pop(k)
        m[k.replace("_sparse", "_dense")] = sp.csr_matrix((a["data"], a["indices"], a["indptr"]),
                                                          shape=tuple(a["shape"])).toarray()
    return m


class MMPOI:

    MOD_NAMES = ("image_embed", "review_embed", "meta_embed", "poi_embed", "geo_embed", "user_embed", "time_embed",
                 "cat_embed", "fuse_mm", "fuse1", "fuse2", "seq_model")

    def __init__(self, meta: dict, hp: dict, device: str):
        M = upstream("model")
        self.meta, self.hp, self.device = meta, hp, device

        def t(a):
            return torch.as_tensor(np.asarray(a, dtype=np.float32)).to(device=device, dtype=torch.float)
        self.X, self.A = t(meta["X"]), t(meta["A_dense"])
        self.X_geo, self.A_geo = t(meta["X_geo"]), t(meta["A_geo_dense"])
        self.X_image, self.X_review, self.X_meta = t(meta["X_image"]), t(meta["X_review"]), t(meta["X_meta"])
        mmd = hp["multimodal_embed_dim"]
        self.image_embed_model = M.GCN(self.X_image.shape[1], hp["gcn_nhid"], mmd, hp["gcn_dropout"])
        self.review_embed_model = M.GCN(self.X_review.shape[1], hp["gcn_nhid"], mmd, hp["gcn_dropout"])
        self.meta_embed_model = M.GCN(self.X_meta.shape[1], hp["gcn_nhid"], mmd, hp["gcn_dropout"])
        self.poi_embed_model = M.GCN(self.X.shape[1], hp["gcn_nhid"], hp["poi_embed_dim"], hp["gcn_dropout"])
        self.geo_embed_model = M.GCN(self.X_geo.shape[1], hp["gcn_nhid"], hp["geo_embed_dim"], hp["gcn_dropout"])
        n_users, n_pois, n_geos = len(meta["user_ids"]), len(meta["poi_venue"]), self.X_geo.shape[0]
        self.user_embed_model = M.UserEmbeddings(n_users, hp["user_embed_dim"])
        self.time_embed_model = M.Time2Vec('sin', out_dim=hp["time_embed_dim"])
        self.cat_embed_model = M.CategoryEmbeddings(meta["num_cats"], hp["cat_embed_dim"],
                                                    np.asarray(meta["X"], dtype=np.float64))
        self.embed_fuse_multimodal_model = M.FuseMultimodalEmbeddings(mmd, 3)
        self.embed_fuse_model1 = M.FuseEmbeddings(hp["user_embed_dim"], hp["poi_embed_dim"])
        self.embed_fuse_model2 = M.FuseEmbeddings(hp["time_embed_dim"], hp["geo_embed_dim"])
        seq_in = hp["poi_embed_dim"] + hp["user_embed_dim"] + hp["time_embed_dim"] + hp["geo_embed_dim"] + mmd * 3
        self.seq_model = M.TransformerModel(n_pois, meta["num_cats"], n_geos, seq_in, hp["transformer_nhead"],
                                            hp["transformer_nhid"], hp["transformer_nlayers"], hp["use_time"],
                                            hp["use_cat"], hp["geo_graph_enabled"], dropout=hp["transformer_dropout"])
        self.task_weights = [torch.nn.Parameter(torch.tensor(TASK_WEIGHT_INIT, dtype=torch.float32, device=device))
                             for _ in range(4)]
        for m in self.modules():
            m.to(device=device)
        self.poi_geo = torch.as_tensor(np.asarray(meta["poi_geo"], dtype=np.int64), device=device)

    def modules(self):
        return [self.image_embed_model, self.review_embed_model, self.meta_embed_model, self.poi_embed_model,
                self.geo_embed_model, self.user_embed_model, self.time_embed_model, self.cat_embed_model,
                self.embed_fuse_multimodal_model, self.embed_fuse_model1, self.embed_fuse_model2, self.seq_model]

    def optim_parameters(self):
        ps = []
        for m in (self.poi_embed_model, self.geo_embed_model, self.user_embed_model, self.image_embed_model,
                  self.meta_embed_model, self.review_embed_model, self.time_embed_model,
                  self.embed_fuse_multimodal_model, self.embed_fuse_model1, self.embed_fuse_model2, self.seq_model):
            ps += list(m.parameters())
        return ps + self.task_weights

    def train(self, mode=True):
        for m in self.modules():
            m.train(mode)

    def state_dict(self):
        sd = {n: m.state_dict() for n, m in zip(self.MOD_NAMES, self.modules())}
        sd["task_weights"] = [w.detach().cpu().clone() for w in self.task_weights]
        return sd

    def load_state_dict(self, sd):
        for n, m in zip(self.MOD_NAMES, self.modules()):
            m.load_state_dict(sd[n])
        with torch.no_grad():
            for w, v in zip(self.task_weights, sd["task_weights"]):
                w.copy_(v)

    def graph_embeddings(self) -> dict:
        return dict(geo=self.geo_embed_model(self.X_geo, self.A_geo), poi=self.poi_embed_model(self.X, self.A),
                    image=self.image_embed_model(self.X_image, self.A),
                    review=self.review_embed_model(self.X_review, self.A),
                    meta=self.meta_embed_model(self.X_meta, self.A))

    def embed(self, user_idx, poi_idx, times, E):
        user = self.user_embed_model(user_idx)
        poi = E["poi"][poi_idx]
        geo = E["geo"][self.poi_geo[poi_idx]]
        t = self.time_embed_model(times.view(-1, 1))
        mm = self.embed_fuse_multimodal_model(E["image"][poi_idx], E["meta"][poi_idx], E["review"][poi_idx])
        return torch.cat((self.embed_fuse_model1(user, poi), self.embed_fuse_model2(t, geo), mm), dim=-1)

    def seq_forward(self, x):
        sm = self.seq_model
        with sdpa_kernel(ATTN_BACKENDS):
            src_mask = sm.generate_square_subsequent_mask(x.shape[0]).to(x.device)
            return sm(x, src_mask)

    def precisions(self):
        return [torch.exp(-(w - w.detach() + TASK_WEIGHT_INIT)) for w in self.task_weights]


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
    poi_cat, poi_geo = np.asarray(meta["poi_cat"]), np.asarray(meta["poi_geo"])
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
        lp = poi[s + 1:e]
        stats["n_labels_used"] += int(ok.sum())
        stats["n_labels_masked"] += int((~ok).sum())
        seqs.append(dict(key=f"{int(user[s])}_{int(traj[s])}", traj_id=int(traj[s]), user=uid2idx[int(user[s])],
                         poi=poi[s:e - 1], t=tfeat[s:e - 1], y_poi=np.where(ok, lp, -1),
                         y_time=np.where(ok, tfeat[s + 1:e], -1.0), y_cat=np.where(ok, poi_cat[lp], -1),
                         y_geo=np.where(ok, poi_geo[lp], -1)))
    seqs.sort(key=lambda d: d["key"])
    stats["n_sequences"] = len(seqs)
    return seqs, stats


def collate(model: MMPOI, seqs: list, E: dict):
    dev = model.device
    lens = [len(s["poi"]) for s in seqs]
    B, L = len(seqs), max(lens)
    users = torch.as_tensor(np.concatenate([np.full(n, s["user"]) for s, n in zip(seqs, lens)]), device=dev)
    pois = torch.as_tensor(np.concatenate([s["poi"] for s in seqs]), device=dev)
    ts = torch.as_tensor(np.concatenate([s["t"] for s in seqs]), dtype=torch.float, device=dev)
    e = model.embed(users, pois, ts, E)
    valid = torch.zeros(B, L, dtype=torch.bool, device=dev)
    for i, n in enumerate(lens):
        valid[i, :n] = True
    x = torch.full((B, L, e.shape[-1]), -1.0, device=dev, dtype=e.dtype)
    x[valid] = e

    def pad(key, dtype):
        return pad_sequence([torch.as_tensor(np.asarray(s[key])) for s in seqs], batch_first=True,
                            padding_value=-1).to(device=dev, dtype=dtype)
    return x, pad("y_poi", torch.long), pad("y_time", torch.float), pad("y_cat", torch.long), pad("y_geo", torch.long)


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
def predict_scores(model: MMPOI, qs: dict, chunk: int = PRED_CHUNK):
    model.train(False)
    n = len(qs["query_id"])
    E = model.graph_embeddings()
    usable = [i for i in range(n) if qs["user"][i] >= 0 and len(qs["poi"][i]) > 0]
    scores = [None] * n
    bsz = model.hp["batch"]
    groups = {}
    for i in usable:
        groups.setdefault((len(qs["poi"][i]) - 1) % bsz, []).append(i)
    for s, idx in sorted(groups.items()):
        for a in range(0, len(idx), chunk):
            ii = idx[a:a + chunk]
            last_poi = torch.as_tensor([int(qs["poi"][i][-1]) for i in ii], device=model.device)
            e = model.embed(torch.as_tensor([qs["user"][i] for i in ii], device=model.device), last_poi,
                            torch.as_tensor([float(qs["t"][i][-1]) for i in ii], dtype=torch.float,
                                            device=model.device), E)
            x = torch.full((s + 1, len(ii), e.shape[-1]), -1.0, device=model.device)
            x[s] = e
            yp = model.seq_forward(x)[0]
            for j, i in enumerate(ii):
                scores[i] = yp[s, j]
    return scores


def rank_lists(model: MMPOI, qs: dict, scores, cands: dict | None):
    poi_venue = np.asarray(model.meta["poi_venue"], dtype=np.int64)
    node_venue = np.asarray(model.meta["node_venue"], dtype=np.int64)
    cnt = np.asarray(model.meta["node_checkin_cnt"], dtype=np.int64)
    pop = node_venue[np.lexsort((node_venue, -cnt))][:TOPK].tolist()
    v2p = np.full(model.meta["n_venues_all"], -1, dtype=np.int64)
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


def load_model(ckpt_path: str, device: str = "cpu") -> MMPOI:
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = MMPOI(meta_from_ckpt(ck["meta"]), ck["hp"], device)
    model.load_state_dict(ck["state"])
    model.train(False)
    model.ckpt_info = {k: ck[k] for k in ("epoch", "variant", "ds", "policy", "monitor_score") if k in ck}
    return model


def default_ckpt(ds: str, policy: str, variant: str) -> str:
    return os.path.join(work_dir(ds, policy), variant, "smoke", "best.pt")


def predict(export_dir, split, query_ids, work_dir=None, ckpt=None, device=None, variant=None, with_cand=True):
    ex = Export(export_dir)
    variant = variant or os.environ.get("MMPOI_VARIANT", "image")
    ckpt = ckpt or os.environ.get("MMPOI_CKPT") or default_ckpt(ex.manifest["ds"], ex.manifest["policy"], variant)
    model = ckpt if isinstance(ckpt, MMPOI) else load_model(ckpt, device or os.environ.get("MMPOI_DEVICE", "cpu"))
    qs = query_sequences(ex, split, query_ids, model.meta)
    scores = predict_scores(model, qs)
    cands = None
    if with_cand:
        cd = ex.candidates(split)
        cands = dict(zip(cd["query_id"].to_numpy(np.int64).tolist(), cd["cand_venue_ids"]))
    df, n_fb = rank_lists(model, qs, scores, cands)
    df.attrs.update(n_fallback=n_fb, n_context_rows_dropped=qs["n_dropped"])
    return df


def train(ex: Export, ed: str, wdir: str, variant: str, device: str, epochs: int | None = None,
          user_frac: float | None = None, out_sub: str = "train", emb_root: str = ROOT, log=print) -> dict:
    from poibench.evaluate import val_feedback
    hp = upstream_defaults()
    check_default_code_path(hp)
    if epochs is not None:
        hp["epochs"] = int(epochs)
    ds = ex.manifest["ds"]
    emb, emb_man = load_embeddings(ds, ex.venues, emb_root)
    odir = os.path.join(wdir, variant, out_sub)
    os.makedirs(odir, exist_ok=True)
    done_p = os.path.join(odir, "DONE.json")
    run_key = dict(adapter_version=ADAPTER_VERSION, variant=variant, hp=hp, user_frac=user_frac, seed=SEED,
                   export_files_sha256=ex.manifest.get("files", {}), embeddings_sha256=emb_man["embeddings_sha256"])
    rk = json.loads(json.dumps(run_key, default=str))
    if os.path.exists(done_p):
        with open(done_p) as f:
            d = json.load(f)
        if d.get("run_key") == rk:
            log(f"[mmpoi] {odir}: training already done (best epoch {d.get('best_epoch')})")
            return d
    pdir = os.path.join(wdir, "prepare")
    pmeta = prepare(ex, wdir)
    seed_all(SEED)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    meta = build_meta(ex, pdir, variant, emb)
    users = None
    if user_frac is not None:
        allu = np.sort(ex.rows["user_id"].unique())
        users = np.sort(np.random.default_rng(SEED).choice(allu, size=max(1, math.ceil(user_frac * len(allu))),
                                                           replace=False))
    seqs, sstats = train_sequences(ex, meta, hp, users)
    seed_all(SEED)
    model = MMPOI(meta, hp, device)
    optimizer = torch.optim.Adam(params=model.optim_parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    criterion_poi = torch.nn.CrossEntropyLoss(ignore_index=-1)
    criterion_cat = torch.nn.CrossEntropyLoss(ignore_index=-1)
    criterion_geo = torch.nn.CrossEntropyLoss(ignore_index=-1)
    criterion_time = upstream("utils").maksed_mse_loss
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=hp["lr_scheduler_factor"])

    val_q = ex.queries("val")
    if users is not None:
        val_q = val_q[val_q["user_id"].isin(set(int(u) for u in users))]
    val_ids = val_q["query_id"].to_numpy(np.int64).tolist()
    vqs = query_sequences(ex, "val", val_ids, meta)
    vroot = root_of_export(ed)

    state = dict(epoch=-1, best_score=-np.inf, best_epoch=None, history=[], sessions=[], patience=0,
                 prev_es_score=0.0, stopped_early=False)
    last_p, best_p = os.path.join(odir, "last.pt"), os.path.join(odir, "best.pt")
    if os.path.exists(last_p):
        ck = torch.load(last_p, map_location="cpu", weights_only=False)
        if ck.get("run_key") == rk:
            model.load_state_dict(ck["state"])
            optimizer.load_state_dict(ck["optimizer"])
            lr_scheduler.load_state_dict(ck["scheduler"])
            torch.set_rng_state(ck["torch_rng"])
            if device.startswith("cuda") and ck.get("cuda_rng") is not None:
                torch.cuda.set_rng_state_all(ck["cuda_rng"])
            np.random.set_state(ck["np_rng"])
            random.setstate(ck["py_rng"])
            state = ck["train_state"]
            log(f"[mmpoi] resume from {last_p}: epoch {state['epoch']} done, best {state['best_epoch']}")
        else:
            log(f"[mmpoi] {last_p} belongs to another configuration; starting fresh")
    ckpt_common = dict(meta=meta_for_ckpt(meta), hp=hp, variant=variant, ds=ds, policy=ex.manifest["policy"],
                       adapter_version=ADAPTER_VERSION, upstream_commit=PINNED_COMMIT,
                       embeddings_sha256=emb_man["embeddings_sha256"])
    rec = CostRecorder(None)
    ngpu = 1 if device.startswith("cuda") else 0
    log(f"[mmpoi] {ds}/{ex.manifest['policy']}/{variant}: {sstats['n_sequences']} train sequences, "
        f"{len(meta['poi_venue'])} POIs, {meta['X_geo'].shape[0]} geo cells, {len(meta['user_ids'])} users, "
        f"{len(val_ids)} val queries, device {device}")
    with rec.stage("train", one_time=True, n_gpus=ngpu, notes="incl. per-epoch val inference for selection") as srec:
        for epoch in range(state["epoch"] + 1, hp["epochs"]):
            if state["stopped_early"]:
                break
            t0 = time.time()
            model.train(True)
            losses = []
            for b in range(0, len(seqs), hp["batch"]):
                batch = seqs[b:b + hp["batch"]]
                E = model.graph_embeddings()
                x, y_poi, y_time, y_cat, y_geo = collate(model, batch, E)
                y_pred_poi, y_pred_time, y_pred_cat, y_pred_geo = model.seq_forward(x)
                loss_poi = criterion_poi(y_pred_poi.transpose(1, 2), y_poi)
                loss_geo = criterion_geo(y_pred_geo.transpose(1, 2), y_geo)
                with torch.no_grad():
                    loss_time = criterion_time(y_pred_time.squeeze(-1), y_time)
                    loss_cat = criterion_cat(y_pred_cat.transpose(1, 2), y_cat)
                p1, _, _, p4 = model.precisions()
                w1, _, _, w4 = model.task_weights
                loss = p1 * loss_poi + w1 + p4 * loss_geo + w4
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
                losses.append([float(loss.detach()), float(loss_poi.detach()), float(loss_geo.detach()),
                               float(loss_time), float(loss_cat)])
            scores = predict_scores(model, vqs)
            vdf, _ = rank_lists(model, vqs, scores, None)
            fb = val_feedback(ds, vdf, ks=(1, 5, 10, 20), root=vroot, query_ids=val_ids)
            monitor_score = float(fb["acc@1"] * 4 + fb["acc@20"])
            es = float(fb["acc@20"] + fb["mrr"])
            state["patience"] = state["patience"] + 1 if es == state["prev_es_score"] else 0
            ml = np.mean(losses, axis=0) if losses else [float("nan")] * 5
            h = dict(epoch=epoch, train_loss=float(ml[0]), train_poi_loss=float(ml[1]), train_geo_loss=float(ml[2]),
                     train_time_loss=float(ml[3]), train_cat_loss=float(ml[4]), val=fb, monitor_score=monitor_score,
                     task_weights=[float(w.detach()) for w in model.task_weights],
                     lr=float(optimizer.param_groups[0]["lr"]), wall_s=round(time.time() - t0, 2))
            state["history"].append(h)
            state["epoch"] = epoch
            if state["patience"] >= EARLY_STOP_PATIENCE:
                state["stopped_early"] = True
                log(f"[mmpoi] early stopping at epoch {epoch}")
            else:
                state["prev_es_score"] = es
                lr_scheduler.step(monitor_score)
                if monitor_score >= state["best_score"]:
                    state["best_score"], state["best_epoch"] = monitor_score, epoch
                    _torch_save(dict(ckpt_common, state=model.state_dict(), epoch=epoch,
                                     monitor_score=monitor_score), best_p)
            _torch_save(dict(run_key=rk, state=model.state_dict(), optimizer=optimizer.state_dict(),
                             scheduler=lr_scheduler.state_dict(), torch_rng=torch.get_rng_state(),
                             cuda_rng=torch.cuda.get_rng_state_all() if device.startswith("cuda") else None,
                             np_rng=np.random.get_state(), py_rng=random.getstate(), train_state=state), last_p)
            with open(os.path.join(odir, "log.jsonl"), "a") as f:
                f.write(json.dumps(h) + "\n")
            log(f"[mmpoi] epoch {epoch}: loss {h['train_loss']:.4f} (poi {h['train_poi_loss']:.4f}, geo "
                f"{h['train_geo_loss']:.4f}) val acc@1 {fb['acc@1']:.4f} acc@20 {fb['acc@20']:.4f} score "
                f"{monitor_score:.4f} lr {h['lr']:.1e} ({h['wall_s']} s)")
        srec["epochs_run_this_session"] = len(state["history"])
    state["sessions"].append(rec.stages[0])
    out = dict(run_key=rk, finished_utc=_now(), best_epoch=state["best_epoch"], best_monitor_score=state["best_score"],
               epochs=hp["epochs"], epochs_run=state["epoch"] + 1, stopped_early=state["stopped_early"],
               seq_stats=sstats, n_val_queries_for_selection=len(val_ids), sessions=state["sessions"],
               prepare=pmeta["cost_stage"], users_subsample=None if users is None else [int(u) for u in users])
    if os.path.exists(last_p):
        ck = torch.load(last_p, map_location="cpu", weights_only=False)
        ck["train_state"] = state
        _torch_save(ck, last_p)
    _json_dump(out, done_p)
    return out


def write_preds(ex: Export, ed: str, wdir: str, variant: str, split: str, device: str, force: bool = False,
                out_root: str = ROOT, emb_root: str = ROOT, log=print) -> str:
    policy, ds = ex.manifest["policy"], ex.manifest["ds"]
    odir = os.path.join(out_root, "preds", f"{method_name(variant)}@{policy}", ds)
    pq, cj = os.path.join(odir, f"{split}.parquet"), os.path.join(odir, f"{split}.cost.json")
    best_p = os.path.join(wdir, variant, "train", "best.pt")
    if not force and os.path.exists(pq) and os.path.exists(cj):
        with open(cj) as f:
            c = json.load(f)
        if c.get("checkpoint_sha256") == _sha256(best_p):
            log(f"[mmpoi] {pq} exists for this checkpoint; skipped")
            return pq
    mname = f"{method_name(variant)}@{policy}"
    with open(os.path.join(wdir, variant, "train", "DONE.json")) as f:
        done = json.load(f)
    with open(os.path.join(embed_dir(ds, emb_root), "embed.json")) as f:
        em = json.load(f)
    qids = ex.queries(split)["query_id"].to_numpy(np.int64).tolist()
    rec = CostRecorder(cj, method=mname, ds=ds, split=split,
                       extra=dict(adapter="methods/mmpoi/run.py", adapter_version=ADAPTER_VERSION, variant=variant,
                                  evidence_policy=policy, topk=TOPK, seed=SEED, upstream_commit=PINNED_COMMIT,
                                  candidates_file=candidates_file_decl(ex, split),
                                  checkpoint_sha256=_sha256(best_p), best_epoch=done["best_epoch"],
                                  best_val_monitor_score=done["best_monitor_score"], epochs=done["epochs"],
                                  epochs_run=done["epochs_run"], stopped_early=done["stopped_early"],
                                  train_sequences=done["seq_stats"], embeddings_sha256=em["embeddings_sha256"],
                                  text_encoder=em["text_encoder"],
                                  captioner=em["captioner"] if variant == "image" else None))
    for st in em["cost_stages"]:
        if variant == "image" or st["name"] == "embed_text_meta_review":
            rec.add_stage(**dict(st, shared_across_runs=True))
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
    rec.add_stage(**tr)
    ngpu = 1 if device.startswith("cuda") else 0
    with rec.stage(f"predict_{split}", one_time=False, n_gpus=ngpu, n_items=len(qids)) as srec:
        model = load_model(best_p, device)
        df = predict(ed, split, qids, ckpt=model)
        srec["n_fallback_popularity"] = int(df.attrs["n_fallback"])
        srec["n_context_rows_not_in_graph"] = int(df.attrs["n_context_rows_dropped"])
    os.makedirs(odir, exist_ok=True)
    df.attrs = {}
    df.to_parquet(pq + ".tmp", index=False)
    os.replace(pq + ".tmp", pq)
    rec.save(n_queries=len(qids))
    log(f"[mmpoi] wrote {pq} ({len(df)} queries, {rec.stages[-1]['n_fallback_popularity']} popularity fallbacks)")
    return pq


def smoke_predict_val(ex: Export, ed: str, wdir: str, variant: str, device: str, users, sub: str = "smoke",
                      log=print) -> dict:
    from poibench.evaluate import val_feedback
    sdir = os.path.join(wdir, variant, sub)
    q = ex.queries("val")
    qids = q.loc[q["user_id"].isin(set(users)), "query_id"].to_numpy(np.int64).tolist()
    df = predict(ed, "val", qids, ckpt=os.path.join(sdir, "best.pt"), device=device)
    df.attrs = {}
    df.to_parquet(os.path.join(sdir, "val_preds.parquet"), index=False)
    fb = val_feedback(ex.manifest["ds"], df, ks=(1, 5, 10, 20), root=root_of_export(ed), query_ids=qids)
    _json_dump(fb, os.path.join(sdir, "val_feedback.json"))
    log(f"[mmpoi] smoke val ({len(qids)} queries of {len(users)} users): {fb}")
    return fb


def main(argv=None):
    ap = argparse.ArgumentParser(description="MMPOI adapter (POIBench exports)")
    ap.add_argument("cmd", nargs="?", default="all", choices=["embed", "prepare", "train", "predict", "all"])
    ap.add_argument("--ds", required=True)
    ap.add_argument("--policy")
    ap.add_argument("--variant", default=os.environ.get("MMPOI_VARIANT", "image"), choices=list(VARIANTS))
    ap.add_argument("--splits", "--split", nargs="+", default=["val", "test"], choices=["val", "test"])
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:N")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke-users", type=float, default=None,
                    help="smoke run: train on this fraction of users (seed 0); outputs under work/ only")
    ap.add_argument("--epochs", type=int, default=None, help="only with --smoke-users (default: upstream 200)")
    ap.add_argument("--canary-prep", action="store_true",
                    help="train the smoke checkpoint predict() loads for the canary (1 epoch, 5%% users, CPU)")
    ap.add_argument("--force", action="store_true", help="redo embed/prepare/predict even if outputs exist")
    a = ap.parse_args(argv)
    if a.seed != SEED:
        ap.error("owner rule: seed 0 only")
    if a.epochs is not None and a.smoke_users is None:
        ap.error("--epochs overrides the upstream default and is allowed only in smoke runs (--smoke-users)")
    device = resolve_device(a.device)
    t0 = time.time()
    if a.cmd == "embed":
        embed(a.ds, device, force=a.force)
        print(f"[mmpoi] embed {a.ds} finished in {time.time() - t0:.0f} s")
        return 0
    if not a.policy:
        ap.error("--policy is required")
    if a.canary_prep:
        device = "cpu" if a.device == "auto" else device
    ed = export_dir(a.ds, a.policy)
    ex = Export(ed)
    wdir = work_dir(a.ds, a.policy)
    if a.canary_prep or a.smoke_users is not None:
        frac = SMOKE_USERS if a.smoke_users is None else a.smoke_users
        ep = SMOKE_EPOCHS if a.epochs is None else a.epochs
        sub = "smoke" if (a.canary_prep or device == "cpu") else "smoke_gpu"
        d = train(ex, ed, wdir, a.variant, device, epochs=ep, user_frac=frac, out_sub=sub)
        if a.canary_prep:
            print(f"[mmpoi] canary-prep done: {os.path.join(wdir, a.variant, 'smoke', 'best.pt')}")
        elif "val" in a.splits:
            smoke_predict_val(ex, ed, wdir, a.variant, device, d["users_subsample"], sub=sub)
        return 0
    if a.cmd in ("prepare", "all"):
        m = prepare(ex, wdir, force=a.force)
        print(f"[mmpoi] prepare: {m['n_nodes']} POI nodes, {m['n_edges']} edges, {m['n_geo_nodes']} geo nodes")
    if a.cmd in ("train", "all"):
        train(ex, ed, wdir, a.variant, device)
    if a.cmd in ("predict", "all"):
        for s in a.splits:
            write_preds(ex, ed, wdir, a.variant, s, device, force=a.force)
    print(f"[mmpoi] {a.cmd} {a.ds}/{a.policy}/{a.variant} finished in {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
