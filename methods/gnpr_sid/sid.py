from __future__ import annotations
import os as _pb_os
def _pb_root():
    p = _pb_os.path.dirname(_pb_os.path.abspath(__file__))
    while not _pb_os.path.isfile(_pb_os.path.join(p, "poibench", "__init__.py")) and _pb_os.path.dirname(p) != p:
        p = _pb_os.path.dirname(p)
    return p
_PB_ROOT = _pb_os.environ.get("POIBENCH_ROOT") or _pb_root()
_PB_MODELS = _pb_os.environ.get("POIBENCH_MODEL_DIR") or _pb_os.path.join(_PB_ROOT, "models")

import json
import os
import pickle
import random
import sys
from collections import Counter

import numpy as np
import pandas as pd

from methods.gnpr_sid import upstream as U

MINILM_DIRNAME = "all-MiniLM-L6-v2"
CATEGORY_PCA_DIM = 64


def model_dir() -> str:
    return os.environ.get("POIBENCH_MODEL_DIR", f"{_PB_MODELS}")


def seed_all(seed: int = 0) -> None:
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sid_args(data_path: str, ckpt_dir: str, device: str, epochs: int | None):
    _, _, _, train_SID = U.sid_modules()
    argv = ["train_SID.py", "--data_path", data_path, "--ckpt_dir", ckpt_dir, "--device", device]
    if epochs is not None:
        argv += ["--epochs", str(int(epochs))]
    old = sys.argv
    try:
        sys.argv = argv
        return train_SID.parse_args("")
    finally:
        sys.argv = old


def build_model(args, dim: int):
    _, crqvae, _, _ = U.sid_modules()
    return crqvae.CRQVAE(in_dim=dim, num_emb_list=args.num_emb_list, e_dim=args.e_dim, layers=args.layers,
                         dropout_prob=args.dropout_prob, bn=args.bn, loss_type=args.loss_type,
                         quant_loss_weight=args.quant_loss_weight, beta=args.beta, kmeans_init=args.kmeans_init,
                         kmeans_iters=args.kmeans_iters, sk_epsilons=args.sk_epsilons, sk_iters=args.sk_iters,
                         use_linear=args.use_linear)


def run_sid(poi_csv: str, out_dir: str, device: str, epochs: int | None = None, seed: int = 0, log=print) -> dict:
    import torch
    from torch.utils.data import DataLoader
    os.makedirs(out_dir, exist_ok=True)
    done = os.path.join(out_dir, "SID_DONE.json")
    if os.path.exists(done):
        with open(done) as f:
            meta = json.load(f)
        sids = {int(k): v for k, v in meta["sids"].items()}
        log(f"SID: reusing {out_dir} ({len(sids)} POIs)")
        return sids
    seed_all(seed)
    emb_dir = os.path.join(out_dir, "emb")
    cat_pkl = os.path.join(emb_dir, "category_to_embedding.pkl")
    if not os.path.exists(cat_pkl):
        c2v = U.category2vec_module()
        c2v.category2vec(csv_path=poi_csv, output_dir=emb_dir, model_name=os.path.join(model_dir(), MINILM_DIRNAME),
                         n_components=CATEGORY_PCA_DIM, category_column="category")
    poi_pkl = os.path.join(emb_dir, "poi_Emb_dict.pkl")
    if not os.path.exists(poi_pkl):
        U.poi2emb_module().main(poi_csv, cat_pkl, None, emb_dir)
    POIdatasets, _, SID_trainer, _ = U.sid_modules()
    ckpt_root = os.path.join(out_dir, "ckpt")
    args = sid_args(poi_pkl, ckpt_root, device, epochs)
    seed_all(seed)
    data = POIdatasets.EmbDataset(args.data_path)
    best = None
    if os.path.isdir(ckpt_root):
        subs = sorted(d for d in os.listdir(ckpt_root) if os.path.exists(os.path.join(ckpt_root, d, "FIT_DONE")))
        if subs:
            best = os.path.join(ckpt_root, subs[-1], "best_loss_model.pth")
    fit = {}
    if best is None:
        model = build_model(args, data.dim)
        loader = DataLoader(data, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True,
                            pin_memory=True)
        trainer = SID_trainer.Trainer(args, model, len(loader))
        best_loss, best_coll = trainer.fit(loader)
        with open(os.path.join(trainer.ckpt_dir, "FIT_DONE"), "w") as f:
            f.write(json.dumps(dict(best_loss=float(best_loss), best_collision_rate=float(best_coll))))
        best = os.path.join(trainer.ckpt_dir, "best_loss_model.pth")
        fit = dict(best_loss=float(best_loss), best_collision_rate=float(best_coll))
        del trainer, model
    seed_all(seed)
    checkpoint = torch.load(best, map_location=args.device, weights_only=False)
    model = build_model(args, data.dim)
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(args.device)
    model.eval()
    loader = DataLoader(data, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    SIDs = {}
    for pids, x in loader:
        pids = pids.tolist()
        _, indices = model.get_indices(x.to(args.device))
        for i, poi in enumerate(pids):
            SIDs[int(poi)] = indices[i].tolist()
    value_counts = Counter(tuple(v) for v in SIDs.values())
    seen_values, updated = {}, {}
    for key in sorted(SIDs.keys()):
        value = SIDs[key]
        vt = tuple(value)
        if value_counts[vt] > 1:
            if vt not in seen_values:
                seen_values[vt] = 0
            else:
                seen_values[vt] += 1
            updated[key] = value + [seen_values[vt]]
        else:
            updated[key] = value
    pd.DataFrame({"pid": list(updated), "sid": [str(v) for v in updated.values()]}).to_csv(
        os.path.join(out_dir, "sid.csv"), index=False)
    n3 = len(SIDs)
    n_coll_groups = sum(1 for c in value_counts.values() if c > 1)
    stats = dict(n_pois=n3, n_unique_3level=len(value_counts), collision_rate_3level=(n3 - len(value_counts)) / n3,
                 n_collision_groups=n_coll_groups, max_group=max(value_counts.values()),
                 n_with_4th_level=sum(len(v) == 4 for v in updated.values()),
                 args={k: (v if isinstance(v, (int, float, str, bool, list, type(None))) else str(v))
                       for k, v in vars(args).items()},
                 checkpoint=os.path.relpath(best, out_dir), fit=fit, emb_dim=int(data.dim))
    if len(set(tuple(v) for v in updated.values())) != len(updated):
        raise RuntimeError("SID assignment is not unique after the collision suffix")
    with open(done + ".tmp", "w") as f:
        json.dump(dict(stats=stats, sids={str(k): v for k, v in updated.items()}), f)
    os.replace(done + ".tmp", done)
    log(f"SID: {stats['n_pois']} POIs, 3-level collision rate {stats['collision_rate_3level']:.4f}, "
        f"{stats['n_with_4th_level']} with a 4th level")
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return updated


def load_sids(out_dir: str) -> tuple[dict, dict]:
    with open(os.path.join(out_dir, "SID_DONE.json")) as f:
        meta = json.load(f)
    return {int(k): v for k, v in meta["sids"].items()}, meta["stats"]
