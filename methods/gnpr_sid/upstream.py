from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
CLONE = os.path.join(ROOT, "third_party", "GNPR-SID")
COMMIT = "641088f08b6f5750ae87bfcee850e152161121df"
UPSTREAM_SHA256 = {
    "V2/POIembedding/category2vec.py": "da6c27060faab9e3abdb982953b756f44e5f9209640da44782c5eb60bce9ba11",
    "V2/POIembedding/POI2emb.py": "d0d30c26d774a0751de2e02d3f8f1d310ec7dda235186600386af87442e32ab7",
    "V2/SID/POIdatasets.py": "19ae2e0594c59074e80d6021e86dc4f6bdc9f6b85b261aae67d258ec81324a5b",
    "V2/SID/SID_trainer.py": "b2477151aadca888d10b03e0c77ae4174bdd230cc554736c678765287c42310c",
    "V2/SID/train_SID.py": "7a53cad4ec2820309f3be7666b0fe20583ebe8351410468bbd6914104c216d98",
    "V2/SID/get_SID.py": "a0adc4bb337ca23fd54c9d5f6af583e0d2081892bb4a6d64ee02a835cbb94f2c",
    "V2/SID/CRQVAE/crqvae.py": "3f0c2c1f3d86ab07ab7790efbc08fed001aacf49b1cbf5d254e473960b24d60f",
    "V2/SID/CRQVAE/cvq.py": "d31fcc7ade8cf9e0b200cdbcb6d23395cdaa426826663e88000cf2594106b082",
    "V2/SID/CRQVAE/mlp.py": "04931666004d398d62aa79089366ed1a4e051c0e678497060b733fc8c99f070c",
    "V2/SID/CRQVAE/rq.py": "728079c01619326b5ed357ffe2a8733d6c0e13d399e7e362560f0a4aa3fee2ed",
    "V2/LLM/train/sft_without_alignment.py": "aad417994f093bf15aa3e957e4cc3dbb0b38a555410e0846aef74471a742c2c6",
}

SFT_EDITS = [
    ("import wandb\n", ""),
    ('    wandb_run_name="sid_sft",\n):\n',
     '    wandb_run_name="sid_sft",\n    poibench_hooks=None,\n):\n'),
    ('    os.environ["WANDB_PROJECT"] = wandb_project\n',
     '    poibench_hooks = dict(poibench_hooks or {})\n'),
    ('        eval_strategy="steps",\n        eval_steps=50,\n',
     '        eval_strategy="no",\n'),
    ('        report_to="wandb",\n',
     '        report_to="none",\n        seed=seed,\n'
     '        max_steps=poibench_hooks.get("max_steps", -1),\n'
     '        save_total_limit=poibench_hooks.get("save_total_limit"),\n'),
    ('    trainer.train()\n',
     '    if poibench_hooks.get("before_train") is not None:\n'
     '        poibench_hooks["before_train"](trainer)\n'
     '    trainer.train(resume_from_checkpoint=poibench_hooks.get("resume_from_checkpoint"))\n'),
]


def _sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def verify_clone(clone: str = CLONE) -> None:
    bad = []
    for rel, want in UPSTREAM_SHA256.items():
        p = os.path.join(clone, rel)
        got = _sha(p) if os.path.exists(p) else "missing"
        if got != want:
            bad.append(f"{rel}: {got[:12]} != {want[:12]}")
    if bad:
        raise RuntimeError(f"{clone} is not the pinned GNPR-SID {COMMIT[:12]}: " + "; ".join(bad))


DEFAULT_BACKBONE = "llama3-8b"
BACKBONE_EDITS = [
    ('def format_fn(batch):\n', '_POIBENCH_EOS = "<|eot_id|>"   # poibench: set to the backbone tokenizer.eos_token\n\n'
                                'def format_fn(batch):\n'),
    ('            f"### Response:\\n{resp.strip()}<|eot_id|>"\n',
     '            f"### Response:\\n{resp.strip()}{_POIBENCH_EOS}"\n'),
]


def backbone() -> str:
    return os.environ.get("GNPR_SID_BACKBONE", DEFAULT_BACKBONE)


def sft_edits() -> list:
    return SFT_EDITS + (BACKBONE_EDITS if backbone() != DEFAULT_BACKBONE else [])


def edits_hash() -> str:
    return hashlib.sha256(json.dumps(sft_edits()).encode()).hexdigest()


def info() -> dict:
    return dict(upstream="wds1996/GNPR-SID", commit=COMMIT, files_sha256=UPSTREAM_SHA256,
                sft_edits_sha256=edits_hash(), n_sft_edits=len(sft_edits()), backbone=backbone())


def _load(name: str, path: str):
    sp = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(sp)
    sys.modules[name] = mod
    sp.loader.exec_module(mod)
    return mod


@contextlib.contextmanager
def _keep_env(var: str):
    had, old = var in os.environ, os.environ.get(var)
    try:
        yield
    finally:
        if had:
            os.environ[var] = old
        else:
            os.environ.pop(var, None)


_MODS = {}


def category2vec_module():
    if "c2v" not in _MODS:
        verify_clone()
        with _keep_env("CUDA_VISIBLE_DEVICES"):
            _MODS["c2v"] = _load("gnpr_category2vec", os.path.join(CLONE, "V2/POIembedding/category2vec.py"))
    return _MODS["c2v"]


def poi2emb_module():
    if "p2e" not in _MODS:
        verify_clone()
        _MODS["p2e"] = _load("gnpr_POI2emb", os.path.join(CLONE, "V2/POIembedding/POI2emb.py"))
    return _MODS["p2e"]


def sid_modules():
    if "sid" not in _MODS:
        verify_clone()
        d = os.path.join(CLONE, "V2", "SID")
        if d not in sys.path:
            sys.path.insert(0, d)
        import importlib
        mods = tuple(importlib.import_module(m) for m in ("POIdatasets", "CRQVAE.crqvae", "SID_trainer", "train_SID"))
        for m in mods:
            if not os.path.abspath(m.__file__).startswith(d):
                raise RuntimeError(f"{m.__name__} imported from {m.__file__}, not from the pinned clone")
        _MODS["sid"] = mods
    return _MODS["sid"]


def edited_sft_text(clone: str = CLONE) -> str:
    verify_clone(clone)
    with open(os.path.join(clone, "V2/LLM/train/sft_without_alignment.py")) as f:
        text = f.read()
    for old, new in sft_edits():
        n = text.count(old)
        if n != 1:
            raise RuntimeError(f"SFT edit does not match exactly once ({n}): {old!r}")
        text = text.replace(old, new)
    return text


def sft_module(root: str = ROOT):
    if "sft" in _MODS:
        return _MODS["sft"]
    text = edited_sft_text()
    d = os.path.join(root, "work", "gnpr_sid", "upstream", f"{COMMIT[:12]}-{edits_hash()[:12]}")
    dst = os.path.join(d, "sft_without_alignment.py")
    if not os.path.exists(dst):
        os.makedirs(d, exist_ok=True)
        tmp = f"{dst}.tmp_{os.getpid()}"
        with open(tmp, "w") as f:
            f.write(text)
        os.replace(tmp, dst)
    with open(dst) as f:
        if f.read() != text:
            raise RuntimeError(f"{dst} differs from the edited upstream text")
    _MODS["sft"] = _load("gnpr_sft_without_alignment", dst)
    return _MODS["sft"]


if __name__ == "__main__":
    verify_clone()
    print(json.dumps(info(), indent=1))
