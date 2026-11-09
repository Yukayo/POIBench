#!/usr/bin/env python
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
CLONE = os.path.join(ROOT, "third_party", "DS-LLM4POI")
PATCH_DIR = os.path.join(HERE, "patches")
COMMIT = "f31ab6ffa3c76ae95d2a086b6eae38005a657a00"
UPSTREAM_SHA256 = {
    "src/utils/prompt_utils.py": "72f1592e068e4ebbdb3128df5911ba10374b2f72ff69b60f9a574647f793e00c",
    "src/utils/data_utils.py": "4264a617866db19fcd59a792ace7f211462e1b726ffd28bb6999300053366647",
    "src/utils/demo_selection.py": "cf020fca044e8af3c9ccaa49fbd6226ea1918c57bc6b122c0a37600e38b27411",
    "src/utils/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "src/calc_traj_similarity.py": "f48e8d1c2dc2501495adb48d8bd161a5ee9455b0b543addb38fb120c6139f3b2",
    "src/run.py": "51121596d91e0ae0ba6fe794e9013217598fe9293513c54e5237c2f8fab378f4",
    "LICENSE": "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4",
    "third_party_licenses/LICENSE": "fe1225d067fc0d8fb2b8944ae7bba7c5536ed536c17d2ea26a387aa2dd4aa892",
}


def _sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def patch_files() -> list:
    return sorted(os.path.join(PATCH_DIR, f) for f in os.listdir(PATCH_DIR) if f.endswith(".patch"))


def patchset_hash() -> str:
    h = hashlib.sha256()
    for p in patch_files():
        h.update(os.path.basename(p).encode() + b"\0" + _sha(p).encode() + b"\n")
    return h.hexdigest()


def target_dir(root: str = ROOT) -> str:
    return os.path.join(root, "work", "ds_llm4poi", "upstream", f"{COMMIT[:12]}-{patchset_hash()[:12]}")


def verify_clone(clone: str = CLONE) -> None:
    bad = []
    for rel, want in UPSTREAM_SHA256.items():
        p = os.path.join(clone, rel)
        got = _sha(p) if os.path.exists(p) else "missing"
        if got != want:
            bad.append(f"{rel}: {got[:12]} != {want[:12]}")
    if bad:
        raise RuntimeError(f"{clone} is not the pinned DS-LLM4POI {COMMIT[:12]}: " + "; ".join(bad))


def _apply(patch: str, d: str) -> None:
    if shutil.which("patch"):
        subprocess.run(["patch", "-p1", "--forward", "--batch", "--silent", "-d", d, "-i", patch], check=True)
    elif shutil.which("git"):
        subprocess.run(["git", "apply", "-p1", "--whitespace=nowarn", patch], check=True, cwd=d)
    else:
        raise RuntimeError("neither `patch` nor `git` is available to apply " + patch)


def ensure_patched(root: str = ROOT, clone: str = CLONE) -> str:
    dst = target_dir(root)
    if os.path.exists(os.path.join(dst, "PATCHED.json")):
        return dst
    verify_clone(clone)
    tmp = f"{dst}.tmp_{os.getpid()}"
    os.makedirs(tmp)
    for rel in UPSTREAM_SHA256:
        os.makedirs(os.path.dirname(os.path.join(tmp, rel)), exist_ok=True)
        shutil.copyfile(os.path.join(clone, rel), os.path.join(tmp, rel))
    for p in patch_files():
        _apply(p, tmp)
    info = dict(upstream="ryonsd/DS-LLM4POI", commit=COMMIT, upstream_sha256=UPSTREAM_SHA256,
                patches={os.path.basename(p): _sha(p) for p in patch_files()},
                patched_sha256={rel: _sha(os.path.join(tmp, rel)) for rel in UPSTREAM_SHA256})
    with open(os.path.join(tmp, "PATCHED.json"), "w") as f:
        json.dump(info, f, indent=1)
    try:
        os.rename(tmp, dst)
    except OSError:
        if not os.path.exists(os.path.join(dst, "PATCHED.json")):
            raise
        shutil.rmtree(tmp, ignore_errors=True)
    return dst


if __name__ == "__main__":
    print(ensure_patched())
    sys.exit(0)
