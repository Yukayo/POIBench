#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))

REPOS = {
    "Graph-Flashback": ("third_party/Graph-Flashback", "3489e9a52bf22c36efc54b1eb1e1dabc94d8a596",
                        "graph_flashback.patch"),
    "Graph-Flashback-KGE": ("third_party/Graph-Flashback-KGE", "9fd927301fe9f2d644716f8d2e89b1d4887e21aa",
                            "kge.patch"),
}
BUILD_VERSION = "b2"
ROOT_ONLY_SKIP = {"data", "results"}  # canary-allow: names of clone subdirectories skipped when copying


def _ignore_for(top: str):
    def ignore(d, names):
        skip = {n for n in names if n in (".git", "__pycache__") or n.endswith(".pyc")}
        if os.path.abspath(d) == os.path.abspath(top):
            skip |= {n for n in names if n in ROOT_ONLY_SKIP}
        return skip
    return ignore


def _sha(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def src_key() -> str:
    h = hashlib.sha256(BUILD_VERSION.encode())
    for name, (_, commit, patch) in sorted(REPOS.items()):
        h.update(f"{name}:{commit}:{_sha(os.path.join(HERE, 'patches', patch))}\n".encode())
    return h.hexdigest()[:12]


def clone_commit(clone: str) -> str | None:
    try:
        out = subprocess.run(["git", "-c", f"safe.directory={clone}", "-C", clone, "rev-parse", "HEAD"],
                             capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except Exception:
        return None


def build_src(root: str = ROOT, verbose: bool = True) -> dict:
    base = os.path.join(root, "work", "graph_flashback", "src", src_key())
    done = os.path.join(base, "DONE.json")
    paths = {name: os.path.join(base, name) for name in REPOS}
    if os.path.exists(done):
        return paths
    tmp = f"{base}.tmp_{os.getpid()}"
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    os.makedirs(tmp)
    info = {}
    for name, (rel, commit, patch) in REPOS.items():
        clone = os.path.join(root, rel)
        head = clone_commit(clone)
        if head is not None and head != commit:
            raise RuntimeError(f"{rel} is at {head}, expected the pinned commit {commit}")
        if head is None and verbose:
            print(f"[apply_patches] WARNING: cannot verify the commit of {rel} (no git / no .git)", file=sys.stderr)
        dst = os.path.join(tmp, name)
        shutil.copytree(clone, dst, ignore=_ignore_for(clone))
        pf = os.path.join(HERE, "patches", patch)
        r = subprocess.run(["patch", "-p1", "--forward", "--batch", "--no-backup-if-mismatch", "-d", dst, "-i", pf],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"patch {patch} failed on {rel}:\n{r.stdout}\n{r.stderr}")
        info[name] = dict(clone=rel, commit=commit, verified_commit=head == commit, patch=f"methods/graph_flashback/"
                          f"patches/{patch}", patch_sha256=_sha(pf), patch_output=r.stdout.strip())
    with open(os.path.join(tmp, "DONE.json"), "w") as f:
        json.dump(info, f, indent=1)
    try:
        os.replace(tmp, base)
    except OSError:
        if not os.path.exists(done):
            raise
        shutil.rmtree(tmp, ignore_errors=True)
    if verbose:
        print(f"[apply_patches] patched copy ready: {os.path.relpath(base, root)}")
    return paths


def main(argv=None):
    ap = argparse.ArgumentParser(description=(__doc__ or " ").splitlines()[0])
    ap.add_argument("--check", action="store_true", help="only print the key and whether the copy exists")
    a = ap.parse_args(argv)
    base = os.path.join(ROOT, "work", "graph_flashback", "src", src_key())
    if a.check:
        print(json.dumps(dict(key=src_key(), path=os.path.relpath(base, ROOT),
                              built=os.path.exists(os.path.join(base, "DONE.json")))))
        return 0
    for k, v in build_src().items():
        print(k, os.path.relpath(v, ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
