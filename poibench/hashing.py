from __future__ import annotations

import hashlib
import os

DATA_SUFFIXES = (".parquet", ".json")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def tree_sums(top: str, suffixes=DATA_SUFFIXES, exclude_dirs=()) -> str:
    lines = []
    for d, dirs, files in os.walk(top):
        dirs[:] = sorted(x for x in dirs if x not in exclude_dirs and x != "__pycache__")
        for fn in sorted(files):
            if suffixes and not fn.endswith(tuple(suffixes)):
                continue
            p = os.path.join(d, fn)
            rel = os.path.relpath(p, top).replace(os.sep, "/")
            lines.append((rel, sha256_file(p)))
    lines.sort()
    return "".join(f"{h}  {rel}\n" for rel, h in lines)


def tree_fingerprint(top: str, suffixes=DATA_SUFFIXES, exclude_dirs=()) -> str:
    return sha256_text(tree_sums(top, suffixes, exclude_dirs))
