import json, os, sys
C = os.path.join(os.path.dirname(os.path.abspath(sys.argv[1])), "dl_cache")
os.makedirs(os.path.join(C, "tmp"), exist_ok=True)
os.environ["HF_HUB_CACHE"] = os.path.join(C, "dl_hf", "hub"); os.environ["HF_XET_CACHE"] = os.path.join(C, "xet")
os.environ["TMPDIR"] = os.path.join(C, "tmp"); os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
from huggingface_hub import HfApi, snapshot_download
SKIP = ("original/", "consolidated.")

def verify(d, manifest):
    return [f for f, s in manifest.items()
            if not os.path.isfile(os.path.join(d, f)) or (s is not None and os.path.getsize(os.path.join(d, f)) != s)]

dest = sys.argv[1]
for spec in sys.argv[2:]:
    repo, sha = spec.split("@")
    d = os.path.join(dest, repo.split("/")[-1]); mk = os.path.join(d, ".poibench_complete")
    if os.path.isfile(mk):
        print(f"[models] {repo}: complete ({json.load(open(mk))['sha']})", flush=True); continue
    os.makedirs(d, exist_ok=True)
    info = HfApi().model_info(repo, revision=sha, files_metadata=True)
    assert info.sha == sha, (repo, info.sha, sha)
    manifest = {s.rfilename: s.size for s in info.siblings if not s.rfilename.startswith(SKIP)}
    if verify(d, manifest):
        snapshot_download(repo, revision=sha, local_dir=d, max_workers=8, ignore_patterns=["original/*", "consolidated.*"])
    bad = verify(d, manifest); assert not bad, f"{repo}: missing/size mismatch {bad}"
    json.dump(dict(repo=repo, sha=sha, files=manifest, source="huggingface"), open(mk, "w"), indent=1)
    print(f"[models] {repo}: OK {len(manifest)} files {sum(v or 0 for v in manifest.values())/1e9:.2f} GB sha={sha}", flush=True)
