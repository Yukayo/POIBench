# Third-party notices

This repository contains no copy of any upstream method repository. `scripts/fetch_upstream.sh` clones each
repository below at the pinned commit into `third_party/`; the adapters import, copy-and-patch (in `work/`) or
read those clones at run time. Patch files under `methods/*/patches/` contain short context lines of the
upstream files they modify.

| upstream | commit | licence (as found in the repository) | use |
|---|---|---|---|
| [GETNext](https://github.com/songyangco/GETNext) | `6a79081aec69` | none found | imported by path (sha256-checked); never modified |
| [Graph-Flashback](https://github.com/kevin-xuan/Graph-Flashback) | `3489e9a52bf2` | none found | copied to work/ and patched there (methods/graph_flashback/patches) |
| [Graph-Flashback-KGE](https://github.com/kevin-xuan/KGE) | `9fd927301fe9` | none found | copied to work/ and patched there (methods/graph_flashback/patches) |
| [DS-LLM4POI](https://github.com/ryonsd/DS-LLM4POI) | `f31ab6ffa3c7` | Apache-2.0 (with an MIT-licensed part, see its third_party_licenses/) | copied to work/ and patched there (methods/ds_llm4poi/patches) |
| [LLM4POI](https://github.com/neolifer/LLM4POI) | `83a44f4eece0` | Apache-2.0 | prompt templates and configuration files read at run time |
| [GNPR-SID](https://github.com/wds1996/GNPR-SID) | `641088f08b6f` | none found | imported by path (sha256-checked); never modified |
| [GA-LLM](https://github.com/liuzh368/GA-LLM) | `360f5c3e4d70` | Apache-2.0 | copied to work/ and patched there by methods/gallm/run.py |
| [MMPOI](https://github.com/zzmylq/MMPOI) | `32f9c76e8768` | none found | imported by path (sha256-checked); never modified |
| [LLM-Augmented-POIRec](https://github.com/wzehui/LLM-Augmented-POIRec) | `c7dac9daf9dc` | none found | caption prompt read at run time (VLM re-ranker, arm B) |

Apache-2.0 upstreams (DS-LLM4POI, LLM4POI, GA-LLM): the patch files `methods/ds_llm4poi/patches/*.patch`, the
patch strings in `methods/gallm/run.py` and `methods/gallm/src_add/poibench_vec_geo.py` modify or extend files
of these projects and are provided under the Apache License 2.0 of the respective upstream project; the
upstream LICENSE files are copied next to every patched working copy (`work/`) by the adapters.

Repositories marked "none found" carried no licence file at the pinned commit. They are therefore never
redistributed here; only patch files, short re-statements of prompt templates needed for comparability, and
fetch scripts are included.

Model weights (downloaded separately from the Hugging Face Hub at the revisions pinned in the adapters) and
datasets are subject to their own licences and terms of use; see README.md.
