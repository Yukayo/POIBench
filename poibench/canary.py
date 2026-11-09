from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import subprocess
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import ROOT
from .data import SPLITS, Arrays, policy_vis_from
from .exports import (Export, build_export, export_dir, export_is_current, load_sources, write_export)

CHECK_COLS = ("ranked_venue_ids", "cand_ranked_venue_ids")
INVARIANT_FILES = ("visible_rows", "venues_public", "train_labels", "query_inputs_train", "candidates_train")

LINT_PATTERNS = [
    (r"\bcheckins\.parquet\b", "reads data/<ds>/checkins.parquet (all val/test rows and targets)"),
    (r"(?<!query_inputs_)\bqueries_(train|val|test)\b", "reads data/<ds>/queries_<split>.parquet (targets)"),
    (r"\b(load_checkins|load_queries|load_venues|load_candidates|load_policy|query_evidence|evidence_slices|"
     r"policy_vis_from|load_split|load_sources)\b", "uses a poibench loader of data/<ds>/"),
    (r"\bpoibench\.data\b|from\s+poibench\s+import\s+[^\n#]*\bdata\b", "imports poibench.data"),
    (r"\b(target_venue_id|target_t_local|target_row_id|in_user_history|user_hist_len|n_train_visits|"
     r"user_visible_from_row_id)\b", "target / hidden-information column of data/<ds>/"),
    (r"""["']data["']\s*[,)]|["']data/""", "builds a path into ROOT/data"),
    (r"\bid_map_(users|venues)\b", "raw-id maps are not adapter inputs"),
    (r"[/\"']evidence[/\"']", "reads data/<ds>/evidence/ (per-user visibility bounds)"),
]
ALLOW_TAG = "canary-allow:"


def lint_sources(paths, exclude_dirs=("patches", "__pycache__")) -> list:
    files = []
    for p in paths:
        if os.path.isfile(p):
            files.append(p)
            continue
        for d, dirs, fs in os.walk(p):
            dirs[:] = sorted(x for x in dirs if x not in exclude_dirs)
            files += [os.path.join(d, f) for f in sorted(fs) if f.endswith((".py", ".sh"))]
    hits = []
    for f in files:
        with open(f, encoding="utf-8", errors="replace") as fh:
            for i, line in enumerate(fh, 1):
                if ALLOW_TAG in line:
                    continue
                for pat, why in LINT_PATTERNS:
                    if re.search(pat, line):
                        hits.append((os.path.relpath(f, ROOT), i, why, line.strip()[:160]))
                        break
    return hits


def load_entry(spec: str, root: str = ROOT):
    if spec.startswith("cmd:"):
        template = spec[4:]

        def run_cmd(export_dir, split, query_ids, work_dir):
            os.makedirs(work_dir, exist_ok=True)
            qf = os.path.join(work_dir, "query_ids.txt")
            with open(qf, "w") as f:
                f.write("\n".join(str(int(x)) for x in query_ids) + "\n")
            out = os.path.join(work_dir, "preds.parquet")
            cmd = template.format(export_dir=export_dir, split=split, query_ids=qf, out=out, work_dir=work_dir)
            subprocess.run(cmd, shell=True, check=True, cwd=root)
            return pd.read_parquet(out)
        return run_cmd
    path, func = spec.rsplit(":", 1)
    if not os.path.isabs(path):
        path = os.path.join(root, path)
    name = "canary_entry_" + re.sub(r"\W", "_", os.path.relpath(path, root))
    sp = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(mod)
    return getattr(mod, func)


def entry_lint_paths(spec: str, root: str = ROOT) -> list:
    if spec.startswith("cmd:") or spec.startswith("builtin:"):
        return []
    path = spec.rsplit(":", 1)[0]
    path = path if os.path.isabs(path) else os.path.join(root, path)
    rel = os.path.relpath(path, os.path.join(root, "methods"))
    if rel.startswith(".."):
        return [path]
    return [os.path.join(root, "methods", rel.split(os.sep)[0])]


def _builtin_prev_then_pop(export_dir, split, query_ids, work_dir):
    ex = Export(export_dir)
    q = ex.queries(split).set_index("query_id").loc[list(query_ids)]
    ven = ex.venues
    pop = ven["venue_id"].to_numpy()[np.lexsort((ven["venue_id"].to_numpy(), -ven["n_visible_train_visits"].to_numpy()))]
    lists = []
    for pv in q["prev_venue_id"].to_numpy():
        lists.append([int(pv)] + [int(v) for v in pop[:60] if v != pv][:49])
    return pd.DataFrame({"query_id": list(query_ids), "ranked_venue_ids": lists})


def _builtin_leaky_sibling(export_dir, split, query_ids, work_dir):
    ex = Export(export_dir)
    ctx = ex.context(split)
    by_row = ctx.drop_duplicates("row_id").set_index("row_id")["venue_id"]
    base = _builtin_prev_then_pop(export_dir, split, query_ids, work_dir)
    out = []
    for qid, lst in zip(base["query_id"], base["ranked_venue_ids"]):
        t = by_row.get(int(qid))
        out.append(([int(t)] + [v for v in lst if v != t])[:50] if t is not None else lst)
    return base.assign(ranked_venue_ids=out)


BUILTINS = {"builtin:prev_then_pop": _builtin_prev_then_pop, "builtin:leaky_sibling": _builtin_leaky_sibling}


def choose_queries(queries: pd.DataFrame, n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    q = queries.sort_values("query_id")
    picks = []
    for _, g in q.groupby("traj_id", sort=True):
        ids = g["query_id"].to_numpy(np.int64)
        picks.append(ids[rng.integers(0, len(ids) - 1)] if len(ids) >= 2 else ids[0])
    picks = np.array(picks, dtype=np.int64)
    if len(picks) > n:
        picks = rng.choice(picks, size=n, replace=False)
    return np.sort(picks)


def perturb_sources(ck: pd.DataFrame, venues: pd.DataFrame, queries: dict, policy: dict, S: np.ndarray,
                    seed: int = 0) -> tuple:
    rng = np.random.default_rng(seed + 1)
    ck = ck.sort_values("row_id").reset_index(drop=True)
    A = Arrays.from_checkins(ck)
    rows = np.arange(len(A.user), dtype=np.int64)
    if policy["kind"] == "H":
        vf = policy_vis_from(policy, len(A.user_first))
        hidden = A.is_train & (rows < vf[A.user])
    else:
        hidden = np.zeros(len(rows), dtype=bool)
    future = ~A.is_train
    protect = np.zeros(len(rows), dtype=bool)
    for q in S:
        protect[A.traj_start[q]:q] = True
    mut = (hidden | future) & ~protect
    idx = np.flatnonzero(mut)
    vt = venues.sort_values("venue_id").reset_index(drop=True)
    catalog = vt["venue_id"].to_numpy(np.int64)[vt["n_train_visits"].to_numpy() > 0]
    old = A.venue[idx]
    new = catalog[rng.integers(0, len(catalog), size=len(idx))]
    same = new == old
    while same.any():
        new[same] = catalog[rng.integers(0, len(catalog), size=int(same.sum()))]
        same = new == old
    ck2 = ck.copy()
    ck2.loc[idx, "venue_id"] = new.astype(ck["venue_id"].dtype)
    for c in ("category", "lat", "lon"):
        ck2.loc[idx, c] = vt[c].to_numpy()[new]
    ntraj = int(A.traj.max()) + 1
    shift_h = rng.integers(1, 13, size=ntraj)
    sign = np.where(hidden[idx], -1, 1)
    delta = pd.to_timedelta(sign * shift_h[A.traj[idx]], unit="h")
    ck2.loc[idx, "t_utc"] = ck["t_utc"].to_numpy()[idx] + delta.to_numpy()
    ck2.loc[idx, "t_local"] = ck["t_local"].to_numpy()[idx] + delta.to_numpy()
    q2 = {}
    v2 = ck2["venue_id"].to_numpy()
    for s, q in queries.items():
        q = q.copy()
        q["target_venue_id"] = v2[q["target_row_id"].to_numpy(np.int64)].astype(q["target_venue_id"].dtype)
        q2[s] = q
    info = dict(n_rows_replaced=int(len(idx)), n_hidden_replaced=int((hidden & mut).sum()),
                n_future_replaced=int((future & mut).sum()), n_protected_rows=int(protect.sum()))
    return ck2, q2, info


def _frames_equal(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    if list(a.columns) != list(b.columns) or len(a) != len(b):
        return False
    for c in a.columns:
        x, y = a[c].to_numpy(), b[c].to_numpy()
        if x.dtype == object:
            if not all(np.array_equal(np.asarray(u), np.asarray(v)) if hasattr(u, "__len__") and not isinstance(u, str)
                       else u == v for u, v in zip(x, y)):
                return False
        elif not np.array_equal(x, y):
            return False
    return True


def _norm_list(v) -> list:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return []
    return [int(x) for x in v]


def run_canary(entry: str, ds: str, policy: str, split: str = "val", n: int = 300, seed: int = 0,
               root: str = ROOT, keep: bool = False, lint_paths: list | None = None, tag: str | None = None,
               write_report: bool = True, expect_config: dict | None = None) -> dict:
    if split not in ("val", "test"):
        raise ValueError("the canary runs on val or test queries (train targets are allowed inputs)")
    t0 = time.time()
    exp = export_dir(ds, policy, root)
    if not export_is_current(ds, policy, root):
        raise RuntimeError(f"{exp} missing or stale: run scripts/export_inputs.py --datasets {ds} --policies {policy}")
    fn = BUILTINS[entry] if entry in BUILTINS else load_entry(entry, root)
    tag = tag or re.sub(r"[^\w.-]+", "_", entry.split(":")[0] if not entry.startswith(("cmd:", "builtin:"))
                        else entry)[-60:]
    work = os.path.join(root, "work", "canary",
                        f"{tag}__{ds}__{policy}__{split}__{datetime.now(timezone.utc):%Y%m%dT%H%M%S}_{os.getpid()}")
    os.makedirs(work)

    ck, venues, queries, pol, cands, meta = load_sources(ds, policy, root)
    S = choose_queries(queries[split], n, seed)
    ck2, q2, pinfo = perturb_sources(ck, venues, queries, pol, S, seed)
    pexp = os.path.join(work, "export_perturbed")
    write_export(build_export(ck2, venues, q2, pol, cands), pexp,
                 dict(ds=ds, policy=policy, perturbed_for_canary=True, n_protected_queries=int(len(S))))

    A0, A1 = Export(exp), Export(pexp)
    inv = {}
    for name in INVARIANT_FILES + (f"candidates_{split}",):
        inv[name] = _frames_equal(A0._pq(name), A1._pq(name))
    sset = set(int(x) for x in S)
    qa, qb = A0.queries(split), A1.queries(split)
    inv[f"query_inputs_{split}[S]"] = _frames_equal(qa[qa.query_id.isin(sset)].reset_index(drop=True),
                                                     qb[qb.query_id.isin(sset)].reset_index(drop=True))
    ca, cb = A0.context(split), A1.context(split)
    inv[f"context_{split}[S]"] = _frames_equal(ca[ca.query_id.isin(sset)].reset_index(drop=True),
                                               cb[cb.query_id.isin(sset)].reset_index(drop=True))
    exposed = int(ca[(ca.row_id.isin(sset))]["row_id"].nunique())
    exposed_after = int((cb[cb.row_id.isin(sset)].merge(ca[ca.row_id.isin(sset)], on=["query_id", "row_id"],
                                                         suffixes=("_b", "_a"))
                         .eval("venue_id_b != venue_id_a")).sum())

    qids = [int(x) for x in S]
    t1 = time.time()
    pa = fn(export_dir=exp, split=split, query_ids=qids, work_dir=os.path.join(work, "run_real"))
    t2 = time.time()
    pb = fn(export_dir=pexp, split=split, query_ids=qids, work_dir=os.path.join(work, "run_perturbed"))
    t3 = time.time()
    cfg_a, cfg_b = pa.attrs.get("adapter_config"), pb.attrs.get("adapter_config")
    config_check = "not requested"
    if expect_config is not None:
        if cfg_a is None or cfg_b is None:
            config_check = "FAIL: the adapter does not report its configuration (attrs['adapter_config'])"
        elif cfg_a != cfg_b:
            config_check = f"FAIL: the two runs used different configurations {cfg_a} / {cfg_b}"
        else:
            bad = {k: [cfg_a.get(k), v] for k, v in expect_config.items() if cfg_a.get(k) != v}
            config_check = f"FAIL: used {bad} (actual, expected)" if bad else "PASS"
    ma, mb = pa.set_index("query_id"), pb.set_index("query_id")
    missing = sorted(sset - set(int(x) for x in ma.index)) + sorted(sset - set(int(x) for x in mb.index))
    cols = [c for c in CHECK_COLS if c in ma.columns or c in mb.columns]
    mism = {c: [] for c in cols}
    for c in cols:
        if c not in ma.columns or c not in mb.columns:
            mism[c] = ["column present in only one run"]
            continue
        for qid in qids:
            if qid in ma.index and qid in mb.index and _norm_list(ma.at[qid, c]) != _norm_list(mb.at[qid, c]):
                mism[c].append(qid)
    lint = lint_sources(lint_paths if lint_paths is not None else entry_lint_paths(entry, root))
    ok_inv = all(inv.values())
    ok_pred = not missing and all(len(v) == 0 for v in mism.values())
    ok_cfg = config_check in ("PASS", "not requested")
    rep = dict(
        result="PASS" if (ok_inv and ok_pred and not lint and ok_cfg) else "FAIL",
        entry=entry, ds=ds, policy=policy, split=split, n_queries=len(qids), seed=seed,
        adapter_config=cfg_a, expected_config=expect_config, config_check=config_check,
        created_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        export_manifest_sha=A0.manifest.get("files", {}), perturbation=pinfo,
        exporter_invariance=inv, s_targets_in_other_contexts=exposed, s_target_copies_changed=exposed_after,
        missing_predictions=missing[:20], n_missing=len(missing),
        mismatches={c: (v[:20] if v and isinstance(v[0], str) else [int(x) for x in v[:20]]) for c, v in mism.items()},
        n_mismatch={c: len(v) for c, v in mism.items()},
        lint=[dict(file=f, line=i, why=w, text=t) for f, i, w, t in lint],
        seconds=dict(setup=round(t1 - t0, 1), real=round(t2 - t1, 1), perturbed=round(t3 - t2, 1)))
    if write_report:
        d = os.path.join(root, "logs", "canary")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, f"{tag}__{ds}__{policy}__{split}.json"), "w") as f:
            json.dump(rep, f, indent=1)
    if not keep:
        shutil.rmtree(work, ignore_errors=True)
    return rep


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="adapter leakage canary (review H1)")
    ap.add_argument("--entry", help="path/to/run.py:predict | cmd:<template> | builtin:prev_then_pop|leaky_sibling")
    ap.add_argument("--ds", required=True)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lint", nargs="*", default=None, help="paths to lint (default: the entry's methods/<m>/)")
    ap.add_argument("--keep", action="store_true", help="keep the perturbed export under work/canary/")
    ap.add_argument("--self-test", action="store_true",
                    help="positive/negative control: builtin:prev_then_pop must PASS, builtin:leaky_sibling FAIL")
    a = ap.parse_args(argv)
    if a.self_test:
        good = run_canary("builtin:prev_then_pop", a.ds, a.policy, a.split, a.n, a.seed, keep=a.keep)
        bad = run_canary("builtin:leaky_sibling", a.ds, a.policy, a.split, a.n, a.seed, keep=a.keep)
        ok = good["result"] == "PASS" and bad["result"] == "FAIL" and bad["n_mismatch"]["ranked_venue_ids"] > 0
        print(json.dumps({"clean": good["result"], "leaky": bad["result"],
                          "leaky_mismatches": bad["n_mismatch"], "self_test": "PASS" if ok else "FAIL"}, indent=1))
        return 0 if ok else 1
    if not a.entry:
        ap.error("--entry is required (or --self-test)")
    rep = run_canary(a.entry, a.ds, a.policy, a.split, a.n, a.seed, keep=a.keep, lint_paths=a.lint)
    print(json.dumps({k: rep[k] for k in ("result", "entry", "ds", "policy", "split", "n_queries", "n_mismatch",
                                          "n_missing", "exporter_invariance", "perturbation",
                                          "s_targets_in_other_contexts", "s_target_copies_changed", "seconds")},
                     indent=1))
    for h in rep["lint"]:
        print(f"LINT {h['file']}:{h['line']}: {h['why']}: {h['text']}")
    return 0 if rep["result"] == "PASS" else 1
