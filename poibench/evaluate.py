from __future__ import annotations

import json
import math
import os
import re
import subprocess
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

METRICS = ("acc@1", "acc@5", "acc@10", "ndcg@10", "mrr", "ndcg@5", "map@20")
DISPLAY_ORDER = ("acc@1", "acc@5", "acc@10", "ndcg@5", "ndcg@10", "map@20", "mrr")
MODES = ("e2e", "cand")
FROZEN_LINE_RE = re.compile(r"Status: FROZEN v(?P<version>\S+) (?P<date>\d{4}-\d{2}-\d{2}) "
                            r"data=(?P<data>[0-9a-f]{64}) eval=(?P<eval>[0-9a-f]{64}) git=(?P<git>[0-9a-f]{7,40})")
STATUS_SEARCH_LINES = 10


def frozen_status(root: str) -> tuple[bool, str]:
    path = os.path.join(root, "PROTOCOL.md")
    if not os.path.exists(path):
        return False, f"{path} does not exist"
    with open(path, encoding="utf-8") as f:
        head = [next(f, "") for _ in range(STATUS_SEARCH_LINES)]
    status = [ln.rstrip("\r\n") for ln in head if ln.startswith("Status:")]
    if not status:
        return False, f"no line starting with 'Status:' in the first {STATUS_SEARCH_LINES} lines of PROTOCOL.md"
    m = FROZEN_LINE_RE.fullmatch(status[0])
    if not m:
        return False, f"status line is not an exact freeze line: {status[0][:120]!r}"
    from .hashing import sha256_file, tree_fingerprint
    data_fp = tree_fingerprint(os.path.join(root, "data"))
    if data_fp != m["data"]:
        return False, f"data fingerprint {data_fp} != frozen {m['data']}"
    eval_fp = sha256_file(os.path.abspath(__file__))
    if eval_fp != m["eval"]:
        return False, f"sha256 of {os.path.abspath(__file__)} = {eval_fp} != frozen {m['eval']}"
    try:
        r = subprocess.run(["git", "-c", f"safe.directory={os.path.abspath(root)}", "-C", root, "cat-file", "-e",
                            f"{m['git']}^{{commit}}"], capture_output=True, timeout=60)
        ok = r.returncode == 0
    except (OSError, subprocess.SubprocessError):
        ok = False
    if not ok:
        return False, f"git commit {m['git']} not found in {root}"
    return True, f"frozen v{m['version']} {m['date']}"


def protocol_frozen(root: str) -> bool:
    return frozen_status(root)[0]


def assert_split_allowed(root: str, split: str) -> None:
    if split not in ("train", "val", "test"):
        raise ValueError(f"unknown split {split!r}")
    if split == "test":
        ok, why = frozen_status(root)
        if not ok:
            raise PermissionError("Refusing to compute metrics on the test split: PROTOCOL.md is not frozen "
                                  f"({why}).")


def _norm_id(x):
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if isinstance(x, (float, np.floating)):
        if math.isnan(x):
            return None
        if float(x).is_integer():
            return str(int(x))
        return str(x)
    return str(x)


def _as_list(v):
    if v is None:
        return []
    if isinstance(v, float) and math.isnan(v):
        return []
    return list(v)


def _dedupe(seq):
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def rank_of(target, lst) -> float:
    try:
        return float(lst.index(target) + 1)
    except ValueError:
        return np.inf


def metrics_from_ranks(ranks: np.ndarray) -> np.ndarray:
    r = np.asarray(ranks, dtype=float)
    found = np.isfinite(r)
    out = np.zeros((len(r), len(METRICS)))
    out[:, 0] = r <= 1
    out[:, 1] = r <= 5
    out[:, 2] = r <= 10
    with np.errstate(divide="ignore", invalid="ignore"):
        out[:, 3] = np.where(r <= 10, 1.0 / np.log2(np.where(found, r, 1.0) + 1.0), 0.0)
        out[:, 4] = np.where(found, 1.0 / np.where(found, r, 1.0), 0.0)
        out[:, 5] = np.where(r <= 5, 1.0 / np.log2(np.where(found, r, 1.0) + 1.0), 0.0)
        out[:, 6] = np.where(r <= 20, 1.0 / np.where(found, r, 1.0), 0.0)
    return out


def weighted_terciles(values: np.ndarray, weights: np.ndarray):
    order = np.argsort(values, kind="stable")
    v = np.asarray(values, float)[order]
    w = np.asarray(weights, float)[order]
    cw = np.cumsum(w)
    tot = cw[-1]
    q1 = v[np.searchsorted(cw, tot / 3.0)]
    q2 = v[np.searchsorted(cw, 2.0 * tot / 3.0)]
    return float(q1), float(q2)


@dataclass
class EvalSet:
    queries: pd.DataFrame
    cand_lists: list
    strata: dict
    strata_info: dict = field(default_factory=dict)
    user_codes: np.ndarray = None
    n_users: int = 0


def build_eval_set(queries: pd.DataFrame, candidates: pd.DataFrame | None, venues: pd.DataFrame | None,
                   cand_k: int = 20, checkins: pd.DataFrame | None = None) -> EvalSet:
    q = queries.reset_index(drop=True).copy()
    if q["query_id"].duplicated().any():
        raise ValueError("duplicate query_id in queries")
    q["_target"] = [_norm_id(x) for x in q["target_venue_id"]]
    n = len(q)

    if candidates is not None:
        cmap = dict(zip(candidates["query_id"], candidates["cand_venue_ids"]))
        missing = [qid for qid in q["query_id"] if qid not in cmap]
        if missing:
            raise ValueError(f"{len(missing)} queries have no candidate list (e.g. {missing[:3]})")
        cand_lists = [_dedupe([_norm_id(x) for x in _as_list(cmap[qid])[:cand_k]]) for qid in q["query_id"]]
    else:
        cand_lists = None

    strata = {"all": np.ones(n, bool)}
    info = {}
    ih = q["in_user_history"].astype(bool).to_numpy()
    strata["hist=repeat"] = ih
    strata["hist=explore"] = ~ih

    if venues is not None:
        vpop = dict(zip((_norm_id(x) for x in venues["venue_id"]), venues["n_train_visits"].astype(float)))
        pos = venues["n_train_visits"].to_numpy(float)
        pos = pos[pos > 0]
        if len(pos):
            q1, q2 = weighted_terciles(pos, pos)
        else:
            q1 = q2 = 0.0
        tp = np.array([vpop.get(t, 0.0) for t in q["_target"]])
        strata["pop=cold"] = tp <= 0
        strata["pop=tail"] = (tp > 0) & (tp <= q1)
        strata["pop=mid"] = (tp > q1) & (tp <= q2)
        strata["pop=head"] = tp > q2
        info["pop_edges"] = {"q1": q1, "q2": q2, "weighting": "train visits", "source": "venues.n_train_visits"}

    if checkins is not None:
        last_row = checkins.groupby("traj_id")["row_id"].max()
        lr = q["traj_id"].map(last_row).to_numpy()
        if pd.isna(lr).any():
            raise ValueError("query traj_id not found in checkins")
        is_last = q["target_row_id"].to_numpy() == lr.astype(np.int64)
        strata["pos=last"] = is_last
        strata["pos=inner"] = ~is_last
        info["pos"] = "pos=last: target is the last check-in of its trajectory (LLM4POI evaluation unit)"

    h = q["user_hist_len"].to_numpy(float)
    if n:
        h1, h2 = np.quantile(h, [1 / 3, 2 / 3])
    else:
        h1 = h2 = 0.0
    strata["hlen=short"] = h <= h1
    strata["hlen=mid"] = (h > h1) & (h <= h2)
    strata["hlen=long"] = h > h2
    info["hlen_edges"] = {"q1": float(h1), "q2": float(h2), "source": "queries.user_hist_len of this split"}

    if cand_lists is not None:
        hit = np.array([t in set(c) for t, c in zip(q["_target"], cand_lists)])
        strata["cand=hit"] = hit
        strata["cand=miss"] = ~hit
        info["cand_k"] = cand_k
        info["cand_recall"] = float(hit.mean()) if n else float("nan")

    ucodes, uniq = pd.factorize(q["user_id"])
    return EvalSet(queries=q, cand_lists=cand_lists, strata=strata, strata_info=info,
                   user_codes=np.asarray(ucodes), n_users=len(uniq))


def per_query_ranks(es: EvalSet, preds: pd.DataFrame, e2e_cutoff: int = 50, allow_missing: bool = False):
    if preds["query_id"].duplicated().any():
        raise ValueError("duplicate query_id in predictions")
    qids = set(es.queries["query_id"])
    extra = [x for x in preds["query_id"] if x not in qids]
    if extra:
        raise ValueError(f"{len(extra)} predicted query_ids not in queries (e.g. {extra[:3]})")
    has_cand_col = "cand_ranked_venue_ids" in preds.columns
    pmap = dict(zip(preds["query_id"], preds["ranked_venue_ids"]))
    cmap = dict(zip(preds["query_id"], preds["cand_ranked_venue_ids"])) if has_cand_col else {}

    n = len(es.queries)
    r_e2e = np.full(n, np.inf)
    r_cand = np.full(n, np.inf)
    n_missing = n_short = n_dupe = 0
    fill_frac = []
    for i, (qid, tgt) in enumerate(zip(es.queries["query_id"], es.queries["_target"])):
        if qid not in pmap:
            n_missing += 1
            continue
        raw = [_norm_id(x) for x in _as_list(pmap[qid])]
        lst = _dedupe(raw)
        n_dupe += len(raw) != len(lst)
        n_short += len(lst) < 10
        r_e2e[i] = rank_of(tgt, lst[:e2e_cutoff])
        if es.cand_lists is not None:
            cands = es.cand_lists[i]
            cset = set(cands)
            src = lst
            if has_cand_col:
                cv = _as_list(cmap.get(qid))
                if cv:
                    src = _dedupe([_norm_id(x) for x in cv])
            ordered = [x for x in src if x in cset]
            oset = set(ordered)
            fill = [x for x in cands if x not in oset]
            full = ordered + fill
            fill_frac.append(len(fill) / len(cands) if cands else 0.0)
            r_cand[i] = rank_of(tgt, full)
    if n_missing and not allow_missing:
        raise ValueError(f"{n_missing}/{n} queries have no prediction (pass allow_missing=True to score them as misses)")
    diag = {"n_queries": n, "n_missing": n_missing, "n_short_lists": n_short, "n_lists_with_duplicates": n_dupe,
            "e2e_cutoff": e2e_cutoff, "used_cand_ranked_column": has_cand_col}
    out = {"e2e": r_e2e}
    if es.cand_lists is not None:
        out["cand"] = r_cand
        diag["cand_fill_rate"] = float(np.mean(fill_frac)) if fill_frac else float("nan")
        diag["cand_any_fill_frac"] = float(np.mean(np.array(fill_frac) > 0)) if fill_frac else float("nan")
    return out, diag


def bootstrap_weights(n_users: int, B: int = 1000, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    W = np.empty((B, n_users), dtype=np.float64)
    for b in range(B):
        W[b] = np.bincount(rng.integers(0, n_users, size=n_users), minlength=n_users)
    return W


def _per_user_sums(vals: np.ndarray, ucodes: np.ndarray, mask: np.ndarray, n_users: int):
    u = ucodes[mask]
    C = np.bincount(u, minlength=n_users).astype(float)
    S = np.stack([np.bincount(u, weights=vals[mask, j], minlength=n_users) for j in range(vals.shape[1])], axis=1)
    return S, C


def _ci(a: np.ndarray, alpha: float = 0.05):
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return float("nan"), float("nan")
    lo, hi = np.percentile(a, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def evaluate(methods: dict, queries: pd.DataFrame, candidates: pd.DataFrame | None = None,
             venues: pd.DataFrame | None = None, cand_k: int = 20, e2e_cutoff: int = 50, B: int = 1000,
             seed: int = 0, ref: str | None = None, pairs: str = "all", allow_missing: bool = False,
             strata: list | None = None, checkins: pd.DataFrame | None = None, *, split: str,
             root: str | None = None) -> dict:
    if root is None:
        from poibench import ROOT as root
    assert_split_allowed(root, split)
    es = build_eval_set(queries, candidates, venues, cand_k=cand_k, checkins=checkins)
    snames = [s for s in es.strata if strata is None or s in strata]
    W = bootstrap_weights(es.n_users, B=B, seed=seed) if B > 0 else None

    WC = {}
    for s in snames:
        m = es.strata[s]
        C = np.bincount(es.user_codes[m], minlength=es.n_users).astype(float)
        WC[s] = W @ C if W is not None else None

    rows, boots, diags, perq = [], {}, {}, {}
    for name, preds in methods.items():
        ranks, diag = per_query_ranks(es, preds, e2e_cutoff=e2e_cutoff, allow_missing=allow_missing)
        diags[name] = diag
        for mode, r in ranks.items():
            vals = metrics_from_ranks(r)
            perq[(name, mode)] = vals
            for s in snames:
                m = es.strata[s]
                nq = int(m.sum())
                nu = int(len(np.unique(es.user_codes[m])))
                point = vals[m].mean(axis=0) if nq else np.full(len(METRICS), np.nan)
                if W is not None and nq:
                    S, _ = _per_user_sums(vals, es.user_codes, m, es.n_users)
                    with np.errstate(invalid="ignore", divide="ignore"):
                        bt = (W @ S) / WC[s][:, None]
                    boots[(name, mode, s)] = bt
                else:
                    bt = None
                for j, met in enumerate(METRICS):
                    lo, hi = _ci(bt[:, j]) if bt is not None else (float("nan"), float("nan"))
                    rows.append({"method": name, "mode": mode, "stratum": s, "metric": met,
                                 "value": float(point[j]), "ci_lo": lo, "ci_hi": hi,
                                 "n_queries": nq, "n_users": nu})

    prow = []
    names = list(methods)
    if pairs == "ref":
        if ref not in methods:
            raise ValueError(f"ref method {ref!r} not among methods")
        plist = [(a, ref) for a in names if a != ref]
    elif pairs == "all":
        plist = [(names[i], names[j]) for i in range(len(names)) for j in range(i + 1, len(names))]
    else:
        plist = []
    for a, b in plist:
        for mode in MODES:
            if (a, mode) not in perq:
                continue
            for s in snames:
                m = es.strata[s]
                nq = int(m.sum())
                if not nq:
                    continue
                d_point = perq[(a, mode)][m].mean(axis=0) - perq[(b, mode)][m].mean(axis=0)
                ba, bb = boots.get((a, mode, s)), boots.get((b, mode, s))
                for j, met in enumerate(METRICS):
                    rec = {"a": a, "b": b, "mode": mode, "stratum": s, "metric": met,
                           "diff": float(d_point[j]), "n_queries": nq}
                    if ba is not None and bb is not None:
                        d = ba[:, j] - bb[:, j]
                        d = d[np.isfinite(d)]
                        rec["ci_lo"], rec["ci_hi"] = _ci(d)
                        if len(d):
                            p = 2 * min(np.mean(d <= 0), np.mean(d >= 0))
                            rec["p_boot"] = float(min(1.0, p))
                    prow.append(rec)

    return {
        "config": {"cand_k": cand_k, "e2e_cutoff": e2e_cutoff, "B": B, "seed": seed,
                   "bootstrap_unit": "user", "ci": "95% percentile", "metrics": list(METRICS)},
        "n_queries": int(len(es.queries)), "n_users": int(es.n_users),
        "strata_info": es.strata_info,
        "strata_sizes": {s: int(es.strata[s].sum()) for s in snames},
        "diagnostics": diags,
        "results": rows,
        "pairs": prow,
    }


CONFIRMATORY_ALPHA = 0.05
MAIN_DATASETS = ("nyc", "tky", "ca", "sin", "istanbul", "sao_paulo")
DATANEEDS_DATASETS = ("nyc", "tky", "istanbul")
LLM_METHODS = ("ds-llm4poi-lcs", "llm4poi", "gnpr-sid")
NONLLM_TRAINED = ("getnext", "graph-flashback")
REFERENCE_SIMPLE = "hybrid"
TRAINED_DATANEEDS = ("getnext", "graph-flashback", "ds-llm4poi-lcs", "llm4poi", "gnpr-sid")
SIMPLE_DATANEEDS = ("repeat", "markov", "hybrid")


def confirmatory_tests() -> list:
    T = []
    for ds in MAIN_DATASETS:
        for m in LLM_METHODS:
            for r in (REFERENCE_SIMPLE,) + NONLLM_TRAINED:
                T.append(dict(family="Q1", ds=ds, a=f"{m}@H-100", b=f"{r}@H-100", mode="e2e", stratum="all",
                              metric="acc@1", group="H-100"))
        for m in NONLLM_TRAINED:
            T.append(dict(family="Q1", ds=ds, a=f"{m}@H-100", b=f"{REFERENCE_SIMPLE}@H-100", mode="e2e",
                          stratum="all", metric="acc@1", group="H-100"))
    for ds in DATANEEDS_DATASETS:
        for m in SIMPLE_DATANEEDS + TRAINED_DATANEEDS:
            T.append(dict(family="Q2", ds=ds, a=f"{m}@H-10", b=f"{m}@H-100", mode="e2e", stratum="all",
                          metric="ndcg@10", group=f"policies-{m}"))
            if m in TRAINED_DATANEEDS:
                T.append(dict(family="Q2", ds=ds, a=f"{m}@H-10", b=f"{m}@S-10", mode="e2e", stratum="all",
                              metric="ndcg@10", group=f"policies-{m}"))
    for arm in ("B", "C"):
        T.append(dict(family="MM", ds="nyc", a=f"qwen2.5-vl-ref-{arm}@H-100", b="qwen2.5-vl-ref-A@H-100", mode="cand",
                      stratum="all", metric="acc@1", group="H-100"))
    T.append(dict(family="MM", ds="nyc", a="mmpoi@H-100", b="mmpoi-noimage@H-100", mode="e2e", stratum="all",
                  metric="acc@1", group="H-100"))
    for m in ("ds-llm4poi-lcs", "llm4poi"):
        T.append(dict(family="FLIP", ds="nyc", a=f"{m}-tt@H-100", b=f"{m}@H-100", mode="e2e", stratum="all",
                      metric="acc@1", group="H-100"))
    return T


def holm(pvals) -> list:
    idx = [i for i, p in enumerate(pvals) if p is not None and not (isinstance(p, float) and math.isnan(p))]
    m = len(idx)
    out = [None] * len(pvals)
    running = 0.0
    for rank, i in enumerate(sorted(idx, key=lambda k: (pvals[k], k))):
        running = max(running, min(1.0, (m - rank) * float(pvals[i])))
        out[i] = running
    return out


def load_policy_json(root: str, ds: str, policy: str) -> dict:
    with open(os.path.join(root, "data", ds, "evidence", f"{policy}.json")) as f:
        return json.load(f)


def candidates_path(root: str, ds: str, split: str, policy: str | dict | None = None) -> str:
    d = os.path.join(root, "data", ds)
    if policy is not None:
        pol = load_policy_json(root, ds, policy) if isinstance(policy, str) else policy
        rel = (pol.get("candidates") or {}).get(split)
        if rel:
            return os.path.join(d, rel)
    return os.path.join(d, f"candidates_{split}.parquet")


def load_split(root: str, ds: str, split: str, policy: str | None = None):
    d = os.path.join(root, "data", ds)
    queries = pd.read_parquet(os.path.join(d, f"queries_{split}.parquet"))
    cpath = candidates_path(root, ds, split, policy)
    candidates = pd.read_parquet(cpath) if os.path.exists(cpath) else None
    venues = pd.read_parquet(os.path.join(d, "venues.parquet"))
    checkins = pd.read_parquet(os.path.join(d, "checkins.parquet"), columns=["row_id", "traj_id"])
    return queries, candidates, venues, checkins


def load_preds(root: str, method: str, ds: str, split: str) -> pd.DataFrame:
    return pd.read_parquet(os.path.join(root, "preds", method, ds, f"{split}.parquet"))


def check_candidate_files(root: str, ds: str, split: str, method_names: list, policy: str | None) -> list:
    want = os.path.normpath(os.path.relpath(candidates_path(root, ds, split, policy), root))
    bad = []
    for m in method_names:
        cp = os.path.join(root, "preds", m, ds, f"{split}.cost.json")
        if not os.path.exists(cp):
            continue
        with open(cp) as f:
            got = json.load(f).get("candidates_file")
        if got and os.path.normpath(got) != want:
            bad.append((m, got, want))
    return bad


def _is_mock_marker(v) -> bool:
    return isinstance(v, str) and v.strip().lower().startswith("mock")


def mock_artifacts(root: str, ds: str, split: str, method_names: list) -> list:
    bad = []
    for m in method_names:
        cp = os.path.join(root, "preds", m, ds, f"{split}.cost.json")
        if not os.path.exists(cp):
            continue
        with open(cp) as f:
            d = json.load(f)
        g = d.get("generation") if isinstance(d.get("generation"), dict) else {}
        if any(_is_mock_marker(v) for v in (d.get("backend"), d.get("model_fingerprint"), g.get("model_fingerprint"))):
            bad.append(m)
    return bad


def missing_candidate_file_decl(root: str, ds: str, split: str, methods: dict) -> list:
    bad = []
    for m, preds in methods.items():
        if "cand_ranked_venue_ids" not in preds.columns:
            continue
        cp = os.path.join(root, "preds", m, ds, f"{split}.cost.json")
        got = None
        if os.path.exists(cp):
            with open(cp) as f:
                got = json.load(f).get("candidates_file")
        if not got:
            bad.append(m)
    return bad


def val_feedback(ds: str, preds: pd.DataFrame, ks=(1, 5, 10, 20), root: str | None = None,
                 query_ids=None) -> dict:
    if root is None:
        from poibench import ROOT as root
    q = pd.read_parquet(os.path.join(root, "data", ds, "queries_val.parquet"), columns=["query_id", "target_venue_id"])
    if query_ids is not None:
        q = q[q["query_id"].isin(set(int(x) for x in query_ids))]
    pmap = dict(zip(preds["query_id"], preds["ranked_venue_ids"]))
    ranks = np.array([rank_of(_norm_id(t), _dedupe([_norm_id(x) for x in _as_list(pmap.get(qid))]))
                      for qid, t in zip(q["query_id"], q["target_venue_id"])])
    out = {f"acc@{k}": float(np.mean(ranks <= k)) if len(ranks) else float("nan") for k in ks}
    out["mrr"] = float(np.mean(np.where(np.isfinite(ranks), 1.0 / np.where(np.isfinite(ranks), ranks, 1.0), 0.0)))
    out["n_queries"] = int(len(q))
    out["n_missing"] = int(sum(qid not in pmap for qid in q["query_id"]))
    return out


def evaluate_files(root: str, ds: str, split: str, method_names: list, out_prefix: str | None = None,
                   policy: str | None = None, e2e_only: bool = False, **kw) -> dict:
    assert_split_allowed(root, split)
    mock = mock_artifacts(root, ds, split, method_names)
    if mock:
        raise ValueError("refusing predictions written by a mock (CPU stand-in) backend: " + ", ".join(mock))
    queries, candidates, venues, checkins = load_split(root, ds, split, policy)
    if e2e_only:
        candidates = None
    else:
        bad = check_candidate_files(root, ds, split, method_names, policy)
        if bad:
            raise ValueError("candidate-file mismatch (pass --policy matching the run, or --e2e-only): "
                             + "; ".join(f"{m} ran on {g}, evaluation uses {w}" for m, g, w in bad))
    if policy is not None and split == "train":
        allowed = set(int(x) for x in load_policy_json(root, ds, policy)["train_query_ids"])
        queries = queries[queries["query_id"].isin(allowed)]
    methods = {m: load_preds(root, m, ds, split) for m in method_names}
    if not e2e_only:
        unknown = missing_candidate_file_decl(root, ds, split, methods)
        if unknown:
            raise ValueError("candidate mode: methods with a cand_ranked_venue_ids column must name their "
                             "candidates_file in <split>.cost.json (review L3): " + ", ".join(unknown))
    res = evaluate(methods, queries, candidates, venues, checkins=checkins, split=split, root=root, **kw)
    res["config"].update({"ds": ds, "split": split, "policy": policy, "e2e_only": bool(e2e_only),
                          "candidates_file": None if e2e_only else
                          os.path.relpath(candidates_path(root, ds, split, policy), root)})
    if out_prefix:
        os.makedirs(os.path.dirname(os.path.abspath(out_prefix)), exist_ok=True)
        with open(out_prefix + ".json", "w") as f:
            json.dump(res, f, indent=1, default=float)
        pd.DataFrame(res["results"]).to_csv(out_prefix + ".csv", index=False)
        if res["pairs"]:
            pd.DataFrame(res["pairs"]).to_csv(out_prefix + ".pairs.csv", index=False)
    return res


def format_table(res: dict, mode: str = "e2e", stratum: str = "all") -> str:
    df = pd.DataFrame(res["results"])
    df = df[(df["mode"] == mode) & (df["stratum"] == stratum)]
    if df.empty:
        return "(no rows)"
    df = df.assign(cell=[f"{v:.4f} [{lo:.4f},{hi:.4f}]" for v, lo, hi in zip(df.value, df.ci_lo, df.ci_hi)])
    return df.pivot(index="method", columns="metric", values="cell")[list(DISPLAY_ORDER)].to_string()


def main(argv=None):
    import argparse
    from poibench import ROOT
    ap = argparse.ArgumentParser(description=(__doc__ or " ").split("\n")[0])
    ap.add_argument("ds")
    ap.add_argument("split", choices=["train", "val", "test"])
    ap.add_argument("methods", nargs="+", help="method directory names under ROOT/preds")
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--policy", default=None,
                    help="evidence policy whose candidate file is used (default: data/<ds>/candidates_<split>.parquet = H-100)")
    ap.add_argument("--cand-k", type=int, default=20)
    ap.add_argument("--e2e-cutoff", type=int, default=50)
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pairs", choices=["all", "ref", "none"], default="all")
    ap.add_argument("--ref", default=None)
    ap.add_argument("--allow-missing", action="store_true")
    ap.add_argument("--e2e-only", action="store_true",
                    help="skip within-candidate mode (needed when methods ran under different candidate files)")
    ap.add_argument("--out", default=None, help="output prefix (default ROOT/results/<ds>/<split>/eval_<methods>)")
    a = ap.parse_args(argv)
    out = a.out
    if out is None:
        tag = "+".join(a.methods) if len(a.methods) <= 4 else f"{len(a.methods)}methods"
        if a.e2e_only:
            tag += "__e2e"
        elif a.policy:
            tag += f"__cands-{a.policy}"
        out = os.path.join(a.root, "results", a.ds, a.split, f"eval_{tag}")
    res = evaluate_files(a.root, a.ds, a.split, a.methods, out_prefix=out, policy=a.policy, cand_k=a.cand_k,
                         e2e_cutoff=a.e2e_cutoff, B=a.B, seed=a.seed, pairs=a.pairs, ref=a.ref,
                         allow_missing=a.allow_missing, e2e_only=a.e2e_only)
    for mode in (("e2e",) if a.e2e_only else MODES):
        print(f"== {a.ds}/{a.split} mode={mode} stratum=all")
        print(format_table(res, mode))
    print("candidate recall@%d: %s" % (a.cand_k, res["strata_info"].get("cand_recall")))
    print("written:", out + ".json")


if __name__ == "__main__":
    main()
