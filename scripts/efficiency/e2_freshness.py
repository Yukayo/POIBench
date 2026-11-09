#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

import numpy as np
import pandas as pd

GAP = 24 * 3600
WEEK = 7 * 86400.0
WEEKLY_METHODS = ("hybrid", "getnext", "getnext-zfeat", "graph-flashback", "ds-llm4poi-lcs", "llm4poi", "gnpr-sid")


def load(ds):
    from poibench.baselines_simple import SimpleBaselines
    from poibench.evaluate import load_policy_json
    d = C.ROOT / "data" / ds
    ck = pd.read_parquet(d / "checkins.parquet")
    venues = pd.read_parquet(d / "venues.parquet")
    pol = load_policy_json(str(C.ROOT), ds, "H-100")
    return ck, venues, pol, SimpleBaselines


class Stream:

    def __init__(self, model):
        self.m = model
        self.pop = model.pop.copy()
        self.markov = {k: (a.copy(), b.copy()) for k, (a, b) in model.markov.items()}
        self.n_tr = model.n_transitions

    def insert(self, v, prev):
        self.pop[v] += 1
        if prev >= 0:
            s = self.markov.get(prev)
            if s is None:
                self.markov[prev] = (np.array([v], np.int64), np.array([1], np.int64))
            else:
                sv, sc = s
                j = int(np.searchsorted(sv, v))
                if j < len(sv) and sv[j] == v:
                    sc[j] += 1
                else:
                    self.markov[prev] = (np.insert(sv, j, v), np.insert(sc, j, 1))
            self.n_tr += 1


def cmd_update(datasets):
    rows = []
    for ds in datasets:
        ck, venues, pol, SB = load(ds)
        t0 = time.perf_counter()
        model = SB(ck, venues, policy=pol)
        fit_s = time.perf_counter() - t0
        A = model.A
        st = Stream(model)
        new = np.flatnonzero(~A.is_train)
        order = new[np.lexsort((new, A.t[new]))]
        same_user = np.r_[False, A.user[1:] == A.user[:-1]]
        prev_ok = np.r_[False, same_user[1:] & ((A.t[1:] - A.t[:-1]) <= GAP)]
        assert (prev_ok == np.r_[False, A.traj[1:] == A.traj[:-1]]).all(), "24-h rule differs from traj ids"
        user_log = {}
        lat = np.empty(len(order))
        venue, user = A.venue, A.user
        pc = time.perf_counter
        T0 = pc()
        for i, r in enumerate(order):
            a = pc()
            st.insert(int(venue[r]), int(venue[r - 1]) if prev_ok[r] else -1)
            user_log.setdefault(int(user[r]), []).append(int(venue[r]))
            lat[i] = pc() - a
        total = pc() - T0
        t0 = time.perf_counter()
        pop_order = np.lexsort((np.arange(model.V), -st.pop))
        resort_s = time.perf_counter() - t0
        ck2 = ck.copy()
        ck2["split"] = "train"
        ref = SB(ck2, venues, policy=None)
        assert (ref.pop == st.pop).all(), "popularity differs from refit"
        assert set(ref.markov) == set(st.markov)
        for k, (a, b) in ref.markov.items():
            sa, sb = st.markov[k]
            assert (a == sa).all() and (b == sb).all(), f"transition counts of {k} differ"
        assert ref.n_transitions == st.n_tr
        t0 = time.perf_counter()
        SB(ck2, venues, policy=None)
        refit_s = time.perf_counter() - t0
        days = (A.t[order].max() - A.t[order].min()) / 86400.0
        r = {"ds": ds, "n_inserts": len(order), "replay_days": days, "inserts_per_week": len(order) / days * 7,
             "insert_us_mean": 1e6 * total / len(order), "insert_us_p50": 1e6 * float(np.percentile(lat, 50)),
             "insert_us_p95": 1e6 * float(np.percentile(lat, 95)), "insert_us_p99": 1e6 * float(np.percentile(lat, 99)),
             "inserts_per_s_1core": len(order) / total, "replay_total_s": total,
             "week_of_inserts_cpu_s": total / days * 7, "pop_resort_ms": 1000 * resort_s,
             "fit_train_s": fit_s, "refit_all_s": refit_s, "n_venues": model.V, "state_equals_refit": True,
             **{f"hw_{k}": v for k, v in C.hardware().items()}}
        print(f"{ds:18s} inserts={len(order):>8d} mean={r['insert_us_mean']:.2f}us p99={r['insert_us_p99']:.2f}us "
              f"week={r['week_of_inserts_cpu_s']:.3f}s resort={r['pop_resort_ms']:.2f}ms refit={refit_s:.3f}s OK", flush=True)
        rows.append(r)
    C.write_csv("e2_update_cost.csv", rows)


def cmd_fresh(datasets, check=True):
    from poibench.data import evidence_slices
    for ds in datasets:
        ck, venues, pol, SB = load(ds)
        model = SB(ck, venues, policy=pol)
        A = model.A
        q = pd.read_parquet(C.ROOT / "data" / ds / "queries_test.parquet")
        r_all = q["target_row_id"].to_numpy(np.int64)

        def rank(st, vis, prev, pop, pop_order):
            model.pop, model.pop_order, model.markov = pop, pop_order, st
            pool = model._pool("hybrid", vis, prev, 50)
            return model._order("hybrid", pool, vis, prev, -1, False)[:50]

        if check:
            sl = evidence_slices(r_all, A, None)
            ref = pd.read_parquet(C.ROOT / "preds" / "hybrid@H-100" / ds / "test.parquet").set_index("query_id")
            bad = 0
            pop0, po0, mk0 = model.pop.copy(), model.pop_order.copy(), dict(model.markov)
            for i in range(0, len(r_all), max(1, len(r_all) // 500)):
                r = r_all[i]
                vis = np.concatenate([A.venue[sl["hist_lo"][i]:sl["hist_hi"][i]], A.venue[sl["prefix_lo"][i]:sl["prefix_hi"][i]]])
                got = rank(mk0, vis, int(A.venue[r - 1]), pop0, po0)
                bad += list(got) != list(ref.loc[r, "ranked_venue_ids"])
            assert bad == 0, f"{ds}: {bad} sampled queries differ from preds/hybrid"
            print(f"{ds}: static re-implementation matches preds/hybrid@H-100 on the sampled queries", flush=True)
        st = Stream(model)
        new = np.flatnonzero(~A.is_train)
        ins = new[np.lexsort((new, A.t[new]))]
        prev_ok = np.r_[False, A.traj[1:] == A.traj[:-1]]
        qo = np.argsort(A.t[r_all] * 0 + A.t[r_all], kind="stable")
        out = [None] * len(r_all)
        j = 0
        dirty = True
        po = None
        t0 = time.perf_counter()
        for k in qo:
            r = r_all[k]
            tq = A.t[r]
            while j < len(ins) and A.t[ins[j]] < tq:
                x = ins[j]
                st.insert(int(A.venue[x]), int(A.venue[x - 1]) if prev_ok[x] else -1)
                j += 1
                dirty = True
            if dirty:
                po = np.lexsort((np.arange(model.V), -st.pop))
                dirty = False
            u = A.user[r]
            vis = A.venue[A.user_first[u]:r]
            out[k] = rank(st.markov, vis, int(A.venue[r - 1]), st.pop, po).tolist()
        el = time.perf_counter() - t0
        d = C.OUT / "fresh_hybrid" / ds
        d.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"query_id": q["query_id"].to_numpy(), "ranked_venue_ids": out}).to_parquet(d / "test.parquet", index=False)
        print(f"{ds}: fresh Hybrid {len(r_all)} queries, {j} inserts interleaved, {el:.1f}s", flush=True)


def per_query_hits(ds, preds: dict):
    from poibench import evaluate as ev
    ev.assert_split_allowed(str(C.ROOT), "test")
    queries, cands, venues, checkins = ev.load_split(str(C.ROOT), ds, "test", "H-100")
    es = ev.build_eval_set(queries, cands, venues, checkins=checkins)
    hits = {}
    for name, df in preds.items():
        ranks, _ = ev.per_query_ranks(es, df)
        hits[name] = ranks["e2e"]
    return es, hits


def cmd_weekly(datasets, B=1000):
    from poibench import evaluate as ev
    wrows, drows = [], []
    for ds in datasets:
        preds = {}
        for m in WEEKLY_METHODS:
            p = C.ROOT / "preds" / f"{m}@H-100" / ds / "test.parquet"
            if p.exists():
                preds[m] = pd.read_parquet(p, columns=["query_id", "ranked_venue_ids"])
        fp = C.OUT / "fresh_hybrid" / ds / "test.parquet"
        if fp.exists():
            preds["hybrid-fresh"] = pd.read_parquet(fp)
        es, ranks = per_query_hits(ds, preds)
        meta = C.meta(ds)
        tval = pd.Timestamp(meta["t_val_start_utc"]).value // 10 ** 9
        ttest = pd.Timestamp(meta["t_test_start_utc"]).value // 10 ** 9
        ckt = pd.read_parquet(C.ROOT / "data" / ds / "checkins.parquet", columns=["row_id", "t_utc"]).sort_values("row_id")
        t = ckt["t_utc"].to_numpy("datetime64[s]").astype(np.int64)[es.queries["target_row_id"].to_numpy(np.int64)]
        x = (t - tval) / WEEK
        wk = np.floor(x).astype(int)
        uc, nu = es.user_codes, es.n_users
        W = ev.bootstrap_weights(nu, B=B, seed=0)

        def usum(v):
            return np.bincount(uc, weights=v, minlength=nu)

        n1 = usum(np.ones(len(x)))
        sx, sxx = usum(x), usum(x * x)
        Wn, Wx, Wxx = W @ n1, W @ sx, W @ sxx
        slope_b, slope_pt = {}, {}
        for m, rk in ranks.items():
            y = (rk <= 1).astype(float)
            sy, sxy = usum(y), usum(x * y)
            Wy, Wxy = W @ sy, W @ sxy
            slope_b[m] = (Wxy - Wx * Wy / Wn) / (Wxx - Wx ** 2 / Wn)
            slope_pt[m] = float(np.polyfit(x, y, 1)[0])
            for w in np.unique(wk):
                msk = wk == w
                nw = usum(msk.astype(float))
                yw = usum(y * msk)
                bw = (W @ yw) / np.maximum(W @ nw, 1e-12)
                wrows.append({"ds": ds, "method": m, "week": int(w), "in_test": bool((t[msk] >= ttest).all()),
                              "n_queries": int(msk.sum()), "n_users": int((nw > 0).sum()), "acc@1": float(y[msk].mean()),
                              "ci_lo": float(np.percentile(bw, 2.5)), "ci_hi": float(np.percentile(bw, 97.5))})
        for m in ranks:
            sb = slope_b[m]
            r = {"ds": ds, "method": m, "n_queries": len(x), "weeks_min": float(x.min()), "weeks_max": float(x.max()),
                 "acc1": float((ranks[m] <= 1).mean()), "slope_per_week": slope_pt[m],
                 "slope_ci_lo": float(np.percentile(sb, 2.5)), "slope_ci_hi": float(np.percentile(sb, 97.5))}
            for ref in ("hybrid", "hybrid-fresh"):
                if ref in slope_b and m != ref:
                    d = sb - slope_b[ref]
                    r[f"dslope_vs_{ref}"] = slope_pt[m] - slope_pt[ref]
                    r[f"dslope_vs_{ref}_ci_lo"] = float(np.percentile(d, 2.5))
                    r[f"dslope_vs_{ref}_ci_hi"] = float(np.percentile(d, 97.5))
            if m == "hybrid-fresh":
                y1, y0 = (ranks[m] <= 1).astype(float), (ranks["hybrid"] <= 1).astype(float)
                gb = (W @ usum(y1) - W @ usum(y0)) / Wn
                r["dacc1_vs_hybrid"] = float(y1.mean() - y0.mean())
                r["dacc1_vs_hybrid_ci_lo"] = float(np.percentile(gb, 2.5))
                r["dacc1_vs_hybrid_ci_hi"] = float(np.percentile(gb, 97.5))
            if "hybrid-fresh" in ranks and m not in ("hybrid-fresh",):
                y1, y0 = (ranks["hybrid-fresh"] <= 1).astype(float), (ranks[m] <= 1).astype(float)
                gb = (W @ usum(y1) - W @ usum(y0)) / Wn
                r["fresh_minus_m_acc1"] = float(y1.mean() - y0.mean())
                r["fresh_minus_m_ci_lo"] = float(np.percentile(gb, 2.5))
                r["fresh_minus_m_ci_hi"] = float(np.percentile(gb, 97.5))
            drows.append(r)
            print(f"{ds:10s} {m:16s} acc1={r['acc1']:.4f} slope/wk={r['slope_per_week']:+.5f} "
                  f"[{r['slope_ci_lo']:+.5f},{r['slope_ci_hi']:+.5f}] "
                  f"d_vs_hyb={r.get('dslope_vs_hybrid', float('nan')):+.5f} "
                  f"[{r.get('dslope_vs_hybrid_ci_lo', float('nan')):+.5f},{r.get('dslope_vs_hybrid_ci_hi', float('nan')):+.5f}]",
                  flush=True)
    C.write_csv("e2_weekly_acc.csv", wrows)
    C.write_csv("e2_drift.csv", drows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["update", "fresh", "weekly"])
    ap.add_argument("--datasets", nargs="+")
    a = ap.parse_args()
    if a.cmd == "update":
        cmd_update(a.datasets or list(C.MAIN + C.BIG))
    elif a.cmd == "fresh":
        cmd_fresh(a.datasets or list(C.MAIN))
    else:
        cmd_weekly(a.datasets or list(C.MAIN))


if __name__ == "__main__":
    main()
