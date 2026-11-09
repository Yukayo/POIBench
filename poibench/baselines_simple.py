from __future__ import annotations

import multiprocessing as mp

import numpy as np
import pandas as pd

from poibench.data import Arrays, evidence_slices, policy_vis_from

METHODS = ("pop", "repeat", "markov", "hybrid")
TT_METHODS = ("hybrid-tt",)
TT_WINDOW_H = 1.0
_USER_EVIDENCE = ("repeat", "hybrid", "hybrid-tt")
_STATE = {}


class SimpleBaselines:
    def __init__(self, checkins: pd.DataFrame, venues: pd.DataFrame, policy: dict | None = None):
        self.A = A = Arrays.from_checkins(checkins)
        vids = np.sort(venues["venue_id"].to_numpy(np.int64))
        if not (vids == np.arange(len(vids))).all():
            raise ValueError("venue_id must be dense 0..V-1 (fixed schema)")
        if len(A.venue) and A.venue.max() >= len(vids):
            raise ValueError("check-in venue_id not in venues.parquet")
        self.V = len(vids)
        self.hour = None
        if "t_local" in checkins.columns:
            tl = pd.to_datetime(checkins.sort_values("row_id")["t_local"])
            self.hour = (tl.dt.hour + tl.dt.minute / 60.0).to_numpy(np.float64)
        n_users = len(A.user_first)
        self.policy_name = None if policy is None else policy.get("policy")
        self.policy_kind = None if policy is None else policy.get("kind")
        if policy is not None and len(policy.get("user_visible_from_row_id", {})) not in (0, n_users):
            raise ValueError("policy has a different number of users than checkins (wrong dataset?)")
        self.vis_from = policy_vis_from(policy, n_users) if (policy is not None and policy["kind"] == "H") else None
        self.train_query_ids = None if policy is None else set(int(x) for x in policy["train_query_ids"])

        rows = np.arange(len(A.user), dtype=np.int64)
        vis = A.is_train.copy()
        if self.vis_from is not None:
            vis &= rows >= self.vis_from[A.user]
        self.pop_visible = vis
        self.pop = np.bincount(A.venue[vis], minlength=self.V).astype(np.int64)
        self.pop_order = np.lexsort((np.arange(self.V), -self.pop))

        pm = np.r_[False, (A.traj[1:] == A.traj[:-1]) & vis[1:] & vis[:-1]]
        self.trans_visible = pm
        a, b = A.venue[rows[pm] - 1], A.venue[rows[pm]]
        self.n_transitions = int(pm.sum())
        self.markov = {}
        if len(a):
            uk, cnt = np.unique(a * self.V + b, return_counts=True)
            pa, pb = uk // self.V, uk % self.V
            bounds = np.r_[0, np.flatnonzero(np.diff(pa)) + 1, len(pa)]
            for s, e in zip(bounds[:-1], bounds[1:]):
                self.markov[int(pa[s])] = (pb[s:e].astype(np.int64), cnt[s:e].astype(np.int64))

    def _order(self, method, codes, vis, prev, loo_v, loo_tr, vis_hour=None, th=None):
        popv = self.pop[codes]
        if loo_v >= 0:
            popv = popv - (codes == loo_v)
        if method == "pop":
            return codes[np.lexsort((codes, -popv))]
        uf = np.zeros(len(codes))
        ur = np.full(len(codes), -1.0)
        if method in _USER_EVIDENCE and len(vis):
            uv, cnt = np.unique(vis, return_counts=True)
            _, first_rev = np.unique(vis[::-1], return_index=True)
            last = len(vis) - 1 - first_rev
            j = np.minimum(np.searchsorted(uv, codes), len(uv) - 1)
            hit = uv[j] == codes
            uf[hit] = cnt[j[hit]]
            ur[hit] = last[j[hit]]
        mk = np.zeros(len(codes))
        sm = 0.0
        if method in ("markov", "hybrid", "hybrid-tt"):
            succ = self.markov.get(int(prev))
            if succ is not None:
                sv, sc = succ
                sm = float(sc.sum()) - (1.0 if loo_tr else 0.0)
                j = np.minimum(np.searchsorted(sv, codes), len(sv) - 1)
                hit = sv[j] == codes
                mk[hit] = sc[j[hit]]
                if loo_tr:
                    mk = mk - (codes == loo_v)
        if method == "repeat":
            idx = np.lexsort((codes, -popv, -ur, -uf))
        elif method == "markov":
            idx = np.lexsort((codes, -popv, -mk))
        elif method in ("hybrid", "hybrid-tt"):
            su = float(len(vis))
            if method == "hybrid-tt" and len(vis):
                d = np.abs(vis_hour - th)
                win = vis[np.minimum(d, 24.0 - d) <= TT_WINDOW_H]
                if len(win):
                    uw, cw = np.unique(win, return_counts=True)
                    j = np.minimum(np.searchsorted(uw, codes), len(uw) - 1)
                    hit = uw[j] == codes
                    uf = np.zeros(len(codes))
                    uf[hit] = cw[j[hit]]
                    su = float(len(win))
            score = (uf / su if su > 0 else 0.0) + (mk / sm if sm > 0 else 0.0)
            idx = np.lexsort((codes, -popv, -ur, -score))
        else:
            raise ValueError(method)
        return codes[idx]

    def _pool(self, method, vis, prev, topk):
        parts = [self.pop_order[:topk + 1]]
        if method in _USER_EVIDENCE:
            parts.append(vis)
        if method in ("markov", "hybrid", "hybrid-tt"):
            succ = self.markov.get(int(prev))
            if succ is not None:
                parts.append(succ[0])
        return np.unique(np.concatenate(parts))

    def _predict_range(self, method, sl, cand_codes, topk, lo, hi):
        A = self.A
        out_top, out_cand, nvis = [], [], []
        for i in range(lo, hi):
            r = int(sl["target"][i])
            prev = int(A.venue[r - 1])
            vis_hour = th = None
            if method in _USER_EVIDENCE:
                vis = np.concatenate([A.venue[sl["hist_lo"][i]:sl["hist_hi"][i]],
                                      A.venue[sl["prefix_lo"][i]:sl["prefix_hi"][i]]])
                if method == "hybrid-tt":
                    vis_hour = np.concatenate([self.hour[sl["hist_lo"][i]:sl["hist_hi"][i]],
                                               self.hour[sl["prefix_lo"][i]:sl["prefix_hi"][i]]])
                    th = float(sl["th"][i])
            else:
                vis = A.venue[:0]
            loo_v = int(A.venue[r]) if self.pop_visible[r] else -1
            loo_tr = bool(self.trans_visible[r])
            pool = self._pool(method, vis, prev, topk)
            out_top.append(self._order(method, pool, vis, prev, loo_v, loo_tr, vis_hour, th)[:topk])
            if cand_codes is not None:
                c = cand_codes[i]
                out_cand.append(self._order(method, c, vis, prev, loo_v, loo_tr, vis_hour, th) if len(c) else c)
            nvis.append(int(len(vis)) if method in _USER_EVIDENCE else
                        int(sl["hist_hi"][i] - sl["hist_lo"][i] + sl["prefix_hi"][i] - sl["prefix_lo"][i]))
        return out_top, out_cand, nvis

    def predict(self, method: str, queries: pd.DataFrame, candidates: pd.DataFrame | None = None,
                topk: int = 50, n_jobs: int = 1, target_time: pd.DataFrame | None = None) -> pd.DataFrame:
        if method not in METHODS + TT_METHODS:
            raise ValueError(f"unknown method {method}")
        A = self.A
        q = queries.reset_index(drop=True)
        r = q["target_row_id"].to_numpy(np.int64)
        if (r <= 0).any() or (r >= len(A.user)).any():
            raise ValueError("target_row_id out of range")
        if (q["prev_row_id"].to_numpy(np.int64) != r - 1).any() or (A.traj[r - 1] != A.traj[r]).any():
            raise ValueError("prev_row_id must be target_row_id-1 in the same trajectory (fixed schema)")
        tr = A.split[r] == 0
        if tr.any() and self.train_query_ids is not None:
            bad = [int(x) for x in q["query_id"].to_numpy()[tr] if int(x) not in self.train_query_ids]
            if bad:
                raise ValueError(f"{len(bad)} train queries are not in the policy's train_query_ids")
        sl = evidence_slices(r, A, self.vis_from)
        sl["target"] = r
        if method in TT_METHODS:
            if target_time is None or self.hour is None:
                raise ValueError(f"{method} needs target_time (exports_ablation target_time_<split>) and t_local")
            tmap = dict(zip(target_time["query_id"].to_numpy(),
                            (lambda t: (t.dt.hour + t.dt.minute / 60.0).to_numpy(np.float64))(
                                pd.to_datetime(target_time["target_t_local"]))))
            miss = [int(x) for x in q["query_id"].to_numpy() if x not in tmap]
            if miss:
                raise ValueError(f"{len(miss)} queries have no target time")
            sl["th"] = np.array([tmap[x] for x in q["query_id"].to_numpy()], dtype=np.float64)
        if ((sl["prefix_hi"] - sl["prefix_lo"]) < 1).any():
            raise ValueError("previous check-in not visible for some query")
        cand_codes = None
        if candidates is not None:
            cmap = dict(zip(candidates["query_id"].to_numpy(), candidates["cand_venue_ids"]))
            cand_codes = []
            for qid in q["query_id"].to_numpy():
                if qid not in cmap:
                    raise ValueError(f"query {qid} has no candidates")
                codes = np.asarray(cmap[qid], dtype=np.int64)
                if len(codes) and (codes.min() < 0 or codes.max() >= self.V):
                    raise ValueError("candidate venue id out of range")
                _, first = np.unique(codes, return_index=True)
                cand_codes.append(codes[np.sort(first)])
        n = len(q)
        n_jobs = max(1, min(int(n_jobs), 24))
        if n_jobs == 1 or n < 2000:
            top, cand, nvis = self._predict_range(method, sl, cand_codes, topk, 0, n)
        else:
            _STATE["self"] = self
            _STATE["args"] = (method, sl, cand_codes, topk)
            bounds = np.linspace(0, n, n_jobs * 4 + 1).astype(int)
            try:
                with mp.get_context("fork").Pool(n_jobs) as pool:
                    parts = pool.map(_worker, list(zip(bounds[:-1], bounds[1:])))
            finally:
                _STATE.clear()
            top = [x for pt in parts for x in pt[0]]
            cand = [x for pt in parts for x in pt[1]]
            nvis = [x for pt in parts for x in pt[2]]
        df = pd.DataFrame({"query_id": q["query_id"].to_numpy(),
                           "ranked_venue_ids": [c.tolist() for c in top]})
        if cand_codes is not None:
            df["cand_ranked_venue_ids"] = [c.tolist() for c in cand]
        df["n_visible_own"] = np.asarray(nvis, dtype=np.int64)
        return df

    def allowed_queries(self, queries: pd.DataFrame) -> pd.DataFrame:
        if self.train_query_ids is None:
            return queries
        r = queries["target_row_id"].to_numpy(np.int64)
        keep = (self.A.split[r] != 0) | queries["query_id"].isin(self.train_query_ids).to_numpy()
        return queries[keep]

    def summary(self) -> dict:
        return {"policy": self.policy_name, "policy_kind": self.policy_kind, "n_venues": self.V,
                "n_visible_train_checkins": int(self.pop_visible.sum()), "n_transitions": self.n_transitions}


def _worker(rng):
    self = _STATE["self"]
    method, sl, cand_codes, topk = _STATE["args"]
    return self._predict_range(method, sl, cand_codes, topk, rng[0], rng[1])
