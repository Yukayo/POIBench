from __future__ import annotations
import os as _pb_os
def _pb_root():
    p = _pb_os.path.dirname(_pb_os.path.abspath(__file__))
    while not _pb_os.path.isfile(_pb_os.path.join(p, "poibench", "__init__.py")) and _pb_os.path.dirname(p) != p:
        p = _pb_os.path.dirname(p)
    return p
_PB_ROOT = _pb_os.environ.get("POIBENCH_ROOT") or _pb_root()
_PB_RAW = _pb_os.environ.get("POIBENCH_RAW_ROOT") or _pb_os.path.join(_PB_ROOT, "data_raw")

import argparse
import datetime as dt
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd

from poibench import data as D


def log(msg):
    print(f"[{dt.datetime.now(dt.timezone.utc):%H:%M:%S}] {msg}", flush=True)


def write_parquet(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)


def build(ds: str, workers: int, raw_root: str | None):
    out = D.data_dir(ds, ROOT)
    os.makedirs(os.path.join(out, "evidence"), exist_ok=True)
    t0 = time.time()
    log(f"{ds}: load raw")
    raw, stats = D.load_raw(ds, raw_root)
    sums = D.raw_checksums(ds, raw_root)
    cuts = None
    if ds in D.VARIANTS:
        with open(os.path.join(D.data_dir(D.base_dataset(ds), ROOT), "meta.json")) as f:
            bm = json.load(f)
        cuts = (pd.Timestamp(bm["t_val_start_utc"]), pd.Timestamp(bm["t_test_start_utc"]))
        log(f"{ds}: variant of {D.base_dataset(ds)}, cuts {cuts[0]} / {cuts[1]}, filter counts t_utc < {cuts[0]}")
    ck, venues, meta, idm = D.preprocess(raw, stats, cuts=cuts, freq_filter=ds not in D.FLASHBACK)
    if cuts is not None:
        meta["variant_of"] = D.base_dataset(ds)
        meta["cuts_from"] = f"data/{D.base_dataset(ds)}/meta.json"
    if ds == "ca":
        meta["source"] = dict(kind="gowalla_ca_sthgcn", checkins=D.GOWALLA_SNAP, categories=D.GOWALLA_SPOTS,
                              polygons=D.CA_POLYGONS_KEY, states=list(D.CA_STATES), tz=D.CA_TZ)
    elif ds in D.TIST_COUNTRY:
        meta["source"] = dict(kind="foursquare_tist2015", dir=D.TIST_KEY,
                              country=D.TIST_COUNTRY[ds], local_time="t_utc + per-row offset")
    elif ds in D.FLASHBACK:
        meta["source"] = dict(kind=f"flashback_{ds}", filter=D.FLASHBACK[ds],
                              note="Flashback/Graph-Flashback preprocessing in load_raw; PROTOCOL step 2 not applied",
                              **({"tz_table": D.GOWALLA_TZ_KEY, "categories": D.GOWALLA_SPOTS,
                                  "no_category_label": D.GOWALLA_NO_CATEGORY} if ds == "gowalla" else
                                 {"dir": D.WWW2019_KEY, "local_time": "t_utc + per-row offset"}))
    elif ds in D.MASSIVE_STEPS and ds != "istanbul":
        meta["source"] = dict(kind="massive_steps", dir=D.MASSIVE_STEPS[ds][0], tz=D.MASSIVE_STEPS[ds][1],
                              trail_prefix=D.CFG["istanbul_trail_prefix"])
    ck = ck.sort_values("row_id").reset_index(drop=True)
    A = D.Arrays.from_checkins(ck)
    log(f"{ds}: {len(ck)} check-ins, {ck.user_id.nunique()} users, {len(venues)} venues; queries")
    queries, qstats = D.build_queries(ck, venues, A)
    stats.update(qstats)

    write_parquet(ck, os.path.join(out, "checkins.parquet"))
    write_parquet(venues, os.path.join(out, "venues.parquet"))
    write_parquet(idm["users"], os.path.join(out, "id_map_users.parquet"))
    write_parquet(idm["venues"], os.path.join(out, "id_map_venues.parquet"))
    for s, q in queries.items():
        write_parquet(q, os.path.join(out, f"queries_{s}.parquet"))

    pols = D.make_policies(A, queries["train"], meta["t_val_start_utc"])
    recall = {}
    hist_share = {}
    for x in D.POLICY_X:
        hname = f"H-{x}"
        hp = pols[hname]
        vis_from = None if x == 100 else D.policy_vis_from(hp, len(A.user_first))
        st = D.prepare_retriever(A, venues, vis_from)
        cand_paths = {}
        for s in D.SPLITS:
            q = queries[s]
            if s == "train":
                q = q[q["query_id"].isin(set(hp["train_query_ids"]))]
            log(f"{ds}: retrieve {hname} {s} ({len(q)} queries)")
            c = D.retrieve(A, venues, q["target_row_id"].to_numpy(), vis_from, workers=workers, st=st)
            assert len(c) == len(q) and (c["cand_venue_ids"].map(len) == D.RETRIEVER["out_len"]).all()
            assert (c["cand_venue_ids"].map(lambda v: len(set(v))) == D.RETRIEVER["out_len"]).all()
            if x == 100:
                rel = f"candidates_{s}.parquet"
            else:
                rel = f"evidence/{hname}.candidates_{s}.parquet"
            write_parquet(c, os.path.join(out, rel))
            cand_paths[s] = rel
            if s == "val":
                qv = queries["val"]
                r = D.candidate_recall(qv, c, row_split=A.split)
                for flag, name in ((True, "repeat"), (False, "explore")):
                    sub = qv[qv["in_user_history"] == flag]
                    rr = D.candidate_recall(sub, c, row_split=A.split)
                    r.update({f"{k}|{name}": v for k, v in rr.items()})
                recall[hname] = r
                sl = D.evidence_slices(qv["target_row_id"].to_numpy(), A, vis_from)
                hl, inh = D._hist_stats(qv["target_row_id"].to_numpy(), sl, A)
                hist_share[hname] = dict(in_user_history=float(inh.mean()), median_hist_len=float(np.median(hl)))
        for name in (hname, f"S-{x}"):
            p = dict(pols[name])
            p["description"] = D.POLICY_TEXT[p["kind"]]
            p["candidates"] = cand_paths if p["kind"] == "H" else {s: f"candidates_{s}.parquet" for s in D.SPLITS}
            with open(os.path.join(out, "evidence", f"{name}.json"), "w") as f:
                json.dump(p, f)

    counts = {}
    for s in D.SPLITS:
        sub = ck[ck["split"] == s]
        q = queries[s]
        counts[s] = dict(checkins=int(len(sub)), users=int(sub.user_id.nunique()), venues=int(sub.venue_id.nunique()),
                         trajectories_touching=int(sub.traj_id.nunique()), queries=int(len(q)),
                         query_users=int(q.user_id.nunique()),
                         t_utc_min=str(sub.t_utc.min()), t_utc_max=str(sub.t_utc.max()))
    qv = queries["val"]
    hl = qv["user_hist_len"].to_numpy()
    utrain = ck[ck.split == "train"].groupby("user_id").size().to_numpy()
    tl = ck.groupby("traj_id").size().to_numpy()
    meta.update(
        dataset=ds, built_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        config=D.CFG, retriever=D.RETRIEVER, raw_files=sums, preprocess_stats=stats, counts=counts,
        totals=dict(checkins=int(len(ck)), users=int(ck.user_id.nunique()), venues=int(len(venues)),
                    catalog_venues=int((venues.n_train_visits > 0).sum()), trajectories=int(ck.traj_id.nunique()),
                    categories=int(venues.category.nunique())),
        traj_len=dict(zip(["min", "p25", "median", "p75", "p90", "max", "mean"],
                          [float(v) for v in [tl.min(), *np.percentile(tl, [25, 50, 75, 90]), tl.max(), tl.mean()]])),
        user_train_checkins=dict(zip(["min", "p10", "p25", "median", "p75", "p90", "max", "mean"],
                                     [float(v) for v in [utrain.min(), *np.percentile(utrain, [10, 25, 50, 75, 90]),
                                                         utrain.max(), utrain.mean()]])),
        val_user_hist_len=dict(zip(["min", "p10", "p25", "median", "p75", "p90", "max", "mean"],
                                   [float(v) for v in [hl.min(), *np.percentile(hl, [10, 25, 50, 75, 90]),
                                                       hl.max(), hl.mean()]])),
        val_in_user_history=float(qv["in_user_history"].mean()),
        val_target_is_prev_venue=float((qv["target_venue_id"].to_numpy() == A.venue[qv["prev_row_id"].to_numpy()]).mean()),
        val_by_policy=hist_share, val_candidate_recall=recall,
        policies={k: dict(n_train_queries=v["n_train_queries"], frac_train_queries=v["frac_train_queries"],
                          frac_visible_precutoff_checkins=v["frac_visible_precutoff_checkins"]) for k, v in pols.items()},
        future_filter=future_filter_stats(raw, venues, idm, meta["t_val_start_utc"], qv,
                                          min_poi=D.FLASHBACK[ds]["min_poi"] if ds in D.FLASHBACK else None),
        build_seconds=round(time.time() - t0, 1),
    )
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1, default=str)
    log(f"{ds}: done in {time.time() - t0:.0f}s; val recall {recall.get('H-100')}")


def future_filter_stats(raw: pd.DataFrame, venues: pd.DataFrame, idm: dict, t_val: str, qv: pd.DataFrame,
                        min_poi: int | None = None) -> dict:
    pre = raw.loc[raw["t_utc"] < pd.Timestamp(t_val), "venue_raw"].value_counts()
    vr = idm["venues"].set_index("venue_id")["venue_raw"]
    precount = vr.reindex(venues["venue_id"]).map(pre).fillna(0).to_numpy()
    weak = precount < (D.CFG["min_poi_checkins"] if min_poi is None else min_poi)
    cat = venues["n_train_visits"].to_numpy() > 0
    return dict(min_poi=int(D.CFG["min_poi_checkins"] if min_poi is None else min_poi),
                catalog_venues_below_min_before_cutoff=int((weak & cat).sum()),
                val_target_share_at_such_venues=float(weak[qv["target_venue_id"].to_numpy()].mean()))


PUBLISHED = {
    "nyc": "GETNext release (also used by LLM4POI): 1,075 users, 5,099 POIs, 103,941 rows "
           "(train 83,228 / val 10,339 / test 10,374), 14,130 trajectories (min length 3).",
    "tky": "GETNext release: 2,280 users, 7,842 POIs, 360,820 rows (train 288,987 / val 35,840 / test 35,993), "
           "44,608 trajectories (min length 3). LLM4POI regenerates TKY from raw with the same rules as ours.",
    "nyc_trainfilter": "Variant of nyc (review M1): same raw file and cut timestamps; the >= 10 venue/user "
                       "filter counts only pre-cutoff check-ins. Not comparable to published numbers.",
    "tky_trainfilter": "Variant of tky (review M1): same raw file and cut timestamps; the >= 10 venue/user "
                       "filter counts only pre-cutoff check-ins. Not comparable to published numbers.",
    "istanbul": "Massive-STEPS Istanbul is released with a random trail-level split over 2012-2013 and "
                "2017-2018 trails; our chronological re-split of the 2013-release trails is not comparable to it.",
    "sao_paulo": "Massive-STEPS Sao Paulo (same handling as istanbul: 2013-release trails only, chronological "
                 "re-split); not comparable to the Massive-STEPS paper's random trail-level split.",
    "ca": "STHGCN / LLM4POI / GNPR-SID Gowalla-CA: 6,592 users, 14,027 POIs after their >= 10 filter (= our "
          "after_filter counts, reproduced exactly); our selection equals the authors' released "
          "dataset_gowalla_ca_ne.csv row for row (636,512 rows). Differences: local time = America/Los_Angeles "
          "(theirs: UTC used as local), chronological UTC cut. GETNext's own Gowalla-CA release is a different "
          "cut (4,315 users, 9,922 POIs, 241,610 rows) and is not comparable.",
    "sin": "Foursquare global-scale (Yang et al., TIST 2015) check-ins at venues with country code SG "
           "(2012-04 .. 2013-09); no single published preprocessing of this cut exists (SIN in other papers is often "
           "the 2010-11 Singapore set of Yuan et al., SIGIR 2013, a different collection).",
    "gowalla": "Flashback (IJCAI 2020) / Graph-Flashback (KDD 2022) Gowalla: the authors' released checkins-gowalla.txt "
               "(52,979 users / 121,851 POIs / 3,300,986 rows, Flashback Table 1) = SNAP with exact duplicates dropped "
               "+ iterative 10/10 core (reproduced row for row); their loader's >= 101 rule gives 7,768 users / "
               "106,994 POIs / 1,823,598 check-ins (Graph-Flashback and later papers) = our after_filter counts "
               "exactly. Our frequency filter is not applied on top. Differences: chronological 80/10/10 UTC cut + "
               "24 h trajectories + isolated / unseen-target removal (theirs: per-user 80/20 split of fixed-length "
               "sequences); local time from the spot's IANA zone (theirs: UTC); categories from Liu et al. where "
               "available, else 'Unknown' (see preprocess_stats).",
    "foursquare_global": "Flashback / Graph-Flashback Foursquare: the authors' released checkins-4sq.txt (46,065 users / "
               "69,005 POIs / 9,450,342 rows, Flashback Table 1) = the WWW2019 raw check-in release + iterative "
               "50/100 core (same users, POIs and row count; their timestamps are 0-6 s earlier than the official "
               "raw file in 99.98 % of rows, ours are the official ones). Published after their loader: 45,343 users "
               "/ 68,879 POIs / 9,361,228 check-ins; their loader never registers the last user block of the file "
               "(user 2733324, 17,014 check-ins), we keep that user, so our after_filter counts are 45,344 users / "
               "9,378,242 check-ins (POIs: see funnel). Same protocol differences as gowalla; local time = UTC + "
               "per-row offset; category from raw_POIs.txt.",
}


M1_REVIEW = {
    "nyc": dict(catalog_venues_below_min_before_cutoff=977, val_target_share_at_such_venues=0.1915),
    "tky": dict(catalog_venues_below_min_before_cutoff=1329, val_target_share_at_such_venues=0.0657),
    "istanbul": dict(catalog_venues_below_min_before_cutoff=564, val_target_share_at_such_venues=0.0548),
}


def fmt_q(d: dict, keys=("min", "p25", "median", "p75", "p90", "max", "mean")) -> str:
    return " / ".join(f"{d[k]:.0f}" if k != "mean" else f"{d[k]:.1f}" for k in keys if k in d)


def data_card(datasets):
    metas = {}
    for ds in datasets:
        p = os.path.join(D.data_dir(ds, ROOT), "meta.json")
        if os.path.exists(p):
            with open(p) as f:
                metas[ds] = json.load(f)
    L = ["# POIBench data card", "",
         f"Generated by `scripts/build_data.py` on {dt.datetime.now(dt.timezone.utc):%Y-%m-%d %H:%M} UTC from `data/<ds>/meta.json`. "
         "Protocol: `PROTOCOL.md` (v0 draft). Only validation-split target statistics are reported; the test split is "
         "counted but never scored.", ""]
    L += ["## Totals", "",
          "| dataset | check-ins | users | venues (catalog = >=1 train visit) | categories | trajectories | traj length min/p25/median/p75/p90/max/mean |",
          "|---|---|---|---|---|---|---|"]
    for ds, m in metas.items():
        t = m["totals"]
        L.append(f"| {ds} | {t['checkins']:,} | {t['users']:,} | {t['venues']:,} ({t['catalog_venues']:,}) | "
                 f"{t['categories']} | {t['trajectories']:,} | {fmt_q(m['traj_len'])} |")
    L += ["", "## Per split", "",
          "| dataset | split | t_utc range | check-ins | users | venues | queries | users with queries |",
          "|---|---|---|---|---|---|---|---|"]
    for ds, m in metas.items():
        for s in D.SPLITS:
            c = m["counts"][s]
            L.append(f"| {ds} | {s} | {c['t_utc_min'][:16]} .. {c['t_utc_max'][:16]} | {c['checkins']:,} | "
                     f"{c['users']:,} | {c['venues']:,} | {c['queries']:,} | {c['query_users']:,} |")
    L += ["", "## Preprocessing funnel", "", "| dataset | step | value |", "|---|---|---|"]
    for ds, m in metas.items():
        for k, v in m["preprocess_stats"].items():
            L.append(f"| {ds} | {k} | {v:,} |" if isinstance(v, (int, float)) else f"| {ds} | {k} | {v} |")
        L.append(f"| {ds} | val/test cut (UTC) | {m['t_val_start_utc']} / {m['t_test_start_utc']} |")
    L += ["", "## Validation targets and history (policy H-100 unless stated)", "",
          "| dataset | target already in visible user evidence | target = previous venue | val history length min/p10/p25/median/p75/p90/max/mean | user train check-ins min/p10/p25/median/p75/p90/max/mean |",
          "|---|---|---|---|---|"]
    keys8 = ("min", "p10", "p25", "median", "p75", "p90", "max", "mean")
    for ds, m in metas.items():
        L.append(f"| {ds} | {m['val_in_user_history']:.3f} | {m['val_target_is_prev_venue']:.3f} | "
                 f"{fmt_q(m['val_user_hist_len'], keys8)} | {fmt_q(m['user_train_checkins'], keys8)} |")
    L += ["", "## Evidence policies (data-needs study)", "",
          "| dataset | policy | visible pre-cutoff check-ins | train queries usable (share) | val: target in visible evidence | val: median visible history |",
          "|---|---|---|---|---|---|"]
    for ds, m in metas.items():
        for x in D.POLICY_X:
            for kind in ("H", "S"):
                name = f"{kind}-{x}"
                p = m["policies"][name]
                hs = m["val_by_policy"].get(f"H-{x}" if kind == "H" else "H-100", {})
                L.append(f"| {ds} | {name} | {p['frac_visible_precutoff_checkins']:.3f} | "
                         f"{p['frac_train_queries']:.3f} | {hs.get('in_user_history', float('nan')):.3f} | "
                         f"{hs.get('median_hist_len', float('nan')):.0f} |")
    L += ["", "## Common retriever r1: candidate recall on validation", "",
          "Recall = share of val queries whose target is in the first k candidates (ground truth never inserted). "
          "repeat/explore = target in / not in the user's visible evidence under H-100.", "",
          "| dataset | policy | R@20 | R@50 | R@20 repeat | R@20 explore | R@50 repeat | R@50 explore |",
          "|---|---|---|---|---|---|---|---|"]
    for ds, m in metas.items():
        for pol, r in m["val_candidate_recall"].items():
            L.append(f"| {ds} | {pol} | {r['recall@20']:.3f} | {r['recall@50']:.3f} | {r['recall@20|repeat']:.3f} | "
                     f"{r['recall@20|explore']:.3f} | {r['recall@50|repeat']:.3f} | {r['recall@50|explore']:.3f} |")
    L += ["", "## Frequency filter and later check-ins (review M1)", "",
          "Base datasets apply the >= 10 venue/user filter to counts that include val/test check-ins (GETNext/LLM4POI "
          "convention). `*_trainfilter` variants count only pre-cutoff check-ins (same raw file and cut timestamps). "
          "Venues below the threshold before the cutoff pass only because of later check-ins.", "",
          "| dataset | catalog venues with < min_poi raw check-ins before T_val (min_poi = 10; foursquare_global 50) | share of val targets at those venues |",
          "|---|---|---|"]
    for ds, m in metas.items():
        ff = m.get("future_filter")
        if ff is None and ds in M1_REVIEW:
            ff = M1_REVIEW[ds]
        if ff:
            L.append(f"| {ds} | {ff['catalog_venues_below_min_before_cutoff']:,} | "
                     f"{ff['val_target_share_at_such_venues']:.3f} |")
    L += ["", "## Comparison with published preprocessing", ""]
    for ds in metas:
        L.append(f"- **{ds}**: {PUBLISHED[ds]}")
    L += ["", "## Raw-file snapshot (sha256)", "", f"| dataset | file (under {_PB_RAW} unless absolute) | bytes | sha256 |",
          "|---|---|---|---|"]
    for ds, m in metas.items():
        for rel, v in m["raw_files"].items():
            L.append(f"| {ds} | {rel} | {v['bytes']:,} | `{v['sha256']}` |")
    L.append("")
    with open(os.path.join(ROOT, "data", "DATA_CARD.md"), "w") as f:
        f.write("\n".join(L))
    log("wrote data/DATA_CARD.md")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(D.DATASETS), choices=list(D.ALL_DATASETS))
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--raw_root", default=None, help="one directory holding every raw file (default: $POIBENCH_RAW_ROOT, else RAW_ROOT + ROOT/data_raw, see poibench.data.raw_path)")
    ap.add_argument("--card-only", action="store_true")
    a = ap.parse_args()
    a.workers = min(a.workers, 24)
    if not a.card_only:
        for ds in a.datasets:
            build(ds, a.workers, a.raw_root)
    data_card(D.ALL_DATASETS)


if __name__ == "__main__":
    main()
