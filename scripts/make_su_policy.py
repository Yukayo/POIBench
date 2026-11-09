#!/usr/bin/env python
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from poibench import data as D


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(D.SU_DATASETS), choices=list(D.SU_DATASETS))
    a = ap.parse_args(argv)
    for ds in a.datasets:
        ck = D.load_checkins(ds)
        A = D.Arrays.from_checkins(ck)
        qtr = D.load_queries(ds, "train")
        with open(os.path.join(D.data_dir(ds), "meta.json")) as f:
            meta = json.load(f)
        pol = D.make_su_policy(A, qtr, meta["t_val_start_utc"])
        p = os.path.join(D.data_dir(ds), "evidence", "S-u.json")
        with open(p + ".tmp", "w") as f:
            json.dump(pol, f)
        os.replace(p + ".tmp", p)
        s = pol["subsample"]
        print(f"{ds}: k={s['k']} of {s['n_candidate_users']} users ({s['frac_users']:.1%}), "
              f"{s['n_sft_trajectories']} SFT trajectories, {pol['n_train_queries']} train queries "
              f"({pol['frac_train_queries']:.1%}) -> {os.path.relpath(p, ROOT)}", flush=True)


if __name__ == "__main__":
    main()
