#!/usr/bin/env python
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import sys
import tarfile
import zipfile
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data_raw", "mmpoi")
DERIVED = os.path.join(RAW, "derived")
LOGD = os.path.join(ROOT, "logs", "mmpoi_audit")
ARCH = {"nyc": ("NYC_WWW2024.zip", "NYC", "24591ecd81b9fcb53d76b1e918ebb4124a7d96b06fb5241b59d4bfda85e623e9"),
        "tky": ("TKY_WWW2024.zip", "TKY", "7de8dcc762a0201ad58f457e805e6d636af25c58c7ea7d8fd8820565d1ba14a4")}
PROCESSED = ("NYC_processed.zip", "3998e67a1572dc7d368df17de6b94c399cbd0b3eb46f12c7e2fa55642150baca")
NAME_RE = re.compile(r"^gmap_(\d+)_([0-9a-fA-F]{24})_(\d+)\.png$")
MIN_SIDE = 64
SEED = 0
N_SAMPLE = 60


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(16 << 20), b""):
            h.update(b)
    return h.hexdigest()


def check_hash(path, want):
    got = sha256_file(path)
    if got != want:
        raise SystemExit(f"sha256 mismatch for {path}: {got} != {want}")


def _decode(blob: bytes):
    from PIL import Image
    try:
        im = Image.open(io.BytesIO(blob))
        im.load()
        w, h = im.size
        g = im.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
        a = np.asarray(g, dtype=np.int16)
        bits = (a[:, 1:] > a[:, :-1]).flatten()
        v = 0
        for b in bits:
            v = (v << 1) | int(b)
        return True, w, h, int(np.uint64(v).astype(np.int64)), ""
    except Exception as e:
        return False, 0, 0, 0, type(e).__name__


def iter_images(city):
    zname, prefix, want = ARCH[city]
    zp = os.path.join(RAW, "archives", zname)
    with zipfile.ZipFile(zp) as z, z.open(f"{prefix}/image.tgz") as tz:
        with tarfile.open(fileobj=tz, mode="r|gz") as tf:
            for m in tf:
                if not m.isreg():
                    yield m.name, None
                    continue
                yield m.name, tf.extractfile(m).read()


def id_map(city):
    m = pd.read_parquet(os.path.join(ROOT, "data", city, "id_map_venues.parquet"))
    return dict(zip(m.venue_raw.str.lower(), m.venue_id.astype(int)))


def cmd_inventory(cities=("nyc", "tky")):
    os.makedirs(DERIVED, exist_ok=True)
    for city in cities:
        check_hash(os.path.join(RAW, "archives", ARCH[city][0]), ARCH[city][2])
        rows, blobs, nonreg = [], [], 0
        for name, blob in iter_images(city):
            if blob is None:
                nonreg += 1
                continue
            base = os.path.basename(name)
            mt = NAME_RE.match(base)
            rows.append(dict(member=name, raw_venue=(mt.group(2).lower() if mt else ""),
                             k=int(mt.group(1)) if mt else -1, j=int(mt.group(3)) if mt else -1,
                             nbytes=len(blob), sha256=hashlib.sha256(blob).hexdigest()))
            blobs.append(blob)
        with ProcessPoolExecutor(16) as ex:
            dec = list(ex.map(_decode, blobs, chunksize=256))
        df = pd.DataFrame(rows)
        df["ok"] = [d[0] for d in dec]
        df["w"] = [d[1] for d in dec]
        df["h"] = [d[2] for d in dec]
        df["dhash"] = np.array([d[3] for d in dec], dtype=np.int64)
        df["err"] = [d[4] for d in dec]
        vm = id_map(city)
        df["venue_id"] = df.raw_venue.map(vm).fillna(-1).astype(int)
        good = df.ok & (np.minimum(df.w, df.h) >= MIN_SIDE)
        nv_sha = df[df.ok].groupby("sha256").raw_venue.nunique()
        nv_dh = df[df.ok].groupby("dhash").raw_venue.nunique()
        df["sha_venues"] = df.sha256.map(nv_sha).fillna(0).astype(int)
        df["dhash_venues"] = df.dhash.map(nv_dh).fillna(0).astype(int)
        df["usable"] = good & (df.sha_venues <= 1) & (df.dhash_venues <= 1) & (df.raw_venue != "")
        df = df.sort_values(["raw_venue", "j", "sha256"], kind="stable").reset_index(drop=True)
        rep = df[df.usable].groupby("raw_venue", sort=False).head(1).index
        df["rep"] = False
        df.loc[rep, "rep"] = True
        df.to_parquet(os.path.join(DERIVED, f"{city}_images.parquet"), index=False)
        print(json.dumps(dict(city=city, images=len(df), non_regular=nonreg, ok=int(df.ok.sum()),
                              usable=int(df.usable.sum()), venues=int(df.raw_venue.nunique()),
                              rep=int(df.rep.sum()))), flush=True)


def _haversine_m(lat1, lon1, lat2, lon2):
    r = 6371000.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def _raw_venues(city):
    zname, prefix, _ = ARCH[city]
    s = set()
    with zipfile.ZipFile(os.path.join(RAW, "archives", zname)) as z, \
            z.open(f"{prefix}/dataset_TSMC2014_{prefix}.txt") as f:
        for line in f:
            s.add(line.decode("latin-1").split("\t")[1].lower())
    return s


def _text_venues(city):
    zname, prefix, _ = ARCH[city]
    out = {}
    bad = 0
    with zipfile.ZipFile(os.path.join(RAW, "archives", zname)) as z, z.open(f"{prefix}/review_summary.json") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                d = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                bad += 1
                continue
            for key, val in d.items():
                vid = key.split("_", 1)[-1].lower()
                out[vid] = sum(1 for x in val if isinstance(x, str) and x.strip()) if isinstance(val, list) else 0
    return out, bad


def coverage_city(city, rep_by_vid, has_any_by_vid):
    exp = os.path.join(ROOT, "exports", city, "H-100")
    vp = pd.read_parquet(os.path.join(exp, "venues_public.parquet"))
    cat = vp[vp.in_catalog]
    res = {"catalog_venues": int(len(cat)),
           "catalog_cov_usable": float(cat.venue_id.isin(rep_by_vid).mean()),
           "catalog_cov_any_valid": float(cat.venue_id.isin(has_any_by_vid).mean())}
    for split in ("val", "test"):
        c = pd.read_parquet(os.path.join(exp, f"candidates_{split}.parquet"))
        top = np.stack([np.asarray(x[:20]) for x in c.cand_venue_ids])
        cov = np.isin(top, list(rep_by_vid))
        res[f"cand20_{split}_slot_cov"] = float(cov.mean())
        res[f"cand20_{split}_queries_all20"] = float(cov.all(1).mean())
        res[f"cand20_{split}_queries_ge10"] = float((cov.sum(1) >= 10).mean())
        res[f"_cand_{split}"] = set(np.unique(top).tolist())
    allc = res.pop("_cand_val") | res.pop("_cand_test")
    res["cand20_valtest_distinct_venues"] = len(allc)
    res["cand20_valtest_distinct_cov_usable"] = float(np.mean([v in rep_by_vid for v in allc]))
    res["cand20_valtest_distinct_cov_any_valid"] = float(np.mean([v in has_any_by_vid for v in allc]))
    vr = pd.read_parquet(os.path.join(exp, "visible_rows.parquet"), columns=["venue_id"])
    res["visible_rows_cov_usable"] = float(vr.venue_id.isin(rep_by_vid).mean())
    q = pd.read_parquet(os.path.join(ROOT, "data", city, "queries_val.parquet"), columns=["target_venue_id"])
    res["val_target_cov_usable"] = float(q.target_venue_id.isin(rep_by_vid).mean())
    return res


def cmd_coverage():
    os.makedirs(LOGD, exist_ok=True)
    out = {}
    for city in ("nyc", "tky"):
        df = pd.read_parquet(os.path.join(DERIVED, f"{city}_images.parquet"))
        raw_v = _raw_venues(city)
        ids = set(df.raw_venue[df.raw_venue != ""])
        txt, txt_bad = _text_venues(city)
        rep = df[df.rep & (df.venue_id >= 0)]
        rep_by_vid = set(rep.venue_id.tolist())
        anyv = set(df[df.ok & (df.venue_id >= 0)].venue_id.tolist())
        imgs_per = df[df.ok].groupby("raw_venue").size()
        sha_groups = df[df.ok].groupby("sha256").raw_venue.nunique()
        dh_groups = df[df.ok].groupby("dhash").raw_venue.nunique()
        big = df[df.ok & (df.sha_venues > 1)].groupby("sha256").agg(n_venues=("raw_venue", "nunique"),
                                                                     n_files=("member", "size"),
                                                                     example=("member", "first"))
        big = big.sort_values("n_venues", ascending=False).head(10)
        r = dict(
            archive_images=int(len(df)), unparsed_names=int((df.raw_venue == "").sum()),
            decode_fail=int((~df.ok).sum()), small_lt64=int((df.ok & (np.minimum(df.w, df.h) < MIN_SIDE)).sum()),
            dims_200x200_share=float(((df.w == 200) & (df.h == 200)).mean()),
            image_venues=len(ids), image_venues_joined_raw=float(np.mean([v in raw_v for v in ids])),
            image_venues_in_our_idmap=int(df[df.venue_id >= 0].raw_venue.nunique()),
            images_per_venue_median=float(imgs_per.median()), images_per_venue_max=int(imgs_per.max()),
            exact_dup_excess_files=int(len(df[df.ok]) - df[df.ok].sha256.nunique()),
            exact_dup_cross_venue_groups=int((sha_groups > 1).sum()),
            exact_dup_cross_venue_files=int((df.ok & (df.sha_venues > 1)).sum()),
            dhash_cross_venue_groups=int((dh_groups > 1).sum()),
            dhash_cross_venue_files=int((df.ok & (df.dhash_venues > 1)).sum()),
            usable_files=int(df.usable.sum()), venues_with_usable=int(df[df.usable].raw_venue.nunique()),
            venues_lost_by_dedup=int(df[df.ok].raw_venue.nunique() - df[df.usable].raw_venue.nunique()),
            top_cross_venue_sha_groups=[dict(sha256=k[:16], **{kk: (int(vv) if kk != "example" else vv)
                                                              for kk, vv in row.items()})
                                        for k, row in big.iterrows()],
            text_venues=len(txt), text_venues_nonempty=int(sum(1 for v in txt.values() if v > 0)),
            text_lines_unparsable=txt_bad, text_venues_joined_raw=float(np.mean([v in raw_v for v in txt])),
        )
        r.update(coverage_city(city, rep_by_vid, anyv))
        out[city] = r
    vp = pd.read_parquet(os.path.join(ROOT, "exports", "nyc", "H-100", "venues_public.parquet"))
    vm = id_map("nyc")
    with zipfile.ZipFile(os.path.join(RAW, "processed_dl", PROCESSED[0])) as z:
        meta = [json.loads(line) for line in z.open("NYC/meta-NYC.json")]
    md = pd.DataFrame(meta)
    md["venue_id"] = md.gmap_id.str.lower().map(vm)
    md = md.dropna(subset=["venue_id"]).astype({"venue_id": int})
    md["mlat"], md["mlon"] = md.latitude.astype(float), md.longitude.astype(float)
    nyc_img = pd.read_parquet(os.path.join(DERIVED, "nyc_images.parquet"))
    covered = set(nyc_img[nyc_img.rep & (nyc_img.venue_id >= 0)].venue_id)
    j = vp[vp.in_catalog & vp.venue_id.isin(covered)].merge(md, on="venue_id", how="left", suffixes=("", "_m"))
    has_meta = j.mlat.notna()
    jj = j[has_meta]
    dist = _haversine_m(jj.lat.values, jj.lon.values, jj.mlat.values, jj.mlon.values)
    named = {c: int((md[c] != "null").sum()) for c in ("name", "address", "description", "avg_rating", "price")}
    out["mapping_nyc"] = dict(
        covered_catalog_venues=int(len(j)), with_meta_row=int(has_meta.sum()),
        category_equal=float((jj.category == jj.category_m).mean()),
        coord_within_200m=float((dist <= 200).mean()), coord_median_m=float(np.median(dist)),
        coord_p99_m=float(np.quantile(dist, 0.99)), meta_rows=int(len(meta)),
        meta_nonnull_fields=named, meta_url_prefixes=sorted({u.rsplit("/", 1)[0] for u in md.url})[:5],
        category_mismatch_examples=jj.loc[jj.category != jj.category_m, ["venue_id", "category", "category_m"]]
        .head(10).to_dict("records"))
    with open(os.path.join(LOGD, "coverage.json"), "w") as f:
        json.dump(out, f, indent=1, default=str)
    print(json.dumps(out, indent=1, default=str))


def _blobs_for(city, members):
    want = set(members)
    got = {}
    for name, blob in iter_images(city):
        if name in want:
            got[name] = blob
    return got


def cmd_sheets():
    from PIL import Image, ImageDraw
    os.makedirs(LOGD, exist_ok=True)
    df = pd.read_parquet(os.path.join(DERIVED, "nyc_images.parquet"))
    vp = pd.read_parquet(os.path.join(ROOT, "exports", "nyc", "H-100", "venues_public.parquet"))
    rep = df[df.rep & (df.venue_id >= 0)].merge(vp[vp.in_catalog][["venue_id", "category"]], on="venue_id")
    rep = rep.sort_values("venue_id").reset_index(drop=True)
    idx = np.sort(np.random.RandomState(SEED).choice(len(rep), N_SAMPLE, replace=False))
    s = rep.iloc[idx].reset_index(drop=True)
    s["sample_no"] = np.arange(1, len(s) + 1)
    blobs = _blobs_for("nyc", s.member.tolist())
    per = 20
    for si in range(0, len(s), per):
        part = s.iloc[si:si + per]
        sheet = Image.new("RGB", (5 * 230, 4 * 250), "white")
        d = ImageDraw.Draw(sheet)
        for k, (_, row) in enumerate(part.iterrows()):
            im = Image.open(io.BytesIO(blobs[row.member])).convert("RGB").resize((200, 200))
            x, y = (k % 5) * 230 + 15, (k // 5) * 250 + 5
            sheet.paste(im, (x, y))
            d.text((x, y + 205), f"#{row.sample_no} {row.category}"[:34], fill="black")
        sheet.save(os.path.join(LOGD, f"sheet_{si // per + 1}.png"))
    s[["sample_no", "venue_id", "raw_venue", "category", "member", "sha256"]].to_csv(
        os.path.join(LOGD, "sample.tsv"), sep="\t", index=False)
    print(f"wrote {len(s)} samples")


def cmd_assets(city="nyc"):
    df = pd.read_parquet(os.path.join(DERIVED, f"{city}_images.parquet"))
    vp = pd.read_parquet(os.path.join(ROOT, "exports", city, "H-100", "venues_public.parquet"))
    rep = df[df.rep & (df.venue_id >= 0)].set_index("venue_id")
    outd = os.path.join(ROOT, "exports_mm", city)
    imgd = os.path.join(outd, "images")
    os.makedirs(imgd, exist_ok=True)
    blobs = _blobs_for(city, rep.member.tolist())
    recs = []
    for vid in vp.venue_id.astype(int):
        if vid in rep.index:
            r = rep.loc[vid]
            fn = f"images/{vid}.png"
            p = os.path.join(outd, fn)
            b = blobs[r.member]
            if not (os.path.exists(p) and sha256_file(p) == r.sha256):
                with open(p, "wb") as f:
                    f.write(b)
            recs.append(dict(venue_id=vid, has_image=True, image=fn, sha256=r.sha256, w=int(r.w), h=int(r.h)))
        else:
            recs.append(dict(venue_id=vid, has_image=False, image="", sha256="", w=0, h=0))
    out = pd.DataFrame(recs)
    out.to_parquet(os.path.join(outd, "venue_images.parquet"), index=False)
    man = dict(source=f"MMPOI {ARCH[city][0]} image.tgz", archive_sha256=ARCH[city][2],
               rule="LOG 2026-10-27 15:12 UTC item 4", n_venues=int(len(out)), n_with_image=int(out.has_image.sum()),
               venue_images_sha256=sha256_file(os.path.join(outd, "venue_images.parquet")))
    with open(os.path.join(outd, "MANIFEST.json"), "w") as f:
        json.dump(man, f, indent=1)
    print(json.dumps(man))


if __name__ == "__main__":
    {"inventory": cmd_inventory, "coverage": cmd_coverage, "sheets": cmd_sheets, "assets": cmd_assets}[sys.argv[1]](*sys.argv[2:])
