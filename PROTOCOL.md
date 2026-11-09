# POIBench protocol

Status: PUBLIC COPY of the protocol frozen as v1.0 on 2026-10-31, amended as v1.0.1 on 2026-11-08 (evaluator input check only; not a freeze line; see "Test lock" in README.md)

> Public copy. References to `LOG.md`, `reports/`, `methods/*/CARD.md`, `methods/*/SPEC.md`, `data/DATA_CARD.md` and job
> identifiers point to the authors' pre-registration log and internal notes, which are not part of this repository.
> The frozen evaluator (`poibench/evaluate.py`) refuses test-split metrics until this file carries a freeze line of
> your own copy (README.md, "Test lock").

Benchmark question: *when is an LLM worth it for next-POI prediction?* (effectiveness, data needs, full cost).
Every decision below lists its **source**: `GETNext` (Yang et al., SIGIR 2022, released code and data),
`LLM4POI` (Li et al., SIGIR 2024, `preprocessing/` code derived from STHGCN), or `ours` (a benchmark choice,
with the reason). Fixed constants were written to `LOG.md` (2026-10-27 12:20 UTC) before any build or metric
run. Implementation: `poibench/data.py`, `scripts/build_data.py`; checks: `tests/test_data.py`; counts:
`data/DATA_CARD.md`.

## 0. Owner rules (apply to everything)

| rule | detail |
|---|---|
| seed | seed 0 only, for every method and every random choice (sampling, init, bootstrap). |
| no baseline tuning | authors' default hyperparameters; where none exist, an arbitrary value written to `LOG.md` before any run. Never chosen by looking at results (val or test). |
| test lock | no metric on test until this file is frozen; validation numbers are for sanity checks only. `poibench/evaluate.py` (`evaluate_files`, `evaluate`) and `poibench.data.candidate_recall` refuse test unless the freeze line of section 9 is present and its hashes verify. |
| pre-registration | ROOT is a git repository (since 2026-10-27 13:44 UTC, `bash scripts/git.sh`). Every `LOG.md` decision is committed **before** the run it governs; LOG timestamps come from `date -u`. |
| adapter inputs | method adapters read only `exports/<ds>/<policy>/` (section 3.1) and must pass the leakage canary (section 7) before their first real run. |
| compute | data building is CPU only; GPU jobs are scripted to be portable (relative paths under ROOT, one command per job, env spec in `envs/`). |

## 1. Datasets and raw sources

| dataset | raw source (read-only, sha256 in DATA_CARD) | notes | source |
|---|---|---|---|
| nyc | Foursquare TSMC2014 `dataset_TSMC2014_NYC.txt` (2012-04 .. 2013-02) | latin-1 encoded; UTC time + timezone offset in minutes | GETNext / LLM4POI |
| tky | Foursquare TSMC2014 `dataset_TSMC2014_TKY.txt` | same format | GETNext / LLM4POI |
| istanbul | Massive-STEPS Istanbul `{train,validation,test}.parquet`, merged | see 1.1 | ours |
| ca | Gowalla SNAP `loc-gowalla_totalCheckins` + Liu et al. `gowalla_spots_subset1.csv` (2009-02 .. 2010-10) | see 1.2; California + Nevada | STHGCN / LLM4POI / GNPR-SID |
| sin | Foursquare global-scale `dataset_TIST2015` (Yang et al., TIST 2015; 2012-04 .. 2013-09), venues with country SG | see 1.3 | ours (city from the global set) |
| sao_paulo | Massive-STEPS Sao Paulo `{train,validation,test}.parquet`, merged | as istanbul (1.1), tz America/Sao_Paulo | ours |
| gowalla | Gowalla SNAP `loc-gowalla_totalCheckins` (global, 2009-02 .. 2010-10) + Liu et al. categories where available | see 1.4; Flashback-line filter | Flashback / Graph-Flashback |
| foursquare_global | Foursquare WWW2019 raw check-in release `raw_Checkins_anonymized.txt` + `raw_POIs.txt` (Yang et al., WWW 2019; global, 2012-04 .. 2014-01) | see 1.5; Flashback-line filter | Flashback / Graph-Flashback |

**Main study vs appendix (owner decision 2026-10-28 ~10:40 UTC; supersedes the 08:19 split).** Main study = nyc,
tky, ca, sin, istanbul, sao_paulo (every method, GPU runs). gowalla and foursquare_global (Flashback-line large
datasets, 1.4 / 1.5; the most-used global sets in the CCF-A survey `reports/dataset_survey/SUMMARY.md`) are built and
exported with the same protocol, but LLM training on them is too slow for this round: no GPU job runs on them (all
held); only the CPU simple baselines are reported, in an appendix.

### 1.1 Istanbul specifics (ours)
* Only trails whose `trail_id` starts with `2013` (Semantic Trails 2013 release; check-ins 2012-04 .. 2013-09,
  the same Yang et al. global-scale collection era as TSMC). The `2018` trails (2017-10 .. 2018-12) are dropped:
  a chronological split over both eras would put almost all val/test in a different collection with different
  user ids.
* Massive-STEPS' own random, trail-level split is **not** used; Istanbul is re-split chronologically like NYC/TKY.
  Numbers are therefore not comparable with the Massive-STEPS paper.
* Massive-STEPS `timestamp` is UTC: its NYC hour-of-day histogram matches the TSMC UTC histogram, not the local
  one (checked 2026-10-27). Local time = `zoneinfo('Europe/Istanbul')` (UTC+3 summer / UTC+2 winter in 2012-13).
* Coordinates: a venue whose coordinates are missing on some rows takes them from its other rows; check-ins at
  venues with no coordinates at all are dropped **before** filtering (57,975 rows, 11,339 venues; 34,352 of these
  rows are at venues with >= 10 check-ins, mostly `Bridge` and `Home (private)`). Reason: GETNext / Graph-Flashback
  / the retriever all need coordinates. Listed as an open issue (section 10).
* Massive-STEPS' trails already exclude single-check-in trails; our own trajectory rule (24 h gap) re-segments.

### 1.2 CA specifics (added 2026-10-28, LOG entry of that date)
* Selection = LLM4POI `preprocessing/generate_ca_raw.py` (from STHGCN): inner join of the SNAP check-ins with the
  spot metadata on spot id, point inside the California or Nevada polygon of the authors' `us_state_polygon_json.json`
  (official LLM4POI `datasets.zip`, `data_raw/llm4poi_release/`). Checked: identical rows to the authors'
  `dataset_gowalla_ca_ne.csv`; the step-2 filter reproduces the published 6,592 users / 14,027 POIs.
* Category = Gowalla leaf name (ids sharing a name are suffixed `(gowalla:<id>)`). Local time = zoneinfo
  `America/Los_Angeles` (the authors use UTC as local time; we do not).

### 1.3 SIN specifics (added 2026-10-28)
* `dataset_TIST2015.zip` from the authors' page (`data_raw/foursquare_tist2015/`, sha256 in SHA256SUMS). Area =
  POI-file country code `SG`; venue category / coordinates from the POI file; local time = UTC + per-row offset
  (as TSMC). Not the 2010-11 Singapore collection of Yuan et al. (SIGIR 2013) that some papers call SIN.

### 1.4 gowalla specifics (added 2026-10-28)
* Flashback (Yang et al., IJCAI 2020) / Graph-Flashback (Rao et al., KDD 2022) preprocessing, reproduced against the
  authors' released `checkins-gowalla.txt` (`data_raw/flashback_release/`, identical to Graph-Flashback's
  `flashback_data.zip`): SNAP rows, exact (user, time, spot) duplicates dropped, iterative core POIs >= 10 / users
  >= 10 (= their file, row for row: 52,979 users / 121,851 POIs / 3,300,986 rows), then their loader's
  `min_checkins = 101` (users with >= 101 check-ins): 7,768 users / 106,994 POIs / 1,823,598 check-ins, the numbers
  Graph-Flashback and later papers publish. **This replaces step 2 below** (our >= 10 filter is not applied on top).
* Category = Liu et al.'s spot category (as for ca) where the spot id is in `gowalla_spots_subset1.csv`; spots
  without metadata (3.1 % of the 106,994; 2.0 % of check-ins) get the category `Unknown`. The join is by the same
  SNAP spot id, with exactly one category per spot (no conflicts).
* Local time: the spot's IANA zone (timezonefinder 6.5.2 on its median coordinates; `scripts/gowalla_venue_tz.py`).
  Flashback / Graph-Flashback use UTC hours; we do not copy that (global data spans many zones).

### 1.5 foursquare_global specifics (added 2026-10-28)
* The Flashback line's "Foursquare" is the WWW2019 global release (Apr 2012 .. Jan 2014), not TIST2015 (Apr 2012 ..
  Sep 2013). Their released `checkins-4sq.txt` (46,065 users / 69,005 POIs / 9,450,342 rows) equals the 90M-row
  **raw** check-in file of that release (`raw_Checkins_anonymized.txt`, not the 22.8M-row
  `dataset_WWW_Checkins_anonymized.txt`) after an iterative core POIs >= 50 / users >= 100 without de-duplication
  (same users, POIs and row count; their timestamps are 0-6 s earlier than the official file in 99.98 % of rows; we
  use the official ones). Then users >= 101 (their loader). Rows with an unparseable timestamp are dropped first.
* Published after the loader: 45,343 users / 68,879 POIs / 9,361,228 check-ins. Their loader never registers the
  last user block of its file (user 2733324, 17,014 check-ins); we keep that user (difference reported in DATA_CARD).
* Category and coordinates from `raw_POIs.txt` (one row per venue); local time = UTC + per-row offset (as TSMC/TIST).

## 2. Preprocessing (order matters)

| # | step | choice | source |
|---|---|---|---|
| 1 | venue attributes | one category per venue = modal raw category (ties alphabetical); coordinates = median of the venue's raw rows. Raw TSMC rows of a venue differ slightly. The mode/median use **all** raw rows, including val/test-period ones: venue metadata (id, category, coordinates) is treated as a public catalog, as in the original papers' POI tables, under every policy. No count or popularity is part of that catalog (review M3). | ours (LLM4POI keeps per-row values) |
| 2 | frequency filter | one pass: drop venues with < 10 check-ins, then users with < 10 check-ins (on the venue-filtered data). Not iterated. **The counts include val/test check-ins** (GETNext/LLM4POI convention, kept for comparability with published numbers): 977 of 5,066 nyc catalog venues (tky 1,329 of 7,832; istanbul 564 of 3,931) have < 10 raw check-ins before T_val and pass only because of later check-ins, and **19.1 % (nyc) / 6.6 % (tky) / 5.5 % (istanbul) of val targets** are at such venues (review M1; val only). Variants `nyc_trainfilter` / `tky_trainfilter` count only check-ins before T_val (same raw file and cut timestamps as the base dataset; everything else unchanged) and serve as a protocol-sensitivity check. | LLM4POI `do_filter` (`min_*_freq: 9` with strict `>`). Reproduces GETNext's published counts exactly: NYC 1,083 users / 5,135 POIs / 147,938 check-ins; TKY 2,293 / 7,873 / 447,570. Variants: ours. |
| 3 | split | global chronological 80/10/10 by check-in count over all filtered check-ins: cut times = `t_utc` of check-in #floor(0.8N) and #floor(0.9N); `train: t < T_val`, `val: T_val <= t < T_test`, `test: t >= T_test` (timestamp ties go to the later split). | LLM4POI `split_train_test` (positional cut on local time); ours: cut by UTC timestamp so ties are never split. |
| 4 | isolated check-ins | removed after the split: a check-in with no other check-in of the same user within 24 h before or after it. Equivalent to dropping single-check-in trajectories. The neighbour may lie across the cutoff (a last train check-in survives because of a val check-in within 24 h), also in the trainfilter variants. | LLM4POI `split_train_test` |
| 5 | trajectories | per user, a new trajectory starts when the gap to the previous check-in is > 24 h (computed on UTC). Trajectories may cross split boundaries. | LLM4POI `generate_id` (`session_time_interval: 1440`, on local time); GETNext release uses the same 24 h gap rule (checked on its NYC files: 99.96 % agreement) but additionally drops trajectories with < 3 check-ins. |
| 6 | ids | dense `user_id`, `venue_id` (maps in `id_map_*.parquet`); `row_id` ordered by (user, t_utc, raw order) so each user is one contiguous chronological block. | ours |
| 7 | queries | every check-in after the first of its trajectory is a query; its split is the split of the target check-in. | GETNext (every position) and LLM4POI `ignore_first`; note LLM4POI's prompt builder evaluates only the LAST check-in of each trajectory, so a `last-of-trajectory` stratum should be added to the evaluator for comparability (open item 6). |
| 8 | unseen targets | val/test queries whose user has no train check-in or whose target venue has no train check-in are dropped (context rows with unseen venues stay). | LLM4POI `remove_unseen_user_poi`, GETNext |
| 9 | duplicates | exact duplicate rows (same user, venue, second) are kept (NYC 250, TKY 577). | LLM4POI (no dedup) |
| 10 | catalog | candidate universe = venues with >= 1 train check-in (`venues.n_train_visits > 0`); catalog metadata (id, coordinates, category) is public under every policy. | ours |

**gowalla / foursquare_global:** step 2 is replaced by the Flashback-line filter of 1.4 / 1.5 (which also counts
val/test check-ins, as step 2 does); steps 1 and 3-10 are unchanged. The Flashback line's own evaluation (per-user
80/20 split of fixed-length 20-check-in sequences, no sessions) is not used.

**Differences from the GETNext NYC release** (which LLM4POI also uses for NYC): GETNext drops trajectories with
< 3 check-ins and cuts its split after that removal, so its windows differ (its val starts 2012-11-09, ours
2012-11-23). For TKY, LLM4POI regenerates data from raw with steps 2-8, i.e. our pipeline up to UTC-vs-local
time ordering. Published NYC numbers are therefore approximately, not exactly, comparable (see section 10).

## 3. Query input and visible evidence

* **Target time is not given** to any model by default (ours, matching GETNext, which predicts the next POI
  without its time). LLM4POI's prompt does give it ("at {target_time}, which POI will user X visit?",
  `v2/convert_prompt_llm4poi.py`); a with-target-time variant is an optional, clearly labelled ablation, never the
  default. The time of the previous check-in is available. The target's category is not given.
* **Static evidence cutoff** (LLM4POI `convert_prompt_llm4poi.build_history`: history = train rows before the
  trajectory start). For a val or test query the visible own evidence is
  `hist` = the user's train-split check-ins before the current trajectory, plus
  `prefix` = the current trajectory up to the previous check-in (always visible; it is the query input, even if it
  starts before the cutoff). Val-period check-ins of earlier trajectories are **not** visible to val or test queries.
* For a train query the visible own evidence is all of the user's check-ins before the target (all in train).
* Population evidence (other users; graphs, transition and popularity statistics, retrieval stores, similar-
  trajectory pools, training samples) = train-split check-ins only. Val is used only for early stopping /
  sanity checks with each method's default procedure; models are not refit on train+val for test.
* `queries_<split>.in_user_history` = target venue occurs in the visible own evidence (hist or prefix);
  `user_hist_len` = number of visible own check-ins. Both refer to policy H-100.
* `poibench.data.query_evidence(ds, split, policy)` returns the per-query row ranges
  `[hist_lo, hist_hi) U [prefix_lo, prefix_hi)` (used by poibench itself: retriever, simple baselines).

### 3.1 Adapter inputs (exports; review H1)

`data/<ds>/` contains every val/test target, so the evidence rule is enforced by construction rather than by
convention: `scripts/export_inputs.py` (`poibench/exports.py`, format `e1`) writes, per dataset and policy,
`exports/<ds>/<policy>/` with

| file | content |
|---|---|
| `visible_rows.parquet` | the train-period check-ins visible under the policy (H-x: latest ceil(x%) per user; S-x: all), schema columns without `split` |
| `venues_public.parquet` | `venue_id, category, lat, lon, in_catalog, n_visible_train_visits` (count from `visible_rows` only; no `n_train_visits`) |
| `query_inputs_<split>.parquet` | `query_id, user_id, traj_id, prev_row_id, prev_venue_id, prev_t_utc, prev_t_local, hist_lo, hist_hi, prefix_lo, prefix_hi, n_visible_own`; **no target column, no target time, no `in_user_history`**; train = the policy's `train_query_ids` |
| `context_<split>.parquet` (val, test) | each query's own trajectory prefix up to but excluding the target, one copy per `query_id` |
| `candidates_<split>.parquet` | the policy's common-retriever candidates |
| `train_labels.parquet` | `label_*` (row, venue, time) of the train queries: the only targets an adapter sees |
| `policy_public.json`, `MANIFEST.json` | policy name/kind/x/description (no per-user visibility bounds); source and file sha256 |

Rules: an adapter reads only its `exports/<ds>/<policy>/` (through `poibench.exports.Export`), never `data/`;
population statistics come only from `visible_rows`; a val/test query may use `visible_rows` and its own context
rows (`query_id == q`; a later query's context copy contains this query's target); model selection on val only
through `poibench.evaluate.val_feedback` (acc@k from predictions; the adapter never reads val targets). Ids
(`row_id`, `query_id`, `traj_id`) are the original identifiers, needed to score predictions; they are keys, not
features (their values encode row counts, a residual side channel that the canary cannot detect).
`methods/export_hybrid/run.py` is the reference adapter: it reproduces the `hybrid` simple baseline from the
exports alone, list for list.

## 4. Common candidate retriever (version r1)

One retriever builds the candidates of every method and every split (`candidates_<split>.parquet`: 50 ranked
venue ids, the first 20 are the main candidate set). The ground-truth venue is **never inserted**; a query whose
target is not retrieved is a miss in candidate mode. All constants were fixed in `LOG.md` before the first build.

| part | definition |
|---|---|
| U: user history | catalog venues in the query's visible own evidence; score = sum over visits of 2^(t_visit / 14 days) (exponentially time-decayed visit count, half-life 14 d); top 50 |
| T: transitions | counts of `prev_venue -> next_venue` over consecutive pairs inside training trajectories (both rows train and visible under the policy), from the query's previous venue; top 50 |
| G: geo-popular | the 100 nearest catalog venues (haversine) to the previous venue, ranked by visible train visit count; top 50 |
| fusion | reciprocal rank fusion, score = sum over lists 1/(60 + rank) (k = 60, Cormack et al. 2009); ties by visible popularity then venue id; top 50, padded with globally popular venues if the union is smaller |
| train queries | leave-one-out: the query's own transition and own target visit are subtracted from T and G |
| policies | under H-x every statistic uses visible rows only; separate files `evidence/H-x.candidates_<split>.parquet` |

Validation candidate recall (H-100; repeat = target in visible own evidence):

| dataset | R@20 | R@50 | R@20 repeat | R@20 explore |
|---|---|---|---|---|
| nyc | 0.702 | 0.796 | 0.862 | 0.070 |
| tky | 0.626 | 0.757 | 0.776 | 0.182 |
| istanbul | 0.556 | 0.694 | 0.848 | 0.247 |
| ca | 0.513 | 0.643 | 0.750 | 0.198 |
| sin | 0.521 | 0.662 | 0.737 | 0.208 |
| sao_paulo | 0.653 | 0.762 | 0.861 | 0.202 |

These are reported, never optimised against. Candidate-mode results must always be shown next to this ceiling.
Checked: val candidates are unchanged when val target venues are altered, and H-10 val candidates are unchanged
when invisible pre-cutoff rows are altered (`tests/test_data.py`).

## 5. Evidence-access policies (data-needs study)

Files: `data/<ds>/evidence/<policy>.json` with `user_visible_from_row_id` (per user), `train_query_ids`,
`candidates` (relative paths), counts and a description.

| policy | visible evidence | training samples (gradient updates, also few-shot demonstrations) |
|---|---|---|
| H-x | for each user only the latest ceil(x% * n_u) pre-cutoff (train-split) check-ins are visible to ANY component: graphs, statistics, retrieval stores, profiles, prompts, training samples. Row r of user u is visible iff r >= `user_visible_from_row_id[u]`. A val/test query's own trajectory prefix stays visible. | train queries whose previous check-in is visible (`train_query_ids`); their input prefix is truncated to visible rows |
| S-x | everything pre-cutoff visible (as H-100) | exactly the H-x `train_query_ids` set |

x in {10, 70, 100}; H-100 = S-100 = default. **Ours (design):** S-x uses the same sample set as H-x, so
H-x vs S-x isolates "how much history is visible" with gradient samples held fixed, and S-x vs S-100 isolates
"how many samples are used" with history held fixed. Realised train-query shares: H-10 0.091 / 0.095 / 0.085
and H-70 0.691 / 0.696 / 0.697 (nyc / tky / istanbul). Methods without training use the policy's evidence only.
Val/test query sets are identical under every policy.

### 5.1 S-u: user-subsample training budget for the large datasets (owner decision 2026-10-28)
**Not used this round** (owner decision ~10:40 UTC: no GPU run on gowalla / foursquare_global). Policy files, exports
and checks exist and are kept for a later round; nothing is queued under S-u.
On gowalla and foursquare_global the LLM fine-tunes (LLM4POI, GNPR-SID) do not fit the GPU budget at full scale
(reports/STATUS.md). They run under policy **S-u** (`data/<ds>/evidence/S-u.json`, `poibench.data.make_su_policy`,
`scripts/make_su_policy.py`): kind S, so every pre-cutoff check-in is visible exactly as in H-100 / S-x (visible rows,
statistics, candidates, prompts, SID features); gradient samples are only the train queries of a seed-0 user subsample.
Users = the first k of `numpy.random.default_rng(0).permutation(sorted ids of users with >= 1 train query)`, k = the
smallest prefix whose number of train trajectories with >= 1 train query reaches tky H-100's 51,611 (LLM4POI's SFT
set size; the same users for GNPR-SID). Val/test: all queries, default candidates, so S-u results are scored on the
same query set as every other method. Results under S-u are labelled "subsampled training" (section sign in the
tables). Every other method runs on gowalla / foursquare_global under H-100 as usual. The export
`exports/<ds>/S-u/` equals `exports/<ds>/S-100/` except the train-query files (`tests/test_data.py check_su`).

## 6. Prediction format and evaluation (summary; the evaluator's docstring is authoritative)

* Predictions: `preds/<method>/<ds>/<split>.parquet` (`query_id`, `ranked_venue_ids` with >= 10 ids, optional
  scores) and `<split>.cost.json` (stages `name, wall_s, cpu_core_s, gpu_s, peak_mem_gb, one_time`).
* Metrics (per query, micro-averaged over queries): Acc@1, Acc@5, Acc@10, NDCG@5, NDCG@10, MAP@20 and MRR.
  MAP@20 follows GETNext / LSTPM (`utils.mAP_metric`): 1/rank if rank <= 20 else 0 (one relevant item), i.e. MRR
  truncated at 20. Two modes: **end-to-end** (`e2e`, primary: the method's own `ranked_venue_ids`, de-duplicated and
  truncated to the stored top-50, so MRR is computed over the top-50 list and approximates full-ranking MRR) and
  **within-candidate** (`cand`: the method's order over the first 20 retriever candidates of its policy; misses count;
  unranked candidates appended in retriever order; MRR exact over the 20). Strata: repeat vs explore (=
  first-visit; `in_user_history`), venue popularity (cold/tail/mid/head), history length terciles, candidate
  hit/miss, and `pos=last` / `pos=inner` (target is / is not the last check-in of its trajectory; `pos=last` is
  LLM4POI's evaluation unit). The `hist` and `hlen` strata always use the H-100 query fields, also when a method ran
  under H-x. Uncertainty: paired user-level bootstrap (B = 1000, seed 0, one resampling matrix shared by every
  method, mode and stratum of an evaluation call), 95% percentile intervals, two-sided bootstrap p-values. Only the
  pre-registered tests of 6.2 are hypothesis tests (Holm-corrected); all other differences are descriptive.
  See `poibench/evaluate.py`.
* Policy matching: candidate mode must use the candidate file of the policy the method ran under
  (`--policy <P>`; H-10/H-70 have their own files, S-x and H-100 share the default). The evaluator refuses a method
  whose `cost.json` names a different `candidates_file`. Comparisons of one method across H-x policies are
  end-to-end only (`--e2e-only`).
* Cost: one-time (fitting, indexing, fine-tuning) vs per-query (inference) stages, see `poibench/cost.py`;
  reported as totals and amortised per 1k queries.
* Cost of record: `results/cost_exclusive/<method>@<policy>/<ds>/` (cost-only re-runs holding their GPU
  exclusively) where it exists; otherwise the `preds/` cost.json, flagged in every table as possibly co-located
  (`results/cost_colocation/` sidecars, `reports/COLOCATION.tsv`). Incremental cost per additional correct top-1
  prediction (ICPC) is computed against the reference simple baseline (6.2: hybrid) on the test split.

### 6.1 Pre-specified analyses (fixed at freeze v1.0; all on the test split, validation shown for reference)

Run by `scripts/eval_all.py` (one evaluator call per dataset x policy over every finished prediction directory, plus
one e2e call per method across its policies) and rendered by `paper/scripts/make_tables.py`. Nothing else is
reported as a benchmark result; any further analysis is labelled **exploratory** in the paper and in reports.

| id | analysis | data | methods | metrics |
|---|---|---|---|---|
| A1 | main effectiveness table | 6 main datasets, H-100 | pop, repeat, markov, hybrid, GETNext (released; normfeat / fixed as labelled variants), Graph-Flashback, DS-LLM4POI (lcs primary; random where run), LLM4POI, GNPR-SID | all 7, e2e; Acc@1 and MRR also cand, next to retriever R@20 |
| A2 | repeat vs first-visit and last vs inner strata | as A1 | as A1 | Acc@1 and NDCG@10, e2e |
| A3 | data needs: H-x vs S-x | nyc, tky, istanbul (simple baselines also ca, sin, sao_paulo at H-10 / S-10) | every method with H-x / S-x runs | NDCG@10 (and Acc@1), e2e |
| A4 | full cost and ICPC | as A1 | as A1 | one-time and per-query CPU / GPU / memory; ICPC vs hybrid |
| A5 | multimodal | nyc, H-100 | VLM re-ranker arms A/B/C; MMPOI with / without images; GETNext as MMPOI's backbone reference | cand (VLM), e2e (MMPOI), all 7 |
| A6 | paper conditions vs unified protocol | nyc (target time), every main dataset (pos=last) | DS-LLM4POI lcs-tt and LLM4POI-tt vs default; every A1 method on pos=last vs all; strict vs lenient parse; trainfilter variant (simple baselines); GETNext fixed / normfeat | Acc@1 e2e |
| A7 | appendix | gowalla, foursquare_global, H-100 / H-10 / S-10 | simple baselines | all 7 |

### 6.2 Confirmatory tests (Holm-Bonferroni within each family, alpha 0.05)

Defined in code by `poibench.evaluate.confirmatory_tests()` (hashed by the freeze line) and written to
`results/confirmatory_test.csv` with raw and Holm-adjusted bootstrap p-values. Reference simple baseline = hybrid
(highest validation e2e Acc@1 under H-100 among the simple baselines on all eight datasets, fixed before any test
metric). A test whose predictions do not exist when the paper is built is reported as not run and is not counted in
its family.

| family | tests | comparison (difference a - b) | metric |
|---|---|---|---|
| Q1 | 66 | per main dataset (H-100): each LLM method (DS-LLM4POI lcs, LLM4POI, GNPR-SID) vs hybrid, vs GETNext, vs Graph-Flashback; GETNext and Graph-Flashback vs hybrid | e2e Acc@1, all |
| Q2 | 39 | nyc / tky / istanbul: trained methods H-10 vs H-100 and H-10 vs S-10; repeat / markov / hybrid H-10 vs H-100 | e2e NDCG@10, all |
| MM | 3 | nyc H-100: VLM arm B vs A and C vs A (cand Acc@1); MMPOI vs MMPOI-noimage (e2e Acc@1) | as stated |
| FLIP | 2 | nyc H-100: DS-LLM4POI lcs-tt vs lcs; LLM4POI-tt vs LLM4POI | e2e Acc@1, all |

### 6.3 Exploratory analyses
Anything not in 6.1 / 6.2 (e.g. other strata, other metrics as test targets, per-category breakdowns, analyses
suggested by the test results) is exploratory: reported with CIs only, labelled as such, and never used to change
a method, a policy or the protocol.

## 7. Methods

Each method entry must list: input information (own history / population / text / images), processing (none,
graph, retrieval, fine-tuning, prompting), predictor, the hyperparameter source (paper default or LOG.md entry),
and its mapping onto the evidence policy (which component reads which rows).

**Leakage gate (review H1).** Every adapter exposes `predict(export_dir, split, query_ids, work_dir)` (or a
`cmd:` template) and must PASS `scripts/canary.py --entry ... --ds <ds> --policy <P>` on 300 val queries (seed 0;
greedy decoding for LLMs) for H-100 and for one H-x policy before its first real run: the predictions must be
identical when every hidden pre-cutoff row and every val/test row outside the queries' own prefixes (including
their targets) is replaced by another venue and shifted in time; the perturbed rebuild must leave the export
unchanged where the queries may look; and the source lint (reads of `data/`, target columns) must be clean.
Reports go to `logs/canary/`.

**Simple baselines** (`pop`, `repeat`, `markov`, `hybrid`; `poibench/baselines_simple.py`, LOG 2026-10-27 12:20 UTC).
Parameter-free counting rules over the policy's visible rows: popularity; the user's visible own venues by frequency
(recency ties); first-order transitions from the previous venue with popularity back-off; hybrid = equal-weight
(0.5/0.5, fixed a priori) mixture of the user's visit distribution and the Markov row. Every list ends with the
retriever's popularity tie rule; the same order re-ranks the candidates. S-x = H-100 for them (no training samples).
`methods/export_hybrid/run.py` reproduces `hybrid` from the exports alone (reference adapter for the canary).

**GETNext** (`getnext`, Yang et al. SIGIR 2022; card `methods/getnext/CARD.md`, adapter g2, LOG 2026-10-27 "GETNext
adapter owner" and later g2 entries). Input: the query's own trajectory prefix, a learned user embedding, and
population evidence (global transition graph and node features: visit count, category, coordinates of the visible
rows). Processing: GCN over the transition graph + node-attention map; predictor: transformer trained from scratch
(authors' param_parser defaults: 200 epochs, batch 20, lr 1e-3), checkpoint = best `4*Acc@1 + Acc@20` on validation
via `val_feedback`. Released behaviour (attention over the batch axis, node-order misalignment emulated with a fixed
seed-0 permutation) is the primary configuration; labelled variants `getnext-fixed` (time-axis attention; nyc) and
`getnext-normfeat` (node feature log(1 + count); every dataset except nyc) run next to it, never instead. Attention
on the MATH SDPA backend (g2: the fused kernel overflows with raw counts). Policy mapping: H-x = graph, counts,
sequences and labels from the H-x visible rows; S-x = graph/sequences from all train rows, gradient labels only at
the H-x train queries. No target time (its time loss uses train labels only).

**Graph-Flashback** (`graph-flashback`, Rao et al. KDD 2022; card `methods/graph_flashback/CARD.md`, LOG 2026-10-27
14:25 UTC). Input: own history (last <= 20 visible own check-ins: history + own prefix) and population (KGE graphs).
Processing: TransE knowledge-graph embedding over interact / temporal triplets of the visible rows and spatial
triplets of the public catalog, then a top-20 POI-POI and a top-100 user-POI graph; predictor: the Flashback RNN
(trained from scratch). Hyperparameters: authors' foursquare defaults; deviations min_checkins 21 (code 101), batch
= trained users - 1 when <= 512 users, loss only at train-query labels, val-Acc@1 checkpoint every 5 epochs. Policy
mapping: H-x = triplets, TransE, both graphs, training windows and labels from the H-x visible rows only (public
catalog shared); S-x = KGE / graphs / input windows from all train-period rows (shared with H-100), gradient updates
only at the H-x train queries. No target time. The canary runs both on predict() and on the full pipeline (retrained
from the perturbed export).

**DS-LLM4POI** (`ds-llm4poi-<cfg>`, Nishida et al. arXiv 2604.06207; card `methods/ds_llm4poi/CARD.md`, LOG 2026-10-27
14:32 UTC). Input: own history only (the user's own visible trajectories as demonstrations + the query's own prefix);
no population evidence, no candidates in the prompt. Processing: prompting (zero-shot in-context learning); demonstration
selection by LCS of the prefix against each bank trajectory (cfg `lcs`, primary, paper recommendation) or random (cfg
`random`, code default), k = 5, plus our implementation of the missing similarity merge. Predictor: Qwen2.5-7B-Instruct
(vLLM, greedy, max_tokens 2000), one answer parsed leniently (authors' strict parse as the `-strictparse` twin);
`ranked_venue_ids` = [answer] + the policy's common-retriever candidates. Hyperparameters: authors' code defaults /
paper. Policy mapping: H-x = demonstrations from the H-x visible rows only, each ending in an H-x train query; S-x =
demonstrations from all train-period rows but only those ending in the H-x train-query set (H-100 = S-100 prompts).
No target time (patch 0001); the labelled ablation `lcs-tt` reads it from `exports_ablation/<ds>/` only.

**LLM4POI v2** (`llm4poi`, Li et al. SIGIR 2024, v2 recipe; card `methods/llm4poi/CARD.md`, LOG 2026-10-27 15:26
UTC). Input: own history only (the user's last <= 50 visible check-ins before the current trajectory + the query's own
prefix); no population evidence at inference, no candidates in the prompt; population = the SFT samples. Processing:
full fine-tuning of Qwen2.5-7B-Instruct with ms-swift (authors' v2/sft.sh flags: 3 epochs, lr 1e-5, effective batch
1,024, ZeRO-3), one sample per training trajectory (its last training query); predictor: the fine-tuned model, greedy,
one answer parsed leniently (authors' exact-string parse as the `-strictparse` twin); `ranked_venue_ids` = [answer] +
the policy's common-retriever candidates. Hyperparameters: authors' sft.sh / eval defaults, swift 3.10.2 defaults
elsewhere; final step, no checkpoint selection. Policy mapping: H-x = SFT samples from the H-x train queries with
history / prefix truncated to visible rows, query prompts from visible rows; S-x = all train-period rows visible,
SFT samples restricted to the trajectories of the H-x train-query set. No target time (template clause removed); the
labelled ablation `llm4poi-tt` (J61) keeps the authors' original prompt and reads val/test times from
`exports_ablation/<ds>/` only.

**qwen2.5-vl-ref** (multimodal module, our reference design; `qwen2.5-vl-ref-{A,B,C}`; card
`methods/qwen25_vl_ref/CARD.md`, audit `reports/MMPOI_AUDIT.md`, LOG 2026-10-27 15:12 and 15:21 UTC). Input: own
history (last 20 visible own check-ins: history + own prefix), public catalog (category, coordinates), the policy's
first 20 common-retriever candidates, and public venue photos (MMPOI, one per venue, `exports_mm/<ds>/`, built by
`scripts/mmpoi_audit.py`; static, policy-independent). No names (none exist), no population statistics, no target
time. Processing: zero-shot prompting of Qwen2.5-VL-7B-Instruct (vLLM, greedy) to rank the 20 candidates (shown in a
seed-0 per-query permutation); paired arms on identical queries / candidates / text: A text only, B + a caption of
each shown photo (same VLM, LLM-Augmented-POIRec photo-summary prompt; one-time caption cost in B's cost.json),
C + the photos (<= 4 history + <= 20 candidates). Primary output `cand_ranked_venue_ids` (within-candidate mode;
unranked candidates appended in retriever order); `ranked_venue_ids` = that + candidates 21-50. Runs: nyc H-100 only.

**MMPOI** (`mmpoi`, `mmpoi-noimage`; Xu et al. WWW 2024; card `methods/mmpoi/CARD.md`, LOG 2026-10-30 12:53, 15:20,
15:25 UTC). GETNext-style transformer whose POI representation fuses pre-computed modality embeddings: venue
category (meta), all tips of the venue (review) and captions of all photos of the venue (image), encoded with the
encoders identified by fingerprinting the authors' released NYC vectors (all-mpnet-base-v2; BLIP-2 OPT-2.7b
captions; gate V-mmpoi: cos >= 0.99 vs the released vectors). Released-code behaviour and defaults kept (lr 1e-4,
batch 256, 200 epochs, early stopping on validation via `val_feedback`); `mmpoi-noimage` = identical code path with
every image vector set to the authors' zero sentinel. Runs: nyc H-100 only (photos and tips exist for NYC only).
Venue photos / tips are static, policy-independent public content crawled for the 2024 paper (not label leakage;
not redistributed).

**GNPR-SID** (`gnpr-sid`, Wang et al. KDD 2025, V2 code; card `methods/gnpr_sid/CARD.md`, LOG 2026-10-27 15:33 /
17:06 UTC). Input: own history (the last 49 visible own check-ins written as semantic IDs, local times; user id) and
population (SID codebook features: catalog category text, coordinates, hour histogram of visible check-ins; LoRA SFT
on visible training sequences). Processing: SID quantisation (CRQVAE, authors' V2 defaults) + LoRA fine-tuning of
Meta-Llama-3-8B-Instruct (authors' values: 5 epochs, lr 2e-5, effective batch 16). Predictor: deterministic beam
search (10) constrained to the SID trie; `ranked_venue_ids` = beam venues + policy-candidate fill (primary),
`cand_ranked_venue_ids` = the 50 policy candidates by length-normalised SID log-probability. Policy mapping: H-x = SID
hour features, SFT input rows and targets from the H-x visible rows only (catalog public); S-x = features and inputs
from all train-period rows (= H-100 SIDs), SFT targets only at the H-x train-query set. No target time (the authors'
"When {time}" clause removed). Final SFT model, no val-based selection.

## 8. Sanity checks done on validation only

Data invariants and leakage tests (`tests/test_data.py`), candidate recall (section 4), dataset statistics
(`data/DATA_CARD.md`), simple baselines under H-100 / H-10 / S-10 in both modes (`reports/REPORT_simple_val.md`,
2026-10-27), every finished method on validation with the evaluator of this version (`scripts/eval_all.py --split
val`, `results/<ds>/val/paper_*`, 2026-10-31; used only for the reference choice of 6.2 among the simple baselines
and for sanity checks). No test-split metric was computed before the freeze line below was committed.

## 9. Freezing

Freeze = replace the "Status" line at the top (the first line starting with `Status:`, within the first 10 lines)
by exactly

    Status: FROZEN v<version> <YYYY-MM-DD> data=<sha256> eval=<sha256> git=<commit>

as printed by `python -I scripts/hash_tree.py --freeze-line <version>` after committing everything: `data` = sha256
of the SHA256SUMS text of every `.parquet`/`.json` under `data/` (also written to `data/SHA256SUMS`), `eval` =
sha256 of `poibench/evaluate.py`, `git` = the commit. The evaluator verifies all three on every test-split call;
any other wording (e.g. `**Status: FROZEN**`, `- Status: FROZEN ...`) keeps the test split locked, and any later
change of a data file or of the evaluator re-locks it. After freezing, any change is a new version with a
changelog entry and test results are re-reported.

## 10. Decisions at freeze (v1.0, 2026-10-31; formerly "open decisions")

1. Evidence for test queries: **static** (LLM4POI), as implemented in `poibench.data.evidence_slices` and used by
   every component; test queries do not see the user's val-period check-ins. (It is the only implementation and
   every run used it.)
2. Exact-reproduction variants `nyc_getnext` / `tky_getnext`: **not built** this round. Comparability with published
   numbers is discussed, not claimed (section 2 notes); the protocol-flip analysis (6.1 A6) quantifies the main
   differences instead.
3. Istanbul coordinate-less venues: **dropped** (as built; 1.1). Reported as a limitation.
4. Random-sample S-x variant: **not run** this round (S-x = latest-sample set as H-x, section 5).
5. Retriever recall: reported next to every within-candidate number (tables A1, A5).
6. Strata: as in section 6 (`pos=last` added 2026-10-27); confirmed.
7. LLM primary outputs: single answer + policy-candidate fill (DS-LLM4POI, LLM4POI), constrained beam + fill
   (GNPR-SID), candidate ranking (VLM re-ranker); fixed before their first GPU runs (LOG entries cited in section 7).
8. Trainfilter variants: simple baselines only (H-100, val + test; LOG 2026-10-31); no GPU method runs on them.
9. Residual id side channel of the exports: documented rule only (3.1); not re-keyed this round.

## Changelog
* v1.0.1 (2026-11-08, FROZEN; amendment after the release review of 2026-11-08, LOG entry of that date): one input
  check in `poibench/evaluate.py` only: `evaluate_files` refuses prediction files whose `<split>.cost.json` names a
  mock (CPU stand-in) backend (`mock_artifacts`: `backend`, `model_fingerprint` or `generation.model_fingerprint`
  starting with "mock"). Scoring code unchanged: no existing prediction file is refused, and an A/B run of the v1.0 and
  v1.0.1 evaluators on every prediction file (6 cities x val/test x all policies) gave identical results. No data or
  candidate file changed (data= unchanged); eval= is the sha256 of the amended file.
* v1.0 (2026-10-31, FROZEN): metrics as implemented (Acc@1/5/10, NDCG@5/10, MAP@20, MRR over the stored top-50
  in e2e mode); pre-specified analyses (6.1), confirmatory families with Holm correction (6.2, code in
  `poibench.evaluate.confirmatory_tests` / `holm`), exploratory rule (6.3); method entries for the simple baselines,
  GETNext and MMPOI (7); decisions at freeze (10). No data or candidate file changed since v0.3b.
* v0.3b (2026-10-28, still DRAFT): main study = nyc, tky, ca, sin, istanbul, sao_paulo; gowalla / foursquare_global
  appendix only (CPU simple baselines); S-u kept but unused.
* v0.3a (2026-10-28, still DRAFT): policy S-u (5.1) for the LLM fine-tunes on gowalla / foursquare_global.
* v0.3 (2026-10-28, still DRAFT): datasets ca / sin / sao_paulo (1.2, 1.3) and gowalla / foursquare_global (1.4, 1.5);
  main (nyc, tky, ca, sin, gowalla, foursquare_global) vs supplementary (istanbul, sao_paulo) datasets (section 1).
  No data or candidate file of an existing dataset changed.
* v0 (2026-10-27): first draft; data built with retriever r1.
* v0.1 (2026-10-27, integration, still DRAFT): section 6 states MRR cut-offs, the `pos` stratum and the
  policy-matching rule (now enforced by the evaluator); section 8 adds the simple-baseline val report; open item 1
  updated (single evidence implementation), item 6 resolved, item 7 added. No data or candidate file changed.
* v0.2 (2026-10-27, review fixes, still DRAFT): adapter inputs = exports (3.1) and the leakage gate (7) (H1);
  disclosure of the future-count frequency filter and the trainfilter variants (2, M1); public venue catalog (2, M3);
  exact freeze line with verified hashes (9, L2); git-tracked pre-registration (0, M2). No data or candidate file
  of nyc/tky/istanbul changed.
