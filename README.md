# POIBench

POIBench is a benchmark for next point-of-interest (POI) prediction that asks when a large language model (LLM) is
worth it. It re-runs twelve methods from their official code, with the authors' defaults, on six city check-in
datasets under one protocol: a chronological split, history-retention and training-sample budgets that vary
separately, a common candidate-generation stage (the common retriever), the request time as an explicit switch (off by
default; labelled "-TT" variants switch it on), leakage canaries that every adapter must pass before its first run, and
CPU, GPU and memory accounting for every pipeline stage. An exploratory efficiency and scalability study measures
training, serving, maintenance and storage cost on top of the same runs.

This repository contains the protocol, the data-building and export code, the common retriever, the frozen evaluator
and its statistics, the leakage canary, the cost recorder, the adapters of all methods, the efficiency scripts and the
generators of the paper's tables. It contains **no data, no predictions and no upstream method code** (see
[Data](#data) and [Upstream code](#upstream-code-and-licences)).

The full protocol is in [`PROTOCOL.md`](PROTOCOL.md).

## Methods

| method | family | adapter | upstream (pinned) |
|---|---|---|---|
| Pop, Repeat, Markov, Hybrid | parameter-free counting baselines | `poibench/baselines_simple.py`, `scripts/run_simple.py` | - |
| GETNext | graph + transformer | `methods/getnext/` | songyangco/GETNext |
| Graph-Flashback | knowledge-graph embedding + RNN | `methods/graph_flashback/` | kevin-xuan/Graph-Flashback, kevin-xuan/KGE |
| DS-LLM4POI | LLM, in-context (Qwen2.5-7B-Instruct, frozen) | `methods/ds_llm4poi/` | ryonsd/DS-LLM4POI |
| LLM4POI | LLM, full fine-tuning (Qwen2.5-7B-Instruct) | `methods/llm4poi/` | neolifer/LLM4POI |
| GNPR-SID | LLM with semantic IDs (Llama-3-8B-Instruct, LoRA) | `methods/gnpr_sid/` | wds1996/GNPR-SID |
| GA-LLM (exploratory, added after the protocol freeze) | LLM with geo-encoder (QLoRA) | `methods/gallm/` | liuzh368/GA-LLM |
| MMPOI | multimodal transformer | `methods/mmpoi/` | zzmylq/MMPOI |
| VLM re-ranker, arms A/B/C | Qwen2.5-VL-7B-Instruct, frozen | `methods/qwen25_vl_ref/` | caption prompt from wzehui/LLM-Augmented-POIRec |
| export_hybrid | reference adapter (reproduces Hybrid from the exports alone) | `methods/export_hybrid/` | - |

## Repository layout

| path | content |
|---|---|
| `PROTOCOL.md` | the benchmark protocol (datasets, preprocessing, split, evidence policies, retriever, metrics, analyses) |
| `poibench/data.py` | raw loaders, preprocessing, chronological split, trajectories, queries, evidence policies, common retriever (r1) |
| `poibench/exports.py` | adapter inputs `exports/<ds>/<policy>/` (no targets; the only files an adapter may read) |
| `poibench/evaluate.py` | frozen evaluator: metrics, modes, strata, paired user bootstrap, confirmatory tests with Holm correction, test lock |
| `poibench/canary.py` | leakage canary (perturbation test, exporter invariance, source lint) |
| `poibench/cost.py`, `poibench/exclusive.py` | per-stage cost recorder (wall, CPU core, GPU, memory; one-time vs per-query) and GPU-exclusivity monitor |
| `poibench/baselines_simple.py` | Pop, Repeat, Markov, Hybrid (and Hybrid-TT) |
| `poibench/hashing.py` | sha256 helpers of the freeze line |
| `methods/<method>/` | adapters; `patches/` holds patch files for upstream code |
| `scripts/` | data building, export, simple baselines, canary, evaluation, statistics, multimodal assets, model download, upstream fetch |
| `scripts/efficiency/` | exploratory efficiency / scalability study (E1-E8) and its paper outputs |
| `scripts/gpu_jobs/` | canary gates per adapter (`canary_gate*.sh`) and the GPU cost-recorder check |
| `paper/scripts/` | generators of the paper's tables, number macros and figures |
| `tests/` | unit tests on synthetic data (plus checks of built data when present) |
| `envs/` | pip requirement files of the four Python environments |
| `data_raw/RAW_SHA256.tsv` | sha256 of every raw file the build reads |

## Installation

Four Python environments were used (Linux, CUDA GPUs). Each `envs/<name>.txt` is the complete `pip freeze` of the
environment the runs used; `envs/req_<name>.txt` lists the direct requirements it was built from.

| variable | Python | requirements | used for |
|---|---|---|---|
| `PY_BASE` | 3.12.15 | `envs/base.txt` (torch 2.13, vLLM 0.30, transformers 5.20) | data, exports, retriever, evaluator, simple baselines, canary, GETNext, Graph-Flashback, DS-LLM4POI, MMPOI, VLM re-ranker, GA-LLM data preparation, efficiency scripts, paper tables |
| `PY_LLM4POI` | 3.12 | `envs/llm4poi.txt` (ms-swift 3.10.2, deepspeed, vLLM 0.11) | LLM4POI |
| `PY_GNPR` | 3.12 | `envs/gnpr_sid.txt` (transformers 4.49, trl 0.11, vLLM 0.7.3) | GNPR-SID |
| `PY_GALLM` | 3.9.20 | `envs/gallm.txt` (torch 2.4 cu121, flash-attn 2.6.3, bitsandbytes) | GA-LLM training and inference (called by `methods/gallm/run.py`) |

```bash
conda create -y -p envs/py_base -c conda-forge python=3.12.15 pip
envs/py_base/bin/pip install -r envs/base.txt
export PY_BASE=$PWD/envs/py_base/bin/python
# likewise envs/llm4poi.txt -> PY_LLM4POI, envs/gnpr_sid.txt -> PY_GNPR, envs/gallm.txt (Python 3.9) -> PY_GALLM
```

The data building, exports, simple baselines, evaluator, canary, unit tests and paper tables run on CPU only.
GPU methods were run on NVIDIA H100 NVL GPUs (one machine type with two H100 NVL GPUs and an AMD EPYC 9V84 CPU);
LLM4POI's full fine-tuning uses both GPUs (DeepSpeed ZeRO-3).

### Environment variables

| variable | default | meaning |
|---|---|---|
| `POIBENCH_RAW_ROOT` | `data_raw/` | raw datasets (layout below) |
| `POIBENCH_MODEL_DIR` | `models/` for GNPR-SID; must be set for the other LLM adapters | local model copies `<dir>/<repo name>/` with a `.poibench_complete` marker (written by `scripts/download_models.py`) |
| `POIBENCH_MODEL_DIR_X` | unset | optional second model directory (backbone-scaling study) |
| `POIBENCH_CACHE` | `work/cache/` | local cache of photos for the multimodal methods |
| `POIBENCH_LLM4POI_CKPT_ROOT`, `POIBENCH_GALLM_CKPT_ROOT` | `work/<method>/ckpt/` | checkpoint roots |
| `PY_GALLM` | `python` | interpreter of the GA-LLM environment |
| `POIBENCH_RUNTIME_DIR`, `POIBENCH_SCRATCH` | `work/runtime/`, `work/efficiency_scratch/` | storage-footprint inputs and scratch of the efficiency scripts |

All commands below run from the repository root with `python -I` (isolated mode), as in the experiments.

### Upstream code

```bash
bash scripts/fetch_upstream.sh          # clones every upstream repository at its pinned commit into third_party/
```

The adapters never edit `third_party/`: GETNext, GNPR-SID and MMPOI files are imported by path after a sha256 check;
Graph-Flashback / KGE, DS-LLM4POI and GA-LLM are copied to `work/` and patched there (patch files in
`methods/*/patches/`, patch strings in `methods/gallm/run.py`); LLM4POI and LLM-Augmented-POIRec are read for prompt
templates and configuration files.

### Model weights

Download the backbones at the revisions pinned in the adapters into `$POIBENCH_MODEL_DIR` (gated models need a
Hugging Face account that accepted the model licence; the script uses your saved `hf auth login` token or `HF_TOKEN`
and only redirects the download cache, to `dl_cache/` next to the model directory, which is git-ignored):

```bash
export POIBENCH_MODEL_DIR=$PWD/models
$PY_BASE -I scripts/download_models.py "$POIBENCH_MODEL_DIR" \
  Qwen/Qwen2.5-7B-Instruct@a09a35458c702b33eeacc393d103063234e8bc28 \
  meta-llama/Meta-Llama-3-8B-Instruct@8afb486c1db24fe5011ec46dfbe5b5dccdb575c2 \
  Qwen/Qwen2.5-VL-7B-Instruct@cc594898137f460bfe9f0759e9844b3ce807cfb5 \
  Yukang/Llama-2-7b-longlora-32k-ft@ab48674ffc55568ffe2a1207ef0e711c2febbaaf
```

MMPOI's text and caption encoders (`sentence-transformers/all-mpnet-base-v2`, `Salesforce/blip2-opt-2.7b`) are
fetched by `methods/mmpoi/run.py embed` at their pinned revisions. The backbones of the exploratory model-size study
are listed in `methods/ds_llm4poi/llm.py` (`MODELS`), `methods/llm4poi/run.py` (`BASES`) and `methods/gnpr_sid/llm.py`
(`BACKBONES`).

## Data

No dataset is included. Download the raw files from their providers, accept their terms of use, and place them under
`data_raw/` (or `$POIBENCH_RAW_ROOT`, or `scripts/build_data.py --raw_root <dir>`: one directory that holds every raw
file in the layout below); `data_raw/RAW_SHA256.tsv` lists the sha256 of every file the build reads.

| dataset key | source | files under `data_raw/` |
|---|---|---|
| `nyc`, `tky` | Foursquare TSMC2014 (Yang et al.), `dataset_tsmc2014.zip` from the dataset page of Dingqi Yang (https://sites.google.com/site/yangdingqi/home/foursquare-dataset) | `tsmc2014/dataset_tsmc2014.zip` and its contents in `tsmc2014/extracted/dataset_tsmc2014/` |
| `istanbul`, `sao_paulo` | Massive-STEPS, Hugging Face datasets `CRUISEResearchGroup/Massive-STEPS-Istanbul` and `CRUISEResearchGroup/Massive-STEPS-Sao-Paulo`, configuration `tabular` | `massive_steps_istanbul/{train,validation,test}.parquet`, `massive_steps_sao_paulo/{train,validation,test}.parquet` |
| `ca` | Gowalla check-ins (SNAP, https://snap.stanford.edu/data/loc-gowalla.html) + Gowalla spot metadata of Liu et al. (`gowalla_spots_subset1.csv`, e.g. figshare doi:10.6084/m9.figshare.22126586) + the California / Nevada polygons of LLM4POI's `datasets.zip` (link in the LLM4POI README) | `gowalla_snap/loc-gowalla_totalCheckins.txt.gz`, `gowalla_liu/gowalla_spots_subset1.csv`, `llm4poi_release/ca/us_state_polygon_json.json` |
| `sin` | Foursquare global-scale check-ins (TIST 2015), `dataset_TIST2015.zip` from Dingqi Yang's dataset page | `foursquare_tist2015/dataset_TIST2015.zip`, `foursquare_tist2015/extracted/dataset_TIST2015_{Checkins,POIs}.txt` |
| `gowalla` (efficiency study / appendix) | as `ca` (SNAP + Liu et al.), plus a spot time-zone table built by `scripts/gowalla_venue_tz.py` | `gowalla_global_derived/venue_tz.parquet` (generated) |
| `foursquare_global` (efficiency study / appendix) | Foursquare WWW 2019 release, `dataset_WWW2019.zip` from Dingqi Yang's dataset page | `foursquare_www2019/dataset_WWW2019.zip`, `foursquare_www2019/extracted/raw_{Checkins_anonymized,POIs}.txt` |
| multimodal (NYC) | MMPOI release archives (Google Drive links in the MMPOI README) | `mmpoi/archives/NYC_WWW2024.zip`, `mmpoi/processed_dl/NYC_processed.zip` |

`scripts/gowalla_venue_tz.py` needs `timezonefinder==6.5.2`, installed into `envs/tzlib`:
`pip install --target envs/tzlib timezonefinder==6.5.2` (then remove the `numpy*` and `setuptools*` directories it
pulls in, so that the environment's own numpy is used).

### Build the datasets and the adapter inputs (CPU)

```bash
$PY_BASE -I scripts/gowalla_venue_tz.py                              # only for the gowalla dataset
OMP_NUM_THREADS=1 $PY_BASE -I scripts/build_data.py --datasets nyc tky istanbul ca sin sao_paulo --workers 24
$PY_BASE -I scripts/build_data.py --datasets gowalla foursquare_global nyc_trainfilter tky_trainfilter   # optional
$PY_BASE -I scripts/export_inputs.py --datasets nyc tky istanbul ca sin sao_paulo     # exports/<ds>/<policy>/
$PY_BASE -I scripts/export_target_time.py --ds nyc tky               # only for the labelled -TT variants
```

`build_data.py` writes `data/<ds>/` (check-ins, venues, queries per split, evidence policies H-x / S-x for
x in {10, 70, 100}, common-retriever candidates per policy) and `data/DATA_CARD.md`. The common retriever (version r1:
time-decayed user history, transitions from the previous venue and geo-popular venues, fused by reciprocal rank
fusion; 50 ranked candidates, the first 20 form the main candidate set) is part of this step (`poibench.data.retrieve`).
`export_inputs.py` writes, per dataset and policy, the only files an adapter may read: visible rows, public venue
catalog, per-query inputs without targets, per-query context, candidates and training labels.

Multimodal assets (NYC; photos and tips of the MMPOI release, policy-independent):

```bash
$PY_BASE -I scripts/mmpoi_audit.py inventory && $PY_BASE -I scripts/mmpoi_audit.py coverage
$PY_BASE -I scripts/mmpoi_audit.py assets      # exports_mm/nyc/ (one photo per venue, VLM re-ranker)
$PY_BASE -I scripts/mmpoi_photos_all.py --ds nyc && $PY_BASE -I scripts/mmpoi_texts.py --ds nyc   # MMPOI inputs
```

## Running the methods

Every run uses seed 0 and the authors' default hyper-parameters (nothing is tuned); policies are `H-100` (default),
`H-10`, `H-70` (history retention) and `S-10`, `S-70` (training-sample budget). Predictions go to
`preds/<method>@<policy>/<ds>/<split>.parquet` with a per-stage cost record `<split>.cost.json`.

**Leakage canary first.** Every adapter must pass its canary on 300 NYC validation queries for `H-100` and `H-10`
before its first real run:

```bash
$PY_BASE -I scripts/canary.py --self-test --ds nyc --policy H-10      # clean built-in passes, leaky control fails
$PY_BASE -I scripts/canary.py --entry methods/export_hybrid/run.py:predict --ds nyc --policy H-10 --split val --n 300
GATE_PY=$PY_BASE bash scripts/gpu_jobs/canary_gate.sh getnext       # likewise graph_flashback, ds_llm4poi
GATE_PY=$PY_LLM4POI bash scripts/gpu_jobs/canary_gate.sh llm4poi
GATE_PY=$PY_GNPR bash scripts/gpu_jobs/canary_gate.sh gnpr_sid
GATE_PY=$PY_BASE PY_GALLM=$PY_GALLM bash scripts/gpu_jobs/canary_gate.sh gallm
GATE_PY=$PY_BASE bash scripts/gpu_jobs/canary_gate_variant.sh getnext zfeat   # labelled variants (--cfg)
GATE_PY=$PY_BASE GATE_SKIP_PREP=1 DSLLM4POI_BACKEND=mock bash scripts/gpu_jobs/canary_gate_variant.sh ds_llm4poi random
$PY_BASE -I methods/mmpoi/gate.py                                    # MMPOI (image and noimage)
$PY_BASE -I methods/qwen25_vl_ref/gate.py --backend vllm             # VLM re-ranker arms A/B/C
```

The canary replaces every hidden pre-cutoff row and every validation / test row outside the queries' own prefixes
(including their targets) by another venue at another time, rebuilds the export and requires identical predictions,
an unchanged export where the queries may look, and a clean source lint. Reports go to `logs/canary/`. The variant
gate sets the adapter's own configuration variable (`GETNEXT_CFG`, `GNPR_SID_CFG`, `DSLLM4POI_CFG`, `LLM4POI_CFG`,
`GALLM_CFG`; other methods are refused) and fails unless the report's `adapter_config` shows that configuration in
both runs. Mock backends (`*_BACKEND=mock`, `--backend mock`) are CPU stand-ins for such checks: their outputs stay
under `work/`, and the evaluator refuses any prediction file whose cost record names a mock backend.

**Simple baselines (CPU).**

```bash
for sp in val test; do $PY_BASE -I scripts/run_simple.py nyc $sp H-100; done       # pop, repeat, markov, hybrid
$PY_BASE -I scripts/run_simple.py nyc test H-100 --methods hybrid-tt               # exploratory, needs the target-time file
```

**Adapters (GPU).** One command per dataset and policy; `--splits val test` trains / selects on validation and
predicts both splits.

```bash
$PY_BASE -I methods/getnext/run.py --ds nyc --policy H-100 --seed 0 --splits val test            # --cfg zfeat: GETNext-z
$PY_BASE -I methods/graph_flashback/run.py prepare --ds nyc --policy H-100 --seed 0 --device cpu  # KGE graphs (CPU)
$PY_BASE -I methods/graph_flashback/run.py all --ds nyc --policy H-100 --seed 0 --split val test --device cuda
$PY_BASE -I methods/ds_llm4poi/run.py all --ds nyc --policy H-100 --seed 0 --split val test --cfg lcs --device cuda
$PY_LLM4POI -I methods/llm4poi/run.py --ds nyc --policy H-100 --seed 0 --splits val test         # --cfg tt: LLM4POI-TT
$PY_GNPR -I methods/gnpr_sid/run.py --ds nyc --policy H-100 --seed 0 --splits val test
PY_GALLM=$PY_GALLM $PY_BASE -I methods/gallm/run.py --ds nyc --policy H-100 --seed 0 --splits val test
$PY_BASE -I methods/mmpoi/run.py embed --ds nyc --device cuda                                    # modality embeddings
$PY_BASE -I methods/mmpoi/run.py all --ds nyc --policy H-100 --seed 0 --splits val test --device cuda   # --variant noimage
$PY_BASE -I methods/qwen25_vl_ref/run.py caption --ds nyc --device cuda                          # arm B captions
$PY_BASE -I methods/qwen25_vl_ref/run.py all --ds nyc --policy H-100 --arm A --seed 0 --split val test --device cuda
```

The DS-LLM4POI configurations are `lcs` (primary), `random` and `lcs-tt`; LLM4POI, GNPR-SID and GA-LLM have `default`
and `tt`. The `-TT` variants keep the authors' original prompts with the request time and read it only from
`exports_ablation/<ds>/` (`scripts/export_target_time.py`). Exclusive cost-only re-runs use `--cost-only` and write
to `results/cost_exclusive/`.

## Evaluation and statistics

```bash
$PY_BASE -I scripts/evaluate.py nyc val hybrid@H-100 getnext@H-100 --policy H-100     # one call, paired bootstrap
$PY_BASE -I scripts/eval_all.py --split val                       # every finished prediction set, all datasets
$PY_BASE -I scripts/eval_all.py --split test                      # after the freeze (see "Test lock")
$PY_BASE -I scripts/amend_confirmatory.py --split test            # confirmatory p-values with B = 100,000
$PY_BASE -I scripts/compose_tail.py && $PY_BASE -I scripts/eval_exploratory.py   # exploratory time-aware analysis
$PY_BASE -I scripts/llm_scaling_report.py                         # exploratory model-size study
```

Metrics: Acc@1/5/10, NDCG@5/10, MAP@20 and MRR, in end-to-end mode (the method's own ranked list) and
within-candidate mode (the method's order over the first 20 retriever candidates; candidate recall is reported next to
it). Strata: revisit vs first visit, venue popularity, history length, candidate hit / miss, last vs inner check-in of
a trajectory. Uncertainty: paired user-level bootstrap (B = 1000, seed 0) with 95% percentile intervals; the
pre-registered confirmatory tests (`poibench.evaluate.confirmatory_tests`) are Holm-corrected within each family;
everything else is descriptive or exploratory.

### Test lock

The evaluator refuses test-split metrics unless the first `Status:` line of `PROTOCOL.md` is an exact freeze line
whose hashes match the data, the evaluator file and a commit of the repository. To evaluate on test in your copy,
build the data, commit, and freeze it:

```bash
git add -A && git commit -m "freeze"
$PY_BASE -I scripts/hash_tree.py --freeze-line 1.0   # writes data/SHA256SUMS, prints "Status: FROZEN v1.0 ..."
# replace the "Status:" line of PROTOCOL.md by the printed line
```

## Cost accounting

`poibench/cost.py` records, for every stage of every pipeline, wall time, CPU core time, allocated GPU time (wall time
times GPUs held), busy GPU time (NVML sampling) and peak CPU / GPU memory, and marks each stage as one-time (fitting,
indexing, fine-tuning) or per-query (inference). `scripts/gpu_jobs/g0_cost_check.py` checks the GPU path of the
recorder. Tables report totals and cost per 1,000 queries, preferring exclusive cost-only re-runs
(`results/cost_exclusive/`) over the cost records of the main runs. Model loading is a one-time stage of its own,
timed before (never inside) the per-query inference stage. Answers taken from an adapter's response cache (a resumed
or repeated run) keep their generation cost as stage `infer_cached` (from the cache's per-answer times or, for GA-LLM,
the ledger of the execution that generated them); the run's own time is recorded under `execution`, and a re-run that
took answers from the cache never replaces an earlier record of the same work measured with fewer cache hits: however
many re-runs follow, the originally measured per-query stages and their provenance (`per_query_stages_from`) are
carried forward and each re-run is listed under `reruns` (the same holds for a re-run GA-LLM `--infer-shard` record).
A resumed GA-LLM generation reuses an interrupted attempt's answer only if it answers the same request (model, seed,
query, prompt and generation settings); otherwise the answer is generated again.

## Efficiency and scalability study (exploratory)

| script | measures |
|---|---|
| `scripts/efficiency/e1_hybrid_bench.py`, `e1_scaling.py` | cost against data size and visible evidence |
| `e2_freshness.py` | incremental updates, a Hybrid kept fresh, accuracy per week since the cutoff |
| `e3_model_size.py` | cost against backbone size |
| `e4_retriever.py` | cost and recall of the common retriever |
| `e5_serving.py <method>` | online serving latency and throughput on one GPU |
| `e6_*.py` | candidate-set size, large datasets, complexity, per-query cost vs history, stages and storage, fresh evidence without retraining (`e6_fresh_export.py` builds the fresh exports and runs their canary) |
| `e7_scale_build.py`, `e7_scale_cpu.py`, `e7_scale_report.py` | TKY user-fraction scalability |
| `e8_fresh_gallm.py` | GA-LLM with fresh evidence |
| `make_eff_paper.py` | figures, tables and number macros of the study |

## Regenerating the paper tables

After `eval_all.py` (and the exploratory scripts above) have written `results/`:

```bash
$PY_BASE -I paper/scripts/make_overview.py
$PY_BASE -I paper/scripts/make_tables.py            # --split val renders a stamped validation draft
$PY_BASE -I paper/scripts/make_extra.py
$PY_BASE -I paper/scripts/make_scale.py
$PY_BASE -I scripts/efficiency/make_eff_paper.py
```

Outputs go to `paper/tables/` and `paper/figures/`. `paper/scripts/paper_config.json` maps prediction directories to
table rows. Methods without results render as dashes.

## Tests

```bash
$PY_BASE -I tests/run_tests.py            # all unit tests; CPU only
$PY_BASE -I tests/run_tests.py test_evaluate test_canary
```

The tests use synthetic data (`tests/synth.py`); the adapter tests additionally need the upstream clones
(`scripts/fetch_upstream.sh`), and the checks of built data in `tests/test_data.py` run only when `data/` exists.

## Seeds and determinism

Seed 0 for every method and every random choice (sampling, initialisation, bootstrap); greedy decoding for the LLM
canaries; no hyper-parameter is tuned (authors' defaults; where none exist, a value fixed before any run).

## Upstream code and licences

No upstream repository is redistributed here. `scripts/fetch_upstream.sh` clones the pinned commits; see
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for the commits, the licence found in each repository and how
each is used. Repositories without a licence are used only through fetch scripts and patch files. Model weights and
datasets are subject to the licences and terms of their providers (e.g. the Llama 3 Community License for Meta Llama 3,
the Llama 2 Community License for the GA-LLM base model, and the terms of the Foursquare, Gowalla and Massive-STEPS
data releases).

## Licence

LICENSE: to be chosen by the authors.

## Citation

```bibtex
@inproceedings{poibench2027,
  title     = {TO BE ADDED},
  author    = {TO BE ADDED},
  booktitle = {TO BE ADDED},
  year      = {2027}
}
```
