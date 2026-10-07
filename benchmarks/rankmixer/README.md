# RankMixer GPU benchmark

Modest 1-epoch comparison, meant to fit in a few GPU-hours on one rented GPU. It is a public-data check of this implementation, not a reproduction of the paper's trillion-scale Douyin experiments.

Ali-CCP tables in sections 5–7 are a test-contaminated development study. The validation-first 5-seed run (2026-10-07, seeds 2025–2029, test unread) is section 8 and `results/rigor/`. That study supersedes sections 5–7 for claims about tokenization, gating, and residual pooling. Conversion there is impression-level joint conversion.

| Track | Models | Data |
| --- | --- | --- |
| Single-task | RankMixer, DCNv2, WuKong, DNN | [Criteo_x1](https://huggingface.co/datasets/reczoo/Criteo_x1) (BARS split, ~33.0M train rows) |
| Multi-task | MT-RankMixer, MMoE, PLE, ShareBottom | Ali-CCP click + impression-level conversion |

Shared settings: Adam, learning rate `1e-3`, embedding size 16, batch 8192, 1 epoch, seed 2025, no extra L2. RankMixer uses `T=8`, `D=64`, `L=2`, `k=2`. The paper's 100M model is `D=768`, `T=16`, `L=2`; do not compare these AUC numbers to Table 1 of the paper.

## 1. Criteo_x1

```bash
# From the repo root. The zip is about 2.9 GB.
mkdir -p data/Criteo
wget -O /tmp/Criteo_x1.zip \
  https://huggingface.co/datasets/reczoo/Criteo_x1/resolve/main/Criteo_x1.zip
unzip /tmp/Criteo_x1.zip -d data/Criteo
# Expect data/Criteo/Criteo_x1/{train,valid,test}.csv
# Label column name: label. 13 numeric fields I1-I13, 26 categorical fields C1-C26.
```

The first FuxiCTR launch also builds `feature_map.json` and the parquet cache under `data/Criteo/`. That preprocessing is one-time and can take longer than an epoch.

## 2. Ali-CCP

The existing multi-task configs in this repo only reference `tiny_mtl`. Ali-CCP is the public click/conversion set used for this comparison. The recorded run in section 5 used the PaddleRec public mirror, [https://paddlerec.bj.bcebos.com/datasets/aitm/](https://paddlerec.bj.bcebos.com/datasets/aitm/), which does not require Tianchi student verification. The steps below are an alternative path from Tianchi dataset 408.

1. Download `sample_skeleton_{train,test}.csv` and `common_features_{train,test}.csv` from https://tianchi.aliyun.com/dataset/408 (login required). There is no official validation split. The official test file stays the test split.
2. Join skeleton rows with common features and keep the 18 categorical ids used by the AITM / Torch-RecHub preprocessing:
   `101, 121, 122, 124, 125, 126, 127, 128, 129, 205, 206, 207, 216, 508, 509, 702, 853, 301`.
   A readable reference script is https://github.com/xidongbo/AITM/blob/main/process_public_dataset.py (it names the second label `purchase`; rename that column to `conversion`).
3. The recorded Ali-CCP cache follows that AITM script, not a sequential prefix. `process_public_dataset.py` sets `random.seed(2020)` and writes a processed training row to dev when `random.random() >= 0.9`, so validation is a random ~10% holdout with that fixed seed. The PaddleRec public mirror used for the tables below is the same kind of split. Write:

```text
data/AliCCP/AliCCP_x1/train.csv
data/AliCCP/AliCCP_x1/valid.csv
data/AliCCP/AliCCP_x1/test.csv
```

Header: `click,conversion,101,121,122,124,125,126,127,128,129,205,206,207,216,508,509,702,853,301`.

`min_categr_count` is 10 so rare ids do not dominate the embedding table. The full training file is on the order of 40 million rows.

Semantic ablation (not part of the default script):

```bash
cd model_zoo/multitask/MT_RankMixer
python run_expid.py --expid MTRankMixer_aliccp_semantic --gpu 0
```

That run groups user profile / item / shop-context into 3 tokens.

## 3. Launch

```bash
bash benchmarks/rankmixer/run_gpu_benchmark.sh 0          # all 8 runs
bash benchmarks/rankmixer/run_gpu_benchmark.sh 0 single   # Criteo only
bash benchmarks/rankmixer/run_gpu_benchmark.sh 0 multi    # Ali-CCP only
```

Metrics are appended to `model_config.csv` next to the config that was used. RankMixer writes `model_zoo/RankMixer/config.csv`. Baseline runs write `benchmarks/rankmixer/configs/<Model>/model_config.csv`.

## 4. Expected runtime

Rough single-GPU budget after preprocessing is cached, batch 8192, one epoch:

| Hardware | One Criteo run | One Ali-CCP run | All 8 runs |
| --- | --- | --- | --- |
| T4 / L4 | 40-80 min | 40-90 min | about 8-12 GPU-hours |
| A10 / 3090 | 25-50 min | 25-60 min | about 5-8 GPU-hours |
| A100 40GB | 15-30 min | 15-40 min | about 3-5 GPU-hours |

The dense RankMixer trunk here is small (`2 * k * L * T * D^2 ≈ 2.6e5` FFN parameters). Wall time is dominated by data loading and the embedding tables, so DCNv2 / WuKong / DNN land in the same band.

To stay inside about 3 GPU-hours on a T4:

- Run only the two headline pairs: RankMixer vs DCNv2, and MT-RankMixer vs PLE.
- Or cut each CSV to the first 4 million training rows and keep the full valid/test files. Say so in the result note; AUC will not match a full-epoch run.

```bash
mkdir -p /tmp/criteo_small && head -n 4000001 data/Criteo/Criteo_x1/train.csv > /tmp/criteo_small/train.csv
# Point train_data at that file in the dataset yaml, then rerun.
```

First-time csv preprocessing (vocabulary + parquet cache) can add another 30-90 minutes per dataset and is not included in the table above.

## 5. Recorded run (RTX 4090, 2026-10-05)

One full pipeline on 1 × NVIDIA GeForce RTX 4090 (AutoDL), commit `5bbf507`, no local code changes. Shared settings: 1 epoch, batch 8192, embedding size 16, Adam `lr=1e-3`, seed 2025, no L2, no dropout, AUC monitor, single seed, no tuning. Wall clock 12:15:54–12:57:40 CST (2505 s, about 41.8 min) for 9 tasks, all exit 0. Epoch time below is training only. The Criteo table below is still this 1-epoch, single-seed run. The Ali-CCP table is a preliminary single-seed result. The 3-seed numbers in section 6 superseded it inside the development study, and section 8 supersedes both for Ali-CCP claims. Full write-up: `docs/RankMixer_tech_report.md`.

Criteo_x1 test (BARS split, ~33M train rows). RankMixer is sequential, `T=8`, `D=64`, `L=2`, `k=2`, dense FFN.

| Model | Params | Test AUC | Test logloss | Epoch |
| --- | ---: | ---: | ---: | ---: |
| DCNv2 | 34,745,121 | 0.808973 | 0.442537 | 97 s |
| RankMixer | 33,656,481 | 0.808522 | 0.443048 | 114 s |
| DNN | 33,574,497 | 0.807165 | 0.444495 | 88 s |
| WuKong | 33,582,393 | 0.806889 | 0.444488 | 102 s |

RankMixer is about +0.0014 AUC over DNN and +0.0016 over WuKong, and −0.0005 versus DCNv2.

AliCCP_x1 test (~38M train rows, `min_categr_count=10`), from the PaddleRec mirror above, not the Tianchi original. Semantic groups: user `[101,121,122,124,125,126,127,128,129]`, item `[205,206,207,216]`, context `[508,509,702,853,301]` (`T=3`, `D=48`). Sequential is `T=8`, `D=64`.

| Model | click AUC | conv AUC | conv logloss | Mean AUC | Epoch |
| --- | ---: | ---: | ---: | ---: | ---: |
| MT-RankMixer semantic | 0.619737 | 0.640612 | 0.002067 | 0.630174 | 91 s |
| PLE | 0.621780 | 0.624989 | 0.002128 | 0.623385 | 82 s |
| MT-RankMixer sequential | 0.617890 | 0.627733 | 0.002098 | 0.622811 | 98 s |
| ShareBottom | 0.621760 | 0.619310 | 0.002200 | 0.620535 | 69 s |
| MMoE | 0.619928 | 0.618001 | 0.002148 | 0.618964 | 78 s |

On this single seed, semantic grouping has test conversion AUC 0.640612, about +0.016 over PLE, and the highest mean AUC. That conversion gap did **not** replicate under the 3-seed protocol in section 6. Treat the Ali-CCP rows above as a superseded preliminary table. Criteo stays 1 epoch and seed 2025.

## 6. Multi-seed and anti-collapse (recorded)

Both follow-ups used the existing Ali-CCP parquet cache (no `feature_map.json` rebuild), seeds 2025 / 2026 / 2027, at most 6 epochs, `early_stop_patience: 1`, and the same optimizer settings as section 5. `monitor: AUC` is the unweighted mean of click AUC and conversion AUC. Test metrics come from the restored best checkpoint. Sample standard deviation uses \(n-1\). Archived tables and the summary CSV are in `benchmarks/rankmixer/results/`. Figures are in `docs/img/rankmixer/`.

The multi-seed jobs on the server used expids `*_ms_s{seed}` (gated checkpoint `MTRankMixer_aliccp_semantic_ms_s2025.model`). Hyperparameters match `configs/multiseed/`. The repo script still names its templates `*_es`. Anti-collapse expids match the script: `MTRankMixer_aliccp_semantic_residual_s{seed}` and `MTRankMixer_aliccp_semantic_entropy_s{seed}` (commit `5627b6c`, `gate_entropy_reg: 0.01`).

Test mean AUC, 3 seeds:

| Model | click AUC | conv AUC | mean AUC |
| --- | --- | --- | ---: |
| Shared mean | 0.61824 ± 0.00257 | 0.63168 ± 0.00224 | 0.62496 ± 0.00240 |
| Residual gated pooling | 0.61815 ± 0.00197 | 0.63080 ± 0.00328 | 0.62447 ± 0.00150 |
| Entropy reg 0.01 | 0.61907 ± 0.00145 | 0.62844 ± 0.00868 | 0.62375 ± 0.00506 |
| PLE | 0.61994 ± 0.00039 | 0.62605 ± 0.00630 | 0.62300 ± 0.00333 |
| Original per-task gating | 0.61835 ± 0.00160 | 0.62734 ± 0.00416 | 0.62284 ± 0.00230 |

Original per-task gating loses to shared mean (−0.00212 mean AUC; conversion −0.00434). On the seed-2025 checkpoint, 500k test rows, the conversion gate collapses onto the item token (0.970745) and drops the user token (0.025597). Click prefers user (0.645876). The two tasks learn different preferences; unconstrained softmax then saturates.

Residual pooling is the best gated variant and has the smallest mean-AUC standard deviation (0.00150). It is about +0.0015 over PLE and over the original gate, and −0.00049 versus shared mean, inside one standard deviation. It does not clearly beat shared mean. Seed-2025 λ is 0.531656 (click) and 0.498624 (conversion). Effective conversion weights are about user 0.644 / item 0.183 / context 0.174, while the gate branch itself still saturates on user (0.952436). Entropy regularization at 0.01 flattens both gates (entropy 1.097763 / 1.098189, maximum \(\ln 3 = 1.0986\)), which is mean pooling. Every anti-collapse run peaks at epoch 1. In the earlier multi-seed run the only exception is shared-mean seed 2026, whose best epoch is 2.

```bash
bash benchmarks/rankmixer/run_multiseed.sh 0
bash benchmarks/rankmixer/run_anticollapse.sh 0
```

`run_anticollapse.sh` writes under `/root/autodl-tmp/ac_out/` and then analyzes both seed-2025 checkpoints on the test split (500k rows), which is the development-study protocol. The archived `anticollapse_summary.csv` still marks the two analysis logs as failed. That was a summarizer bug: it treated `gate_*.log` as training runs. `summarize_multiseed.py` now skips `gate_` / `analyze_` / `grad_` logs and only parses `run_expid` training logs. The archived CSV was not rewritten.

## 7. Finer tokens and a 21-run sweep (recorded)

`configs/sweep/` and `run_sweep.sh` repeat the Oct 6 2026 sweep: seeds 2025 / 2026 / 2027, same optimizer and early-stop budget, existing Ali-CCP parquet. Six semantic tokens, `token_dim: 48`: user id `[101]`, user profile `[121,122,124,125,126,127,128,129]`, item id `[205]`, item category/shop/brand `[206,207,216]`, cross features `[508,509,702,853]`, scenario `[301]`. Templates:

| Template expid | What it is |
| --- | --- |
| `MTR_sw_g6_mean` | 6 tokens, shared mean |
| `MTR_sw_g6_gate` | 6 tokens, per-task softmax gate |
| `MTR_sw_g6_residual` | 6 tokens, residual gated pooling |
| `MTR_sw_ent0001` / `MTR_sw_ent0003` | 3-token gate, entropy 0.001 / 0.003 |
| `MTR_sw_res_ent0001` | 3-token residual, entropy 0.001 |
| `MTR_sw_res_temp2` | 3-token residual, `gate_temperature: 2` |

The script runs one job at a time and writes `/root/autodl-tmp/sw_out/`. The recorded run started three jobs at once; nine first attempts died with Ray `LocalRayletDiedError` and were rerun. All 21 final runs exited 0. Two of them (`ent0001` seed 2027, `g6_gate` seed 2027) peak at epoch 2; the other 19 peak at epoch 1.

Test mean AUC:

| Variant | conv AUC | mean AUC |
| --- | --- | ---: |
| g6_residual | 0.63468 ± 0.00680 | 0.62724 ± 0.00359 |
| g6_mean | 0.63473 ± 0.00327 | 0.62687 ± 0.00208 |
| g6_gate | 0.63442 ± 0.00177 | 0.62635 ± 0.00283 |
| shared mean, 3 tokens | 0.63168 ± 0.00224 | 0.62496 ± 0.00240 |
| res_ent0001 | 0.63132 ± 0.00068 | 0.62488 ± 0.00044 |
| PLE | 0.62605 ± 0.00630 | 0.62300 ± 0.00333 |

On this test-contaminated development study, moving shared mean from 3 tokens to 6 tokens is +0.00191 mean AUC and about +0.0031 conversion AUC, on the order of one standard deviation. That reading did not replicate under the validation-first protocol in section 8 (g6_mean − g3_mean mean AUC +0.00072, p=0.692; the random 6-way split scores at least as high as the semantic split). `g6_residual` is +0.00037 over `g6_mean` on this test table (a tie) and +0.00424 over PLE. On the seed-2025 checkpoint, `g6_gate` conversion collapses onto item id (0.966156). `g6_residual` does not: click leans on user id (0.592475), conversion on user profile (0.567906), λ 0.510613 / 0.496720. Entropy 0.001 and 0.003 flatten the 3-token gates; temperature 2 leaves them peaked (entropy 0.894404 / 0.843908) and does not raise AUC. Tables and figures: `benchmarks/rankmixer/results/sweep_results.md`, `docs/img/rankmixer/gate_g6_gate_s2025.png`, `docs/img/rankmixer/gate_g6_residual_s2025.png`.

```bash
bash benchmarks/rankmixer/run_sweep.sh 0
```

## 8. Rigor suite (validation first; recorded 2026-10-07)

Sections 5–7 used the test split for diagnosis and for choosing what to write up. This section does not. The suite ran on AutoDL with `run_rigor_suite.sh`, `SEEDS=2025 2026 2027 2028 2029`, `PARALLEL=2`. All 35 runs exited 0. Training passed `--skip_test`. Gate analysis used validation. The test parquet was not read.

Headline models, existing Ali-CCP parquet: `g3_mean`, `g6_mean`, `g6_residual`, `g6_gate`, `g6_random`, `g6_sequential`, PLE. Budget: batch 8192, embedding 16, Adam `1e-3`, epochs ≤ 10, `early_stop_patience: 3`, EQ loss. `g6_random` and `g6_sequential` are shared-mean controls at T=6. The random partition seed is 42 (Fisher-Yates via `random.Random.random`, then chunks of 3). The sequential groups are contiguous chunks of the 18-field list. Semantic groups have sizes 1, 8, 1, 3, 4, 1, so their projection widths differ from the equal groups of 3.

Validation mean ± sample std (n=5), sorted by mean AUC. Click and conversion are both impression-level. Full per-run rows, paired tests, gate tables, λ, and the gradient audit: `results/rigor/`.

| variant | click AUC | impression-level conversion AUC | mean AUC |
| --- | --- | --- | ---: |
| g6_gate | 0.61886 ± 0.00142 | 0.64139 ± 0.00589 | 0.63013 ± 0.00346 |
| g6_random | 0.61936 ± 0.00224 | 0.63956 ± 0.00502 | 0.62946 ± 0.00303 |
| g6_residual | 0.62009 ± 0.00126 | 0.63829 ± 0.00393 | 0.62919 ± 0.00228 |
| g6_mean | 0.61960 ± 0.00149 | 0.63589 ± 0.00568 | 0.62775 ± 0.00348 |
| g3_mean | 0.61790 ± 0.00114 | 0.63615 ± 0.00237 | 0.62702 ± 0.00154 |
| g6_sequential | 0.61991 ± 0.00225 | 0.63352 ± 0.00503 | 0.62672 ± 0.00211 |
| PLE | 0.61823 ± 0.00274 | 0.63136 ± 0.00651 | 0.62480 ± 0.00441 |

No mean-AUC paired comparison reaches p<0.1 (df=4). The smallest is g6_gate − PLE, +0.00533 ± 0.00653, 5/5, t=+1.83, p=0.142. Three single-metric rows are below 0.1: g6_sequential − g3_mean click p=0.080; g6_gate − PLE conversion p=0.077; g3_mean − PLE conversion p=0.095. The archived `rigor_results.md` says no comparison reaches p<0.1; that sentence disagrees with its table. The table is the record. g6_mean − g3_mean mean AUC is +0.00072 ± 0.00379 (3/5, p=0.692). g6_mean − g6_random is −0.00171 ± 0.00502 (2/5, p=0.488): the random 6-way split scores at least as high as the semantic split. The development-study “finer semantic tokens are the main gain” reading did not replicate.

`g6_gate` conversion puts weight 0.96–0.99 on one token in every seed (entropy 0.121305 ± 0.050015). The token changes with the seed: item_id (2025, 0.976), item_attr (2026, 0.982), user_profile (2027, 0.961), item_attr (2028, 0.984), user_profile (2029, 0.991). Residual pooling raises conversion-gate entropy to 0.699623 ± 0.467691; seeds 2027 and 2029 still exceed 0.9 on user_profile (0.920 and 0.988). Learnable λ stays near its 0.5 init: click 0.506983 ± 0.003786, conversion 0.499401 ± 0.004778.

Gradient audit, one `g6_mean` seed-2025 checkpoint, 4 validation batches, weights not updated: mean trunk L2 click 4.313689e-02, conversion 4.029779e-03, ratio 10.7045. Under EQ, click drives the trunk about 10.7× harder than conversion. `NORM` was not used. 34 of 35 runs peak at epoch 1; PLE seed 2026 peaks at epoch 2 (mean AUC 0.61730, 916 s). Every RankMixer variant’s mean AUC sits above PLE. Paired mean-AUC p-values versus PLE are 0.142 (g6_gate), 0.173 (g6_residual), 0.293 (g6_mean), 0.229 (g3_mean). There is no paired row for g6_random or g6_sequential versus PLE.

Seed-2025 validation figures (not a 5-seed mean): `docs/img/rankmixer/rigor_valid_g6_gate_s2025.png`, `docs/img/rankmixer/rigor_valid_g6_residual_s2025.png`. Final-test numbers are pending. The frozen set is all 7 variants, with no further selection.

```bash
# Already run (seeds 2025–2029, test unread). OUT defaults to /root/autodl-tmp/rg_out.
SEEDS="2025 2026 2027 2028 2029" bash /root/autodl-tmp/FuxiCTR/benchmarks/rankmixer/run_rigor_suite.sh 0
# Once, after freeze, all 7 variants, no further selection:
OUT=/root/autodl-tmp/rg_out bash /root/autodl-tmp/FuxiCTR/benchmarks/rankmixer/run_final_test.sh 0
```
