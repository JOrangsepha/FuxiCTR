# RankMixer GPU benchmark

Modest 1-epoch comparison, meant to fit in a few GPU-hours on one rented GPU. It is a public-data check of this implementation, not a reproduction of the paper's trillion-scale Douyin experiments.

| Track | Models | Data |
| --- | --- | --- |
| Single-task | RankMixer, DCNv2, WuKong, DNN | [Criteo_x1](https://huggingface.co/datasets/reczoo/Criteo_x1) (BARS split, ~33.0M train rows) |
| Multi-task | MT-RankMixer, MMoE, PLE, ShareBottom | Ali-CCP click + conversion |

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

1. Download `sample_skeleton_{train,test}.csv` and `common_features_{train,test}.csv` from https://tianchi.aliyun.com/dataset/408 (login required). There is no official validation split.
2. Join skeleton rows with common features and keep the 18 categorical ids used by the AITM / Torch-RecHub preprocessing:
   `101, 121, 122, 124, 125, 126, 127, 128, 129, 205, 206, 207, 216, 508, 509, 702, 853, 301`.
   A readable reference script is https://github.com/xidongbo/AITM/blob/main/process_public_dataset.py (it names the second label `purchase`; rename that column to `conversion`).
3. Hold out 10% of the training rows, in order, as `valid.csv`. Write:

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

One full pipeline on 1 × NVIDIA GeForce RTX 4090 (AutoDL), commit `5bbf507`, no local code changes. Shared settings: 1 epoch, batch 8192, embedding size 16, Adam `lr=1e-3`, seed 2025, no L2, no dropout, AUC monitor, single seed, no tuning. Wall clock 12:15:54–12:57:40 CST (2505 s, about 41.8 min) for 9 tasks, all exit 0. Epoch time below is training only. Full tables, valid metrics, and caveats are in `docs/RankMixer_tech_report.md` section 7.

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

Semantic grouping has the best conversion AUC (about +0.016 over PLE) and the highest mean AUC. Click AUC is about 0.002 below PLE. Sequential is clearly weaker than semantic on mean AUC, so the semantic token split is the part that moves the multi-task result. These are 1-epoch, single-seed, untuned numbers. Ali-CCP conversion positives are rare, so gaps around 0.001 can be noise. Re-check with multiple seeds, more epochs, and early stopping.
