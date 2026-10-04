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

The existing multi-task configs in this repo only reference `tiny_mtl`. Ali-CCP is the public click/conversion set used for this comparison (Tianchi dataset 408).

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
