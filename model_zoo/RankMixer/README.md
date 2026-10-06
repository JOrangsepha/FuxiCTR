# RankMixer

> Jie Zhu, Zhifang Fan, Xiaoxie Zhu, Yuchen Jiang, Hangyu Wang, et al. [RankMixer: Scaling Up Ranking Models in Industrial Recommenders](https://arxiv.org/abs/2507.15551), CIKM 2025. ByteDance.

RankMixer replaces quadratic self-attention with parameter-free multi-head token mixing and gives every token its own FFN, so model width can grow without mixing heterogeneous feature spaces in one shared MLP. This directory is a FuxiCTR implementation of that architecture, plus an optional Sparse-MoE FFN.

The multi-task extension lives in [`../multitask/MT_RankMixer`](../multitask/MT_RankMixer). A Chinese technical note, including simplifications and the CPU test log, is in [`../../docs/RankMixer_tech_report.md`](../../docs/RankMixer_tech_report.md).

## Architecture

1. Embed each field, flatten in feature-map order, and split into `T` tokens. A shared linear map projects each chunk to dimension `D` (paper Eq. 2). `token_grouping: semantic` instead projects user-specified feature groups.
2. Stack `L` blocks. Each block is `LN(TokenMixing(X) + X)` then `LN(PFFN(S) + S)` (Eq. 1). Token mixing is a reshape with `H = T` heads and has no parameters (Eq. 3-5). `token_dim` must be divisible by `num_tokens`.
3. Each token has its own two-layer GELU FFN of width `k * D` (Eq. 6-7). Parameters are not shared across tokens.
4. Mean-pool the final tokens and apply a prediction MLP. The paper does not specify this head; the default is one linear layer when `mlp_hidden_units` is empty.

Set `use_sparse_moe: true` to replace each per-token FFN with ReLU-routed experts (Eq. 10-11). See the tech report for the dense-training / sparse-inference simplification.

## Configuration

| Parameter | Default | Description |
| --- | --- | --- |
| `num_tokens` | 8 | Token count `T`. Also the number of mixing heads. |
| `token_dim` | 64 | Token size `D`. Must be divisible by `num_tokens`. |
| `num_layers` | 2 | Number of RankMixer blocks `L`. |
| `ffn_multiplier` | 4 | FFN expansion ratio `k`. Hidden size is `k * D`. |
| `token_grouping` | `sequential` | `sequential` chunks the flattened embedding. `semantic` uses `feature_groups`. |
| `feature_groups` | null | List of feature-name lists, or source names such as `user`. Required for semantic mode. Every feature must appear once. |
| `use_sparse_moe` | false | Per-token Sparse-MoE FFN. |
| `num_experts` | 2 | Experts per token when Sparse-MoE is on. |
| `moe_lambda` | 1e-3 | Weight of the ReLU-gate L1 penalty. |
| `mlp_hidden_units` | [] | Prediction MLP. Empty means a linear head on the mean-pooled tokens. |

Other keys (`embedding_dim`, `optimizer`, `loss`, `batch_size`, ...) match the rest of the model zoo.

## Run

CPU smoke test on the tiny parquet split:

```bash
cd model_zoo/RankMixer
python run_expid.py --expid RankMixer_test --gpu -1
python run_expid.py --expid RankMixer_semantic_test --gpu -1
python run_expid.py --expid RankMixer_moe_test --gpu -1
```

Criteo_x1 (download steps are in [`../../benchmarks/rankmixer/README.md`](../../benchmarks/rankmixer/README.md)):

```bash
python run_expid.py --expid RankMixer_criteo_x1 --gpu 0
```

`RankMixer_criteo_x1` is a modest 1-epoch setting (`T=8`, `D=64`, `L=2`, `k=2`, embedding size 16, batch 8192), not the paper's 100M-parameter production model (`D=768`, `T=16`, `L=2`). On the recorded RTX 4090 run (2026-10-05, 1 epoch, seed 2025), test AUC was 0.808522, about +0.0014 over DNN and −0.0005 versus DCNv2. This Criteo number is still single-seed. See the preliminary Criteo table in [`../../docs/RankMixer_tech_report.md`](../../docs/RankMixer_tech_report.md).

## Simplifications

- No separate sequence encoder. Sequence fields must already be pooled to one vector (for example with a `feature_encoder`) so embeddings stack as `(batch, fields, dim)`.
- One shared `Proj` for sequential tokenization. Semantic groups use one projection per group because the group widths differ.
- The Sparse-MoE path mixes a softmax train router and a ReLU inference router. The L1 penalty is averaged over the batch. Details are in the tech report.
- Submodule dropout defaults to 0, which matches the paper.
