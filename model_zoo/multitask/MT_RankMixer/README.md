# MT-RankMixer

MT-RankMixer is an original multi-task extension of [RankMixer](https://arxiv.org/abs/2507.15551) (Zhu et al., CIKM 2025, ByteDance). It is not described in the paper.

The shared trunk is the RankMixer encoder in [`../../RankMixer`](../../RankMixer): field embeddings become `T` tokens, and `L` blocks apply parameter-free token mixing plus a per-token FFN. The paper then mean-pools those tokens once for every task. MT-RankMixer does not. Each task learns its own gate over the output tokens and its own tower:

```
alpha_{k,t} = softmax_t(w_k^T x_t + b_k)
h_k         = sum_t alpha_{k,t} x_t
y_k         = Tower_k(h_k)
```

`gate_type: sigmoid` uses a sigmoid and then L1-normalizes over tokens, so the gate still sums to 1 and the tower input stays on the same scale as a mean pool.

`task_pooling: residual` keeps that gate and mixes it with a shared mean, \(h_k = \lambda_k \mathrm{mean}_t(x_t) + (1-\lambda_k)\sum_t \alpha_{k,t} x_t\), where \(\lambda_k\) is a per-task sigmoid initialized at 0.5. That is the pooling added after the conversion gate collapsed onto one token. Default `task_pooling` is still `gate`.

Token grouping is configurable:

- `sequential` (default): the same chunking as the paper.
- `semantic`: `feature_groups` is a list of feature-name lists, or feature-source names. This is the ablation switch for grouping features by meaning (user / item / context) instead of by position.

The model subclasses `MultiTaskModel` and returns `{label}_pred` for each label. The tiny multi-task split uses `click` and `conversion`.

## Configuration

| Parameter | Default | Description |
| --- | --- | --- |
| `num_tasks` | 2 | Number of tasks. Must match `task` and the dataset labels. |
| `num_tokens` | 8 | Token count `T`. Must divide `token_dim`. In semantic mode, must match the number of groups when set. |
| `token_dim` | 64 | Token size `D`. |
| `num_layers` | 2 | RankMixer blocks in the shared trunk. |
| `ffn_multiplier` | 4 | Per-token FFN expansion ratio `k`. |
| `token_grouping` | `sequential` | `sequential` or `semantic`. |
| `feature_groups` | null | Semantic groups. Every feature is used exactly once. |
| `gate_type` | `softmax` | `softmax`, `sigmoid` (L1-normalized), or `mean`. |
| `task_pooling` | `gate` | `gate` keeps a per-task token gate. `mean` shares one mean-pool across towers. `residual` uses `h = λ * mean(tokens) + (1 - λ) * gated`, with one learned sigmoid scalar λ per task, initialized at 0.5. `gate_type: mean` selects the shared pool. |
| `gate_entropy_reg` | 0 | Coefficient β of `-β * H(α)` added to the loss, summed over tasks after a batch mean. `0` leaves the loss unchanged. The anti-collapse run used `0.01` because ln 3 ≈ 1.099, so the largest bonus per task is about 0.011 (~7% of click logloss ~0.16). That scale overshot to a uniform gate (entropy 1.097763 and 1.098189 vs maximum 1.0986). The later sweep also flattened 3-token gates at `0.003` (entropy 1.095964 / 1.097799) and at `0.001` (1.078908 / 1.096069). `0.1` would rival the click loss. |
| `gate_temperature` | 1 | Divides gate logits before softmax or sigmoid. `1` is the unscaled gate. |
| `tower_hidden_units` | [64] | Hidden units of each task tower. |
| `use_sparse_moe` | false | Sparse-MoE per-token FFNs in the shared trunk. |
| `moe_lambda` | 1e-3 | L1 penalty on the ReLU gates. |

## Run

CPU smoke tests:

```bash
cd model_zoo/multitask/MT_RankMixer
python run_expid.py --expid MTRankMixer_test --gpu -1
python run_expid.py --expid MTRankMixer_group_test --gpu -1
python run_expid.py --expid MTRankMixer_mean_test --gpu -1
python run_expid.py --expid MTRankMixer_residual_test --gpu -1
python run_expid.py --expid MTRankMixer_entropy_test --gpu -1
```

`MTRankMixer_group_test` uses semantic groups (all categorical fields in one token, all numerical fields in the other) and `gate_type: sigmoid`.

Ali-CCP, once the CSV splits are prepared (see [`../../../benchmarks/rankmixer/README.md`](../../../benchmarks/rankmixer/README.md)):

```bash
python run_expid.py --expid MTRankMixer_aliccp --gpu 0
python run_expid.py --expid MTRankMixer_aliccp_semantic --gpu 0
```

`MTRankMixer_aliccp` is the 1-epoch comparison point against MMoE, PLE, and ShareBottom. `MTRankMixer_aliccp_semantic` is the 3-token grouping ablation (user / item / context, `T=3`, `D=48`). The single-seed semantic test conversion AUC 0.640612 was not replicated. On 3 tokens, residual pooling (0.62447 ± 0.00150) does not clearly beat shared mean (0.62496 ± 0.00240).

The later sweep splits the same 18 fields into 6 tokens (`token_dim` stays 48): user id `[101]`, user profile `[121..129]`, item id `[205]`, item category/shop/brand `[206,207,216]`, cross features `[508,509,702,853]`, scenario `[301]`. Reproducible expids are `MTR_sw_g6_mean`, `MTR_sw_g6_gate`, and `MTR_sw_g6_residual` in [`../../../benchmarks/rankmixer/configs/sweep`](../../../benchmarks/rankmixer/configs/sweep). Three seeds, test mean AUC: g6_residual 0.62724 ± 0.00359, g6_mean 0.62687 ± 0.00208, g6_gate 0.62635 ± 0.00283. The gain versus 3-token shared mean is about +0.0019 and is within about one standard deviation. g6_residual is only +0.00037 over g6_mean. On seed 2025, the plain 6-token conversion gate collapses onto item id (0.966), while residual gating stays mixed: click leans on user id (0.59), conversion on user profile (0.57), λ about 0.51 / 0.50. See [`../../../docs/RankMixer_tech_report.md`](../../../docs/RankMixer_tech_report.md).

## How this differs from MMoE

MMoE experts all read the same flattened embedding, and each task gates those experts. MT-RankMixer first builds cross-token features with parameter-free mixing, keeps a separate FFN per token, and only then lets each task gate the tokens. The per-token FFNs are not MMoE experts: they do not share their input.
