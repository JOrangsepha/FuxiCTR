# MT-RankMixer

MT-RankMixer is an original multi-task extension of [RankMixer](https://arxiv.org/abs/2507.15551) (Zhu et al., CIKM 2025, ByteDance). It is not described in the paper.

The shared trunk is the RankMixer encoder in [`../../RankMixer`](../../RankMixer): field embeddings become `T` tokens, and `L` blocks apply parameter-free token mixing plus a per-token FFN. The paper then mean-pools those tokens once for every task. MT-RankMixer does not. Each task learns its own gate over the output tokens and its own tower:

```
alpha_{k,t} = softmax_t(w_k^T x_t + b_k)
h_k         = sum_t alpha_{k,t} x_t
y_k         = Tower_k(h_k)
```

`gate_type: sigmoid` uses a sigmoid and then L1-normalizes over tokens, so the gate still sums to 1 and the tower input stays on the same scale as a mean pool.

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
| `gate_type` | `softmax` | `softmax` or `sigmoid` (L1-normalized). |
| `tower_hidden_units` | [64] | Hidden units of each task tower. |
| `use_sparse_moe` | false | Sparse-MoE per-token FFNs in the shared trunk. |
| `moe_lambda` | 1e-3 | L1 penalty on the ReLU gates. |

## Run

CPU smoke tests:

```bash
cd model_zoo/multitask/MT_RankMixer
python run_expid.py --expid MTRankMixer_test --gpu -1
python run_expid.py --expid MTRankMixer_group_test --gpu -1
```

`MTRankMixer_group_test` uses semantic groups (all categorical fields in one token, all numerical fields in the other) and `gate_type: sigmoid`.

Ali-CCP, once the CSV splits are prepared (see [`../../../benchmarks/rankmixer/README.md`](../../../benchmarks/rankmixer/README.md)):

```bash
python run_expid.py --expid MTRankMixer_aliccp --gpu 0
python run_expid.py --expid MTRankMixer_aliccp_semantic --gpu 0
```

`MTRankMixer_aliccp` is the 1-epoch comparison point against MMoE, PLE, and ShareBottom. `MTRankMixer_aliccp_semantic` is the grouping ablation (user profile / item / shop-context, `T=3`, `D=48`).

## How this differs from MMoE

MMoE experts all read the same flattened embedding, and each task gates those experts. MT-RankMixer first builds cross-token features with parameter-free mixing, keeps a separate FFN per token, and only then lets each task gate the tokens. The per-token FFNs are not MMoE experts: they do not share their input.
