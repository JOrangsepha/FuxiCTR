# RankMixer 与 MT-RankMixer 技术说明

本文说明在本 FuxiCTR fork 中实现的 RankMixer（Zhu 等，CIKM 2025，arXiv:2507.15551，字节跳动）以及原创多任务变体 MT-RankMixer。公式编号与论文第 3 节一致。CPU 自测见第 5 节，RTX 4090 上的公开数据结果见第 7 节。

## 1. 动机

工业排序模型要在延迟和 QPS 约束下把参数做大。DCN、自注意力这一类 CPU 时代的交叉结构，在 GPU 上往往是访存瓶颈，MFU 很低，参数增加并不能换成算力利用率。RankMixer 用两件事换可扩展性：

- **多头 Token Mixing**：不用注意力权重，只做 reshape / 拼接，在 token 之间交换信息。论文报告它比自注意力更准，也更省算力和显存。
- **Per-token FFN**：每个 token 有自己的 FFN 参数。计算量和共享 FFN 相同，参数量大约乘以 token 数，用来避免高频特征空间把长尾特征淹没。

论文在抖音推荐上把稠密参数扩到约 1B，在线活跃天数 +0.3%，使用时长 +1.08%，并把 MFU 从 4.5% 提到 45%。公开的 Criteo / Ali-CCP 实验不能复现这个规模。本仓库的实现可以在 FuxiCTR 里训练；第 5 节是 tiny 数据上的 CPU 闭环，第 7 节是同一套小宽度配置在 RTX 4090 上的 1 epoch 对照。

## 2. 架构

输入嵌入拼成 \(e_{\text{input}}\)，切成 \(T\) 个 token，堆叠为 \(X_0 \in \mathbb{R}^{T \times D}\)。第 \(n\) 个 RankMixer block（论文式 1）：

\[
S_{n-1} = \mathrm{LN}\big(\mathrm{TokenMixing}(X_{n-1}) + X_{n-1}\big), \qquad
X_n = \mathrm{LN}\big(\mathrm{PFFN}(S_{n-1}) + S_{n-1}\big).
\]

**Tokenization（式 2）。** 默认按特征表顺序把字段嵌入展平，再切成等长片段，不够的位置补 0，然后用一个共享线性层投到 \(D\)：

\[
x_i = \mathrm{Proj}\big(e_{\text{input}}[d(i-1):di]\big), \quad i = 1,\ldots,T.
\]

**Token Mixing（式 3-5）。** 每个 token 切成 \(H\) 个头。第 \(h\) 个混合 token 是所有 token 的第 \(h\) 个头的拼接。论文设 \(H = T\)，这样混合后维度仍是 \(D\)，残差能直接相加。因此配置里 `token_dim` 必须能被 `num_tokens` 整除。这一步没有参数。

**Per-token FFN（式 6-7）。** 第 \(t\) 个 token 使用自己的 \(W^{t,1}\in\mathbb{R}^{D\times kD}\)、\(W^{t,2}\in\mathbb{R}^{kD\times D}\) 和 GELU：

\[
v_t = W^{t,2}\,\mathrm{GELU}(W^{t,1} s_t + b^{t,1}) + b^{t,2}.
\]

不同 token 的线性层不共享存储。\(k\) 对应配置项 `ffn_multiplier`。论文给出的稠密参数量约 \(2kLTD^2\)。

**输出。** 论文对最后一层 token 做 mean pooling，再算各任务 logit。单任务 RankMixer 在池化向量上接一个可配置 MLP（`mlp_hidden_units` 为空时就是一层线性 + sigmoid）。

**Sparse-MoE（式 10-11，可选）。** `use_sparse_moe: true` 时，每个 token 有 \(N_e\) 个专家 FFN，推理门控是 ReLU：

\[
G_{i,j} = \mathrm{ReLU}(h(s_i)), \qquad v_i = \sum_j G_{i,j}\, e_{i,j}(s_i),
\]

\[
\mathcal{L} = \mathcal{L}_{\text{task}} + \lambda \sum_{i,j} G_{i,j}.
\]

## 3. MT-RankMixer 相对论文的原创部分

论文把最终 token mean-pool 成一个向量，再接到各个任务头，池化权重对所有任务相同。MT-RankMixer 保留共享的 RankMixer trunk，但**不**做这次共享池化。每个任务自己对输出 token 打分并混合，再进自己的 tower：

\[
\alpha_{k,t} = \mathrm{softmax}_t(w_k^\top x_t + b_k), \qquad
h_k = \sum_t \alpha_{k,t} x_t, \qquad
\hat y_k = \mathrm{Tower}_k(h_k).
\]

`gate_type: sigmoid` 时先做 sigmoid，再在 token 维上做 L1 归一化，门控仍然和为 1，tower 的输入尺度与 softmax 混合一致。

这和 MMoE 不是一回事。MMoE 的专家读的是同一份展平嵌入；RankMixer 的 per-token FFN 每个只看自己的 token，参数和输入一起拆开。任务门控发生在 mixing 之后，门控的对象是 token，不是一份共享输入上的专家。

**分组方式可配置**。论文本身只描述了按语义聚类后再顺序切分。第 7.3 节在 Ali-CCP 上比较了这两种切法：

- `token_grouping: sequential`（默认）：按特征表顺序切块，对应式 2。
- `token_grouping: semantic`：`feature_groups` 里每一项是一组特征名，或一个 `source`（如 `user` / `item` / `context`）。每个组单独投影成一个 token。未覆盖或重复的特征会在建模型时直接报错。YAML 里写成整数的特征号（如 `101`）会转成字符串再匹配列名。

单任务模型同样支持这个开关，`RankMixer_semantic_test` 用 tiny parquet 里的 `user` / `item` / `context` 做了一次端到端训练。

## 4. 实现与文件

与论文对齐的层在 `fuxictr/pytorch/layers/interactions/rankmixer.py`，两个模型都调用它，避免两份公式实现。

| 路径 | 作用 |
| --- | --- |
| `fuxictr/pytorch/layers/interactions/rankmixer.py` | Token 切分、Token Mixing、Per-token FFN、Sparse-MoE、任务门控、分组解析 |
| `fuxictr/pytorch/layers/interactions/__init__.py` | 导出上述符号 |
| `model_zoo/RankMixer/` | 单任务模型、`run_expid.py`、tiny / Criteo 配置、README |
| `model_zoo/multitask/MT_RankMixer/` | 多任务模型、tiny_mtl / Ali-CCP 配置、README |
| `model_zoo/__init__.py`、`model_zoo/multitask/__init__.py` | 注册 `RankMixer`、`MTRankMixer` |
| `tests/unit_tests/models/test_rankmixer.py` | 形状、门控和为 1、FFN 参数独立、分组解析 |
| `tests/test_torch.sh` | 加入 5 个 CPU smoke expid |
| `benchmarks/rankmixer/` | 与 DCNv2 / WuKong / DNN / MMoE / PLE / ShareBottom 的 GPU 对照脚本和配置 |
| `README.md` | 模型表增加第 42 行 RankMixer、第 59 行 MT-RankMixer |

单任务实验号：`RankMixer_test`、`RankMixer_semantic_test`、`RankMixer_moe_test`、`RankMixer_criteo_x1`。

多任务实验号：`MTRankMixer_test`、`MTRankMixer_group_test`（显式特征组 + sigmoid 门控）、`MTRankMixer_aliccp`、`MTRankMixer_aliccp_semantic`。

`MTRankMixer` 继承 `MultiTaskModel`，输出键为 `{label}_pred`，与 MMoE / PLE 相同。tiny_mtl 的标签是 `click` 和 `conversion`。

## 5. 自测

环境只有 CPU。下面的命令和输出是本次提交前在仓库里实际跑出来的。

### 5.1 单元测试

```bash
python tests/unit_tests/models/test_rankmixer.py -v
```

14 个测试全部通过。原始摘要：

```
test_build_feature_tokenizer_rejects_groups_in_sequential_mode ... ok
test_resolve_feature_groups_from_yaml_like_config ... ok
test_semantic_tokenizer_groups_do_not_leak ... ok
test_sequential_chunk_padding_keeps_prefix ... ok
test_source_groups_and_rejections ... ok
test_moe_rankmixer_adds_penalty ... ok
test_mt_rankmixer_shapes_and_semantic_groups ... ok
test_rankmixer_output_shape_and_ffn_independence ... ok
test_experts_are_untied_and_reg_backprops ... ok
test_gate_weights_sum_to_one ... ok
test_block_output_shape ... ok
test_per_token_ffn_parameters_are_independent ... ok
test_token_dim_must_be_divisible_by_num_tokens ... ok
test_token_mixing_matches_formula_and_has_no_parameters ... ok

----------------------------------------------------------------------
Ran 14 tests in 0.952s

OK
```

### 5.2 CPU smoke test（各 1 个 epoch）

```bash
cd model_zoo/RankMixer
python run_expid.py --expid RankMixer_test --gpu -1
python run_expid.py --expid RankMixer_semantic_test --gpu -1
python run_expid.py --expid RankMixer_moe_test --gpu -1

cd model_zoo/multitask/MT_RankMixer
python run_expid.py --expid MTRankMixer_test --gpu -1
python run_expid.py --expid MTRankMixer_group_test --gpu -1
```

下面是 2026-10-04 在 CPU 上的原始日志，进度条和 Ray 预处理刷屏已略去，指标行保持原样。

```
========== RankMixer_test ==========
Total number of parameters: 29489.
Train loss: 0.459504
[Metrics] AUC: 0.937500
****** Validation evaluation ******
[Metrics] logloss: 0.138507 - AUC: 0.937500
******** Test evaluation ********
[Metrics] logloss: 0.138507 - AUC: 0.937500

========== RankMixer_semantic_test ==========
Total number of parameters: 20457.
Train loss: 0.283468
[Metrics] AUC: 1.000000
****** Validation evaluation ******
[Metrics] logloss: 0.109328 - AUC: 1.000000
******** Test evaluation ********
[Metrics] logloss: 0.109328 - AUC: 1.000000

========== RankMixer_moe_test ==========
Total number of parameters: 14073.
Train loss: 0.175467
[Metrics] AUC: 0.997396
****** Validation evaluation ******
[Metrics] logloss: 0.133811 - AUC: 0.997396
******** Test evaluation ********
[Metrics] logloss: 0.133811 - AUC: 0.997396

========== MTRankMixer_test ==========
Total number of parameters: 25484.
Train loss: 1.059085
[Task: click][Metrics] AUC: 0.563830
[Task: conversion][Metrics] AUC: 0.484848
****** Validation evaluation ******
[Task: click][Metrics] logloss: 0.277203 - AUC: 0.563830
[Task: conversion][Metrics] logloss: 0.427931 - AUC: 0.484848
******** Test evaluation ********
[Task: click][Metrics] logloss: 0.277203 - AUC: 0.563830
[Task: conversion][Metrics] logloss: 0.427931 - AUC: 0.484848

========== MTRankMixer_group_test ==========
Total number of parameters: 32300.
Train loss: 0.590076
[Task: click][Metrics] AUC: 0.673759
[Task: conversion][Metrics] AUC: 0.131313
****** Validation evaluation ******
[Task: click][Metrics] logloss: 0.224258 - AUC: 0.673759
[Task: conversion][Metrics] logloss: 0.092097 - AUC: 0.131313
******** Test evaluation ********
[Task: click][Metrics] logloss: 0.224258 - AUC: 0.673759
[Task: conversion][Metrics] logloss: 0.092097 - AUC: 0.131313
```

tiny 集只有约 100 条样本，AUC 接近 1 或很低都只说明训练闭环没有报错，不能当成模型效果。

`tests/test_torch.sh` 里已经加上上述 5 个 expid。整份脚本在 `DCNv3` 处会失败：`model_zoo/DCNv3` 不在这个 fork 里，这是改 RankMixer 之前就存在的问题，后面的模型（包括新加的）不会被 `&&` 链跑到。新模型是单独用上面的命令验证的。

## 6. 与论文的差异和局限

1. **没有序列子网络。** 论文先用 DIN / LONGER 一类模块把行为序列压成向量再进 tokenization。这里要求每个字段已经是一条向量。序列特征如果没有 `feature_encoder` 池化，`FeatureEmbedding` 无法堆成 `(batch, fields, dim)`。
2. **预测头是 FuxiCTR 的 MLP。** 论文只写了 mean pooling 之后算任务 logit，没有规定 MLP 深度。默认线性头；Criteo 配置在池化后加了一层宽度 64 的 ReLU。
3. **顺序切分用一个共享 `Proj`。** 和式 2 的写法一致。语义分组因为各组宽度不同，改成每组一个投影。
4. **Sparse-MoE 的 DTSI 是实用近似。** 论文引用 Pan 等 2024：训练用 \(h_{\text{train}}\)，推理只用 \(h_{\text{infer}}\)，\(\mathcal{L}_{\mathrm{reg}}\) 只加在推理门控上。本实现中 \(h_{\text{train}}\) 是 softmax（稠密，保证每个专家都有任务梯度），\(h_{\text{infer}}\) 是 ReLU。训练时两条门控各占一半混合进残差，稀疏那一半先做了归一化，避免尺度压过 softmax；惩罚项仍用未归一化的 ReLU。推理只用 ReLU 门控，允许全 0（残差支路还在）。\(\mathcal{L}_{\mathrm{reg}}\) 对 batch 取了均值，否则 `moe_lambda` 会随 batch 变大。推理门控的 bias 初始化为 1，让训练初期专家是打开的，再由 L1 压稀疏。论文里的 load-balancing 对比实验和 8 倍稀疏的线上 ROI 没有复现。
5. **没有工业侧的融合 kernel、fp16 服务、异步稀疏更新。** 参数量和 FLOPs 的 scaling 公式只作为文档，不在训练循环里统计 MFU。
6. **公开数据上的宽度远小于论文。** Criteo 配置的稠密 FFN 参数大约 \(2kLTD^2 = 2\times2\times2\times8\times64^2 = 262144\)。论文 100M 配置是 \(D=768, T=16, L=2\)。不要把这边的 AUC 和论文表 1 比。
7. **多任务门控和语义分组都不是论文内容。** 分组是消融开关，默认仍然是论文的顺序切分。

## 7. GPU 实验结果

本节数字来自 2026-10-05 在 AutoDL 上的一次完整流水线，9 个任务全部成功（exit 0）。代码是分支 `cursor/rankmixer-mt-ce74` 的 commit `5bbf507`，服务器上没有本地代码改动。Epoch 时间只计一个 epoch 的训练（日志里从 Epoch start 到 Evaluation），不含数据加载和评估。

### 7.1 设置

- GPU：1 × NVIDIA GeForce RTX 4090（AutoDL）。
- 时间：2026-10-05 12:15:54–12:57:40 CST，流水线总计 2505 秒（约 41.8 分钟）。
- 公共设置：`epochs=1`，`batch_size=8192`，`embedding_dim=16`，Adam，学习率 `1e-3`，`seed=2025`，无正则、无 dropout，监控 AUC（越大越好）。每个模型单次运行、单种子、未调参。
- Criteo_x1：FuxiCTR / BARS 标准切分，全量。4029 batch/epoch，约 3300 万训练样本。
- AliCCP_x1：PaddleRec 公开镜像预处理后的常用采样版，https://paddlerec.bj.bcebos.com/datasets/aitm/ 。这不是需要学生认证的天池原版。4648 batch/epoch，约 3800 万训练样本，`min_categr_count=10`。
- RankMixer（`RankMixer_criteo_x1`）：`num_tokens=8`，`token_dim=64`，`num_layers=2`，`ffn_multiplier=2`，`num_experts=4` 但 `use_sparse_moe=false`（稠密 FFN），`moe_lambda=1e-3`，`token_grouping=sequential`，LayerNorm + 残差，输出 MLP `[64]`。
- MT-RankMixer：`num_tasks=2`，`num_layers=2`，`ffn_multiplier=2`，`num_experts=4`（稠密），`gate_type=softmax`，`moe_lambda=1e-3`，`loss_weight=EQ`，任务 tower `[64]`。sequential 为 `num_tokens=8`、`token_dim=64`。semantic 为 `num_tokens=3`、`token_dim=48`，分组是用户 `[101,121,122,124,125,126,127,128,129]`、商品 `[205,206,207,216]`、组合/上下文 `[508,509,702,853,301]`。

### 7.2 Criteo_x1（单任务 CTR）

| 模型 | 参数量 | Valid AUC | Valid logloss | Test AUC | Test logloss | Epoch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DCNv2 | 34,745,121 | 0.808634 | 0.442998 | 0.808973 | 0.442537 | 97 s |
| RankMixer（新） | 33,656,481 | 0.808138 | 0.443553 | 0.808522 | 0.443048 | 114 s |
| DNN | 33,574,497 | 0.806911 | 0.444872 | 0.807165 | 0.444495 | 88 s |
| WuKong | 33,582,393 | 0.806493 | 0.444985 | 0.806889 | 0.444488 | 102 s |

RankMixer 的 test AUC 比 DNN 高 0.0014（0.808522 − 0.807165），比 WuKong 高 0.0016（0.808522 − 0.806889），比 DCNv2 低 0.0005（0.808522 − 0.808973）。在这组未调参的 1 epoch 设置里，RankMixer 优于 DNN 和 WuKong，与 DCNv2 基本持平。四个模型的参数量都在 3360 万到 3470 万之间，embedding 表占了绝大部分，稠密交叉部分的差别被参数总量拉平了。

### 7.3 AliCCP_x1（多任务 CTR / CVR）

Test：

| 模型 | 参数量 | click AUC | click logloss | conv AUC | conv logloss | 平均 AUC | Epoch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MT-RankMixer（semantic，新） | 20,478,948 | 0.619737 | 0.161928 | 0.640612 | 0.002067 | 0.630174 | 91 s |
| PLE | 20,738,012 | 0.621780 | 0.161666 | 0.624989 | 0.002128 | 0.623385 | 82 s |
| MT-RankMixer（sequential，新） | 20,678,612 | 0.617890 | 0.162420 | 0.627733 | 0.002098 | 0.622811 | 98 s |
| ShareBottom | 20,455,634 | 0.621760 | 0.161650 | 0.619310 | 0.002200 | 0.620535 | 69 s |
| MMoE | 20,628,890 | 0.619928 | 0.161709 | 0.618001 | 0.002148 | 0.618964 | 78 s |

Valid：

| 模型 | click AUC | click logloss | conv AUC | conv logloss | 平均 AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| MT-RankMixer（semantic） | 0.619608 | 0.161800 | 0.643919 | 0.002112 | 0.631764 |
| PLE | 0.621765 | 0.161530 | 0.630583 | 0.002172 | 0.626174 |
| MT-RankMixer（sequential） | 0.617952 | 0.162286 | 0.633717 | 0.002141 | 0.625835 |
| ShareBottom | 0.621575 | 0.161528 | 0.627480 | 0.002239 | 0.624527 |
| MMoE | 0.620067 | 0.161570 | 0.627999 | 0.002188 | 0.624033 |

semantic 分组的 MT-RankMixer 平均 AUC 最高（test 0.630174）。转化 AUC 也最高：test conv AUC 0.640612，比 PLE 的 0.624989 高约 0.016，conv logloss 同样最低（0.002067）。点击 AUC 是 0.619737，比 PLE（0.621780）和 ShareBottom（0.621760）低约 0.002。

sequential 分组的平均 AUC 是 0.622811，低于 semantic 的 0.630174，点击 AUC 在五者里最低。它的转化 AUC 0.627733 仍高于 PLE、ShareBottom 和 MMoE，但和 semantic 的差距说明按语义把特征放进不同 token，而不是按表顺序切块，是这次多任务扩展里起作用的部分。

### 7.4 局限

每个模型只训练 1 个 epoch，只有一个种子，没有调参。Ali-CCP 的转化正样本极少，CVR 的 AUC 和 logloss 波动大。0.001 量级的差距，包括 Criteo 上相对 DCNv2 的 0.0005，以及 Ali-CCP 点击上相对 PLE 的约 0.002，都可能落在噪声里。后续需要多种子、多个 epoch，并打开 early stopping 再复核。这组结果也不能和论文表 1 的抖音 100M / 1B 模型比较。

### 7.5 复现

脚本仍是 `benchmarks/rankmixer/run_gpu_benchmark.sh`，英文步骤在 `benchmarks/rankmixer/README.md`。Criteo_x1 用 BARS 切分。Ali-CCP 这次用的是 PaddleRec 镜像 https://paddlerec.bj.bcebos.com/datasets/aitm/ ，预处理成上述 18 个类别字段和 `click` / `conversion` 两列，而不是天池原版。semantic 消融的 expid 是 `MTRankMixer_aliccp_semantic`。在这张 4090 上，单个 epoch 的训练是 69–114 秒；整条 9 任务流水线含加载和评估共 41.8 分钟。
