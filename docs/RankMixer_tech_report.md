# RankMixer 与 MT-RankMixer 技术说明

本文说明在本 FuxiCTR fork 中实现的 RankMixer（Zhu 等，CIKM 2025，arXiv:2507.15551，字节跳动）以及原创多任务变体 MT-RankMixer。公式编号与论文第 3 节一致。CPU 自测命令和原始输出见第 5 节。

## 1. 动机

工业排序模型要在延迟和 QPS 约束下把参数做大。DCN、自注意力这一类 CPU 时代的交叉结构，在 GPU 上往往是访存瓶颈，MFU 很低，参数增加并不能换成算力利用率。RankMixer 用两件事换可扩展性：

- **多头 Token Mixing**：不用注意力权重，只做 reshape / 拼接，在 token 之间交换信息。论文报告它比自注意力更准，也更省算力和显存。
- **Per-token FFN**：每个 token 有自己的 FFN 参数。计算量和共享 FFN 相同，参数量大约乘以 token 数，用来避免高频特征空间把长尾特征淹没。

论文在抖音推荐上把稠密参数扩到约 1B，在线活跃天数 +0.3%，使用时长 +1.08%，并把 MFU 从 4.5% 提到 45%。公开的 Criteo / Ali-CCP 实验不能复现这个规模，本仓库提供的是可在 FuxiCTR 里训练、可在小数据上核对形状和训练闭环、并预留了 GPU 对照配置的实现。

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

**分组方式可配置**，这是计划中的消融，论文本身只描述了按语义聚类后再顺序切分：

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

## 7. 计划中的 GPU 对照

脚本：`benchmarks/rankmixer/run_gpu_benchmark.sh`。说明和下载步骤：`benchmarks/rankmixer/README.md`。本环境没有 GPU，也没有 Criteo / Ali-CCP 全量文件，这一节的实验还没有跑。

对照关系：

- 单任务，Criteo_x1：`RankMixer_criteo_x1` vs DCNv2 / WuKong / DNN
- 多任务，Ali-CCP（click、conversion）：`MTRankMixer_aliccp` vs MMoE / PLE / ShareBottom

共同设置：Adam，学习率 `1e-3`，embedding 16，batch 8192，1 个 epoch，seed 2025，不加额外 L2。RankMixer 为 `T=8, D=64, L=2, k=2`。

### 数据

Criteo_x1（约 2.9 GB）：

```bash
mkdir -p data/Criteo
wget -O /tmp/Criteo_x1.zip \
  https://huggingface.co/datasets/reczoo/Criteo_x1/resolve/main/Criteo_x1.zip
unzip /tmp/Criteo_x1.zip -d data/Criteo
# data/Criteo/Criteo_x1/{train,valid,test}.csv
# 标签列 label；数值 I1–I13；类别 C1–C26
```

Ali-CCP 需要天池登录（数据集 408）：https://tianchi.aliyun.com/dataset/408 。下载 `sample_skeleton_{train,test}.csv` 和 `common_features_{train,test}.csv`，按 `benchmarks/rankmixer/README.md` 抽 18 个类别特征，第二标签列命名为 `conversion`，并从训练集按顺序留出 10% 做验证。放到：

```text
data/AliCCP/AliCCP_x1/train.csv
data/AliCCP/AliCCP_x1/valid.csv
data/AliCCP/AliCCP_x1/test.csv
```

### 命令

```bash
bash benchmarks/rankmixer/run_gpu_benchmark.sh 0          # 8 个任务
bash benchmarks/rankmixer/run_gpu_benchmark.sh 0 single   # 只跑 Criteo
bash benchmarks/rankmixer/run_gpu_benchmark.sh 0 multi    # 只跑 Ali-CCP

# 语义分组消融（不在默认脚本里）
cd model_zoo/multitask/MT_RankMixer
python run_expid.py --expid MTRankMixer_aliccp_semantic --gpu 0
```

### 预计耗时

预处理缓存写好之后，单卡、batch 8192、1 个 epoch：T4/L4 上 8 个任务大约 8–12 GPU 小时；A10 大约 5–8 小时；A100 40GB 大约 3–5 小时。时间主要花在读数据和 embedding，不在这个小宽度的 FFN 上。第一次从 csv 建词表和 parquet 缓存，每个数据集还要额外 30–90 分钟。

若预算只有大约 3 GPU 小时：只跑 RankMixer vs DCNv2，以及 MT-RankMixer vs PLE；或把训练集截成前 400 万行（验证集和测试集保持全量），并在结果里注明这不是完整 epoch。
