# MT-RankMixer 研究笔记

这篇笔记记的是我在 FuxiCTR 里做的多任务排序，不是论文复现报告。RankMixer（Zhu 等，CIKM 2025，arXiv:2507.15551，字节跳动）我只当作骨干。我自己的部分是 MT-RankMixer：多任务场景下不再让 CTR 和 CVR 共用一次 mean-pooling。

已经跑完的数字只有 2026-10-05 那次 RTX 4090、1 epoch、单种子的对照，写在第 6 节，没有另编。多种子和门控统计的代码在第 8 节，那些实验还没有出结果。

## 1. 我观察到的问题

论文里的 RankMixer 把最后一层 token 做一次 mean-pooling，再接到各个任务头。池化权重对所有任务相同。这在单任务排序里说得通。我要做的是广告里的点击（CTR）和转化（CVR）一起学。

我的观察是：这两个任务要看的特征不是同一套。点击更吃曝光上下文（场景、位置这一类字段）；转化更吃用户和商品上的购买意图，而且正样本极少，logloss 在 0.002 这个量级。共享 mean-pooling 把所有 token 压成同一个向量，两个任务的梯度就在抢这一份表示。CVR 样本少，但一旦把共享表示往购买意图上拉，CTR 未必受益。

所以我假设：如果每个任务自己决定看哪些 token，CVR 可以多看用户和商品，CTR 可以留住上下文，两边不必妥协成同一个池化向量。

于是我设计了两件事。一是任务感知的 token 门控，替代共享 mean-pooling。二是按用户 / 商品 / 上下文把特征做成 token，让门控的对象是语义组，而不是按表顺序切出来的匿名块。第 6 节是这套设计在公开数据上的第一次检验。第 8 节是我准备用来把它和「只是换了个池化」分开的消融。

## 2. 原创贡献

下面三件都不是论文内容。论文的输出是一次共享 mean-pooling。

**(a) 任务感知的 token 门控。** 共享 RankMixer trunk 之后，我不做 mean-pool。任务 \(k\) 对每个输出 token 打分再混合，然后进自己的 tower：

\[
\alpha_{k,t} = \mathrm{softmax}_t(w_k^\top x_t + b_k), \qquad
h_k = \sum_t \alpha_{k,t} x_t, \qquad
\hat y_k = \mathrm{Tower}_k(h_k).
\]

`gate_type: sigmoid` 时先做 sigmoid，再在 token 维上做 L1 归一化，门控仍然和为 1。`task_pooling: mean`（或 `gate_type: mean`）关掉这扇门，所有任务共用 \(h = \mathrm{mean}_t(x_t)\)，tower 仍是各自的。默认是 per-task gate，旧配置的行为不变。

**(b) 语义 token 化。** `token_grouping: semantic` 时，`feature_groups` 的每一组投影成一个 token。Ali-CCP 上我分成三组：用户 `[101,121,122,124,125,126,127,128,129]`、商品 `[205,206,207,216]`、上下文 `[508,509,702,853,301]`。门控因此是在「用户 / 商品 / 上下文」上分配权重，而不是在 8 个顺序切片上。顺序切块 `token_grouping: sequential` 留作对照，对应论文式 2。

**(c) 用来检验 (a)(b) 的消融和门控分析。** 共享 mean-pool 是证明「per-task gating 有用」的对照：主干、分组、tower 宽度都不变，只拿掉门控参数。门控分析脚本在 test 集上统计每个任务对三个 token 的平均权重，并按 `click=1` / `conversion=1` 分层。这两项的协议在第 8 节，数字还没有。

这和 MMoE 不是一回事。MMoE 的专家读的是同一份展平嵌入。这里的 per-token FFN 每个只看自己的 token，任务门控发生在 mixing 之后，门控的对象是 token，不是一份共享输入上的专家。

## 3. 设计中的取舍

**为什么 \(H = T\)。** Token mixing 没有参数：每个 token 切成 \(H\) 个头，第 \(h\) 个混合 token 是所有 token 的第 \(h\) 个头拼起来。论文把 \(H\) 设成 \(T\)，混合后维度仍是 \(D\)，残差可以直接加。我保持这个约束，所以 `token_dim` 必须能被 `num_tokens` 整除。如果 \(H \neq T\)，就要再加一个投影才能残差，那一步有参数，也就不再是论文里的 parameter-free mixing。语义分组只有 3 个 token 时，我把 \(D\) 选成 48，而不是顺序切块用的 64，就是为了满足 \(48 \bmod 3 = 0\)。

**为什么门控放在 mixing 之后。** 如果在 mixing 之前就按任务选 token，每个任务只能看见自己选中的原始组，RankMixer 用来交换组间信息的那一步就被绕开了。放在 mixing 和 per-token FFN 之后，token 已经交换过信息，残差又把混合前的向量加回去。门控选的是「从用户组出发、经过混合、再被残差拉回来」的那条流，不是一个还没交叉过的裸嵌入。

**\(T = 3\) 时 token mixing 在做什么。** 三个 token、三个头，每个头的宽度是 \(D/3\)。混合后的每一个位置都是用户、商品、上下文各取一段拼接起来的，所以混合后的 token 不再是纯语义组。语义身份主要靠残差留在这条流上，再靠不共享参数的 per-token FFN 做不同的变换。我不会把门控权重解释成「模型只看了商品嵌入、完全没看上下文」。它表示的是这三条流的混合比例。这也是第 8 节要画出来核对的东西：如果 CTR 和 CVR 的平均门控几乎一样，说明这个开关没有学到任务差异。

**为什么 sigmoid 门要做 L1 归一化。** Softmax 自然和为 1，尺度和 mean-pool（均匀权重 \(1/T\)）可比。裸 sigmoid 的和取决于有几个 token 被激活，tower 的输入尺度会跟着变。那样的话，sigmoid 和 softmax 的差别里混进了尺度，不只是门的形状。L1 归一化之后，两种门的输出都在概率单纯形上，和共享 mean-pool 的尺度也对齐。分母我用 `clamp_min(1e-6)` 避免全 0。

**参数量。** 我没有把稠密层做成论文的 100M / 1B。Criteo 上顺序 RankMixer 的稠密 FFN 大约 \(2kLTD^2 = 2\times 2\times 2\times 8\times 64^2 = 262144\)，总参数 33,656,481，和 DNN / WuKong / DCNv2 一样，绝大部分是 embedding。Ali-CCP 上 semantic 的稠密 FFN 大约 \(2\times 2\times 2\times 3\times 48^2 = 55296\)，顺序切块仍是 262144。两组对照的 embedding 表同量级（约 2000 万），但稠密宽度并没有对齐。semantic 赢 sequential，不能全部算在分组上，稠密容量不同是混杂因素。第 8 节的 mean-pool 消融把 \(T\)、\(D\)、分组和 tower 都固定，只去掉门控。门控本身只有每个任务一个 `Linear(D, 1)`，semantic 配置下是 \(2\times(48+1) = 98\) 个参数，相对 2000 万可以忽略。所以这组消融比的是「有没有 per-task 权重」，不是比参数量。

**损失。** 两个任务用 FuxiCTR 的 `loss_weight: EQ`，也就是两份 batch 平均的 binary cross-entropy 直接相加。CVR 的正样本极少，EQ 并没有按正样本率加权。这可能让共享 trunk 被 CVR 梯度带偏，也可能让 CVR 根本学不动。我先保持和 MMoE / PLE 配置一致，不在这一轮同时改损失。不对称加权留作未来工作。

## 4. 基础工作：接进 FuxiCTR 的 RankMixer

这一节是骨干，篇幅只保留和后面实验有关的部分。公式编号沿用论文第 3 节。

输入嵌入展平后切成 \(T\) 个 token，堆成 \(X_0 \in \mathbb{R}^{T \times D}\)。第 \(n\) 个 block：

\[
S_{n-1} = \mathrm{LN}\big(\mathrm{TokenMixing}(X_{n-1}) + X_{n-1}\big), \qquad
X_n = \mathrm{LN}\big(\mathrm{PFFN}(S_{n-1}) + S_{n-1}\big).
\]

顺序切分用一个共享线性层，对应式 2。语义分组因为各组宽度不同，改成每组一个投影。Per-token FFN 的两层线性不共享存储，激活是 GELU，\(k\) 是 `ffn_multiplier`。单任务模型在 mean-pool 之后接 FuxiCTR 的 MLP。`use_sparse_moe: true` 时每个 token 换成 ReLU 路由的专家，这一路没有出现在第 6 节的 GPU 对照里（那边 `use_sparse_moe: false`）。

和论文相比，我明确改过或没做的有：没有序列编码器，字段必须已经是一条向量；预测头是可配置 MLP，Criteo 用了宽度 64 的 ReLU；Sparse-MoE 的训练门是 softmax、推理门是 ReLU，惩罚对 batch 取均值，推理门 bias 初始化为 1，这是 DTSI 的实用近似，不是生产 kernel；没有融合 kernel，也不统计 MFU。这些选择不影响第 6 节，因为那次对照用的是稠密 FFN。

层实现在 `fuxictr/pytorch/layers/interactions/rankmixer.py`。单任务模型在 `model_zoo/RankMixer/`，多任务在 `model_zoo/multitask/MT_RankMixer/`。

## 5. 实现里和本轮实验相关的开关

| 配置 | 作用 |
| --- | --- |
| `token_grouping: sequential` | 按特征表顺序切块。`RankMixer_criteo_x1`、`MTRankMixer_aliccp` |
| `token_grouping: semantic` | 按 `feature_groups` 分组。`MTRankMixer_aliccp_semantic` |
| `task_pooling: gate` | 默认。per-task softmax 或 sigmoid 门 |
| `task_pooling: mean` 或 `gate_type: mean` | 共享 mean-pool 消融。`MTRankMixer_aliccp_semantic_mean_es`、CPU 冒烟 `MTRankMixer_mean_test` |
| `MTRankMixer_aliccp_semantic_es` | 和 semantic 相同的结构，epochs 上限 6，`early_stop_patience: 1` |

`monitor: AUC` 在多任务里不是点击 AUC。`MultiTaskModel.evaluate` 会把各任务指标的算术平均写回裸键 `AUC`。Early stopping 看的是 click AUC 和 conversion AUC 的平均。

## 6. 结果分析

数字来自 2026-10-05 12:15:54–12:57:40 CST，AutoDL 上 1 × RTX 4090，分支 `cursor/rankmixer-mt-ce74` 的 commit `5bbf507`，服务器上没有本地代码改动。9 个任务全部 exit 0，流水线 2505 秒（约 41.8 分钟）。公共设置：`epochs=1`，`batch_size=8192`，`embedding_dim=16`，Adam，学习率 `1e-3`，`seed=2025`，无正则、无 dropout，监控 AUC。Epoch 时间只计训练，不含加载和评估。

Criteo_x1 是 FuxiCTR / BARS 全量切分，4029 batch/epoch，约 3300 万训练样本。RankMixer 为 `T=8, D=64, L=2, k=2`，稠密 FFN，顺序切块，输出 MLP `[64]`。

AliCCP_x1 来自 PaddleRec 公开镜像 https://paddlerec.bj.bcebos.com/datasets/aitm/ ，不是需要学生认证的天池原版。4648 batch/epoch，约 3800 万训练样本，`min_categr_count=10`。

### 6.1 Criteo：骨干是否站得住

| 模型 | 参数量 | Valid AUC | Valid logloss | Test AUC | Test logloss | Epoch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DCNv2 | 34,745,121 | 0.808634 | 0.442998 | 0.808973 | 0.442537 | 97 s |
| RankMixer | 33,656,481 | 0.808138 | 0.443553 | 0.808522 | 0.443048 | 114 s |
| DNN | 33,574,497 | 0.806911 | 0.444872 | 0.807165 | 0.444495 | 88 s |
| WuKong | 33,582,393 | 0.806493 | 0.444985 | 0.806889 | 0.444488 | 102 s |

Test AUC 上，RankMixer 比 DNN 高 0.0014（0.808522 − 0.807165），比 WuKong 高 0.0016（0.808522 − 0.806889），比 DCNv2 低 0.0005（0.808522 − 0.808973）。在这组未调参的 1 epoch 设置里，骨干不低于普通 DNN，和 DCNv2 基本持平。我把它当作「主干没有写坏」，不把它当作 RankMixer 相对 DCN 的结论。参数量都在 3360 万到 3470 万，embedding 占绝大部分。

### 6.2 Ali-CCP：语义分组和 per-task 门

Test：

| 模型 | 参数量 | click AUC | click logloss | conv AUC | conv logloss | 平均 AUC | Epoch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MT-RankMixer（semantic） | 20,478,948 | 0.619737 | 0.161928 | 0.640612 | 0.002067 | 0.630174 | 91 s |
| PLE | 20,738,012 | 0.621780 | 0.161666 | 0.624989 | 0.002128 | 0.623385 | 82 s |
| MT-RankMixer（sequential） | 20,678,612 | 0.617890 | 0.162420 | 0.627733 | 0.002098 | 0.622811 | 98 s |
| ShareBottom | 20,455,634 | 0.621760 | 0.161650 | 0.619310 | 0.002200 | 0.620535 | 69 s |
| MMoE | 20,628,890 | 0.619928 | 0.161709 | 0.618001 | 0.002148 | 0.618964 | 78 s |

Valid 上的排序和 test 一致：semantic 的平均 AUC 0.631764、转化 AUC 0.643919 都是最高；PLE 点击 AUC 0.621765 仍略高。完整 valid 表：

| 模型 | click AUC | click logloss | conv AUC | conv logloss | 平均 AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| MT-RankMixer（semantic） | 0.619608 | 0.161800 | 0.643919 | 0.002112 | 0.631764 |
| PLE | 0.621765 | 0.161530 | 0.630583 | 0.002172 | 0.626174 |
| MT-RankMixer（sequential） | 0.617952 | 0.162286 | 0.633717 | 0.002141 | 0.625835 |
| ShareBottom | 0.621575 | 0.161528 | 0.627480 | 0.002239 | 0.624527 |
| MMoE | 0.620067 | 0.161570 | 0.627999 | 0.002188 | 0.624033 |

**semantic 相对 sequential。** 平均 AUC 0.630174 对 0.622811，转化 AUC 0.640612 对 0.627733。我的解释是：顺序切块按特征表把字段切成 8 段，一个 token 里会混进用户字段和商品字段，门控无法把「商品」当成一个对象。语义分组把购买意图相关的用户、商品字段各自收成一个 token，CVR 的门可以同时抬高这两组。sequential 的转化 AUC 0.627733 仍然高于 PLE、ShareBottom 和 MMoE，所以骨干本身不是无效的；semantic 多出来的大约 0.013 转化 AUC，是我目前归到分组上的部分。第 3 节说过，这组对照的稠密 FFN 容量并不相同（55296 对 262144），所以这只是假设，不是隔离之后的因果结论。

**点击 AUC 为什么略输给 PLE。** semantic 的点击 AUC 是 0.619737，PLE 是 0.621780，ShareBottom 是 0.621760，大约低 0.002。同时转化 AUC 高约 0.016，转化 logloss 也最低（0.002067）。我现在的猜测有三条，都还没验证：

1. PLE 有任务专属专家，CTR 可以不经过 CVR 那条路径。我的门只有 3 个 token 的权重，造不出一组新特征。如果点击依赖的组合没有落在这三组的残差流上，门就调不过来。
2. `loss_weight: EQ` 把稀疏的 CVR 损失和 CTR 损失直接相加，共享 trunk 会被转化梯度拉动。这能解释「转化变好、点击变差」这种交换，而不是两个任务一起变好。
3. 上下文 token 对点击更重要。如果 CVR 的门把权重放在用户和商品上，而共享的 per-token FFN 又被这个梯度塑形，点击能用的上下文表示会被挤掉。

**未来工作（还没做）。** 我想试的是：给每个任务加一个 task-specific token，门可以选一条完全私有的流；或者把 CTR tower 做深、CVR tower 保持浅，做成不对称 tower；以及按正样本率改损失权重。这些都还没有实验。第 8 节先做更窄的一件事：在语义分组固定时，拿掉 per-task 门，看平均 AUC 和转化 AUC 掉不掉。

## 7. 局限

每个模型只训练了 1 个 epoch，一个种子，没有调参。Ali-CCP 转化正样本极少，CVR 的 AUC 和 logloss 波动大。0.001 量级的差距，包括 Criteo 上相对 DCNv2 的 0.0005，以及点击上相对 PLE 的约 0.002，都可能是噪声。semantic 对 PLE 的转化差距大约 0.016，比 0.001 大，但仍然是单种子、单 epoch，我不会把它写成稳定提升。

semantic 和 sequential 的稠密宽度没有对齐。公开数据上的宽度也远小于论文表 1 的抖音 100M / 1B 模型，AUC 不能和那张表比。

下一轮需要多种子、多个 epoch，并打开 early stopping。协议已经写进第 8 节，结果还没有。

## 8. 下一轮 GPU 实验（代码已准备，尚未跑）

机器仍是用户的 AutoDL 4090。数据沿用 `data/AliCCP/AliCCP_x1/` 里已经做好的 parquet 和 `feature_map.json`，脚本不会重新预处理。

三个模型，种子 2025、2026、2027：

| 模板 expid | 模型 |
| --- | --- |
| `MTRankMixer_aliccp_semantic_es` | 语义分组 + per-task softmax 门 |
| `MTRankMixer_aliccp_semantic_mean_es` | 同样的语义主干，`task_pooling: mean` |
| `PLE_aliccp_es` | PLE |

其余超参和上一轮相同：batch 8192，embedding 16，Adam `1e-3`，无正则、无 dropout。Epoch 上限 6，`early_stop_patience: 1`。监控指标是两任务 AUC 的算术平均，原因见第 5 节。FuxiCTR 默认还会在指标不提升时降低学习率；patience 为 1 时，测试指标来自恢复出来的最佳 checkpoint，降低后的学习率不会进入这组 test 数字。

每个任务单独日志，并套 `timeout`（默认 4 小时）。汇总表是 `benchmarks/rankmixer/multiseed_summary.csv`，`multiseed_stats.py` 计算均值 ± 样本标准差。

Semantic、per-task 门、种子 2025 的 checkpoint 会留在：

`model_zoo/multitask/MT_RankMixer/checkpoints/AliCCP_x1/MTRankMixer_aliccp_semantic_es_s2025.model`

门控脚本对这个 checkpoint 在 test 集上最多取 50 万行，输出每个任务对用户 / 商品 / 上下文的均值和标准差，并按 `click=1`、`conversion=1` 分层，另存一张 png。共享 mean-pool 的 checkpoint 没有门，脚本会直接退出。

```bash
bash /root/autodl-tmp/FuxiCTR/benchmarks/rankmixer/run_multiseed.sh 0

python /root/autodl-tmp/FuxiCTR/benchmarks/rankmixer/analyze_gates.py --gpu 0 \
  --checkpoint /root/autodl-tmp/FuxiCTR/model_zoo/multitask/MT_RankMixer/checkpoints/AliCCP_x1/MTRankMixer_aliccp_semantic_es_s2025.model \
  --config /root/autodl-tmp/FuxiCTR/benchmarks/rankmixer/configs/multiseed \
  --expid MTRankMixer_aliccp_semantic_es \
  --max_samples 500000
```

## 附录 A. CPU 自测日志

下面是 2026-10-04 在 CPU 上、tiny 数据、各 1 个 epoch 的原始日志。tiny 集大约 100 条，AUC 不能当效果。单元测试 14 个，当时 `Ran 14 tests in 0.952s`，`OK`。后来又加上共享池化、门控汇总和日志解析，当前是 18 个测试。`tests/test_torch.sh` 仍会在缺失的 `model_zoo/DCNv3` 处停住，这是改 RankMixer 之前就有的问题。

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

共享 mean-pool 的 CPU 冒烟是 `MTRankMixer_mean_test`。同一 tiny 主干上，门控版 25484 个参数，mean-pool 版 25418 个，差值 66，等于两个 `Linear(32, 1)`。tiny 上的 AUC 不记录为结果。

## 附录 B. 文件

| 路径 | 作用 |
| --- | --- |
| `fuxictr/pytorch/layers/interactions/rankmixer.py` | Token 切分、Token Mixing、Per-token FFN、Sparse-MoE、任务门控 |
| `model_zoo/RankMixer/` | 单任务模型与 Criteo / tiny 配置 |
| `model_zoo/multitask/MT_RankMixer/` | 多任务模型；`task_pooling` 在 `src/MTRankMixer.py` |
| `tests/unit_tests/models/test_rankmixer.py` | 形状、门控和为 1、FFN 独立、共享池化、汇总函数 |
| `benchmarks/rankmixer/run_gpu_benchmark.sh` | 已经跑完的 1 epoch 对照 |
| `benchmarks/rankmixer/run_multiseed.sh` | 3 模型 × 3 种子，early stopping |
| `benchmarks/rankmixer/configs/multiseed/` | 上述模板配置，数据格式是已有 parquet |
| `benchmarks/rankmixer/analyze_gates.py` | 门控均值、标准差和 png |
| `benchmarks/rankmixer/summarize_multiseed.py` | 写 `multiseed_summary.csv` |
| `benchmarks/rankmixer/multiseed_stats.py` | 均值 ± 样本标准差 |
