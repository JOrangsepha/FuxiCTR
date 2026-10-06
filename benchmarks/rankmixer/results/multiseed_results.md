# MT-RankMixer multi-seed results (Ali-CCP sampled, RTX 4090)

Seeds 2025/2026/2027; up to 6 epochs, early stopping patience 1 on average validation AUC. Test-set metrics. Std is sample std (n-1).

## Per run

| model | seed | click AUC | conv AUC | avg AUC | click logloss | conv logloss | best epoch | epochs run | wall time (s) |
|---|---|---|---|---|---|---|---|---|---|
| MT-RankMixer per-task gating | 2025 | 0.619847 | 0.630532 | 0.625190 | 0.161871 | 0.001997 | 1 | 2 | 461 |
| MT-RankMixer per-task gating | 2026 | 0.618528 | 0.622638 | 0.620583 | 0.161309 | 0.002306 | 1 | 2 | 455 |
| MT-RankMixer per-task gating | 2027 | 0.616664 | 0.628848 | 0.622756 | 0.162280 | 0.002067 | 1 | 2 | 456 |
| MT-RankMixer shared-mean (ablation) | 2025 | 0.620829 | 0.633976 | 0.627402 | 0.162166 | 0.002084 | 1 | 2 | 440 |
| MT-RankMixer shared-mean (ablation) | 2026 | 0.615693 | 0.629497 | 0.622595 | 0.164761 | 0.002122 | 2 | 3 | 589 |
| MT-RankMixer shared-mean (ablation) | 2027 | 0.618209 | 0.631558 | 0.624883 | 0.162030 | 0.002100 | 1 | 2 | 446 |
| PLE | 2025 | 0.619497 | 0.619832 | 0.619665 | 0.162070 | 0.002049 | 1 | 2 | 432 |
| PLE | 2026 | 0.620124 | 0.625897 | 0.623011 | 0.162407 | 0.002042 | 1 | 2 | 450 |
| PLE | 2027 | 0.620211 | 0.632435 | 0.626323 | 0.161919 | 0.002189 | 1 | 2 | 427 |

## Mean ± std over 3 seeds

| model | click AUC | conv AUC | avg AUC | click logloss | conv logloss |
|---|---|---|---|---|---|
| MT-RankMixer per-task gating | 0.61835 ± 0.00160 | 0.62734 ± 0.00416 | 0.62284 ± 0.00230 | 0.16182 ± 0.00049 | 0.00212 ± 0.00016 |
| MT-RankMixer shared-mean (ablation) | 0.61824 ± 0.00257 | 0.63168 ± 0.00224 | 0.62496 ± 0.00240 | 0.16299 ± 0.00154 | 0.00210 ± 0.00002 |
| PLE | 0.61994 ± 0.00039 | 0.62605 ± 0.00630 | 0.62300 ± 0.00333 | 0.16213 ± 0.00025 | 0.00209 ± 0.00008 |

## Differences of means

| comparison | click AUC | conv AUC | avg AUC |
|---|---|---|---|
| gated − PLE | -0.00160 | +0.00128 | -0.00016 |
| gated − shared-mean | +0.00010 | -0.00434 | -0.00212 |

## Gate analysis (seed 2025 gated checkpoint)

Checkpoint: `/root/autodl-tmp/FuxiCTR/model_zoo/multitask/MT_RankMixer/checkpoints/AliCCP_x1/MTRankMixer_aliccp_semantic_ms_s2025.model`

Rows used: 500000 (cap 500000). Sample standard deviation (n - 1). Gates sum to 1 over tokens inside each task.

| slice | task | n | user mean | user std | item mean | item std | context mean | context std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | click | 500000 | 0.645876 | 0.106685 | 0.314944 | 0.115830 | 0.039179 | 0.022231 |
| all | conversion | 500000 | 0.025597 | 0.029230 | 0.970745 | 0.030193 | 0.003658 | 0.002290 |
| click=1 | click | 23552 | 0.660649 | 0.089472 | 0.293571 | 0.096787 | 0.045780 | 0.022247 |
| click=1 | conversion | 23552 | 0.021904 | 0.024672 | 0.974363 | 0.025602 | 0.003733 | 0.002281 |
| conversion=1 | click | 159 | 0.658588 | 0.086070 | 0.303015 | 0.092868 | 0.038397 | 0.024820 |
| conversion=1 | conversion | 159 | 0.029376 | 0.031978 | 0.967195 | 0.033223 | 0.003429 | 0.002125 |

![gate weights](../../../docs/img/rankmixer/multiseed_gate_weights.png)
