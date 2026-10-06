# MT-RankMixer anti-collapse results (Ali-CCP sampled, RTX 4090)

Commit 5627b6c. Seeds 2025/2026/2027; up to 6 epochs, early stopping patience 1 on average validation AUC. Test-set metrics. Std is sample std (n-1).

## Per run

| model | seed | click AUC | conv AUC | avg AUC | click logloss | conv logloss | best epoch | epochs run | wall time (s) |
|---|---|---|---|---|---|---|---|---|---|
| MT-RankMixer residual gated pooling | 2025 | 0.620415 | 0.629259 | 0.624837 | 0.161454 | 0.001992 | 1 | 2 | 595 |
| MT-RankMixer residual gated pooling | 2026 | 0.617089 | 0.628573 | 0.622831 | 0.161906 | 0.002129 | 1 | 2 | 604 |
| MT-RankMixer residual gated pooling | 2027 | 0.616937 | 0.634573 | 0.625755 | 0.162661 | 0.002045 | 1 | 2 | 602 |
| MT-RankMixer gate entropy reg (0.01) | 2025 | 0.617713 | 0.619608 | 0.618661 | 0.161968 | 0.002142 | 1 | 2 | 595 |
| MT-RankMixer gate entropy reg (0.01) | 2026 | 0.620596 | 0.636970 | 0.628783 | 0.161372 | 0.001981 | 1 | 2 | 601 |
| MT-RankMixer gate entropy reg (0.01) | 2027 | 0.618898 | 0.628728 | 0.623813 | 0.161878 | 0.002005 | 1 | 2 | 599 |

## Mean ± std over 3 seeds (this run plus previous multi-seed run)

| model | click AUC | conv AUC | avg AUC | click logloss | conv logloss |
|---|---|---|---|---|---|
| MT-RankMixer residual gated pooling | 0.61815 ± 0.00197 | 0.63080 ± 0.00328 | 0.62447 ± 0.00150 | 0.16201 ± 0.00061 | 0.00206 ± 0.00007 |
| MT-RankMixer gate entropy reg (0.01) | 0.61907 ± 0.00145 | 0.62844 ± 0.00868 | 0.62375 ± 0.00506 | 0.16174 ± 0.00032 | 0.00204 ± 0.00009 |
| MT-RankMixer per-task gating (prev) | 0.61835 ± 0.00160 | 0.62734 ± 0.00416 | 0.62284 ± 0.00230 | 0.16182 ± 0.00049 | 0.00212 ± 0.00016 |
| MT-RankMixer shared-mean ablation (prev) | 0.61824 ± 0.00257 | 0.63168 ± 0.00224 | 0.62496 ± 0.00240 | 0.16299 ± 0.00154 | 0.00210 ± 0.00002 |
| PLE (prev) | 0.61994 ± 0.00039 | 0.62605 ± 0.00630 | 0.62300 ± 0.00333 | 0.16213 ± 0.00025 | 0.00209 ± 0.00008 |

## Differences of means

| comparison | click AUC | conv AUC | avg AUC |
|---|---|---|---|
| MT-RankMixer residual gated pooling − PLE (prev) | -0.00179 | +0.00475 | +0.00147 |
| MT-RankMixer residual gated pooling − MT-RankMixer shared-mean ablation (prev) | -0.00009 | -0.00088 | -0.00049 |
| MT-RankMixer gate entropy reg (0.01) − PLE (prev) | -0.00087 | +0.00239 | +0.00075 |
| MT-RankMixer gate entropy reg (0.01) − MT-RankMixer shared-mean ablation (prev) | +0.00083 | -0.00324 | -0.00121 |
| MT-RankMixer per-task gating (prev) − PLE (prev) | -0.00159 | +0.00129 | -0.00016 |
| MT-RankMixer per-task gating (prev) − MT-RankMixer shared-mean ablation (prev) | +0.00011 | -0.00434 | -0.00212 |

## Gate weights, entropy and λ (seed-2025 checkpoints, 500k rows, slice = all)

| model | task | user | item | context | gate entropy (nats) | λ (weight on shared mean) |
|---|---|---|---|---|---|---|
| per-task gating (prev, collapsed) | click | 0.646 | 0.315 | 0.039 | not measured | n/a |
| per-task gating (prev, collapsed) | conversion | 0.026 | 0.971 | 0.004 | not measured | n/a |
| residual gated pooling | click | 0.677 | 0.271 | 0.052 | 0.659 | 0.532 |
| residual gated pooling | conversion | 0.952 | 0.032 | 0.015 | 0.203 | 0.499 |
| gate entropy reg 0.01 | click | 0.341 | 0.322 | 0.337 | 1.098 | n/a |
| gate entropy reg 0.01 | conversion | 0.331 | 0.334 | 0.335 | 1.098 | n/a |

Max entropy for 3 tokens is ln 3 = 1.0986 nats. Full tables: [gate_residual_s2025.md](gate_residual_s2025.md), [gate_entropy_s2025.md](gate_entropy_s2025.md).

![residual gates](../../../docs/img/rankmixer/gate_residual_s2025.png)

![entropy gates](../../../docs/img/rankmixer/gate_entropy_s2025.png)
## Notes

- Residual pooling: with λ ≈ 0.5, the effective token weights are λ/3 + (1−λ)·gate. For conversion that is about user 0.644, item 0.183, context 0.174, so no token gets almost all the weight anymore. The gated part, though, now concentrates on the user token (0.952) instead of the item token. The gate still saturates; the mean-pooling term is what keeps the mix balanced.
- Entropy regularization at 0.01 pushes both tasks' gates to almost exactly uniform (entropy 1.098 of a 1.0986 maximum). That makes it effectively the same as shared mean pooling, and per-task selectivity is lost.
- Gate tables come from seed 2025 only. Every run's best epoch was 1, and it early-stopped after epoch 2.
- Wall time per run (~600 s) is longer than the previous run (~450 s). The two runs used different drivers, so the times aren't directly comparable.
