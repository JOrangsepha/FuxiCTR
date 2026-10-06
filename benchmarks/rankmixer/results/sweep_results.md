# MT-RankMixer improvement sweep (Ali-CCP sample, Oct 6 2026)

Test metrics, latest log per expid (9 early Ray LocalRayletDiedError crashes were rerun cleanly; all 21 final runs exit code 0). Std is sample std (n-1).

## Per-run results

| variant | seed | best epoch | click AUC | conv AUC | avg AUC |
|---|---|---|---|---|---|
| ent0001 | 2025 | 1 | 0.61906 | 0.63608 | 0.62757 |
| ent0001 | 2026 | 1 | 0.61880 | 0.62865 | 0.62373 |
| ent0001 | 2027 | 2 | 0.61335 | 0.62948 | 0.62141 |
| ent0003 | 2025 | 1 | 0.61891 | 0.62784 | 0.62337 |
| ent0003 | 2026 | 1 | 0.61856 | 0.63041 | 0.62448 |
| ent0003 | 2027 | 1 | 0.61801 | 0.62350 | 0.62075 |
| res_ent0001 | 2025 | 1 | 0.61798 | 0.63149 | 0.62473 |
| res_ent0001 | 2026 | 1 | 0.62018 | 0.63058 | 0.62538 |
| res_ent0001 | 2027 | 1 | 0.61716 | 0.63190 | 0.62453 |
| res_temp2 | 2025 | 1 | 0.61826 | 0.63270 | 0.62548 |
| res_temp2 | 2026 | 1 | 0.61916 | 0.62538 | 0.62227 |
| res_temp2 | 2027 | 1 | 0.61956 | 0.62626 | 0.62291 |
| g6_mean | 2025 | 1 | 0.61976 | 0.63848 | 0.62912 |
| g6_mean | 2026 | 1 | 0.61973 | 0.63321 | 0.62647 |
| g6_mean | 2027 | 1 | 0.61754 | 0.63249 | 0.62501 |
| g6_residual | 2025 | 1 | 0.62021 | 0.64186 | 0.63104 |
| g6_residual | 2026 | 1 | 0.61945 | 0.62834 | 0.62389 |
| g6_residual | 2027 | 1 | 0.61973 | 0.63384 | 0.62679 |
| g6_gate | 2025 | 1 | 0.62119 | 0.63403 | 0.62761 |
| g6_gate | 2026 | 1 | 0.62030 | 0.63635 | 0.62833 |
| g6_gate | 2027 | 2 | 0.61334 | 0.63287 | 0.62311 |

## Combined comparison (mean ± std, 3 seeds each)

| variant | n | click AUC | conv AUC | avg AUC | Δ vs shared mean (3 tok) | Δ vs g6_mean | Δ vs PLE |
|---|---|---|---|---|---|---|---|
| g6_residual: residual gated pooling, 6 tokens | 3 | 0.61980 ± 0.00039 | 0.63468 ± 0.00680 | 0.62724 ± 0.00359 | +0.00228 | +0.00037 | +0.00424 |
| g6_mean: shared mean pooling, 6 tokens | 3 | 0.61901 ± 0.00127 | 0.63473 ± 0.00327 | 0.62687 ± 0.00208 | +0.00191 | +0.00000 | +0.00387 |
| g6_gate: per-task gate, 6 tokens | 3 | 0.61828 ± 0.00430 | 0.63442 ± 0.00177 | 0.62635 ± 0.00283 | +0.00139 | -0.00052 | +0.00335 |
| shared mean (3 tok, prev) | 3 | 0.61824 ± 0.00257 | 0.63168 ± 0.00224 | 0.62496 ± 0.00240 | +0.00000 | -0.00191 | +0.00196 |
| res_ent0001: residual gated pooling + entropy 0.001 (3 tokens) | 3 | 0.61844 ± 0.00156 | 0.63132 ± 0.00068 | 0.62488 ± 0.00044 | -0.00008 | -0.00199 | +0.00188 |
| residual gated pooling (3 tok, prev) | 3 | 0.61815 ± 0.00197 | 0.63080 ± 0.00328 | 0.62447 ± 0.00150 | -0.00049 | -0.00240 | +0.00147 |
| ent0001: per-task gate + entropy reg 0.001 (3 tokens) | 3 | 0.61707 ± 0.00323 | 0.63140 ± 0.00407 | 0.62424 ± 0.00311 | -0.00072 | -0.00263 | +0.00124 |
| entropy reg 0.01 (3 tok, prev) | 3 | 0.61907 ± 0.00145 | 0.62844 ± 0.00868 | 0.62375 ± 0.00506 | -0.00121 | -0.00312 | +0.00075 |
| res_temp2: residual gated pooling, gate temperature 2 (3 tokens) | 3 | 0.61899 ± 0.00066 | 0.62811 ± 0.00399 | 0.62355 ± 0.00170 | -0.00141 | -0.00332 | +0.00055 |
| PLE (prev) | 3 | 0.61994 ± 0.00039 | 0.62605 ± 0.00630 | 0.62300 ± 0.00333 | -0.00196 | -0.00387 | +0.00000 |
| ent0003: per-task gate + entropy reg 0.003 (3 tokens) | 3 | 0.61849 ± 0.00046 | 0.62725 ± 0.00349 | 0.62287 ± 0.00192 | -0.00209 | -0.00400 | -0.00013 |
| per-task gating (3 tok, prev) | 3 | 0.61835 ± 0.00160 | 0.62734 ± 0.00416 | 0.62284 ± 0.00230 | -0.00212 | -0.00403 | -0.00016 |
All 3-token "prev" rows come from /workspace/rankmixer_multiseed/results.md and /workspace/rankmixer_anticollapse/results.md. The 6-token variants use semantic tokens in this order: token0 user id (101), token1 user profile (121-129), token2 item id (205), token3 item category/shop/brand (206...), token4 cross features, token5 scenario (301).

## Gate weights (seed 2025 checkpoint, 500k test rows, mean over rows; max entropy is ln3 = 1.099 for 3 tokens, ln6 = 1.792 for 6 tokens)

| variant | click gate | conversion gate | entropy click / conv | residual λ click / conv |
|---|---|---|---|---|
| ent0001 | user 0.31, item 0.41, ctx 0.28 | user 0.33, item 0.35, ctx 0.32 | 1.079 / 1.096 | n/a |
| ent0003 | user 0.34, item 0.33, ctx 0.32 | 0.33 / 0.34 / 0.33 (uniform) | 1.096 / 1.098 | n/a |
| res_ent0001 | user 0.33, item 0.35, ctx 0.32 | 0.33 / 0.33 / 0.34 (uniform) | 1.092 / 1.097 | 0.553 / 0.493 |
| res_temp2 | user 0.56, item 0.32, ctx 0.12 | user 0.67, item 0.21, ctx 0.13 | 0.894 / 0.844 | 0.522 / 0.497 |
| g6_gate | user id 0.38, scenario 0.46, item id 0.14, rest <0.01 | item id 0.966 (collapsed), rest <0.02 | 1.023 / 0.177 | n/a |
| g6_residual | user id 0.59, cross 0.19, scenario 0.15 | user profile 0.57, item cat/shop/brand 0.16, item id 0.13 | 0.936 / 1.219 | 0.511 / 0.497 |
| g6_mean | no gate (shared mean pooling) | | | |

## Takeaway

- The clearest gain comes from finer tokenization, not from the gate: going from 3 to 6 semantic tokens lifts shared mean pooling from 0.62496 to 0.62687 avg AUC (+0.0019), mostly in conversion AUC (+0.0031). That is close to one std (0.0021-0.0024), so it is suggestive, not conclusive, with 3 seeds.
- Best variant overall is g6_residual at 0.62724 ± 0.00359 (+0.0023 vs 3-token shared mean, +0.0004 vs g6_mean, +0.0042 vs PLE). Versus g6_mean, the same-tokenization comparison, it is within noise; it does not beat it beyond one std. Versus PLE, both g6_residual and g6_mean are ahead by more than one std.
- With 6 tokens, the plain per-task gate collapses again (conversion puts 96.6% on item id) and trails g6_mean by 0.0005, while the residual gate keeps the mean path (λ about 0.5) and learns task-specific, non-collapsed weights (click leans on user id, conversion on user profile). That is a real interpretability point, even though the AUC gain over mean is within noise.
- Smaller entropy coefficients (0.001, 0.003) and gate temperature 2 did not help on 3 tokens; res_ent0001 ties shared mean (-0.0001) but has the lowest seed variance of all variants (std 0.00044).

![g6 per-task gate](../../../docs/img/rankmixer/gate_g6_gate_s2025.png)

![g6 residual gate](../../../docs/img/rankmixer/gate_g6_residual_s2025.png)

![residual temperature 2](../../../docs/img/rankmixer/gate_res_temp2_s2025.png)
