# MT-RankMixer rigor suite — validation-only results

Run on AutoDL, finished 2026-10-07 16:00 CST (ALL_DONE). 7 variants × 5 seeds (2025–2029) = 35 training runs, all exit 0. patience=3, epochs≤10, EQ loss (unnormalized sum of per-task BCE), gate_stage=valid, `--skip_test`.

**The test split was NOT read by this suite.** Every number below is on the validation split. Final-test is a separate, one-time step after the design is frozen (`bash /root/autodl-tmp/FuxiCTR/benchmarks/rankmixer/run_final_test.sh 0`; checkpoints, ~2.7 GB, remain on the AutoDL data disk under `/root/autodl-tmp/rg_out/checkpoints/` and were not copied here).

## Per-variant mean ± std (sample std, n=5), sorted by avg AUC

| variant | n | click AUC | conv AUC | avg AUC |
|---|---|---|---|---|
| g6_gate | 5 | 0.61886 ± 0.00142 | 0.64139 ± 0.00589 | 0.63013 ± 0.00346 |
| g6_random | 5 | 0.61936 ± 0.00224 | 0.63956 ± 0.00502 | 0.62946 ± 0.00303 |
| g6_residual | 5 | 0.62009 ± 0.00126 | 0.63829 ± 0.00393 | 0.62919 ± 0.00228 |
| g6_mean | 5 | 0.61960 ± 0.00149 | 0.63589 ± 0.00568 | 0.62775 ± 0.00348 |
| g3_mean | 5 | 0.61790 ± 0.00114 | 0.63615 ± 0.00237 | 0.62702 ± 0.00154 |
| g6_sequential | 5 | 0.61991 ± 0.00225 | 0.63352 ± 0.00503 | 0.62672 ± 0.00211 |
| PLE | 5 | 0.61823 ± 0.00274 | 0.63136 ± 0.00651 | 0.62480 ± 0.00441 |

## Paired comparisons (same seed), all three metrics

| comparison (A − B, same seed) | metric | mean diff ± sd | A wins | paired t (df=4) | p |
|---|---|---|---|---|---|
| g6_mean − g3_mean | click | +0.00171 ± 0.00194 | 4/5 | +1.96 | 0.121 |
| g6_mean − g3_mean | conv | -0.00026 ± 0.00589 | 3/5 | -0.10 | 0.926 |
| g6_mean − g3_mean | mean | +0.00072 ± 0.00379 | 3/5 | +0.43 | 0.692 |
| g6_mean − g6_random | click | +0.00024 ± 0.00186 | 3/5 | +0.29 | 0.788 |
| g6_mean − g6_random | conv | -0.00367 ± 0.00888 | 1/5 | -0.92 | 0.408 |
| g6_mean − g6_random | mean | -0.00171 ± 0.00502 | 2/5 | -0.76 | 0.488 |
| g6_mean − g6_sequential | click | -0.00031 ± 0.00262 | 2/5 | -0.27 | 0.803 |
| g6_mean − g6_sequential | conv | +0.00237 ± 0.00865 | 1/5 | +0.61 | 0.573 |
| g6_mean − g6_sequential | mean | +0.00103 ± 0.00474 | 2/5 | +0.49 | 0.652 |
| g6_random − g6_sequential | click | -0.00055 ± 0.00256 | 2/5 | -0.48 | 0.655 |
| g6_random − g6_sequential | conv | +0.00604 ± 0.00788 | 3/5 | +1.71 | 0.162 |
| g6_random − g6_sequential | mean | +0.00274 ± 0.00421 | 3/5 | +1.46 | 0.219 |
| g6_random − g3_mean | click | +0.00147 ± 0.00286 | 4/5 | +1.14 | 0.316 |
| g6_random − g3_mean | conv | +0.00341 ± 0.00445 | 4/5 | +1.71 | 0.162 |
| g6_random − g3_mean | mean | +0.00244 ± 0.00326 | 4/5 | +1.67 | 0.170 |
| g6_sequential − g3_mean | click | +0.00202 ± 0.00193 | 4/5 | +2.33 | 0.080 |
| g6_sequential − g3_mean | conv | -0.00263 ± 0.00703 | 2/5 | -0.84 | 0.449 |
| g6_sequential − g3_mean | mean | -0.00031 ± 0.00353 | 2/5 | -0.19 | 0.855 |
| g6_residual − g6_mean | click | +0.00049 ± 0.00152 | 3/5 | +0.72 | 0.510 |
| g6_residual − g6_mean | conv | +0.00240 ± 0.00512 | 3/5 | +1.05 | 0.355 |
| g6_residual − g6_mean | mean | +0.00144 ± 0.00322 | 3/5 | +1.00 | 0.373 |
| g6_gate − g6_mean | click | -0.00074 ± 0.00171 | 1/5 | -0.96 | 0.389 |
| g6_gate − g6_mean | conv | +0.00550 ± 0.00767 | 4/5 | +1.60 | 0.184 |
| g6_gate − g6_mean | mean | +0.00238 ± 0.00451 | 4/5 | +1.18 | 0.303 |
| g6_residual − g6_gate | click | +0.00123 ± 0.00167 | 3/5 | +1.65 | 0.175 |
| g6_residual − g6_gate | conv | -0.00311 ± 0.00762 | 2/5 | -0.91 | 0.414 |
| g6_residual − g6_gate | mean | -0.00094 ± 0.00456 | 2/5 | -0.46 | 0.669 |
| g6_mean − PLE | click | +0.00137 ± 0.00381 | 3/5 | +0.81 | 0.465 |
| g6_mean − PLE | conv | +0.00453 ± 0.00735 | 4/5 | +1.38 | 0.241 |
| g6_mean − PLE | mean | +0.00295 ± 0.00545 | 4/5 | +1.21 | 0.293 |
| g6_residual − PLE | click | +0.00187 ± 0.00367 | 4/5 | +1.14 | 0.319 |
| g6_residual − PLE | conv | +0.00692 ± 0.00867 | 4/5 | +1.79 | 0.149 |
| g6_residual − PLE | mean | +0.00439 ± 0.00593 | 4/5 | +1.66 | 0.173 |
| g6_gate − PLE | click | +0.00064 ± 0.00379 | 2/5 | +0.38 | 0.726 |
| g6_gate − PLE | conv | +0.01003 ± 0.00946 | 5/5 | +2.37 | 0.077 |
| g6_gate − PLE | mean | +0.00533 ± 0.00653 | 5/5 | +1.83 | 0.142 |
| g3_mean − PLE | click | -0.00033 ± 0.00249 | 1/5 | -0.30 | 0.781 |
| g3_mean − PLE | conv | +0.00479 ± 0.00492 | 4/5 | +2.18 | 0.095 |
| g3_mean − PLE | mean | +0.00223 ± 0.00352 | 4/5 | +1.42 | 0.229 |

p-values are two-sided paired t with 4 df; with n=5 these are low-power, and no comparison here reaches p<0.1.

## Per-run validation results

| variant | seed | best epoch | valid click AUC | valid conv AUC | valid avg AUC | train s |
|---|---|---|---|---|---|---|
| PLE | 2025 | 1 | 0.61957 | 0.63555 | 0.62756 | 767 |
| PLE | 2026 | 2 | 0.61355 | 0.62105 | 0.61730 | 916 |
| PLE | 2027 | 1 | 0.61808 | 0.63688 | 0.62748 | 731 |
| PLE | 2028 | 1 | 0.62029 | 0.63441 | 0.62735 | 724 |
| PLE | 2029 | 1 | 0.61965 | 0.62891 | 0.62428 | 669 |
| g3_mean | 2025 | 1 | 0.61857 | 0.63827 | 0.62842 | 790 |
| g3_mean | 2026 | 1 | 0.61724 | 0.63230 | 0.62477 | 784 |
| g3_mean | 2027 | 1 | 0.61685 | 0.63558 | 0.62621 | 771 |
| g3_mean | 2028 | 1 | 0.61725 | 0.63757 | 0.62741 | 769 |
| g3_mean | 2029 | 1 | 0.61957 | 0.63703 | 0.62830 | 775 |
| g6_gate | 2025 | 1 | 0.61839 | 0.63735 | 0.62787 | 875 |
| g6_gate | 2026 | 1 | 0.62051 | 0.64669 | 0.63360 | 873 |
| g6_gate | 2027 | 1 | 0.61901 | 0.64863 | 0.63382 | 871 |
| g6_gate | 2028 | 1 | 0.61966 | 0.63888 | 0.62927 | 889 |
| g6_gate | 2029 | 1 | 0.61675 | 0.63541 | 0.62608 | 887 |
| g6_mean | 2025 | 1 | 0.62122 | 0.64450 | 0.63286 | 799 |
| g6_mean | 2026 | 1 | 0.62102 | 0.63561 | 0.62832 | 853 |
| g6_mean | 2027 | 1 | 0.61943 | 0.63763 | 0.62853 | 842 |
| g6_mean | 2028 | 1 | 0.61790 | 0.63009 | 0.62400 | 858 |
| g6_mean | 2029 | 1 | 0.61843 | 0.63161 | 0.62502 | 850 |
| g6_random | 2025 | 1 | 0.62143 | 0.63864 | 0.63003 | 848 |
| g6_random | 2026 | 1 | 0.62094 | 0.63582 | 0.62838 | 851 |
| g6_random | 2027 | 1 | 0.61771 | 0.63908 | 0.62840 | 856 |
| g6_random | 2028 | 1 | 0.62045 | 0.64814 | 0.63430 | 857 |
| g6_random | 2029 | 1 | 0.61628 | 0.63611 | 0.62620 | 851 |
| g6_residual | 2025 | 1 | 0.62000 | 0.64094 | 0.63047 | 913 |
| g6_residual | 2026 | 1 | 0.62191 | 0.63951 | 0.63071 | 918 |
| g6_residual | 2027 | 1 | 0.61844 | 0.63732 | 0.62788 | 892 |
| g6_residual | 2028 | 1 | 0.61965 | 0.63193 | 0.62579 | 891 |
| g6_residual | 2029 | 1 | 0.62046 | 0.64173 | 0.63109 | 882 |
| g6_sequential | 2025 | 1 | 0.62057 | 0.62668 | 0.62362 | 846 |
| g6_sequential | 2026 | 1 | 0.62126 | 0.63752 | 0.62939 | 835 |
| g6_sequential | 2027 | 1 | 0.61591 | 0.63933 | 0.62762 | 843 |
| g6_sequential | 2028 | 1 | 0.62069 | 0.63169 | 0.62619 | 844 |
| g6_sequential | 2029 | 1 | 0.62113 | 0.63237 | 0.62675 | 828 |

34 of 35 runs hit their best validation AUC at epoch 1 (PLE seed 2026 at epoch 2, and that run is also the PLE outlier at 0.6173 avg). This is the familiar one-epoch overfitting pattern in CTR models; early stopping with patience=3 means each run trained ~4 epochs and kept the epoch-1 checkpoint.

## Gate behaviour across seeds (validation, slice=all)

| variant | seed | task | top token (weight) | 2nd token (weight) | entropy (nats) |
|---|---|---|---|---|---|
| g6_gate | 2025 | click | scenario (0.432) | user_id (0.309) | 1.143 |
| g6_gate | 2025 | conversion | item_id (0.976) | user_id (0.009) | 0.140 |
| g6_gate | 2026 | click | scenario (0.887) | item_attr (0.094) | 0.391 |
| g6_gate | 2026 | conversion | item_attr (0.982) | scenario (0.005) | 0.113 |
| g6_gate | 2027 | click | cross (0.407) | user_id (0.398) | 1.079 |
| g6_gate | 2027 | conversion | user_profile (0.961) | scenario (0.018) | 0.195 |
| g6_gate | 2028 | click | item_id (0.849) | user_id (0.124) | 0.467 |
| g6_gate | 2028 | conversion | item_attr (0.984) | scenario (0.008) | 0.097 |
| g6_gate | 2029 | click | item_id (0.651) | scenario (0.311) | 0.740 |
| g6_gate | 2029 | conversion | user_profile (0.991) | scenario (0.006) | 0.061 |
| g6_residual | 2025 | click | user_id (0.673) | cross (0.139) | 0.815 |
| g6_residual | 2025 | conversion | user_profile (0.536) | item_attr (0.227) | 1.224 |
| g6_residual | 2026 | click | scenario (0.905) | cross (0.033) | 0.406 |
| g6_residual | 2026 | conversion | item_attr (0.614) | user_profile (0.258) | 1.012 |
| g6_residual | 2027 | click | user_id (0.601) | cross (0.159) | 1.103 |
| g6_residual | 2027 | conversion | user_profile (0.920) | scenario (0.024) | 0.381 |
| g6_residual | 2028 | click | user_id (0.956) | item_id (0.014) | 0.185 |
| g6_residual | 2028 | conversion | user_profile (0.710) | scenario (0.170) | 0.806 |
| g6_residual | 2029 | click | item_id (0.947) | user_id (0.030) | 0.170 |
| g6_residual | 2029 | conversion | user_profile (0.988) | cross (0.005) | 0.076 |

Multi-seed aggregate (from `analysis/*_seeds.md`):
- g6_gate conversion gate entropy 0.121 ± 0.050 nats (near one-hot in every seed); click 0.764 ± 0.343.
- g6_residual conversion gate entropy 0.700 ± 0.468; click 0.536 ± 0.410.
- Residual λ (weight on the plain mean path): click 0.507 ± 0.004, conversion 0.499 ± 0.005 across seeds, i.e. it barely moves from its 0.5 initialisation.

## Gradient audit (g6_mean seed 2025 checkpoint, 4 validation batches, no weight updates)

Under EQ loss, click logloss ≈0.16–0.20 vs conversion ≈0.0003–0.007 per batch. Mean trunk gradient L2: click 4.31e-2, conversion 4.03e-3, so **click drives the shared trunk ~10.7× harder than conversion**. Single checkpoint, 4 batches: indicative, not a population estimate. Optional `NORM` loss weighting exists in code but was not used in this suite.

## Honest takeaway

1. **No variant separates from seed noise on validation.** Spread across variants (0.6248–0.6301 avg AUC) is comparable to within-variant std (0.0015–0.0044). Nothing here supports a headline "X beats Y" claim.
2. **The "6-token semantic grouping is the main gain" story is not confirmed.** g6_mean vs g3_mean is +0.0007 (3/5 seeds). The random 6-way partition (0.6295) actually scores above the semantic one (0.6277), and the sequential partition (0.6267) is at g3 level. If token count or grouping matters at all, these data can't say that the *semantic* choice is what helps. The previous test-set sweep result (g6 ≈ +0.0019) should be treated as not replicated under validation-first protocol.
3. **Residual vs mean at T=6:** +0.0014 avg (3/5, p≈0.37), not significant. The residual's real, reproducible effect is diagnostic, not AUC: it softens the conversion-gate collapse (entropy 0.70 vs 0.12). But λ stays at ≈0.5, so the "learnable λ" isn't actually learning a mix; it's effectively a fixed 50/50 blend.
4. **Gate collapse is robust but seed-dependent.** The plain per-task gate's conversion head collapses to a single token (0.96–0.99 weight) in 5/5 seeds, but to a *different* token in different seeds (item_id, item_attr, user_profile). That's a clean, honest finding: the collapse is an identifiability/optimisation artefact, not the model discovering "the" conversion feature.
5. **All RankMixer variants edge PLE on validation** (by +0.002 to +0.005 avg; g6_gate wins 5/5 seeds, p≈0.14), with PLE also the noisiest (one bad seed). Directionally consistent, still not significant at n=5.
6. **Click dominates the shared-trunk gradient ~10×** under EQ loss, which is a plausible reason conversion-side design changes show up mostly as variance rather than consistent gains; loss normalisation is the natural next experiment.

For the resume, this supports selling the research chain and the falsification itself (a claimed gain that didn't survive multi-seed validation-first testing, seed-dependent gate collapse, gradient imbalance), not an AUC improvement. Choose and freeze the design based on these validation numbers before running final test once.

## Files
STATUS, ALL_DONE, rg_driver.log, rigor_summary.csv, grad_audit.md, analysis/ (per-seed gate md/json/png, *_seeds.md aggregates, analysis logs), logs/ (*.log/*.time/*.exit for all 35 runs), configs/ (per-run config groups).
