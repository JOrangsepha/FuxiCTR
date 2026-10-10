# NORM_FLOOR real-data validation results

OUT=`/root/autodl-tmp/nf_out`
Protocol: `--skip_test`, patience=3, epochs≤10, AliCCP_x1 **validation only** (test unread).
Code: `loss_weight: NORM_FLOOR` = per-batch NORM with denominator `clamp_min(1e-2)`.

## EQ baseline (prior rigor, same protocol)

| variant | valid click AUC | valid conv AUC | valid avg AUC |
|---------|-----------------|----------------|---------------|
| g6_mean [EQ] | 0.61960 ± 0.00149 | 0.63589 ± 0.00568 | 0.62775 ± 0.00348 |

W[1,10] numbers: see /root/autodl-tmp/fu_out/followup_results.md

## This suite (mean ± std over seeds)

| variant | n | click AUC | conv AUC | avg AUC | clicked CVR AUC | conv&lt;0.55 |
|---------|---|-----------|----------|---------|-----------------|------------|
| MTR_nf_g6_mean_normfloor | 5 | 0.61998 ± 0.00167 | 0.64082 ± 0.00359 | 0.63040 ± 0.00229 | 0.61048 ± 0.01007 | 0/5 |
| MTR_nf_g6_mean_norm | 5 | 0.61994 ± 0.00042 | 0.56673 ± 0.03616 | 0.59334 ± 0.01816 | 0.50219 ± 0.06199 | 1/5 |
| MTR_nf_g6_gate_normfloor | 5 | 0.62017 ± 0.00150 | 0.64348 ± 0.00570 | 0.63183 ± 0.00305 | 0.61457 ± 0.00689 | 0/5 |
| MTR_nf_g6_residual_normfloor | 5 | 0.62019 ± 0.00131 | 0.64554 ± 0.00373 | 0.63287 ± 0.00198 | 0.61806 ± 0.00663 | 0/5 |

## Per-seed detail

| expid | exit | secs | best_ep | click | conv | avg | clicked_CVR | loss_w | collapsed |
|-------|------|------|---------|-------|------|-----|-------------|--------|-----------|
| MTR_nf_g6_gate_normfloor_s2025 | 0 | 864 | 1 | 0.62094 | 0.64997 | 0.63546 | 0.62165 | NORM_FLOOR | False |
| MTR_nf_g6_gate_normfloor_s2026 | 0 | 844 | 1 | 0.62013 | 0.64240 | 0.63127 | 0.61293 | NORM_FLOOR | False |
| MTR_nf_g6_gate_normfloor_s2027 | 0 | 879 | 1 | 0.61758 | 0.64196 | 0.62977 | 0.60688 | NORM_FLOOR | False |
| MTR_nf_g6_gate_normfloor_s2028 | 0 | 874 | 1 | 0.62107 | 0.64774 | 0.63440 | 0.62185 | NORM_FLOOR | False |
| MTR_nf_g6_gate_normfloor_s2029 | 0 | 859 | 1 | 0.62112 | 0.63533 | 0.62823 | 0.60955 | NORM_FLOOR | False |
| MTR_nf_g6_mean_norm_s2025 | 0 | 844 | 1 | 0.62001 | 0.56615 | 0.59308 | 0.54191 | NORM | False |
| MTR_nf_g6_mean_norm_s2026 | 0 | 837 | 1 | 0.61965 | 0.57008 | 0.59487 | 0.47581 | NORM | False |
| MTR_nf_g6_mean_norm_s2027 | 0 | 845 | 1 | 0.61945 | 0.52005 | 0.56975 | 0.41348 | NORM | True |
| MTR_nf_g6_mean_norm_s2028 | 0 | 833 | 1 | 0.62007 | 0.62091 | 0.62049 | 0.57441 | NORM | False |
| MTR_nf_g6_mean_norm_s2029 | 0 | 837 | 1 | 0.62054 | 0.55648 | 0.58851 | 0.50533 | NORM | False |
| MTR_nf_g6_mean_normfloor_s2025 | 0 | 858 | 1 | 0.61728 | 0.63968 | 0.62848 | 0.60457 | NORM_FLOOR | False |
| MTR_nf_g6_mean_normfloor_s2026 | 0 | 866 | 1 | 0.62030 | 0.63853 | 0.62942 | 0.60346 | NORM_FLOOR | False |
| MTR_nf_g6_mean_normfloor_s2027 | 0 | 869 | 1 | 0.62142 | 0.64043 | 0.63092 | 0.61483 | NORM_FLOOR | False |
| MTR_nf_g6_mean_normfloor_s2028 | 0 | 866 | 1 | 0.61964 | 0.63841 | 0.62902 | 0.60326 | NORM_FLOOR | False |
| MTR_nf_g6_mean_normfloor_s2029 | 0 | 849 | 1 | 0.62126 | 0.64706 | 0.63416 | 0.62630 | NORM_FLOOR | False |
| MTR_nf_g6_residual_normfloor_s2025 | 0 | 864 | 1 | 0.62051 | 0.64153 | 0.63102 | 0.60688 | NORM_FLOOR | False |
| MTR_nf_g6_residual_normfloor_s2026 | 0 | 878 | 1 | 0.62180 | 0.64438 | 0.63309 | 0.61806 | NORM_FLOOR | False |
| MTR_nf_g6_residual_normfloor_s2027 | 0 | 865 | 1 | 0.61825 | 0.64321 | 0.63073 | 0.62042 | NORM_FLOOR | False |
| MTR_nf_g6_residual_normfloor_s2028 | 0 | 882 | 1 | 0.61973 | 0.65083 | 0.63528 | 0.62074 | NORM_FLOOR | False |
| MTR_nf_g6_residual_normfloor_s2029 | 0 | 892 | 1 | 0.62067 | 0.64774 | 0.63420 | 0.62422 | NORM_FLOOR | False |

## vs EQ g6_mean (descriptive Δ of means)

- **MTR_nf_g6_mean_normfloor**: avg AUC 0.63040 (ΔEQ +0.00265); conv AUC 0.64082 (EQ conv 0.63589)
- **MTR_nf_g6_mean_norm**: avg AUC 0.59334 (ΔEQ -0.03441); conv AUC 0.56673 (EQ conv 0.63589)
- **MTR_nf_g6_gate_normfloor**: avg AUC 0.63183 (ΔEQ +0.00408); conv AUC 0.64348 (EQ conv 0.63589)
- **MTR_nf_g6_residual_normfloor**: avg AUC 0.63287 (ΔEQ +0.00512); conv AUC 0.64554 (EQ conv 0.63589)

## Collapse verdict

- Plain NORM: 1/5 seeds with valid conv AUC < 0.55 (prior collapse ~0.50–0.53).
- NORM_FLOOR: 5/5 seeds with valid conv AUC ≥ 0.60.
- Mean conv AUC NORM_FLOOR 0.64082 vs NORM 0.56673.

## Gradient audits

### grad_audit_normfloor.md

# Trunk gradient audit: raw (EQ) vs NORM-weighted per-task contributions

Validation split, 16 batches of 8192 per checkpoint, eval mode, no weight updates. `raw` = ||grad_trunk L_k|| (what EQ sums). `NORM` = ||grad_trunk L_k|| / L_k (what `loss_weight: NORM` sums). Ratio = click / conversion of the batch-mean norms; median ratio is over batches. Trunk = embedding_layer, tokenizer, encoder.

| checkpoint | trained with | mean click BCE | mean conv BCE | raw click | raw conv | raw ratio (median) | NORM click | NORM conv | NORM ratio (median) | ratio under its own loss_weight | ||grad|| EQ total | ||grad|| NORM total |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| NORM_FLOOR g6_mean s2025 | NORM_FLOOR | 0.1881 | 0.00325 | 2.373e-02 | 2.633e-03 | 9.01 (8.84) | 1.257e-01 | 1.332e+00 | 0.094 (0.156) | 9.015 | 2.437e-02 | 1.362e+00 |
| NORM g6_mean s2025 | NORM | 0.1879 | 0.01228 | 2.333e-02 | 5.618e-06 | 4153.43 (6085.50) | 1.221e-01 | 4.723e-04 | 258.469 (318.011) | 258.469 | 2.333e-02 | 1.220e-01 |
| EQ g6_mean s2025 | EQ | 0.1883 | 0.00318 | 3.116e-02 | 3.226e-03 | 9.66 (9.22) | 1.627e-01 | 1.842e+00 | 0.088 (0.184) | 9.659 | 3.162e-02 | 1.865e+00 |

Ratio > 1 means click pushes the shared trunk harder than conversion. Small-sample diagnostic (one checkpoint per row), not a population estimate.


- `grad_audit_norm.md`: missing
- `grad_audit_norm_raw.md`: missing
