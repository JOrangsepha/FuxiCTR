# Trunk gradient audit: raw (EQ) vs NORM-weighted per-task contributions

Validation split, 16 batches of 8192 per checkpoint, eval mode, no weight updates. `raw` = ||grad_trunk L_k|| (what EQ sums). `NORM` = ||grad_trunk L_k|| / L_k (what `loss_weight: NORM` sums). Ratio = click / conversion of the batch-mean norms; median ratio is over batches. Trunk = embedding_layer, tokenizer, encoder.

| checkpoint | trained with | mean click BCE | mean conv BCE | raw click | raw conv | raw ratio (median) | NORM click | NORM conv | NORM ratio (median) | ratio under its own loss_weight | ||grad|| EQ total | ||grad|| NORM total |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| EQ g6_mean s2025 | EQ | 0.1883 | 0.00318 | 3.116e-02 | 3.226e-03 | 9.66 (9.22) | 1.627e-01 | 1.842e+00 | 0.088 (0.184) | 9.659 | 3.162e-02 | 1.865e+00 |
| MTR_fu_g6_mean_w10 s2025 | [1.0, 10.0] | 0.1881 | 0.00310 | 2.899e-02 | 2.825e-03 | 10.26 (10.49) | 1.557e-01 | 1.567e+00 | 0.099 (0.157) | 1.026 | 2.965e-02 | 1.608e+00 |

Ratio > 1 means click pushes the shared trunk harder than conversion. Small-sample diagnostic (one checkpoint per row), not a population estimate.
