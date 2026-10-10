# Trunk gradient audit: raw (EQ) vs NORM-weighted per-task contributions

Validation split, 16 batches of 8192 per checkpoint, eval mode, no weight updates. `raw` = ||grad_trunk L_k|| (what EQ sums). `NORM` = ||grad_trunk L_k|| / L_k (what `loss_weight: NORM` sums). Ratio = click / conversion of the batch-mean norms; median ratio is over batches. Trunk = embedding_layer, tokenizer, encoder.

| checkpoint | trained with | mean click BCE | mean conv BCE | raw click | raw conv | raw ratio (median) | NORM click | NORM conv | NORM ratio (median) | ratio under its own loss_weight | ||grad|| EQ total | ||grad|| NORM total |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| NORM_FLOOR g6_mean s2025 | NORM_FLOOR | 0.1881 | 0.00325 | 2.373e-02 | 2.633e-03 | 9.01 (8.84) | 1.257e-01 | 1.332e+00 | 0.094 (0.156) | 9.015 | 2.437e-02 | 1.362e+00 |
| NORM g6_mean s2025 | NORM | 0.1879 | 0.01228 | 2.333e-02 | 5.618e-06 | 4153.43 (6085.50) | 1.221e-01 | 4.723e-04 | 258.469 (318.011) | 258.469 | 2.333e-02 | 1.220e-01 |
| EQ g6_mean s2025 | EQ | 0.1883 | 0.00318 | 3.116e-02 | 3.226e-03 | 9.66 (9.22) | 1.627e-01 | 1.842e+00 | 0.088 (0.184) | 9.659 | 3.162e-02 | 1.865e+00 |

Ratio > 1 means click pushes the shared trunk harder than conversion. Small-sample diagnostic (one checkpoint per row), not a population estimate.
