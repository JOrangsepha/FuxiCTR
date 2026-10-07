### Evidence: per-batch NORM collapses the conversion head (first launch, 2026-10-07 16:30 CST)

`loss_weight: NORM` = sum_k L_k / stopgrad(L_k), recomputed per batch. Two g6_mean NORM runs (seeds 2025, 2026)
were started and stopped after epoch 1 to reorder the queue. Epoch-1 validation:

| run | train loss (epoch mean; NORM ⇒ should be 2.0) | click AUC | conv AUC |
|---|---|---|---|
| MTR_fu_g6_mean_norm_s2025 | 1.843 | 0.6212 | 0.5007 |
| MTR_fu_g6_mean_norm_s2026 | 1.842 | 0.6202 | 0.5327 |

(EQ g6_mean same seeds: conv AUC 0.6445 / 0.6356.) A mean loss of 1.84 means the conversion BCE was exactly 0.0 in
fp32 on ~16% of batches, i.e. every conversion prediction in those batches underflowed (1 − p == 1). That matches the
share of 8192-row batches with no conversion at all (~2.2e-4 positives/row ⇒ ~1.8 per batch ⇒ P(0) ≈ 17%). On such a
batch L_conv ≈ mean(p) and NORM multiplies its gradient by 1/L_conv, so the logit gradient is ≈ p / mean(p): it does
not vanish as p → 0 and keeps pushing conversion logits to −∞. Logs: /root/autodl-tmp/fu_evidence/norm_perbatch_attempt_logs/.
