# Trunk gradient norms

Checkpoint `/root/autodl-tmp/fu_out/checkpoints/AliCCP_x1/MTR_fu_g6_mean_w10_s2025.model`. Weights were not updated.

`EQ` is the unnormalized sum of per-task mean BCE. It does not divide by the number of tasks and it does not rescale by the loss magnitude. On Ali-CCP, click logloss is about 0.16 and conversion logloss is about 0.002, so click can dominate `∇_trunk`. The trunk is `embedding_layer`, `tokenizer`, and `encoder`. Towers, the token gate, and residual λ are outside it.

Optional hooks, not used by the rigor suite: `loss_weight` may be a list of manual weights, or `NORM` (each task loss divided by its detached absolute value, then summed). Training defaults stay `EQ`.

Stage: `valid`.

| batch | task | mean BCE | trunk grad L2 | full grad L2 |
| --- | --- | --- | --- | --- |
| 0 | click | 1.974953e-01 | 2.914409e-02 | 3.803381e-02 |
| 0 | conversion | 5.465457e-04 | 3.767198e-03 | 6.412750e-03 |
| 0 | EQ sum | 1.980418e-01 | 2.928744e-02 | |
| 1 | click | 1.841721e-01 | 2.679366e-02 | 3.371378e-02 |
| 1 | conversion | 6.700196e-03 | 2.581041e-03 | 3.164319e-03 |
| 1 | EQ sum | 1.908723e-01 | 2.682998e-02 | |
| 2 | click | 1.791836e-01 | 2.989218e-02 | 4.328621e-02 |
| 2 | conversion | 5.406394e-03 | 2.697688e-03 | 2.744464e-03 |
| 2 | EQ sum | 1.845900e-01 | 3.046806e-02 | |
| 3 | click | 2.036274e-01 | 3.552001e-02 | 4.726264e-02 |
| 3 | conversion | 3.210036e-03 | 2.030314e-03 | 2.681034e-03 |
| 3 | EQ sum | 2.068375e-01 | 3.483237e-02 | |

Mean trunk grad L2 over 4 batch(es): click 3.033749e-02, conversion 2.769060e-03, click/conversion ratio 10.9559.
