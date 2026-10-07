# Trunk gradient norms

Checkpoint `/root/autodl-tmp/rg_out/checkpoints/AliCCP_x1/MTR_rg_g6_mean_s2025.model`. Weights were not updated.

`EQ` is the unnormalized sum of per-task mean BCE. It does not divide by the number of tasks and it does not rescale by the loss magnitude. On Ali-CCP, click logloss is about 0.16 and conversion logloss is about 0.002, so click can dominate `∇_trunk`. The trunk is `embedding_layer`, `tokenizer`, and `encoder`. Towers, the token gate, and residual λ are outside it.

Optional hooks, not used by the rigor suite: `loss_weight` may be a list of manual weights, or `NORM` (each task loss divided by its detached absolute value, then summed). Training defaults stay `EQ`.

Stage: `valid`.

| batch | task | mean BCE | trunk grad L2 | full grad L2 |
| --- | --- | --- | --- | --- |
| 0 | click | 1.994854e-01 | 4.367441e-02 | 8.732463e-02 |
| 0 | conversion | 3.527260e-04 | 3.806648e-03 | 5.282336e-03 |
| 0 | EQ sum | 1.998381e-01 | 4.364979e-02 | |
| 1 | click | 1.841090e-01 | 2.862097e-02 | 4.393965e-02 |
| 1 | conversion | 6.954777e-03 | 5.592431e-03 | 7.258824e-03 |
| 1 | EQ sum | 1.910638e-01 | 2.996712e-02 | |
| 2 | click | 2.036224e-01 | 3.919430e-02 | 7.056644e-02 |
| 2 | conversion | 5.704362e-03 | 4.667883e-03 | 5.805979e-03 |
| 2 | EQ sum | 2.093268e-01 | 4.013031e-02 | |
| 3 | click | 2.064197e-01 | 6.105788e-02 | 1.061460e-01 |
| 3 | conversion | 3.225296e-03 | 2.052153e-03 | 2.083440e-03 |
| 3 | EQ sum | 2.096450e-01 | 6.099360e-02 | |

Mean trunk grad L2 over 4 batch(es): click 4.313689e-02, conversion 4.029779e-03, click/conversion ratio 10.7045.
