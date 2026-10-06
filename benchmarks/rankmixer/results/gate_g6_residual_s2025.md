# MT-RankMixer token gate weights

Checkpoint: `/root/autodl-tmp/sw_out/checkpoints/AliCCP_x1/MTR_sw_g6_residual_s2025.model`

Rows used: 500000 (cap 500000). task_pooling: `residual`. Sample standard deviation (n - 1). Gates sum to 1 over tokens inside each task.

| slice | task | n | token0 mean | token0 std | token1 mean | token1 std | token2 mean | token2 std | token3 mean | token3 std | token4 mean | token4 std | token5 mean | token5 std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | click | 500000 | 0.592475 | 0.241905 | 0.027433 | 0.012083 | 0.015348 | 0.008263 | 0.027242 | 0.025257 | 0.187587 | 0.205514 | 0.149916 | 0.140869 |
| all | conversion | 500000 | 0.040151 | 0.022142 | 0.567906 | 0.092607 | 0.133308 | 0.062275 | 0.163197 | 0.090042 | 0.057036 | 0.052308 | 0.038401 | 0.024608 |
| click=1 | click | 23552 | 0.646559 | 0.221468 | 0.029163 | 0.013347 | 0.015846 | 0.009320 | 0.029244 | 0.027410 | 0.149030 | 0.186228 | 0.130158 | 0.123788 |
| click=1 | conversion | 23552 | 0.038519 | 0.021004 | 0.558662 | 0.090215 | 0.138023 | 0.065164 | 0.182303 | 0.090858 | 0.049290 | 0.050337 | 0.033204 | 0.022640 |
| conversion=1 | click | 159 | 0.533936 | 0.248541 | 0.029743 | 0.013721 | 0.015894 | 0.009120 | 0.031162 | 0.027821 | 0.198562 | 0.217013 | 0.190703 | 0.154095 |
| conversion=1 | conversion | 159 | 0.039545 | 0.025572 | 0.580644 | 0.094855 | 0.127857 | 0.060945 | 0.152176 | 0.090633 | 0.056035 | 0.054498 | 0.043743 | 0.026665 |

## Mean gate entropy (nats)

| slice | task | n | entropy |
| --- | --- | --- | --- |
| all | click | 500000 | 0.936289 |
| all | conversion | 500000 | 1.218579 |
| click=1 | click | 23552 | 0.895100 |
| click=1 | conversion | 23552 | 1.212471 |
| conversion=1 | click | 159 | 0.996936 |
| conversion=1 | conversion | 159 | 1.201271 |

## Residual mix weight λ

h = λ * mean(tokens) + (1 - λ) * gated mix. λ is a learned sigmoid scalar per task.

| task | lambda |
| --- | --- |
| click | 0.510613 |
| conversion | 0.496720 |

![gate weights](../../../docs/img/rankmixer/gate_g6_residual_s2025.png)
