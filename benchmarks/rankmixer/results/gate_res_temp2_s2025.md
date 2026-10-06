# MT-RankMixer token gate weights

Checkpoint: `/root/autodl-tmp/sw_out/checkpoints/AliCCP_x1/MTR_sw_res_temp2_s2025.model`

Rows used: 500000 (cap 500000). task_pooling: `residual`. Sample standard deviation (n - 1). Gates sum to 1 over tokens inside each task.

| slice | task | n | user mean | user std | item mean | item std | context mean | context std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | click | 500000 | 0.556061 | 0.146097 | 0.321533 | 0.125221 | 0.122406 | 0.052369 |
| all | conversion | 500000 | 0.665507 | 0.068818 | 0.206966 | 0.057833 | 0.127527 | 0.032811 |
| click=1 | click | 23552 | 0.603181 | 0.129437 | 0.282648 | 0.108409 | 0.114171 | 0.048181 |
| click=1 | conversion | 23552 | 0.664585 | 0.064142 | 0.207235 | 0.055547 | 0.128180 | 0.033424 |
| conversion=1 | click | 159 | 0.545845 | 0.137242 | 0.334465 | 0.113649 | 0.119690 | 0.053837 |
| conversion=1 | conversion | 159 | 0.665699 | 0.069819 | 0.211009 | 0.061192 | 0.123292 | 0.029975 |

## Mean gate entropy (nats)

| slice | task | n | entropy |
| --- | --- | --- | --- |
| all | click | 500000 | 0.894404 |
| all | conversion | 500000 | 0.843908 |
| click=1 | click | 23552 | 0.866533 |
| click=1 | conversion | 23552 | 0.846302 |
| conversion=1 | click | 159 | 0.903759 |
| conversion=1 | conversion | 159 | 0.841598 |

## Residual mix weight λ

h = λ * mean(tokens) + (1 - λ) * gated mix. λ is a learned sigmoid scalar per task.

| task | lambda |
| --- | --- |
| click | 0.522207 |
| conversion | 0.496641 |

![gate weights](../../../docs/img/rankmixer/gate_res_temp2_s2025.png)
