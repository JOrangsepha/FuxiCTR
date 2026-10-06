# MT-RankMixer token gate weights

Checkpoint: `/root/autodl-tmp/ac_out/checkpoints/AliCCP_x1/MTRankMixer_aliccp_semantic_residual_s2025.model`

Rows used: 500000 (cap 500000). task_pooling: `residual`. Sample standard deviation (n - 1). Gates sum to 1 over tokens inside each task.

| slice | task | n | user mean | user std | item mean | item std | context mean | context std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | click | 500000 | 0.676654 | 0.202985 | 0.271431 | 0.191984 | 0.051916 | 0.047561 |
| all | conversion | 500000 | 0.952436 | 0.041424 | 0.032427 | 0.031043 | 0.015138 | 0.017751 |
| click=1 | click | 23552 | 0.731947 | 0.173586 | 0.222869 | 0.162896 | 0.045185 | 0.039620 |
| click=1 | conversion | 23552 | 0.959525 | 0.038798 | 0.026880 | 0.026720 | 0.013595 | 0.019136 |
| conversion=1 | click | 159 | 0.647823 | 0.187384 | 0.304302 | 0.173286 | 0.047875 | 0.039403 |
| conversion=1 | conversion | 159 | 0.948362 | 0.072955 | 0.035760 | 0.040213 | 0.015878 | 0.039380 |

## Mean gate entropy (nats)

| slice | task | n | entropy |
| --- | --- | --- | --- |
| all | click | 500000 | 0.659136 |
| all | conversion | 500000 | 0.203124 |
| click=1 | click | 23552 | 0.614552 |
| click=1 | conversion | 23552 | 0.179609 |
| conversion=1 | click | 159 | 0.698469 |
| conversion=1 | conversion | 159 | 0.206141 |

## Residual mix weight λ

h = λ * mean(tokens) + (1 - λ) * gated mix. λ is a learned sigmoid scalar per task.

| task | lambda |
| --- | --- |
| click | 0.531656 |
| conversion | 0.498624 |

![gate weights](../../../docs/img/rankmixer/gate_residual_s2025.png)
