# MT-RankMixer token gate weights

Stage: validation. Use these numbers for analysis and model selection. The test split stays unread until the design is frozen.

Checkpoint: `/root/autodl-tmp/nf_out/checkpoints/AliCCP_x1/MTR_nf_g6_residual_normfloor_s2027.model`

Rows used: 500000 (cap 500000). task_pooling: `residual`. Sample standard deviation (n - 1) is across rows inside this checkpoint, not across seeds. Gates sum to 1 over tokens inside each task.

| slice | task | n | user_id mean | user_id std | user_profile mean | user_profile std | item_id mean | item_id std | item_attr mean | item_attr std | cross mean | cross std | scenario mean | scenario std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | click | 500000 | 0.371504 | 0.251728 | 0.339133 | 0.177348 | 0.073604 | 0.027596 | 0.068170 | 0.035160 | 0.120710 | 0.101561 | 0.026878 | 0.038535 |
| all | conversion | 500000 | 0.012276 | 0.012376 | 0.951236 | 0.028272 | 0.017221 | 0.007037 | 0.009962 | 0.011035 | 0.006013 | 0.006335 | 0.003293 | 0.005044 |
| click=1 | click | 23611 | 0.352149 | 0.252931 | 0.369983 | 0.181719 | 0.074871 | 0.028601 | 0.074790 | 0.037343 | 0.104035 | 0.092935 | 0.024171 | 0.019065 |
| click=1 | conversion | 23611 | 0.011900 | 0.014149 | 0.954462 | 0.026964 | 0.016020 | 0.006385 | 0.009812 | 0.010475 | 0.005278 | 0.005906 | 0.002529 | 0.003897 |
| conversion=1 | click | 187 | 0.258517 | 0.227610 | 0.425242 | 0.168947 | 0.083735 | 0.027583 | 0.072953 | 0.041827 | 0.139117 | 0.106997 | 0.020437 | 0.017454 |
| conversion=1 | conversion | 187 | 0.008059 | 0.010424 | 0.966167 | 0.019540 | 0.014592 | 0.005188 | 0.005713 | 0.006747 | 0.003801 | 0.003197 | 0.001668 | 0.002368 |

## Mean gate entropy (nats)

| slice | task | n | entropy |
| --- | --- | --- | --- |
| all | click | 500000 | 1.256192 |
| all | conversion | 500000 | 0.250758 |
| click=1 | click | 23611 | 1.248337 |
| click=1 | conversion | 23611 | 0.235405 |
| conversion=1 | click | 187 | 1.277453 |
| conversion=1 | conversion | 187 | 0.183774 |

## Residual mix weight λ

h = λ * mean(tokens) + (1 - λ) * gated mix. λ is a learned sigmoid scalar per task.

| task | lambda |
| --- | --- |
| click | 0.514585 |
| conversion | 0.507616 |
