# MT-RankMixer token gate weights

Checkpoint: `/root/autodl-tmp/ac_out/checkpoints/AliCCP_x1/MTRankMixer_aliccp_semantic_entropy_s2025.model`

Rows used: 500000 (cap 500000). task_pooling: `gate`. Sample standard deviation (n - 1). Gates sum to 1 over tokens inside each task.

| slice | task | n | user mean | user std | item mean | item std | context mean | context std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | click | 500000 | 0.341014 | 0.012763 | 0.322071 | 0.008551 | 0.336916 | 0.011546 |
| all | conversion | 500000 | 0.330860 | 0.008408 | 0.334056 | 0.010508 | 0.335084 | 0.009541 |
| click=1 | click | 23552 | 0.339042 | 0.012411 | 0.323001 | 0.008217 | 0.337957 | 0.011895 |
| click=1 | conversion | 23552 | 0.331577 | 0.009207 | 0.332514 | 0.010900 | 0.335909 | 0.009633 |
| conversion=1 | click | 159 | 0.342376 | 0.013537 | 0.322262 | 0.008967 | 0.335362 | 0.013446 |
| conversion=1 | conversion | 159 | 0.329961 | 0.015377 | 0.336013 | 0.014593 | 0.334026 | 0.009517 |

## Mean gate entropy (nats)

| slice | task | n | entropy |
| --- | --- | --- | --- |
| all | click | 500000 | 1.097763 |
| all | conversion | 500000 | 1.098189 |
| click=1 | click | 23552 | 1.097829 |
| click=1 | conversion | 23552 | 1.098151 |
| conversion=1 | click | 159 | 1.097640 |
| conversion=1 | conversion | 159 | 1.097740 |


![gate weights](../../../docs/img/rankmixer/gate_entropy_s2025.png)
