# MT-RankMixer token gate weights

Checkpoint: `/root/autodl-tmp/sw_out/checkpoints/AliCCP_x1/MTR_sw_res_ent0001_s2025.model`

Rows used: 500000 (cap 500000). task_pooling: `residual`. Sample standard deviation (n - 1). Gates sum to 1 over tokens inside each task.

| slice | task | n | user mean | user std | item mean | item std | context mean | context std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | click | 500000 | 0.334763 | 0.044568 | 0.347068 | 0.035562 | 0.318168 | 0.032211 |
| all | conversion | 500000 | 0.329905 | 0.017785 | 0.327544 | 0.017739 | 0.342551 | 0.019983 |
| click=1 | click | 23552 | 0.335607 | 0.043741 | 0.346208 | 0.032583 | 0.318185 | 0.031497 |
| click=1 | conversion | 23552 | 0.331687 | 0.018402 | 0.326890 | 0.018552 | 0.341423 | 0.020566 |
| conversion=1 | click | 159 | 0.336122 | 0.053085 | 0.350503 | 0.038258 | 0.313375 | 0.034503 |
| conversion=1 | conversion | 159 | 0.330304 | 0.024296 | 0.327591 | 0.019836 | 0.342105 | 0.022891 |

## Mean gate entropy (nats)

| slice | task | n | entropy |
| --- | --- | --- | --- |
| all | click | 500000 | 1.091576 |
| all | conversion | 500000 | 1.096873 |
| click=1 | click | 23552 | 1.092072 |
| click=1 | conversion | 23552 | 1.096786 |
| conversion=1 | click | 159 | 1.089505 |
| conversion=1 | conversion | 159 | 1.096098 |

## Residual mix weight λ

h = λ * mean(tokens) + (1 - λ) * gated mix. λ is a learned sigmoid scalar per task.

| task | lambda |
| --- | --- |
| click | 0.553487 |
| conversion | 0.493151 |
