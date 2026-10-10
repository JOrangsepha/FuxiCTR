# MT-RankMixer token gate weights

Stage: validation. Use these numbers for analysis and model selection. The test split stays unread until the design is frozen.

Checkpoint: `/root/autodl-tmp/nf_out/checkpoints/AliCCP_x1/MTR_nf_g6_gate_normfloor_s2027.model`

Rows used: 500000 (cap 500000). task_pooling: `gate`. Sample standard deviation (n - 1) is across rows inside this checkpoint, not across seeds. Gates sum to 1 over tokens inside each task.

| slice | task | n | user_id mean | user_id std | user_profile mean | user_profile std | item_id mean | item_id std | item_attr mean | item_attr std | cross mean | cross std | scenario mean | scenario std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | click | 500000 | 0.440217 | 0.148575 | 0.045355 | 0.024942 | 0.001113 | 0.000887 | 0.002050 | 0.001914 | 0.507632 | 0.149661 | 0.003633 | 0.002362 |
| all | conversion | 500000 | 0.001522 | 0.002031 | 0.982888 | 0.015912 | 0.000110 | 0.000083 | 0.000098 | 0.000099 | 0.014425 | 0.014917 | 0.000956 | 0.000907 |
| click=1 | click | 23611 | 0.448072 | 0.134825 | 0.054665 | 0.028007 | 0.001379 | 0.001118 | 0.002682 | 0.002280 | 0.488798 | 0.135468 | 0.004404 | 0.002631 |
| click=1 | conversion | 23611 | 0.001443 | 0.001665 | 0.980099 | 0.016492 | 0.000115 | 0.000087 | 0.000083 | 0.000065 | 0.017120 | 0.015670 | 0.001140 | 0.000985 |
| conversion=1 | click | 187 | 0.395153 | 0.126060 | 0.049930 | 0.025158 | 0.001310 | 0.000980 | 0.001928 | 0.001731 | 0.547185 | 0.129607 | 0.004494 | 0.002555 |
| conversion=1 | conversion | 187 | 0.001573 | 0.001267 | 0.976363 | 0.021937 | 0.000111 | 0.000063 | 0.000096 | 0.000068 | 0.020603 | 0.021229 | 0.001255 | 0.000952 |

## Mean gate entropy (nats)

| slice | task | n | entropy |
| --- | --- | --- | --- |
| all | click | 500000 | 0.828533 |
| all | conversion | 500000 | 0.089619 |
| click=1 | click | 23611 | 0.868592 |
| click=1 | conversion | 23611 | 0.101480 |
| conversion=1 | click | 187 | 0.847771 |
| conversion=1 | conversion | 187 | 0.114203 |

