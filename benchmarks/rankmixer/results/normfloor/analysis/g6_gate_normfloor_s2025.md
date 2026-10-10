# MT-RankMixer token gate weights

Stage: validation. Use these numbers for analysis and model selection. The test split stays unread until the design is frozen.

Checkpoint: `/root/autodl-tmp/nf_out/checkpoints/AliCCP_x1/MTR_nf_g6_gate_normfloor_s2025.model`

Rows used: 500000 (cap 500000). task_pooling: `gate`. Sample standard deviation (n - 1) is across rows inside this checkpoint, not across seeds. Gates sum to 1 over tokens inside each task.

| slice | task | n | user_id mean | user_id std | user_profile mean | user_profile std | item_id mean | item_id std | item_attr mean | item_attr std | cross mean | cross std | scenario mean | scenario std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | click | 500000 | 0.273318 | 0.143969 | 0.007926 | 0.004898 | 0.190926 | 0.069086 | 0.009579 | 0.011977 | 0.003987 | 0.002476 | 0.514264 | 0.185879 |
| all | conversion | 500000 | 0.000200 | 0.000411 | 0.000075 | 0.000236 | 0.999213 | 0.001857 | 0.000026 | 0.000026 | 0.000158 | 0.000174 | 0.000330 | 0.001668 |
| click=1 | click | 23611 | 0.286259 | 0.135589 | 0.009227 | 0.005874 | 0.215295 | 0.074146 | 0.012951 | 0.014901 | 0.004482 | 0.002625 | 0.471786 | 0.180187 |
| click=1 | conversion | 23611 | 0.000163 | 0.000322 | 0.000076 | 0.000242 | 0.999410 | 0.000862 | 0.000024 | 0.000023 | 0.000176 | 0.000194 | 0.000150 | 0.000648 |
| conversion=1 | click | 187 | 0.240103 | 0.123211 | 0.008844 | 0.005944 | 0.188877 | 0.065326 | 0.007963 | 0.010232 | 0.004322 | 0.002674 | 0.549890 | 0.169871 |
| conversion=1 | conversion | 187 | 0.000240 | 0.000337 | 0.000082 | 0.000046 | 0.999318 | 0.000715 | 0.000023 | 0.000028 | 0.000185 | 0.000165 | 0.000153 | 0.000536 |

## Mean gate entropy (nats)

| slice | task | n | entropy |
| --- | --- | --- | --- |
| all | click | 500000 | 1.022758 |
| all | conversion | 500000 | 0.006727 |
| click=1 | click | 23611 | 1.076558 |
| click=1 | conversion | 23611 | 0.005498 |
| conversion=1 | click | 187 | 1.012818 |
| conversion=1 | conversion | 187 | 0.006276 |

