# MT-RankMixer token gate weights

Stage: validation. Use these numbers for analysis and model selection. The test split stays unread until the design is frozen.

Checkpoint: `/root/autodl-tmp/nf_out/checkpoints/AliCCP_x1/MTR_nf_g6_gate_normfloor_s2026.model`

Rows used: 500000 (cap 500000). task_pooling: `gate`. Sample standard deviation (n - 1) is across rows inside this checkpoint, not across seeds. Gates sum to 1 over tokens inside each task.

| slice | task | n | user_id mean | user_id std | user_profile mean | user_profile std | item_id mean | item_id std | item_attr mean | item_attr std | cross mean | cross std | scenario mean | scenario std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | click | 500000 | 0.025814 | 0.025886 | 0.000833 | 0.000392 | 0.004466 | 0.006005 | 0.056250 | 0.042182 | 0.005650 | 0.006787 | 0.906985 | 0.060064 |
| all | conversion | 500000 | 0.000235 | 0.000134 | 0.000186 | 0.000164 | 0.000145 | 0.000096 | 0.997740 | 0.001553 | 0.000973 | 0.001091 | 0.000721 | 0.000710 |
| click=1 | click | 23611 | 0.033167 | 0.032298 | 0.000924 | 0.000465 | 0.006088 | 0.007754 | 0.062538 | 0.041155 | 0.007210 | 0.009260 | 0.890073 | 0.064039 |
| click=1 | conversion | 23611 | 0.000248 | 0.000142 | 0.000200 | 0.000171 | 0.000155 | 0.000097 | 0.997664 | 0.001673 | 0.001125 | 0.001317 | 0.000608 | 0.000417 |
| conversion=1 | click | 187 | 0.031104 | 0.027035 | 0.000889 | 0.000487 | 0.005618 | 0.006211 | 0.078572 | 0.039646 | 0.007703 | 0.008264 | 0.876114 | 0.058510 |
| conversion=1 | conversion | 187 | 0.000262 | 0.000144 | 0.000271 | 0.000180 | 0.000154 | 0.000072 | 0.997255 | 0.001794 | 0.001352 | 0.001405 | 0.000706 | 0.000428 |

## Mean gate entropy (nats)

| slice | task | n | entropy |
| --- | --- | --- | --- |
| all | click | 500000 | 0.369959 |
| all | conversion | 500000 | 0.018372 |
| click=1 | click | 23611 | 0.427518 |
| click=1 | conversion | 23611 | 0.018915 |
| conversion=1 | click | 187 | 0.466346 |
| conversion=1 | conversion | 187 | 0.021761 |

