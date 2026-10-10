# MT-RankMixer token gate weights

Stage: validation. Use these numbers for analysis and model selection. The test split stays unread until the design is frozen.

Checkpoint: `/root/autodl-tmp/nf_out/checkpoints/AliCCP_x1/MTR_nf_g6_gate_normfloor_s2028.model`

Rows used: 500000 (cap 500000). task_pooling: `gate`. Sample standard deviation (n - 1) is across rows inside this checkpoint, not across seeds. Gates sum to 1 over tokens inside each task.

| slice | task | n | user_id mean | user_id std | user_profile mean | user_profile std | item_id mean | item_id std | item_attr mean | item_attr std | cross mean | cross std | scenario mean | scenario std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | click | 500000 | 0.035100 | 0.055548 | 0.000479 | 0.000354 | 0.909182 | 0.071173 | 0.022412 | 0.013248 | 0.000928 | 0.000521 | 0.031900 | 0.022488 |
| all | conversion | 500000 | 0.000383 | 0.000420 | 0.000021 | 0.000013 | 0.002603 | 0.001285 | 0.996647 | 0.001704 | 0.000113 | 0.000133 | 0.000234 | 0.000200 |
| click=1 | click | 23611 | 0.051913 | 0.071572 | 0.000572 | 0.000410 | 0.886925 | 0.086132 | 0.024130 | 0.015468 | 0.001059 | 0.000575 | 0.035401 | 0.022745 |
| click=1 | conversion | 23611 | 0.000455 | 0.000466 | 0.000022 | 0.000013 | 0.002910 | 0.001437 | 0.996228 | 0.001877 | 0.000128 | 0.000160 | 0.000258 | 0.000214 |
| conversion=1 | click | 187 | 0.039267 | 0.061839 | 0.000600 | 0.000469 | 0.911508 | 0.071874 | 0.018018 | 0.011034 | 0.001116 | 0.000560 | 0.029490 | 0.017837 |
| conversion=1 | conversion | 187 | 0.000426 | 0.000362 | 0.000024 | 0.000012 | 0.002918 | 0.001122 | 0.996159 | 0.001493 | 0.000159 | 0.000137 | 0.000314 | 0.000245 |

## Mean gate entropy (nats)

| slice | task | n | entropy |
| --- | --- | --- | --- |
| all | click | 500000 | 0.366992 |
| all | conversion | 500000 | 0.024491 |
| click=1 | click | 23611 | 0.429684 |
| click=1 | conversion | 23611 | 0.027189 |
| conversion=1 | click | 187 | 0.359955 |
| conversion=1 | conversion | 187 | 0.027894 |

