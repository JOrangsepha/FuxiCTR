# MT-RankMixer token gate weights

Stage: validation. Use these numbers for analysis and model selection. The test split stays unread until the design is frozen.

Checkpoint: `/root/autodl-tmp/nf_out/checkpoints/AliCCP_x1/MTR_nf_g6_gate_normfloor_s2029.model`

Rows used: 500000 (cap 500000). task_pooling: `gate`. Sample standard deviation (n - 1) is across rows inside this checkpoint, not across seeds. Gates sum to 1 over tokens inside each task.

| slice | task | n | user_id mean | user_id std | user_profile mean | user_profile std | item_id mean | item_id std | item_attr mean | item_attr std | cross mean | cross std | scenario mean | scenario std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | click | 500000 | 0.132400 | 0.083630 | 0.004594 | 0.002458 | 0.851842 | 0.089054 | 0.001166 | 0.001317 | 0.000182 | 0.000312 | 0.009815 | 0.006401 |
| all | conversion | 500000 | 0.000181 | 0.000129 | 0.997399 | 0.001925 | 0.002218 | 0.001771 | 0.000017 | 0.000045 | 0.000044 | 0.000082 | 0.000141 | 0.000100 |
| click=1 | click | 23611 | 0.149178 | 0.087690 | 0.005344 | 0.003019 | 0.832653 | 0.093181 | 0.001424 | 0.001727 | 0.000246 | 0.000456 | 0.011155 | 0.006904 |
| click=1 | conversion | 23611 | 0.000195 | 0.000281 | 0.996840 | 0.002370 | 0.002731 | 0.002183 | 0.000018 | 0.000030 | 0.000056 | 0.000109 | 0.000160 | 0.000112 |
| conversion=1 | click | 187 | 0.139924 | 0.086838 | 0.005389 | 0.002711 | 0.840040 | 0.093663 | 0.001584 | 0.001482 | 0.000244 | 0.000367 | 0.012819 | 0.007529 |
| conversion=1 | conversion | 187 | 0.000207 | 0.000091 | 0.996800 | 0.001896 | 0.002764 | 0.001792 | 0.000018 | 0.000011 | 0.000045 | 0.000050 | 0.000165 | 0.000081 |

## Mean gate entropy (nats)

| slice | task | n | entropy |
| --- | --- | --- | --- |
| all | click | 500000 | 0.449801 |
| all | conversion | 500000 | 0.018932 |
| click=1 | click | 23611 | 0.490873 |
| click=1 | conversion | 23611 | 0.022282 |
| conversion=1 | click | 187 | 0.482010 |
| conversion=1 | conversion | 187 | 0.022720 |

