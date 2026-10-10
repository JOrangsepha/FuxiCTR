# MT-RankMixer token gate weights

Stage: validation. Use these numbers for analysis and model selection. The test split stays unread until the design is frozen.

Checkpoint: `/root/autodl-tmp/nf_out/checkpoints/AliCCP_x1/MTR_nf_g6_residual_normfloor_s2028.model`

Rows used: 500000 (cap 500000). task_pooling: `residual`. Sample standard deviation (n - 1) is across rows inside this checkpoint, not across seeds. Gates sum to 1 over tokens inside each task.

| slice | task | n | user_id mean | user_id std | user_profile mean | user_profile std | item_id mean | item_id std | item_attr mean | item_attr std | cross mean | cross std | scenario mean | scenario std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | click | 500000 | 0.957637 | 0.025431 | 0.006945 | 0.004213 | 0.012764 | 0.007952 | 0.007062 | 0.005327 | 0.007106 | 0.007648 | 0.008486 | 0.005734 |
| all | conversion | 500000 | 0.008677 | 0.004978 | 0.084306 | 0.072175 | 0.305990 | 0.142083 | 0.093029 | 0.107490 | 0.027476 | 0.027520 | 0.480522 | 0.180092 |
| click=1 | click | 23611 | 0.956199 | 0.027573 | 0.007392 | 0.004605 | 0.012897 | 0.008593 | 0.007210 | 0.005586 | 0.008007 | 0.008596 | 0.008295 | 0.005431 |
| click=1 | conversion | 23611 | 0.008653 | 0.004906 | 0.074182 | 0.053220 | 0.266786 | 0.124199 | 0.109818 | 0.118622 | 0.025506 | 0.027897 | 0.515055 | 0.168397 |
| conversion=1 | click | 187 | 0.955778 | 0.041581 | 0.007434 | 0.006604 | 0.011669 | 0.010536 | 0.008577 | 0.008751 | 0.008269 | 0.013071 | 0.008273 | 0.006678 |
| conversion=1 | conversion | 187 | 0.007338 | 0.003078 | 0.076493 | 0.054786 | 0.268372 | 0.136718 | 0.049041 | 0.075437 | 0.018083 | 0.019464 | 0.580673 | 0.166915 |

## Mean gate entropy (nats)

| slice | task | n | entropy |
| --- | --- | --- | --- |
| all | click | 500000 | 0.233188 |
| all | conversion | 500000 | 1.129355 |
| click=1 | click | 23611 | 0.239321 |
| click=1 | conversion | 23611 | 1.122701 |
| conversion=1 | click | 187 | 0.236312 |
| conversion=1 | conversion | 187 | 1.002445 |

## Residual mix weight λ

h = λ * mean(tokens) + (1 - λ) * gated mix. λ is a learned sigmoid scalar per task.

| task | lambda |
| --- | --- |
| click | 0.503483 |
| conversion | 0.517928 |
