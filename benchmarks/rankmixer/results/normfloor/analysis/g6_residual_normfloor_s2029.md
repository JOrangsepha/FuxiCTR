# MT-RankMixer token gate weights

Stage: validation. Use these numbers for analysis and model selection. The test split stays unread until the design is frozen.

Checkpoint: `/root/autodl-tmp/nf_out/checkpoints/AliCCP_x1/MTR_nf_g6_residual_normfloor_s2029.model`

Rows used: 500000 (cap 500000). task_pooling: `residual`. Sample standard deviation (n - 1) is across rows inside this checkpoint, not across seeds. Gates sum to 1 over tokens inside each task.

| slice | task | n | user_id mean | user_id std | user_profile mean | user_profile std | item_id mean | item_id std | item_attr mean | item_attr std | cross mean | cross std | scenario mean | scenario std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | click | 500000 | 0.004971 | 0.009518 | 0.011747 | 0.009480 | 0.930329 | 0.053659 | 0.038703 | 0.034530 | 0.010621 | 0.007585 | 0.003629 | 0.012753 |
| all | conversion | 500000 | 0.205756 | 0.172924 | 0.733758 | 0.165683 | 0.005968 | 0.002268 | 0.010571 | 0.013434 | 0.021817 | 0.018667 | 0.022131 | 0.070587 |
| click=1 | click | 23611 | 0.005669 | 0.014702 | 0.012702 | 0.011324 | 0.926271 | 0.061049 | 0.040363 | 0.036667 | 0.011072 | 0.008630 | 0.003923 | 0.015880 |
| click=1 | conversion | 23611 | 0.228477 | 0.182935 | 0.709213 | 0.175833 | 0.005824 | 0.002337 | 0.012122 | 0.015176 | 0.020578 | 0.018742 | 0.023785 | 0.081781 |
| conversion=1 | click | 187 | 0.004013 | 0.008349 | 0.012360 | 0.008123 | 0.920280 | 0.052086 | 0.049277 | 0.037813 | 0.011613 | 0.006451 | 0.002457 | 0.002157 |
| conversion=1 | conversion | 187 | 0.291618 | 0.170635 | 0.671984 | 0.158739 | 0.005746 | 0.002245 | 0.005491 | 0.006534 | 0.014530 | 0.013975 | 0.010632 | 0.004496 |

## Mean gate entropy (nats)

| slice | task | n | entropy |
| --- | --- | --- | --- |
| all | click | 500000 | 0.314487 |
| all | conversion | 500000 | 0.667129 |
| click=1 | click | 23611 | 0.325302 |
| click=1 | conversion | 23611 | 0.686934 |
| conversion=1 | click | 187 | 0.345226 |
| conversion=1 | conversion | 187 | 0.707568 |

## Residual mix weight λ

h = λ * mean(tokens) + (1 - λ) * gated mix. λ is a learned sigmoid scalar per task.

| task | lambda |
| --- | --- |
| click | 0.501634 |
| conversion | 0.510136 |
