# MT-RankMixer token gate weights

Stage: validation. Use these numbers for analysis and model selection. The test split stays unread until the design is frozen.

Checkpoint: `/root/autodl-tmp/nf_out/checkpoints/AliCCP_x1/MTR_nf_g6_residual_normfloor_s2026.model`

Rows used: 500000 (cap 500000). task_pooling: `residual`. Sample standard deviation (n - 1) is across rows inside this checkpoint, not across seeds. Gates sum to 1 over tokens inside each task.

| slice | task | n | user_id mean | user_id std | user_profile mean | user_profile std | item_id mean | item_id std | item_attr mean | item_attr std | cross mean | cross std | scenario mean | scenario std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | click | 500000 | 0.135841 | 0.119909 | 0.021261 | 0.010202 | 0.123122 | 0.084519 | 0.029582 | 0.020432 | 0.011767 | 0.006476 | 0.678427 | 0.191102 |
| all | conversion | 500000 | 0.099823 | 0.076822 | 0.152397 | 0.055170 | 0.295912 | 0.096488 | 0.260296 | 0.116610 | 0.048167 | 0.020148 | 0.143406 | 0.058635 |
| click=1 | click | 23611 | 0.165091 | 0.140536 | 0.024038 | 0.010962 | 0.146227 | 0.093478 | 0.033654 | 0.021357 | 0.013370 | 0.006681 | 0.617620 | 0.203037 |
| click=1 | conversion | 23611 | 0.085074 | 0.070759 | 0.139249 | 0.047835 | 0.292390 | 0.088881 | 0.285906 | 0.112864 | 0.044506 | 0.019227 | 0.152875 | 0.058801 |
| conversion=1 | click | 187 | 0.098247 | 0.083934 | 0.021402 | 0.008642 | 0.117952 | 0.082115 | 0.023146 | 0.012811 | 0.010535 | 0.005440 | 0.728717 | 0.147447 |
| conversion=1 | conversion | 187 | 0.073187 | 0.076361 | 0.151108 | 0.042884 | 0.334291 | 0.086478 | 0.223414 | 0.103815 | 0.047209 | 0.017613 | 0.170790 | 0.059155 |

## Mean gate entropy (nats)

| slice | task | n | entropy |
| --- | --- | --- | --- |
| all | click | 500000 | 0.921377 |
| all | conversion | 500000 | 1.550685 |
| click=1 | click | 23611 | 1.013180 |
| click=1 | conversion | 23611 | 1.536576 |
| conversion=1 | click | 187 | 0.854847 |
| conversion=1 | conversion | 187 | 1.531159 |

## Residual mix weight λ

h = λ * mean(tokens) + (1 - λ) * gated mix. λ is a learned sigmoid scalar per task.

| task | lambda |
| --- | --- |
| click | 0.498908 |
| conversion | 0.525592 |
