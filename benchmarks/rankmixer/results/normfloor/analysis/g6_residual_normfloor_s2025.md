# MT-RankMixer token gate weights

Stage: validation. Use these numbers for analysis and model selection. The test split stays unread until the design is frozen.

Checkpoint: `/root/autodl-tmp/nf_out/checkpoints/AliCCP_x1/MTR_nf_g6_residual_normfloor_s2025.model`

Rows used: 500000 (cap 500000). task_pooling: `residual`. Sample standard deviation (n - 1) is across rows inside this checkpoint, not across seeds. Gates sum to 1 over tokens inside each task.

| slice | task | n | user_id mean | user_id std | user_profile mean | user_profile std | item_id mean | item_id std | item_attr mean | item_attr std | cross mean | cross std | scenario mean | scenario std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | click | 500000 | 0.308723 | 0.263296 | 0.061586 | 0.023097 | 0.205619 | 0.142232 | 0.115217 | 0.065762 | 0.087318 | 0.060910 | 0.221538 | 0.096387 |
| all | conversion | 500000 | 0.020012 | 0.018699 | 0.189634 | 0.056836 | 0.120292 | 0.099474 | 0.156991 | 0.114333 | 0.018647 | 0.019792 | 0.494424 | 0.125946 |
| click=1 | click | 23611 | 0.305442 | 0.271329 | 0.062496 | 0.024026 | 0.209709 | 0.140559 | 0.112862 | 0.045943 | 0.077000 | 0.050719 | 0.232491 | 0.100879 |
| click=1 | conversion | 23611 | 0.018366 | 0.018003 | 0.185768 | 0.054731 | 0.101840 | 0.080425 | 0.182934 | 0.120142 | 0.016840 | 0.019120 | 0.494251 | 0.128840 |
| conversion=1 | click | 187 | 0.185309 | 0.216539 | 0.074252 | 0.020288 | 0.253651 | 0.127827 | 0.118298 | 0.040258 | 0.092047 | 0.045917 | 0.276443 | 0.088072 |
| conversion=1 | conversion | 187 | 0.012943 | 0.015884 | 0.183957 | 0.059602 | 0.109510 | 0.083897 | 0.113779 | 0.082323 | 0.012810 | 0.012959 | 0.567000 | 0.113360 |

## Mean gate entropy (nats)

| slice | task | n | entropy |
| --- | --- | --- | --- |
| all | click | 500000 | 1.423649 |
| all | conversion | 500000 | 1.245935 |
| click=1 | click | 23611 | 1.414703 |
| click=1 | conversion | 23611 | 1.240084 |
| conversion=1 | click | 187 | 1.500035 |
| conversion=1 | conversion | 187 | 1.146504 |

## Residual mix weight λ

h = λ * mean(tokens) + (1 - λ) * gated mix. λ is a learned sigmoid scalar per task.

| task | lambda |
| --- | --- |
| click | 0.534712 |
| conversion | 0.512440 |
