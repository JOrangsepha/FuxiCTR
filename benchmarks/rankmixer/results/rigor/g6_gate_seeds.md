# Gate weights across seeds

Seeds: 2025, 2026, 2027, 2028, 2029.
Stage: valid.

Stage: validation. Use these numbers for analysis and model selection. The test split stays unread until the design is frozen.

Each cell is the mean of the per-seed mean gate weights, ± the sample standard deviation across seeds (n - 1). It is not the standard deviation across rows inside one checkpoint.

| slice | task | n_seeds | user_id mean±std | user_profile mean±std | item_id mean±std | item_attr mean±std | cross mean±std | scenario mean±std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | click | 5 | 0.173599 ± 0.173046 | 0.039235 ± 0.074564 | 0.345477 ± 0.384844 | 0.025521 ± 0.038515 | 0.085463 ± 0.179921 | 0.330704 ± 0.361829 |
| all | conversion | 5 | 0.003666 ± 0.003682 | 0.391532 ± 0.533526 | 0.197542 ± 0.435054 | 0.393750 ± 0.537691 | 0.004976 ± 0.004745 | 0.008533 ± 0.005536 |
| click=1 | click | 5 | 0.185337 ± 0.177887 | 0.041637 ± 0.077592 | 0.340407 ± 0.369388 | 0.029318 ± 0.042836 | 0.079782 ± 0.165514 | 0.323519 ± 0.352123 |
| click=1 | conversion | 5 | 0.003288 ± 0.003248 | 0.391596 ± 0.533424 | 0.197603 ± 0.435325 | 0.393786 ± 0.537522 | 0.005086 ± 0.004263 | 0.008642 ± 0.006260 |
| conversion=1 | click | 5 | 0.158232 ± 0.154191 | 0.043545 ± 0.082293 | 0.346270 ± 0.379000 | 0.031261 ± 0.051592 | 0.088795 ± 0.184575 | 0.331897 ± 0.348439 |
| conversion=1 | conversion | 5 | 0.003717 ± 0.004014 | 0.388939 ± 0.529382 | 0.196468 ± 0.432969 | 0.393718 ± 0.537452 | 0.005698 ± 0.005665 | 0.011460 ± 0.010634 |

## Mean gate entropy across seeds (nats)

| slice | task | n_seeds | entropy mean±std |
| --- | --- | --- | --- |
| all | click | 5 | 0.764166 ± 0.343040 |
| all | conversion | 5 | 0.121305 ± 0.050015 |
| click=1 | click | 5 | 0.811185 ± 0.333832 |
| click=1 | conversion | 5 | 0.121490 ± 0.047813 |
| conversion=1 | click | 5 | 0.801799 ± 0.332890 |
| conversion=1 | conversion | 5 | 0.137513 ± 0.069713 |
