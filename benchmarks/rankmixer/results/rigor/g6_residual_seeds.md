# Gate weights across seeds

Seeds: 2025, 2026, 2027, 2028, 2029.
Stage: valid.

Stage: validation. Use these numbers for analysis and model selection. The test split stays unread until the design is frozen.

Each cell is the mean of the per-seed mean gate weights, ± the sample standard deviation across seeds (n - 1). It is not the standard deviation across rows inside one checkpoint.

| slice | task | n_seeds | user_id mean±std | user_profile mean±std | item_id mean±std | item_attr mean±std | cross mean±std | scenario mean±std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | click | 5 | 0.457342 ± 0.413767 | 0.026851 ± 0.040415 | 0.205731 ± 0.414367 | 0.031752 ± 0.034524 | 0.069023 ± 0.073841 | 0.209301 ± 0.391884 |
| all | conversion | 5 | 0.020033 ± 0.015838 | 0.682477 ± 0.296853 | 0.035923 ± 0.049562 | 0.187981 ± 0.254226 | 0.020757 ± 0.019242 | 0.052829 ± 0.067791 |
| click=1 | click | 5 | 0.468555 ± 0.430645 | 0.028927 ± 0.045232 | 0.210893 ± 0.425575 | 0.025830 ± 0.035185 | 0.057696 ± 0.063811 | 0.208099 ± 0.399786 |
| click=1 | conversion | 5 | 0.017873 ± 0.013774 | 0.678395 ± 0.305568 | 0.036876 ± 0.054236 | 0.200425 ± 0.268732 | 0.018123 ± 0.016271 | 0.048307 ± 0.063368 |
| conversion=1 | click | 5 | 0.445682 ± 0.416960 | 0.031596 ± 0.050189 | 0.210433 ± 0.423486 | 0.026058 ± 0.036332 | 0.074199 ± 0.084073 | 0.212032 ± 0.398357 |
| conversion=1 | conversion | 5 | 0.016995 ± 0.012668 | 0.681824 ± 0.293679 | 0.035348 ± 0.049531 | 0.183597 ± 0.257393 | 0.019012 ± 0.018258 | 0.063224 ± 0.086325 |

## Mean gate entropy across seeds (nats)

| slice | task | n_seeds | entropy mean±std |
| --- | --- | --- | --- |
| all | click | 5 | 0.535870 ± 0.410238 |
| all | conversion | 5 | 0.699623 ± 0.467691 |
| click=1 | click | 5 | 0.501917 ± 0.422221 |
| click=1 | conversion | 5 | 0.684888 ± 0.461294 |
| conversion=1 | click | 5 | 0.540740 ± 0.456847 |
| conversion=1 | conversion | 5 | 0.695100 ± 0.470858 |

## Residual mix weight λ across seeds

| task | n_seeds | lambda mean±std |
| --- | --- | --- |
| click | 5 | 0.506983 ± 0.003786 |
| conversion | 5 | 0.499401 ± 0.004778 |
