# MT-RankMixer token gate weights

Checkpoint: `/root/autodl-tmp/sw_out/checkpoints/AliCCP_x1/MTR_sw_g6_gate_s2025.model`

Rows used: 500000 (cap 500000). task_pooling: `gate`. Sample standard deviation (n - 1). Gates sum to 1 over tokens inside each task.

| slice | task | n | token0 mean | token0 std | token1 mean | token1 std | token2 mean | token2 std | token3 mean | token3 std | token4 mean | token4 std | token5 mean | token5 std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | click | 500000 | 0.379083 | 0.150101 | 0.007109 | 0.005113 | 0.137500 | 0.050855 | 0.003818 | 0.004058 | 0.008463 | 0.007117 | 0.464027 | 0.146254 |
| all | conversion | 500000 | 0.017516 | 0.013752 | 0.001916 | 0.001864 | 0.966156 | 0.020994 | 0.000880 | 0.005092 | 0.005908 | 0.009282 | 0.007624 | 0.006768 |
| click=1 | click | 23552 | 0.382413 | 0.141139 | 0.008506 | 0.005819 | 0.152946 | 0.053767 | 0.004784 | 0.004596 | 0.008564 | 0.006818 | 0.442787 | 0.134616 |
| click=1 | conversion | 23552 | 0.015905 | 0.012989 | 0.002217 | 0.002094 | 0.968348 | 0.021451 | 0.001093 | 0.007996 | 0.006227 | 0.009108 | 0.006210 | 0.005162 |
| conversion=1 | click | 159 | 0.347304 | 0.144479 | 0.007432 | 0.005434 | 0.142953 | 0.049157 | 0.003696 | 0.005112 | 0.009415 | 0.007529 | 0.489201 | 0.146818 |
| conversion=1 | conversion | 159 | 0.021077 | 0.018687 | 0.002422 | 0.002647 | 0.956131 | 0.052611 | 0.003107 | 0.027731 | 0.008629 | 0.012911 | 0.008634 | 0.006237 |

## Mean gate entropy (nats)

| slice | task | n | entropy |
| --- | --- | --- | --- |
| all | click | 500000 | 1.022955 |
| all | conversion | 500000 | 0.177228 |
| click=1 | click | 23552 | 1.058635 |
| click=1 | conversion | 23552 | 0.169129 |
| conversion=1 | click | 159 | 1.027579 |
| conversion=1 | conversion | 159 | 0.212269 |


![gate weights](../../../docs/img/rankmixer/gate_g6_gate_s2025.png)
