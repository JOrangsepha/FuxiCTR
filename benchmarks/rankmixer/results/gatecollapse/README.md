# 转化门塌缩消融（只看验证集）

协议：`--skip_test`，patience=3，epochs≤10，AliCCP_x1，**验证集**，test 未读。种子 2025–2029。基线 g6_gate EQ 在 `REUSE_BASELINE=1` 时复用 rigor 的 `rg_out`，不是这一轮重新训练的。

数字以 [`results.md`](results.md) 为准。驱动日志是 [`runs.tsv`](runs.tsv)。这一轮归档里没有逐种子的门控 dump，也没有另写 summary csv。

这套实验没有读 test。不能据此改写预注册 test 里「没有 p<0.05」的结论，也不能说已经在最终 test 上超过调公平之后的 PLE。
