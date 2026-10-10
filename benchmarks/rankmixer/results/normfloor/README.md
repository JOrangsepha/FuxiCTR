# NORM_FLOOR real-data validation suite

Protocol: `--skip_test`, patience=3, epochs≤10, AliCCP_x1 **validation only** (test unread).
20/20 training runs exit 0. Code path: `loss_weight: NORM_FLOOR` = per-batch NORM with denominator `clamp_min(1e-2)`.

Authoritative write-up: [`results.md`](results.md). Per-seed table: [`summary.csv`](summary.csv). Driver log: [`runs.tsv`](runs.tsv). Gradient audit: [`analysis/grad_audit_normfloor.md`](analysis/grad_audit_normfloor.md). Gate dumps: `analysis/g6_*_normfloor_s*.md`.

This suite did **not** touch the test split. Do not claim a final-test win over fair PLE from these numbers.
