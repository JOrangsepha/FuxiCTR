# Gate-collapse ablation (validation only)

OUT=`/root/autodl-tmp/gc_out`
Protocol: `--skip_test`, patience=3, epochs≤10, AliCCP_x1 **validation only**.
Success (mitigated): across seeds, conversion-gate max weight mean clearly below ~0.9 **or** entropy clearly above ~0.5, without destroying avg AUC vs baseline.

## Summary (mean ± sample std over seeds)

| variant | n | click AUC | conv AUC | avg AUC | clicked CVR | conv gate H | conv max w | collapsed |
|---------|---|-----------|----------|---------|-------------|-------------|------------|-----------|
| gate EQ (baseline, reused rigor) | 5 | 0.6189 ± 0.0014 | 0.6414 ± 0.0059 | 0.6301 ± 0.0035 | 0.6230 ± 0.0073 | 0.1213 ± 0.0500 | 0.9786 ± 0.0112 | 5/5 |
| gate + entropy β=0.001 | 5 | 0.6172 ± 0.0031 | 0.6364 ± 0.0029 | 0.6268 ± 0.0016 | 0.6151 ± 0.0050 | 1.7868 ± 0.0016 | 0.1811 ± 0.0076 | 0/5 |
| gate + entropy β=0.003 | 5 | 0.6200 ± 0.0008 | 0.6380 ± 0.0024 | 0.6290 ± 0.0013 | 0.6133 ± 0.0105 | 1.7891 ± 0.0011 | 0.1749 ± 0.0065 | 0/5 |
| gate + entropy β=0.01 | 5 | 0.6192 ± 0.0012 | 0.6334 ± 0.0033 | 0.6263 ± 0.0016 | 0.6051 ± 0.0052 | 1.7906 ± 0.0006 | 0.1738 ± 0.0027 | 0/5 |
| gate + temperature T=2 | 5 | 0.6196 ± 0.0007 | 0.6388 ± 0.0034 | 0.6292 ± 0.0016 | 0.6192 ± 0.0137 | 1.1487 ± 0.3473 | 0.5781 ± 0.1891 | 0/5 |
| gate + temperature T=4 | 5 | 0.6196 ± 0.0015 | 0.6401 ± 0.0036 | 0.6299 ± 0.0018 | 0.6147 ± 0.0091 | 1.6334 ± 0.1626 | 0.3152 ± 0.1409 | 0/5 |
| gate + β=0.01 × T=2 | 5 | 0.6197 ± 0.0007 | 0.6373 ± 0.0044 | 0.6285 ± 0.0025 | 0.6057 ± 0.0053 | 1.7912 ± 0.0004 | 0.1688 ± 0.0012 | 0/5 |
| residual + entropy β=0.01 | 5 | 0.6198 ± 0.0013 | 0.6365 ± 0.0016 | 0.6282 ± 0.0013 | 0.6117 ± 0.0102 | 1.7912 ± 0.0002 | 0.1714 ± 0.0026 | 0/5 |

## Mitigation verdict

- gate + entropy β=0.001: **mitigated** (max_w=0.181 < 0.9; H=1.787 > 0.5; avg AUC Δ=-0.0033)
- gate + entropy β=0.003: **mitigated** (max_w=0.175 < 0.9; H=1.789 > 0.5; avg AUC Δ=-0.0012)
- gate + entropy β=0.01: **mitigated** (max_w=0.174 < 0.9; H=1.791 > 0.5; avg AUC Δ=-0.0038)
- gate + temperature T=2: **mitigated** (max_w=0.578 < 0.9; H=1.149 > 0.5; avg AUC Δ=-0.0009)
- gate + temperature T=4: **mitigated** (max_w=0.315 < 0.9; H=1.633 > 0.5; avg AUC Δ=-0.0003)
- gate + β=0.01 × T=2: **mitigated** (max_w=0.169 < 0.9; H=1.791 > 0.5; avg AUC Δ=-0.0016)
- residual + entropy β=0.01: **mitigated** (max_w=0.171 < 0.9; H=1.791 > 0.5; avg AUC Δ=-0.0020)

## Per-seed detail

| expid | exit | secs | click | conv | avg | clicked_CVR | conv H | max w | argmax | collapsed |
|-------|------|------|-------|------|-----|-------------|--------|-------|--------|-----------|
| MTR_gc_g6_gate_s2025 | 0 | 0 | 0.61839 | 0.63734 | 0.62787 | 0.61733 | 0.1402 | 0.9758 | item_id | True |
| MTR_gc_g6_gate_s2026 | 0 | 0 | 0.62051 | 0.64669 | 0.63360 | 0.63372 | 0.1130 | 0.9818 | item_attr | True |
| MTR_gc_g6_gate_s2027 | 0 | 0 | 0.61901 | 0.64863 | 0.63382 | 0.62595 | 0.1947 | 0.9610 | user_profile | True |
| MTR_gc_g6_gate_s2028 | 0 | 0 | 0.61966 | 0.63888 | 0.62927 | 0.62254 | 0.0975 | 0.9837 | item_attr | True |
| MTR_gc_g6_gate_s2029 | 0 | 0 | 0.61675 | 0.63541 | 0.62608 | 0.61546 | 0.0611 | 0.9907 | user_profile | True |
| MTR_gc_g6_gate_ent0001_s2025 | 0 | 879 | 0.61652 | 0.63563 | 0.62607 | 0.62114 | 1.7851 | 0.1869 | user_id | False |
| MTR_gc_g6_gate_ent0001_s2026 | 0 | 1078 | 0.61272 | 0.63924 | 0.62598 | 0.60821 | 1.7894 | 0.1788 | item_attr | False |
| MTR_gc_g6_gate_ent0001_s2027 | 0 | 858 | 0.62089 | 0.63770 | 0.62930 | 0.61769 | 1.7862 | 0.1883 | user_id | False |
| MTR_gc_g6_gate_ent0001_s2028 | 0 | 853 | 0.61687 | 0.63768 | 0.62727 | 0.61226 | 1.7860 | 0.1822 | user_id | False |
| MTR_gc_g6_gate_ent0001_s2029 | 0 | 850 | 0.61894 | 0.63173 | 0.62533 | 0.61644 | 1.7871 | 0.1693 | user_profile | False |
| MTR_gc_g6_gate_ent0003_s2025 | 0 | 853 | 0.62035 | 0.63688 | 0.62862 | 0.61091 | 1.7887 | 0.1726 | item_attr | False |
| MTR_gc_g6_gate_ent0003_s2026 | 0 | 869 | 0.62085 | 0.64011 | 0.63048 | 0.62464 | 1.7896 | 0.1761 | user_id | False |
| MTR_gc_g6_gate_ent0003_s2027 | 0 | 872 | 0.61889 | 0.63976 | 0.62932 | 0.62324 | 1.7896 | 0.1706 | item_id | False |
| MTR_gc_g6_gate_ent0003_s2028 | 0 | 852 | 0.61937 | 0.63437 | 0.62687 | 0.60029 | 1.7876 | 0.1857 | user_id | False |
| MTR_gc_g6_gate_ent0003_s2029 | 0 | 860 | 0.62041 | 0.63869 | 0.62955 | 0.60717 | 1.7903 | 0.1697 | scenario | False |
| MTR_gc_g6_gate_ent001_s2025 | 0 | 854 | 0.61845 | 0.62952 | 0.62398 | 0.60584 | 1.7899 | 0.1754 | item_id | False |
| MTR_gc_g6_gate_ent001_s2026 | 0 | 852 | 0.62087 | 0.63035 | 0.62561 | 0.60016 | 1.7912 | 0.1718 | item_id | False |
| MTR_gc_g6_gate_ent001_s2027 | 0 | 857 | 0.61827 | 0.63460 | 0.62643 | 0.60108 | 1.7900 | 0.1778 | user_id | False |
| MTR_gc_g6_gate_ent001_s2028 | 0 | 840 | 0.62006 | 0.63548 | 0.62777 | 0.60498 | 1.7909 | 0.1719 | cross | False |
| MTR_gc_g6_gate_ent001_s2029 | 0 | 864 | 0.61827 | 0.63697 | 0.62762 | 0.61340 | 1.7910 | 0.1719 | user_id | False |
| MTR_gc_g6_gate_T2_s2025 | 0 | 853 | 0.61965 | 0.63905 | 0.62935 | 0.62903 | 1.3374 | 0.4324 | user_profile | False |
| MTR_gc_g6_gate_T2_s2026 | 0 | 826 | 0.62035 | 0.63991 | 0.63013 | 0.60695 | 1.5382 | 0.3841 | item_attr | False |
| MTR_gc_g6_gate_T2_s2027 | 0 | 834 | 0.62001 | 0.63802 | 0.62901 | 0.61212 | 1.2911 | 0.5247 | user_profile | False |
| MTR_gc_g6_gate_T2_s2028 | 0 | 851 | 0.61861 | 0.64304 | 0.63082 | 0.63825 | 0.8756 | 0.7295 | item_attr | False |
| MTR_gc_g6_gate_T2_s2029 | 0 | 836 | 0.61938 | 0.63378 | 0.62658 | 0.60947 | 0.7011 | 0.8199 | user_profile | False |
| MTR_gc_g6_gate_T4_s2025 | 0 | 840 | 0.62097 | 0.63828 | 0.62963 | 0.60333 | 1.6923 | 0.2367 | scenario | False |
| MTR_gc_g6_gate_T4_s2026 | 0 | 840 | 0.61736 | 0.64199 | 0.62968 | 0.61528 | 1.7619 | 0.1998 | user_id | False |
| MTR_gc_g6_gate_T4_s2027 | 0 | 851 | 0.62045 | 0.64523 | 0.63284 | 0.62643 | 1.7455 | 0.2303 | item_attr | False |
| MTR_gc_g6_gate_T4_s2028 | 0 | 850 | 0.61896 | 0.63910 | 0.62903 | 0.61989 | 1.6026 | 0.3717 | user_profile | False |
| MTR_gc_g6_gate_T4_s2029 | 0 | 846 | 0.62027 | 0.63602 | 0.62815 | 0.60845 | 1.3646 | 0.5376 | user_profile | False |
| MTR_gc_g6_gate_ent001_T2_s2025 | 0 | 866 | 0.62093 | 0.64267 | 0.63180 | 0.61138 | 1.7914 | 0.1685 | item_id | False |
| MTR_gc_g6_gate_ent001_T2_s2026 | 0 | 836 | 0.61934 | 0.63733 | 0.62834 | 0.61141 | 1.7915 | 0.1683 | user_id | False |
| MTR_gc_g6_gate_ent001_T2_s2027 | 0 | 863 | 0.61919 | 0.63844 | 0.62882 | 0.60335 | 1.7913 | 0.1706 | user_id | False |
| MTR_gc_g6_gate_ent001_T2_s2028 | 0 | 853 | 0.61991 | 0.63775 | 0.62883 | 0.60109 | 1.7914 | 0.1674 | user_profile | False |
| MTR_gc_g6_gate_ent001_T2_s2029 | 0 | 855 | 0.61932 | 0.63031 | 0.62482 | 0.60137 | 1.7906 | 0.1694 | cross | False |
| MTR_gc_g6_residual_ent001_s2025 | 0 | 855 | 0.61849 | 0.63521 | 0.62685 | 0.61746 | 1.7914 | 0.1702 | user_id | False |
| MTR_gc_g6_residual_ent001_s2026 | 0 | 851 | 0.61857 | 0.63595 | 0.62726 | 0.60161 | 1.7912 | 0.1737 | user_id | False |
| MTR_gc_g6_residual_ent001_s2027 | 0 | 870 | 0.62060 | 0.63935 | 0.62997 | 0.61665 | 1.7913 | 0.1706 | user_profile | False |
| MTR_gc_g6_residual_ent001_s2028 | 0 | 864 | 0.62135 | 0.63659 | 0.62897 | 0.62253 | 1.7913 | 0.1680 | cross | False |
| MTR_gc_g6_residual_ent001_s2029 | 0 | 781 | 0.61996 | 0.63564 | 0.62780 | 0.60002 | 1.7908 | 0.1744 | item_attr | False |

## Notes

- Baseline checkpoints/logs/gate JSON are reused from `rg_out` when `REUSE_BASELINE=1`.
- Impression-level conversion AUC is CTCVR; clicked CVR is conversion AUC on `click==1` rows.
- Collapsed = conversion max weight ≥ 0.9 (or max≥0.85 with entropy ≤ 0.5).

