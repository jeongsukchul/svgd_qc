# antmaze-giant experiment archive

All runs: 1M offline steps, `--discount=0.995`, 50 eval episodes, eval every 100k.

Machine: single RTX 3090, env `/home/sukchul/miniconda3/envs/fql`, driver `main.py`.


## Headline table (completed runs, mean over seeds of last-5 evals)

| agent | t1 | t2 | t3 |
|---|---|---|---|
| `rebrac.py` | **0.244** (2sd) | **0.832** (2sd) | **0.426** (2sd) |
| `anq_stdfp + clip fix` | **0.088** (2sd) | **0.818** (2sd) | **0.312** (2sd) |
| `anq_stdfp as-is` | **0.128** (2sd) | **0.700** (1sd) | **0.192** (2sd) |
| `mani_stdfp metric (scale-free)` | — | — | **0.004** (1sd) |

## All runs

| run group | task | agent | seed | n | last3 | last5 | curve |
|---|---|---|---|---|---|---|---|
| `N_pretanh_t1` | t1 | anq_stdfp + clip fix | 0 | 11 | 0.107 | 0.092 | 0.00 0.26 0.08 0.12 0.12 0.00 0.02 0.12 0.14 0.10 0.08 |
| `N1_pretanh_t1_s1` | t1 | anq_stdfp + clip fix | 1 | 11 | 0.080 | 0.084 | 0.04 0.02 0.00 0.00 0.00 0.00 0.02 0.16 0.08 0.06 0.10 |
| `M_base_t1` | t1 | anq_stdfp as-is | 0 | 11 | 0.140 | 0.084 | 0.00 0.04 0.00 0.00 0.00 0.00 0.00 0.00 0.16 0.16 0.10 |
| `M1_base_t1_s1` | t1 | anq_stdfp as-is | 1 | 11 | 0.267 | 0.172 | 0.00 0.14 0.04 0.00 0.00 0.00 0.00 0.06 0.36 0.18 0.26 |
| `T_bs00_t1` | t1 | anq_stdfp base_scale=0.0 *(running)* | 0 | 8 | 0.293 | 0.196 | 0.02 0.00 0.00 0.08 0.02 0.26 0.20 0.42 |
| `T1_bs00_t1_s1` | t1 | anq_stdfp base_scale=0.0 *(running)* | 1 | 1 | 0.060 | 0.060 | 0.06 |
| `V_bs05_t1` | t1 | anq_stdfp base_scale=0.5 *(running)* | 0 | 4 | 0.007 | 0.005 | 0.00 0.02 0.00 0.00 |
| `R_rebrac_t1` | t1 | rebrac.py | 0 | 11 | 0.220 | 0.232 | 0.00 0.08 0.00 0.00 0.18 0.26 0.34 0.16 0.20 0.14 0.32 |
| `R1_rebrac_t1_s1` | t1 | rebrac.py | 1 | 11 | 0.327 | 0.256 | 0.02 0.00 0.00 0.00 0.04 0.08 0.24 0.06 0.28 0.26 0.44 |
| `P_pretanh_t2` | t2 | anq_stdfp + clip fix | 0 | 11 | 0.847 | 0.840 | 0.00 0.60 0.00 0.50 0.80 0.88 0.88 0.78 0.92 0.86 0.76 |
| `P1_pretanh_t2_s1` | t2 | anq_stdfp + clip fix | 1 | 11 | 0.773 | 0.796 | 0.02 0.02 0.46 0.54 0.62 0.62 0.84 0.82 0.80 0.74 0.78 |
| `O_base_t2` | t2 | anq_stdfp as-is | 0 | 11 | 0.707 | 0.700 | 0.00 0.28 0.26 0.24 0.72 0.60 0.68 0.70 0.70 0.68 0.74 |
| `O1_base_t2_s1` | t2 | anq_stdfp as-is *(running)* | 1 | 1 | 0.000 | 0.000 | 0.00 |
| `Q_rebrac_t2` | t2 | rebrac.py | 0 | 11 | 0.833 | 0.844 | 0.00 0.00 0.78 0.26 0.82 0.72 0.88 0.84 0.84 0.82 0.84 |
| `Q1_rebrac_t2_s1` | t2 | rebrac.py | 1 | 11 | 0.833 | 0.820 | 0.00 0.52 0.24 0.24 0.42 0.52 0.82 0.78 0.70 0.92 0.88 |
| `A_lam5_t3` | t3 | [retired] anq_stdfp lam=5.0 *(running)* | 0 | 5 | 0.000 | 0.000 | 0.00 0.00 0.00 0.00 0.00 |
| `C_stdfp3_t3` | t3 | [retired] anq_stdfp3 (=ReBRAC) *(running)* | 0 | 4 | 0.000 | 0.000 | 0.00 0.00 0.00 0.00 |
| `D_lam5_pretanh_t3` | t3 | [retired] lam=5.0 +pretanh *(running)* | 0 | 5 | 0.000 | 0.000 | 0.00 0.00 0.00 0.00 0.00 |
| `G_mani_data_t30` | t3 | [retired] mani strong-lam *(running)* | 0 | 1 | 0.000 | 0.000 | 0.00 |
| `E_rebrac_a5_t3` | t3 | [retired] rebrac alpha=5.0 *(running)* | 0 | 6 | 0.000 | 0.000 | 0.00 0.00 0.00 0.00 0.00 0.00 |
| `J_lam001_pretanh_t3` | t3 | anq_stdfp + clip fix | 0 | 11 | 0.387 | 0.380 | 0.00 0.00 0.02 0.02 0.28 0.44 0.36 0.38 0.34 0.44 0.38 |
| `J1_pretanh_t3_s1` | t3 | anq_stdfp + clip fix | 1 | 11 | 0.253 | 0.244 | 0.00 0.02 0.04 0.00 0.40 0.32 0.32 0.14 0.16 0.36 0.24 |
| `B_lam001_t3` | t3 | anq_stdfp as-is | 0 | 11 | 0.187 | 0.156 | 0.00 0.02 0.00 0.00 0.04 0.22 0.08 0.14 0.20 0.14 0.22 |
| `B1_lam001_t3_s1` | t3 | anq_stdfp as-is | 1 | 11 | 0.253 | 0.228 | 0.00 0.00 0.00 0.12 0.12 0.10 0.28 0.10 0.28 0.20 0.28 |
| `S_bs00_t3` | t3 | anq_stdfp base_scale=0.0 *(running)* | 0 | 8 | 0.353 | 0.296 | 0.00 0.00 0.00 0.18 0.24 0.38 0.28 0.40 |
| `S1_bs00_t3_s1` | t3 | anq_stdfp base_scale=0.0 *(running)* | 1 | 3 | 0.013 | 0.013 | 0.00 0.00 0.04 |
| `U_bs05_t3` | t3 | anq_stdfp base_scale=0.5 *(running)* | 0 | 8 | 0.053 | 0.072 | 0.00 0.00 0.00 0.00 0.20 0.08 0.06 0.02 |
| `K_mani_norm_t3` | t3 | mani_stdfp metric (scale-free) | 0 | 11 | 0.000 | 0.004 | 0.00 0.00 0.00 0.02 0.00 0.00 0.02 0.00 0.00 0.00 0.00 |
| `K1_mani_norm_t3_s1` | t3 | mani_stdfp metric (scale-free) *(running)* | 1 | 6 | 0.000 | 0.000 | 0.00 0.00 0.00 0.00 0.00 0.00 |
| `I_mani_weak_t3` | t3 | mani_stdfp metric (unnormalised) *(running)* | 0 | 5 | 0.000 | 0.000 | 0.00 0.00 0.00 0.00 0.00 |
| `I1_mani_weak_t3_s1` | t3 | mani_stdfp metric (unnormalised) *(running)* | 1 | 5 | 0.000 | 0.000 | 0.00 0.00 0.00 0.00 0.00 |
| `L_rebrac_a001_t3` | t3 | rebrac.py | 0 | 11 | 0.500 | 0.488 | 0.00 0.00 0.02 0.34 0.48 0.46 0.52 0.42 0.54 0.50 0.46 |
| `L1_rebrac_a001_t3_s1` | t3 | rebrac.py | 1 | 11 | 0.327 | 0.364 | 0.00 0.00 0.00 0.16 0.32 0.28 0.48 0.36 0.24 0.50 0.24 |
| `L2_rebrac_t3_s2` | t3 | rebrac.py *(running)* | 2 | 5 | 0.240 | 0.152 | 0.00 0.04 0.02 0.18 0.52 |
