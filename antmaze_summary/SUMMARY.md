# antmaze-giant: complete summary (portable)

Protocol: `antmaze-giant-navigate-singletask-task{N}-v0`, horizon_length=1 (no
chunking), discount=0.995, 1M offline steps, eval every 100k (50-100 episodes),
num_qs=2, best_of_n=1.  Score = mean of last-5 evals; n = independent seeds.
184 antmaze runs indexed in `all_runs_index.json`; per-run eval.csv+flags.json
under `runs/`; exact winner flags + agent sources under `configs/`.

## Winning anq_rfs config (validated)
```
--agent=agents/anq_rfs.py  --discount=0.995 --horizon_length=1
--agent.refine_output_mode=action  --agent.bc_anchor=data  --agent.lam=0.01
--agent.latent_reg=kl  --agent.target_multiplier=0.125
--agent.drift_temps=3.0  --agent.q_agg=min
# all else default: adam_eps=1e-8, drift_force_norm=unit, unsquashed latent,
# latent_deterministic=True(exec) / train_latent_stochastic=True,
# residual_sees_stopped_base=True, gen_per_label=8, lr=3e-4, 4x512 MLPs
```

## Head-to-head vs ReBRAC (same protocol, paired-capable seeds)

| task | anq_rfs (n) | rebrac (n) | delta |
|---|---|---|---|
| t1 | 0.325 (11) | 0.184 (7) | +0.141 |
| t2 | 0.838 (5) | 0.806 (3) | +0.032 |
| t3 | 0.386 (10) | 0.269 (9) | +0.116 |
| t4 | 0.538 (6) | 0.497 (3) | +0.041 |
| t5 | 0.873 (5) | 0.862 (3) | +0.011 |
| **agg** | **0.592** | 0.524 | +0.068 |

Validated earlier with seed-paired stats: anq_rfs wins 18/23 pairs across the 5
tasks, sign test p = 0.011 (t1 p=0.037; t3 p=0.055; t2/t4/t5 ties).

## vs RQL-paper Full Result Table (arXiv 2606.17551, their protocol, num_qs=10)
Table best per task: t1 62 (QSM), t2 73 (ReBRAC), t3 21 (RQL), t4 74 (ReBRAC),
t5 89 (ReBRAC); best aggregate 57 (ReBRAC).  Ours (num_qs=2): see table above --
t2 and t3 beat every algorithm in their table; aggregate exceeds their best.

## Key antmaze-specific findings (all measured here)
1. bc_anchor=data with lam=0.01 is required.  lam window is sharp: 0.003 and
   0.03 both collapse to ~0.  Residual anchor at lam=0.01 collapses (~0) --
   but that test is confounded by lam scale; at proper lam it was never run
   under the final protocol except via livebase (below).
2. drift_temps=3.0 (blurry BC): the decoder collapses to a near-deterministic
   s->a map (decoder-to-data distance 0.147 ~ fit error).  Latent steering is
   then impotent; ALL RL improvement flows through the refine head.  This is
   why the weak data anchor wins: it frees the refine head.
3. adam_eps MUST stay 1e-8: global eps=3e-3 collapses antmaze to 0.000 with
   late critic divergence (dq ~ -179).  Decoder-only eps is safe but useless.
   (Opposite of cube-double, where eps=3e-3 is the fix -- domain-matched knob.)
4. Unified recipe (raw force + drift_adam_eps=1e-4, global 1e-8): 0.336 (n=5)
   vs 0.378 -- costs ~0.04 on antmaze, near-parity.
5. residual_sees_stopped_base=False (live base): 0.421-0.465 across n=6-9 vs
   0.378 (n=8) -- consistently positive but ~1 SEM; not statistically separable.
   Best candidate for further seeds in a new setting.
6. target_multiplier insensitive over 0.03-0.25 (0.44/0.38/0.40 vs 0.378);
   0.5 slightly worse (n=1); tm=1.0 helps t1 specifically (0.780 vs 0.691),
   tm=2.0 collapses.
7. q_agg=min required (mean scored 0.000 in the matched pair CDN30/CDNm30).
8. Latent 2x2 (t3): KL regulariser is load-bearing (drop it: 0.378->0.20-0.27);
   sampling is nearly free to remove (deterministic z + KL = 0.340).
9. gen_per_label: 8 default fine; 32 collapses training (decoder overfit +
   latent dual divergence) on cube -- untested at 32 on antmaze; 16 untested.

## Per-arm aggregate table

| arm | task | n | last5 |
|---|---|---|---|
| W_rfs_t1 | t1 | 11 | 0.325 |
| AMU_rawdeps_t1 | t1 | 2 | 0.286 |
| V_abs_t1 | t1 | 1 | 0.224 |
| W_reb_t1 | t1 | 6 | 0.202 |
| E | t1 | 1 | 0.200 |
| B_cur_t1 | t1 | 1 | 0.176 |
| A_rebrac_t1 | t1 | 1 | 0.076 |
| V_abs_t2 | t2 | 1 | 0.864 |
| W_rfs_t2 | t2 | 5 | 0.838 |
| W_reb_t2 | t2 | 3 | 0.806 |
| MIS_CDX_win_t2_s0_antmaze-giant-navigate-singletask-task2-v0 | t2 | 1 | 0.300 |
| MIS_antmaze-giant-navigate-singletask-task2-v0 | t2 | 32 | 0.191 |
| MIS_CDX_win_t2_s1_antmaze-giant-navigate-singletask-task2-v0 | t2 | 1 | 0.140 |
| MIS_CDX_win_t2_s2_antmaze-giant-navigate-singletask-task2-v0 | t2 | 1 | 0.088 |
| R07_act_data_kl_t3 | t3 | 1 | 0.524 |
| S01b | t3 | 1 | 0.516 |
| R17b_livebase_t3 | t3 | 1 | 0.496 |
| R05_abs_data_no_t3 | t3 | 1 | 0.456 |
| TM03_t3 | t3 | 3 | 0.441 |
| LB_livebase_t3 | t3 | 6 | 0.421 |
| TM25_t3 | t3 | 3 | 0.404 |
| R04_abs_data_kl_t3 | t3 | 1 | 0.400 |
| A_rebrac_t3 | t3 | 1 | 0.396 |
| W_rfs_t3 | t3 | 10 | 0.386 |
| TM06_t3 | t3 | 3 | 0.377 |
| R11b_deephead_t3 | t3 | 1 | 0.344 |
| DETK_detlat_kl_t3 | t3 | 3 | 0.340 |
| R12b | t3 | 1 | 0.332 |
| AMU_deps33_t3 | t3 | 1 | 0.332 |
| AMU_rawdeps_t3 | t3 | 7 | 0.315 |
| R09b_nobasecond_t3 | t3 | 1 | 0.304 |
| R15b_looselatent_t3 | t3 | 1 | 0.292 |
| B_cur_t3 | t3 | 1 | 0.280 |
| V_abs_t3 | t3 | 1 | 0.278 |
| R08_livebase_t3 | t3 | 1 | 0.272 |
| DET_detlat_t3 | t3 | 3 | 0.267 |
| W_reb_t3 | t3 | 8 | 0.254 |
| S04b_kl | t3 | 2 | 0.198 |
| NOR_stoch_noreg_t3 | t3 | 3 | 0.197 |
| R01_pre_data_kl_t3 | t3 | 1 | 0.184 |
| E | t3 | 1 | 0.168 |
| R02_pre_data_no_t3 | t3 | 1 | 0.100 |
| SEes20_t3 | t3 | 2 | 0.074 |
| SE_ent_lat_t3 | t3 | 3 | 0.069 |
| R16b_nq4_t3 | t3 | 1 | 0.060 |
| SEes05_t3 | t3 | 2 | 0.052 |
| R03_pre_res_kl_t3 | t3 | 1 | 0.028 |
| R06_abs_res_kl_t3 | t3 | 1 | 0.004 |
| R14b_lam03_t3 | t3 | 1 | 0.004 |
| R13b_lam003_t3 | t3 | 1 | 0.000 |
| S02b | t3 | 1 | 0.000 |
| S05b | t3 | 1 | 0.000 |
| S06b | t3 | 1 | 0.000 |
| SIG_det_sigreg_t3 | t3 | 3 | 0.000 |
| SIGS_stoch_sigreg_t3 | t3 | 3 | 0.000 |
| SIGc01_t3 | t3 | 2 | 0.000 |
| SIGc10_t3 | t3 | 2 | 0.000 |
| AR_resid10_t3 | t3 | 2 | 0.000 |
| AME_eps33_t3 | t3 | 2 | 0.000 |
| AME_lbeps_t3 | t3 | 1 | 0.000 |
| W_rfs_t4 | t4 | 6 | 0.538 |
| W_reb_t4 | t4 | 3 | 0.497 |
| W_rfs_t5 | t5 | 5 | 0.873 |
| W_reb_t5 | t5 | 3 | 0.862 |
| AMU_rawdeps_t5 | t5 | 2 | 0.858 |
