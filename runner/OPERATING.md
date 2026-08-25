# OPERATING RULES — read at EVERY wake before doing anything else

## Rule 0: CURRENT TARGETS + ALGORITHM (user, 2026-08-25)
- THE algorithm is mani_stdfp (anq_stdfp + manifold-metric refine). anq_rfs = ablation only.
- Headline comparisons: cube-double vs DSRL(our protocol, agg .72); antmaze-giant vs
  rebrac (.52, already beaten); humanoidmaze-medium vs Q-Flow (.81).
- humanoidmaze-large: DROPPED by user.
- Antmaze coverage via degenerate metric (normalize + ridge=100 = Euclidean).

## Prime directive
- Standing protocol doc: docs/OGBench_Researcher_Prompt_v2.md (round format = its §11; dead axes = 부록 B).
 (user mandate, standing)
Autonomously tune the two-component algorithm to SOTA. Do not wait for user replies.

## Rule 1: GPUs MUST NEVER be idle  (violated 3x — the user is rightly angry)
- The queue (`experiments.tsv` ON rows) must NEVER be empty. Target: >= 15 ON rows at all times.
- Seed-depth work is ALWAYS useful: every task leader and every dsrl baseline
  accumulates seeds toward n=8 whenever no probe is pending. Queue it in bulk.
- At every wake: FIRST check `ON` count and live count. If ON < 10, refill with
  seed depth BEFORE any analysis. Analysis never justifies an empty queue.
- Small "information-optimal" batches are the failure mode. Queue deep; insert
  probes ABOVE the standing backlog when a hypothesis needs testing.

## Rule 2: the algorithm structure is FIXED
- drift-BC decoder + in-support latent actor (DSRL-like) + out-of-support refine
  actor (anq_stdfp / anq_rfs). Both components present. Instantiations allowed:
  anq_stdfp (separate nets) and anq_rfs (shared trunk). stdfp/dsrl runs are
  baselines/diagnostics only.
- num_qs=2 and best_of_n=1 are FIXED for fair comparison. 1M steps only.

## Rule 1b: SEED BUDGET (user correction — depth was crowding out exploration)
- Screening: n=2-3 (kills n=1 flattery; never rank on n=1).
- Promising arm: confirm at n=4-5. STOP there unless it is a final headline claim.
- Final headline claims only (the per-task leaders in the paper table): n=8 cap.
  NOTHING goes past n=8. SEM gain beyond that is ~0.005 — pure waste.
- Backlog filler priority: (1) untested cells/probes at n=2, (2) confirmations
  to n=4-5, (3) headline depth to n=8. Depth is the LAST resort filler, not the
  default.

## Rule 2b: USER-VETOED directions (do NOT test these)
- Multi-scale drift kernel (drift_multi_temp / R_list with several temps): user strictly vetoed.
- n-step TD returns via h>1 on locomotion (breaks the h=1 protocol): vetoed.
- Refine-head per-module lr: allowed but user judges it won't matter -- lowest priority.
- Critic expectile > 0.5: allowed, but user expects it to fail; keep to the n=1 probes already queued.
- State-dependent latent std: allowed.

## Rule 3: analysis discipline
- Read the FULL metric set (dashboard.py): eval + decoder mse + latent KL vs
  budget + delta_rms + critic q & drift.
- Never conclude from n=1 or mid-run curves (repeatedly burned: n=1 flattery
  0.516->0.069, 0.840->0.756; cube arms sit at 0 until 400k).
- Verify env_name from each run's flags.json before aggregating (57-run
  mislaunch happened; done-markers must be eval-row counts, not log strings).
- Baselines need seed depth too (dsrl n=1 numbers moved by +-0.07).

## Rule 3c: isolation runs get the component's OWN config sweep
When testing a component in isolation (e.g. stdfp without the refine head), do
NOT inherit the parent recipe's constants (drift_temps, agg, freeze) and then
judge the component -- sweep its own key axes first (its defaults + the domain
candidates).  Premature call made 2026-08-24: "stdfp is weak everywhere on
humanoid" from ONE inherited config at n<=2 mid-run.  User corrected it.

## Rule 3b: after EDITING run_plan.sh, ALWAYS kill + restart the runner
The runner is a long-lived bash process; it never re-reads domain cases.  The
57-run mislaunch AND the HM_rfs_t1 mislaunch both came from stale runners.
After restart, verify the FIRST launched row's env dir matches its domain.

## Rule 4: hygiene at every wake
- `find exp -name "*.pkl" -delete`  (checkpoints disabled but old-code runs write them)
- Confirm `run_plan.sh` alive (restart with IDLE_EXIT=999999999) and sync daemon alive.
- Disk check; HF sync via daemon.

## Current targets (dsrl on OUR protocol, cube-double)
t1 0.924 (tied) | t2 ~0.856 (ours 0.780) | t3 0.865 (ours 0.799) |
t4 0.295 (ours 0.263, within noise) | t5 0.707 (ours 0.743 WON)
Antmaze-giant: agg 58.6 vs table-best 57 — banked.
