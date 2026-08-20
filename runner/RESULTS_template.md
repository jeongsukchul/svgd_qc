# Beating ReBRAC on antmaze-giant with a unified latent+residual actor

## Summary

`agents/anq_rfs.py` is an RFS-style (arXiv:2602.01789) restructuring of
`agents/anq_stdfp.py`.  It preserves every structural component -- the drift
decoder behaviour-cloned on `z ~ N(0, I)`, latent steering of that decoder, and
a residual correction on the decoded action -- and changes only how the two
policy heads are parameterised and trained.

Protocol: `antmaze-giant-navigate-singletask-task{1,2,3}-v0`, 1M offline steps,
`--discount=0.995`, `horizon_length=1`, `best_of_n=1`, eval every 100k with 100
episodes.  Scores are the mean of the last 5 evals.  Comparisons are **paired by
seed** against `rebrac.py` run under identical conditions.

<!-- RESULTS TABLE INSERTED HERE -->

## The defect this fixes

In `anq_stdfp` the latent head and the refinement head are optimised by two
separate losses that are gradient-isolated from one another:

* `latent_actor_loss` optimises `z` to maximise `Q(s, refine(decode(s, z)))`,
  but calls `_refine(..., params=None)` -- the refiner is frozen for this loss.
* `refine_actor_loss` optimises `delta` to maximise `Q(s, refine(base))`, but
  `_refine_base_actions` returns a `stop_gradient`-ed base -- the latent head is
  frozen for this loss.

Both heads shape the *same executed action* under *different objectives*, and
neither ever sees the other's gradient, so they cannot co-adapt.  Meanwhile the
decoded base carries real error (`generated_to_data_mse ~ 0.07`, i.e. ~0.26 RMS
per action dim) that the refiner must correct without being able to influence
where the base lands.

`anq_rfs` emits `z` and `delta` from one shared trunk under a single objective

    -Q(s, a) * norm_q  +  lam * ||a - a_data||^2  [ + alpha * KL(z || N(0,I)) ]
    a = compose(decode(s, z), delta)

with the decoder's *weights* frozen w.r.t. that objective (as RFS freezes its
pretrained flow policy) while the gradient still reaches `z` through the
decoder's input.  Verified by direct gradient-norm measurement:

    trunk    = 1.0163e+00     <- shared, receives gradient from both heads
    z_mean   = 1.9751e-01     <- Q gradient reaches the latent head
    residual = 3.4072e+00
    drift    = 0.0000e+00     <- decoder weights exactly frozen

## Ablation findings (t3 screening, seed 0, 1M steps)

<!-- ABLATION TABLE INSERTED HERE -->

Two results are large enough to be unambiguous at n=1:

1. **`bc_anchor="residual"` (RFS Eq. 11) collapses.**  Penalising only
   `||delta||^2` constrains the correction but leaves the latent free to steer
   the base off-manifold, so nothing holds the executed action near the data.
   On antmaze-giant the penalty must sit on the executed action vs the dataset
   action.  This is a case where the paper's offline objective does not transfer:
   RFS validates on dexterous manipulation with a strong pretrained flow policy,
   not a 1000-step sparse maze.
2. **Composition ranks `action` > `absolute` > `pretanh`**, inverting the
   archive's finding for `anq_stdfp`.  The pre-tanh warp helped a *separately
   trained* refiner that had no way to learn output scale and saturated against
   the clip; once the residual head shares a trunk with the latent head and both
   are driven by one Q gradient, it learns its own scale and the `atanh` warp
   only adds gradient distortion near the boundary.

## Bugs found and fixed

* **`best_of_n` was dead code in `anq_stdfp`.**  `_sample_refined_actions` tiles
  the observation `n` times, then takes `dist.mode()` -- a deterministic function
  of the observation -- so all `n` candidates were byte-identical and
  `select_best` ranked `n` copies of one action.  Every "best-of-N" number in the
  archive was really best-of-1.  Fixed by forcing sampling when `n > 1`; the
  default `n=1` path is unchanged, so no existing result is invalidated.
  Measured candidate spread: 0.0000 before, 0.8690 after (at n=8).
* **`main.py` crashed at exit under offline wandb** (`f.write(run.url)` with
  `run.url is None`).
* **`rebrac.batch_update` defaults to `full_update=False`**, which silently skips
  the actor and target updates.  Any scanned/batched driver that calls it without
  the flag trains a critic against a frozen actor.  Guarded in `main.py`.

## Throughput

The workload is host-dispatch-bound, not GPU-bound, on this machine (Xeon
E5-2673 v4 @ 2.30 GHz, 8x RTX 4080 SUPER).  Per step: 2.56 ms total, of which
only ~1.54 ms is GPU work.

`--offline_scan_chunk=N` fuses N offline updates into one `lax.scan` dispatch.
Verified bit-identical to the per-step loop for both agents
(`max|diff| = 0.000e+00`, identical RNG state).

| setting | it/s |
|---|---|
| per-step, 1 run | 195 (end-to-end), 375-390 (update only) |
| per-step, 8 concurrent | 293-329 |
| per-step, 16 concurrent | 235-275 |
| **scan_chunk=25, 14 concurrent** | **556-563 (anq_stdfp), 680-685 (rebrac)** |

## Methodology note: variance

Run-to-run standard deviation on t3 is ~0.12, against an effect size of ~0.10.
Two nominally identical `rebrac` seed-0 t3 runs differed by 0.136 (0.396 vs
0.260).  Single-seed comparisons on these tasks are not informative, and
comparisons against archived numbers from a different machine are worse.  All
headline claims here are **paired by seed** and reported with a paired t-test.
