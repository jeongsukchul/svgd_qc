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

## Results

Paired by seed against `rebrac.py` under identical conditions (same task, same
seed, same protocol).  Score = mean of the last 5 evals of a 1M-step run.

| task | `anq_rfs` | `rebrac.py` | delta | t-test | seeds+ | verdict |
|---|---|---|---|---|---|---|
| t1 | **0.319 +- 0.048** (6) | 0.202 +- 0.045 | **+0.117** | p=0.037 | 5/6 | win |
| t2 | 0.821 +- 0.007 (3) | 0.806 +- 0.010 | +0.015 | p=0.123 | 3/3 | tie |
| t3 | **0.378 +- 0.024** (8) | 0.254 +- 0.055 | **+0.124** | p=0.055 | 6/8 | trend |
| t4 | 0.527 +- 0.234 (3) | 0.497 +- 0.180 | +0.030 | p=0.643 | 2/3 | tie |
| t5 | 0.877 +- 0.015 (3) | 0.862 +- 0.027 | +0.015 | p=0.735 | 2/3 | tie |

**Pooled over all tasks and seeds: `anq_rfs` wins 18 of 23 paired
comparisons, sign test p = 0.0106.**

The sign test is the right pooled statistic here: pairs are matched by seed
*and* task, and it is immune to the scale differences between tasks (t2 deltas
~0.015 vs t3 ~0.12) that would let t3 dominate a pooled mean.

### How to read this

Gains concentrate on the **hard** tasks and vanish on the easy ones:

* t1 and t3 are where `rebrac` is weak (0.20 / 0.25) and where `anq_rfs` adds
  +0.12 on both.  These are also the two tasks where `rebrac` catastrophically
  failed on individual seeds (t3 seed 5 scored 0.000 across all ten evals with
  a normally-trained critic; seed 7 scored 0.082).
* t2, t4 and t5 are tasks both agents largely solve (0.50-0.88).  There is
  little headroom and the two methods are indistinguishable.

t4 and t5 were run **after** the configuration was fixed and were never used for
any selection decision, so they are a clean out-of-sample test.  The result
there is neutral rather than negative: no advantage, but no degradation either.

### Honest limitations

* Only t1 clears p<0.05 on its own, and its rank test is marginal (wilcoxon
  p=0.062).  t3 misses at p=0.055.  The per-task tests are underpowered.
* Run-to-run std is ~0.12 against effects of ~0.12, so per-task n=3-8 is simply
  not enough; the pooled test is where the evidence lives.
* The configuration was selected by screening on t3, so t1/t2 are only partial
  generalization evidence.  t4/t5 are the clean test and they came back neutral.
* `anq_rfs` is not immune to the collapse mode either: `W_rfs_t4_s0` peaked at
  0.53 and then scored 0.000 for its last five evals.


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

| arm (t3, seed 0, 1M steps) | last5 | vs winner |
|---|---|---|
| `R17b` live base (no stop-gradient on the residual head's base input) | **0.496** | +0.018 |
| `R07` winner (action + data anchor + KL), seed 0 | 0.478 | -- |
| `R11b` deeper residual head (512,512) | 0.344 | -0.134 |
| `R12b` sharper drift BC (drift_temps=0.3) | 0.332 | -0.146 |
| `R09b` residual head not conditioned on base | 0.304 | -0.174 |
| `R15b` looser latent budget (target_multiplier=0.5) | 0.292 | -0.186 |
| `R04` absolute output + data + KL | 0.400 | -0.078 |
| `R05` absolute output + data + no reg | 0.456 | -0.022 |
| `R01` pretanh + data + KL | 0.184 | -0.294 |
| `R02` pretanh + data + no reg | 0.100 | -0.378 |
| `R08` pretanh, live base | 0.272 | -0.206 |
| `R03` pretanh + **residual anchor** | 0.028 | collapse |
| `R06` absolute + **residual anchor** | 0.004 | collapse |
| `R13b` lam=0.003 | **0.000** | collapse |
| `R14b` lam=0.03 | 0.008 | collapse |
| `S02b` SAC entropy on latent + residual | 0.000 | collapse |

**The collapse cluster is the clearest signal in the whole sweep.**  Five
configurations scored ~0.00: `bc_anchor="residual"` (twice), `lam=0.003`,
`lam=0.03`, and SAC entropy over both heads.  Every one of them either weakens
or redirects the quadratic pull of the *executed action* toward the *dataset
action*.  `lam=0.01` sits in a narrow viable window -- 3x lower and 3x higher
both destroy the agent.

This is also a negative result for the SAC-style variant: an entropy bonus
actively rewards moving away from the BC anchor, which is exactly wrong in this
offline sparse-reward setting.  Notably RFS itself uses PPO for its *online*
fine-tuning but a plain BC-regularised objective offline, which is consistent.


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

## Raw paired comparison output

```
==============================================================================
PAIRED COMPARISON  (mean +/- sem of last-5 evals over seeds; * = incomplete)
==============================================================================
arm                                         t1            t2            t3            t4            t5
anq_rfs (action+data+kl)         0.319+-0.048(6) 0.821+-0.007(3) 0.378+-0.024(8) 0.527+-0.234(3) 0.877+-0.015(3)
rebrac.py                        0.202+-0.045(6) 0.806+-0.010(3) 0.254+-0.055(8) 0.497+-0.180(3) 0.862+-0.027(3)
anq_rfs (absolute+data+none)     0.224+-0.000(1) 0.864+-0.000(1) 0.278+-0.000(1)            --            --
------------------------------------------------------------------------------
delta (anq_rfs - rebrac), paired by seed where both exist:
  t1: +0.117 +- 0.041  (n=6)   [s0: 0.238v0.094, s1: 0.340v0.232, s2: 0.184v0.056, s3: 0.242v0.312, s4: 0.482v0.324, s5: 0.428v0.194]  t=2.83 p=0.037  [5/6 seeds+] wilcoxon p=0.062
  t2: +0.015 +- 0.006  (n=3)   [s0: 0.834v0.824, s1: 0.814v0.788, s2: 0.814v0.806]  t=2.57 p=0.123  [3/3 seeds+]  (n.s.)
  t3: +0.124 +- 0.054  (n=8)   [s0: 0.478v0.260, s1: 0.280v0.162, s2: 0.416v0.408, s3: 0.402v0.336, s4: 0.330v0.378, s5: 0.310v0.000, s6: 0.374v0.402, s7: 0.432v0.082]  t=2.30 p=0.055  [6/8 seeds+] wilcoxon p=0.078  (n.s.)
  t4: +0.030 +- 0.055  (n=3)   [s0: 0.078v0.146, s1: 0.636v0.602, s2: 0.866v0.742]  t=0.54 p=0.643  [2/3 seeds+]  (n.s.)
  t5: +0.015 +- 0.038  (n=3)   [s0: 0.902v0.860, s1: 0.850v0.910, s2: 0.878v0.816]  t=0.39 p=0.735  [2/3 seeds+]  (n.s.)
POOLED over all tasks/seeds: anq_rfs wins 18/23 pairs   sign-test p=0.0106
  (wilcoxon on pooled deltas p=0.0043 -- scale-mixed, sign test is the cleaner read)
==============================================================================
```


## Latent-actor ablations (t3, 3 seeds each unless noted)

The winner's latent head is a stochastic Gaussian with a KL-to-prior penalty and
a learned dual.  The 2x2 below separates the two factors -- does `z` need to be
*sampled*, and does it need to be *regularised*?

| latent head | no regulariser | KL to prior |
|---|---|---|
| deterministic | 0.267 +- 0.046 | **0.340 +- 0.055** |
| stochastic | 0.197 +- 0.035 | **0.378 +- 0.024** (n=8) |

**The KL is load-bearing; the sampling is not.**  Removing the regulariser costs
0.07 (deterministic) to 0.18 (stochastic).  Making the latent deterministic
costs ~0.04, which is inside the noise floor.  A DDPG-style deterministic latent
head with a KL penalty is therefore a valid simplification: it drops the
reparameterised sampling, the entropy/KL dual and `train_latent_stochastic`
with no meaningful loss.

`target_multiplier` (the KL budget, `budget = target_multiplier * action_dim`)
is **not** a sensitive knob over 0.03-0.25: 0.441 / 0.377 / 0.404 against the
default's 0.378, non-monotone, all within noise.  Only the extreme 0.5 looked
bad (0.292, n=1).  It was never tuned and does not need to be.

### sigreg does not work in this architecture

`agents/stdfp.py` regularises its DDPG latent branch with `_sigreg_strong_loss`,
an empirical-characteristic-function match of the *batch marginal* of `z` to
N(0, I).  That is the natural choice for a deterministic head, where a
per-sample KL degenerates.  Ported here as `latent_reg="sigreg"`, it fails
completely: **11 runs, 4 configurations, every eval 0.00, `best` never above
0.00.**  Coefficient 0.1 / 1.0 / 10 make no difference, and it fails with a
stochastic latent too, so it is neither a scale problem nor a
deterministic-pairing problem.

The training trace shows why -- `SIG_det_sigreg_t3_s0`:

```
step      actor_q    bc_offset   delta_rms   latent_mean_abs
170000     -201.8      0.654       0.112        0.800
500000     -282.2      0.626       0.121        0.800
830000     -591.7      8.658       1.408      136,600
995000     -281.6     10.720       0.996    4,819,000
```

`z` is stable at ~0.8 until ~700k and then diverges to 4.8e6, dragging
`bc_offset` from 0.63 to 10.7.  The ECF objective is **periodic in `z`**
(`exp(i * proj * t)`), so once projections alias past the `t in [-5, 5]` grid
there is no restoring force at any coefficient.  A per-sample KL has an
unbounded `||mean||^2` term that always pulls inward; the ECF objective does
not, and nothing else in `anq_rfs` bounds `||z||`.  Using it here would require
an explicit magnitude bound (tanh-squashed latent head, or a small `||z||^2`
term alongside it).

### Full ablation ranking

```
arm                                           n    mean    sem   seeds
live base (no stop-grad on base input)        3   0.465  0.099   0.61 0.51 0.28
target_multiplier 0.03                        3   0.441  0.046   0.53 0.42 0.37
target_multiplier 0.25                        3   0.404  0.034   0.47 0.37 0.37
winner: action + data + KL, stochastic z      8   0.378  0.024   0.48 0.43 0.42 0.40 0.37 0.33 0.31 0.28
target_multiplier 0.06                        3   0.377  0.071   0.50 0.38 0.25
DETERMINISTIC z + KL                          3   0.340  0.055   0.45 0.30 0.27
DETERMINISTIC z, no reg                       3   0.267  0.046   0.35 0.25 0.20
stochastic z, no reg                          3   0.197  0.035   0.23 0.23 0.13
SAC entropy on latent                         3   0.069  0.030   0.12 0.07 0.02
sigreg coeff 0.1 (det z)                      2   0.000  0.000   0.00 0.00
sigreg coeff 1.0 (det z)                      2   0.000  0.000   0.00 0.00
sigreg coeff 10  (det z)                      2   0.000  0.000   0.00 0.00
sigreg (stochastic z)                         2   0.000  0.000   0.00 0.00
```

Two caveats on this table:

* `live base` tops it at 0.465 but with n=3 and a 0.33 seed spread (0.61 / 0.51
  / 0.28) it is not separable from the winner's 0.378 at n=8.  It is a genuine
  candidate, not a confirmed improvement.
* Arms were **selected into this table by being the top of an earlier n=1
  screen**, so regression to the mean is expected.  `SAC entropy on latent`
  demonstrates it: it screened at 0.516, the best single-seed result in the
  project, and replicated at **0.069** over three seeds.  Single-seed rankings
  on this benchmark are close to worthless.
