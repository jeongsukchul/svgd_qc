#!/usr/bin/env bash
# Phase 1: 16 runs, 2 per GPU, tasks t1 and t3, seed 0.
L=/workspace/svgd_qc/runner/launch.sh
CA="--agent.drift_temps=3.0 --agent.lam=0.01 --agent.refine_anchor=data --agent.refine_residual_space=pretanh"
ANQ=agents/anq_stdfp.py
REB=agents/rebrac.py
MF=0.11
g=0
launch () {  # launch <group> <agent> <task> <extra...>
  $L $((g % 8)) $MF "$1" "$2" "$3" 0 "${@:4}"
  g=$((g+1)); sleep 3
}
for T in 1 3; do
  launch "A_rebrac_t$T"     $REB $T
  launch "B_cur_t$T"        $ANQ $T $CA
  launch "C_bs0_t$T"        $ANQ $T $CA --agent.base_scale=0.0
  launch "D_bon32_t$T"      $ANQ $T $CA --agent.best_of_n=32 --agent.latent_deterministic=False
  launch "E_sharp_t$T"      $ANQ $T --agent.drift_temps=0.3 --agent.lam=0.01 --agent.refine_anchor=data --agent.refine_residual_space=pretanh
  launch "F_h5_t$T"         $ANQ $T $CA --horizon_length=5
  launch "G_rebrac_h5_t$T"  $REB $T --horizon_length=5 --agent.action_chunking=True
  launch "H_bonsharp_t$T"   $ANQ $T --agent.drift_temps=0.3 --agent.lam=0.01 --agent.refine_anchor=data --agent.refine_residual_space=pretanh --agent.best_of_n=32 --agent.latent_deterministic=False
done
