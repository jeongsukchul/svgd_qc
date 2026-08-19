#!/usr/bin/env bash
# Phase 2a: core anq_rfs grid on t3 (8 arms). GPU2 is free; spread the rest.
L=/workspace/svgd_qc/runner/launch.sh
A=agents/anq_rfs.py
T=3; MF=0.10
C="--agent.drift_temps=3.0 --agent.lam=0.01"
i=0
GPUS=(2 2 6 1 4 5 3 7)
launch () { $L ${GPUS[$i]} $MF "$1" $A $T 0 $C "${@:2}"; i=$((i+1)); sleep 4; }
launch "R01_pre_data_kl_t$T"  --agent.refine_output_mode=pretanh  --agent.bc_anchor=data     --agent.latent_reg=kl
launch "R04_abs_data_kl_t$T"  --agent.refine_output_mode=absolute --agent.bc_anchor=data     --agent.latent_reg=kl
launch "R02_pre_data_no_t$T"  --agent.refine_output_mode=pretanh  --agent.bc_anchor=data     --agent.latent_reg=none
launch "R03_pre_res_kl_t$T"   --agent.refine_output_mode=pretanh  --agent.bc_anchor=residual --agent.latent_reg=kl
launch "R05_abs_data_no_t$T"  --agent.refine_output_mode=absolute --agent.bc_anchor=data     --agent.latent_reg=none
launch "R06_abs_res_kl_t$T"   --agent.refine_output_mode=absolute --agent.bc_anchor=residual --agent.latent_reg=kl
launch "R07_act_data_kl_t$T"  --agent.refine_output_mode=action   --agent.bc_anchor=data     --agent.latent_reg=kl
launch "R08_livebase_t$T"     --agent.refine_output_mode=pretanh  --agent.residual_sees_stopped_base=False
