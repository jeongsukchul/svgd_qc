#!/usr/bin/env bash
# Phase 2: broad anq_rfs (unified latent+residual actor) sweep, screened on t3.
# t3 has the widest dynamic range in the archive: rebrac 0.426 vs anq_stdfp 0.312.
L=/workspace/svgd_qc/runner/launch.sh
A=agents/anq_rfs.py
T=${1:-3}
MF=0.11
# common: match the archive's BC sharpness and BC weight
C="--agent.drift_temps=3.0 --agent.lam=0.01"
g=0
launch () { $L $((g % 8)) $MF "$1" $A $T 0 $C "${@:2}"; g=$((g+1)); sleep 3; }

# --- composition x anchor x latent-regulariser grid ---
launch "R01_pre_data_kl_t$T"    --agent.refine_output_mode=pretanh  --agent.bc_anchor=data     --agent.latent_reg=kl
launch "R02_pre_data_no_t$T"    --agent.refine_output_mode=pretanh  --agent.bc_anchor=data     --agent.latent_reg=none
launch "R03_pre_res_kl_t$T"     --agent.refine_output_mode=pretanh  --agent.bc_anchor=residual --agent.latent_reg=kl
launch "R04_abs_data_kl_t$T"    --agent.refine_output_mode=absolute --agent.bc_anchor=data     --agent.latent_reg=kl
launch "R05_abs_data_no_t$T"    --agent.refine_output_mode=absolute --agent.bc_anchor=data     --agent.latent_reg=none
launch "R06_abs_res_kl_t$T"     --agent.refine_output_mode=absolute --agent.bc_anchor=residual --agent.latent_reg=kl
launch "R07_act_data_kl_t$T"    --agent.refine_output_mode=action   --agent.bc_anchor=data     --agent.latent_reg=kl
# --- gradient-coupling / architecture ablations (all on the pretanh+data+kl base) ---
launch "R08_livebase_t$T"       --agent.refine_output_mode=pretanh --agent.residual_sees_stopped_base=False
launch "R09_nobasecond_t$T"     --agent.refine_output_mode=pretanh --agent.condition_residual_on_base=False
launch "R10_bon16_t$T"          --agent.refine_output_mode=pretanh --agent.best_of_n=16 --agent.latent_deterministic=False
launch "R11_deephead_t$T"       --agent.refine_output_mode=pretanh --agent.residual_head_hidden_dims="(512,512)"
# --- scalar sweeps ---
launch "R12_sharpbc_t$T"        --agent.refine_output_mode=pretanh --agent.drift_temps=0.3
launch "R13_lam003_t$T"         --agent.refine_output_mode=pretanh --agent.lam=0.003
launch "R14_lam03_t$T"          --agent.refine_output_mode=pretanh --agent.lam=0.03
launch "R15_looselatent_t$T"    --agent.refine_output_mode=pretanh --agent.target_multiplier=0.5
launch "R16_nq4_t$T"            --agent.refine_output_mode=pretanh --agent.num_qs=4
