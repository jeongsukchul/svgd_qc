#!/usr/bin/env bash
# Periodically push results to HF while runs are in flight; exit once
# everything has been idle for two consecutive checks.
INTERVAL=${INTERVAL:-1800}
idle=0
while true; do
  /workspace/svgd_qc/runner/sync_hf.sh >> /workspace/svgd_qc/runner/sync.log 2>&1
  echo "$(date) synced (live=$(pgrep -cf 'main.py --agent'))" >> /workspace/svgd_qc/runner/sync.log
  if [ "$(pgrep -cf 'main.py --agent')" -eq 0 ] && ! pgrep -f "bash .*run_plan.sh" >/dev/null; then
    idle=$((idle+1))
    [ "$idle" -ge 2 ] && { echo "$(date) all idle, daemon exit" >> /workspace/svgd_qc/runner/sync.log; break; }
  else
    idle=0
  fi
  sleep "$INTERVAL"
done
