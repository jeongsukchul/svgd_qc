#!/usr/bin/env bash
# Keep the archive current while runs are in flight; exit ~20 min after the last one ends.
A=/home/sukchul/qc/experiment_logs
idle=0
while true; do
  "$A/sync.sh" >> "$A/sync.log" 2>&1
  if pgrep -f "main.py --agent" > /dev/null; then idle=0; else idle=$((idle+1)); fi
  [ "$idle" -ge 2 ] && { echo "$(date) all runs finished, final sync done" >> "$A/sync.log"; break; }
  sleep 600
done
