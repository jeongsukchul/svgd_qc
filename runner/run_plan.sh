#!/usr/bin/env bash
# Launch every ON row in runner/experiments.tsv, keeping <= MAXJOBS alive.
# Flips each launched row to RUNNING, so re-running this script never
# double-launches.  Rows are taken top-to-bottom, so put the priority work first.
#
# Phase-3 rows (group starting W_ or V_) get eval_episodes=100 for tighter
# confidence intervals; screening rows stay at 50.
PLAN=/workspace/svgd_qc/runner/experiments.tsv
LOG=/workspace/svgd_qc/runner/run_plan.log
MAXJOBS=${MAXJOBS:-16}
MF=${MF:-0.10}
echo "$(date) run_plan start (max=$MAXJOBS)" >> "$LOG"
while true; do
  row=$(grep -nE '^ON[[:space:]]*\|' "$PLAN" | head -1)
  if [ -z "$row" ]; then
    # Don't exit the instant the queue drains -- rows are often appended while
    # runs are still going, and exiting here races those additions.  Idle-wait
    # up to IDLE_EXIT seconds (default 2h) for new ON rows before giving up.
    waited=0
    while [ -z "$row" ] && [ "$waited" -lt "${IDLE_EXIT:-7200}" ]; do
      sleep 60; waited=$((waited+60))
      row=$(grep -nE '^ON[[:space:]]*\|' "$PLAN" | head -1)
    done
    [ -z "$row" ] && { echo "$(date) queue idle ${waited}s, exiting" >> "$LOG"; break; }
    echo "$(date) new rows appeared after ${waited}s idle" >> "$LOG"
  fi
  ln=${row%%:*}
  body=${row#*:}
  IFS='|' read -r _ grp agent task seed flags <<< "$body"
  grp=$(echo "$grp" | xargs); agent=$(echo "$agent" | xargs)
  task=$(echo "$task" | xargs); seed=$(echo "$seed" | xargs)
  flags=$(echo "$flags" | sed 's/#.*//')

  while [ "$(pgrep -cf 'main.py --agent')" -ge "$MAXJOBS" ]; do sleep 60; done

  # least-busy GPU by live compute-process count
  best=0; bestc=99
  for g in 0 1 2 3 4 5 6 7; do
    c=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$g" 2>/dev/null | wc -l)
    [ "$c" -lt "$bestc" ] && { bestc=$c; best=$g; }
  done

  case "$grp" in
    W_*|V_*) EV=100 ;;
    *)       EV=50  ;;
  esac

  # Domain is selected by group prefix.  CD_ = cube-double manipulation:
  # horizon_length=5 with action chunking, discount 0.99 (per DSRL's own
  # command_dsrl.sh).  Everything else stays antmaze-giant at h=1 / 0.995.
  case "$grp" in
    HM_*)  DOM="ENV_PREFIX=humanoidmaze-medium-navigate DISCOUNT=0.995 HORIZON=1" ;;
    HL_*)  DOM="ENV_PREFIX=humanoidmaze-large-navigate DISCOUNT=0.995 HORIZON=1" ;;
    CQF_*) DOM="ENV_PREFIX=cube-double-play DISCOUNT=0.99 HORIZON=1" ;;   # qflow standard setting: flat 1-step
    CDX_*) DOM="ENV_PREFIX=cube-double-play DISCOUNT=0.99 HORIZON=5 STEPS=3000000" ;;
    CD*)   DOM="ENV_PREFIX=cube-double-play DISCOUNT=0.99 HORIZON=5" ;;
    *)    DOM="" ;;
  esac

  echo "$(date) gpu$best ev=$EV <- $grp" >> "$LOG"
  eval "$DOM EVAL_EPISODES=$EV /workspace/svgd_qc/runner/launch.sh $best $MF $grp $agent $task $seed $flags" >> "$LOG" 2>&1
  sed -i "${ln}s/^ON /RUNNING /" "$PLAN"
  sleep 25
done
