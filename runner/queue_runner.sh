#!/usr/bin/env bash
# Keeps at most MAXJOBS main.py runs alive, launching queued arms as slots free.
MAXJOBS=${MAXJOBS:-22}
QUEUE=/workspace/svgd_qc/runner/queue.txt
LOG=/workspace/svgd_qc/runner/queue_runner.log
touch "$QUEUE"
echo "$(date) queue_runner start (max=$MAXJOBS)" >> "$LOG"
while true; do
  # strip blank/comment lines
  line=$(grep -vE '^\s*(#|$)' "$QUEUE" | head -1)
  [ -z "$line" ] && { echo "$(date) queue empty, exiting" >> "$LOG"; break; }
  n=$(pgrep -cf 'main.py --agent')
  if [ "$n" -lt "$MAXJOBS" ]; then
    # pick the GPU with the fewest live runs
    best=0; bestc=9999
    for g in 0 1 2 3 4 5 6 7; do
      c=$(pgrep -af 'main.py --agent' | grep -c "GPU$g=" 2>/dev/null || true)
      c=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader -i $g 2>/dev/null | wc -l)
      if [ "$c" -lt "$bestc" ]; then bestc=$c; best=$g; fi
    done
    echo "$(date) launching on gpu$best ($n live): $line" >> "$LOG"
    # shellcheck disable=SC2086
    /workspace/svgd_qc/runner/launch.sh $best 0.10 $line >> "$LOG" 2>&1
    # remove that line from the queue
    grep -vxF "$line" "$QUEUE" > "$QUEUE.tmp" && mv "$QUEUE.tmp" "$QUEUE"
    sleep 30
  else
    sleep 60
  fi
done
