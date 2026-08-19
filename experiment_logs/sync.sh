#!/usr/bin/env bash
# Archive experiment logs + run artifacts into experiment_logs/, keeping up with
# runs still in progress. Safe to re-run; overwrites with latest.
cd /home/sukchul/qc
SP=/tmp/claude-1000/-home-sukchul-qc/682f546f-7056-4a11-8268-2787858681d7/scratchpad
A=/home/sukchul/qc/experiment_logs
mkdir -p $A/logs $A/runs $A/analysis $A/queues
for f in $SP/*.log; do
  [ -f "$f" ] || continue
  b=$(basename "$f")
  { tr '\r' '\n' < "$f" | grep -vE "it/s\]|it/s,|\?it/s" | grep -v '^\s*$'
    echo "--- last progress lines ---"
    tr '\r' '\n' < "$f" | grep -E "it/s" | tail -3
  } > "$A/logs/$b" 2>/dev/null
done
cp $SP/*.py $A/analysis/ 2>/dev/null
cp $SP/queue_*.sh $A/queues/ 2>/dev/null
for d in exp/beat/ant2/*/*/*/; do
  [ -d "$d" ] || continue
  g=$(echo "$d" | cut -d/ -f4)
  mkdir -p "$A/runs/$g"
  for f in eval.csv offline_agent.csv flags.json agent_source_path.txt; do
    [ -f "$d/$f" ] && cp "$d/$f" "$A/runs/$g/" 2>/dev/null
  done
  [ -d "$d/source_snapshot" ] && cp -r "$d/source_snapshot" "$A/runs/$g/" 2>/dev/null
done
# agent sources as currently patched
mkdir -p $A/agents
cp agents/anq_stdfp.py agents/mani_stdfp.py agents/anq_stdfp3.py agents/rebrac.py $A/agents/ 2>/dev/null
/home/sukchul/miniconda3/envs/fql/bin/python /home/sukchul/qc/experiment_logs/make_summary.py
