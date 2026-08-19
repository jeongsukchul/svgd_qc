#!/usr/bin/env bash
# Assemble a lightweight results tree and push it to the Hugging Face Hub.
# Re-runnable: rebuilds the tree and re-uploads (HF dedupes unchanged files).
set -u
export HF_HOME=/workspace/.hf_home
SRC=/workspace/svgd_qc
OUT=$SRC/hf_upload
REPO=${HF_REPO:-tjrcjf410/svgd-qc-antmaze-giant}

rm -rf "$OUT"; mkdir -p "$OUT"/{runs,agents,runner,logs,patches}

# 1. per-run metrics + config + the exact agent source that run used
for ev in "$SRC"/exp/beat/ant2/*/*/*/eval.csv; do
  [ -f "$ev" ] || continue
  d=$(dirname "$ev"); g=$(basename "$(dirname "$(dirname "$d")")")
  # $d = .../ant2/<group>/<env>/<expname>
  g=$(echo "$ev" | awk -F'/ant2/' '{print $2}' | cut -d/ -f1)
  mkdir -p "$OUT/runs/$g"
  cp -f "$d/eval.csv" "$d/flags.json" "$OUT/runs/$g/" 2>/dev/null
  cp -f "$d/offline_agent.csv" "$OUT/runs/$g/" 2>/dev/null
  [ -d "$d/source_snapshot" ] && cp -rf "$d/source_snapshot" "$OUT/runs/$g/" 2>/dev/null
done

# 2. agent sources under test
cp -f "$SRC"/agents/{anq_rfs,anq_stdfp,rebrac}.py "$OUT/agents/" 2>/dev/null

# 3. runner scripts + the editable experiment plan
cp -f "$SRC"/runner/*.sh "$SRC"/runner/*.py "$SRC"/runner/experiments.tsv "$OUT/runner/" 2>/dev/null

# 4. training stdout, tqdm progress stripped (keep last 3 progress lines)
for f in "$SRC"/exp_logs/*.log; do
  [ -f "$f" ] || continue
  n=$(basename "$f")
  { grep -vE '^\s*[0-9]+%\|| it/s\]|\[A' "$f" | head -400
    echo "... [progress lines stripped] ..."
    grep -oE 'offline: +100%.*' "$f" | tail -3
  } > "$OUT/logs/$n" 2>/dev/null
done

# 5. diffs of the two driver files we modified
( cd "$SRC" && git diff -- main.py > "$OUT/patches/main.py.diff" 2>/dev/null )
( cd "$SRC" && git diff -- log_utils.py > "$OUT/patches/log_utils.py.diff" 2>/dev/null )

# 6. regenerate the summary
SUMMARY_OUT="$OUT/SUMMARY.md" /venv/main/bin/python "$SRC/runner/make_summary.py"

du -sh "$OUT"
/venv/main/bin/python - "$REPO" "$OUT" <<'PY'
import os, sys
os.environ.setdefault("HF_HOME", "/workspace/.hf_home")
from huggingface_hub import HfApi
repo, folder = sys.argv[1], sys.argv[2]
api = HfApi()
api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
api.upload_folder(folder_path=folder, repo_id=repo, repo_type="dataset",
                  commit_message="sync antmaze-giant anq_rfs results")
print(f"uploaded -> https://huggingface.co/datasets/{repo}  (private)")
PY
