#!/usr/bin/env bash
# After r64 ep8 continue: merge adapter and full Model Utility eval.
set -euo pipefail
ROOT=/opt/data/private/DQB
PY=/root/miniconda3/envs/unlearning/bin/python
CFG="$ROOT/configs/full_tofu_lora_r64_ep8.yaml"
ADAPTER="$ROOT/outputs/full_tofu_r64_ep8/adapter"
CKPT="$ROOT/outputs/full_tofu_r64_ep8/run/checkpoint-125"
MERGED="$ROOT/outputs/full_tofu_r64_ep8/merged"
LOG="$ROOT/logs/full_tofu_r64_ep8"
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$LOG"

SRC="$ADAPTER"
if [[ -d "$CKPT" ]]; then
  SRC="$CKPT"
fi

echo "=== merge ep8 from $SRC ==="
"$PY" "$ROOT/scripts/merge_full_tofu_lora.py" \
  --config "$CFG" \
  --adapter_dir "$SRC" \
  --output_dir "$MERGED" \
  2>&1 | tee "$LOG/merge.log"

echo "=== full utility eval ep8 ==="
"$PY" -u "$ROOT/scripts/light_tofu_eval.py" \
  --model "$MERGED" \
  --run_id "full_tofu_r64_ep8" \
  --config "$CFG" \
  --full_utility \
  --batch_size 4 \
  2>&1 | tee "$LOG/full_eval.log"

"$PY" - <<'PY'
import json
from pathlib import Path
root = Path("/opt/data/private/DQB/results/model_utility_optimization/evals")
print(f"{'run':28} {'MU':>8} {'R-ROUGE':>8} {'R-Prob':>8} {'RA-ROUGE':>9} {'RW-ROUGE':>9}")
for name in ("full_tofu_r64_ep5", "full_tofu_r64_ep7", "full_tofu_r64_ep8"):
    p = root / name / "summary.json"
    if not p.exists():
        print(f"{name:28} MISSING")
        continue
    mu = json.loads(p.read_text())["model_utility"]
    print(
        f"{name:28} {mu['Model Utility']:8.4f} {mu['ROUGE Retain']:8.4f} "
        f"{mu['Prob. Retain']:8.4f} {mu['ROUGE Real Authors']:9.4f} {mu['ROUGE Real World']:9.4f}"
    )
PY
