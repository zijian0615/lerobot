#!/usr/bin/env bash
# Record the single fixed screw with a small per-episode nudge, pack the episodes
# that ended in the bin, make the 224x224 copy, and push both to the Hub.
#
#   tmux new-session -d -s screw_single "bash fanuc_er4ia_sim/run_screw_single.sh 2>&1 | tee fanuc_er4ia_sim/data/screw_single.log"
set -euo pipefail
cd "$(dirname "$0")/.."

EPISODES=${EPISODES:-50}
JITTER_XY=${JITTER_XY:-0.01}      # metres
JITTER_YAW=${JITTER_YAW:-15}      # degrees
NAME=${NAME:-screw_v4}                         # episode dir under fanuc_er4ia_sim/data
REPO=${REPO:-zijian2022/fanuc_er4ia_screw_single}
EXTRA=${EXTRA:-}                              # extra recorder flags, e.g. "--bg-plates data/bg_plates"
SRC=fanuc_er4ia_sim/data/$NAME
FULL=fanuc_er4ia_sim/data/${REPO#*/}
SMALL=${FULL}_224

echo "[run] record $EPISODES episodes, jitter +/-${JITTER_XY} m, +/-${JITTER_YAW} deg -> $SRC"
(cd fanuc_er4ia_sim && OMNI_KIT_ACCEPT_EULA=YES LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1 \
  ~/isaacsim-venv/bin/python record_screw_episode.py --out "data/$NAME" --single --screw 0 \
  --jitter-xy "$JITTER_XY" --jitter-yaw "$JITTER_YAW" --episodes "$EPISODES" --max-speed 50 $EXTRA) \
  2>&1 | grep --line-buffered -E "\[record\]|Traceback|Error:" || true

if [ -n "${RESIZE:-}" ]; then           # e.g. RESIZE="480 640": one dataset at that size, no 224 copy
  echo "[run] pack at $RESIZE -> $FULL"
  uv run python fanuc_er4ia_sim/pack_lerobot_episode.py --src "$SRC" --root "$FULL" --repo-id "$REPO" --resize $RESIZE --push
  echo "[run] done: https://huggingface.co/datasets/$REPO"
else
  echo "[run] pack -> $FULL"
  uv run python fanuc_er4ia_sim/pack_lerobot_episode.py --src "$SRC" --root "$FULL" --repo-id "$REPO" --push
  echo "[run] 224x224 -> $SMALL"
  uv run python fanuc_er4ia_sim/resize_dataset.py --src-root "$FULL" --src-repo "$REPO" \
    --dst-root "$SMALL" --dst-repo "${REPO}_224" --push
  echo "[run] done: https://huggingface.co/datasets/$REPO and ${REPO}_224"
fi
