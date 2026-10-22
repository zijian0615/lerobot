#!/usr/bin/env bash
# Single screw at a random position (and yaw) every episode, bin fixed, for fine-tuning on many layouts.
# Records in batches; after each batch every episode so far is packed at 480x640 and pushed, so a stop
# loses at most the current batch.
#
#   tmux new-session -d -s screw_randpos "bash fanuc_er4ia_sim/run_screw_randpos.sh 2>&1 | tee fanuc_er4ia_sim/data/screw_randpos.log"
set -euo pipefail
cd "$(dirname "$0")/.."

TOTAL=${TOTAL:-200}
BATCH=${BATCH:-100}
START=${START:-0}                                # first episode index (resume after a stop)
BOX=${BOX:-"0.30 0.55 -0.25 0.22"}               # x0 x1 y0 y1, metres, robot frame
NAME=${NAME:-screw_randpos}
REPO=${REPO:-zijian2022/fanuc_er4ia_screw_randpos}

for ((s = START; s < TOTAL; s += BATCH)); do
  e=$((s + BATCH < TOTAL ? s + BATCH : TOTAL))
  echo "[randpos] episodes $s..$((e - 1)) of $TOTAL, screw box $BOX"
  EPISODES=$e NAME=$NAME REPO=$REPO RESIZE="480 640" \
    EXTRA="--start $s --random-xy $BOX --bg-plates data/bg_plates_v2 --tone data/cam_tone_v2.json" \
    bash fanuc_er4ia_sim/run_screw_single.sh
done
echo "[randpos] all $TOTAL episodes recorded and pushed to $REPO"
