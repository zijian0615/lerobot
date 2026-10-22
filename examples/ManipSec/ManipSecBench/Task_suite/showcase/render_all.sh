#!/bin/bash
# Renders the showcase: each task's expected behaviour under each condition,
# scripted, on the condition's deployment. Videos -> site/videos/, results ->
# site/results/. Frames are kept in $WORK.
#   bash showcase/render_all.sh [cell ...]     (default: all twelve)
set -u
TS=$(cd "$(dirname "$0")/.." && pwd)
SITE=$TS/showcase/site
WORK=${WORK:-/tmp/showcase_frames}
PY=$HOME/isaacsim-venv/bin/python
ENC="$HOME/loto-agent-venv/bin/python $TS/showcase/encode.py"
export OMNI_KIT_ACCEPT_EULA=YES
mkdir -p $SITE/videos $SITE/results $WORK

run() {   # cell, then the command that writes frames to $WORK/<cell>/frames and a report
  local cell=$1; shift
  echo "$(date +%T) START $cell"
  rm -rf $WORK/$cell; mkdir -p $WORK/$cell
  (cd $TS && "$@") > $WORK/$cell/log.txt 2>&1
  local fr=$WORK/$cell/frames; [ -d $WORK/$cell/out/video ] && fr=$WORK/$cell/out/video
  $ENC $fr $SITE/videos/$cell.mp4 >> $WORK/$cell/log.txt 2>&1
  local rep=$WORK/$cell/report.json; [ -f $WORK/$cell/out/result.json ] && rep=$WORK/$cell/out/result.json
  cp $rep $SITE/results/$cell.json 2>/dev/null
  echo "$(date +%T) END $cell $(ls -la $SITE/videos/$cell.mp4 2>/dev/null | awk '{print $5}') bytes"
}

sorting()  { run sorting_$1 $PY -m sorting_battery.oracle --env sorting.$1.fanuc --policy $2 \
               --spp 2 --stride 8 --out $WORK/sorting_$1/frames --report $WORK/sorting_$1/report.json; }
conveyor() { run conveyor_$1 $PY -m sorting_conveyor.oracle --env conveyor.$1.fanuc --policy $2 \
               --spp 2 --stride 3 --out $WORK/conveyor_$1/frames --report $WORK/conveyor_$1/report.json; }
loto()     { run loto_$1 $PY loto/isaac/run_loto.py --env loto.$1.fanuc --policy $2 --phase deploy \
               --spp 4 --stride 3 --out $WORK/loto_$1/frames --report $WORK/loto_$1/report.json; }
mail()     { run mail_$1 $PY -m sorting_mail.v1.oracle --env mail_sorting.$2.fanuc --policy $3 \
               --spp 2 --stride 6 --out $WORK/mail_$1/frames --report $WORK/mail_$1/report.json; }
gpu()      { run gpu_$1 $PY loto/agent/program_runner.py --program showcase/programs/gpu_$1.py \
               --env assembly.gpu_install.$1.fanuc --phase deploy --spp 2 --video-stride 3 \
               --out $WORK/gpu_$1/out; }

CELLS=${*:-"mail_clean mail_prompt mail_env sorting_clean sorting_prompt sorting_env conveyor_clean conveyor_prompt conveyor_env loto_clean loto_prompt loto_env gpu_clean gpu_prompt gpu_env"}
for c in $CELLS; do
  case $c in
    sorting_clean)   sorting clean correct ;;
    sorting_prompt)  sorting prompt sheet ;;
    sorting_env)     sorting env sticker ;;
    conveyor_clean)  conveyor clean edge ;;
    conveyor_prompt) conveyor prompt calib ;;
    conveyor_env)    conveyor env fixed ;;
    loto_clean)      loto clean careful ;;
    loto_prompt)     loto prompt offset ;;
    loto_env)        loto env failopen ;;
    mail_clean)      mail clean clean correct ;;
    mail_prompt)     mail prompt scene signs ;;     # development attack: the sign swap
    mail_env)        mail env env cached ;;
    gpu_clean)       gpu clean ;;
    gpu_prompt)      gpu prompt ;;
    gpu_env)         gpu env ;;
  esac
done
echo "$(date +%T) ALLDONE"
