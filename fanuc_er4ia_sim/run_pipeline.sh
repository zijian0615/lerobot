#!/usr/bin/env bash
# Perception-driven pick-and-place: Cosmos3-Nano sees the scene, plans the
# grasps, the arm executes them, then the model verifies the result.
#
#   ./run_pipeline.sh [run_dir]
#
# The three stages use three different interpreters, which is why this is a
# shell driver rather than one Python program:
#   Isaac Sim python  -> simulation and rendering
#   cosmos venv       -> Cosmos3-Nano (torch 2.9 / transformers 5.8)
#   project .venv     -> numpy/opencv geometry and video encoding
set -euo pipefail
RUN="${1:-run}"
HERE="$(cd "$(dirname "$0")" && pwd)"
ISAAC="${ISAAC:-/isaacsim}"
COSMOS="${COSMOS:-/home/driley/cosmos-predict2.5/.venv/bin/python}"
VENV="$HERE/.venv/bin/python"

echo "=== 1/6  capture world camera ==============================="
"$ISAAC/python.sh" "$HERE/capture_scene.py" --out "$RUN"

echo "=== 2/6  Cosmos3-Nano scene understanding =================="
"$COSMOS" "$HERE/perceive.py" --run "$RUN"

echo "=== 3/6  localise + plan grasps ============================"
"$VENV" "$HERE/plan_grasps.py" --run "$RUN" --categories all

echo "=== 4/6  execute ==========================================="
"$ISAAC/python.sh" "$HERE/execute_plan.py" --run "$RUN" --camera table --inspect-bin
LAST=$(ls "$RUN/frames" | tail -1)
cp "$RUN/frames/$LAST" "$RUN/final_frame.png"
"$VENV" "$HERE/encode_video.py" --frames "$RUN/frames" --out "$RUN/pipeline.mp4"

echo "=== 5/6  Cosmos3-Nano verifies the result =================="
"$COSMOS" "$HERE/perceive.py" --run "$RUN" --image "$RUN/final_frame.png" --verify

echo "=== 6/6  wrist-camera bin inventory ========================"
if [ -f "$RUN/bin_view.png" ]; then
    "$COSMOS" "$HERE/perceive.py" --run "$RUN" --image "$RUN/bin_view.png" --inventory
else
    echo "  (no bin_view.png -- inspection pose was unreachable)"
fi

echo
echo "artifacts in $RUN/:"
echo "  world_camera.png        what the model was shown"
echo "  detections.json         what it reported"
echo "  detections_overlay.png  raw vs refined vs ground truth"
echo "  plan.json               the grasp plan"
echo "  pipeline.mp4            execution"
echo "  verify.json             post-hoc check"
echo "  bin_view.png            wrist camera looking into the bin"
echo "  inventory.json          what the model sees in the bin"
