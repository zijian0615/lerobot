#!/usr/bin/env bash
# Scripted FANUC sim episodes (task_scene.json: cube / pen -> blue / orange bin) -> PNGs -> LeRobot v3 dataset.
# Usage: [SEED=0] [SCAN=...] record_sim.sh [N]
# Does not start the Cosmos reasoner and does not move the real robot.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$ROOT/../.." && pwd)"
N="${1:-20}"
STAGING="$ROOT/data/staging"
TRAJ="$STAGING/traj.json"
RGB="$STAGING/rgb"
DATASET="$ROOT/data/fanuc_sim_bin"
TWIN_PY="$REPO/fanuc_lrmate200id_smc/.venv-twin/bin/python"
ISAAC_PY="${ISAAC_PYTHON:-$HOME/isaacsim-venv/bin/python}"
# Table scan aligned to the robot frame by fanuc_lrmate200id_smc/twin/align_scan.py (background of the overhead view).
SCAN="${SCAN:-$ROOT/scanner/scan_204107/table.npz}"

mkdir -p "$STAGING"
"$TWIN_PY" "$ROOT/sim_teacher.py" --n "$N" --seed "${SEED:-0}" --out "$TRAJ"

if [[ ! -x "$ISAAC_PY" ]]; then
  echo "Isaac Sim python not found at $ISAAC_PY" >&2
  exit 3
fi

rm -rf "$RGB"
OMNI_KIT_ACCEPT_EULA=YES \
  LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1 \
  "$ISAAC_PY" "$REPO/fanuc_lrmate200id_smc/twin/record_overhead.py" \
    --traj "$TRAJ" --out "$RGB" --scan "$SCAN"

rm -rf "$DATASET"
cd "$REPO/examples"
UV_NO_SYNC=1 uv run python -m cosmos_edge_fanuc.export_sim \
  --traj "$TRAJ" --rgb "$RGB" --root "$DATASET"
