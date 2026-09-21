#!/usr/bin/env bash
# Creates ../.venv-twin with mujoco/numpy/pytest and links libpython so that `mjpython` (the macOS GUI launcher)
# works. Needed because uv's standalone Python does not ship libpython next to the venv.
set -euo pipefail
cd "$(dirname "$0")/.."
uv venv .venv-twin --python 3.12 -q
uv pip install -q --python .venv-twin/bin/python mujoco numpy pytest
BASE=$(.venv-twin/bin/python -c "import sys; print(sys.base_prefix)")
ln -sf "$BASE/lib/libpython3.12.dylib" .venv-twin/lib/libpython3.12.dylib
echo "ready: .venv-twin/bin/mjpython twin/twin.py --source demo"
