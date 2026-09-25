#!/usr/bin/env bash
# Creates ../.venv-twin with mujoco/numpy/pytest/pillow.
# On macOS, links libpython so that `mjpython` (the Cocoa GUI launcher) works.
# On Linux, `mjpython` is not shipped; we alias it to the venv python.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ ! -x .venv-twin/bin/python ]]; then
  uv venv .venv-twin --python 3.12 -q
fi
uv pip install -q --python .venv-twin/bin/python mujoco numpy pytest pillow
if [[ "$(uname -s)" == Darwin ]]; then
  BASE=$(.venv-twin/bin/python -c "import sys; print(sys.base_prefix)")
  ln -sf "$BASE/lib/libpython3.12.dylib" .venv-twin/lib/libpython3.12.dylib
else
  ln -sfn python .venv-twin/bin/mjpython
fi
echo "ready: .venv-twin/bin/mjpython twin/twin.py --source demo"
