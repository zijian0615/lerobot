#!/usr/bin/env bash
# FANUC Cosmos Edge policy server. Does not start the Cosmos Nano reasoner.
# Refuses port 8000 and any SO-101 checkpoint. Does not install cosmos-framework.
set -euo pipefail

PORT=8001
ROOT="$(cd "$(dirname "$0")" && pwd)"
STATS="$ROOT/fanuc_lerobot_stats.json"
VIEW="Two cameras: a fixed overhead camera looking down at the printed table, and a wrist camera on the gripper looking at the fingers."

if [[ "$PORT" == "8000" ]]; then
  echo "refusing port 8000: that is the Cosmos Nano reasoner" >&2
  exit 2
fi

CHECKPOINT="${1:-}"
if [[ -z "$CHECKPOINT" ]]; then
  echo "usage: $0 CHECKPOINT_DIR" >&2
  echo "Pass a FANUC post-train. SO-101 checkpoints are refused." >&2
  exit 2
fi

folded="$(printf '%s' "$CHECKPOINT" | tr '[:upper:]' '[:lower:]')"
case "$folded" in
  *so101*|*cosmos_edge_policy_so101*)
    echo "refusing SO-101 checkpoint: $CHECKPOINT" >&2
    echo "Those weights are domain 22 / 6-D. FANUC is domain 23 / 7-D." >&2
    exit 2
    ;;
esac

if [[ ! -f "$STATS" ]]; then
  echo "missing $STATS" >&2
  exit 2
fi

PY="${COSMOS_FRAMEWORK_PYTHON:-$HOME/cosmos-framework/.venv/bin/python}"
if [[ ! -x "$PY" ]]; then
  echo "cosmos-framework is not installed at $PY" >&2
  echo "This GB10 has Python 3.12 only. CUDA groups in cosmos-framework need Python 3.13." >&2
  echo "Policy port would be $PORT. Do not reuse the reasoner on 8000." >&2
  echo "Not starting a server, and not downloading the training stack." >&2
  exit 3
fi

if ! "$PY" -c 'import cosmos_framework' >/dev/null 2>&1; then
  echo "$PY cannot import cosmos_framework" >&2
  exit 3
fi

help_text="$("$PY" -m cosmos_framework.scripts.action_policy_server_robolab --help 2>&1 || true)"
if ! grep -q -- '--domain-name' <<<"$help_text"; then
  echo "installed action_policy_server_robolab has no --domain-name." >&2
  echo "Upstream cosmos-framework would start a DROID server and return the wrong actions." >&2
  echo "FANUC flags are not in that build. Not starting it." >&2
  exit 3
fi

exec "$PY" -m cosmos_framework.scripts.action_policy_server_robolab \
  --checkpoint-path "$CHECKPOINT" \
  --port "$PORT" \
  --domain-name fanuc \
  --action-dim 7 \
  --arm-joint-dim 6 \
  --action-space joint_pos \
  --conditioning-fps 30 \
  --no-flip-gripper \
  --action-normalization minmax \
  --normalizer-stats-path "$STATS" \
  --view-description "$VIEW" \
  --no-guardrails
