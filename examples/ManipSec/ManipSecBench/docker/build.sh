#!/usr/bin/env bash
# Build the sandbox images.
#   ./build.sh                 manipsec-agent:latest (shared by every task),
#                              manipsec-sim-base:latest (Isaac Sim + cell + harness),
#                              manipsec-sim-<task>:latest for each task in TASKS
#   TASKS=sorting ./build.sh   only that task's simulator image (base rebuilt if changed)
#   INSTALL_ISAAC=0 SIM_TAG_SUFFIX=kinematic ./build.sh
#                              simulators without Isaac Sim (kinematic backend only)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
BENCH="$(dirname "$HERE")"                       # examples/ManipSec/ManipSecBench
REPO="$(cd "$BENCH/../../.." && pwd)"
SUFFIX="${SIM_TAG_SUFFIX:-latest}"
AGENT_TAG="${AGENT_TAG:-manipsec-agent:latest}"
BASE_TAG="manipsec-sim-base:$SUFFIX"
TASKS="${TASKS:-loto sorting assembly}"
IDS=(--build-arg "UID=$(id -u)" --build-arg "GID=$(id -g)")

docker build -f "$HERE/Dockerfile.agent" "${IDS[@]}" \
    --build-arg "CLAUDE_CODE_VERSION=${CLAUDE_CODE_VERSION:-latest}" \
    --build-context api="$BENCH/Task_suite/loto/agent" \
    -t "$AGENT_TAG" "$HERE"

docker build -f "$HERE/Dockerfile.sim-base" "${IDS[@]}" \
    --build-arg "INSTALL_ISAAC=${INSTALL_ISAAC:-1}" \
    --build-context suite="$BENCH/Task_suite" \
    --build-context fanuc="$REPO/fanuc_er4ia_sim" \
    -t "$BASE_TAG" "$HERE"

for task in $TASKS; do
    docker build -f "$HERE/Dockerfile.sim-task" "${IDS[@]}" \
        --build-arg "BASE=$BASE_TAG" --build-arg "TASK=$task" \
        --build-context suite="$BENCH/Task_suite" \
        -t "manipsec-sim-$task:$SUFFIX" "$HERE"
done
