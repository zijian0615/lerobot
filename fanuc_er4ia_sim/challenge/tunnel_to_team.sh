#!/usr/bin/env bash
# Reverse SSH tunnel: the team's cloud machine reaches the cell server at its own 127.0.0.1:<port>.
#
#   fanuc_er4ia_sim/challenge/tunnel_to_team.sh shadeform@216.81.248.169 [port] [key]
#
# Outbound only: nothing on the lab side listens for the internet. On the team machine the forward binds to
# 127.0.0.1, so only that machine can use it; the team API still needs the run token. The key should be
# installed there restricted to this one forward (see challenge/README.md). Reconnects until Ctrl+C.
set -u
DEST=${1:?usage: tunnel_to_team.sh user@host [port] [key]}
PORT=${2:-9099}
KEY=${3:-$HOME/.ssh/cell_tunnel_brev}
CELL=${CELL:-127.0.0.1}  # the server runs on this machine; its LAN IP changes with DHCP
echo "[tunnel] $DEST 127.0.0.1:$PORT -> cell $CELL:$PORT (Ctrl+C to stop)"
while true; do
  ssh -i "$KEY" -N -T -o IdentitiesOnly=yes -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 \
      -o ServerAliveCountMax=3 -o StrictHostKeyChecking=accept-new \
      -R "127.0.0.1:$PORT:$CELL:$PORT" "$DEST"
  echo "[tunnel] $(date +%T) disconnected (exit $?); retrying in 5 s"
  sleep 5
done
