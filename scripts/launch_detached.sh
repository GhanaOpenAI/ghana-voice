#!/usr/bin/env bash
# Detached launcher for long training or encoding jobs.
# Runs in its own session, decoupled from the terminal.
#
# Usage:
#   bash scripts/launch_detached.sh <log_name> <command...>
#
# Example:
#   bash scripts/launch_detached.sh train_twi \
#     ghana-voice-trainer train --data-dir path/to/data --run-name my-twi-run

set -euo pipefail

LOG_NAME="${1:?usage: launch_detached.sh <log-name> <command...>}"
shift

mkdir -p logs
export PYTHONUNBUFFERED=1

setsid nohup "$@" > "logs/${LOG_NAME}.log" 2>&1 < /dev/null &
PID=$!
disown || true
echo "Launched (detached): PID=$PID  log=logs/${LOG_NAME}.log"
echo "To follow log: tail -f logs/${LOG_NAME}.log"
