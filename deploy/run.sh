#!/bin/bash
set -euo pipefail
export LIGHTSHOW_HOME="$(cd "$(dirname "$0")" && pwd)"
export LIGHTSHOW_ENV_FILE="$LIGHTSHOW_HOME/config.env"
export LIGHTSHOW_STATE_DIR="$LIGHTSHOW_HOME/state"
export TZ=Asia/Shanghai
umask 077
exec "$LIGHTSHOW_HOME/.venv/bin/python" "$LIGHTSHOW_HOME/current/app/cloud_runner.py" "$@"
