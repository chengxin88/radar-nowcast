#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export LIGHTSHOW_HOME="${LIGHTSHOW_HOME:-$PROJECT_ROOT}"
export LIGHTSHOW_ENV_FILE="${LIGHTSHOW_ENV_FILE:-$LIGHTSHOW_HOME/config.env}"
export LIGHTSHOW_STATE_DIR="${LIGHTSHOW_STATE_DIR:-$LIGHTSHOW_HOME/state}"
export TZ=Asia/Shanghai
umask 077
exec "${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}" "$PROJECT_ROOT/app/cloud_runner.py" "$@"
