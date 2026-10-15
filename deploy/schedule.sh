#!/bin/bash
set -euo pipefail
export LIGHTSHOW_HOME="$(cd "$(dirname "$0")" && pwd)"
umask 077
exec "$LIGHTSHOW_HOME/.venv/bin/python" "$LIGHTSHOW_HOME/current/app/manage_schedule.py" "$@"
