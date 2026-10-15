#!/bin/bash
set -euo pipefail
exec "${LIGHTSHOW_HOME:-/home/gbamwf/lightshow-duty}/run.sh" run --mode manual "$@"
