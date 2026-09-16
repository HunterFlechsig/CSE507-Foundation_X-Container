#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIRECTORY=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIRECTORY/../common.sh"
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <seed>" >&2
  exit 2
fi
fx_launch_experiment "candidptx_seg" "$1" true
