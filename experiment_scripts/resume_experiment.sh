#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIRECTORY=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIRECTORY/common.sh"

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <experiment> <seed> [extra args...]" >&2
  echo "Experiments:" >&2
  fx_known_experiments >&2
  exit 2
fi

experiment_name="$1"
seed="$2"
shift 2

fx_launch_experiment "$experiment_name" "$seed" true "$@"
