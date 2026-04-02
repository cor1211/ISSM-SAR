#!/usr/bin/env bash
set -euo pipefail

if [[ $# -eq 0 ]]; then
  set -- pipeline --help
fi

case "$1" in
  pipeline)
    shift
    default_config="/app/config/pipeline_config_stac_runtime.yaml"
    has_config=0
    for arg in "$@"; do
      if [[ "$arg" == "--config" ]]; then
        has_config=1
        break
      fi
    done
    if [[ ${has_config} -eq 0 ]]; then
      set -- --config "$default_config" "$@"
    fi
    exec python /app/sar_pipeline.py "$@"
    ;;
  workflow)
    shift
    default_config="/app/config/pipeline_config_stac_runtime.yaml"
    has_config=0
    for arg in "$@"; do
      if [[ "$arg" == "--config" ]]; then
        has_config=1
        break
      fi
    done
    if [[ $# -eq 0 ]]; then
      set -- --help
    elif [[ ${has_config} -eq 0 ]]; then
      set -- --config "$default_config" "$@"
    fi
    exec python /app/sr_workflow.py "$@"
    ;;
  publish)
    shift
    exec python /app/tools/publish_sr_outputs.py "$@"
    ;;
  bash|shell)
    shift
    exec /bin/bash "$@"
    ;;
  python)
    shift
    exec python "$@"
    ;;
  sh)
    shift
    exec /bin/sh "$@"
    ;;
  *)
    exec "$@"
    ;;
esac
