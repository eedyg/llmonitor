#!/usr/bin/env bash
# Start the monitor (UI + gateway), or use a subcommand.
#   ./run.sh                         # config.yaml
#   ./run.sh --verbose               # debug logging
#   ./run.sh engine --engine vllm --profile mtp --ctx 128k   # print the launch plan
#   ./run.sh mcp                     # MCP over stdio
#   MONITOR_UI_PORT=9010 ./run.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

PY="${PY:-../.venv/bin/python}"
if [[ ! -x "${PY}" ]]; then
    PY="$(command -v python3)"
fi

case "${1:-}" in
    engine) shift; exec "${PY}" -m monitor.launcher "$@" ;;
    mcp)    shift; exec "${PY}" -m monitor.mcp_server "$@" ;;
esac

exec "${PY}" -m monitor --config "${MONITOR_CONFIG:-config.yaml}" "$@"
