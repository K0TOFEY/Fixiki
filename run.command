#!/bin/bash
set -e

cd "$(dirname "$0")"
PORT=5001

PYTHON=""
if [ -x ".venv311/bin/python" ] && ".venv311/bin/python" -c "import pandas" >/dev/null 2>&1; then
    PYTHON=".venv311/bin/python"
elif [ -x ".venv/bin/python" ] && ".venv/bin/python" -c "import pandas" >/dev/null 2>&1; then
    PYTHON=".venv/bin/python"
elif command -v python3 >/dev/null 2>&1 && python3 -c "import pandas" >/dev/null 2>&1; then
    PYTHON="$(command -v python3)"
else
    echo "No usable Python environment with pandas was found."
    echo "Create a stable Python 3.11 environment and install the project dependen=cies:"
    echo "  python3.11 -m venv .venv311"
    echo "  .venv311/bin/python -m pip install -r requirements.txt"
    exit 1
fi

if ! command -v lsof >/dev/null 2>&1; then
    echo "lsof is required to check whether port $PORT is in use."
    exit 1
fi

PIDS="$(lsof -tiTCP:"$PORT" -sTCP:LISTEN || true)"
if [ -n "$PIDS" ]; then
    for pid in $PIDS; do
        PROCESS="$(ps -p "$pid" -o command= || true)"
        case "$PROCESS" in
            *dashboard.py*) ;;
            *)
                echo "Port $PORT is occupied by another application (PID $pid)."
                echo "It was not stopped. Choose another port or stop it manually."
                exit 1
                ;;
        esac
    done

    echo "Stopping existing dashboard process(es) on port $PORT: $PIDS"
    kill $PIDS

    for attempt in 1 2 3 4 5; do
        PIDS="$(lsof -tiTCP:"$PORT" -sTCP:LISTEN || true)"
        [ -z "$PIDS" ] && break
        sleep 1
    done

    if [ -n "$PIDS" ]; then
        echo "Forcibly stopping process(es) still using port $PORT: $PIDS"
        kill -9 $PIDS
        sleep 1
    fi

    if lsof -tiTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
        echo "Could not free port $PORT. Check permissions and try again."
        exit 1
    fi
fi

export PORT
exec "$PYTHON" dashboard.py
