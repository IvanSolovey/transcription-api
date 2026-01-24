#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$PROJECT_ROOT/venv"
PYTHON_BIN="$VENV_DIR/bin/python3"
UVICORN_BIN="$VENV_DIR/bin/uvicorn"
LOG_DIR="$PROJECT_ROOT/logs"
LOG_FILE="$LOG_DIR/server.log"
PID_FILE="$PROJECT_ROOT/data/server.pid"
ENV_FILE="$PROJECT_ROOT/.env"
REQUIREMENTS_FILE="$PROJECT_ROOT/requirements.txt"
DOWNLOAD_MODELS_SCRIPT="$PROJECT_ROOT/download_models.py"

SYSTEM_PACKAGES_DEBIAN=(ffmpeg libsndfile1 libsndfile1-dev)
SYSTEM_PACKAGES_RHEL=(ffmpeg libsndfile)

command_exists() {
    command -v "$1" >/dev/null 2>&1
}

print_section() {
    echo "\n========== $1 =========="
}

install_system_packages() {
    print_section "Installing system packages"
    if command_exists apt-get; then
        sudo apt-get update
        sudo apt-get install -y "${SYSTEM_PACKAGES_DEBIAN[@]}"
    elif command_exists dnf; then
        sudo dnf install -y "${SYSTEM_PACKAGES_RHEL[@]}"
    elif command_exists yum; then
        sudo yum install -y "${SYSTEM_PACKAGES_RHEL[@]}"
    elif command_exists brew; then
        brew install ffmpeg libsndfile
    else
        echo "WARN: Package manager not detected. Install ffmpeg and libsndfile manually."
    fi
}

ensure_directories() {
    mkdir -p "$LOG_DIR" "$PROJECT_ROOT/data" "$PROJECT_ROOT/temp"
}

create_venv() {
    if [[ -x "$PYTHON_BIN" ]]; then
        return
    fi
    print_section "Creating virtual environment"
    python3 -m venv "$VENV_DIR"
}

install_python_dependencies() {
    print_section "Installing Python dependencies"
    "$PYTHON_BIN" -m pip install --upgrade pip wheel
    "$PYTHON_BIN" -m pip install -r "$REQUIREMENTS_FILE"
}

run_db_migrations() {
    print_section "Initializing database"
    "$PYTHON_BIN" -m app.db.init_db
}

download_models() {
    if [[ -f "$DOWNLOAD_MODELS_SCRIPT" ]]; then
        print_section "Downloading speech models"
        "$PYTHON_BIN" "$DOWNLOAD_MODELS_SCRIPT"
    else
        echo "INFO: download_models.py not found, skipping model download"
    fi
}

source_env_file() {
    if [[ -f "$ENV_FILE" ]]; then
        set -a
        # shellcheck disable=SC1090
        source "$ENV_FILE"
        set +a
    fi
}

start_server() {
    ensure_directories
    source_env_file
    local port="${PORT:-8000}"
    local workers="${UVICORN_WORKERS:-1}"
    local host="${HOST:-0.0.0.0}"

    if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" >/dev/null 2>&1; then
        echo "Server already running with PID $(cat "$PID_FILE"). Use restart or stop first."
        exit 0
    fi

    print_section "Starting FastAPI server"
    nohup "$UVICORN_BIN" main:app \
        --host "$host" \
        --port "$port" \
        --workers "$workers" \
        --log-config "$PROJECT_ROOT/logging.conf" \
        --access-log \
        --timeout-keep-alive 30 \
        --limit-concurrency 10 \
        > "$LOG_FILE" 2>&1 &

    echo $! > "$PID_FILE"
    echo "Server started on http://$host:$port (PID $(cat "$PID_FILE"))"
    echo "Logs: $LOG_FILE"
}

stop_server() {
    if [[ -f "$PID_FILE" ]]; then
        local pid
        pid=$(cat "$PID_FILE")
        if kill -0 "$pid" >/dev/null 2>&1; then
            print_section "Stopping server (PID $pid)"
            kill "$pid"
            rm -f "$PID_FILE"
            return
        fi
    fi

    echo "Server is not running."
}

restart_server() {
    stop_server || true
    sleep 1
    start_server
}

tail_logs() {
    ensure_directories
    print_section "Streaming logs"
    tail -f "$LOG_FILE"
}

show_status() {
    if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" >/dev/null 2>&1; then
        echo "Server is running (PID $(cat "$PID_FILE"))"
    else
        echo "Server is stopped"
    fi
}

usage() {
    cat <<'EOF'
Usage: scripts/server_manager.sh <command>

Commands:
  deploy   Install system deps, Python packages, initialize DB, download models
  start    Launch uvicorn with background logging
  stop     Stop running uvicorn instance
  restart  Restart the server
  status   Show server status
  logs     Tail the server log file
EOF
}

main() {
    if [[ $# -lt 1 ]]; then
        usage
        exit 1
    fi

    local action="$1"
    shift || true

    case "$action" in
        deploy)
            install_system_packages
            ensure_directories
            create_venv
            install_python_dependencies
            run_db_migrations
            download_models
            echo "Deployment completed successfully. Use '$0 start' to launch the API."
            ;;
        start)
            create_venv
            install_python_dependencies
            run_db_migrations
            start_server
            ;;
        stop)
            stop_server
            ;;
        restart)
            restart_server
            ;;
        logs)
            tail_logs
            ;;
        status)
            show_status
            ;;
        *)
            usage
            exit 1
            ;;
    esac
}

main "$@"
