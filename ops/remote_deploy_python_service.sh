#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  remote_deploy_python_service.sh <app_path> <branch> <venv_path> <requirements_path> <service_manager> <service_name_or_command> <healthcheck_url> [post_restart_delay_seconds]

Examples:
  remote_deploy_python_service.sh /home/ubuntu/Stoody/stoody-backend dev /home/ubuntu/Stoody/stoody-backend/venv requirements.txt systemctl stoody-backend http://127.0.0.1:5001/health 5
  remote_deploy_python_service.sh /home/ubuntu/Stoody/stoody-backend dev /home/ubuntu/Stoody/stoody-backend/venv requirements.txt supervisorctl skillbot-development http://127.0.0.1:5001/health 5

Optional environment variables:
  MENTOR_AI_ENABLED=true|false
  MENTOR_AI_PATH=/home/ubuntu/Stoody/stoody-backend/mentorAI
  MENTOR_AI_BASE_PATH=/mentor-ai
  MENTOR_AI_HOST=127.0.0.1
  MENTOR_AI_PORT=3000
  MENTOR_AI_HEALTHCHECK_URL=http://127.0.0.1:3000/mentor-ai/api/health
  NGINX_SITE_PATH=/etc/nginx/sites-available/skillbot_nginx_alb.conf
  STOODY_UPLOAD_ROOT=/var/lib/stoody/uploads
  STOODY_UPLOAD_OWNER=<app owner, defaults to APP_PATH owner>
  STOODY_UPLOAD_GROUP=clamav
  STOODY_CLAMAV_HARDEN_SOCKET=true
  STOODY_CLAMAV_CONF=/etc/clamav/clamd.conf
  STOODY_CLAMAV_SOCKET=/var/run/clamav/clamd.ctl
  STOODY_CLAMAV_GROUP=clamav
  STOODY_CLAMAV_SOCKET_MODE=660
  STOODY_DEPLOY_LOCK_PATH=/tmp/stoody-backend-deploy.lock
  STOODY_DEPLOY_LOCK_TIMEOUT_SECONDS=900
  STOODY_UPLOAD_CLEANUP_ENABLED=true
  STOODY_UPLOAD_CLEANUP_UNIT=stoody-upload-cleanup
  STOODY_UPLOAD_CLEANUP_TIMER=*-*-* 03:20:00
  STOODY_UPLOAD_CLEANUP_RANDOMIZED_DELAY_SEC=15m
EOF
}

APP_PATH="${1:-}"
BRANCH="${2:-}"
VENV_PATH="${3:-}"
REQUIREMENTS_PATH="${4:-requirements.txt}"
SERVICE_MANAGER="${5:-}"
SERVICE_NAME_OR_COMMAND="${6:-}"
HEALTHCHECK_URL="${7:-}"
POST_RESTART_DELAY="${8:-5}"

if [[ -z "$APP_PATH" || -z "$BRANCH" || -z "$SERVICE_MANAGER" || -z "$SERVICE_NAME_OR_COMMAND" || -z "$HEALTHCHECK_URL" ]]; then
  usage
  exit 1
fi

if [[ ! -d "$APP_PATH" ]]; then
  echo "App path does not exist: $APP_PATH" >&2
  exit 1
fi

acquire_deploy_lock() {
  local lock_path="${STOODY_DEPLOY_LOCK_PATH:-/tmp/stoody-backend-deploy.lock}"
  local lock_timeout="${STOODY_DEPLOY_LOCK_TIMEOUT_SECONDS:-900}"

  if ! command -v flock >/dev/null 2>&1; then
    echo "flock is required to serialize deploys on this host" >&2
    exit 1
  fi

  echo "Acquiring deploy lock: $lock_path"
  exec 9>"$lock_path"
  if ! flock -w "$lock_timeout" 9; then
    echo "Timed out waiting ${lock_timeout}s for deploy lock: $lock_path" >&2
    exit 1
  fi
}

is_git_dirty() {
  [[ -n "$(git diff --name-only)" || -n "$(git diff --cached --name-only)" ]]
}

sync_git_repo() {
  local repo_path="$1"
  local repo_branch="$2"

  if [[ ! -d "$repo_path/.git" ]]; then
    echo "Not a git repository: $repo_path" >&2
    exit 1
  fi

  cd "$repo_path"

  if is_git_dirty; then
    echo "Refusing to deploy from a dirty git worktree at $repo_path" >&2
    exit 1
  fi

  echo "Fetching latest code for branch: $repo_branch ($repo_path)"
  git fetch origin "$repo_branch"

  local current_branch
  current_branch="$(git rev-parse --abbrev-ref HEAD || true)"
  if [[ "$current_branch" != "$repo_branch" ]]; then
    if git show-ref --verify --quiet "refs/heads/$repo_branch"; then
      git checkout "$repo_branch"
    else
      git checkout -b "$repo_branch" --track "origin/$repo_branch"
    fi
  fi

  git pull --ff-only origin "$repo_branch"
}

app_owner() {
  local upload_owner="${STOODY_UPLOAD_OWNER:-}"
  if [[ -z "$upload_owner" ]]; then
    upload_owner="$(stat -c '%U' "$APP_PATH")"
  fi
  echo "$upload_owner"
}

set_clamd_config_value() {
  local config_path="$1"
  local key="$2"
  local value="$3"

  if sudo grep -Eq "^[[:space:]]*${key}[[:space:]]+" "$config_path"; then
    sudo sed -i -E "s@^[[:space:]]*${key}[[:space:]].*@${key} ${value}@" "$config_path"
  else
    printf '%s %s\n' "$key" "$value" | sudo tee -a "$config_path" >/dev/null
  fi
}

ensure_clamav_socket_hardening() {
  local enabled="${STOODY_CLAMAV_HARDEN_SOCKET:-true}"
  if [[ "$enabled" != "true" ]]; then
    echo "Skipping ClamAV socket hardening because STOODY_CLAMAV_HARDEN_SOCKET=$enabled"
    return
  fi

  local clamav_config="${STOODY_CLAMAV_CONF:-/etc/clamav/clamd.conf}"
  local clamav_socket="${STOODY_CLAMAV_SOCKET:-/var/run/clamav/clamd.ctl}"
  local clamav_group="${STOODY_CLAMAV_GROUP:-${STOODY_UPLOAD_GROUP:-clamav}}"
  local socket_mode="${STOODY_CLAMAV_SOCKET_MODE:-660}"
  local upload_owner
  upload_owner="$(app_owner)"

  if [[ ! -f "$clamav_config" ]]; then
    echo "ClamAV config not found: $clamav_config" >&2
    exit 1
  fi

  if ! getent passwd "$upload_owner" >/dev/null; then
    echo "Backend service user does not exist: $upload_owner" >&2
    exit 1
  fi

  if ! getent group "$clamav_group" >/dev/null; then
    echo "ClamAV group does not exist: $clamav_group" >&2
    exit 1
  fi

  echo "Hardening ClamAV socket $clamav_socket with group $clamav_group and mode $socket_mode"
  set_clamd_config_value "$clamav_config" "LocalSocket" "$clamav_socket"
  set_clamd_config_value "$clamav_config" "LocalSocketMode" "$socket_mode"
  set_clamd_config_value "$clamav_config" "LocalSocketGroup" "$clamav_group"

  if systemctl cat clamav-daemon.socket >/dev/null 2>&1; then
    echo "Configuring systemd ClamAV socket activation permissions"
    local systemd_socket_mode="$socket_mode"
    if [[ "$systemd_socket_mode" != 0* ]]; then
      systemd_socket_mode="0$systemd_socket_mode"
    fi
    sudo mkdir -p /etc/systemd/system/clamav-daemon.socket.d
    printf '[Socket]\nSocketMode=%s\nSocketUser=clamav\nSocketGroup=%s\n' "$systemd_socket_mode" "$clamav_group" \
      | sudo tee /etc/systemd/system/clamav-daemon.socket.d/upload-security.conf >/dev/null
    sudo systemctl daemon-reload
    sudo systemctl stop clamav-daemon.service clamav-daemon.socket
    sudo systemctl start clamav-daemon.socket
  fi

  if ! id -nG "$upload_owner" | tr ' ' '\n' | grep -Fx "$clamav_group" >/dev/null; then
    echo "Adding $upload_owner to $clamav_group for ClamAV socket access"
    sudo usermod -aG "$clamav_group" "$upload_owner"
  fi

  sudo systemctl restart clamav-daemon.service

  for _ in {1..10}; do
    if [[ -S "$clamav_socket" ]]; then
      break
    fi
    sleep 1
  done

  if [[ ! -S "$clamav_socket" ]]; then
    echo "ClamAV socket was not created: $clamav_socket" >&2
    exit 1
  fi

  local actual_group actual_mode
  actual_group="$(stat -c '%G' "$clamav_socket")"
  actual_mode="$(stat -c '%a' "$clamav_socket")"
  if [[ "$actual_group" != "$clamav_group" || "$actual_mode" != "$socket_mode" ]]; then
    echo "Unexpected ClamAV socket permissions: group=$actual_group mode=$actual_mode" >&2
    exit 1
  fi

  sudo -u "$upload_owner" clamdscan --fdpass /etc/hosts >/dev/null
}

ensure_node_runtime() {
  local install_node=0

  if ! command -v node >/dev/null 2>&1; then
    install_node=1
  else
    local node_major
    node_major="$(node -p "process.versions.node.split('.')[0]")"
    if [[ "$node_major" -lt 20 ]]; then
      install_node=1
    fi
  fi

  if [[ "$install_node" -eq 1 ]]; then
    echo "Installing Node.js 20 runtime"
    curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
    sudo apt-get install -y nodejs build-essential
  fi

  if ! command -v pnpm >/dev/null 2>&1; then
    echo "Installing pnpm"
    if command -v corepack >/dev/null 2>&1; then
      sudo corepack enable
      sudo corepack prepare pnpm@10.28.0 --activate
    else
      sudo npm install -g pnpm@10.28.0
    fi
  fi
}

stop_existing_mentor_ai() {
  local pid_file="$1"
  if [[ ! -f "$pid_file" ]]; then
    return
  fi

  local pid
  pid="$(cat "$pid_file")"
  if [[ -n "$pid" ]] && kill -0 "$pid" >/dev/null 2>&1; then
    echo "Stopping existing MentorAI process: $pid"
    kill "$pid"

    local attempts=0
    while kill -0 "$pid" >/dev/null 2>&1; do
      attempts=$((attempts + 1))
      if [[ "$attempts" -ge 10 ]]; then
        echo "MentorAI process did not stop cleanly: $pid" >&2
        exit 1
      fi
      sleep 1
    done
  fi

  rm -f "$pid_file"
}

show_mentor_ai_logs() {
  local mentor_log_file="$1"

  if [[ -f "$mentor_log_file" ]]; then
    echo "Last 200 lines from MentorAI log:" >&2
    tail -n 200 "$mentor_log_file" >&2 || true
  else
    echo "MentorAI log file not found: $mentor_log_file" >&2
  fi
}

start_mentor_ai_process() {
  local mentor_host="$1"
  local mentor_port="$2"
  local mentor_base_path="$3"
  local mentor_log_file="$4"

  if [[ -f ".next/standalone/server.js" ]]; then
    echo "Starting MentorAI with standalone output"
    env \
      NODE_ENV=production \
      HOSTNAME="$mentor_host" \
      PORT="$mentor_port" \
      NEXT_PUBLIC_BASE_PATH="$mentor_base_path" \
      nohup node .next/standalone/server.js >"$mentor_log_file" 2>&1 </dev/null &
  else
    echo "Standalone output not found; starting MentorAI with next start"
    env \
      NODE_ENV=production \
      HOSTNAME="$mentor_host" \
      PORT="$mentor_port" \
      NEXT_PUBLIC_BASE_PATH="$mentor_base_path" \
      nohup pnpm exec next start --hostname "$mentor_host" --port "$mentor_port" >"$mentor_log_file" 2>&1 </dev/null &
  fi
}

wait_for_mentor_ai_health() {
  local mentor_pid="$1"
  local mentor_healthcheck_url="$2"
  local mentor_log_file="$3"

  local attempts=0
  local max_attempts=30

  while [[ "$attempts" -lt "$max_attempts" ]]; do
    if curl --fail --silent "$mentor_healthcheck_url" >/dev/null 2>&1; then
      echo "MentorAI health check passed"
      return 0
    fi

    if ! kill -0 "$mentor_pid" >/dev/null 2>&1; then
      echo "MentorAI process exited before becoming healthy" >&2
      show_mentor_ai_logs "$mentor_log_file"
      return 1
    fi

    attempts=$((attempts + 1))
    sleep 2
  done

  echo "MentorAI health check did not pass after $((max_attempts * 2)) seconds: $mentor_healthcheck_url" >&2
  show_mentor_ai_logs "$mentor_log_file"
  return 1
}

deploy_mentor_ai() {
  local mentor_enabled="${MENTOR_AI_ENABLED:-true}"
  mentor_enabled="$(printf '%s' "$mentor_enabled" | tr '[:upper:]' '[:lower:]')"

  if [[ "$mentor_enabled" == "false" ]]; then
    echo "MentorAI deployment disabled"
    return
  fi

  local mentor_repo_path="${MENTOR_AI_PATH:-$APP_PATH/mentorAI}"
  local mentor_base_path="${MENTOR_AI_BASE_PATH:-/mentor-ai}"
  local mentor_host="${MENTOR_AI_HOST:-127.0.0.1}"
  local mentor_port="${MENTOR_AI_PORT:-3000}"
  local mentor_healthcheck_url="${MENTOR_AI_HEALTHCHECK_URL:-http://$mentor_host:$mentor_port$mentor_base_path/api/health}"
  local mentor_pid_file="${MENTOR_AI_PID_FILE:-$mentor_repo_path/.mentor-ai.pid}"
  local mentor_log_dir="$mentor_repo_path/logs"
  local mentor_log_file="$mentor_log_dir/mentor-ai.log"

  if [[ ! -d "$mentor_repo_path" ]]; then
    echo "MentorAI path not found: $mentor_repo_path" >&2
    exit 1
  fi

  ensure_node_runtime

  cd "$mentor_repo_path"

  if [[ ! -f "package.json" || ! -f "pnpm-lock.yaml" ]]; then
    echo "MentorAI repository is missing package.json or pnpm-lock.yaml: $mentor_repo_path" >&2
    exit 1
  fi

  mkdir -p "$mentor_log_dir"

  echo "Installing MentorAI dependencies"
  pnpm install --frozen-lockfile

  echo "Building MentorAI"
  NEXT_PUBLIC_BASE_PATH="$mentor_base_path" pnpm build

  stop_existing_mentor_ai "$mentor_pid_file"

  echo "Starting MentorAI on $mentor_host:$mentor_port"
  start_mentor_ai_process "$mentor_host" "$mentor_port" "$mentor_base_path" "$mentor_log_file"

  local mentor_pid=$!
  echo "$mentor_pid" > "$mentor_pid_file"

  echo "Running MentorAI health check: $mentor_healthcheck_url"
  wait_for_mentor_ai_health "$mentor_pid" "$mentor_healthcheck_url" "$mentor_log_file"
}

reload_nginx_if_config_provided() {
  local nginx_site_path="${NGINX_SITE_PATH:-}"
  if [[ -z "$nginx_site_path" ]]; then
    return
  fi

  if [[ ! -f /tmp/skillbot_nginx_alb.conf ]]; then
    echo "Nginx config payload not found at /tmp/skillbot_nginx_alb.conf" >&2
    exit 1
  fi

  echo "Updating nginx config: $nginx_site_path"
  sudo cp /tmp/skillbot_nginx_alb.conf "$nginx_site_path"
  sudo nginx -t
  sudo systemctl reload nginx
}

ensure_private_upload_dirs() {
  local upload_root="${STOODY_UPLOAD_ROOT:-/var/lib/stoody/uploads}"
  local upload_owner
  local upload_group="${STOODY_UPLOAD_GROUP:-clamav}"
  upload_owner="$(app_owner)"

  if ! getent passwd "$upload_owner" >/dev/null; then
    echo "Upload directory owner does not exist: $upload_owner" >&2
    exit 1
  fi

  if ! getent group "$upload_group" >/dev/null; then
    echo "Upload directory group does not exist: $upload_group" >&2
    exit 1
  fi

  echo "Ensuring private upload directories under $upload_root ($upload_owner:$upload_group)"
  sudo install -d -o "$upload_owner" -g "$upload_group" -m 2750 \
    "$upload_root" \
    "$upload_root/quarantine" \
    "$upload_root/clean" \
    "$upload_root/rejected"

  sudo find "$upload_root" -type d -exec chmod 2750 {} +
  sudo chown -R "$upload_owner:$upload_group" "$upload_root"

  if sudo nginx -T 2>/dev/null | grep -Eq "^[[:space:]]*(root|alias)[[:space:]]+$upload_root(/|;)|/var/lib/stoody"; then
    echo "Nginx appears to expose private upload storage; refusing to continue" >&2
    exit 1
  fi
}

ensure_upload_cleanup_timer() {
  local enabled="${STOODY_UPLOAD_CLEANUP_ENABLED:-true}"
  if [[ "$enabled" != "true" ]]; then
    echo "Skipping upload cleanup timer because STOODY_UPLOAD_CLEANUP_ENABLED=$enabled"
    return
  fi

  local unit_base="${STOODY_UPLOAD_CLEANUP_UNIT:-stoody-upload-cleanup}"
  local timer_calendar="${STOODY_UPLOAD_CLEANUP_TIMER:-*-*-* 03:20:00}"
  local randomized_delay="${STOODY_UPLOAD_CLEANUP_RANDOMIZED_DELAY_SEC:-15m}"
  local upload_owner upload_group python_bin
  upload_owner="$(app_owner)"
  upload_group="${STOODY_UPLOAD_GROUP:-clamav}"

  if [[ -n "$VENV_PATH" && -x "$VENV_PATH/bin/python" ]]; then
    python_bin="$VENV_PATH/bin/python"
  else
    python_bin="$(command -v python3 || command -v python || true)"
  fi

  if [[ -z "$python_bin" ]]; then
    echo "Python interpreter not found for upload cleanup timer" >&2
    exit 1
  fi

  if [[ ! -f "$APP_PATH/scripts/cleanup_upload_storage.py" ]]; then
    echo "Upload cleanup script not found: $APP_PATH/scripts/cleanup_upload_storage.py" >&2
    exit 1
  fi

  echo "Installing upload cleanup timer: ${unit_base}.timer ($timer_calendar)"
  sudo tee "/etc/systemd/system/${unit_base}.service" >/dev/null <<SERVICE
[Unit]
Description=Stoody private upload storage cleanup

[Service]
Type=oneshot
User=$upload_owner
Group=$upload_group
WorkingDirectory=$APP_PATH
EnvironmentFile=-$APP_PATH/.env
ExecStart=$python_bin $APP_PATH/scripts/cleanup_upload_storage.py --execute
Nice=10
IOSchedulingClass=best-effort
SERVICE

  sudo tee "/etc/systemd/system/${unit_base}.timer" >/dev/null <<TIMER
[Unit]
Description=Run Stoody private upload storage cleanup

[Timer]
OnCalendar=$timer_calendar
Persistent=true
RandomizedDelaySec=$randomized_delay
Unit=${unit_base}.service

[Install]
WantedBy=timers.target
TIMER

  sudo systemctl daemon-reload
  sudo systemctl enable --now "${unit_base}.timer"
  systemctl is-enabled "${unit_base}.timer" >/dev/null
  systemctl is-active "${unit_base}.timer" >/dev/null
}

acquire_deploy_lock
sync_git_repo "$APP_PATH" "$BRANCH"

if [[ ! -f "$APP_PATH/$REQUIREMENTS_PATH" ]]; then
  echo "Requirements file not found: $APP_PATH/$REQUIREMENTS_PATH" >&2
  exit 1
fi

if [[ -n "$VENV_PATH" ]]; then
  if [[ ! -f "$VENV_PATH/bin/activate" ]]; then
    echo "Virtualenv activate script not found: $VENV_PATH/bin/activate" >&2
    exit 1
  fi

  # shellcheck disable=SC1090
  source "$VENV_PATH/bin/activate"
fi

cd "$APP_PATH"

echo "Installing Python dependencies"
python -m pip install --disable-pip-version-check -r "$REQUIREMENTS_PATH"

ensure_clamav_socket_hardening
ensure_private_upload_dirs
ensure_upload_cleanup_timer

echo "Restarting service via $SERVICE_MANAGER"
case "$SERVICE_MANAGER" in
  systemctl)
    sudo systemctl restart "$SERVICE_NAME_OR_COMMAND"
    ;;
  supervisorctl)
    sudo supervisorctl restart "$SERVICE_NAME_OR_COMMAND"
    ;;
  command)
    bash -lc "$SERVICE_NAME_OR_COMMAND"
    ;;
  *)
    echo "Unsupported service manager: $SERVICE_MANAGER" >&2
    exit 1
    ;;
esac

sleep "$POST_RESTART_DELAY"

echo "Running backend health check: $HEALTHCHECK_URL"
curl --fail --silent --show-error "$HEALTHCHECK_URL" >/dev/null

deploy_mentor_ai
reload_nginx_if_config_provided

echo "Deployment completed successfully for $APP_PATH"
