#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  remote_deploy_python_service.sh <app_path> <branch> <venv_path> <requirements_path> <service_manager> <service_name_or_command> <healthcheck_url> [post_restart_delay_seconds]

Examples:
  remote_deploy_python_service.sh /home/ubuntu/Stoody/stoody-backend dev /home/ubuntu/Stoody/stoody-backend/venv requirements.txt systemctl stoody-backend http://127.0.0.1:5001/health 5
  remote_deploy_python_service.sh /home/ubuntu/Stoody/stoody-backend dev /home/ubuntu/Stoody/stoody-backend/venv requirements.txt supervisorctl skillbot-development http://127.0.0.1:5001/health 5
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

cd "$APP_PATH"

if [[ -n "$(git diff --name-only)" || -n "$(git diff --cached --name-only)" ]]; then
  echo "Refusing to deploy from a dirty git worktree at $APP_PATH" >&2
  exit 1
fi

echo "Fetching latest code for branch: $BRANCH"
git fetch origin "$BRANCH"

current_branch="$(git rev-parse --abbrev-ref HEAD || true)"
if [[ "$current_branch" != "$BRANCH" ]]; then
  if git show-ref --verify --quiet "refs/heads/$BRANCH"; then
    git checkout "$BRANCH"
  else
    git checkout -b "$BRANCH" --track "origin/$BRANCH"
  fi
fi

git pull --ff-only origin "$BRANCH"

if [[ ! -f "$REQUIREMENTS_PATH" ]]; then
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

echo "Installing Python dependencies"
python -m pip install --disable-pip-version-check -r "$REQUIREMENTS_PATH"

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

echo "Running health check: $HEALTHCHECK_URL"
curl --fail --silent --show-error "$HEALTHCHECK_URL" >/dev/null

echo "Deployment completed successfully for $APP_PATH"
