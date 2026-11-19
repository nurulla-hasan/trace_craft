#!/usr/bin/env bash
set -euo pipefail

TARGET="${1:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

usage() {
  echo "Usage: $0 {frontend|backend|all}"
  exit 1
}

pull_repo() {
  local dir="$1"
  if [ -d "$dir/.git" ]; then
    echo "==> Pulling latest changes in $dir..."
    git -C "$dir" pull --ff-only
  fi
}

case "$TARGET" in
  frontend)
    if [ -d ".git" ]; then
      pull_repo "."
    else
      pull_repo "frontend"
    fi
    echo "==> Rebuilding and restarting frontend..."
    docker compose up -d --build --no-deps frontend
    ;;
  backend)
    if [ -d ".git" ]; then
      pull_repo "."
    else
      pull_repo "backend"
    fi
    echo "==> Rebuilding and restarting backend..."
    docker compose up -d --build --no-deps backend
    ;;
  all)
    if [ -d ".git" ]; then
      pull_repo "."
    else
      pull_repo "backend"
      pull_repo "frontend"
    fi
    echo "==> Rebuilding and restarting all services..."
    docker compose up -d --build
    ;;
  *)
    usage
    ;;
esac

echo "==> Cleaning up dangling Docker images..."
docker image prune -f >/dev/null 2>&1 || true

echo "==> Current status:"
docker compose ps
