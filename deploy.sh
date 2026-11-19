#!/usr/bin/env bash
set -euo pipefail

# TraceCraft 1-Click VPS Deployment Script
# Usage:
#   curl -sSL https://raw.githubusercontent.com/nurulla-hasan/trace_craft/main/deploy.sh | bash
#   OR
#   ./deploy.sh [YOUR_VPS_IP_OR_DOMAIN]

echo "=========================================================="
echo "         TraceCraft - 1-Click VPS Deployment"
echo "=========================================================="

# 1. Determine target IP / Domain
INPUT_HOST="${1:-}"
if [ -n "$INPUT_HOST" ]; then
  HOST="$INPUT_HOST"
else
  echo "==> Detecting public IP address..."
  HOST=$(curl -s4 --max-time 5 https://api.ipify.org || \
         curl -s4 --max-time 5 https://ifconfig.me || \
         curl -s4 --max-time 5 https://icanhazip.com || true)
  if [ -z "$HOST" ]; then
    echo "Warning: Could not automatically detect public IP."
    read -rp "Please enter your VPS public IP or domain: " HOST
  fi
fi

echo "==> Using host: $HOST"

# 2. Check / Install Docker and Docker Compose
command_exists() {
  command -v "$1" >/dev/null 2>&1
}

if ! command_exists docker; then
  echo "==> Docker not found. Installing Docker..."
  curl -fsSL https://get.docker.com | sh
  if command_exists systemctl; then
    systemctl start docker || true
    systemctl enable docker || true
  fi
fi

# Ensure docker compose (v2) is available
if ! docker compose version >/dev/null 2>&1; then
  echo "==> Installing docker compose plugin..."
  if command_exists apt-get; then
    apt-get update && apt-get install -y docker-compose-plugin
  elif command_exists yum; then
    yum install -y docker-compose-plugin
  fi
fi

# 3. Setup Project Directory
REPO_URL="https://github.com/nurulla-hasan/trace_craft.git"
INSTALL_DIR="${INSTALL_DIR:-$HOME/tracecraft}"

if [ -f "docker-compose.yml" ]; then
  WORK_DIR="$(pwd)"
  echo "==> Already in repository directory: $WORK_DIR"
else
  if [ -d "$INSTALL_DIR/.git" ]; then
    echo "==> Updating existing repository in $INSTALL_DIR..."
    cd "$INSTALL_DIR"
    git pull origin main || true
    WORK_DIR="$INSTALL_DIR"
  else
    echo "==> Cloning repository into $INSTALL_DIR..."
    git clone "$REPO_URL" "$INSTALL_DIR"
    cd "$INSTALL_DIR"
    WORK_DIR="$INSTALL_DIR"
  fi
fi

cd "$WORK_DIR"

# 4. Generate or update .env file
if [ ! -f ".env" ]; then
  echo "==> Creating .env from .env.docker.example..."
  if [ -f ".env.docker.example" ]; then
    cp .env.docker.example .env
  else
    echo "Error: .env.docker.example not found!"
    exit 1
  fi

  # Auto-generate secure random keys
  PG_PASS=$(openssl rand -hex 16 2>/dev/null || echo "pg_$(date +%s)")
  JWT_ACC=$(openssl rand -hex 32 2>/dev/null || echo "jwt_acc_$(date +%s)")
  JWT_REF=$(openssl rand -hex 32 2>/dev/null || echo "jwt_ref_$(date +%s)")

  # Replace placeholder values in .env
  sed -i "s|YOUR_VPS_IP|$HOST|g" .env
  sed -i "s|change_this_strong_postgres_password|$PG_PASS|g" .env
  sed -i "s|replace-with-32-plus-char-random-access-secret|$JWT_ACC|g" .env
  sed -i "s|replace-with-32-plus-char-random-refresh-secret|$JWT_REF|g" .env
  echo "==> Configured .env with host $HOST and generated secure secrets."
else
  echo "==> Existing .env found. Keeping current settings."
fi

# Ensure update.sh is executable
if [ -f "update.sh" ]; then
  chmod +x update.sh
fi

# 5. Build and launch all services with Docker Compose
echo "==> Building and launching containers (Postgres, Redis, Engine, Backend, Frontend)..."
docker compose up -d --build

echo "==> Waiting for services to initialize..."
sleep 5

# 6. Display Status and Access Info
echo ""
echo "=========================================================="
echo "          TraceCraft Deployed Successfully!               "
echo "=========================================================="
echo ""
echo "  Frontend URL:     http://$HOST:3010"
echo "  Backend API:      http://$HOST:5010/api/v1"
echo "  Vector Engine:    http://$HOST:8000/docs"
echo ""
echo "  Default Admin:    admin@example.com"
echo "  Default Password: ChangeThisAdminPassword123!"
echo "  (You can change these credentials in .env)"
echo ""
echo "  Useful Commands:"
echo "    - View logs:    docker compose logs -f"
echo "    - Update app:   ./update.sh all"
echo "    - Stop app:     docker compose down"
echo "    - Restart app:  docker compose restart"
echo "=========================================================="
