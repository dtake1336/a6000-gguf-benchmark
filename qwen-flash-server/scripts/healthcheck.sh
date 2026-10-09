#!/usr/bin/env bash
set -euo pipefail

BASE_URL=${BASE_URL:-http://127.0.0.1:${PORT:-8080}}
curl --fail --silent --show-error "$BASE_URL/health"
printf '\n'