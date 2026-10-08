#!/usr/bin/env bash
# P0 preflight: raw machine observation + python-level diagnostics.
# Read-only: performs no installs and touches nothing outside the project.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${REPO_ROOT}/outputs/diagnostics"
mkdir -p "${OUT}"

{
  echo "=== uname ===";      uname -a
  echo "=== os-release ==="; cat /etc/os-release
  echo "=== whoami ===";     whoami
  echo "=== nvidia-smi ==="; nvidia-smi
  echo "=== nvidia query ==="; nvidia-smi --query-gpu=index,name,memory.total,memory.free,driver_version --format=csv
  echo "=== free ===";       free -h
  echo "=== df ===";         df -h
  echo "=== python ===";     python3 --version
  echo "=== which python ==="; which python3
  echo "=== git ===";        (command -v git && git --version) || echo "git missing"
} > "${OUT}/raw_preflight.txt" 2>&1
echo "raw output -> ${OUT}/raw_preflight.txt"

PY="${REPO_ROOT}/.venv/bin/python"
[ -x "${PY}" ] || PY="python3"

"${PY}" -m src.pipeline.preflight --out "${OUT}"
code=$?
echo "python preflight exit=${code}"
exit ${code}
