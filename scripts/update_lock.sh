#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: uv is required: https://docs.astral.sh/uv/getting-started/installation/" >&2
    exit 1
fi

# pyproject.toml is the single source of truth for the minimum uv version.
uv lock
uv sync --locked --group dev
uv lock --check

echo
echo "uv.lock is current and .venv matches it."
echo "Next: git add pyproject.toml uv.lock scripts/ && git commit"
