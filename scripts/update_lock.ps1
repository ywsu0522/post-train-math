$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Push-Location $repoRoot

try {
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        throw "uv is required: https://docs.astral.sh/uv/getting-started/installation/"
    }

    # pyproject.toml is the single source of truth for the minimum uv version.
    uv lock
    if ($LASTEXITCODE -ne 0) { throw "uv lock failed" }

    uv sync --locked --group dev
    if ($LASTEXITCODE -ne 0) { throw "uv sync failed" }

    uv lock --check
    if ($LASTEXITCODE -ne 0) { throw "uv lock check failed" }

    Write-Host ""
    Write-Host "uv.lock is current and .venv matches it."
    Write-Host "Next: git add pyproject.toml uv.lock scripts/ && git commit"
}
finally {
    Pop-Location
}
