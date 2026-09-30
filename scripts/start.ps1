param([switch]$BuildOnly)
$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..')
if (-not $env:BUILDX_BUILDER) { $env:BUILDX_BUILDER = 'default' }
if (-not $env:BUILDX_CONFIG) {
    # Keep transient Buildx activity files away from Dropbox/Windows file locks.
    $env:BUILDX_CONFIG = Join-Path ([IO.Path]::GetTempPath()) ('review-authenticity-buildx-' + [Guid]::NewGuid().ToString('N'))
}
docker compose build
if ($LASTEXITCODE -ne 0) { throw 'Docker build failed. Start Docker Desktop and retry.' }
if (-not (Test-Path 'artifacts/baseline/demo-v1/metadata.json')) {
    Write-Host 'Building a tiny SYNTHETIC DEMO baseline once. Its scores are not real evaluation results.'
    docker compose --profile tools run --rm trainer train-baseline --dataset data/demo/reviews.csv --version demo-v1 --demo
    if ($LASTEXITCODE -ne 0) { throw 'Demo bootstrap failed.' }
}
if (-not $BuildOnly) {
    docker compose up -d --wait
    if ($LASTEXITCODE -ne 0) { throw 'Application did not become healthy. Run docker compose logs.' }
    Write-Host 'Open http://localhost:3000 (API docs: http://localhost:8000/docs)'
}
