#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")/.."
export BUILDX_BUILDER="${BUILDX_BUILDER:-default}"
docker compose build
if [ ! -f artifacts/baseline/demo-v1/metadata.json ]; then
  echo 'Building a tiny SYNTHETIC DEMO baseline once; these are not real evaluation results.'
  docker compose --profile tools run --rm trainer train-baseline --dataset data/demo/reviews.csv --version demo-v1 --demo
fi
if [ "${1:-}" != '--build-only' ]; then
  docker compose up -d --wait
  echo 'Open http://localhost:3000 (API docs: http://localhost:8000/docs)'
fi
