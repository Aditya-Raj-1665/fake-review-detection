# Verification record

Executed on 2026-09-25 in this Windows workspace. This record distinguishes observed
results from optional functionality that has not been run.

| Check | Observed result |
|---|---|
| Python unit/integration suite | 35 passed |
| Python coverage | 81%; optional transformer module excluded from the coverage denominator |
| Ruff lint and formatting | Passed for package, tests, and HTTP smoke script |
| Frontend component suite | 2 passed with Vitest 4.1.11 |
| ESLint | Passed |
| TypeScript and Vite production build | Passed |
| Locked frontend install/audit | Passed; zero reported vulnerabilities at execution time |
| Python lock consistency | `uv lock --check` passed |
| Compose configuration | `docker compose config --quiet` passed |
| Docker image builds | Baseline backend and nginx frontend built successfully |
| Container health | Both application containers reached healthy state |
| Windows startup command | `powershell -ExecutionPolicy Bypass -File scripts/start.ps1` completed successfully |
| Live API/proxy smoke | Passed health, model info, single/batch predictions, input errors, and eight concurrent requests through nginx |
| Docker trainer | Created `docker-smoke-v1` from 40 synthetic rows, with metadata, metrics, and a confusion matrix |
| Artifact permission regression | Newly host-trained `windows-permissions-v1` loaded and predicted in Docker as UID 10001 without manual ACL repair |
| Real training | Imported Cornell corpus; trained/calibrated both baselines; selected LR on validation; evaluated selected model on test |
| Report commands | `evaluate` and `compare` executed successfully |
| Figure inspection | Saved test confusion matrix visually inspected; labels and counts match computed metrics |

The Python tests used `--basetemp=.cache/pytest-run-4` because this agent's restricted
environment could not access the default Windows pytest temporary directory. This is
an environment override, not a special test implementation. The suite emitted one
Starlette deprecation warning about the currently supported HTTPX test transport.
Frontend build/test tools required execution outside the restricted sandbox to read
ancestor directories; no source-code workaround or disabled check was used.

## Re-run

With the development tools installed as described in the README:

```powershell
uv sync --locked
uv run --locked pytest
uv run --locked ruff check src tests scripts/smoke.py
uv run --locked ruff format --check src tests scripts/smoke.py
npm --prefix frontend ci
npm --prefix frontend test
npm --prefix frontend run lint
npm --prefix frontend run build
docker compose config --quiet
powershell -ExecutionPolicy Bypass -File scripts/start.ps1
python scripts/smoke.py
```

For another Docker trainer smoke run, choose a fresh immutable version:

```powershell
docker compose --profile tools run --rm trainer train-baseline --dataset data/demo/reviews.csv --version your-smoke-version --demo
```

## Boundaries

* DistilBERT training, pretrained weight download, GPU inference, and the optional
  transformer Docker image were **not executed**, as requested. The temperature-fitting
  helper has an inexpensive numerical test; this does not verify the neural pipeline.
* No browser automation surface was connected, so browser visual inspection and an
  actual browser click-through were unavailable. Component tests exercised entry,
  submission, loading, result rendering, and errors; HTTP checks exercised the live proxy.
* The GitHub Actions workflow was written but was not run on GitHub.
* No public deployment or load benchmark was performed. Eight concurrent smoke requests
  verify basic serving behavior, not production capacity or a p95 latency guarantee.
* The real held-out metrics concern this historical hotel-review corpus and review-level
  split. They do not establish general fraud detection or external-domain accuracy.

## Issues found and resolved

Docker's initial image download failed to resolve its registry; retrying with the
built-in builder succeeded. Windows file locking also affected package/build caches.
The Windows launcher now uses a fresh temporary Buildx configuration outside Dropbox
unless the caller has explicitly configured one; the engine retains its image/build cache.
A patched Vitest version replaced the initial version with a reported advisory.
Most importantly, a private Windows ACL from `tempfile.mkdtemp` prevented the non-root
container from opening a host-trained artifact. Publication now creates a unique staging
directory with inherited permissions; the existing artifacts' inheritance was repaired.
The backend remains non-root, and its artifact mount remains read-only.
