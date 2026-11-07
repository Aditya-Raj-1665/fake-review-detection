FROM python:3.12-slim-bookworm AS builder
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=0
RUN pip install --no-cache-dir uv==0.12.19
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project
COPY src ./src
RUN uv sync --locked --no-dev --no-editable

FROM python:3.12-slim-bookworm AS runtime
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    MPLCONFIGDIR=/tmp/matplotlib APP_ENV=production
WORKDIR /app
RUN groupadd --gid 10001 app && useradd --uid 10001 --gid app --create-home app \
    && mkdir artifacts reports && chown -R app:app /app
COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --chown=app:app configs ./configs
COPY --chown=app:app data/demo ./data/demo
USER app
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=5s --start-period=40s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=3)"
CMD ["python", "-m", "review_authenticity.server"]

FROM builder AS transformer-builder
RUN uv sync --locked --no-dev --extra transformer --no-editable

FROM runtime AS transformer
COPY --from=transformer-builder --chown=app:app /app/.venv /app/.venv


# Cloud deployment: train and package the Cornell baseline

FROM builder AS cloud-trainer

COPY configs ./configs

RUN /app/.venv/bin/reviews download-cornell

RUN /app/.venv/bin/reviews train-baseline \
    --dataset data/raw/cornell.csv \
    --version cornell-v1

FROM runtime AS cloud

COPY --from=cloud-trainer --chown=app:app \
    /app/artifacts/baseline/cornell-v1 \
    /app/artifacts/baseline/cornell-v1

ENV MODEL_PATH=/app/artifacts/baseline/cornell-v1
ENV MODEL_TYPE=baseline
