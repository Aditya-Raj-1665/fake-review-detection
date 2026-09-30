DATASET ?= data/raw/cornell.csv
VERSION ?= v1
ARTIFACT ?= artifacts/baseline/$(VERSION)
UV ?= uv
.PHONY: setup bootstrap run stop logs test lint format train-baseline train-transformer evaluate evaluate-baseline dev-backend dev-frontend
setup bootstrap:
	sh scripts/start.sh --build-only
run:
	docker compose up -d --wait
stop:
	docker compose down
logs:
	docker compose logs -f
test:
	$(UV) run --locked pytest
	npm --prefix frontend ci
	npm --prefix frontend test
lint:
	$(UV) run --locked ruff check src tests
	$(UV) run --locked ruff format --check src tests
	npm --prefix frontend run lint
format:
	$(UV) run --locked ruff check --fix src tests
	$(UV) run --locked ruff format src tests
train-baseline:
	docker compose --profile tools run --rm trainer train-baseline --dataset $(DATASET) --version $(VERSION)
train-transformer:
	$(UV) run --locked --extra transformer reviews train-transformer --dataset $(DATASET) --version $(VERSION)
evaluate evaluate-baseline:
	$(UV) run --locked reviews evaluate --artifact $(ARTIFACT)
dev-backend:
	$(UV) run --locked uvicorn review_authenticity.api:create_app --factory --reload --port 8000
dev-frontend:
	npm --prefix frontend run dev
