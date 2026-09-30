"""Exercise lifespan loading, privacy-preserving errors, and request boundaries."""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from review_authenticity.api import create_app
from review_authenticity.inference import InferenceService


def test_api_lifespan_and_batch(settings):
    with patch("review_authenticity.api.InferenceService", wraps=InferenceService) as loader:
        with TestClient(create_app(settings)) as client:
            assert client.get("/health").json() == {"status": "healthy", "model_loaded": True}
            assert client.get("/api/v1/model/info").json()["demo"] is True
            assert (
                client.post("/api/v1/predict", json={"text": "Works after two weeks."}).status_code
                == 200
            )
            response = client.post("/api/v1/predict/batch", json={"reviews": ["Nice", "Bad"]})
            assert response.status_code == 200
            assert len(response.json()) == 2
            assert loader.call_count == 1


@pytest.mark.parametrize(
    "payload",
    [
        {"text": ""},
        {"text": " \t\n"},
        {"text": "a" * 101},
        {},
        {"text": None},
        {"text": 42},
        {"text": "fine", "unknown": True},
    ],
)
def test_invalid_review(settings, payload):
    with TestClient(create_app(settings)) as client:
        assert client.post("/api/v1/predict", json=payload).status_code == 422


def test_batch_limits_and_no_input_echo(settings):
    with TestClient(create_app(settings)) as client:
        for reviews in ([], ["a"] * 4, ["hello", " "]):
            assert (
                client.post("/api/v1/predict/batch", json={"reviews": reviews}).status_code == 422
            )
        response = client.post("/api/v1/predict", json={"text": {"secret": "PRIVATE REVIEW"}})
        assert "PRIVATE REVIEW" not in response.text
        assert client.post("/api/v1/predict", content="{").status_code == 422


def test_body_limit(settings):
    limited = settings.model_copy(update={"max_request_bytes": 1024})
    with TestClient(create_app(limited)) as client:
        assert client.post("/api/v1/predict", content="x" * 1025).status_code == 413


def test_inference_failure_is_safe(settings):
    with TestClient(create_app(settings)) as client:
        with patch.object(InferenceService, "predict", side_effect=RuntimeError("internal path")):
            response = client.post("/api/v1/predict", json={"text": "a review"})
            assert response.status_code == 500
            assert "internal path" not in response.text
