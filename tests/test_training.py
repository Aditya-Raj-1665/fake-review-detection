"""Verify actual fitted artifacts, metrics conventions, and probability semantics."""

import json
import shutil

import pytest
from pydantic import ValidationError

from review_authenticity.config import Settings, TrainingConfig
from review_authenticity.errors import ArtifactError
from review_authenticity.evaluation import measure
from review_authenticity.inference import InferenceService
from review_authenticity.labels import Label, predicted_label
from review_authenticity.training import train_baselines


def test_metadata_and_roundtrip(artifact, settings):
    metadata = json.loads((artifact / "metadata.json").read_text())
    assert metadata["demo"] is True
    assert set(metadata["validation_comparison"]) == {"logistic_regression", "naive_bayes"}
    assert metadata["test_metrics"]["sample_count"] == 6
    service = InferenceService(settings)
    results = service.predict(["This lamp works but the switch is stiff."])
    result = results[0]
    assert result["genuine_probability"] + result["deceptive_probability"] == pytest.approx(1)
    assert result["authenticity_score"] == round(result["genuine_probability"] * 100, 2)
    assert (
        service.predict(["This lamp works but the switch is stiff."])[0]["genuine_probability"]
        == result["genuine_probability"]
    )


def test_chosen_label_confidence_under_custom_threshold(settings):
    service = InferenceService(settings.model_copy(update={"prediction_threshold": 0.99}))
    result = service.predict(["I washed this twice and the seam stayed straight."])[0]
    assert result["label"] == "deceptive"
    assert result["confidence"] == result["deceptive_probability"]
    assert predicted_label(0.5, 0.5) == Label.GENUINE
    assert predicted_label(0.6, 0.7) == Label.DECEPTIVE


def test_missing_artifact(tmp_path):
    with pytest.raises(ArtifactError, match="train-baseline"):
        InferenceService(Settings(_env_file=None, model_path=tmp_path))


def test_corrupted_artifact_rejected(artifact, tmp_path):
    copied = tmp_path / "copy"
    shutil.copytree(artifact, copied)
    (copied / "model.joblib").write_bytes(b"not a model")
    with pytest.raises(ArtifactError, match="checksum"):
        InferenceService(Settings(_env_file=None, model_path=copied))


def test_refuse_overwrite(artifact):
    with pytest.raises(FileExistsError):
        train_baselines(artifact, artifact.parent.parent, artifact.name, TrainingConfig())


def test_deceptive_positive_metrics():
    metrics = measure([0, 0, 1, 1], [0.1, 0.8, 0.2, 0.9], 0.5)
    assert metrics["false_positives"] == 1
    assert metrics["false_negatives"] == 1
    assert metrics["f1"] == 0.5
    assert metrics["confusion_matrix"] == [[1, 1], [1, 1]]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"prediction_threshold": 1.2},
        {"max_review_length": 0},
        {"app_env": "production", "cors_origins": ["*"]},
    ],
)
def test_invalid_settings(kwargs):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **kwargs)


def test_invalid_split_config():
    with pytest.raises(ValidationError):
        TrainingConfig(validation_ratio=0.5, test_ratio=0.4)
