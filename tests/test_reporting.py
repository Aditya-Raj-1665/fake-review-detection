"""Reporting and lightweight optional-calibration checks without neural dependencies."""

import json

import numpy as np
import pytest
from scipy.special import softmax
from sklearn.metrics import log_loss

from review_authenticity.cli import compare_artifacts
from review_authenticity.download import download_cornell
from review_authenticity.transformer import fit_temperature


def test_comparison_uses_real_validation_results(artifact, tmp_path):
    output = tmp_path / "comparison.md"
    compare_artifacts([artifact], output)
    report = json.loads(output.with_suffix(".json").read_text())
    assert report["split"] == "validation"
    assert {row["model"] for row in report["models"]} == {"logistic_regression", "naive_bayes"}
    assert "naive_bayes" in output.read_text()


def test_download_rejects_changed_archive(tmp_path):
    archive = tmp_path / "untrusted.zip"
    archive.write_bytes(b"unexpected upstream response")
    with pytest.raises(ValueError, match="checksum"):
        download_cornell(tmp_path / "reviews.csv", archive)
    assert not (tmp_path / "reviews.csv").exists()


def test_temperature_reduces_overconfident_log_loss():
    logits = np.array([[8.0, 0.0], [8.0, 0.0], [0.0, 8.0], [0.0, 8.0]])
    labels = np.array([0, 1, 1, 0])
    temperature = fit_temperature(logits, labels)
    assert 0 < temperature <= 10
    assert log_loss(labels, softmax(logits / temperature, axis=1)) < log_loss(
        labels, softmax(logits, axis=1)
    )
