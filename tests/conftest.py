"""Small real calibrated artifacts shared by inexpensive integration tests."""

from pathlib import Path

import pytest

from review_authenticity.config import Settings, TrainingConfig
from review_authenticity.training import train_baselines


@pytest.fixture(scope="session")
def artifact(tmp_path_factory):
    root = tmp_path_factory.mktemp("artifact")
    return train_baselines(
        Path("data/demo/reviews.csv"),
        root,
        "test-v1",
        TrainingConfig(),
        demo=True,
        report_root=root / "reports",
    )


@pytest.fixture
def settings(artifact):
    return Settings(
        _env_file=None, app_env="test", model_path=artifact, max_review_length=100, max_batch_size=3
    )
