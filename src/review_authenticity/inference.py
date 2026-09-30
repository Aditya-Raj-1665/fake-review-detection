"""Load trusted local artifacts once and expose a common prediction interface."""

import hashlib
import json
import logging
from importlib.metadata import version
from pathlib import Path
from threading import Lock
from time import perf_counter

import joblib
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from review_authenticity.config import Settings
from review_authenticity.errors import ArtifactError
from review_authenticity.labels import LABEL_NAMES, LABEL_VALUES, Label, predicted_label

logger = logging.getLogger(__name__)


class ArtifactMetadata(BaseModel):
    """Validate the artifact contract before deserializing model files."""

    model_config = ConfigDict(extra="allow")
    schema_version: int
    model_type: str
    model_name: str
    version: str
    demo: bool
    threshold: float = Field(gt=0, lt=1)
    label_mapping: dict[str, int]
    calibration: str
    libraries: dict[str, str]
    model_sha256: str | None = None
    max_tokens: int | None = Field(default=None, ge=16, le=512)
    temperature: float = Field(default=1.0, gt=0)


class InferenceService:
    """Serve a loaded model without training or per-request disk access."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.lock = Lock()
        path = settings.model_path
        try:
            self.metadata = ArtifactMetadata.model_validate_json(
                (path / "metadata.json").read_text(encoding="utf-8")
            )
            meta = self.metadata
            if meta.schema_version != 1 or meta.label_mapping != LABEL_VALUES:
                raise ValueError("Unsupported schema or label mapping")
            if meta.model_type != settings.model_type:
                raise ValueError("MODEL_TYPE does not match artifact metadata")
            if meta.model_type == "baseline":
                if meta.libraries["scikit-learn"] != version("scikit-learn"):
                    raise ValueError(
                        "scikit-learn differs from training; use the training lockfile"
                    )
                model_file = path / "model.joblib"
                digest = hashlib.sha256(model_file.read_bytes()).hexdigest()
                if digest != meta.model_sha256:
                    raise ValueError("Model checksum mismatch")
                # Checksums detect damage; they do NOT make untrusted pickle files safe.
                self.model = joblib.load(model_file)
                if set(self.model.classes_) != set(LABEL_VALUES.values()):
                    raise ValueError("Model classes do not match the artifact label mapping")
            else:
                from review_authenticity.transformer import TransformerPredictor

                self.model = TransformerPredictor(path, meta.max_tokens, meta.temperature)
        except Exception as exc:
            command = (
                "uv run reviews train-baseline --dataset data/demo/reviews.csv "
                "--version demo-v1 --demo"
                if settings.model_type == "baseline"
                else "uv run --extra transformer reviews train-transformer "
                "--dataset data/raw/reviews.csv --version v1"
            )
            raise ArtifactError(
                f"Cannot load trusted {settings.model_type} artifact at {path}. "
                f"Create one with: {command}; then set MODEL_PATH to its directory. Reason: {exc}"
            ) from exc
        self.threshold = settings.prediction_threshold or self.metadata.threshold
        logger.info(
            "Model loaded: name=%s version=%s demo=%s",
            self.metadata.model_name,
            self.metadata.version,
            self.metadata.demo,
        )

    def info(self) -> dict:
        """Expose public model facts without local paths or training text."""
        return {
            "model_name": self.metadata.model_name,
            "model_version": self.metadata.version,
            "model_type": self.metadata.model_type,
            "demo": self.metadata.demo,
            "threshold": self.threshold,
            "calibration": self.metadata.calibration,
            "max_review_length": self.settings.max_review_length,
            "max_batch_size": self.settings.max_batch_size,
            "max_tokens": self.metadata.max_tokens,
        }

    def _signals(self, text: str) -> list[dict]:
        """Show signed TF-IDF contributions to the uncalibrated LR decision score."""
        if self.metadata.model_name != "logistic_regression":
            return []
        pipeline = self.model.calibrated_classifiers_[0].estimator
        vectorizer, classifier = pipeline["tfidf"], pipeline["classifier"]
        vector = vectorizer.transform([text])
        names = vectorizer.get_feature_names_out()
        contributions = vector.data * classifier.coef_[0, vector.indices]
        ranking = np.argsort(np.abs(contributions))[::-1][:5]
        return [
            {
                "term": str(names[vector.indices[index]]),
                "direction": "genuine" if contributions[index] > 0 else "deceptive",
                "weight": round(float(contributions[index]), 5),
            }
            for index in ranking
            if contributions[index] != 0
        ]

    def predict(self, texts: list[str]) -> list[dict]:
        """Return calibrated probabilities and the probability of the chosen label."""
        started = perf_counter()
        if self.metadata.model_type == "baseline":
            column = list(self.model.classes_).index(int(Label.GENUINE))
            probabilities = self.model.predict_proba(texts)[:, column]
        else:
            # Bound GPU memory and avoid concurrent tokenizer/model mutations.
            with self.lock:
                probabilities = self.model.predict(texts)
        probabilities = np.asarray(probabilities, dtype=float)
        if probabilities.shape != (len(texts),) or not np.all(np.isfinite(probabilities)):
            raise RuntimeError("Model returned invalid probability shape or non-finite values")
        if np.any((probabilities < 0) | (probabilities > 1)):
            raise RuntimeError("Model returned probabilities outside [0, 1]")
        elapsed = round((perf_counter() - started) * 1000, 2)
        predictions = []
        for text, probability in zip(texts, probabilities, strict=True):
            p = float(probability)
            label = predicted_label(p, self.threshold)
            predictions.append(
                {
                    "label": LABEL_NAMES[label],
                    "authenticity_score": round(p * 100, 2),
                    "genuine_probability": p,
                    "deceptive_probability": 1 - p,
                    "confidence": p if label == Label.GENUINE else 1 - p,
                    "model_name": self.metadata.model_name,
                    "model_version": self.metadata.version,
                    "demo": self.metadata.demo,
                    "threshold": self.threshold,
                    "inference_ms": elapsed,
                    "model_signals": self._signals(text),
                }
            )
        return predictions


def read_metadata(path: Path) -> dict:
    """Read training metadata for offline reporting."""
    return json.loads((path / "metadata.json").read_text(encoding="utf-8"))
