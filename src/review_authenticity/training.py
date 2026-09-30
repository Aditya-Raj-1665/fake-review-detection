"""Train calibrated baselines, select on validation, and test the winner once."""

import hashlib
import logging
import platform
import re
import shutil
import uuid
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from review_authenticity.config import TrainingConfig
from review_authenticity.data import load_dataset, normalize_text, split_dataset
from review_authenticity.evaluation import measure, save_report, write_json
from review_authenticity.labels import LABEL_VALUES, Label

logger = logging.getLogger(__name__)
BASELINES = ("logistic_regression", "naive_bayes")


def build_pipeline(name: str, config: TrainingConfig) -> CalibratedClassifierCV:
    """Calibrate out-of-fold probabilities without touching validation or test data."""
    if name not in BASELINES:
        raise ValueError(f"Unknown baseline: {name}")
    classifier = (
        LogisticRegression(
            C=config.logistic_c,
            class_weight=config.class_weight,
            max_iter=1000,
            random_state=config.seed,
        )
        if name == "logistic_regression"
        else MultinomialNB(alpha=config.nb_alpha)
    )
    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    preprocessor=normalize_text,
                    lowercase=False,
                    token_pattern=r"(?u)\b\w+\b|[^\w\s]",
                    max_features=config.max_features,
                    ngram_range=config.ngram_range,
                    min_df=config.min_df,
                    max_df=config.max_df,
                    sublinear_tf=True,
                ),
            ),
            ("classifier", classifier),
        ]
    )
    return CalibratedClassifierCV(
        pipeline,
        method="sigmoid",
        ensemble=False,
        cv=StratifiedKFold(
            n_splits=config.calibration_folds, shuffle=True, random_state=config.seed
        ),
    )


def genuine_probabilities(model, texts):
    """Locate the genuine column from the model's actual class ordering."""
    return model.predict_proba(list(texts))[:, list(model.classes_).index(int(Label.GENUINE))]


def validate_version(value: str) -> str:
    """Keep versions usable as single safe directory names."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", value):
        raise ValueError("Version must be 1–64 letters, digits, dots, underscores or hyphens.")
    return value


def train_baselines(
    dataset_path: Path,
    output_root: Path,
    model_version: str,
    config: TrainingConfig,
    *,
    demo: bool = False,
    report_root: Path = Path("reports"),
) -> Path:
    """Publish an immutable selected artifact and auditable evaluation reports."""
    validate_version(model_version)
    destination = output_root / "baseline" / model_version
    if destination.exists():
        raise FileExistsError(f"Artifact already exists: {destination}. Choose a new version.")
    logger.info("Training baselines: version=%s demo=%s", model_version, demo)
    dataset = load_dataset(dataset_path)
    splits = split_dataset(dataset, config)
    candidates, comparison = {}, {}
    for name in BASELINES:
        model = build_pipeline(name, config)
        model.fit(splits["train"].text.tolist(), splits["train"].label)
        started = perf_counter()
        p = genuine_probabilities(model, splits["validation"].text)
        latency = (perf_counter() - started) * 1000 / len(p)
        metrics = measure(splits["validation"].label, p, config.threshold)
        metrics.update({"inference_ms_per_review": round(latency, 3), "split": "validation"})
        comparison[name] = metrics
        candidates[name] = model
    selected = max(BASELINES, key=lambda name: comparison[name][config.selection_metric])
    model = candidates[selected]
    test_metrics = measure(
        splits["test"].label, genuine_probabilities(model, splits["test"].text), config.threshold
    )
    test_metrics.update({"split": "test", "model": selected, "demo": demo})
    metadata = {
        "schema_version": 1,
        "model_type": "baseline",
        "model_name": selected,
        "version": model_version,
        "trained_at": datetime.now(UTC).isoformat(),
        "demo": demo,
        "threshold": config.threshold,
        "label_mapping": LABEL_VALUES,
        "dataset_sha256": dataset.fingerprint,
        "dataset_statistics": dataset.statistics,
        "training_config": config.model_dump(),
        "split_sizes": {name: len(part) for name, part in splits.items()},
        "selection_metric": config.selection_metric,
        "validation_comparison": comparison,
        "test_metrics": test_metrics,
        "calibration": "sigmoid_out_of_fold_train_only",
        "libraries": {name: version(name) for name in ("scikit-learn", "numpy", "joblib")},
        "python": platform.python_version(),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    # mkdir inherits Windows ACLs; mkdtemp's private ACL breaks Docker bind-mount access.
    staging = destination.parent / f".training-{uuid.uuid4().hex}"
    staging.mkdir(mode=0o755)
    try:
        joblib.dump(model, staging / "model.joblib")
        metadata["model_sha256"] = hashlib.sha256(
            (staging / "model.joblib").read_bytes()
        ).hexdigest()
        write_json(staging / "metadata.json", metadata)
        write_json(
            staging / "splits.json", {name: part.text_id.tolist() for name, part in splits.items()}
        )
        staging.rename(destination)
    except Exception:
        shutil.rmtree(staging)
        raise
    report_dir = report_root / "baseline" / model_version
    save_report(report_dir, test_metrics)
    write_json(report_dir / "validation_comparison.json", comparison)
    write_json(report_dir / "dataset_statistics.json", dataset.statistics)
    logger.info(
        "Training complete: selected=%s artifact=%s reports=%s", selected, destination, report_dir
    )
    return destination
