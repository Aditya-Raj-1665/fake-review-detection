"""Strict CSV validation, conservative normalization, and leakage-safe splits."""

import csv
import hashlib
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from review_authenticity.config import TrainingConfig
from review_authenticity.errors import DatasetValidationError
from review_authenticity.labels import LABEL_NAMES, LABEL_VALUES


def normalize_text(text: str) -> str:
    """Normalize Unicode, case, and whitespace while retaining other signals."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip().lower()


def text_id(text: str) -> str:
    """Identify equivalent text without writing reviews to split manifests."""
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


@dataclass
class Dataset:
    """Hold validated rows and descriptive statistics."""

    frame: pd.DataFrame
    statistics: dict
    fingerprint: str


def load_dataset(path: Path) -> Dataset:
    """Validate CSV rows, reject conflicting labels, and remove duplicate text."""
    try:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream, strict=True)
            columns = reader.fieldnames or []
            if not {"text", "label"}.issubset(columns):
                raise DatasetValidationError("Expected columns: text, label.")
            if len(columns) != len(set(columns)):
                raise DatasetValidationError("CSV contains duplicate column names.")
            rows = []
            for number, row in enumerate(reader, start=2):
                if None in row or any(value is None for value in row.values()):
                    raise DatasetValidationError(f"Malformed CSV record {number}: column mismatch.")
                text = row["text"]
                if not normalize_text(text):
                    raise DatasetValidationError(f"Record {number}: text is empty.")
                label = row["label"].strip().lower()
                if label not in LABEL_VALUES:
                    raise DatasetValidationError(
                        f"Record {number}: label must be genuine or deceptive."
                    )
                rows.append({"text": text, "label": LABEL_VALUES[label], "text_id": text_id(text)})
    except (OSError, UnicodeError, csv.Error) as exc:
        raise DatasetValidationError(f"Cannot read UTF-8 CSV at {path}: {exc}") from exc
    if not rows:
        raise DatasetValidationError("Dataset contains no reviews.")
    frame = pd.DataFrame(rows)
    if (frame.groupby("text_id")["label"].nunique() > 1).any():
        raise DatasetValidationError("Identical normalized text has conflicting labels.")
    original_count = len(frame)
    frame = frame.drop_duplicates("text_id").sort_values("text_id").reset_index(drop=True)
    counts = frame["label"].value_counts()
    if len(counts) != 2 or counts.min() < 10:
        raise DatasetValidationError(
            "Need at least 10 unique reviews per class for safe splitting."
        )
    lengths = frame["text"].str.len()
    fingerprint = hashlib.sha256(
        "\n".join(f"{r.text_id}:{r.label}" for r in frame.itertuples()).encode()
    ).hexdigest()
    statistics = {
        "input_rows": original_count,
        "unique_rows": len(frame),
        "duplicates_removed": original_count - len(frame),
        "class_counts": {LABEL_NAMES[label]: int(count) for label, count in counts.items()},
        "minority_fraction": float(counts.min() / len(frame)),
        "text_length": {
            "min": int(lengths.min()),
            "median": float(lengths.median()),
            "max": int(lengths.max()),
        },
    }
    return Dataset(frame, statistics, fingerprint)


def split_dataset(dataset: Dataset, config: TrainingConfig) -> dict[str, pd.DataFrame]:
    """Make deterministic stratified partitions before learning any vocabulary."""
    try:
        train_val, test = train_test_split(
            dataset.frame,
            test_size=config.test_ratio,
            stratify=dataset.frame.label,
            random_state=config.seed,
        )
        train, validation = train_test_split(
            train_val,
            test_size=config.validation_ratio / (1 - config.test_ratio),
            stratify=train_val.label,
            random_state=config.seed,
        )
    except ValueError as exc:
        raise DatasetValidationError("Split ratios need more examples of each class.") from exc
    splits = {"train": train, "validation": validation, "test": test}
    for name, part in splits.items():
        if part.label.nunique() != 2:
            raise DatasetValidationError(f"{name} split must contain both classes.")
    if train.label.value_counts().min() < config.calibration_folds:
        raise DatasetValidationError("Training split needs more rows per calibration fold.")
    return splits
