"""Tests for leakage prevention and actionable input errors."""

import csv
from pathlib import Path

import pytest

from review_authenticity.config import TrainingConfig
from review_authenticity.data import load_dataset, normalize_text, split_dataset
from review_authenticity.errors import DatasetValidationError


@pytest.mark.parametrize(
    "content,message",
    [
        ("review,label\nsomething,genuine\n", "Expected columns"),
        ("text,label\nhello,unknown\n", "label must"),
        ('text,label\n"   ",genuine\n', "text is empty"),
        ("text,label\nhello\n", "column mismatch"),
        ("text,label\nhello,genuine,extra\n", "column mismatch"),
        ("text,label\n", "no reviews"),
        ("text,label\nhello,genuine\n", "at least 10"),
        ('text,label\n"unterminated,genuine\n', "Cannot read"),
    ],
)
def test_invalid_csv(tmp_path, content, message):
    path = tmp_path / "bad.csv"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(DatasetValidationError, match=message):
        load_dataset(path)


def test_duplicate_and_conflicting_labels(tmp_path):
    original = Path("data/demo/reviews.csv").read_text(encoding="utf-8")
    rows = list(csv.DictReader(original.splitlines()))
    path = tmp_path / "reviews.csv"
    duplicate = rows[0]["text"].upper() + "  "
    path.write_text(original + f'"{duplicate}",genuine\n', encoding="utf-8")
    assert load_dataset(path).statistics["duplicates_removed"] == 1
    path.write_text(original + f'"{duplicate}",deceptive\n', encoding="utf-8")
    with pytest.raises(DatasetValidationError, match="conflicting"):
        load_dataset(path)


def test_disjoint_deterministic_splits():
    dataset = load_dataset(Path("data/demo/reviews.csv"))
    first = split_dataset(dataset, TrainingConfig())
    second = split_dataset(dataset, TrainingConfig())
    for name in first:
        assert first[name].text_id.tolist() == second[name].text_id.tolist()
        assert first[name].label.nunique() == 2
    sets = [set(frame.text_id) for frame in first.values()]
    assert sum(map(len, sets)) == len(set.union(*sets)) == 40


def test_conservative_normalization():
    assert normalize_text("  NOT\nGood!!! 🙂 <b>Hi</b> https://a.test ") == (
        "not good!!! 🙂 <b>hi</b> https://a.test"
    )
