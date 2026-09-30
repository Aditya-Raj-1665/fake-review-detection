"""Optional, checksum-pinned import of the public Cornell corpus mirror."""

import csv
import hashlib
import io
import urllib.request
import zipfile
from pathlib import Path

from review_authenticity.evaluation import write_json

CORPUS_URL = "https://www.kaggle.com/api/v1/datasets/download/rtatman/deceptive-opinion-spam-corpus"
CORPUS_SHA256 = "f93674ef50e365886aeef5e651ac8990e8430d36e439ae1ca5fa6962a76a0a8e"


def download_cornell(destination: Path, archive: Path | None = None) -> Path:
    """Import a verified archive without extracting untrusted archive paths."""
    if destination.exists():
        raise FileExistsError(f"Dataset already exists: {destination}")
    if archive is None:
        with urllib.request.urlopen(CORPUS_URL, timeout=60) as response:
            content = response.read(10_000_001)
    else:
        content = archive.read_bytes()
    if hashlib.sha256(content).hexdigest() != CORPUS_SHA256:
        raise ValueError("Dataset checksum changed. Inspect the upstream release before importing.")
    with zipfile.ZipFile(io.BytesIO(content)) as zipped:
        source = csv.DictReader(io.StringIO(zipped.read("deceptive-opinion.csv").decode("utf-8")))
        rows = []
        for row in source:
            label = {"truthful": "genuine", "deceptive": "deceptive"}[row["deceptive"]]
            rows.append(
                {
                    "text": row["text"],
                    "label": label,
                    "hotel": row["hotel"],
                    "polarity": row["polarity"],
                    "source": row["source"],
                }
            )
    if len(rows) != 1600:
        raise ValueError("Expected 1,600 reviews in this pinned release.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(
        destination.with_suffix(".source.json"),
        {
            "dataset": "Cornell Deceptive Opinion Spam Corpus v1.4 (Kaggle mirror)",
            "url": CORPUS_URL,
            "archive_sha256": CORPUS_SHA256,
            "rows": len(rows),
            "research": [
                "https://aclanthology.org/P11-1032/",
                "https://aclanthology.org/N13-1053/",
            ],
            "notice": "Research corpus; upstream rights apply. Raw data is not committed.",
        },
    )
    return destination
