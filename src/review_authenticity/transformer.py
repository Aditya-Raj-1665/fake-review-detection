"""Optional DistilBERT training; heavyweight imports stay out of baseline serving."""

import logging
import shutil
import uuid
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

from review_authenticity.config import TrainingConfig
from review_authenticity.data import load_dataset, split_dataset
from review_authenticity.evaluation import measure, save_report, write_json
from review_authenticity.labels import LABEL_NAMES, LABEL_ORDER, LABEL_VALUES, Label
from review_authenticity.training import validate_version

logger = logging.getLogger(__name__)
PRETRAINED_MODEL = "distilbert/distilbert-base-uncased"


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Fit one positive temperature on a reserved training calibration subset."""
    result = minimize_scalar(
        lambda t: log_loss(labels, softmax(logits / t, axis=1), labels=LABEL_ORDER),
        bounds=(0.05, 10.0),
        method="bounded",
    )
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError("Temperature calibration did not converge.")
    return float(result.x)


class TransformerPredictor:
    """Run locally saved DistilBERT weights on CUDA when available, otherwise CPU."""

    def __init__(self, path: Path, max_tokens: int | None, temperature: float):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if max_tokens is None:
            raise ValueError("Transformer artifact must declare max_tokens")
        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            path, local_files_only=True, use_safetensors=True
        ).to(self.device)
        expected = {int(key): value for key, value in LABEL_NAMES.items()}
        if self.model.config.id2label != expected:
            raise ValueError("Transformer label mapping does not match application classes")
        self.model.eval()
        self.max_tokens, self.temperature = max_tokens, temperature

    def predict(self, texts: list[str]) -> np.ndarray:
        """Predict in small batches; inputs beyond max_tokens are explicitly truncated."""
        results = []
        with self.torch.inference_mode():
            for start in range(0, len(texts), 8):
                tokens = self.tokenizer(
                    texts[start : start + 8],
                    padding=True,
                    truncation=True,
                    max_length=self.max_tokens,
                    return_tensors="pt",
                )
                logits = self.model(**tokens.to(self.device)).logits / self.temperature
                results.extend(
                    self.torch.softmax(logits, dim=-1)[:, int(Label.GENUINE)].cpu().tolist()
                )
        return np.asarray(results)


def train_transformer(
    dataset_path: Path,
    output_root: Path,
    model_version: str,
    config: TrainingConfig,
    *,
    epochs: int = 2,
    batch_size: int = 8,
    max_tokens: int = 256,
    learning_rate: float = 2e-5,
    revision: str = "main",
    report_root: Path = Path("reports"),
) -> Path:
    """Select epochs on validation, calibrate on reserved train rows, and test once."""
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, set_seed

    validate_version(model_version)
    if epochs < 1 or batch_size < 1 or not 16 <= max_tokens <= 512 or learning_rate <= 0:
        raise ValueError("Need positive epochs/batch size/rate and 16 <= max_tokens <= 512.")
    destination = output_root / "distilbert" / model_version
    if destination.exists():
        raise FileExistsError(f"Artifact already exists: {destination}")
    dataset = load_dataset(dataset_path)
    splits = split_dataset(dataset, config)
    train, calibration = train_test_split(
        splits["train"],
        test_size=0.15,
        stratify=splits["train"].label,
        random_state=config.seed,
    )
    set_seed(config.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL, revision=revision)
    model = AutoModelForSequenceClassification.from_pretrained(
        PRETRAINED_MODEL,
        revision=revision,
        num_labels=2,
        id2label={int(k): v for k, v in LABEL_NAMES.items()},
        label2id=LABEL_VALUES,
        use_safetensors=True,
    ).to(device)
    resolved_revision = model.config._commit_hash
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    def batches(frame, shuffle=False):
        """Tokenize raw text with the model tokenizer only."""
        rows = list(zip(frame.text, frame.label, strict=True))

        def collate(items):
            """Pad each batch and attach its supervised labels."""
            tokens = tokenizer(
                [item[0] for item in items],
                padding=True,
                truncation=True,
                max_length=max_tokens,
                return_tensors="pt",
            )
            tokens["labels"] = torch.tensor([item[1] for item in items], dtype=torch.long)
            return {key: value.to(device) for key, value in tokens.items()}

        return DataLoader(
            rows,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate,
            generator=torch.Generator().manual_seed(config.seed),
        )

    def get_logits(frame):
        """Evaluate the current checkpoint without gradient tracking."""
        model.eval()
        values = []
        with torch.inference_mode():
            for batch in batches(frame):
                batch.pop("labels")
                values.append(model(**batch).logits.cpu().numpy())
        return np.concatenate(values)

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".training-{uuid.uuid4().hex}"
    staging.mkdir(mode=0o755)
    history, best_score = [], -1.0
    try:
        for epoch in range(epochs):
            logger.info("DistilBERT training: epoch=%s device=%s", epoch + 1, device)
            model.train()
            for batch in batches(train, shuffle=True):
                optimizer.zero_grad(set_to_none=True)
                loss = model(**batch).loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            metrics = measure(
                splits["validation"].label,
                softmax(get_logits(splits["validation"]), axis=1)[:, int(Label.GENUINE)],
                config.threshold,
            )
            history.append({"epoch": epoch + 1, **metrics})
            if metrics[config.selection_metric] > best_score:
                best_score = metrics[config.selection_metric]
                model.save_pretrained(staging, safe_serialization=True)
                tokenizer.save_pretrained(staging)
        # Only the selected checkpoint reaches calibration and the final test set.
        model = AutoModelForSequenceClassification.from_pretrained(
            staging, local_files_only=True, use_safetensors=True
        ).to(device)
        temperature = fit_temperature(get_logits(calibration), calibration.label.to_numpy())
        started = perf_counter()
        validation_logits = get_logits(splits["validation"])
        elapsed = (perf_counter() - started) * 1000 / len(validation_logits)
        validation_metrics = measure(
            splits["validation"].label,
            softmax(validation_logits / temperature, axis=1)[:, int(Label.GENUINE)],
            config.threshold,
        )
        validation_metrics.update(
            {"inference_ms_per_review": round(elapsed, 3), "split": "validation"}
        )
        test_metrics = measure(
            splits["test"].label,
            softmax(get_logits(splits["test"]) / temperature, axis=1)[:, int(Label.GENUINE)],
            config.threshold,
        )
        test_metrics.update({"split": "test", "model": "distilbert", "demo": False})
        metadata = {
            "schema_version": 1,
            "model_type": "distilbert",
            "model_name": "distilbert",
            "version": model_version,
            "trained_at": datetime.now(UTC).isoformat(),
            "demo": False,
            "threshold": config.threshold,
            "label_mapping": LABEL_VALUES,
            "calibration": "temperature_reserved_train_subset",
            "temperature": temperature,
            "max_tokens": max_tokens,
            "pretrained_model": PRETRAINED_MODEL,
            "requested_revision": revision,
            "resolved_revision": resolved_revision,
            "dataset_sha256": dataset.fingerprint,
            "dataset_statistics": dataset.statistics,
            "training_config": config.model_dump(),
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "calibration_rows": len(calibration),
            "split_sizes": {key: len(value) for key, value in splits.items()},
            "validation_comparison": {"distilbert": validation_metrics},
            "test_metrics": test_metrics,
            "epoch_history": history,
            "libraries": {key: version(key) for key in ("torch", "transformers")},
            "device": str(device),
        }
        write_json(staging / "metadata.json", metadata)
        write_json(
            staging / "splits.json",
            {
                **{key: value.text_id.tolist() for key, value in splits.items()},
                "calibration": calibration.text_id.tolist(),
            },
        )
        staging.rename(destination)
    except Exception:
        shutil.rmtree(staging)
        raise
    save_report(report_root / "distilbert" / model_version, test_metrics)
    return destination
