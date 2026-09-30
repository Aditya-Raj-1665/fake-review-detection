"""Explicit commands for data, training, and reports; never invoked by API startup."""

import argparse
import json
import sys
from pathlib import Path

from review_authenticity.config import TrainingConfig
from review_authenticity.data import load_dataset
from review_authenticity.download import download_cornell
from review_authenticity.evaluation import save_report, write_json
from review_authenticity.inference import read_metadata
from review_authenticity.logging_config import configure_logging
from review_authenticity.training import train_baselines


def compare_artifacts(paths: list[Path], output: Path) -> None:
    """Compare validation results only when the dataset and split identities match."""
    records, reference = [], None
    for path in paths:
        metadata = read_metadata(path)
        splits = json.loads((path / "splits.json").read_text(encoding="utf-8"))
        signature = (
            metadata["dataset_sha256"],
            sorted(splits["validation"]),
            sorted(splits["test"]),
        )
        if reference is not None and signature != reference:
            raise ValueError("Cannot compare models trained with different datasets or splits.")
        reference = signature
        for name, metrics in metadata["validation_comparison"].items():
            records.append(
                {"model": name, "version": metadata["version"], "demo": metadata["demo"], **metrics}
            )
    write_json(output.with_suffix(".json"), {"split": "validation", "models": records})
    lines = [
        "# Validation comparison",
        "",
        "Latency is batch wall time per review; compare on the same hardware.",
        "",
        "| Model | Version | F1 | Precision | Recall | ms/review | Demo |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for item in records:
        lines.append(
            f"| {item['model']} | {item['version']} | {item['f1']:.3f} | "
            f"{item['precision']:.3f} | {item['recall']:.3f} | "
            f"{item['inference_ms_per_review']:.3f} | {item['demo']} |"
        )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Dispatch validated command-line arguments to the shared ML package."""
    parser = argparse.ArgumentParser(description="Review authenticity training and reporting")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("train-baseline", "train-transformer"):
        train = commands.add_parser(name)
        train.add_argument("--dataset", type=Path, required=True)
        train.add_argument("--config", type=Path, default=Path("configs/baseline.json"))
        train.add_argument("--version", required=True)
        train.add_argument("--output", type=Path, default=Path("artifacts"))
        train.add_argument("--reports", type=Path, default=Path("reports"))
        if name == "train-baseline":
            train.add_argument("--demo", action="store_true")
        else:
            train.add_argument("--epochs", type=int, default=2)
            train.add_argument("--batch-size", type=int, default=8)
            train.add_argument("--max-tokens", type=int, default=256)
            train.add_argument("--learning-rate", type=float, default=2e-5)
            train.add_argument("--revision", default="main")
    download = commands.add_parser("download-cornell")
    download.add_argument("--output", type=Path, default=Path("data/raw/cornell.csv"))
    download.add_argument("--archive", type=Path)
    inspect = commands.add_parser("inspect-data")
    inspect.add_argument("--dataset", type=Path, required=True)
    inspect.add_argument("--output", type=Path, default=Path("reports/dataset_statistics.json"))
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--artifact", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, default=Path("reports/evaluation"))
    compare = commands.add_parser("compare")
    compare.add_argument("artifacts", type=Path, nargs="+")
    compare.add_argument("--output", type=Path, default=Path("reports/comparison.md"))
    args = parser.parse_args()
    configure_logging()
    try:
        if args.command.startswith("train-"):
            config = TrainingConfig.model_validate_json(args.config.read_text(encoding="utf-8"))
            if args.command == "train-baseline":
                demo = args.demo or "demo" in args.dataset.parts
                path = train_baselines(
                    args.dataset,
                    args.output,
                    args.version,
                    config,
                    demo=demo,
                    report_root=args.reports,
                )
            else:
                from review_authenticity.transformer import train_transformer

                path = train_transformer(
                    args.dataset,
                    args.output,
                    args.version,
                    config,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    max_tokens=args.max_tokens,
                    learning_rate=args.learning_rate,
                    revision=args.revision,
                    report_root=args.reports,
                )
            print(f"Artifact created: {path}")
        elif args.command == "download-cornell":
            print(download_cornell(args.output, args.archive))
        elif args.command == "inspect-data":
            dataset = load_dataset(args.dataset)
            write_json(args.output, {**dataset.statistics, "sha256": dataset.fingerprint})
            print(args.output)
        elif args.command == "evaluate":
            # Re-render the original held-out result, rather than repeatedly querying test data.
            save_report(args.output, read_metadata(args.artifact)["test_metrics"])
            print(f"Exported original held-out evaluation: {args.output}")
        elif args.command == "compare":
            compare_artifacts(args.artifacts, args.output)
            print(args.output)
    except (ValueError, OSError, RuntimeError, ImportError) as exc:
        parser.exit(1, f"{type(exc).__name__}: {exc}\n")


if __name__ == "__main__":
    sys.exit(main())
