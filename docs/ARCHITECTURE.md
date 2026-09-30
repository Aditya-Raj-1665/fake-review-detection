# Architecture

## Components

The Python package is the single source of truth. `data.py` validates and normalizes
data; `labels.py` owns label encoding. `training.py` fits calibrated classical pipelines;
`transformer.py` provides an optional independent HF/PyTorch pipeline. `evaluation.py`
owns metrics and report rendering. `inference.py` loads trusted artifacts; `api.py`
validates requests and delegates predictions. `cli.py` performs explicit offline work.

The React form fetches model info, posts a review, and renders uncertainty, errors, and
model provenance. nginx serves built static files and proxies API requests. Compose runs
only frontend/backend normally; the `tools` profile provides a short-lived trainer.

## Training lifecycle

1. Strictly read CSV; reject missing/invalid records and conflicting text labels.
2. Normalize only for duplicate keys; preserve original text for transformers.
3. Sort hashes, deduplicate, fingerprint the corpus, and stratify with the saved seed.
4. Classical models fit vocabulary and classifier within training calibration folds.
   Out-of-fold scores fit a sigmoid; the final base pipeline fits the full training set.
5. Compare validation deceptive-class F1 (configurable), resolve ties deterministically.
6. Evaluate the selected unchanged model on test once. Save metadata, split identities,
   model checksum, reports, and figure. Rename a staging directory into a new version.

DistilBERT shares outer splits, reserves part of training for temperature fitting,
chooses the checkpoint by validation performance, and tests only that checkpoint after
calibration. The optional comparison tool checks dataset and held-out split identities.

## Serving lifecycle

Environment → typed Settings → FastAPI lifespan → metadata validation → model load →
ready health endpoint. Missing/incompatible artifacts fail startup with a training command.
Each worker owns one model. There is no global singleton shared between application objects,
no startup training, no database, and no external model calls during inference.

Request → bounded body buffer → typed JSON validation → text/count checks → model
probability → configured threshold → typed response. Review text is never deliberately
logged. Operational errors are logged server-side; clients get short safe messages.

## Trade-offs

* A single Python package keeps train/serve preprocessing consistent and avoids packaging
  two competing ML implementations.
* The baseline image includes report libraries so the same image can run the trainer.
  PyTorch/Transformers stay optional to keep normal startup lightweight.
* Joblib is simple but unsafe for untrusted files. Only operator-configured local artifacts
  are accepted; checksum checks detect accidental damage, not malicious replacements.
* One process conserves model memory. Baselines allow thread-pool concurrent inference;
  transformer calls use a lock and small batches. There is no distributed queue.
* A random stratified review split is a reproducible initial benchmark, not a measure of
  generalization to unseen hotels/authors. Grouped/domain holdouts are future work.
* Reports can be re-exported without rerunning test inference. Comparing validation results
  avoids selecting on test, but repeatedly tuning to validation can still overfit it.
* Dependency locks pin package resolutions; major/minor container tags can still receive
  image rebuilds. Record image digests for byte-identical long-term deployment provenance.
