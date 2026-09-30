# Cornell baseline model card

## Intended use

Portfolio/educational experimentation with English hotel-review text. Predictions
must not be used as definitive evidence against a reviewer or business. Other product
domains, languages, current AI-written reviews, and population prevalence require
separate validation.

## Executed experiment

Version `cornell-v1`, trained on 2026-09-25 with the locked Python environment and
`configs/baseline.json`. The real run (not synthetic demo) ingested 1,600 reviews,
removed four normalized duplicates, and retained 800 deceptive and 796 genuine reviews.
Seed 42; 1,116 training, 240 validation, and 240 test rows. Dataset fingerprint:

```text
e16952444fffdf9b20b1744215bf7c894151a5cf3f3ad09aa1e7007f6b48fc54
```

Both models used unigram/bigram TF-IDF with punctuation retained, up to 30,000 features,
and training-only sigmoid calibration with three folds. Deceptive is the positive class.

| Validation model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.9083 | 0.9153 | 0.9000 | 0.9076 |
| Multinomial Naive Bayes | 0.9042 | 0.8880 | 0.9250 | 0.9061 |

Logistic Regression won by validation F1, by a very small margin. This is not evidence of
a statistically significant advantage; no confidence interval or repeated-fold comparison
was performed. The selected model was evaluated once on the test split:

| Test metric | Observed value |
|---|---:|
| Accuracy | 0.8958 |
| Deceptive precision | 0.8926 |
| Deceptive recall | 0.9000 |
| Deceptive F1 | 0.8963 |
| Brier score | 0.0774 |
| Log loss | 0.2548 |
| Genuine reviews incorrectly flagged | 13 |
| Deceptive reviews missed | 12 |

Actual rows × predicted columns, order `[deceptive, genuine]`: `[[108, 12], [13, 107]]`.
The raw computed outputs and metadata snapshot are in [results](results/cornell-v1/metadata.json).
No transformer was trained or benchmarked. Do not substitute the paper's scores or
synthetic-demo scores for these observations.

## Evaluation limits

This is a review-level random stratified holdout, not the original papers' evaluation
protocol and not a hotel-grouped, author-grouped, temporal, or external-domain holdout.
Source style and hotel identity can appear in multiple splits. Near-duplicates may remain.
Crowd-worker deception and historically collected review text do not represent all modern
fraud. Calibration was fitted without test data, but reliability under distribution shift
has not been demonstrated. A balanced benchmark differs from a low-fraud review stream.

The model's default threshold is 0.5. Scores mean estimated `P(genuine)` on patterns learned
from this corpus. LR feature signals refer to its pre-calibration score and are associations.

## Reproduce

Download with `reviews download-cornell`, then train with a fresh version using
`reviews train-baseline --dataset data/raw/cornell.csv --version your-version`.
Use `uv sync --locked`, keep the config unchanged, and compare dataset/split fingerprints.
Per-review latency recorded in metadata is batched validation prediction wall time on
the training machine; it is not an HTTP latency or cross-machine performance promise.
