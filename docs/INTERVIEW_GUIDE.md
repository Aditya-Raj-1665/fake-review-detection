# Interview guide

## A 60-second explanation

“I built a supervised NLP pipeline that estimates review authenticity from text. I
validated a public hotel-review dataset, removed normalized duplicates, and made fixed
stratified training, validation, and test splits. I compared calibrated TF-IDF Logistic
Regression and Naive Bayes on validation F1, then evaluated only the winner on the held-out
test set. I saved the model with its configuration, dataset fingerprint, and metrics.
FastAPI loads that artifact once and exposes single/batch inference; a small React page
shows probabilities and limitations. Docker separates explicit training from serving.
I implemented an optional DistilBERT pipeline, but did not train or benchmark it in the
baseline run. This is a research classifier, not proof that an individual review is fake.”

Adapt this explanation to what you have personally studied and reproduced. Do not claim
to have authored, tested, or understood components you cannot explain.

## Architecture and deployment

React → nginx API proxy → FastAPI → inference service → trusted local artifact. Offline:
CSV → validation/deduplication → split → calibrated models → validation selection → test
evaluation → artifact. The same package owns train and inference preprocessing.

**Why FastAPI?** Typed request/response validation, generated OpenAPI docs, lifespan
resource ownership, and a simple ASGI deployment fit a small stateless model server.
It is not the reason the model performs well; data and evaluation design determine that.

**How is the model loaded?** Once per worker at lifespan startup. Metadata, checksum,
class mapping, and scikit-learn version are checked. A missing model stops startup with
an actionable command. No training happens on a prediction or health request.

**Why one worker?** Every worker holds its own model. Extra copies of a transformer can
exhaust memory. Baseline requests use FastAPI's thread pool; transformer calls are locked
per worker. Docker serves compiled frontend assets through nginx, not Vite's dev server.

## Dataset and preprocessing

The real dataset has 1,600 English positive/negative hotel reviews from the Cornell
research corpus mirror. Crowdsourced deceptive text and collected truthful reviews have
different provenance. Four normalized duplicates were removed in the recorded run.
See the dataset/model cards for exact counts and caveats.

**Preprocessing:** Unicode NFKC, lowercase, and whitespace normalization. No stopword,
punctuation, URL, HTML, or emoji removal. Negation is useful: “not good” differs from “good.”
TF-IDF tokenization captures word and punctuation tokens with unigram/bigram features.
Transformers use their own tokenizer on raw text, not the TF-IDF preprocessing function.

**TF-IDF:** Term frequency says how present a token is in a document; inverse document
frequency downweights tokens that occur across many documents. Here sublinear TF reduces
the effect of repetition. Vocabulary and IDF must be learned only from training folds.

## Models

**Logistic Regression:** A weighted sum of sparse TF-IDF features gives a decision score.
The logistic link estimates a class probability; regularization controls coefficient
size. Our `C` config controls inverse regularization strength. Balanced class weights
change the training loss, not the held-out class distribution.

**Multinomial Naive Bayes:** Combines class priors with feature likelihoods under a
conditional independence assumption. That assumption is approximate for text, but the
model is fast and often useful. Smoothing (`alpha`) avoids zero probabilities. TF-IDF
features are nonnegative, which this estimator requires. Raw NB probabilities can be
overconfident; we calibrate them before serving.

**DistilBERT:** A smaller pretrained transformer that contextualizes tokens using attention.
Fine-tuning adds a classification objective. Our optional code selects epochs on validation,
reserves training data for temperature calibration, saves model/tokenizer, and uses
CPU/CUDA inference without gradients. It truncates to the configured token limit. No
DistilBERT performance or latency claim exists until that pipeline is actually run.

## Metrics and class imbalance

Deceptive is the positive class for reporting, even though its numeric code is 0.

| Term | Meaning here |
|---|---|
| True positive | Deceptive review correctly flagged |
| False positive | Genuine review incorrectly flagged deceptive |
| False negative | Deceptive review incorrectly called genuine |
| True negative | Genuine review correctly called genuine |
| Precision | TP / (TP + FP): how trustworthy a deceptive flag is |
| Recall | TP / (TP + FN): fraction of deceptive reviews caught |
| F1 | 2 × precision × recall / (precision + recall) |
| Accuracy | Fraction of all labels correct |
| Macro F1 | Average class F1 giving each class equal weight |

The confusion matrix fixes row/column order as `[deceptive, genuine]`. Accuracy can hide
failure on a rare class. Report class counts and precision/recall/F1 together. This corpus
is nearly balanced after deduplication; many real review streams are not. No resampling
is implemented. If later added, it belongs inside the training fold, never on held-out data.

## Splits, leakage, and reproducibility

Training fits model parameters. Validation chooses model/configuration/checkpoint. Test
estimates final held-out performance. Our default is about 70/15/15, stratified by label,
with a fixed seed. Sorting hashes makes row-order changes irrelevant.

Leakage can mean identical text in train and test, vocabulary fitted on all rows, calibration
using test labels, or choosing a model by test performance. The code prevents these cases.
It does **not** prevent all near-duplicate, author, hotel, or temporal leakage. This is why
the recorded result is explicitly a review-level benchmark, not unseen-hotel validation.

Locks, saved configs, dataset fingerprints, split identities, and immutable artifact versions
support reproducibility. Hardware/library differences can still change neural computation.
GPU determinism warnings are not a guarantee of bit-identical training.

## Probability calibration and interpretation

Calibration asks whether predictions around 70% are correct around 70% of the time on
similar data. Ranking performance and calibration are different. Our classical models
generate out-of-fold training scores for sigmoid calibration, then refit the underlying
pipeline on the training partition. Validation/test rows are not calibration data.
The transformer reserves a separate portion of training for a scalar temperature.

We report Brier score and log loss, but have not demonstrated universal calibration or
measured calibration improvement against every uncalibrated alternative. Class-prevalence
and domain shifts can break reliability. The UI's authenticity score is `100 × P(genuine)`;
confidence is probability of the threshold-selected class. Threshold changes trade off
false positives and false negatives and should be chosen before looking at final test results.

LR signals are feature value × learned coefficient contributions before the calibration
sigmoid. A high-weight phrase shows what the model learned, not why the author wrote it
and not evidence that the phrase itself proves fraud.

## Limitations and scaling

Source/domain bias, short text, sarcasm, AI-generated text, hidden duplicates, and
unverified truthful labels remain. Only English and text features are supported.
The local application has no auth because it stores no accounts or predictions.
No submitted text is deliberately logged. An untrusted pickle can execute code, so the
API never accepts model uploads and only operators set the local artifact path.

For higher load, measure p95 latency, memory, and throughput first. Classical models may
scale with additional replicas/workers; transformer batching and GPU memory need different
treatment. TLS/rate controls belong at the public hosting edge. A database or queue should
be added only when a concrete requirement (history, long-running jobs) justifies it.

## Follow-up questions to practice

1. Why can an old hotel-review benchmark overestimate modern product-review performance?
2. What would a hotel-grouped split change, and how would you preserve class balance?
3. Why calibrate inside training folds rather than reuse the model-selection validation set?
4. How do precision/recall change when the genuine threshold rises?
5. Why is `confidence` not always `max(probabilities)` with a custom threshold?
6. Why keep stop words and punctuation? What ablation would test that decision?
7. Where is the vectorizer fitted, and why must it be inside the calibration CV estimator?
8. How do you know the saved model is the one that produced the reported test score?
9. Why doesn't repeated test evaluation provide independent evidence?
10. How would you detect drift without having immediate labels for new reviews?
11. What experiment would justify paying DistilBERT's latency and memory cost?
12. What can a checksum protect against, and what can it not protect against?
13. Which tests are meaningful, and which optional paths remain unexecuted?
14. How would you move from probabilistic assistance to a responsibly evaluated moderation system?
