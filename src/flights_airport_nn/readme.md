# Airport prediction from segmented ADS-B flights

**Recommended broad checkpoint:** `runs/2026-09-17-broad-finetune-v1/selected_model.pt` improves on nearest airport across large, medium and small airport groups on fresh June 5–6, 2026 flights. The earlier commercial-only checkpoint is evaluated separately below.

A shared GRU ranks airport candidates for **takeoff and landing**. It uses relative
geometry, altitude above airport elevation, ground speed, vertical rate, heading,
coverage and missing-value indicators, airport size and scheduled service. It does
not use airport IDs, aircraft IDs, callsigns, absolute coordinates, routes, or label
frequencies as model inputs.

The model combines the endpoint observation, a recurrent trajectory encoding and
attention pooling, with a learned correction to a distance prior. Each endpoint
is trained with a softmax loss over its candidate airports. AdamW, dropout,
gradient clipping, coverage augmentation and early stopping limit overfitting.
The commercial baseline selects its checkpoint by validation loss. The broader
run uses the size-balanced validation rule described below. Validation data also
fits one confidence temperature. Test data is never used for either operation.

## Setup

Use the repository virtual environment with PyTorch installed (CUDA is optional):

```bash
export OUTPUT_DIR=/mnt/storage1/planequery-data
export PYTHONPATH=src
export POLARS_MAX_THREADS=4
```

`OUTPUT_DIR` is the parent of `data`, not the data directory itself. Training reads
local cached ground-truth flights, algorithm segments and ADS-B parquet files.
It does not rebuild labels or download missing source data. Feature caches live in
`data/processed/flights_airport_nn/v2/` and contain no Python pickle objects.
Source file metadata, feature source code metadata and retrieval settings validate
cache reuse. Changing the feature representation requires a feature-version bump.

## Train and evaluate

Pass explicit, disjoint dates. Use several weeks for training and later dates for
validation and testing. This short example exercises the complete workflow:

```bash
.venv/bin/python -m flights_airport_nn.experiment \
  --train-dates 2026-02-01 2026-02-02 2026-02-03 \
  --validation-dates 2026-02-07 2026-02-08 \
  --test-dates 2026-03-01 2026-03-02 \
  --epochs 20 --patience 4 \
  --output-dir /mnt/storage1/planequery-data/data/models/flights_airport_nn/runs/my-run
```

`--prepare-only` builds caches without fitting or evaluating a model. A run writes
`config.json`, `history.json`, `model.pt`, validation metrics/predictions and, when
test dates are supplied, test metrics/predictions. Existing model files are never
silently overwritten. Inspect individual mistakes in the prediction parquet
files. Accuracy includes labeled endpoints for which candidate retrieval fails.
Reports include nearest-airport and top-three accuracy, candidate recall, macro
accuracy over airports, confidence calibration, takeoff/landing results, and
unseen aircraft/airport subsets. Empty subsets report zero examples rather than
inventing an accuracy.

## Predict without labels

```python
from flights_airport_nn.features.features import create_features
from flights_airport_nn.inference import load_model, predict_endpoints, inference

model, config = load_model('/path/to/run/model.pt')
features = create_features(segmented_flights_df, adsb_messages_df)
predictions = predict_endpoints(features, model, config, top_k=3)
# Or populate both airport fields on the Flight objects:
flights_by_id = inference(features, model, config)
```

The flight dataframe uses the repository's `Flight` schema. Airport identifiers
may be null. ADS-B requires `icao`, `time`, `lat`, and `lon`; optional motion and
altitude values can be missing. `predict_endpoints` returns airport, conditional
confidence and alternatives for each endpoint. If no usable observations or
candidates exist, it returns `airport_ident=None` and zero confidence. `inference`
also clears that airport field. `predict` returns ranking logits, not independent
binary probabilities. Checkpoints from the previous four-feature prototype are
rejected because their preprocessing is incompatible.

## Data and generalization boundaries

* Observations stay inside the supplied segment and within ten minutes of the
  relevant endpoint. Up to 32 points are sampled, with extra detail near the
  endpoint. Landing sequences run backward from arrival.
* Candidates are selected from airports within 200 km of the first endpoint
  observation: 24 nearest fields, then up to eight additional medium/large fields,
  filling remaining spaces by distance. The correct label is **never injected**.
  The airport catalog currently covers small, medium and large airports, excluding
  heliports and seaplane bases.
* Ground-truth airport labels are joined onto segmented times using the existing
  temporal flight matcher with airport comparison disabled. Only matched segments
  are supervised. This evaluates airport assignment, not segmentation quality.
  The commercial BTS/SFDPS ground-truth builder also checks ADS-B proximity to
  labeled airports and flight continuity. This selects a cleaner commercial
  population. The broader FAA evaluation uses its own time-only match and
  includes a separate reception-loss check. Neither score measures all raw
  segments.
* The commercial baseline uses BTS/SFDPS agreement; the broader run uses
  single-source FAA SFDPS labels. Random endpoint splits would be misleading
  because routes and aircraft repeat. Keep chronological validation/test splits
  and report unfamiliar aircraft/airports. Even good held-out results do not
  establish worldwide or general-aviation accuracy; those need independent
  labels before making that claim.
* Confidence is conditional on candidate retrieval and the labeling population.
  Test confidence calibration alongside accuracy. A distant/missing endpoint,
  novel airport type or unrepresented flight population can change reliability.
* Reported binomial intervals treat endpoints as independent. They do not capture
  correlation between the two ends of a flight or repeated aircraft/routes.

Run the focused checks with:

```bash
.venv/bin/python -m pytest tests/flights_airport_nn -q
```

## Audit airports excluded from supervision

Use the separate audit CLI to hold out a deterministic airport fold. With five
folds, the default excludes roughly 20% of airport identities from **positive
training labels, validation checkpoint selection and temperature calibration**.
Those fields remain in the geographic candidate catalog; this measures transfer
of a shared scorer rather than discovering airports missing from the catalog.

```bash
.venv/bin/python -m flights_airport_nn.generalization \
  --train-dates 2026-02-01 2026-02-02 2026-02-03 2026-02-04 2026-02-05 2026-02-06 \
  --validation-dates 2026-02-07 2026-02-08 \
  --airport-folds 5 --holdout-fold 0 \
  --output-dir /mnt/storage1/planequery-data/data/models/flights_airport_nn/runs/airport-audit
```

A successful audit within this commercial population still does not measure
transfer to general aviation. Both CLIs support `--cache-workers` for parallel day
preparation; each worker can use several GB of memory. Day inference supports
`aircraft_batch_size` and batches endpoint scores to limit memory use.

To evaluate a checkpoint on additional untouched dates:

```bash
.venv/bin/python -m flights_airport_nn.evaluate \
  --model /path/to/run/model.pt --dates 2026-03-03 \
  --output-dir /path/to/new-evaluation
```

This evaluator uses the commercial BTS/SFDPS cache format. For the broader
FAA checkpoint, use `flights_airport_nn.stress_broad` as shown below. Evaluators
reject training/validation dates. Repeatedly selecting models from the same
test results turns that test into validation; reserve new dates for the next
final assessment.

Generate an exportable learning-curve figure, error CSV, and Markdown report after
testing:

```bash
.venv/bin/python -m flights_airport_nn.report \
  --run-dir /path/to/run --audit-dir /path/to/airport-audit
```

## Measured run: 2026-09-17

Training used February 1–24 (137,888 endpoints), validation used February 25–28
(20,386 endpoints), and the final test used March 1–8 (43,074 endpoints). There
were no shared flight IDs between these splits. Validation loss selected epoch 2;
early stopping ended training after epoch 6. No changes were selected using the
test results.

| Final test group | Endpoints | Neural network | Nearest airport |
|---|---:|---:|---:|
| All | 43,074 | 99.9350% | 98.3494% |
| Takeoff | 21,537 | 99.9536% | 99.0853% |
| Landing | 21,537 | 99.9164% | 97.6134% |
| Aircraft absent from training | 2,494 | 99.9599% | 97.5140% |

There were 28 test errors, including two candidate retrieval misses, compared with
711 nearest-airport errors. A separate audit excluded 44 airport identities from
positive training labels, model selection and calibration: 99.9230% accuracy on
2,599 endpoints, versus 98.1916% for nearest-airport assignment.

These are **filtered US commercial-flight results**, with the population and
coverage limitations described above. They are not worldwide or general-aviation
accuracy estimates. Only 12 main-test endpoints used airports absent from training;
the separate airport holdout audit provides the more substantial transfer check.

Artifacts are under
`$OUTPUT_DIR/data/models/flights_airport_nn/runs/2026-09-17-generalization/`:
`model.pt`, `results.md`, `evaluation.png`, `evaluation.svg`, `test_metrics.json`,
`test_predictions.parquet`, `test_errors.csv`, and `split_audit.json`.
The separate audit is in the sibling `2026-09-17-airport-holdout-audit/` directory.

## Broader frozen-checkpoint assessment

Use single-source FAA SFDPS flights to include smaller airports and registration
callsigns outside the commercial BTS/FAA agreement set:

```bash
.venv/bin/python -m flights_airport_nn.broader_evaluation \
  --model /path/to/run/model.pt \
  --dates 2026-03-15 2026-04-01 2026-06-01 \
  --output-dir /path/to/broader-evaluation
.venv/bin/python -m flights_airport_nn.error_audit \
  --evaluation-dir /path/to/broader-evaluation
```

Matching requires a unique aircraft/time pair within 15 minutes at both segment
boundaries. It does not require airport agreement or nearby positions. Airport
labels are removed before feature construction. The evaluator rejects the original
training, validation and recorded test dates and checks that the checkpoint hash
has not changed. A deterministic 10% flight sample additionally loses the first
and last two or five minutes of reception; candidates are rebuilt for each case.
All stress comparisons use the same flights and retain unavailable observations
as failures.

These are **agreement scores with single-source labels**. Registration callsigns
are a proxy, not a verified general-aviation classification. Unresolvable airport
codes are counted separately. The disagreement audit retains raw ADS-B evidence,
cached FAA labels and temporal matching metadata, and assigns conservative,
tentative evidence categories. It never relabels the test or reports corrected
accuracy. A missing nearby observation does not establish a flawed label.

The existing checkpoint's training labels contain no small-airport endpoints.
Airport identity holdouts within its commercial population do not address that
gap. Dates used in this broader assessment must become development data if the
findings inform the next model; reserve other dates for its final test.

### Frozen checkpoint results on broader flights

The March 15, April 1, and June 1, 2026 evaluation matched 46,605 segmented
flights to FAA SFDPS records by aircraft and time. There were 92,872 endpoints
with resolvable airport labels; 338 unresolved endpoints were counted separately.
The original network checkpoint, feature extraction, candidate retrieval and
confidence calibration were unchanged.

| FAA label cohort | Endpoints | Network | Nearest airport | Candidate recall |
|---|---:|---:|---:|---:|
| All resolved | 92,872 | 87.43% | 94.04% | 99.68% |
| Large airport | 65,351 | 99.53% | 97.73% | 99.89% |
| Medium airport | 20,544 | 78.57% | 85.57% | 99.66% |
| Small airport | 6,969 | 0.20% | 84.52% | 97.93% |
| Registration callsign proxy | 17,470 | 53.30% | 86.49% | 98.88% |
| Small airport with a nearby, low ADS-B observation | 3,512 | 0.06% | 99.80% | 100% |

The network predicted a small airport only 14 times among 92,872 resolved
endpoints. Its training labels included **no small-airport endpoints**, even
though candidate retrieval supplied the correct small airport for 97.93% of
small-airport evaluation endpoints. This is a ranking failure and a severe
population shift. The nearby, low-observation row is a conditional
corroboration check, not a replacement for overall accuracy.

The same 9,134 endpoints were evaluated with 0, 2, and 5 minutes of reception
removed at both ends. Network agreement was 87.62%, 83.46%, and 70.37%; nearest
airport agreement was 93.85%, 55.15%, and 8.89%. The network handles missing
near-terminal reception better than the nearest-observation rule but remains
unreliable for small airports.

Independent airline-reported BTS records matched 33,538 endpoints on March 15
and April 1. They agreed with the FAA labels in all matched cases and supported
the FAA label in all 77 network disagreements in that overlap. This cross-check
covers commercial flights and does not validate the small-airport population.
The disagreement audit applies track evidence symmetrically to the label and
prediction and keeps insufficient cases unresolved. Of 11,673 disagreements,
1,702 had sustained evidence for the labeled airport, 158 suggested a label
or segmentation problem, one had conflicting evidence, and 9,812 were
unresolved. It preserves the original agreement scores. Cached FAA labels can have code-resolution and segmentation
errors; the `audit/` artifacts document suspected cases. Original FAA messages
were locally available for a partial June 1 sample only.

Complete metrics, per-endpoint predictions, original disagreements, paired
reception-loss results, source cross-checks and plots are in
`$OUTPUT_DIR/data/experiments/flights_airport_nn/broader-2026-09-17-complete/`.
Any model changes informed by these findings need a new, unused test period.

## Broader model: small-airport training and fresh-date test

The earlier commercial checkpoint had zero positive small-airport examples. A new
run fine-tuned the same candidate-relative GRU on unique aircraft/time matches to
FAA SFDPS flights from March 15, April 1, June 1 and June 2, 2026. Candidates
were retrieved without labels, and labels outside the candidate set were
excluded from the training loss. The four training dates supplied 87,809 large,
26,757 medium and **8,704 small** retrieved positive endpoints; the small
positives span 1,132 distinct airports. Loss weights were 1:2:6 for large,
medium and small labels.

June 3–4 selected the epoch and fit a confidence temperature. The predefined
selection rule maximized average validation gain over nearest airport among
epochs that beat nearest in every size group and retained at least 99% large-
airport agreement. Epoch 4 was selected. June 5–6 was scored once as the main
held-out test; its 66,098 scored endpoints came from 33,178 matched flights.

| Held-out group | Endpoints | Nearest airport | Original NN | Broadly trained NN |
|---|---:|---:|---:|---:|
| All | 66,098 | 94.72% | 89.65% | **98.64%** |
| Large airport | 47,933 | 98.37% | 99.76% | **99.74%** |
| Medium airport | 13,817 | 85.92% | 82.63% | **98.30%** |
| Small airport | 4,348 | 82.47% | 0.41% | **87.70%** |

The new network improved on nearest airport by 5.81 percentage points on June 5
small-airport endpoints and 4.59 points on June 6. Across both dates, the
aircraft-cluster paired bootstrap 95% interval for the small-airport gain was
4.38 to 6.11 points. For 344 small-airport endpoints whose airport identity was
absent from broad fine-tuning, agreement was 65.41% versus 56.98% for nearest;
candidate recall there was 86.63%, indicating room to improve retrieval as well.
The validation-selected nearest-airport hybrid scored 98.62% overall, slightly
below the new network alone, so the network is the simpler choice.

The full-track assessment scored 66,100 resolvable endpoints, including two
heliport labels that the candidate catalog cannot represent. Its airport-size
rows match the cached 66,098-endpoint comparison above. On the same 6,451
endpoints with 0, 2 and 5 minutes of reception removed at both ends, network
agreement was 98.56%, 95.68% and 77.29%; nearest agreement was 94.56%, 53.46%
and 9.56%. A diagnostic subset of 2,132 small-airport endpoints with an already
close, low observation favored nearest: 99.62% versus 98.92% for the network.
That subset is defined using the true airport and cannot itself be an inference
rule. The new model wins across the measured airport-size groups, not every
individual flight or condition.

Of 899 network disagreements, a conservative terminal-track screen flagged 99
as possible label or segment errors and left 800 unresolved. The same screen
left 2,671 of 2,676 network-only agreements unresolved; missing terminal
reception often prevents direct track corroboration. Seven scored endpoints had
a three-letter code collision: the resolver chose an ident thousands of
kilometers from the observed track, while both methods selected the local
airport matching an alternative catalog code resolution. These seven count as
errors for both methods in the frozen scores. The audit preserves all source
labels and does not report adjusted accuracy. Its track evidence comes from
the same ADS-B source as the network and is not independent proof.

On the earlier, previously viewed March 1–8 filtered commercial benchmark, the
new checkpoint scored 99.9025% on 43,074 endpoints, versus 98.3494% for nearest
airport. The older commercial-only checkpoint scored 99.9350% on that sample,
14 additional correct endpoints. This secondary regression check was not used
to select the broader checkpoint.

A paired March 1 source comparison used the same 2,257 labeled ADSB.lol
segments and 4,514 endpoints for both observation networks. The selected broad
network scored 99.89% on either ADSB.lol or ADSBx; nearest scored 98.56% on
ADSB.lol and 98.29% on ADSBx. ADSBx provided a usable candidate on 4,513
endpoints. The four predictions and source-limited results are in
`$OUTPUT_DIR/data/experiments/flights_airport_nn/source-matrix-commercial-2026-03-01/`.

An additional March 1 source comparison used the **broader FAA SFDPS labels**:
24,402 ADSB.lol segments matched FAA flights uniquely by aircraft and time.
After excluding unresolved labels and unsupported airport types, the same 48,625
endpoints were scored with each network's observations. Missing observations
count as errors. The selected broad checkpoint gave:

| FAA label group | Endpoints | ADSB.lol NN | ADSB.lol nearest | ADSBx NN | ADSBx nearest |
|---|---:|---:|---:|---:|---:|
| All | 48,625 | 98.35% | 94.50% | 98.34% | 94.21% |
| Large | 32,924 | 99.53% | 98.37% | 99.51% | 98.11% |
| Medium | 11,416 | 97.91% | 86.59% | 97.88% | 86.25% |
| Small | 4,285 | 90.41% | 85.79% | 90.55% | 85.41% |

ADSBx had an endpoint observation for 48,611 of the 48,625 scored endpoints.
The FAA labels agreed with the original BTS/SFDPS commercial labels on all
4,490 endpoints shared by the two March 1 matched-flight sets; that overlap
does not validate the small-airport labels, which are outside the commercial set.
These are agreement scores with one FAA label source on a date already used in
the earlier commercial assessment, so they are an exploratory source comparison
rather than a new held-out result. Per-endpoint paired predictions, counts and
matching metadata are in
`$OUTPUT_DIR/data/experiments/flights_airport_nn/source-matrix-faa-2026-03-01/`.
Reproduce them with `python -m flights_airport_nn.compare_sources_broad`.

April 2 and June 7–8 were also withheld from training and selection, but their
segmented-flight files are partial. They contain 3,616 scored endpoints in total;
the new network scored 96.54% versus 88.86% for nearest, including 91.00% versus
87.20% at small airports. This is a limited-coverage check, not a second
full-day estimate.

The main report, per-endpoint predictions, paired uncertainty and exportable
figure are under
`$OUTPUT_DIR/data/experiments/flights_airport_nn/broad-improvement-2026-09-17-v1/primary-test/`.
The full-track report, reception-loss metrics and label audit are in the sibling
`reception-audit/` directory; see `results.md`, `metrics.json` and `audit/`.
The partial-date check is in `limited-coverage-check/`, and the earlier
commercial regression check is in `commercial-regression-check/`.
The selected model is under
`$OUTPUT_DIR/data/models/flights_airport_nn/runs/2026-09-17-broad-finetune-v1/selected_model.pt`.
The same `load_model` and `predict_endpoints` inference functions load the new
checkpoint. FAA SFDPS remains one label source, airport identities recur across
dates, and no finite test can guarantee correctness on every flight or in every
country.

The flight evaluator exposes the checkpoint as `algorithm-airports-nn`. It
starts from algorithm segments without airport assignments, reads endpoint
observations from `--adsb-src`, and returns the normal flight schema:

```bash
.venv/bin/python -m flights.evaluation.evaluation \
  --test-src algorithm-airports-nn --adsb-src adsblol \
  --gold-src ground_truth --test-dates 2026-03-01
```

The default checkpoint is the selected broad model above. Use
`--airport-nn-model /path/to/checkpoint.pt` to evaluate another checkpoint.
For ADSB.lol, the evaluator first writes a reusable full-day file under
`$OUTPUT_DIR/data/flights/algorithm-nn/year=YYYY/month=MM/day=DD/flights.parquet`.
Progress is printed after each aircraft batch, and completed batches resume
after interruption. Generate the March 1 file without running an evaluation:

```bash
.venv/bin/python -m flights_airport_nn.derive_day --date 2026-03-01
```

The flight evaluator saves its summary CSV in
`$OUTPUT_DIR/data/experiments/comparison/`. Each run also has a sibling directory
with the same filename stem. Its `index.json` identifies each comparison by
date and source pair; the listed Parquet files preserve the full `df0`/`df1`
columns and `match_status` returned by `df_flights_comparison`. For example:

```python
import json
from pathlib import Path
import polars as pl

run = Path("/path/to/2026-..._flights_evaluation")
entry = json.loads((run / "index.json").read_text())["comparisons"][0]
comparison = pl.read_parquet(run / entry["path"])
```

To repeat the broad training or score a new day with FAA source data, use:

```bash
.venv/bin/python -m flights_airport_nn.broad_cache \
  --dates 2026-03-15 2026-04-01 2026-06-01 2026-06-02 2026-06-03 2026-06-04
.venv/bin/python -m flights_airport_nn.train_broad \
  --train-dates 2026-03-15 2026-04-01 2026-06-01 2026-06-02 \
  --validation-dates 2026-06-03 2026-06-04 \
  --initial-model /path/to/commercial/model.pt --output-dir /path/to/broad-run
.venv/bin/python -m flights_airport_nn.select_broad --run-dir /path/to/broad-run
.venv/bin/python -m flights_airport_nn.stress_broad \
  --model /path/to/broad-run/selected_model.pt \
  --dates 2026-06-09 --output-dir /path/to/new-broad-evaluation
```

The source files for a requested day must exist locally. `stress_broad` scores
normal reception and a fixed 10% reception-loss sample, retains candidate
failures in the denominator, and writes the scored endpoints for audit.
