# Airport prediction from segmented ADS-B flights

A shared GRU ranks airport candidates for **takeoff and landing**. It uses relative
geometry, altitude above airport elevation, ground speed, vertical rate, heading,
coverage and missing-value indicators, airport size and scheduled service. It does
not use airport IDs, aircraft IDs, callsigns, absolute coordinates, routes, or label
frequencies as model inputs.

The model combines the endpoint observation, a recurrent trajectory encoding and
attention pooling, with a learned correction to a distance prior. Each endpoint
is trained with a softmax loss over its candidate airports. AdamW, dropout,
gradient clipping, coverage augmentation and early stopping limit overfitting.
Validation loss selects the checkpoint; validation data also fits one confidence
temperature. Test data is never used for either operation.

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
  The upstream ground-truth builder also checks ADS-B proximity to labeled
  airports and flight continuity. This selects a cleaner coverage population;
  reported accuracy does not measure performance on all raw segments or severe
  reception gaps.
* The present labels are BTS/SFDPS agreement for US commercial flights. Random
  endpoint splits would be misleading because routes and aircraft repeat. Keep
  chronological validation/test splits and report unfamiliar aircraft/airports.
  Even good held-out results do not establish worldwide or general-aviation
  accuracy; those need independent labels before making that claim.
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

The evaluator rejects training/validation dates. Repeatedly selecting models from
the same test results turns that test into validation; reserve new dates for the
next final assessment.

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
