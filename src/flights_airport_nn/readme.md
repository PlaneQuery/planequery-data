# Airport neural network

A shared GRU ranks nearby airports using up to 32 observations from each end of
an independently segmented ADS-B flight. The inputs are relative geometry,
altitude, motion and airport characteristics. Aircraft IDs, airport IDs, dates
and reference airport labels are not model inputs.
The default model has 38,082 trainable parameters.

## Code

- `features.py`: one fixed representation and candidate search (32 airports).
- `model.py`: shared trajectory ranker and checkpoint loading.
- `data.py`: February reference matching, quality checks and feature caches.
- `train.py`: fresh training, validation selection and held-out scoring.
- `evaluate.py`: endpoint metrics and inspectable Parquet results.
- `inference.py`: predictions preserving the standard Flight schema.
- `derive_day.py`: resumable daily inference and evaluator integration.

## February 2026 model

All observations come from ADSB.lol. Training uses February 1–20, checkpoint
selection uses February 21–24, and the untouched final test uses February 25–28.
Training starts with random weights and an explicit nearest-distance prior.
There is no pretrained checkpoint or airport-specific embedding.

BTS airline reports and conservative SFDPS airport codes supply labels. Flights
are segmented from the full UTC day before reference matching; reference times
and airports never define segments. Matching uses aircraft and times only, with
a 15-minute tolerance at both ends and unique matches. Cross-midnight flights,
overlapping references and conflicting labels are excluded from supervision.

BTS labels can be used directly. SFDPS-only labels additionally require at least
three low/ground observations over ten seconds within 2 km of the reported
airport. This is a corroborated subset, with a bias toward good reception; its
scores must not be described as accuracy over all flights. Uncorroborated label
agreement is reported separately. Candidate retrieval failures stay in the
accuracy denominator. Medium/small airports receive loss weights of 2/4 versus
1 for large airports. Temporal holdouts prevent flight reuse, but do not ensure
that airports or aircraft are unseen.

```bash
python -m flights_airport_nn.data --workers 4
python -m flights_airport_nn.train
```

The dataset loader rejects dates outside February 2026. Cache manifests record
source files and feature/segmentation code; incomplete caches are rejected.
Training refuses an existing model directory, so a rerun requires a new
`--output` directory. Install the optional `airport-nn` dependencies for PyTorch.

Default checkpoint:
`data/models/flights_airport_nn/february-2026/model.pt`, beneath the configured
storage root. The directory also contains training provenance, epoch history,
and `validation/` and `test/` reports. Each report has `endpoints.parquet`,
`references.parquet` (including unmatched references), and `metrics.json`.

## Inference

```bash
python -m flights_airport_nn.derive_day --date 2026-03-01 --source adsblol
python src/flights/evaluation/evaluation.py \
  --test-src algorithm-airports-nn --gold-src ground_truth \
  --test-dates 2026-03-01
```

Inference supports ADSB.lol and ADSBX. Only the two airport identifier columns
are replaced; flight boundaries and the standard Flight schema are preserved.
Missing observations produce null airport identifiers. Daily results are stored
under `data/flights/algorithm-nn/{source}/year=…/month=…/day=…/`, with a cache key
covering model contents, flight contents, observations, airport catalog and code.
Old experimental models are intentionally unsupported.

## Frozen-model network comparison

After training, run the independent benchmark against the existing commercial
ground-truth files and the broader BTS/SFDPS reference set:

```bash
python -m flights.evaluation.airport_benchmark --date 2026-03-01 --networks adsblol adsbx --workers 8
python -m flights.evaluation.airport_benchmark --date 2026-03-02 --networks adsblol --workers 8
```

The benchmark lives with evaluation code and cannot write training caches.
Both methods use the same independently generated segments. Nearest means the
closest supported airport to the first/last valid observed segment position.
Both networks use the same reference definitions, including the existing
ADSB.lol-derived commercial ground truth. That commercial set was itself
filtered using ADSB.lol reception, so it is not an unbiased coverage sample.

Reports include per-network and shared-flight comparisons, endpoint accuracy,
both-airports-correct flight accuracy, candidate recall, matching coverage, and
airport-size breakdowns. SFDPS-only primary labels need ground corroboration in
at least one evaluated network; this shared eligibility rule does not use model
or nearest predictions. Uncorroborated agreement is separate. Results are
conditional on unambiguous temporal matches, with unmatched references retained
in `references.parquet` alongside predictions and `match_status`.

Results live under
`data/experiments/flights_airport_nn/february-2026-march-test/{date}/`.
No March data are used for optimization or checkpoint selection.
