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

All optimizer observations come from ADSB.lol. Training uses February 1–20.
Checkpoint selection uses February 21–27 (BTS/SFDPS) and June 12/15, 2024
(ADSB.lol with EUROCONTROL labels). February 28 is an unused buffer.
The selection loss gives equal weight to the US and European validation regions,
then equal weight to days within each region. Airport-size weights remain 1/2/4
for large/medium/small airports. Final benchmark dates are March 1, 2026 and
June 1, 2024, scored only after the checkpoint is frozen. Both benchmark dates
were evaluated in earlier experiments, so they are not untouched tests.
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

The training dataset builder rejects dates outside February 2026. European
validation caches are prepared separately by
`flights.evaluation.eurocontrol_benchmark`; they never enter optimizer batches.
Cache manifests record
source files and feature/segmentation code; incomplete caches are rejected.
Training refuses an existing model directory, so a rerun requires a new
`--output` directory. Install the optional `airport-nn` dependencies for PyTorch.

Default checkpoint:
`data/models/flights_airport_nn/february-2026-europe-validation/model.pt`, beneath
the configured storage root. The previous model remains in `february-2026/`.
The new directory contains training provenance, epoch history,
and per-day `validation/` reports. Each report has `endpoints.parquet`,
`references.parquet` (including unmatched references), and `metrics.json`.

The split, daily reference-availability audit, European acquisition/cache data,
and final comparisons are stored under
`data/experiments/flights_airport_nn/split-2026-09-20/`.
The [completed retraining report and Parquet results](/mnt/storage1/planequery-data/data/experiments/flights_airport_nn/split-2026-09-20/results.md)
record epoch 3 selected from six completed epochs. The new checkpoint improves
March 1 small-airport agreement, while some medium/large-airport results regress.
Nearest still leads on the small EUROCONTROL airport subsets.
EUROCONTROL matching allows native takeoff from five minutes before to sixty
minutes after off-block and landing within fifteen minutes. Only unique
aircraft/time matches are used. Primary European scoring does not require
ground reception; missing/ambiguous matches remain in the recovery denominator.

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
python -m flights.evaluation.airport_benchmark --date 2026-03-01 \
  --networks adsblol adsbx --workers 8 \
  --output /mnt/storage1/planequery-data/data/experiments/flights_airport_nn/split-2026-09-20/final-us
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
The network benchmark also requires overlapping reference/segment intervals
and records how many observed positions fall inside each reference interval,
separating absent observations from observed but unmatched flights.

The earlier February-only model's results live under
`data/experiments/flights_airport_nn/february-2026-march-test/{date}/`.
No March data are used for optimization or checkpoint selection. For a new
model, pass a fresh `--output` directory to keep model-specific reports separate.

The previous run's [report and Parquet results](/mnt/storage1/planequery-data/data/experiments/flights_airport_nn/february-2026-march-test/results.md)
include reference-flight recovery, shared-network cohorts, unseen-airport and
aircraft breakdowns, and an audit of reference/endpoint disagreements. The
March 1 small-airport subset remains a weakness: nearest has higher reference
agreement than this NN on both networks. March 2 has a more selective reference
cohort because only ADSB.lol corroboration is available.
