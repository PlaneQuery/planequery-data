# Flight evaluation

`flights.evaluation.evaluation` compares each requested test/gold source pair
across all aircraft ICAOs in the gold source. Test-source flights are limited
to non-null ICAOs present in gold, so gold-only aircraft contribute false
negatives while test-only aircraft are outside the evaluation cohort. Daily
comparison details and metric rows are cached under
`OUTPUT_DIR/data/cache/evaluation`.

BTS excludes an aircraft's entire UTC takeoff day when a non-cancelled,
non-diverted report is missing `air_time`, `wheels_on`, or `arr_delay`. The
exclusion uses the mapped ICAO and also removes that aircraft's diversion legs
on the affected day, while retaining its other days. Departure timing determines
the UTC day; when the departure day is uncertain, the exclusion covers the
reported local day and a possible overnight departure (including their UTC date
boundaries). Diversion reports use their separate timing fields and do not
trigger this missing-field rule. BTS cache version 16 and BTS-verified version 2
apply this policy.

## Get or create evaluation data

A normal run checks the cache first and calculates only missing combinations of
date, test source, gold source, and, where applicable, ADS-B source. Multiple
values form the full combination set. The CSV contains every metric column by
default.

`takeoff_airport_id_null_count` and `landing_airport_id_null_count` count
missing resolved airport IDs in retained **test-source** flights, including
unmatched test flights. Their corresponding `_null_pct` columns divide by
`true_positive + false_positive` (stored as fractions from 0 to 1; zero when
there are no retained test flights). Gold-only rows are excluded, as are
test-only aircraft and flights discarded by the comparison's boundary grace
period. This includes OpenSky flights with missing or unresolved airports.
Existing airport match metrics still use matched flights only, and null
airports remain incorrect matches. Console output also displays null counts
and formatted percentages. These metrics are included in CSVs, cached stats,
and daily averages.

`takeoff_airport_id_non_null_match_pct` and
`landing_airport_id_non_null_match_pct` report accuracy among matched flights
whose corresponding **test-source** airport ID is non-null. Each divides the
airport match count by its `_non_null_count` column. Unmatched flights are
excluded; a non-null test airport with a null gold airport still counts as
incorrect. Values are fractions from 0 to 1, or zero if no eligible airports
exist. These columns are available for all sources and are also displayed as
formatted percentages for OpenSky. They are included in CSVs, cached stats,
and daily averages. Cache version 8 recalculates older evaluations on demand.

```bash
python -m flights.evaluation.evaluation \
  --test-source algorithm-nn \
  --gold-source bts \
  --adsb-source adsblol adsbx \
  --test-dates 2026-02-01 2026-03-01
```

The CSV is written to the timestamped default location under
`OUTPUT_DIR/data/experiments/comparison`.

Select and order CSV columns with `--columns`:

```bash
python -m flights.evaluation.evaluation \
  --test-source algorithm-nn \
  --gold-source bts \
  --adsb-source adsblol adsbx \
  --test-dates 2026-02-01 2026-03-01 \
  --columns test_date adsb_src precision recall f1 \
    takeoff_airport_id_match_pct landing_airport_id_match_pct
```

Use `--refresh-cache` to recalculate and replace matching entries.

## Compare training and nontraining airports

Add `--airport-nn-training /path/to/training.json` to optionally report airport
accuracy by exposure to positive training labels:

```bash
python -m flights.evaluation.evaluation \
  --test-source algorithm-nn \
  --gold-source sfdps bts-verified \
  --adsb-source adsblol \
  --test-dates 2026-03-01 \
  --airport-nn-training data/models/flights_airport_nn/learning-curve-10-20/days-20/training.json
```

Use the manifest belonging to the evaluated checkpoint (selected separately
with `--airport-nn-model` when needed). Only Parquet metadata files explicitly
listed in `inputs["train"]` are read. A relative manifest argument resolves under
`OUTPUT_DIR`. Each metadata file is located by its dataset/date/filename suffix
under `OUTPUT_DIR/data/processed/flights_airport_nn`, for example
`OUTPUT_DIR/data/processed/flights_airport_nn/february-2026/2026-02-01/part-0000.parquet`.
The recorded machine prefix does not affect local lookup, and the raw manifest
is never modified. The files must exist under the configured output directory.
Membership uses rows with `eligible == True` and `target >= 0`, combining
takeoff and landing labels. Numeric `truth_id` labels are used directly; airport
codes in `truth` are resolved to IDs through the catalog for each training date.
Unresolved codes raise an error. Validation files and NPZ feature arrays are
not required.

Two additional CSVs are written alongside the normal stats CSV, using the same
filename stem:

- `*_airport_exposure.csv`: six rows per daily comparison—takeoff and landing,
  each split into `training`, `nontraining`, and `unknown` gold-airport groups.
- `*_training_airports.csv`: the training airport IDs and their contributing
  endpoint counts, preserving the membership used for that report.

Each endpoint is grouped by its **gold** airport ID, independently of the
other endpoint or the prediction. The summary preserves date, test/gold
source, ADS-B provider, and input-window settings, and records the manifest
path and SHA-256. Counts and rates have these definitions:

| Column | Meaning |
| --- | --- |
| `gold_endpoint_count` | Retained matched and unmatched gold endpoints in the group |
| `gold_airport_count` | Distinct known gold airport IDs across those endpoints |
| `matched_endpoint_count` | Endpoints on matched flights |
| `matched_airport_count` | Distinct known gold airport IDs on matched flights |
| `correct_count` | Matched endpoints whose predicted airport ID equals gold |
| `incorrect_count` | Matched endpoints with an incorrect or null prediction |
| `null_prediction_count` | Matched endpoints with no predicted airport ID |
| `airport_accuracy` | Correct / matched endpoints (flight weighted) |
| `airport_macro_accuracy` | Mean of each airport's matched-endpoint accuracy (equal airport weight) |
| `flight_match_coverage` | Matched / all retained gold endpoints |
| `correct_endpoint_recall` | Correct / all retained gold endpoints |

Unknown gold airport IDs have counts and flight-match coverage but null
correct/incorrect counts and accuracy/recall metrics. Rates with an empty
denominator are null, rather than zero. Rates are fractions from 0 to 1.
Test-only rows and flights previously discarded by the boundary grace period
are excluded. Results remain daily; the report does not average daily rates.

`nontraining` means absent from positive training labels. It can include airports
seen during validation or as negative candidates, so it is not a claim of no
model exposure. Use evaluation dates outside the training dates when measuring
generalization; comparing airport types within each group can help distinguish
airport composition from training effects.

The report is derived after matching and reuses cached comparisons. It does not
change cache keys or cached stats. Without the option, training files are never
read and no additional CSVs are written. `--columns` still controls only the
normal stats CSV. `--list-cache` lists stats only and cannot be combined with
this option; run the normal evaluation command to reuse detailed cache entries.

The same report can be created from an existing Python result without running
another evaluation:

```python
from pathlib import Path
from flights.evaluation.airport_exposure import (
    airport_exposure_report,
    load_training_airports,
)

training = load_training_airports(Path("/path/to/training.json"))
report = airport_exposure_report(results, training)
```

## Inspect existing cache entries

`--list-cache` is read-only. With no filters it lists every day in the active
cache version; source, provider, and date options narrow the results. Omit
`--columns` to export all metric and cache-provenance columns.

```bash
python -m flights.evaluation.evaluation --list-cache \
  --test-source algorithm-nn \
  --gold-source bts \
  --adsb-source adsblol adsbx
```

Cache listings are written to stdout. Use `export_cached_evaluations` when a
specific destination file is needed programmatically.

If a requested day is not cached, use the normal command above with that date;
it will reuse hits and create only the missing entries.

The equivalent Python entry points are `evaluate`, `list_cached_evaluations`,
and `export_cached_evaluations` in `flights.evaluation.evaluation`.

`evaluate` returns `EvaluationResults`, whose entries pair each detailed
comparison with a typed `EvaluationStats` object. DataFrames are an explicit
reporting boundary rather than the primary stats interface. It never writes a
CSV; the command-line interface handles CSV reporting separately.

```python
from datetime import date

from flights.evaluation.evaluation import evaluate

results = evaluate(
    test_source="algorithm-nn",
    gold_source="bts",
    test_dates=date(2026, 3, 1),
)

stats = results.stats[0]
print(stats.precision, stats.recall, stats.airport_id_match_pct)

# Convert only when tabular output is needed.
frame = results.to_frame()
```

The result models live in `flights.evaluation.models`. For object-based cache
discovery, `EvaluationCache.find_stats(...)` returns `CachedEvaluationStats`
objects without loading the detailed comparison files. `list_entries(...)`
is the tabular adapter for display and CSV export.
