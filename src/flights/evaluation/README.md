# Flight evaluation

`flights.evaluation.evaluation` compares each requested test/gold source pair
across all aircraft ICAOs in the gold source. Test-source flights are limited
to non-null ICAOs present in gold, so gold-only aircraft contribute false
negatives while test-only aircraft are outside the evaluation cohort. Daily
comparison details and metric rows are cached under
`OUTPUT_DIR/data/cache/evaluation`.

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
