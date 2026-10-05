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

```bash
python -m flights.evaluation.evaluation \
  --test-source algorithm-nn \
  --gold-source bts \
  --adsb-source adsblol adsbx \
  --test-dates 2026-02-01 2026-03-01 \
  --output-csv /tmp/evaluation.csv
```

Select and order CSV columns with `--columns`:

```bash
python -m flights.evaluation.evaluation \
  --test-source algorithm-nn \
  --gold-source bts \
  --adsb-source adsblol adsbx \
  --test-dates 2026-02-01 2026-03-01 \
  --output-csv /tmp/evaluation-summary.csv \
  --columns test_date adsb_src precision recall f1 \
    takeoff_airport_id_match_pct landing_airport_id_match_pct
```

Use `--refresh-cache` to replace matching entries or `--no-cache` for a run
that neither reads nor writes the persistent cache.

## Inspect existing cache entries

`--list-cache` is read-only. With no filters it lists every day in the active
cache version; source, provider, and date options narrow the results. Omit
`--columns` to export all metric and cache-provenance columns.

```bash
python -m flights.evaluation.evaluation --list-cache \
  --test-source algorithm-nn \
  --gold-source bts \
  --adsb-source adsblol adsbx \
  --output-csv /tmp/cached-evaluations.csv
```

If a requested day is not cached, use the normal command above with that date;
it will reuse hits and create only the missing entries.

The equivalent Python entry points are `evaluate`, `list_cached_evaluations`,
and `export_cached_evaluations` in `flights.evaluation.evaluation`.
