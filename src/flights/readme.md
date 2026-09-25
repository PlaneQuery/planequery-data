# Flights

Create daily flight Parquet files for a date range with:

```bash
python -m flights.cli START_DATE END_DATE SOURCE [OPTIONS]
```

For example, create airport-NN flights for March 1 through March 7:

```bash
python -m flights.cli 2026-03-01 2026-03-08 algorithm-nn \
  --adsb-source adsblol
```

The start date is inclusive and the end date is exclusive. Run
`python -m flights.cli --help` to list sources and options.

`algorithm-nn` and `algorithm-nearest` process days in parallel. Their default
worker count uses half of the machine's logical CPUs, is limited to roughly one
worker per 4 GiB of RAM while retaining 20% memory headroom, and can be
overridden with `--workers`.

## Eurocontrol

Place each raw monthly archive at:

```text
$OUTPUT_DIR/data/raw/eurocontrol/YYYYMM.zip
```

Then create every daily partition in that month with an exclusive end date:

```bash
python -m flights.cli 2025-08-01 2025-09-01 eurocontrol
```

The Eurocontrol source extracts only the required `Flights_*.csv.gz` member to
a temporary directory, creates `data/intermediate/eurocontrol/YYYYMM.parquet`,
and writes daily files under `data/flights/source=eurocontrol/version=3/`.
Temporary gzip/CSV files are removed automatically. Existing legacy extracted
inputs remain supported as fallbacks.
