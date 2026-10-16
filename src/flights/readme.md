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

For ADS-B-backed sources, pass `--no-save-adsb-downloads` to remove release
archives, extracted traces, and canonical ADS-B Parquet downloaded or generated
by that invocation after the flight files have been written. ADS-B cache paths
that existed before the command and the generated flight Parquet are preserved.

`algorithm-nn` and `algorithm-nearest` use the worker budget at the widest
available level. A one-day range processes complete-aircraft ADS-B batches in
parallel; a multi-day range processes days in parallel and keeps each day
single-process to avoid nested pools. The default uses half of the machine's
logical CPUs, is limited to roughly one worker per 4 GiB of RAM while retaining
20% memory headroom, and can be overridden with `--workers`.

When a matching `algorithm-nearest` day and airport-catalog manifest already
exist for the same ADS-B provider, `algorithm-nn` reuses those flight boundaries
and metadata and reads only that provider's ADS-B columns needed for airport
inference. For example, OpenSky inference reuses only OpenSky segmentation.
Missing or stale nearest output falls back to normal segmentation of the same
ADS-B provider.

## ADS-B Exchange published flights

The `adsbx` flight source is ADS-B Exchange's published flight product, not
the ADSBX state-vector source used by ADS-B-backed algorithms. The upstream
files are partitioned by arrival date, and only the first day of each month is
available. The provider therefore creates the complete same-day cohort from
that one file: flights whose takeoff and landing both occur on date `D`.
Evaluation applies the same restriction to every source, so overnight flights
are excluded consistently rather than counted as missing ADSBX flights.

Normalized arrival artifacts are stored under:

```text
$OUTPUT_DIR/data/intermediate/adsbx/published-arrivals/version=1/
```

Older artifacts under `data/flights/adsbx/v1/` remain readable as a migration
fallback. New artifacts are written only to the intermediate-data location.
Aircraft flags and canonical airport IDs are added by the versioned `adsbx`
flight provider and are not persisted in the normalized arrivals artifact.

## Eurocontrol

Place each raw monthly archive at:

```text
$OUTPUT_DIR/data/raw/eurocontrol/YYYYMM.zip
```

Then create every daily partition in that month with an exclusive end date:

```bash
python -m flights.cli 2025-08-01 2025-09-01 eurocontrol
```

The Eurocontrol source extracts the `Flights_*.csv.gz` and
`Flight_FIRs_Actual_*.csv.gz` members to a temporary directory (uncompressed CSV
members are also supported). It joins them by `ECTRL ID` and uses `Exit Time`
where `FIR ID` is `TAXI_OUT` for `takeoff_time`; missing or invalid exit times
remain null. `landing_time` uses `ACTUAL ARRIVAL TIME` from the flights summary.
Daily partitions are selected by the takeoff date.

The monthly cache is `data/intermediate/eurocontrol/version=2/YYYYMM.parquet`,
and daily files are written under `data/flights/source=eurocontrol/version=5/`.
These versions bypass older caches that used off-block time for takeoff.
Temporary CSV files are removed automatically. If the monthly ZIP is absent,
the source reads the two required files directly from
`data/raw/eurocontrol/YYYYMM/`, supporting both `.csv.gz` and `.csv` files.
Existing extracted inputs are retained unchanged.
