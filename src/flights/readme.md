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
and daily files are written under
`data/flights/source=eurocontrol/adsb_source=adsblol/version=6/` by default.
These versions bypass older caches that used off-block time for takeoff.
Temporary CSV files are removed automatically. If the monthly ZIP is absent,
the source reads the two required files directly from
`data/raw/eurocontrol/YYYYMM/`, supporting both `.csv.gz` and `.csv` files.
Existing extracted inputs are retained unchanged.

Every EUROCONTROL build enriches callsigns using locally cached ADS-B data.
ADSB.lol is the default provider; use `--adsb-source` to choose another:

```bash
python -m flights.cli 2024-06-01 2024-06-02 eurocontrol \
  --adsb-source opensky
```

Supported providers are `adsblol`, `opensky`, and `adsbx`. Enrichment scans only complete local
canonical ADS-B partitions for the aircraft and dates covered by the flights,
including arrival dates for overnight flights. It does not download missing
partitions. Flights without sufficient coverage remain in the output with an
empty callsign.

Observations must match the aircraft's ICAO address and fall in its
`[takeoff_time, landing_time)` interval. Callsigns are uppercased with whitespace
removed; valid values contain 1–8 letters or digits. Each distinct callsign
receives at most one vote per minute per flight. A callsign needs at least
three supporting minutes and 80% of valid votes; otherwise it remains empty.
Missing observations are never filled from another flight.

Enriched files use a separate provider-specific path, for example
`data/flights/source=eurocontrol/adsb_source=adsblol/version=6/`, and their
manifests track the ADS-B provider and Parquet version. Rerun the CLI to rebuild
enriched flights after adding previously missing ADS-B partitions.

From Python, `EurocontrolFlightSource()` always enriches using ADSB.lol.
Pass `EurocontrolFlightSource(callsign_source=ADSBSource.OPENSKY)` to
`get_flights`/`write_flights` to select OpenSky, or call
`enrich_eurocontrol_callsigns(flights, ADSBSource.ADSBLOL)` from
`flights.providers.eurocontrol_callsigns` on an existing flight DataFrame.

Use `eurocontrol-verified` to apply the verified-flight consistency checks:

```bash
python -m flights.cli 2024-06-01 2024-06-02 eurocontrol-verified
```

It requires known, consecutive airports and at least four flights per aircraft,
with the flight callsign sequence matching the ADS-B callsign history exactly.
Unlike `bts-verified` and `sfdps-verified`, it accepts airports outside US FAA
airspace. It sorts flights and observations and includes overnight arrivals.
The same `--adsb-source` provider is used for enrichment and verification;
verification reads ADS-B through the standard reader, which can fetch missing
partitions. Callsigns are refreshed from those observations before checking the
sequence, so this is a consistency check, not independent callsign ground truth.
Aircraft that fail the checks are excluded.

The source is also available as `EUROCONTROL_VERIFIED_SOURCE` from
`flights.sources` and by name in the evaluation CLI. Its default output path is
`data/flights/source=eurocontrol-verified/adsb_source=adsblol/version=1/`.
