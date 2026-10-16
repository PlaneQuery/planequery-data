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

The Eurocontrol source extracts only the required `Flights_*.csv.gz` member to
a temporary directory, creates `data/intermediate/eurocontrol/YYYYMM.parquet`,
and writes daily files under `data/flights/source=eurocontrol/version=3/`.
Temporary gzip/CSV files are removed automatically. Existing legacy extracted
inputs remain supported as fallbacks.

## OpenSky reconstruction

`opensky-reverse-eng` derives flights from raw OpenSky `state_vectors_data4`
using the rules inferred on June 1, 2024 and validated on March 1, 2026:

```bash
python -m flights.cli 2026-03-01 2026-03-02 opensky-reverse-eng \
  --adsb-source opensky --workers 4
```

It is an `ADSBFlightSource` and supports the normal flight cache, airport
provenance, evaluation, and aircraft-batch workers. It currently requires the
OpenSky provider. The existing `opensky` source continues to read published
`flights_data4` records.

The reconstruction cannot use canonical `read_adsb(source=OPENSKY)`: that
pipeline removes the snapshot clock and original fractional position clock,
collapses repeated updates, rounds altitudes, cleans tracks, and normalizes
callsigns. Those changes affect density, median filtering, gap decisions, and
endpoint times. The source-specific reader reuses the existing raw OpenSky file
validation and ADS-B aircraft buckets. It preserves both clocks, metre
altitudes, and original callsigns; no ground flag, reported velocity, or
published flight records are needed for inference.

For takeoffs on day D, all 24 raw hours must exist for D−1 through D+2. Two
three-day replays reconstruct OpenSky arrival partitions D and D+1, then select
takeoffs on D, matching the existing published OpenSky provider's convention.
Missing raw hours fail explicitly. For the example above, the required raw
partitions are February 28 through March 3, 2026. The lower-level
`reconstruct_arrival_day` function accepts just the three-day context and is
used to compare against the research arrival-day benchmark.

The reader builds a separate filtered, aircraft-sorted cache once per day:

```text
$OUTPUT_DIR/data/intermediate/opensky/reverse-eng-states/version=1/
```

Only finite positions with `snapshot time - lastposupdate < 1 second` enter
this cache; duplicates and valid ground positions remain. Hourly writes bound
memory, and a per-day lock protects overlapping workers. The normal flight
output is stored at:

```text
$OUTPUT_DIR/data/flights/source=opensky-reverse-eng/adsb_source=opensky/version=1/
```

Flight manifests identify the raw-input cache format/version rather than a
canonical ADS-B Parquet version. Bump the flight-source version if changes to
input processing affect predictions. `--no-save-adsb-downloads` retains this
intermediate cache; that option removes canonical ADS-B downloads, which this
source does not generate.

The inferred algorithm uses four or more samples per epoch-aligned 30-second
bin, seven-point median checks, and removal of three leading and six trailing
samples per replay batch. Gaps split only when longer than 600 seconds and
slower than one quarter of the batch's median displacement speed. Unfinished
raw tails are replayed the next day, including days without new observations.
Finalized tracks must be strictly longer than 600 integer seconds. This differs
from the paper's 15-minute description.

Airport assignment uses the retained endpoints with horizontal/vertical gates
of 15 km/2 km for departure and 30 km/2.5 km for arrival. The shared dated
OurAirports catalog supplies stable IDs and canonical identifiers, restricted
to fixed-wing airports with GPS codes and elevations. It is an approximation
of OpenSky's unavailable airport catalog, and differs from the fixed June 2024
catalog used in the research benchmark. First/last contact times populate
both the flight boundary and message-time fields; they are not observed runway
events. Callsign mode is computed before trimming its surrounding whitespace.
Unknown aircraft metadata uses the standard flight-schema defaults.

Run the production source's regression tests with:

```bash
python -m pytest tests/flights/test_opensky_reverse_eng.py
```

The exploratory scripts have been retired. Saved integration results remain
under `data/processed/opensky_reverse_engineering/2026-03-01/` as
`source_validation.json` and `source_validation_{frozen,shared}.parquet`.

The March 1, 2026 integration check covered 2,048 aircraft and 5,241 published
flights with known starts. All 5,237 reconstructed boundary pairs, callsigns,
and airport labels match the saved research predictions when using the frozen
catalog (after canonical callsign normalization). With the shared dated
catalog, 5,200/5,241 reference flights have exact boundaries (99.22%), and
5,219/5,241 match within 30 seconds (99.58%). Exact boundaries, callsign, and
both airport labels agree for 4,735/5,241 flights (90.35%); the frozen catalog
scores 4,761/5,241 (90.84%). These are sampled arrival-day results, not a
full-population accuracy estimate or an exact reproduction of OpenSky's
candidate-distance fields.
