
# Testing Data to focus on for now
Febuary 28th -> March 4th. So I can test March 1, 2, 3

# Setup
pip install -e .

# Canonical ADS-B Parquet

Downloaded provider files remain under `data/raw`. All ADS-B readers consume
the same derived layout:

```text
data/processed/adsb/parquet_output/
  source=adsblol/version=6/year=YYYY/month=MM/day=DD/icao_bucket=N/*.parquet
  source=adsbx/version=1/year=YYYY/month=MM/day=DD/icao_bucket=N/*.parquet
  source=opensky/version=4/year=YYYY/month=MM/day=DD/icao_bucket=N/*.parquet
```

Every completed day contains `_SUCCESS`, uses the shared ADS-B schema, and uses
the unchanged eight-bucket ICAO partitioning scheme (`icao_bucket=0` through
`icao_bucket=7`). `read_adsb()` creates a missing provider/day partition and
then uses the common scanner.

# Historical OurAirports data

Airport data is resolved through an immutable `AirportCatalogView` for each
flight day. On first use, `AirportHistory.open()` clones
`davidmegginson/ourairports-data` and derives a change-only temporal catalog
under `data/raw/ourairports/history/`. The derived files contain airport
versions, code/ident aliases, source commits, and a commit manifest; they are
not daily copies of `airports.csv`. Later refreshes process only new airport
commits. Empty upstream exports are recorded but do not replace the last valid
state.

Each flight-day view combines two end-of-day repository states:

- The state observed on the flight day supplies historically correct airport
  metadata and determines which fixed-wing airports can be spatial matches.
- The state one calendar month later supplies late-discovered airports and
  code aliases. For flights less than one month old, the repository's latest
  state is used instead.

All OurAirports facility types are retained. Closed airports can be resolved
directly for historical records but are excluded from nearest-airport spatial
matching when they were closed on the flight day. An airport that was active
on the flight day remains eligible even if it closed during the one-month
knowledge window.

Registered flight sources persist the stable OurAirports `id` in
`takeoff_airport_id` and `landing_airport_id`, with the date-specific `ident`
retained as a label. Every registered flight-day parquet has an adjacent
`airport_catalog.json` recording the flight-day commit, knowledge-cutoff
commit, available catalog date, and policy version.
