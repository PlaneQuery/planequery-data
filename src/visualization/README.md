# ADS-B paper figures

`adsb_track.py` overlays barometric altitude (blue, left axis) and ground speed
(charcoal, right axis) in one graph. A narrow strip inside its bottom edge shows
`on_ground`: orange for true and light blue for false; missing states stay blank. Defaults
are a 7.2 × 4.8 inch figure with serif text, embedded
TrueType fonts in vector PDF, editable SVG text, and a 300 dpi PNG preview.
No LaTeX installation is required. Outputs default to `paper/figures/`.

Reproduce the inspected example from the repository root:

```sh
.venv/bin/python src/visualization/adsb_track.py \
  --icao a7ede5 --date 2026-03-01 --source adsbx \
  --observations tmp/clear_adsbx_20260301/a7ede5_observations.parquet \
  --flights data/flights/algorithm/v1/adsbx/year=2026/month=03/day=01/flights.parquet \
  --event-source 'algorithm flight estimates' \
  --start 01:30 --end 06:30
```

For another aircraft, omit `--observations` to use the repository's canonical
provider reader. Set `OUTPUT_DIR` to the data root, as with other data scripts:

```sh
OUTPUT_DIR=/Volumes/T2-SSD/planequery .venv/bin/python src/visualization/adsb_track.py \
  --icao a7ede5 --date 2026-03-01 --source adsbx \
  --flights data/flights/algorithm/v1/adsbx/year=2026/month=03/day=01/flights.parquet
```

The reader can create/download a missing canonical partition. To plot only an
existing file without that behavior, pass `--observations` explicitly.
Flight input is a Parquet table with `icao`, `takeoff_time`, `landing_time`,
`takeoff_airport_ident`, and `landing_airport_ident`. Supply the flight source you
want to assess and identify it using `--event-source`. Missing airport codes are
labeled `unknown airport`; missing event times are omitted. Without `--flights`,
only observations are plotted: the script does not guess flight boundaries or
airports from ground-state flips.

All timestamps are UTC (naive input timestamps are interpreted as UTC). An
annotation marks the supplied flight timestamp, not a guarantee of exact liftoff
or touchdown. The state strip independently shows the reported flags, including
brief flips. Null flags stay blank. It does not debounce or smooth them.
The provider reader determines the meaning of its ground flags: for ADSBX the
canonical conversion maps `alt_baro == "ground"` to true. Thus false is a
reported/conversion state, not independently verified evidence of flight.

Missing altitude is left missing, including on-ground samples; it is never set
to zero. Negative barometric altitudes are retained. Lines break across missing
values and observation gaps over `--gap-seconds` (default 30). Gaps stay blank,
without shading; the ground-state strip does not bridge them. Points show actual
samples, while connecting lines only aid reading. No state is inferred outside
the observed interval. Use `--start 05:00 --end 07:00` to inspect a dense or
ambiguous section, particularly when many event labels crowd a full-day plot.

Use `--width` and `--height` to set the intended physical publication size,
`--formats pdf` for PDF only, or `--output paper/figures/custom_name` to set the
output stem. For a two-column paper:

```tex
\begin{figure*}[t]
  \centering
  \includegraphics[width=\textwidth]{figures/a7ede5_20260301_adsbx.pdf}
  \caption{ADSBX observations for N61RJ (ICAO A7EDE5) on 1 March 2026 UTC.
  Altitude uses the left axis and ground speed the right axis. Dotted lines mark
  algorithm-estimated flight boundaries; the strip inside the bottom edge
  shows reported ground states. Observation gaps longer than 30 seconds
  are left blank.}
  \label{fig:adsbx-a7ede5}
\end{figure*}
```

For notebooks, import `visualization.adsb_track.plot_track`; it accepts Polars
observation and flight frames and returns a Matplotlib figure.

## Altitude anomaly example

Add `--show-geometric` to compare both altitude channels on the same left axis,
keeping ground speed on the right. Neither channel is treated as ground truth.
The AA216C example shows the entire departure/arrival context around its anomalous
barometric reports (18:26:08–18:29:08 UTC). Flight markers are algorithm estimates;
in particular, its estimated 18:20 departure precedes the observed ground-to-air
transition around 18:26.

```sh
.venv/bin/python src/visualization/adsb_track.py \
  --icao aa216c --date 2026-03-01 --source adsbx \
  --observations tmp/altitude_anomaly_20260301/aa216c_observations.parquet \
  --flights data/flights/algorithm/v1/adsbx/year=2026/month=03/day=01/flights.parquet \
  --event-source 'algorithm flight estimates' \
  --show-geometric --start 18:20 --end 18:38
```
