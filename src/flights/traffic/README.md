# Traffic flight source

`TrafficFlightSource` (source name `traffic`) calls **Traffic 2.13's
`Traffic(data).iterate()` with no overrides**. The upstream implementation is
installed as an optional dependency, not copied or reimplemented:

```bash
pip install -e '.[traffic]'
PYTHONPATH=src python -m flights.cli \
  2026-03-01 2026-03-02 traffic --adsb-source adsblol --workers 4

PYTHONPATH=src python -m flights.evaluation.evaluation \
  --test-source traffic --gold-source bts sfdps bts-verified sfdps-verified \
  --adsb-source adsblol --test-dates 2026-03-01
```

The extra also constrains pandas to 2.x: Traffic 2.13 imports `DatetimeTZBlock`,
an internal class removed in pandas 3.

Traffic offers multiple trajectory-analysis methods, rather than one universal
takeoff/landing detector. This source uses its default flight iteration:

- Sort observations by timestamp, group by aircraft **and callsign**, and split
  each group at gaps **strictly greater than ten minutes**.
- Preserve upstream handling of singleton segments (omitted), null callsigns
  (omitted by pandas grouping), empty callsigns (retained), duplicate timestamps,
  and recurring callsigns. A callsign that changes away and back can therefore
  produce interleaved trajectories.
- Preserve ground reports and missing positions/altitudes. Continuous reports
  through a turnaround can join physical flight legs. No phase classifier,
  airborne filter, resampling, smoothing, airport rule, or additional duration
  threshold changes these trajectories. In particular, this does not call
  `Flight.phases()`, which uses the OpenAP classifier already exposed separately.

The adapter changes only the input/output representation: canonical `icao` and
`time` become `icao24` and UTC `timestamp`; rows without aircraft identity or
timestamp are omitted. The first and last observation times populate
`takeoff_time`/`landing_time` and the message-time fields. **These are trajectory
bounds, not detected liftoff/touchdown.** Endpoint coordinates and altitudes are
preserved, and the first non-null registration is retained. Nearest-airport
assignment uses the project's dated catalog after segmentation and cannot
change flight boundaries. It is not Traffic's airport inference.

The source uses complete-aircraft batches of 500, the standard flight cache and
evaluation interfaces, and `--target-day-adsb`. By default it reads the previous,
target, and following UTC days for ADSB.lol/OpenSky, or only the target day for
ADSB Exchange. Only trajectories starting on the requested UTC date are returned.
Changing the pinned Traffic version requires a source-version bump to invalidate
caches. Importing the registry does not require installing Traffic; nonempty
inference reports the installation command if it is missing.

Upstream references: [Traffic.iterate](https://traffic-viz.github.io/api/traffic.core.traffic.html#traffic.core.Traffic.iterate),
[Flight.split](https://traffic-viz.github.io/api/traffic.core.flight.html#traffic.core.Flight.split),
and [the MIT-licensed repository](https://github.com/xoolive/traffic).
