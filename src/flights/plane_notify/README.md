# Plane Notify segmentation baseline

`__init__.py` exposes `PlaneNotifyFlightSource`; `segmentation.py` replays
archived observations, and `core.py` contains the upstream event detector.
The public import is `from flights.plane_notify import PlaneNotifyFlightSource`.

```bash
python -m flights.evaluation.evaluation \
  --test-source plane-notify \
  --gold-source sfdps bts-verified sfdps-verified \
  --adsb-source adsblol \
  --test-dates 2026-03-01
```

`plane-notify` version 1 uses the takeoff/landing branches extracted directly
from [Jxck-S/plane-notify, revision e1aa8c6](https://github.com/Jxck-S/plane-notify/blob/e1aa8c668896cd6eaa80c42db813eb0487fb02d9/planeClass.py#L313-L372).
The copied core is in [`core.py`](core.py), with its
upstream GPL-3.0 license alongside it. Notifications, maps, route lookup, and
unrelated alerts are omitted; the event conditions are unchanged. A regression
test executes the unmodified upstream branches against 5,000 deterministic
replay states and compares their events and state with the extracted core.

The adapter uses the example configuration's 60-second polling interval and
five-minute data-loss timeout. At each UTC minute boundary `t`, it replays the
latest observation in `(t - 60 seconds, t]`; no observation means `run_empty()`.
This is an explicit approximation of a live API: archived positions do not
record whether the API would retain an aircraft with stale position data.
The ADSBX-v2 airborne-speed guard (`<= 10 kt`) is retained. Null/invalid
coordinates or required airborne fields are skipped, and canonical on-ground
observations map to upstream's `alt_baro = "ground"`.

The implementation uses **15,000 ft**, despite the older upstream pseudocode
describing 10,000 ft. Its altitude equality behavior, persistent
`landing_plausible` flag, and replacement of a pending takeoff on another
takeoff event are deliberately preserved. No duration, distance, altitude, or
airport-based post-filter is added. Only a detected takeoff followed by a
detected landing with positive duration produces a flight. A first airborne
observation needs an earlier empty poll to trigger data acquisition. Unpaired
landings and pending takeoffs are not converted into partial flights.

Takeoff and observed-ground landing times use the replay clock. Data-loss
landing time uses the last position timestamp, matching upstream's flight-time
calculation. Ground-transition takeoff coordinates come from the preceding
check; other endpoints use the current/last position. The dated airport
catalog's nearest fixed-wing airport replaces upstream's live CSV/geodesic
lookup; airport IDs do not affect segmentation matching. This can change
airport assignments near distance ties, but the detector's altitude condition
also accepts any altitude at or below 15,000 ft regardless of airport elevation.

The source reads the preceding, target, and following UTC days for ADSB.lol
and OpenSky. ADSBX public samples and `--target-day-adsb` use only the target
day. Input-window metadata is carried through batching; no empty polls are
created beyond the requested window. Flights belong to their UTC takeoff day.
The source uses the normal flight and evaluation caches and source registry.

Evaluation retains the gold source's aircraft cohort, including gold-only
aircraft as false negatives. Matching requires both endpoint times to be
within one hour and does not require matching airports. Full-day matching is
followed by scoring departures in the inclusive
01:00–23:00 UTC window, using the gold departure when present and the test
departure otherwise.
Precision, recall, and F1 therefore measure flight detection under this
matching policy, rather than exact takeoff/landing timestamp accuracy.
