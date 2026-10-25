# OpenAP / traffic phase segmentation

`__init__.py` exposes `OpenAPFlightSource`; `segmentation.py` contains the
phase-to-flight adapter. The public import is
`from flights.openap import OpenAPFlightSource`.

The `openap` flight source uses OpenAP 2.6.2's `FlightPhase.set_trajectory()`
and `phaselabel()` directly, with its default 60-second windows. This is the
same classifier called by
[traffic's default phase wrapper](https://github.com/xoolive/traffic/blob/master/src/traffic/algorithms/navigation/phases.py),
so these are one baseline, not two independent algorithms. Install the optional
dependency with `pip install -e '.[openap]'`.

The adapter is deliberately small:

- Inputs are timestamps in seconds, altitude in feet, groundspeed in knots
  (as in traffic), and vertical rate in feet/minute. Barometric measurements
  take precedence over geometric fallbacks. Missing ground altitude/rate are
  restored to zero from the canonical `on_ground` flag. Other incomplete or
  nonfinite detector inputs are omitted; there is no interpolation, resampling,
  additional smoothing, or airport-elevation correction.
- Each aircraft is split at observation gaps **over 10 minutes**, matching
  [traffic's default `Flight.split()`](https://github.com/xoolive/traffic/blob/master/src/traffic/core/flight.py).
  OpenAP is run independently on each resulting trace.
- Consecutive `CL`, `CR`, `DE`, and `LVL` observations form a flight. `GND`,
  `NA`, and gaps separate flights. Changes between airborne phases do not split
  flights. First/last airborne observations estimate takeoff/landing; a segment
  must have positive duration. No minimum flight duration or distance is imposed.
- The upstream classifier leaves the last time window `NA`, including every
  point of traces shorter than 60 seconds. This behavior is preserved, so
  endpoints can lose up to approximately a minute of observations.
- Nearest-airport assignment uses the existing dated airport catalog after
  segmentation. Airport guesses do not change the segments or flight matching.

The source supports the normal cache, evaluation, worker, and `--target-day-adsb`
interfaces. Its default input windows match `algorithm-nearest`/`algorithm-nn`:
previous, target, and following UTC days for ADSB.lol/OpenSky, target day for
ADSB Exchange. Only flights starting on the target date are returned.

```bash
# Generate the full flight cache with within-day parallelism.
PYTHONPATH=src python -m flights.cli \
  2026-03-01 2026-03-02 openap --adsb-source adsblol --workers 4

# Evaluate all four references using the standard matcher and metrics.
PYTHONPATH=src python -m flights.evaluation.evaluation \
  --test-source openap --gold-source bts sfdps bts-verified sfdps-verified \
  --adsb-source adsblol --test-dates 2026-03-01

# Export a comparison with algorithm-nn, endpoint timing errors, and provenance.
PYTHONPATH=src python scripts/paper/evaluate_openap.py
```

This measures the phase-to-flight adapter, not phase-label classification
accuracy: the references provide flight intervals, not per-message phase labels.
The standard evaluator uses one-to-one chronological matching with both endpoints
within one hour, restricts predictions to each gold source's aircraft cohort,
and scores departures in the inclusive 01:00–23:00 UTC window after full-day
matching (gold departure when present, otherwise test departure). Thus
flight F1 alone does not establish precise takeoff/landing timing. Verified
references are filtered subsets with their own aircraft cohorts, so precision
can differ even for identical predictions.
