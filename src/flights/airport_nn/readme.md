# Airport neural network

A 37,314-parameter GRU ranks nearby airports from each end of a segmented ADS-B
flight. Inputs are airport-relative position, altitude, motion and endpoint state.
Airport size/type, scheduled-service status, aircraft/airport identities, and
reference labels are not inputs.
Training, evaluation and inference automatically prefer CUDA, then Apple MPS,
and fall back to the CPU.

## Training

Install PyTorch with `pip install -e '.[airport-nn]'`.

The fixed split is:

- Optimizer: February 1–20, 2026, ADSB.lol with BTS/SFDPS labels.
- Validation: February 21–27, 2026.
- Unused buffer: February 28, 2026.
- Benchmarks: March 1, 2026 and June 1, 2024. These have been examined previously
  and are not untouched tests.

Prepare February data from local ADSB.lol observations and derived BTS/SFDPS
flights under the configured storage root:

```bash
python -m flights.airport_nn.data --workers 4 --output /path/to/february-cache
```

Train from the prepared cache:

```bash
python -m flights.airport_nn.train \
  --cache /path/to/february-cache --output /path/to/new-model
```

Training starts from random weights and refuses to overwrite a model directory.
It saves `model.pt`, input provenance, epoch history and validation predictions.
Checkpoint selection averages daily cross entropy. Every eligible endpoint has equal
loss weight, regardless of airport size.
Early stopping follows three epochs without improvement.

To sweep **1, 2, 5, 10, 15, 20 and 24 training days**, run from the repository
root with a fresh output directory:

```bash
python scripts/train_sweep.py --output outputs/airport-nn-sweep-24
```

The sweep uses the configured February cache (override with `--cache`). Each
model trains on February 1 through N and validates on **February 25–27** so all
seven sizes share the same held-out dates. This differs from the standalone
trainer's split above; rerun smaller sizes with this split for a comparable curve.
Each `days-NN/` directory contains `history.json`, `model.pt`, `training.json`
and per-day validation reports in `validation/YYYY-MM-DD/metrics.json`.

Preparation independently segments observations before matching references by
aircraft and time, allowing 15 minutes at each end. Ambiguous matches, overlapping
references and cross-UTC-day flights are excluded. BTS and SFDPS labels are accepted
without requiring ground/low-altitude observations near the labeled airport.
Conflicting airport labels are excluded. This supervision is imperfect; measured
agreement is not accuracy on all flights.

Caches contain 20-column feature arrays, endpoint labels and a source manifest.
Caches produced by earlier feature versions are incompatible and must be rebuilt.
Labels,
candidate membership, conflicts, targets and evaluation correctness all use the
persistent OurAirports `id`; airport identifiers are retained only for display.
Changed preparation inputs require a new cache directory. Candidate misses remain
in accuracy denominators but cannot contribute to training cross entropy.

## Inference

```bash
python -m flights.airport_nn.derive_day 2026-03-01 --source adsblol \
  --model /path/to/new-model/model.pt
python -m flights.evaluation.evaluation \
  --test-source algorithm-nn --gold-source ground-truth \
  --adsb-source adsblol \
  --test-dates 2026-03-01 --airport-nn-model /path/to/new-model/model.pt
```

Daily evaluation results are cached under `OUTPUT_DIR/data/cache/evaluation`,
so the same date and source combination does not need to be recomputed. A
normal evaluation reads cache hits and creates only the missing entries. Use
`python -m flights.evaluation.evaluation --list-cache` to list all cached days,
or add `--test-source`, `--gold-source`, and `--adsb-source` to filter the list.
Both normal evaluations and cache listings accept `--output-csv`. Add
`--columns` to choose and order CSV columns; when it is omitted, every
applicable column is written.

```bash
python -m flights.evaluation.evaluation \
  --test-source algorithm-nn --gold-source bts \
  --adsb-source adsblol adsbx --test-dates 2026-03-01 \
  --airport-nn-model /path/to/new-model/model.pt \
  --output-csv /tmp/algorithm-nn-evaluation.csv \
  --columns test_date adsb_src precision recall

python -m flights.evaluation.evaluation --list-cache \
  --test-source algorithm-nn --gold-source bts \
  --adsb-source adsblol adsbx \
  --output-csv /tmp/cached-algorithm-nn-evaluation.csv \
  --columns test_date adsb_src precision recall
```

Inference supports `adsblol`, `adsbx`, and `opensky`. It preserves the standard Flight schema
and all fields except the two airport references (stable IDs and identifiers).
Missing observations produce null airport references. Daily results use the
canonical FlightSource layout under
`data/flights/source=algorithm-nn/adsb_source=…/version=9/`. This version and
`opensky-nn` version 4 use the February 2026 20-day checkpoint by default.
Custom checkpoints still require regenerating the corresponding flight cache.
The default checkpoint is
`outputs/airport-nn-february-2026-10-21-days-20/model.pt`.

For a provider comparison with the same ADS-B observation window, use:

```bash
python -m flights.evaluation.evaluation \
  --test-source algorithm-nn --gold-source bts \
  --adsb-source adsblol adsbx opensky --test-dates 2026-03-01 \
  --target-day-adsb
```

This reads exactly `[2026-03-01 00:00, 2026-03-02 00:00)` UTC for every
provider, before segmentation and airport inference. The flag applies to
ADS-B flight sources on either side of the comparison. It changes the input
window, not the flight-scoring rules. Omitting it preserves provider-specific
windows. The flag takes no value. Each date in `--test-dates` gets its own window.

Target-day-only runs have separate flight and evaluation caches, including
reused `algorithm-nearest` segments. CSV output includes the boolean
`target_day_adsb`. Use `--list-cache --target-day-adsb` to list only evaluations
using the target day's observations.

To generate flight files with the same target-day-only inputs, use:

```bash
python -m flights.cli 2026-03-01 2026-03-02 algorithm-nn \
  --adsb-source adsblol --target-day-adsb
```

The flag applies separately to each day in the `[start, end)` range and requires
an ADS-B flight source. Bad-day checks, release preparation, and
`--no-save-adsb-downloads` cleanup use only the target days in this mode.

The canonical `flights.cli` command automatically reuses a matching cached
`algorithm-nearest` day and reads only the trajectory columns needed by the NN.
If that day is absent or its airport-catalog manifest is stale, inference falls
back to segmenting the ADS-B observations normally.
Reuse-only reads use larger aircraft batches because they do not materialize
the segmentation message objects. For a single-day `flights.cli` range these
batches run in parallel; multi-day ranges parallelize whole days instead.
Within-day NN workers evaluate on CPU to avoid several processes contending for
the Mac's single MPS command queue.

To score a prepared validation cache:

```bash
python -m flights.airport_nn.evaluate --model /path/to/new-model/model.pt \
  --cache /path/to/february-cache --dates 2026-02-21 2026-02-22 \
  --output /path/to/scores
```

For a one-day preparation/training smoke run:

```bash
python -m flights.airport_nn.data --dates 2026-03-01 --workers 1 \
  --output /tmp/airport-nn-cache
python -m flights.airport_nn.train --cache /tmp/airport-nn-cache \
  --output /tmp/airport-nn-model --epochs 1 --train-dates 2026-03-01 \
  --validation-dates
```

### OpenSky segments with NN airports

The `opensky-nn` source keeps the published `opensky` flights (including their
`firstseen`/`lastseen` boundaries, row order, and metadata) and replaces their
departure/arrival airport IDs and identifiers with the same NN used by
`algorithm-nn`. It always uses canonical OpenSky ADS-B observations. The
`--adsb-source` option does not change this source's observation provider.

```bash
python -m flights.cli 2026-03-01 2026-03-02 opensky-nn
python -m flights.evaluation.evaluation \
  --test-source opensky-nn --gold-source bts \
  --test-dates 2026-03-01
```

Both commands accept `--airport-nn-model /path/to/model.pt`; the default is the
same checkpoint as `algorithm-nn`. Published flights are read through the
catalog-aware `opensky` cache. Building that cache requires the departure day's
and following day's downloaded `flights_data4` partitions. OpenSky observation
data must cover the flights through their latest arrivals, including overnight
arrivals.

Inference reads observations in batches of at most 2,048 published flights.
For each flight, it selects the same aircraft's first and last ten minutes,
clipped to the supplied boundaries, and samples at most 32 observations per
endpoint. Original OpenSky airport labels are not model inputs. Flights without
usable endpoint observations remain in the output with null airport references.

Output is cached under
`data/flights/source=opensky-nn/adsb_source=opensky/version=4/` with OpenSky
observation provenance. As with `algorithm-nn`, regenerate flight files with
`flights.cli` when changing checkpoints, then use evaluation's `--refresh-cache`.
`--workers` parallelizes days for this source; each day's flight batches run
sequentially. `--target-day-adsb` is not supported for `opensky-nn`.

## Code

- `features.py`: candidate search and trajectory features shared by training/inference.
- `model.py`: ranker and checkpoint loading.
- `data.py`: reference labels, matching and feature preparation.
- `train.py`, `evaluate.py`: optimization, checkpoint selection and scoring.
- `inference.py`, `derive_day.py`: airport assignment and cached daily Flight output.

Run focused checks with:

```bash
pytest -q tests/flights/airport_nn tests/flights/evaluation/test_algorithm_airports_nn.py
```
