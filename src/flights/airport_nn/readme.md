# Airport neural network

A 38,082-parameter GRU ranks nearby airports from each end of a segmented ADS-B
flight. Inputs are airport-relative position, altitude, motion and airport
characteristics. Aircraft/airport identities and reference labels are not inputs.
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
Checkpoint selection averages daily weighted cross entropy. Large/medium/small
airports have loss weights 1/2/4.
Early stopping follows three epochs without improvement.

Preparation independently segments observations before matching references by
aircraft and time, allowing 15 minutes at each end. Ambiguous matches, overlapping
references and cross-UTC-day flights are excluded. BTS labels are accepted
directly; SFDPS-only labels require three ground/low observations within 2 km over
at least ten seconds. This supervision is imperfect and favors some traffic and
reception conditions; measured agreement is not accuracy on all flights.

Caches contain feature arrays, endpoint labels and a source manifest. Labels,
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
`data/flights/source=algorithm-nn/adsb_source=…/version=5/`. The model is not yet
part of the source version, so regenerate a day when changing checkpoints. The
default checkpoint remains
`data/models/flights_airport_nn/algorithm-airport-refactored/model.pt`.

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
