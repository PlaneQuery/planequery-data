# Evaluation table for the paper

The script now produces **one compact summary table**, grouped by reference
flight dataset: BTS, BTS-verified, SFDPS, SFDPS-verified, Eurocontrol, and
Eurocontrol-verified. Each method/ADS-B provider has a row within its group.
Only reference datasets present in the input are included.
The published OpenSky flights row appears directly before Nearest/OpenSky and
NN/OpenSky within each reference dataset, keeping the OpenSky comparisons together.

With `OUTPUT_DIR` set as for evaluation, read the existing evaluation cache:

```bash
.venv/bin/python scripts/paper/csvs_to_latex_table.py \
  --test-source opensky algorithm-nearest algorithm-nn \
  --adsb-source opensky none \
  --gold-source bts bts-verified sfdps sfdps-verified eurocontrol eurocontrol-verified \
  --reference-date bts=2026-03-01 \
  --reference-date sfdps=2026-03-01 \
  --reference-date eurocontrol=2024-06-01 \
  --target-day-adsb false \
  --require-complete \
  --prefix evaluation_target_day_adsb_false
```

No CSV filenames are needed. The script calls `EvaluationCache.find_stats()`
from the evaluation setup, using `$OUTPUT_DIR/data/cache/evaluation`. It reads
only entries supported by the current evaluation cache version, including the
cache reader's existing validity checks. It never runs evaluations or modifies
cache files. Missing results cause an error if nothing matches; individual
uncached source/date combinations are not invented or recomputed.

Use `--cache-dir /path/to/evaluation` to override the cache root. Cache mode
requires the project's Python environment and `OUTPUT_DIR`, just like the
evaluator. Multiple cached configurations for the same date/method/reference/
provider/input-window combination use the newest cache creation time by default.
Use `--cache-policy error` to reject these ambiguities instead. The manifest
records the selected cache paths, full keys (including source/config versions),
creation times, and stats hashes so the table's inputs can be traced. Cache
selection can include comparisons absent from earlier CSV exports.

To reproduce a particular exported run, pass CSVs explicitly instead:

```bash
.venv/bin/python scripts/paper/csvs_to_latex_table.py \
  first_flights_evaluation.csv second_flights_evaluation.csv \
  --test-source opensky algorithm-nn \
  --adsb-source opensky none
```

Relative filenames resolve under `$OUTPUT_DIR/data/experiments/comparison`;
absolute paths also work without `OUTPUT_DIR`. CSV mode uses only the standard
library. Put filenames before options accepting multiple values. CSV inputs and
`--cache-dir` cannot be combined; input modes are not silently mixed.

All outputs go to `paper/tables/<prefix>/` in the repository. For the command
above, the folder is `paper/tables/evaluation_target_day_adsb_false/`:

- `<prefix>_reference_summary.latex`: the single paper table.
- `<prefix>_reference_summary.csv`: separate numeric columns, including exact
  denominator counts; percentage columns contain values from 0 to 100.
- `<prefix>_manifest.json`: CSV hashes or cache provenance, selected dates, filters and definitions.

Rerunning replaces the same named outputs. Previously generated detailed tables
are left in place; use the new `_reference_summary.latex` file in the paper.

## What appears in the table

The reference heading gives the dataset, date, and total gold ICAOs once.
Columns are:

| Column | Meaning |
| --- | --- |
| Method / ADS-B | Evaluated method and provider; OpenSky alone is its published flight dataset |
| Flights G / T | Retained gold and test flight totals |
| Segmentation P / R (%) | Precision and recall |
| Zero flights ICAOs (%) | Gold ICAOs with no test flights / all gold ICAOs |
| Airport acc. TO / LD (%) | Correct takeoff and landing airports / matched flights |
| Null airports TO / LD (%) | Null takeoff and landing outputs / retained test flights |
| Time bias TO / LD (min) | Existing mean signed takeoff and landing errors; negative means early |

Counts are not repeated next to every percentage. Recover matched flights as
`G * recall / 100`; false negatives as `G - matched`, and false positives as
`T - matched`. Airport-correct counts use matched flights as denominator;
null-output counts use T; missing-aircraft counts use the gold ICAOs in the
heading. Rounded percentages yield approximate counts. The CSV retains exact
G, T, and matched-flight counts and unrounded percentages. Gold-flight totals
can differ by method because the evaluator applies boundary filtering.

Timing uses existing signed-bias columns, not mean absolute error. No new
evaluation metrics or cache changes are required.

## Dates and filters

`--reference-date SOURCE=YYYY-MM-DD` applies to both the reference dataset and
its verified variant. Repeat for each family. If omitted, the script chooses
the latest date shared by the available variants of that family. It never
averages different dates. The chosen dates are printed and recorded in the
manifest. To reproduce the paper selection, retain the three date options in
the example above.

`target_day_adsb=false` is the default. Arguments are inclusion-based: list what
you want. Multiple values and repeated options add to the same selection.
Values within one option are alternatives; different filters apply together.
Omitting a source/provider/date filter keeps all available values for that
field. Filtering happens before duplicate checks and automatic date selection.

| Argument | Example |
| --- | --- |
| `--test-source` | `--test-source algorithm-nn opensky` |
| `--gold-source` | `--gold-source bts bts-verified sfdps sfdps-verified` |
| `--adsb-source` | `--adsb-source opensky none` |
| `--date` | `--date 2024-06-01 2026-03-01` |
| `--target-day-adsb` | `--target-day-adsb false` (also accepts `true` or `unknown`) |

For example, `--test-source opensky --test-source algorithm-nn` includes both
methods. `--tables` and `--metrics` also accumulate repeated values.

Use `--adsb-source opensky none` to include OpenSky-message methods and published
datasets with no provider. Published OpenSky flights have no ADS-B provider
value, so `--adsb-source opensky` alone does not select them. Pair this with
`--test-source` to choose which published datasets and methods you want.
Gold-source filters use exact names; `--reference-date` applies to the whole
family including its verified variant.

In CSV mode, exact duplicate rows are deduplicated and conflicting daily results
are rejected. Cache mode follows the explicit cache-selection policy above.
Existing `average` rows are excluded. `--require-complete` fails before writing
when selected metrics are missing, except undefined timing when no flights
match. Without it, missing metrics produce a warning, blank CSV cells, and
dashes in LaTeX. A dataset's ICAO count must agree across its methods to be
shown once in the heading.

`--decimals` controls LaTeX rounding (default 2). `--prefix` names the subfolder,
files, and labels; without it, the prefix contains the selected date range.
`--output-dir` changes the parent directory; the `<prefix>/` subfolder is always
created beneath it. `--metrics` restricts displayed metrics;
`--list-metrics` lists available keys. CSV denominator counts are preserved.

Optional detailed tables remain accessible with
`--tables segmentation aircraft_coverage airport_accuracy timing_accuracy`.
They retain all filtered dates unless `--reference-date` is supplied. The
default is `--tables reference_summary`.

## Include in the JOAS paper

The JOAS class already loads `booktabs`. Add to the preamble:

```latex
\usepackage{longtable}
```

After copying the generated prefix folder into the paper project's `tables/`:

```latex
\input{tables/evaluation_target_day_adsb_false/evaluation_target_day_adsb_false_reference_summary.latex}
```

Reference it with:

```latex
Table~\ref{tab:evaluation-target-day-adsb-false-reference-summary}
```

Do not wrap the input in a `table` float. The table uses `longtable` and needs
a single-column layout, as used by the JOAS template.
