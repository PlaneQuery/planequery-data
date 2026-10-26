# Evaluation table for the paper

Generated tables default to `$OUTPUT_DIR/paper/tables/<prefix>/`, and figures
to `$OUTPUT_DIR/paper/figures/<figure-name>/`. Set `OUTPUT_DIR` to the data root
used for evaluation. Explicit `--output-dir` overrides still work; the candidate
input figure uses `--output`. For standalone table generation from absolute CSV
paths without `OUTPUT_DIR`, supply `--output-dir` explicitly.

Explanatory notes beneath tables are commented out in generated `.tex` files.
Remove the leading `%` from the note line to show them; definitions remain in
the manifests and the commented source.
All table exporters exclude verified flight sources, including from CSV inputs.
Use base reference names (`bts`, `sfdps`, `eurocontrol`) in source filters.

## Airport-NN training curve

`airport_nn_training_curve.py` plots neural-network validation accuracy versus
the number of distinct training days, with a nearest-airport baseline. Each
point is a separate model from the saved training-size sweep:

```bash
.venv/bin/python scripts/paper/airport_nn_training_curve.py
```

Outputs are a vector PDF with embedded fonts, PNG, editable SVG, plotted values
as CSV, and a LaTeX figure snippet. Without overrides,
the `--sweep-dir` defaults to
`$OUTPUT_DIR/data/experiments/flights_airport_nn/airport-nn-sweep-59`
and the output directory to `$OUTPUT_DIR/paper/figures/airport_nn_training_curve`
(set `OUTPUT_DIR`, or supply both paths explicitly).
The figure is 142 mm wide by 65 mm tall, matching the JOAS template's text width,
with 8--9 pt labels and a dashed model curve with visible circular markers.
The gray baseline has no markers, keeping it distinct in grayscale printing.
Connecting lines are a visual guide between measured model results.
It uses the template's Linux Libertine font when available through
`kpsewhich`, with DejaVu Serif as a fallback. Training details appear in the
LaTeX caption rather than an in-plot title. Copy the output folder into the
paper's `figures/` directory, then insert:

```latex
\input{figures/airport_nn_training_curve/airport_nn_training_curve.tex}
```

The sweep uses 1, 2, 5, 10, 15, 20, 30, 40, 50, and 59 training dates, taking
chronological prefixes starting February 1, 2026. For each model, the script
selects the epoch with minimum mean daily validation cross-entropy and verifies
that its history metrics match the saved validation reports. Accuracy pools
the saved primary endpoint metrics with equal weight per endpoint, matching
the sweep's `summary.csv`. All models validate on the same 207,714 endpoints
from June 1–5, 2026, against BTS and FAA SFDPS reference airport labels using
ADSB.lol observations. The script checks that validation dates, label sources,
network, daily endpoint counts, and nearest-airport baselines agree across models.
The CSV records each model's actual training-day count, selected epoch, accuracy,
and source directory. The caption identifies the validation sources and
checkpoint-selection rule.

## Reference-flight aircraft categories and airport characteristics

`reference_flight_breakdown_table.py` exports two matching `booktabs`/`longtable`
tables for **BTS and SFDPS on 2026-03-01** and
**Eurocontrol on 2024-06-01**:

```bash
OUTPUT_DIR=/Volumes/T2-SSD/planequery .venv/bin/python scripts/paper/reference_flight_breakdown_table.py
```

The aircraft table uses ADS-B emitter `category` (A1, A2, A3, etc.), with category
names, illustrative aircraft, and weight definitions. For example, Boeing 737
is an A3 example; `aircraft_type` instead contains model codes such as B738.
Reported categories are preserved, including anomalies and unrecognized codes;
missing categories are not imputed from aircraft models. Categories are enriched
by the reference providers: OpenAirframes for BTS/Eurocontrol and ADS-B
observations for SFDPS. These are not independently verified classifications.
The LaTeX table combines B0 through D2 into one "Other reported categories" row,
calculated from their summed counts before rounding. The CSV retains each code.
Each table has one block with three dataset columns. Percentages use each
dataset's flight count, shown once in its column heading.

The airport table shows **takeoff / landing percentages** for small, medium,
large, and unknown airports. Heliport through Closed rows are omitted.
Code availability has one row, **No IATA or ICAO**, counting resolved airports
with both explicit `iata_code` and `icao_code` fields blank. Unresolved airports
and absent code metadata are excluded from this numerator. LAX / KLAX is an
example of an airport with both codes. An `ident` or `gps_code` is not treated
as evidence of an ICAO code. The CSV retains all airport types and code groups;
all displayed percentages still use the full flight count as denominator.

Airport sizes use the flight-date catalog. Code availability uses **one shared
dated snapshot for all three datasets** (the latest in local history by default).
The 2024 OurAirports schema lacked an explicit `icao_code` field, so treating its
nulls as missing codes would falsely classify major airports as IATA only.
The table records the selected snapshot date and states that code availability
describes that snapshot, not historical assignments. Use
`--airport-code-date 2026-10-25` to pin it; snapshots without ICAO metadata are rejected.

Both tables describe **all flights in the current-version reference caches**,
after provider preprocessing but before evaluation matching/boundary filtering.
Each flight contributes once to the aircraft table and once to each airport
endpoint distribution. Repeated airport visits count repeatedly. Missing values
remain in denominators; each full CSV distribution sums to 100% before rounding.
The airport table displays selected rows, so its displayed rows need not sum to 100%.
These are flight-weighted shares, not shares of unique aircraft or airports.

The generator reads existing cache files and airport history without refreshing,
downloading, or regenerating data. It checks flight manifests against the
flight-date catalog, including the existing one-month grace policy, and fails
if a required cache is absent or stale. Eurocontrol uses its default ADSB.lol
enrichment cache. Its population retains the provider's registration-matching
requirements, so this is not a count of all raw Eurocontrol records.

Outputs default to `$OUTPUT_DIR/paper/tables/reference_flight_breakdown/`:

- `reference_flight_breakdown_aircraft.tex`: category percentages and definitions.
- `reference_flight_breakdown_airports.tex`: airport size/type and code percentages.
- `reference_flight_breakdown.csv`: exact counts, denominators, and unrounded percentages (0–100).
- `reference_flight_breakdown_manifest.json`: source versions, dates, cache/history hashes,
  catalog provenance, definitions, and documentation URLs.

Use `--output-dir outputs/paper/tables` for a workspace-local export, `--prefix`
to change the folder/file names, and `--decimals` to change displayed precision.
Include in a full-width/single-column area without a surrounding table float:

```latex
% Preamble:
\usepackage{booktabs,longtable,array}
% Body:
\input{tables/reference_flight_breakdown/reference_flight_breakdown_aircraft.tex}
\input{tables/reference_flight_breakdown/reference_flight_breakdown_airports.tex}
```

Labels are `tab:reference-flight-breakdown-aircraft` and
`tab:reference-flight-breakdown-airports`. Definitions follow
[ADS-B Exchange's emitter-category documentation](https://support.adsbexchange.com/hc/en-us/articles/44705224053517-Emitter-Category-ADS-B-DO-260B-2-2-3-2-5-2)
and the [OurAirports data dictionary](https://ourairports.com/help/data-dictionary.html).

## Airport NN candidate input figure

`airport_nn_candidate_view.py` creates separate figures for A96262's departure
from US-8169 and its preceding arrival on April 1, 2026, using the ADS-B Exchange
extract already saved for the paper. Each figure shows six candidates.
It calls the production endpoint sampler and feature builder, with identical
plot scales across airports. It reads the existing airport-history snapshot
without refreshing it and never downloads data.

```bash
OUTPUT_DIR=/Volumes/T2-SSD/planequery .venv/bin/python scripts/paper/airport_nn_candidate_view.py
```

PNG, editable SVG, vector PDF, all 32 candidate records, sampled input CSV, and provenance
are saved to `$OUTPUT_DIR/paper/figures/a96262_airport_nn_inputs/`. The source extract omits
heading and vertical rate, so the figure displays the available spatial inputs
and does not compute NN scores. The figure includes KCXO and highlights US-8169
as the correct airport, without displaying distance-rank numbers.
The departure uses its first ten minutes and the six nearest candidates, with
the X labeled “First Message” and color labeled “Minutes from first message.”
The arrival uses its last ten minutes and the original six arrival candidates,
with “Last Message” and “Minutes from last message” (negative offsets).
The arrival at US-8169 is inferred from the next departure; touchdown is
unobserved. Both figures use the production sampler.

Outputs use the basenames `a96262_airport_nn_takeoff_candidate_inputs` and
`a96262_airport_nn_landing_candidate_inputs`, including separate candidate CSVs,
sampled-input CSVs, and provenance manifests. Both are generated by default;
use `--endpoint takeoff` or `--endpoint landing` to generate only one.

## Algorithm comparison on OpenSky

`algoirthm_comparision.py` compares **Sun, plane-notify, and OpenAP**
in the same grouped layout as the other paper tables.
It shows retained gold flights once in each dataset heading, **Test flights T**, segmentation
**P / R / F1 (%)**, and airport matches **TO / LD (%)**.
Sun uses source `junzis`, and OpenAP uses source `openap`.
All three methods use OpenSky ADS-B messages.

```bash
OUTPUT_DIR=/Volumes/T2-SSD/planequery PYTHONPATH=src .venv/bin/python scripts/paper/algoirthm_comparision.py
```

The references are BTS and SFDPS on **2026-03-01**, and Eurocontrol on
**2024-06-01**. Verified datasets are excluded from this preset.
All nine comparisons must be present before files are written. Use `--gold-source`
to select fewer reference datasets; all three methods are still required for each.
The default input-window setting is `--target-day-adsb false`, matching the
overall evaluation table. Pass `--target-day-adsb true` only when those results
have been evaluated. The generator reads existing current-version caches or
explicit evaluation CSVs; it never runs missing evaluations. Cache selection
uses the shared exporter's validity checks and latest-run policy.

Outputs are `algorithm_comparison.tex`, `algorithm_comparison.csv`, and
`algorithm_comparison_manifest.json` under
`$OUTPUT_DIR/paper/tables/algorithm_comparison/`. The CSV retains source IDs,
exact denominator counts, and unrounded percentages; the manifest records input
provenance. Airport percentages divide by matched flights, with null predictions
counted incorrect. Segmentation uses the evaluator's retained flight counts.

Use `--output-dir outputs/paper/tables` for a workspace-local export.
`--prefix`, `--decimals`, `--cache-dir`, and `--cache-policy latest|error` follow
the other table presets. Absolute CSV inputs work without `OUTPUT_DIR` when
`--output-dir` is supplied.

```latex
\usepackage{booktabs,longtable,multirow}
% In a single-column/full-width area, without a table float:
\input{tables/algorithm_comparison/algorithm_comparison.tex}
```

Reference with `Table~\ref{tab:algorithm-comparison}`.

## Table 1: ADS-B source comparison

`adsb_source_comparision_table.py` is the preset for Table 0's ADS-B-provider
companion. It reuses `csvs_to_latex_table.py` for cache/CSV loading, metric
definitions, percentage formatting, and the grouped LaTeX layout.

```bash
PYTHONPATH=src .venv/bin/python scripts/paper/adsb_source_comparision_table.py
```

Set `OUTPUT_DIR` to the evaluation data root (for example,
`/Volumes/T2-SSD/planequery`). The preset selects:

- BTS and SFDPS on **2026-03-01**.
- Eurocontrol on **2024-06-01**.
- ADSB.lol, OpenSky messages, and ADSBX, all with **target-day-only ADS-B**.
- Nearest and NN, showing retained gold counts once in the dataset heading and **Test flights T**,
  segmentation **P / R / F1 (%)**, **Zero flights ICAOs (%)**, and airport accuracy
  **takeoff / landing (%)**. Zero flights is the percentage of gold ICAOs with
  no test flights before matching or boundary filtering.
  Shared test-flight counts and segmentation cells span the two methods.

Use `--test-source algorithm-nn` for NN only. The preset requires all 18
comparisons (9 for NN only) and their selected metrics and denominator counts
before writing anything. The generic exporter can display these same metrics
with `--metrics` and also requires its selected reference/provider/method grid.

By default it reads only the current supported evaluation cache version; it
does not run missing evaluations or substitute old cache versions. Explicit
evaluation CSVs can be supplied before the options, with the same path and
conflicting-duplicate rules as the generic exporter. Use compatible evaluation
runs; CSV inputs cannot establish that evaluator/source versions agree.

To populate missing results with the current evaluator, run the two date groups
separately (these commands may process data and run NN inference):

```bash
PYTHONPATH=src .venv/bin/python src/flights/evaluation/evaluation.py \
  --test-source algorithm-nearest algorithm-nn \
  --gold-source bts sfdps \
  --adsb-source adsblol opensky adsbx \
  --test-dates 2026-03-01 --target-day-adsb

PYTHONPATH=src .venv/bin/python src/flights/evaluation/evaluation.py \
  --test-source algorithm-nearest algorithm-nn \
  --gold-source eurocontrol \
  --adsb-source adsblol opensky adsbx \
  --test-dates 2024-06-01 --target-day-adsb
```

Then rerun the table preset. Outputs default to
`$OUTPUT_DIR/paper/tables/table_1_adsb_sources/`:

- `table_1_adsb_sources.tex`: the paper table.
- `table_1_adsb_sources.csv`: exact counts and unrounded percentages (0–100).
- `table_1_adsb_sources_manifest.json`: filters, dates, definitions, and input provenance.

`--output-dir` changes the parent directory, for example to
`/Users/jonahgoode/Documents/PlaneQuery/Code/planequery-data/paper/tables`
to place the folder beside Table 0. `--prefix` defaults to `table_1_adsb_sources`.
`--cache-dir`, `--cache-policy latest|error`, and `--decimals` are also supported.

With `\usepackage{booktabs,longtable,multirow}`, include the table using:

```latex
\input{tables/table_1_adsb_sources/table_1_adsb_sources.tex}
```

Reference `Table~\ref{tab:table-1-adsb-sources}`. Use a full-width/single-column
area without a surrounding `table` float.

## Training versus nontraining airport accuracy

`training_vs_non_training_table.py` compares Nearest and NN on **Eurocontrol only**
(excluding its verified variant). It reuses the existing airport-exposure evaluator and the summary script's
cache selection, reference dates, ordering, and LaTeX escaping. No new evaluation
metrics are introduced.

From the repository root, using the same `OUTPUT_DIR` as evaluation:

```bash
.venv/bin/python scripts/paper/training_vs_non_training_table.py \
  --adsb-source opensky \
  --gold-source eurocontrol \
  --reference-date eurocontrol=2024-06-01 \
  --require-complete \
  --prefix table_1
```

By default, it reads current-version cached comparisons and uses `training.json`
beside the current `DEFAULT_MODEL_PATH`. It does not regenerate flights, run
inference, or compute missing comparisons. It can create the breakdown even if
evaluation was originally run without `--airport-nn-training`. For a custom
model, supply its matching `--airport-nn-training /absolute/path/training.json`
and select the corresponding evaluation cache. The script does not infer or
validate checkpoint ownership from a training manifest. Relative manifest paths
resolve under `OUTPUT_DIR`, as in evaluation.

Alternatively, supply evaluator-generated `*_airport_exposure.csv` files before
the options. Relative filenames resolve under
`$OUTPUT_DIR/data/experiments/comparison`; absolute paths work without
`OUTPUT_DIR` when `--output-dir` is supplied. CSV mode uses the training membership already recorded in those
files, requires one shared manifest hash, and rejects conflicting duplicate rows.
It does not combine CSV inputs with cache or training-manifest options.

The default methods are `algorithm-nearest` and `algorithm-nn`. `--gold-source`
defaults to and accepts only `eurocontrol`. Inclusion filters `--test-source`,
`--adsb-source`, and `--date` accept multiple values or repeated options.
`--reference-date eurocontrol=YYYY-MM-DD` selects the date; otherwise the latest
available date is used. `--target-day-adsb` defaults to `false`.
`--cache-dir`, `--cache-policy latest|error`, `--decimals`, and `--output-dir`
follow the summary script's conventions. `--require-complete` checks that all
explicitly selected reference sources and all selected methods/endpoints/groups
are present within the available reference/date/provider contexts. A group
with zero matched endpoints is valid and displays a dash, rather than 0%.

The compact table names Eurocontrol and its date in the caption, with no dataset
section heading or continuation headings. It has Nearest and NN rows, with training
and nontraining matched counts and accuracy percentages, each shown as takeoff / landing.
Identical count pairs share a cell spanning Nearest and NN for the same ADS-B
provider. Both endpoint counts must match exactly; differing or unavailable
counts stay in separate cells. Accuracy and drop cells remain method-specific.
The final **Drop (pp)** column gives training minus nontraining accuracy for
each method, separately for takeoff and landing. Positive means lower accuracy
on nontraining airports; negative means higher accuracy. Differences use the
unrounded rates and are undefined if either group has no matched endpoints.
The CSV includes `accuracy_drop_pp` on both group rows for each method/endpoint;
it is the same paired difference, not a value to sum across groups.
Accuracy is the existing `airport_accuracy`: correct airport IDs divided by
matched endpoints with known gold IDs, with null predictions counted incorrect.
Membership is based on eligible positive NN training labels, combining takeoff
and landing labels; validation labels and unknown gold IDs are excluded.
The same training-airport set is applied to both methods. One ADS-B provider is
named once in the note; multiple providers are named beside each method.

Outputs are under `$OUTPUT_DIR/paper/tables/<prefix>/`:

- `<prefix>_training_vs_non_training.tex`: compact paper table.
- `<prefix>_training_vs_non_training.csv`: one row per method/reference/group/
  endpoint, preserving exact counts and provenance. All rate columns are renamed
  with `_percent` and scaled to 0–100. This is a table export, not an evaluator
  input CSV.
- `<prefix>_training_vs_non_training_manifest.json`: input/cache provenance,
  training manifest hash, filters, dates, and metric definitions.

Include with `\usepackage{booktabs,longtable,multirow}` and
`\input{tables/table_1/table_1_training_vs_non_training.tex}` after copying the
folder into the paper project. Use a single-column/full-width area, without a
surrounding `table` float. Reference with
`Table~\ref{tab:table-1-training-vs-non-training}`.

## Overall evaluation table

The script now produces **one compact summary table**, grouped by reference
flight dataset: BTS, SFDPS, and Eurocontrol. Each method/ADS-B provider has a row within its group.
Explicitly requested methods and reference datasets must all be present.
The published OpenSky flights and OpenSky-NN rows appear directly before
Nearest/OpenSky and NN/OpenSky within each reference dataset. OpenSky-NN keeps
OpenSky's flight boundaries and replaces the airport predictions with the NN.

With `OUTPUT_DIR` set as for evaluation, read the existing evaluation cache:

```bash
.venv/bin/python scripts/paper/csvs_to_latex_table.py \
  --test-source opensky opensky-nn algorithm-nearest algorithm-nn \
  --adsb-source opensky none \
  --gold-source bts sfdps eurocontrol \
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
cache files. Missing requested method/reference/provider comparisons cause an
error before any output is written. Missing selected metrics also fail by default.
Current means the current evaluation cache version; no extra source/model
fingerprint freshness policy is applied by the exporter.

`src/flights/cli.py ... opensky-nn` generates flight files, not evaluation
results. To evaluate already-generated OpenSky-NN flights, run these two date
groups in the same environment and checkout used to generate the flights.
Valid flight caches are reused; missing or stale flight caches may trigger
regeneration. These commands create the three missing reference comparisons:

```bash
PYTHONPATH=src .venv/bin/python src/flights/evaluation/evaluation.py \
  --test-source opensky-nn \
  --gold-source bts sfdps \
  --test-dates 2026-03-01

PYTHONPATH=src .venv/bin/python src/flights/evaluation/evaluation.py \
  --test-source opensky-nn \
  --gold-source eurocontrol \
  --test-dates 2024-06-01
```

Use the same `OUTPUT_DIR` as the table export. Omit `--target-day-adsb` for these
sources. Then rerun the table command above (use `--prefix table_0` for Table 0).
An explicitly requested `opensky-nn` now fails if any selected comparison is
missing. Published OpenSky and OpenSky-NN use the provider-independent evaluation
metadata value `none`, so keep `--adsb-source opensky none` when exporting them
alongside Nearest/NN. The model still uses OpenSky observations internally.

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
absolute paths also work without `OUTPUT_DIR` when `--output-dir` is supplied. CSV mode uses only the standard
library. Put filenames before options accepting multiple values. CSV inputs and
`--cache-dir` cannot be combined; input modes are not silently mixed.

All outputs default to `$OUTPUT_DIR/paper/tables/<prefix>/`. For the command
above, the folder is `$OUTPUT_DIR/paper/tables/evaluation_target_day_adsb_false/`:

- `<prefix>_reference_summary.tex`: the single paper table.
- `<prefix>_reference_summary.csv`: separate numeric columns, including exact
  denominator counts; percentage columns contain values from 0 to 100.
- `<prefix>_manifest.json`: CSV hashes or cache provenance, selected dates, filters and definitions.

Rerunning replaces the same named outputs. Previously generated detailed tables
are left in place; use the new `_reference_summary.tex` file in the paper.

## What appears in the table

The reference heading gives the dataset, date, total gold ICAOs, and retained gold
flights once. Gold-flight counts must agree across methods and providers in each
reference/date/input-window group; mismatches fail before output files are replaced.
Columns are:

| Column | Meaning |
| --- | --- |
| Method | Evaluated method; OpenSky denotes its published flight dataset |
| Test flights T | Retained test flight total; gold total G is in the heading |
| Segmentation P / R / F1 (%) | Precision, recall, and their harmonic mean (F1) |
| Zero flights ICAOs (%) | Gold ICAOs with no test flights / all gold ICAOs |
| Airport acc. TO / LD (%) | Correct takeoff and landing airports / matched flights |
| Null airports TO / LD (%) | Null takeoff and landing outputs / retained test flights |
| Time bias TO / LD (min) | Existing mean signed takeoff and landing errors; negative means early |

When all ADS-B-based methods use one provider, the table identifies that provider
once in its note and uses short row labels such as `OpenSky`, `Nearest`, and `NN`.
When multiple providers are present, the column becomes `Method / ADS-B` and
provider suffixes remain on the rows. CSV metadata always retains the provider.

For adjacent Nearest and NN rows using the same provider, and for adjacent
OpenSky and OpenSky-NN rows, all metrics from the segmentation, aircraft-coverage,
and timing families can share cells spanning both rows.
This covers counts, precision/recall/F1, zero-flight ICAOs, and timing
errors or threshold recalls. Airport-dependent metrics stay separate, including
any scores requiring correct airports. Cells merge only when the underlying selected values match exactly,
not merely after rounding; missing values are not merged. CSV rows remain complete.

Counts are not repeated next to every percentage. Recover matched flights as
`G * recall / 100`; false negatives as `G - matched`, and false positives as
`T - matched`. Airport-correct counts use matched flights as denominator;
null-output counts use T; missing-aircraft counts use the gold ICAOs in the
heading. Rounded percentages yield approximate counts. The CSV retains exact
G, T, and matched-flight counts and unrounded percentages for every row.
Detailed segmentation and timing tables also put their shared gold count in
the group heading. Reference-population breakdowns already show their counts
once per dataset; airport-exposure counts remain specific to each method/group.

Timing uses existing signed-bias columns, not mean absolute error. No new
evaluation metrics or cache changes are required.

## Dates and filters

`--reference-date SOURCE=YYYY-MM-DD` chooses the date for that reference dataset.
Repeat for each dataset. If omitted, the script chooses its latest available date. It never
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
| `--gold-source` | `--gold-source bts sfdps` |
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
dataset.

In CSV mode, exact duplicate rows are deduplicated and conflicting daily results
are rejected. Cache mode follows the explicit cache-selection policy above.
Existing `average` rows are excluded. The generic exporter always fails before
writing if a requested comparison or selected metric is missing, except undefined
timing when no flights match. `--require-complete` remains accepted for compatibility;
it is now the default and cannot be disabled. Explicit methods/references/providers
define the required grid; omitted filters use the values present in the selected
input. Summary tables still select one date per reference family; detailed exports
require every explicitly selected date. Each dataset's ICAO count must agree across
its methods to be shown once in the heading. Standalone CSV inputs do not carry
cache-version provenance; the current-version check applies to cache input mode.

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
\usepackage{longtable,multirow}
```

After copying the generated prefix folder into the paper project's `tables/`:

```latex
\input{tables/evaluation_target_day_adsb_false/evaluation_target_day_adsb_false_reference_summary.tex}
```

Reference it with:

```latex
Table~\ref{tab:evaluation-target-day-adsb-false-reference-summary}
```

Do not wrap the input in a `table` float. The table uses `longtable` and needs
a single-column layout, as used by the JOAS template.
