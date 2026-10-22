# Evaluation table for the paper

The script now produces **one compact summary table**, grouped by reference
flight dataset: BTS, BTS-verified, SFDPS, SFDPS-verified, Eurocontrol, and
Eurocontrol-verified. Each method/ADS-B provider has a row within its group.
Only reference datasets present in the input are included.
The published OpenSky flights row appears directly before Nearest/OpenSky and
NN/OpenSky within each reference dataset, keeping the OpenSky comparisons together.

```bash
.venv/bin/python scripts/paper/csvs_to_latex_table.py \
  2026-10-22_02-01-25-644740_ae715f2f_flights_evaluation.csv \
  2026-10-22_02-13-57-160131_ae715f2f_flights_evaluation.csv \
  2026-10-22_02-55-08-101981_ae715f2f_flights_evaluation.csv \
  2026-10-22_03-59-47-862124_ae715f2f_flights_evaluation.csv \
  2026-10-22_04-29-08-704468_ae715f2f_flights_evaluation.csv \
  2026-10-22_04-30-03-530001_ae715f2f_flights_evaluation.csv \
  2026-10-22_04-38-11-779872_ae715f2f_flights_evaluation.csv \
  --reference-date bts=2026-03-01 \
  --reference-date sfdps=2026-03-01 \
  --reference-date eurocontrol=2024-06-01 \
  --require-complete \
  --prefix evaluation_target_day_adsb_false
```

Relative CSV filenames resolve under `$OUTPUT_DIR/data/experiments/comparison`.
Absolute paths also work. Put filenames before options that accept multiple
values. The script uses only the standard library and does not run evaluations.

Outputs go to `paper/figures/evaluation_tables/` in the repository:

- `<prefix>_reference_summary.latex`: the single paper table.
- `<prefix>_reference_summary.csv`: separate numeric columns, including exact
  denominator counts; percentage columns contain values from 0 to 100.
- `<prefix>_manifest.json`: input hashes, selected dates, filters and definitions.

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

`target_day_adsb=false` is the default. Other filters remain available:
`--test-source`, `--gold-source`, `--adsb-source`, `--date`, and
`--target-day-adsb true|false|unknown`.

Exact duplicate rows are deduplicated; conflicting daily results are rejected.
Existing `average` rows are excluded. `--require-complete` fails before writing
when selected metrics are missing, except undefined timing when no flights
match. Without it, missing metrics produce a warning, blank CSV cells, and
dashes in LaTeX. A dataset's ICAO count must agree across its methods to be
shown once in the heading.

`--decimals` controls LaTeX rounding (default 2). `--prefix` names the files and
labels; without it, the prefix contains the selected date range. `--output-dir`
changes the destination. `--metrics` restricts displayed metrics;
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

After copying the generated summary into the paper project's `figures/tables/`:

```latex
\input{figures/tables/evaluation_target_day_adsb_false_reference_summary.latex}
```

Reference it with:

```latex
Table~\ref{tab:evaluation-target-day-adsb-false-reference-summary}
```

Do not wrap the input in a `table` float. The table uses `longtable` and needs
a single-column layout, as used by the JOAS template.
