"""Show readsb legs and the algorithm's refined endpoints for A0275C.

Reads the cached ADS-B Exchange day; does not download or modify source data.
The first two legs are plotted, but segmentation runs on the entire day.
Pass --observations to reproduce from the exported day extract.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import hashlib
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
import numpy as np
import polars as pl


ROOT = Path(__file__).resolve().parents[2]
STEM = "a0275c_20260301_readsb_segments"
ICAO = "a0275c"
START = datetime(2026, 3, 1)
END = START + timedelta(days=1)
# Saved algorithm-NN airport predictions, keyed by exact endpoint timestamps so
# changed flight boundaries cannot silently inherit labels from a different flight.
AIRPORT_SOURCE = "data/flights/source=algorithm-nn/adsb_source=adsbx/target_day_adsb=true/version=9/year=2026/month=03/day=01/data.parquet"
AIRPORT_LABELS = {
    ("takeoff", "2026-03-01T00:40:33.934000"): "KBOS",
    ("landing", "2026-03-01T06:44:53.580000"): "KLAX",
    ("takeoff", "2026-03-01T08:27:43.655000"): "KLAX",
    ("landing", "2026-03-01T13:08:48.520000"): "KJFK",
}
BLUE, GROUND, TAKEOFF, LANDING = "#0072B2", "#D55E00", "#009E73", "#CC79A7"
STYLE = {"font.family": "serif", "font.size": 9, "axes.labelsize": 9,
         "xtick.labelsize": 8, "ytick.labelsize": 8, "pdf.fonttype": 42,
         "ps.fonttype": 42, "svg.fonttype": "none", "axes.spines.top": False,
         "axes.spines.right": False}


def read_observations(path: Path | None):
    # Production imports require OUTPUT_DIR even when using a portable extract.
    if path is not None:
        os.environ.setdefault("OUTPUT_DIR", str(ROOT))
    elif not os.environ.get("OUTPUT_DIR"):
        raise ValueError("Set OUTPUT_DIR to the existing data root or pass --observations.")
    from data_engineering.adsb.canonical import ADSBSource, adsb_day_files, scan_adsb_parquet
    from flights.segmentation.algorithm import ADSB_ALGORITHM_COLUMNS

    if path is not None:
        sources = [path.resolve()]
        scan = pl.scan_parquet(path)
    else:
        sources = adsb_day_files(ADSBSource.ADSBX, START.date(), icaos=[ICAO])
        scan = scan_adsb_parquet(ADSBSource.ADSBX, [START.date()],
                                 ADSB_ALGORITHM_COLUMNS, icaos=[ICAO])
    if scan.collect_schema()["time"].time_zone is not None:
        scan = scan.with_columns(pl.col("time").dt.convert_time_zone("UTC")
                                 .dt.replace_time_zone(None))
    # Match read_adsb's [start, end) window and _segment_adsb_df's validity filter.
    data = scan.filter((pl.col("icao").str.to_lowercase() == ICAO)
                       & (pl.col("time") >= START) & (pl.col("time") < END)
                       & pl.col("lat").is_not_null() & pl.col("lon").is_not_null()
                       ).sort("time").collect()
    if data.is_empty():
        raise ValueError("No cached ADS-B Exchange observations for A0275C on 2026-03-01.")
    return data, sources


def segment_day(data):
    from flights.segmentation.algorithm import _rows_to_position_messages, identify_flight_segments
    from flights.segmentation.readsb_legs import mark_leg_indexes

    messages = _rows_to_position_messages(data)
    markers = mark_leg_indexes(messages)
    cuts = [0, *markers, len(messages)]
    legs = [{"segment": i, "start_index": a, "end_index_exclusive": b,
             "first_message": messages[a].time, "last_message": messages[b - 1].time}
            for i, (a, b) in enumerate(zip(cuts, cuts[1:]), 1) if a < b]
    flights = identify_flight_segments(messages)
    if len(legs) < 2:
        raise ValueError("This case requires at least two readsb segments.")
    visible = [f for f in flights if legs[0]["first_message"] <= f.takeoff_time
               and f.landing_time <= legs[1]["last_message"]]
    if len(visible) != 2:
        raise ValueError(f"Expected two flights in the illustrated segments, got {len(visible)}.")
    for flight in visible:
        if flight.takeoff_message.on_ground is not False or flight.landing_message.on_ground is not True:
            raise ValueError("Illustrated endpoints must have observed airborne/ground transitions.")
    return legs, flights, visible


def stamp_label(stamp):
    # Round display labels to the nearest second; exports retain milliseconds.
    return (stamp + timedelta(microseconds=500_000)).strftime("%H:%M:%S")


def plot_case(data, legs, flights):
    shown = data.slice(0, legs[1]["end_index_exclusive"])
    t = mdates.date2num(shown["time"].to_list())
    boundary = legs[1]["first_message"]
    xmax = legs[1]["last_message"] + timedelta(minutes=10)
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(7.2, 3.8))
        ax = fig.add_axes([.105, .255, .87, .43])
        states = fig.add_axes([.105, .195, .87, .035], sharex=ax)
        leg_ax = fig.add_axes([.105, .865, .87, .06], sharex=ax)
        fig.text(.105, .975, "A0275C · ADS-B Exchange · 1 March 2026", va="top", fontsize=11)

        ax.scatter(t, shown["baro_altitude_ft"], color=BLUE, s=2.2, linewidths=0)
        ax.set(xlim=(START, xmax), ylim=(-500, 39000), ylabel="Barometric altitude (ft)")
        ax.set_yticks([0, 10000, 20000, 30000])
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
        ax.grid(axis="y", color=".9", lw=.5)
        ax.tick_params(axis="x", labelbottom=False, bottom=False)

        # Each bar spans observed members of a readsb leg, including ground messages.
        for leg in legs[:2]:
            x0, x1 = mdates.date2num([leg["first_message"], leg["last_message"]])
            leg_ax.plot([x0, x1], [.45, .45], color=".6" if leg["segment"] == 1 else ".78",
                        lw=4, solid_capstyle="butt")
            leg_ax.text((x0 + x1) / 2, .70, f"readsb leg {leg['segment']}",
                        ha="center", va="bottom", fontsize=9)
        leg_ax.plot([mdates.date2num(boundary)] * 2, [.1, .8], color=".4", lw=.85)
        leg_ax.set(ylim=(0, 1.8))
        leg_ax.axis("off")
        for axis in (ax, states):
            axis.axvline(boundary, color=".4", ls="--", lw=.85, zorder=1)
        ax.annotate(f"readsb boundary\n{stamp_label(boundary)}", (mdates.date2num(boundary), .98),
                    xycoords=("data", "axes fraction"), ha="center", va="top", color=".35",
                    fontsize=8, bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5})

        # The shaded dwell illustrates why a leg boundary is not a flight endpoint.
        landing, takeoff = flights[0].landing_time, flights[1].takeoff_time
        ax.axvspan(landing, takeoff, color=GROUND, alpha=.07, zorder=0)
        ax.text(mdates.date2num(landing + (takeoff - landing) / 2), .40, "On ground",
                transform=ax.get_xaxis_transform(), ha="center", va="center",
                color=GROUND, fontsize=8, bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5})

        for flight in flights:
            for kind, color in (("takeoff", TAKEOFF), ("landing", LANDING)):
                stamp = getattr(flight, f"{kind}_time")
                for axis in (ax, states):
                    axis.axvline(stamp, color=color, ls=":", lw=1, zorder=2)
                # Keep the outer labels inside the figure, with short leaders.
                offset, align = (3, "left") if stamp == flights[0].takeoff_time else (
                    (-3, "right") if stamp == flights[-1].landing_time else (0, "center"))
                airport = AIRPORT_LABELS[(kind, stamp.isoformat())]
                ax.annotate(f"{kind.capitalize()}\n{airport}\n{stamp_label(stamp)}",
                            xy=(mdates.date2num(stamp), 1.01), xycoords=("data", "axes fraction"),
                            xytext=(offset, 8), textcoords="offset points", ha=align, va="bottom",
                            color=color, fontsize=8,
                            arrowprops={"arrowstyle": "-", "color": color, "lw": .6})

        # No filling across absent observations, and no imputation of ground altitudes.
        flags = shown["on_ground"].to_list()
        gaps = np.diff(t) * 86400
        for flag, color in ((False, "#56B4E9"), (True, GROUND)):
            mask = np.array([value is flag for value in flags])
            spans = [[(t[i], .5), (t[i + 1], .5)] for i in range(len(t) - 1)
                     if mask[i] and mask[i + 1] and gaps[i] <= 30]
            states.add_collection(LineCollection(spans, colors=color, linewidths=3))
            states.scatter(t[mask], np.full(mask.sum(), .5), marker="|", s=8,
                           linewidths=.6, color=color)
        states.set(ylim=(0, 1), yticks=[], xlabel="Time (UTC)")
        states.spines["left"].set_visible(False)
        states.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 2)))
        states.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        states.tick_params(axis="x", length=3)
        fig.legend(handles=[
            Line2D([], [], color=".4", ls="--", lw=.85, label="readsb boundary"),
            Line2D([], [], color=".4", ls=":", lw=1, label="Algorithm estimates"),
            Line2D([], [], color="#56B4E9", lw=3, label="on_ground=False"),
            Line2D([], [], color=GROUND, lw=3, label="on_ground=True"),
        ], loc="lower center", bbox_to_anchor=(.53, .01), ncol=4,
            frameon=False, fontsize=7, handlelength=1.8, columnspacing=1.5)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "paper" / "figures")
    args = parser.parse_args()
    data, sources = read_observations(args.observations)
    legs, all_flights, visible = segment_day(data)
    fig = plot_case(data, legs, visible)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "svg", "png"):
        target = args.output_dir / f"{STEM}.{extension}"
        fig.savefig(target, dpi=300, facecolor="white")
        print(target)
    plt.close(fig)
    extract = args.output_dir / f"{STEM}_observations.parquet"
    # Read completes before any output write, allowing reproduction in the same directory.
    data.write_parquet(extract)
    records = [{"segment": leg["segment"], "readsb_start": leg["first_message"],
                "readsb_last_message": leg["last_message"],
                "takeoff_time": flight.takeoff_time,
                "takeoff_airport_icao": AIRPORT_LABELS[("takeoff", flight.takeoff_time.isoformat())],
                "landing_time": flight.landing_time,
                "landing_airport_icao": AIRPORT_LABELS[("landing", flight.landing_time.isoformat())]}
               for leg, flight in zip(legs[:2], visible)]
    pl.DataFrame(records).write_csv(args.output_dir / f"{STEM}_times.csv")
    code_paths = [Path(__file__), ROOT / "src/flights/segmentation/algorithm.py",
                  ROOT / "src/flights/segmentation/readsb_legs.py"]
    manifest = {
        "icao": ICAO, "date": str(START.date()), "adsb_source": "adsbx", "timezone": "UTC",
        "source_paths": [str(path) for path in sources], "observation_count": data.height,
        "observation_sha256": hashlib.sha256(extract.read_bytes()).hexdigest(),
        "code_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in code_paths},
        "readsb_segments": legs, "illustrated_segments": [1, 2],
        "airport_label_source": AIRPORT_SOURCE,
        "illustrated_flights": records,
        "all_algorithm_flights": [{"takeoff_time": f.takeoff_time, "landing_time": f.landing_time,
                                    "landing_on_ground": f.landing_message.on_ground}
                                   for f in all_flights],
        "notes": "Recomputed using the production Python port of readsb mark_legs and identify_flight_segments. "
                 "First two legs shown; full day used for segmentation. Event labels rounded to nearest second. "
                 "Flight times are estimates from observed transitions, not independently verified wheels-off/on. "
                 "Null altitudes remain missing. Ground-state strip does not bridge gaps over 30 s. "
                 "Bars span first to last observed message in each readsb leg. "
                 "The unplotted final flight has a trace-end landing fallback, not an observed landing."
    }
    (args.output_dir / f"{STEM}_manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    (args.output_dir / f"{STEM}.tex").write_text(
        "\\begin{figure}[htbp]\n  \\centering\n"
        f"  \\includegraphics[width=\\linewidth]{{figures/{STEM}}}\n"
        "  \\caption{The first two readsb legs for \\texttt{a0275c} on 1 March 2026, "
        "using ADS-B Exchange observations. Gray bars show the legs, which include messages "
        f"while the aircraft is on the ground. The readsb boundary at {stamp_label(legs[1]['first_message'])} UTC falls within "
        f"the ground stop, between the algorithm's estimated landing at {stamp_label(visible[0].landing_time)} and next takeoff "
        f"at {stamp_label(visible[1].takeoff_time)}. The proposed algorithm therefore searches within each leg to select "
        "takeoff and landing messages (dotted lines). Airport codes are the neural network's "
        "predictions. The bottom strip shows the reported "
        "\\texttt{on\\_ground} state; missing altitudes remain blank. All times are UTC.}\n"
        "  \\label{fig:a0275c-readsb-segments}\n\\end{figure}\n")
    print(pl.DataFrame(records))


if __name__ == "__main__":
    main()
