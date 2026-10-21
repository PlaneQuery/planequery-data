"""Publication figures for an aircraft's ADS-B observations (all times UTC).

Flight annotations are supplied estimates, independent of observed ground flags.
See README.md beside this module for CLI examples and interpretation.
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, time
from pathlib import Path
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import numpy as np
import polars as pl


def prepare_observations(frame: pl.DataFrame, icao: str, day: date) -> pl.DataFrame:
    required = {"icao", "time", "baro_altitude_ft", "ground_speed_kt", "on_ground"}
    if missing := required - set(frame.columns):
        raise ValueError(f"Missing observation columns: {sorted(missing)}")
    if frame.schema["time"].time_zone is not None:
        frame = frame.with_columns(pl.col("time").dt.convert_time_zone("UTC").dt.replace_time_zone(None))
    frame = frame.filter(
        (pl.col("icao").str.to_lowercase() == icao.lower())
        & (pl.col("time").dt.date() == day)
    ).sort("time")
    if frame.is_empty():
        raise ValueError(f"No observations for {icao} on {day}")
    return frame


def flight_events(flights: pl.DataFrame | None, icao: str, start: datetime, end: datetime):
    """Return supplied flight boundaries; never infer airport codes from flags."""
    events = []
    if flights is None:
        return events
    for row in flights.filter(pl.col("icao").str.to_lowercase() == icao.lower()).to_dicts():
        for kind in ("takeoff", "landing"):
            stamp = row.get(f"{kind}_time")
            if stamp is None:
                continue
            if stamp.tzinfo is not None:
                from datetime import UTC
                stamp = stamp.astimezone(UTC).replace(tzinfo=None)
            if start <= stamp <= end:
                events.append((stamp, kind, row.get(f"{kind}_airport_ident") or "unknown airport"))
    return sorted(set(events))


def plot_track(
    observations: pl.DataFrame, *, icao: str, day: date,
    flights: pl.DataFrame | None = None, gap_seconds: float = 30,
    width: float = 7.2, height: float = 4.8, title: str | None = None,
    start: datetime | None = None, end: datetime | None = None,
    event_source: str = "supplied flight estimates",
):
    """Return a Matplotlib figure without writing files.

    Null values stay missing. Lines stop at missing values and gaps larger than
    gap_seconds; an inset state strip displays known ground-state observations.
    No smoothing, interpolation, or ground-flag debouncing is performed.
    """
    if gap_seconds <= 0:
        raise ValueError("gap_seconds must be positive")
    d = prepare_observations(observations, icao, day)
    start = start or d["time"].min()
    end = end or d["time"].max()
    if end <= start:
        raise ValueError("The plotting interval must have positive duration")
    d = d.filter(pl.col("time").is_between(start, end))
    if d.is_empty():
        raise ValueError("No observations in the requested interval")
    t = mdates.date2num(d["time"].to_list())
    gaps = np.diff(t) * 86400 > gap_seconds
    # Insert NaNs between disconnected samples without discarding endpoints.
    breaks = np.flatnonzero(gaps) + 1
    tx = np.insert(t, breaks, np.nan)
    flags = d["on_ground"].to_list()
    state = np.array([0 if v is False else 1 if v is True else .5 for v in flags], dtype=float)
    events = flight_events(flights, icao, start, end)
    style = {"font.family": "serif", "font.size": 9, "axes.labelsize": 9,
             "xtick.labelsize": 8, "ytick.labelsize": 8, "pdf.fonttype": 42,
             "ps.fonttype": 42, "svg.fonttype": "none", "axes.spines.top": False,
             "axes.spines.right": False}
    with plt.rc_context(style):
        fig, primary = plt.subplots(figsize=(width, height))
        secondary = primary.twinx()
        axes = [primary, secondary]
        secondary.spines['right'].set_visible(True)
        fig.subplots_adjust(left=.12, right=.88, bottom=.20, top=.73 if events else .85)
        for ax, col, label, color in zip(axes[:2],
                ["baro_altitude_ft", "ground_speed_kt"],
                ["Barometric altitude\n(ft)", "Ground speed\n(kt)"], ["#0072B2", "#333333"]):
            y = d[col].cast(pl.Float64).to_numpy()
            ax.plot(tx, np.insert(y, breaks, np.nan), color=color, lw=.85,
                    ls='-' if ax is primary else '--')
            ax.scatter(t, y, color=color, s=2, linewidths=0, zorder=3)
            ax.set_ylabel(label, color=color)
            ax.tick_params(axis='y', colors=color)
            ax.margins(y=.12)
        primary.grid(axis="y", color=".9", lw=.5)
        # A rug inside the shared plotting area carries state independently of
        # either numeric scale. Never extend it through a coverage gap.
        state_styles = [(0, '#56B4E9', '|', 'on_ground=False'),
                        (1, '#D55E00', '|', 'on_ground=True')]
        legend = []
        for value, color, marker, label in state_styles:
            mask = state == value
            segments = [[(t[i], .025), (t[i+1], .025)]
                        for i in range(len(t)-1) if mask[i] and not gaps[i]]
            strip = LineCollection(segments, colors=color, linewidths=4,
                                   transform=secondary.get_xaxis_transform(), zorder=5)
            strip.set_gid(label)
            secondary.add_collection(strip)
            secondary.scatter(t[mask], np.full(mask.sum(), .025), color=color,
                              marker=marker, s=16, linewidths=.8, zorder=6,
                              transform=secondary.get_xaxis_transform())
            legend.append(Line2D([], [], color=color, lw=4, label=label))
        fig.legend(handles=legend, loc='lower center', bbox_to_anchor=(.5, .015),
                   ncol=2, frameon=False, fontsize=7)
        for i, (stamp, kind, airport) in enumerate(events):
            x = mdates.date2num(stamp)
            color = "#009E73" if kind == "takeoff" else "#CC79A7"
            primary.axvline(x, color=color, ls=":", lw=.75, alpha=.9)
            # Distribute labels across a separate annotation lane to avoid close
            # turnaround labels colliding; leaders retain exact event positions.
            axes[0].annotate(f"{kind.title()}\n{airport}\n{stamp:%H:%M:%S}",
                xy=(x, 1), xycoords=("data", "axes fraction"),
                xytext=((i + .5) / len(events), 1.07 + .18 * (i % 2)),
                textcoords="axes fraction", ha="center", va="bottom", fontsize=7,
                color=color, annotation_clip=False,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 2},
                arrowprops={"arrowstyle": "-", "lw": .6, "color": color,
                            "connectionstyle": "angle,angleA=0,angleB=90,rad=2"})
        axes[-1].set_xlim(mdates.date2num(start), mdates.date2num(end))
        locator = mdates.AutoDateLocator(minticks=4, maxticks=9)
        axes[-1].xaxis.set_major_locator(locator)
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        primary.set_xlabel(f"Time (UTC), {day.isoformat()}")
        fig.suptitle(title or f"{icao.upper()} · ADS-B observations", y=.99, fontsize=11)
        if events:
            secondary.text(.5, .98, f"Dotted lines: {event_source}.",
                           transform=secondary.transAxes, ha='center', va='top',
                           fontsize=7, color='.35',
                           bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': 2})
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--icao", required=True)
    parser.add_argument("--date", required=True, type=date.fromisoformat)
    parser.add_argument("--source", default="adsbx", choices=["adsbx", "adsblol", "opensky"])
    parser.add_argument("--observations", type=Path, help="Optional observation Parquet; otherwise read canonical provider data")
    parser.add_argument("--flights", type=Path, help="Flight Parquet with takeoff/landing_time and takeoff/landing_airport_ident")
    parser.add_argument("--event-source", default="supplied flight estimates")
    parser.add_argument("--output", type=Path, help="Output stem; defaults to paper/figures/ICAO_DATE_SOURCE")
    parser.add_argument("--formats", nargs="+", default=["pdf", "svg", "png"], choices=["pdf", "svg", "png"])
    parser.add_argument("--gap-seconds", type=float, default=30)
    parser.add_argument("--width", type=float, default=7.2, help="Figure width in inches")
    parser.add_argument("--height", type=float, default=4.8)
    parser.add_argument("--start", type=time.fromisoformat, help="UTC HH:MM[:SS]")
    parser.add_argument("--end", type=time.fromisoformat, help="UTC HH:MM[:SS]")
    args = parser.parse_args()
    if args.output is None:
        args.output = Path("paper/figures") / f"{args.icao.lower()}_{args.date:%Y%m%d}_{args.source}"
    if args.observations:
        observations = pl.read_parquet(args.observations)
    else:
        from data_engineering.adsb.read_adsb import ADSBSource, read_adsb
        observations = read_adsb(args.date, source=ADSBSource(args.source), icaos=[args.icao.lower()])
    flights = pl.read_parquet(args.flights) if args.flights else None
    if flights is None:
        warnings.warn("No --flights supplied: plotting observed flags without inferred airport/event labels.")
    fig = plot_track(observations, icao=args.icao, day=args.date, flights=flights,
                     gap_seconds=args.gap_seconds, width=args.width, height=args.height,
                     start=datetime.combine(args.date, args.start) if args.start else None,
                     end=datetime.combine(args.date, args.end) if args.end else None,
                     title=f"{args.icao.upper()} · {args.source.upper()}", event_source=args.event_source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for fmt in args.formats:
        target = Path(f"{args.output}.{fmt}")
        fig.savefig(target, dpi=300, facecolor="white")
        print(target)
    plt.close(fig)


if __name__ == "__main__":
    main()
