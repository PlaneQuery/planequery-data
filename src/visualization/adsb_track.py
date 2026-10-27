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
    event_source: str = "flight estimates",
    show_geometric: bool = False,
    show_map: bool = False, airports: pl.DataFrame | None = None,
    basemap: bool = False, map_zoom: int = 11,
    map_events_only: bool = False, nearby_airport_limit: int = 8,
    map_below: bool = False, include_airports: tuple[str, ...] = (),
    compress_gaps: bool = False,
):
    """Return a Matplotlib figure without writing files.

    Null values stay missing. Lines stop at missing values and gaps larger than
    gap_seconds; an inset state strip displays known ground-state observations.
    No smoothing, interpolation, or ground-flag debouncing is performed.
    """
    if gap_seconds <= 0:
        raise ValueError("gap_seconds must be positive")
    d = prepare_observations(observations, icao, day)
    if show_geometric and 'geom_altitude_ft' not in d.columns:
        raise ValueError('Geometric overlay requires geom_altitude_ft in observations')
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
    event_labels = {}
    if flights is not None:
        for row in flights.filter(pl.col('icao').str.to_lowercase() == icao.lower()).to_dicts():
            for kind in ('takeoff', 'landing'):
                if row.get(f'{kind}_event_label'):
                    event_labels[(row.get(f'{kind}_time'), kind)] = row[f'{kind}_event_label']
    style = {"font.family": "serif", "font.size": 9, "axes.labelsize": 9,
             "xtick.labelsize": 8, "ytick.labelsize": 8, "pdf.fonttype": 42,
             "ps.fonttype": 42, "svg.fonttype": "none", "axes.spines.top": False,
             "axes.spines.right": False}
    with plt.rc_context(style):
        fig, primary = plt.subplots(figsize=(width, height))
        secondary = primary.twinx()
        axes = [primary, secondary]
        secondary.spines['right'].set_visible(True)
        fig.subplots_adjust(left=.12, right=.88, bottom=.20, top=.73 if events and not map_events_only else .85)
        for ax, col, label, color in zip(axes[:2],
                ["baro_altitude_ft", "ground_speed_kt"],
                ["baro_altitude_ft", "ground_speed_kt"], ["#0072B2", "#333333"]):
            y = d[col].cast(pl.Float64).to_numpy()
            ax.scatter(t, y, color=color, s=2, linewidths=0, zorder=3)
            ax.set_ylabel(label, color=color)
            ax.tick_params(axis='y', colors=color)
            ax.margins(y=.12)
        primary.grid(axis="y", color=".9", lw=.5)
        legend = []
        if show_geometric:
            y = d['geom_altitude_ft'].cast(pl.Float64).to_numpy()
            primary.scatter(t, y, color='#882255', s=2, linewidths=0, zorder=3)
            primary.set_ylabel('baro_altitude_ft / geom_altitude_ft', color='.2')
            primary.tick_params(axis='y', colors='.2')
            legend.extend([
                Line2D([], [], color='#0072B2', marker='.', ls='none', label='baro_altitude_ft'),
                Line2D([], [], color='#882255', marker='.', ls='none', label='geom_altitude_ft'),
                Line2D([], [], color='#333333', marker='.', ls='none', label='ground_speed_kt'),
            ])
        # A rug inside the shared plotting area carries state independently of
        # either numeric scale. Never extend it through a coverage gap.
        state_styles = [(0, '#56B4E9', '|', 'on_ground=False'),
                        (1, '#D55E00', '|', 'on_ground=True')]
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
                   ncol=3 if show_geometric else 2, frameon=False, fontsize=7)
        for i, (stamp, kind, airport) in enumerate(events):
            x = mdates.date2num(stamp)
            color = "#009E73" if kind == "takeoff" else "#CC79A7"
            primary.axvline(x, color=color, ls="--" if map_events_only else ':', lw=.9, alpha=.9)
            if map_events_only:
                # Keep codes tied to the actual event lines without restoring
                # verbose callouts or a separate annotation lane.
                near_right = x > mdates.date2num(start) + .9 * (mdates.date2num(end) - mdates.date2num(start))
                primary.annotate(airport, xy=(x, .97),
                                 xycoords=('data', 'axes fraction'),
                                 xytext=(-4 if near_right else 4, 0),
                                 textcoords='offset points',
                                 ha='right' if near_right else 'left', va='top',
                                 fontsize=7, color=color, zorder=7)
                continue
            # Distribute labels across a separate annotation lane to avoid close
            # turnaround labels colliding; leaders retain exact event positions.
            label = event_labels.get((stamp, kind), kind.title())
            axes[0].annotate(f"{label}\n{airport}\n{stamp:%H:%M:%S}",
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
        if compress_gaps:
            # Compress only long message-free intervals; all artists retain
            # original UTC coordinates, including flight event annotations.
            long_gaps = np.flatnonzero(np.diff(t) * 86400 > 30 * 60)
            intervals = [(t[i], t[i + 1]) for i in long_gaps]
            retained = 12 / 1440  # Leave visible separation between flights.

            def forward(values):
                values = np.asarray(values)
                result = values.copy()
                for left, right in intervals:
                    result = result - np.clip((values - left) / (right - left), 0, 1) * (right - left - retained)
                return result

            compressed = []
            removed = 0
            for left, right in intervals:
                compressed.append((left - removed, right - left - retained))
                removed += right - left - retained

            def inverse(values):
                values = np.asarray(values)
                result = values.copy()
                for left, reduction in compressed:
                    result = result + np.clip((values - left) / retained, 0, 1) * reduction
                return result

            for ax in axes:
                ax.set_xscale('function', functions=(forward, inverse))
            # Fixed UTC ticks exclude hidden-gap interiors, avoiding a pileup
            # of clock labels in each compressed interval.
            from matplotlib.ticker import FixedLocator
            tick_start = np.ceil(mdates.date2num(start) * 48) / 48
            ticks = np.arange(tick_start, mdates.date2num(end), 1 / 48)
            ticks = np.round(ticks * 86400) / 86400
            ticks = [tick for tick in ticks if not any(left < tick < right for left, right in intervals)]
            primary.xaxis.set_major_locator(FixedLocator(ticks))
            primary.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            for left, right in intervals:
                primary.text((left + right) / 2, 0, '//',
                             transform=primary.get_xaxis_transform(),
                             ha='center', va='center', fontsize=12, zorder=10,
                             bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': 1})
            if intervals:
                primary.set_xlabel(f"Time (UTC), {day.isoformat()} · // long gaps compressed")
        if map_events_only and events:
            secondary.legend(handles=[
                Line2D([], [], color='#009E73', ls='--', lw=.9, label='Estimated takeoff'),
                Line2D([], [], color='#CC79A7', ls='--', lw=.9, label='Estimated landing'),
            ], loc='upper center', ncol=2, frameon=False, fontsize=7)
        fig.suptitle(title or f"{icao.upper()} · ADS-B observations", y=.99, fontsize=11)
        if events and not map_events_only:
            secondary.text(.5, .98, f"Dotted lines: {event_source}.",
                           transform=secondary.transAxes, ha='center', va='top',
                           fontsize=7, color='.35',
                           bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': 2})
        if not map_events_only and flights is not None and 'landing_assignment_method' in flights.columns:
            reviewed = flights.filter(pl.col('icao').str.to_lowercase() == icao.lower()).sort('takeoff_time').to_dicts()
            for i, row in enumerate(reviewed[:-1]):
                if row.get('landing_time') is not None or row.get('landing_assignment_method') != 'matched_to_next_departure':
                    continue
                left, right = row['takeoff_time'], reviewed[i+1]['takeoff_time']
                if left is None or right is None:
                    continue
                midpoint = left + (right-left)/2
                if start <= midpoint <= end:
                    secondary.text(mdates.date2num(midpoint), .65,
                                   f"Arrival inferred: {row['landing_airport_ident']}\nTime unobserved",
                                   transform=secondary.get_xaxis_transform(), ha='center',
                                   fontsize=7, color='.35',
                                   bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': 2})
        if show_map:
            if not {'lat', 'lon'} <= set(d.columns):
                raise ValueError('Map requires lat and lon observation columns')
            # Preserve the original panel's physical width when adding the map.
            fig.set_size_inches(width + 4, height)
            primary.set_position([.08, .20, .45, .53 if events and not map_events_only else .65])
            secondary.set_position(primary.get_position())
            fig.legends[0].set_bbox_to_anchor((.305, .015))
            map_height = .65 if map_events_only else .53
            map_ax = fig.add_axes([.67, .20 if map_events_only else .26, .29, map_height])
            if map_below:
                callouts = bool(events) and not map_events_only
                fig.set_size_inches(width, height + (5.2 if callouts else 4))
                primary.set_position([.12, .61, .76, .25] if callouts else [.12, .66, .76, .27])
                secondary.set_position(primary.get_position())
                fig.legends[0].set_bbox_to_anchor((.5, .52 if callouts else .555))
                map_ax.set_position([.12, .12, .76, .35] if callouts else [.12, .14, .76, .36])
            lon = d['lon'].cast(pl.Float64).to_numpy()
            lat = d['lat'].cast(pl.Float64).to_numpy()
            valid = np.isfinite(lon) & np.isfinite(lat)
            if not valid.any():
                raise ValueError('No valid positions in the plotting interval')
            if basemap:
                from visualization.basemap import project, add_basemap
                from matplotlib.ticker import FuncFormatter
                px, py = project(lon, lat)
            else:
                px, py = lon, lat
            # Break on both missing coordinates and time gaps; never interpolate
            # a route across unobserved portions of the flight.
            from matplotlib import patheffects
            map_ax.plot(np.insert(px, breaks, np.nan), np.insert(py, breaks, np.nan),
                        color='#273469', lw=1.7, zorder=2,
                        path_effects=[patheffects.Stroke(linewidth=3.5, foreground='white'),
                                      patheffects.Normal()])
            points = map_ax.scatter(px[valid], py[valid], c=t[valid], cmap='plasma' if basemap else 'viridis',
                                    s=14 if map_events_only or map_below else 25,
                                    edgecolors='none' if map_events_only or map_below else 'black' if basemap else 'white',
                                    linewidths=.5, zorder=3)
            points.set_gid('observed-map-positions')
            if airports is not None:
                codes = {event[2] for event in events}
                if flights is not None:
                    # Airport assessments can have no event time when touchdown
                    # occurs in missing coverage, yet still belong on the map.
                    for row in flights.filter(pl.col('icao').str.to_lowercase() == icao.lower()).to_dicts():
                        for kind in ('takeoff', 'landing'):
                            if row.get(f'{kind}_airport_ident'):
                                codes.add(row[f'{kind}_airport_ident'])
                for airport in airports.filter(pl.col('ident').is_in(codes)).to_dicts():
                    x, y = airport['longitude_deg'], airport['latitude_deg']
                    if x is None or y is None:
                        continue
                    if basemap:
                        x, y = project(x, y)
                    map_ax.scatter([x], [y], marker='*', s=110, color='#006633',
                                   edgecolors='white', linewidths=.8, zorder=4)
                    label = airport['ident']
                    label_left = basemap and x > np.nanmean(px)
                    map_ax.annotate(label, (x, y), xytext=(-9, 10) if label_left else (7, -12),
                                    ha='right' if label_left else 'left',
                                    va='bottom' if label_left else 'top',
                                    textcoords='offset points', fontsize=7,
                                    arrowprops={'arrowstyle': '-', 'lw': .7, 'color': '.2'} if basemap else None,
                                    bbox={'facecolor': 'white', 'edgecolor': '.6', 'alpha': .95, 'pad': 2},
                                    zorder=6)
            map_ax.set_aspect(1 if basemap else 1 / np.cos(np.deg2rad(np.nanmean(lat))), adjustable='box')
            map_ax.margins(.18)
            if basemap:
                # Fill the map panel even for nearly east-west or north-south
                # tracks, expanding geographic context rather than stretching it.
                xmin, xmax = map_ax.get_xlim()
                ymin, ymax = map_ax.get_ylim()
                panel = map_ax.get_position(original=True)
                panel_ratio = (panel.width * fig.get_figwidth()) / (panel.height * fig.get_figheight())
                span_x, span_y = xmax - xmin, ymax - ymin
                span_x = max(span_x, span_y * panel_ratio)
                span_y = max(span_y, span_x / panel_ratio)
                cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
                map_ax.set_xlim(cx - span_x / 2, cx + span_x / 2)
                map_ax.set_ylim(cy - span_y / 2, cy + span_y / 2)
            if airports is not None and 'type' in airports.columns:
                # Add context only within the established flight viewport; nearby
                # airports must not recursively expand the map extent.
                xlim, ylim = map_ax.get_xlim(), map_ax.get_ylim()
                nearby = airports.filter(
                    pl.col('type').is_in(['small_airport', 'medium_airport', 'large_airport'])
                    & ~pl.col('ident').is_in(codes)
                )
                nearby_positions = []
                # Rank by the ends of observed segments, not by overflights.
                # Use exactly the same time-gap/missing-position boundaries as
                # the drawn track; a singleton segment is itself an endpoint.
                connected = valid[:-1] & valid[1:]
                connected[np.asarray(breaks, dtype=int) - 1] = False
                endpoints = valid & (~np.r_[False, connected] | ~np.r_[connected, False])
                for airport in nearby.to_dicts():
                    x, y = airport['longitude_deg'], airport['latitude_deg']
                    if x is None or y is None:
                        continue
                    if basemap:
                        x, y = project(x, y)
                    if not (xlim[0] <= x <= xlim[1] and ylim[0] <= y <= ylim[1]):
                        continue
                    distance = np.min((px[endpoints]-x)**2 + (py[endpoints]-y)**2)
                    nearby_positions.append((distance, airport['ident'], x, y))
                # Explicit context requests remain visible even when small
                # airfields near coverage gaps dominate the distance ranking.
                ranked = sorted(nearby_positions, key=lambda p: (p[1] not in include_airports, p[0]))
                for _, code, x, y in ranked[:nearby_airport_limit]:
                    map_ax.scatter([x], [y], marker='D', s=35, facecolors='white',
                                   edgecolors='#404040', linewidths=1, zorder=4)
                    map_ax.annotate(code, (x, y), xytext=(-6, -10),
                                    textcoords='offset points', ha='right', va='top', fontsize=7,
                                    bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': .9, 'pad': 1},
                                    zorder=6)
                map_ax.set_xlim(xlim)
                map_ax.set_ylim(ylim)
                map_ax.legend(handles=[
                    Line2D([], [], marker='*', color='#006633', ls='none', markersize=8,
                           label='Flight airports' if map_events_only or map_below else 'Estimated endpoints'),
                    Line2D([], [], marker='D', markerfacecolor='white', color='#404040',
                           ls='none', markersize=4, label='Other airports'),
                ], loc='upper left', fontsize=6, framealpha=.95, borderpad=.4)
            map_ax.set_xlabel('Longitude (°)')
            map_ax.set_ylabel('Latitude (°)')
            map_ax.set_title('Airports & observed positions' if basemap else 'Observed positions', fontsize=10)
            map_ax.ticklabel_format(useOffset=False, style='plain')
            map_ax.grid(color='.9', lw=.5)
            map_ax.locator_params(axis='both', nbins=5)
            if basemap:
                add_basemap(map_ax, zoom=map_zoom)
                map_ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{np.rad2deg(x / 6378137):.2f}'))
                map_ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{np.rad2deg(np.arctan(np.sinh(y / 6378137))):.2f}'))
                map_ax.grid(False)
            color_ax = fig.add_axes([.70, .085 if map_events_only else .105, .23, .02])
            if map_below:
                color_ax.set_position([.25, .065, .5, .012])
            colorbar = fig.colorbar(points, cax=color_ax, orientation='horizontal')
            colorbar.set_ticks([t[valid].min(), t[valid].max()])
            colorbar.ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            colorbar.set_label('Observation time (UTC)', fontsize=8)
            # Resolve airport-label collisions after the geographic limits and
            # aspect ratio are final. Endpoint annotations retain first priority.
            from matplotlib.text import Annotation
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            occupied = []
            from matplotlib.transforms import Bbox
            # Reserve the observed track corridor, not just other text boxes.
            track_pixels = map_ax.transData.transform(np.column_stack([px[valid], py[valid]]))
            occupied.extend(Bbox.from_bounds(x-5, y-5, 10, 10) for x, y in track_pixels)
            if map_ax.get_legend() is not None:
                occupied.append(map_ax.get_legend().get_window_extent(renderer))
            for label in map_ax.texts:
                if not isinstance(label, Annotation):
                    occupied.append(label.get_window_extent(renderer))
            for label in map_ax.texts:
                if not isinstance(label, Annotation):
                    continue
                for dx, dy in [(7, 8), (-7, 8), (7, -8), (-7, -8),
                               (7, 24), (-7, 24), (7, -24), (-7, -24),
                               (7, 40), (-7, 40), (7, -40), (-7, -40)]:
                    label.set_position((dx, dy))
                    label.set_ha('left' if dx > 0 else 'right')
                    label.set_va('bottom' if dy > 0 else 'top')
                    label.update_positions(renderer)
                    label.update_bbox_position_size(renderer)
                    box = label.get_bbox_patch().get_window_extent(renderer).expanded(1.05, 1.15)
                    if (map_ax.bbox.contains(box.x0, box.y0) and map_ax.bbox.contains(box.x1, box.y1)
                            and not any(box.overlaps(other) for other in occupied)):
                        break
                occupied.append(box)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--icao", required=True)
    parser.add_argument("--date", required=True, type=date.fromisoformat)
    parser.add_argument("--source", default="adsbx", choices=["adsbx", "adsblol", "opensky"])
    parser.add_argument("--observations", type=Path, help="Optional observation Parquet; otherwise read canonical provider data")
    parser.add_argument("--flights", type=Path, help="Flight Parquet with takeoff/landing_time and takeoff/landing_airport_ident")
    parser.add_argument("--event-source", default="flight estimates")
    parser.add_argument("--output", type=Path, help="Output stem; defaults to paper/figures/ICAO_DATE_SOURCE")
    parser.add_argument("--formats", nargs="+", default=["pdf", "svg", "png"], choices=["pdf", "svg", "png"])
    parser.add_argument("--gap-seconds", type=float, default=30)
    parser.add_argument('--show-geometric', action='store_true', help='Overlay geometric altitude on the altitude axis')
    parser.add_argument('--show-map', action='store_true', help='Add a lat/lon map beside the time series (adds 4 inches to width)')
    parser.add_argument('--map-events-only', action='store_true', help='Label takeoff/landing airports on the map instead of the time series')
    parser.add_argument('--nearby-airport-limit', type=int, default=8, help='Maximum other airports to label on map')
    parser.add_argument('--map-below', action='store_true', help='Place a larger map below the time series')
    parser.add_argument('--compress-gaps', action='store_true', help='Compress message gaps over 30 minutes to twelve minutes of axis width, preserving UTC labels')
    parser.add_argument('--include-airports', nargs='+', default=[], help='Prioritize these context-airport codes within the map viewport')
    parser.add_argument('--basemap', action='store_true', help='Use an OpenStreetMap basemap (downloads and caches viewport tiles); implies --show-map')
    parser.add_argument('--map-zoom', type=int, default=11, help='Basemap tile zoom (default 11)')
    parser.add_argument('--airports', type=Path, default=Path('data/raw/ourairports/airports.csv'),
                        help='Airport CSV for map labels (ident, latitude_deg, longitude_deg)')
    parser.add_argument("--width", type=float, default=7.2, help="Figure width in inches")
    parser.add_argument("--height", type=float, default=4.8)
    parser.add_argument("--start", type=time.fromisoformat, help="UTC HH:MM[:SS]")
    parser.add_argument("--end", type=time.fromisoformat, help="UTC HH:MM[:SS]")
    args = parser.parse_args()
    args.show_map = args.show_map or args.basemap
    args.show_map = args.show_map or args.map_events_only or args.map_below
    if args.output is None:
        args.output = Path("paper/figures") / f"{args.icao.lower()}_{args.date:%Y%m%d}_{args.source}"
    if args.observations:
        observations = pl.read_parquet(args.observations)
    else:
        from data_engineering.adsb.read_adsb import ADSBSource, read_adsb
        observations = read_adsb(args.date, source=ADSBSource(args.source), icaos=[args.icao.lower()],
                                 additional_columns=['geom_altitude_ft'] if args.show_geometric else None)
    flights = pl.read_parquet(args.flights) if args.flights else None
    if flights is None:
        warnings.warn("No --flights supplied: plotting observed flags without inferred airport/event labels.")
    fig = plot_track(observations, icao=args.icao, day=args.date, flights=flights,
                     gap_seconds=args.gap_seconds, width=args.width, height=args.height,
                     start=datetime.combine(args.date, args.start) if args.start else None,
                     end=datetime.combine(args.date, args.end) if args.end else None,
                     title=f"{args.icao.upper()} · {'ADS-B Exchange' if args.source == 'adsbx' else args.source.upper()}", event_source=args.event_source,
                     show_geometric=args.show_geometric, show_map=args.show_map,
                     map_events_only=args.map_events_only, nearby_airport_limit=args.nearby_airport_limit,
                     map_below=args.map_below, include_airports=tuple(c.upper() for c in args.include_airports),
                     compress_gaps=args.compress_gaps,
                     basemap=args.basemap, map_zoom=args.map_zoom,
                     airports=pl.read_csv(args.airports, infer_schema_length=10000)
                     if args.show_map and args.airports.is_file() else None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for fmt in args.formats:
        target = Path(f"{args.output}.{fmt}")
        fig.savefig(target, dpi=300, facecolor="white")
        print(target)
    plt.close(fig)


if __name__ == "__main__":
    main()
