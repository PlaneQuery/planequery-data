"""Plot the available NN inputs for A96262's arrival and departure at US-8169 offline.

Run with OUTPUT_DIR pointing at the existing airport-history storage root.
The figure extract lacks heading and vertical rate: do not infer scores from it.
"""

import argparse
from dataclasses import asdict
from datetime import timedelta
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import numpy as np
import polars as pl

from airports.catalog import AirportCatalogView
from airports.history import AirportHistory
from airports.history.schema import HistoryPaths
from data_engineering.utils import OUTPUT_DIR
from flights.airport_nn.features import VERSION, clean_messages, trajectories


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=Path, default=Path("tmp/adsbx_a96262_20260401/observations.parquet"))
    parser.add_argument("--flights", type=Path, default=Path("tmp/adsbx_a96262_20260401/figure_flights.parquet"))
    parser.add_argument("--output", type=Path,
                        default=OUTPUT_DIR.expanduser() / "paper/figures/a96262_airport_nn_inputs",
                        help="Output directory (default: $OUTPUT_DIR/paper/figures/a96262_airport_nn_inputs).")
    parser.add_argument("--endpoint", choices=("both", "takeoff", "landing"), default="both")
    args = parser.parse_args()
    all_flights = pl.read_parquet(args.flights).sort("takeoff_time")
    departure = all_flights.filter(
        (pl.col("icao") == "a96262") & (pl.col("takeoff_airport_ident") == "US-8169")
    )
    if departure.height != 1:
        raise ValueError("Expected exactly one A96262 departure from US-8169")
    arrival = all_flights.filter(
        (pl.col("icao") == "a96262")
        & (pl.col("landing_airport_ident") == "US-8169")
        & (pl.col("landing_time") <= departure["takeoff_time"][0])
    ).tail(1)
    if arrival.height != 1:
        raise ValueError("Expected a preceding A96262 arrival at US-8169")
    observations = pl.read_parquet(args.observations)
    absent = [name for name in ("track_deg", "baro_rate_fpm") if name not in observations.columns]
    # Null placeholders let the production sampler run. Their derived features
    # are excluded from the export/plots, and this script never runs inference.
    observations = observations.with_columns([pl.lit(None, dtype=pl.Float64).alias(name) for name in absent])
    # Read the existing history snapshot directly; open() can refresh/download.
    history = AirportHistory(HistoryPaths.defaults())
    for name, flights in (("takeoff", departure), ("landing", arrival)):
        if args.endpoint in ("both", name):
            render_endpoint(args, flights, observations, absent, history, name)


def render_endpoint(args, flights, observations, absent, history, name):
    flight = flights.row(0, named=True)
    landing = name == "landing"
    stem = f"a96262_airport_nn_{name}_candidate_inputs"
    catalog = AirportCatalogView.from_history(history, flight["takeoff_time"].date(), history.latest_available_date)
    endpoint = next(e for e in trajectories(flights, observations, catalog) if e.endpoint == name)
    values = endpoint.values
    assert endpoint.candidate_idents and "US-8169" in endpoint.candidate_idents
    assert np.isfinite(values).all()
    shown = (("US-3394", "TT75", "TE85", "89TA", "KCXO", "US-8169") if landing
             else ("16XS", "US-6846", "US-8445", "US-1134", "KCXO", "US-8169"))
    selected = [endpoint.candidate_idents.index(ident) for ident in shown]
    boundary = flight[f"{name}_time"]
    start = max(flight["takeoff_time"], boundary - timedelta(minutes=10))
    end = min(flight["landing_time"], boundary + timedelta(minutes=10))
    window = clean_messages(observations).filter((pl.col("icao") == flight["icao"]) & (pl.col("time") >= start) & (pl.col("time") <= end))
    elapsed = values[0, :, 10] * 600
    anchor = window["time"].max() if landing else window["time"].min()
    signed_elapsed = -elapsed if landing else elapsed
    times = [anchor + timedelta(seconds=float(seconds)) for seconds in signed_elapsed]
    distances = np.expm1(values[:, :, 4] * 5)
    altitude = values[:, :, 5] * 10000
    east, north = values[:, :, 1] * 100, values[:, :, 0] * 100
    args.output.mkdir(parents=True, exist_ok=True)
    candidates = []
    for i, ident in enumerate(endpoint.candidate_idents):
        airport = catalog.id_index[endpoint.candidate_ids[i]]
        candidates.append(dict(endpoint=name, candidate_distance_rank=i + 1, airport_ident=ident, airport_id=airport.id, name=airport.name, elevation_ft=airport.elevation_ft, endpoint_message_distance_km=float(distances[i, 0]), endpoint_message_baro_height_ft=float(altitude[i, 0]), displayed=i in selected))
    pl.DataFrame(candidates).write_csv(args.output / f"{stem}_candidates.csv")
    rows = []
    for i, ident in enumerate(endpoint.candidate_idents):
        for t, stamp in enumerate(times):
            rows.append(dict(endpoint=name, airport_ident=ident, model_sequence_index=t, time_utc=stamp, seconds_from_endpoint_message=float(signed_elapsed[t]), north_km=float(north[i, t]), east_km=float(east[i, t]), distance_km=float(distances[i, t]), baro_height_above_airport_ft=float(altitude[i, t]), radial_speed_km_s=float(values[i, t, 2] * .2), ground_speed_kt=float(values[i, t, 6] * 300), on_ground=float(values[i, t, 11]), altitude_missing=bool(values[i, t, 14]), speed_missing=bool(values[i, t, 15])))
    pl.DataFrame(rows).write_csv(args.output / f"{stem}_sampled_inputs.csv")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#aeb6bf", "axes.labelcolor": "#344054", "xtick.color": "#667085", "ytick.color": "#667085", "svg.fonttype": "none"})
    plt.rcParams["pdf.fonttype"] = 42
    fig = plt.figure(figsize=(14.5, 9.5), facecolor="white")
    fig.text(.06, .925, f"A96262  ·  {name.capitalize()}  ·  ADS-B Exchange  ·  {start:%Y-%m-%d}  ·  {min(times):%H:%M:%S}-{max(times):%H:%M:%S} UTC", fontsize=11.5, color="#475467")
    grid = fig.add_gridspec(2, 3, left=.065, right=.97, bottom=.17, top=.855, hspace=.45, wspace=.26)
    cmap, norm = plt.get_cmap("plasma"), Normalize(-10, 0) if landing else Normalize(0, 10)
    colors = cmap(norm(signed_elapsed / 60))
    xlim = (np.floor(min(0, east[selected].min()) / 10) * 10 - 5, np.ceil(max(0, east[selected].max()) / 10) * 10 + 5)
    ylim = (np.floor(min(0, north[selected].min()) / 10) * 10 - 5, np.ceil(max(0, north[selected].max()) / 10) * 10 + 5)
    max_distance = np.ceil(distances[selected].max() / 10) * 10 + 5
    for slot, i in enumerate(selected):
        item = candidates[i]
        inner = grid[slot // 3, slot % 3].subgridspec(2, 1, height_ratios=[1.25, 1], hspace=.55)
        plan, profile = fig.add_subplot(inner[0]), fig.add_subplot(inner[1])
        highlight = item["airport_ident"] == "US-8169"
        title = item["airport_ident"] + ("  ·  Correct airport" if highlight else "")
        plan.text(0, 1.13, title, transform=plan.transAxes, fontsize=12.5, weight="bold", color="#087f72" if highlight else "#182230")
        for ax in (plan, profile):
            ax.set_facecolor("#f5fbf9" if highlight else "#fafbfc")
            ax.grid(color="#e5e9ef", linewidth=.6, zorder=0)
            ax.tick_params(labelsize=8)
        plan.plot(east[i], north[i], color="#b7bec8", lw=.8, zorder=1)
        plan.scatter(east[i], north[i], c=colors, s=18, zorder=3, edgecolors="none")
        plan.scatter(east[i, 0], north[i, 0], c=[colors[0]], marker="X", s=75, edgecolors="#182230", linewidths=.7, zorder=4)
        plan.scatter(0, 0, marker="*", s=150, color="#087f72", edgecolors="white", linewidths=.6, zorder=5)
        plan.set(xlim=xlim, ylim=ylim, xlabel="East of airport (km)", ylabel="North (km)")
        plan.set_aspect("equal", adjustable="box")
        profile.plot(distances[i], altitude[i], color="#b7bec8", lw=.8)
        profile.scatter(distances[i], altitude[i], c=colors, s=18, zorder=3, edgecolors="none")
        profile.scatter(distances[i, 0], altitude[i, 0], c=[colors[0]], marker="X", s=75, edgecolors="#182230", linewidths=.7, zorder=4)
        profile.scatter(0, 0, marker="*", s=150, color="#087f72", edgecolors="white", linewidths=.6, zorder=5)
        profile.set(xlim=(-3, max_distance), ylim=(-350, 5400 if landing else 3000), xlabel="Distance to airport (km)", ylabel="Baro height (ft)", yticks=[0, 2500, 5000] if landing else [0, 1000, 2000, 3000])
    cax = fig.add_axes([.065, .075, .43, .013])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal", ticks=np.arange(norm.vmin, norm.vmax + 1, 2))
    cb.set_label("Minutes from last message" if landing else "Minutes from first message", fontsize=10)
    cb.ax.tick_params(labelsize=9)
    fig.legend(handles=[Line2D([], [], marker="*", linestyle="", color="#087f72", markersize=13, label="Airport"), Line2D([], [], marker="X", linestyle="", color=colors[0], markeredgecolor="#182230", markersize=9, label="Last Message" if landing else "First Message")], loc="center", bbox_to_anchor=(.745, .075), ncol=2, frameon=False, fontsize=10)
    for extension in ("png", "svg", "pdf"):
        fig.savefig(args.output / f"{stem}.{extension}", dpi=180, facecolor="white")
    plt.close(fig)
    manifest = dict(feature_version=VERSION, observation_path=str(args.observations.resolve()), observation_sha256=sha256(args.observations), flight_path=str(args.flights.resolve()), flight_sha256=sha256(args.flights), catalog=asdict(catalog.provenance), raw_window_observations=window.height, sampled_observations=len(times), candidates=len(candidates), selected=[endpoint.candidate_idents[i] for i in selected], missing_source_columns=absent, model_scores_computed=False, endpoint=name, label_provenance="Reviewed destination matched to next departure; touchdown unobserved" if landing else "Reviewed departure from US-8169", endpoint_boundary=str(boundary), note="Production feature builder and sampler; unavailable heading/rate inputs are not displayed or scored. Physical units decoded from normalized float32 inputs.")
    (args.output / f"{stem}_manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    print(json.dumps(manifest, indent=2, default=str))


if __name__ == "__main__":
    main()
