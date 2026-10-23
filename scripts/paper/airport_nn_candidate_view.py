"""Plot the available NN inputs for the paper's A96262 departure from US-8169, entirely offline.

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
from flights.airport_nn.features import VERSION, clean_messages, trajectories


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=Path, default=Path("tmp/adsbx_a96262_20260401/observations.parquet"))
    parser.add_argument("--flights", type=Path, default=Path("tmp/adsbx_a96262_20260401/figure_flights.parquet"))
    parser.add_argument("--output", type=Path, default=Path("paper/figures/a96262_airport_nn_inputs"))
    args = parser.parse_args()
    flights = pl.read_parquet(args.flights).filter(
        (pl.col("icao") == "a96262") & (pl.col("takeoff_airport_ident") == "US-8169")
    )
    if flights.height != 1:
        raise ValueError("Expected exactly one A96262 departure from US-8169")
    flight = flights.row(0, named=True)
    observations = pl.read_parquet(args.observations)
    absent = [name for name in ("track_deg", "baro_rate_fpm") if name not in observations.columns]
    # Null placeholders let the production sampler run. Their derived features
    # are excluded from the export/plots, and this script never runs inference.
    observations = observations.with_columns([pl.lit(None, dtype=pl.Float64).alias(name) for name in absent])
    # Read the existing history snapshot directly; open() can refresh/download.
    history = AirportHistory(HistoryPaths.defaults())
    catalog = AirportCatalogView.from_history(history, flight["takeoff_time"].date(), history.latest_available_date)
    endpoint = next(e for e in trajectories(flights, observations, catalog) if e.endpoint == "takeoff")
    values = endpoint.values
    assert endpoint.candidate_idents and "US-8169" in endpoint.candidate_idents
    assert np.isfinite(values).all()
    selected = [endpoint.candidate_idents.index(ident) for ident in ("16XS", "US-6846", "US-8445", "US-1134", "KCXO", "US-8169")]
    start = flight["takeoff_time"]
    end = min(flight["landing_time"], start + timedelta(minutes=10))
    window = clean_messages(observations).filter((pl.col("icao") == flight["icao"]) & (pl.col("time") >= start) & (pl.col("time") <= end))
    elapsed = values[0, :, 10] * 600
    times = [window["time"].min() + timedelta(seconds=float(seconds)) for seconds in elapsed]
    distances = np.expm1(values[:, :, 4] * 5)
    altitude = values[:, :, 5] * 10000
    east, north = values[:, :, 1] * 100, values[:, :, 0] * 100
    args.output.mkdir(parents=True, exist_ok=True)
    candidates = []
    for i, ident in enumerate(endpoint.candidate_idents):
        airport = catalog.id_index[endpoint.candidate_ids[i]]
        candidates.append(dict(candidate_distance_rank=i + 1, airport_ident=ident, airport_id=airport.id, name=airport.name, elevation_ft=airport.elevation_ft, first_distance_km=float(distances[i, 0]), first_baro_height_ft=float(altitude[i, 0]), displayed=i in selected))
    pl.DataFrame(candidates).write_csv(args.output / "candidates.csv")
    rows = []
    for i, ident in enumerate(endpoint.candidate_idents):
        for t, stamp in enumerate(times):
            rows.append(dict(airport_ident=ident, model_sequence_index=t, time_utc=stamp, seconds_after_first_observation=float(elapsed[t]), north_km=float(north[i, t]), east_km=float(east[i, t]), distance_km=float(distances[i, t]), baro_height_above_airport_ft=float(altitude[i, t]), radial_speed_km_s=float(values[i, t, 2] * .2), ground_speed_kt=float(values[i, t, 6] * 300), on_ground=float(values[i, t, 11]), altitude_missing=bool(values[i, t, 14]), speed_missing=bool(values[i, t, 15])))
    pl.DataFrame(rows).write_csv(args.output / "sampled_inputs.csv")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#aeb6bf", "axes.labelcolor": "#344054", "xtick.color": "#667085", "ytick.color": "#667085", "svg.fonttype": "none"})
    plt.rcParams["pdf.fonttype"] = 42
    fig = plt.figure(figsize=(14.5, 9.5), facecolor="white")
    fig.text(.06, .925, f"A96262  ·  ADS-B Exchange  ·  {start:%Y-%m-%d}  ·  {times[0]:%H:%M:%S}-{times[-1]:%H:%M:%S} UTC", fontsize=11.5, color="#475467")
    grid = fig.add_gridspec(2, 3, left=.065, right=.97, bottom=.17, top=.855, hspace=.45, wspace=.26)
    cmap, norm = plt.get_cmap("plasma"), Normalize(0, 10)
    colors = cmap(norm(elapsed / 60))
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
        profile.set(xlim=(-3, max_distance), ylim=(-350, 3000), xlabel="Distance to airport (km)", ylabel="Baro height (ft)", yticks=[0, 1000, 2000, 3000])
    cax = fig.add_axes([.065, .075, .43, .013])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal", ticks=[0, 2, 4, 6, 8, 10])
    cb.set_label("Minutes from first message", fontsize=10)
    cb.ax.tick_params(labelsize=9)
    fig.legend(handles=[Line2D([], [], marker="*", linestyle="", color="#087f72", markersize=13, label="Airport"), Line2D([], [], marker="X", linestyle="", color=colors[0], markeredgecolor="#182230", markersize=9, label="First Message")], loc="center", bbox_to_anchor=(.745, .075), ncol=2, frameon=False, fontsize=10)
    for extension in ("png", "svg", "pdf"):
        fig.savefig(args.output / f"a96262_airport_nn_candidate_inputs.{extension}", dpi=180, facecolor="white")
    plt.close(fig)
    manifest = dict(feature_version=VERSION, observation_path=str(args.observations.resolve()), observation_sha256=sha256(args.observations), flight_path=str(args.flights.resolve()), flight_sha256=sha256(args.flights), catalog=asdict(catalog.provenance), raw_window_observations=window.height, sampled_observations=len(times), candidates=len(candidates), selected=[endpoint.candidate_idents[i] for i in selected], missing_source_columns=absent, model_scores_computed=False, endpoint="takeoff", takeoff_label_provenance="Reviewed departure from US-8169", endpoint_boundary=str(start), note="Production feature builder and sampler; unavailable heading/rate inputs are not displayed or scored. Physical units decoded from normalized float32 inputs.")
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    print(json.dumps(manifest, indent=2, default=str))


if __name__ == "__main__":
    main()
