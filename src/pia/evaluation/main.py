from datetime import timedelta
import os
from pathlib import Path
from data_engineering.flights.flight_type import Flight
from pia.resolver import build_graph
from pia.evaluation.pia_graph_0 import flights_0
import polars as pl
from data_engineering.utils import OUTPUT_DIR
os.environ.setdefault("OUTPUT_DIR", str(Path("output").resolve()))
os.environ.setdefault("MPLCONFIGDIR", str(Path("tmp/matplotlib").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Patch, Rectangle
import networkx as nx


def visualize_graph(G: nx.DiGraph, output_path: Path) -> None:
    if G.number_of_nodes() == 0:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, "No flights in graph", ha="center", va="center", fontsize=16)
        ax.axis("off")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=180)
        plt.close(fig)
        return

    PIA_COLOR = "#2563eb"
    REAL_COLOR = "#64748b"
    SELECTED_EDGE_COLOR = "#0f172a"
    CANDIDATE_EDGE_COLOR = "#94a3b8"
    SOURCE_COLOR = "#16a34a"
    SINK_COLOR = "#dc2626"
    ISOLATED_COLOR = "#7c3aed"
    BAR_HEIGHT = 0.62
    MIN_BAR_WIDTH = timedelta(minutes=20).total_seconds() / 86400

    has_resolved_mapping = any(
        "mapped_icao" in attrs
        for _, attrs in G.nodes(data=True)
    )
    selected_edge_count = sum(
        "mapped_icao" in attrs
        for _, _, attrs in G.edges(data=True)
    )
    has_selected_edges = selected_edge_count > 0
    has_candidate_edges = selected_edge_count < G.number_of_edges()

    def lane_key(node: str) -> str:
        attrs = G.nodes[node]
        flight = attrs["flight"]
        return attrs.get("mapped_icao") or flight.icao

    def node_color(flight: Flight) -> str:
        if flight.pia:
            return PIA_COLOR
        return REAL_COLOR

    def node_border_color(node: str) -> str:
        is_source = G.in_degree(node) == 0
        is_sink = G.out_degree(node) == 0
        if is_source and is_sink:
            return ISOLATED_COLOR
        if is_source:
            return SOURCE_COLOR
        if is_sink:
            return SINK_COLOR
        return "#f8fafc"

    def edge_gap_label(delta: timedelta) -> str:
        minutes = int(delta.total_seconds() // 60)
        if minutes < 60:
            return f"{minutes}m"
        hours = minutes // 60
        remaining_minutes = minutes % 60
        if hours < 24:
            return f"{hours}h {remaining_minutes}m"
        days = hours // 24
        remaining_hours = hours % 24
        return f"{days}d {remaining_hours}h"

    def flight_label(node: str) -> str:
        attrs = G.nodes[node]
        flight = attrs["flight"]
        mapped_icao = attrs.get("mapped_icao")
        if flight.pia and mapped_icao and mapped_icao != flight.icao:
            first_line = f"PIA {flight.icao} -> {mapped_icao}"
        elif flight.pia:
            first_line = f"PIA {flight.icao}"
        else:
            first_line = flight.icao
        return (
            f"{first_line}\n"
            f"{flight.takeoff_airport_ident}->{flight.landing_airport_ident} "
            f"{flight.takeoff_time:%m-%d %H:%M}-{flight.landing_time:%H:%M}"
        )

    lane_stats = {}
    for node, attrs in G.nodes(data=True):
        flight = attrs["flight"]
        key = lane_key(node)
        stats = lane_stats.setdefault(
            key,
            {
                "count": 0,
                "pia_count": 0,
                "first_takeoff": flight.takeoff_time,
                "last_landing": flight.landing_time,
            },
        )
        stats["count"] += 1
        if flight.pia:
            stats["pia_count"] += 1
        stats["first_takeoff"] = min(stats["first_takeoff"], flight.takeoff_time)
        stats["last_landing"] = max(stats["last_landing"], flight.landing_time)

    lane_keys = sorted(
        lane_stats,
        key=lambda key: (
            lane_stats[key]["first_takeoff"],
            key,
        ),
    )
    lane_y = {key: index for index, key in enumerate(lane_keys)}

    starts = [
        mdates.date2num(attrs["flight"].takeoff_time)
        for _, attrs in G.nodes(data=True)
    ]
    ends = [
        mdates.date2num(attrs["flight"].landing_time)
        for _, attrs in G.nodes(data=True)
    ]
    min_time = min(starts)
    max_time = max(ends)
    time_span = max(max_time - min_time, 1 / 24)
    min_visible_bar_width = min(max(MIN_BAR_WIDTH, time_span * 0.006), 6 / 24)

    figure_width = min(max(14, time_span * 0.55), 34)
    figure_height = min(max(7, len(lane_keys) * 0.58 + 3.0), 44)
    fig, ax = plt.subplots(figsize=(figure_width, figure_height))
    ax.set_facecolor("white")

    for index, key in enumerate(lane_keys):
        if index % 2 == 0:
            ax.axhspan(index - 0.5, index + 0.5, color="#f8fafc", zorder=0)

    node_positions = {}
    ordered_nodes = sorted(
        G.nodes,
        key=lambda node: (
            lane_y[lane_key(node)],
            G.nodes[node]["flight"].takeoff_time,
            G.nodes[node]["flight"].icao,
        ),
    )
    show_all_labels = G.number_of_nodes() <= 70

    for node in ordered_nodes:
        attrs = G.nodes[node]
        flight = attrs["flight"]
        start = mdates.date2num(flight.takeoff_time)
        landing = mdates.date2num(flight.landing_time)
        width = max(landing - start, min_visible_bar_width)
        y = lane_y[lane_key(node)]
        node_positions[node] = {
            "takeoff_x": start,
            "landing_x": landing,
            "draw_end_x": start + width,
            "y": y,
        }
        rect = Rectangle(
            (start, y - BAR_HEIGHT / 2),
            width,
            BAR_HEIGHT,
            facecolor=node_color(flight),
            edgecolor=node_border_color(node),
            linewidth=2.0,
            zorder=4,
        )
        ax.add_patch(rect)

        is_important = (
            show_all_labels
            or flight.pia
            or G.in_degree(node) == 0
            or G.out_degree(node) == 0
        )
        if not is_important:
            continue

        label_inside = width >= time_span * 0.035
        if label_inside:
            text_x = start + width / 2
            text_color = "white"
            ha = "center"
            bbox = None
            clip_on = True
        else:
            text_x = start + width + max(time_span * 0.004, 10 / 1440)
            text_color = "#0f172a"
            ha = "left"
            bbox = {
                "boxstyle": "round,pad=0.25",
                "facecolor": "white",
                "edgecolor": "#cbd5e1",
                "linewidth": 0.5,
                "alpha": 0.88,
            }
            clip_on = False

        ax.text(
            text_x,
            y,
            flight_label(node),
            color=text_color,
            fontsize=6.4,
            fontweight="bold",
            ha=ha,
            va="center",
            linespacing=1.15,
            bbox=bbox,
            clip_on=clip_on,
            zorder=6,
        )

    edge_label_limit = 90 if has_selected_edges else 35
    for source, target, attrs in G.edges(data=True):
        source_flight = G.nodes[source]["flight"]
        target_flight = G.nodes[target]["flight"]
        source_pos = node_positions[source]
        target_pos = node_positions[target]
        source_xy = (source_pos["landing_x"], source_pos["y"])
        target_xy = (target_pos["takeoff_x"], target_pos["y"])
        selected = "mapped_icao" in attrs
        color = SELECTED_EDGE_COLOR if selected else CANDIDATE_EDGE_COLOR
        linewidth = 1.4 if selected else 0.8
        alpha = 0.78 if selected else 0.24
        lane_delta = target_pos["y"] - source_pos["y"]
        rad = 0.18 if lane_delta >= 0 else -0.18
        if source_pos["y"] == target_pos["y"]:
            rad = 0.10
        arrow = FancyArrowPatch(
            source_xy,
            target_xy,
            arrowstyle="-|>",
            mutation_scale=7,
            linewidth=linewidth,
            color=color,
            alpha=alpha,
            connectionstyle=f"arc3,rad={rad}",
            zorder=2,
        )
        ax.add_patch(arrow)

        if G.number_of_edges() <= edge_label_limit and (selected or G.number_of_edges() <= 35):
            label_x = (source_xy[0] + target_xy[0]) / 2
            label_y = (source_xy[1] + target_xy[1]) / 2
            if source_xy[1] == target_xy[1]:
                label_y -= 0.34
            gap = target_flight.takeoff_time - source_flight.landing_time
            ax.text(
                label_x,
                label_y,
                edge_gap_label(gap),
                color="#334155",
                fontsize=5.5,
                ha="center",
                va="center",
                bbox={
                    "boxstyle": "round,pad=0.15",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.78,
                },
                zorder=5,
            )

    lane_labels = []
    for key in lane_keys:
        stats = lane_stats[key]
        label = f"{key}  {stats['count']} flights"
        if stats["pia_count"]:
            label += f", {stats['pia_count']} PIA"
        lane_labels.append(label)
    ax.set_yticks(list(range(len(lane_keys))))
    ax.set_yticklabels(lane_labels, fontsize=7 if len(lane_keys) <= 35 else 5.5)
    ax.set_ylim(-0.7, len(lane_keys) - 0.3)
    ax.invert_yaxis()

    padding = max(time_span * 0.025, 1 / 24)
    ax.set_xlim(min_time - padding, max_time + padding)
    locator = mdates.AutoDateLocator(minticks=5, maxticks=12)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax.grid(axis="x", color="#cbd5e1", linewidth=0.7, alpha=0.75)
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", length=0)

    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#cbd5e1")

    title_prefix = (
        "Resolved PIA continuity graph"
        if has_resolved_mapping
        else "Candidate PIA continuity graph"
    )
    ax.set_title(
        (
            f"{title_prefix}\n"
            f"{G.number_of_nodes()} flights, {G.number_of_edges()} edges, "
            f"{len(lane_keys)} identity lanes"
        ),
        loc="left",
        fontsize=14,
        fontweight="bold",
        pad=18,
    )

    legend_handles = [
        Patch(facecolor=PIA_COLOR, edgecolor="none", label="PIA flight"),
        Patch(facecolor=REAL_COLOR, edgecolor="none", label="Real ICAO flight"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="white", markeredgecolor=SOURCE_COLOR, markeredgewidth=2, label="Source"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="white", markeredgecolor=SINK_COLOR, markeredgewidth=2, label="Sink"),
    ]
    if has_selected_edges:
        legend_handles.insert(
            2,
            Line2D(
                [0],
                [0],
                color=SELECTED_EDGE_COLOR,
                lw=1.6,
                label="Selected continuity edge",
            ),
        )
    if has_candidate_edges:
        legend_handles.insert(
            2 if not has_selected_edges else 3,
            Line2D(
                [0],
                [0],
                color=CANDIDATE_EDGE_COLOR,
                lw=1.6,
                label="Candidate continuity edge",
            ),
        )
    ax.legend(
        handles=legend_handles,
        loc="upper right",
        bbox_to_anchor=(1, 1.11),
        ncol=5,
        frameon=False,
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

from pia.evaluation.pia_graph_0 import flights_2, flights_3
def main(flights = flights_3):
    print(f"{len(flights)} nodes")
    pia_flights = [flight for flight in flights if flight.pia]
    G = build_graph(flights)
    output_path = OUTPUT_DIR / Path("tmp/pia_graph_0.png")
    # G, mapping_dict = process_graph(G)
    visualize_graph(G, output_path)
    print(f"outputed to {output_path}")

def evaluation(mapping_dict, gold_dict):
    comparison_rows = []
    for flight_id in sorted(set(gold_dict) | set(mapping_dict)):
        expected_icao = gold_dict.get(flight_id)
        predicted_icao = mapping_dict.get(flight_id)
        comparison_rows.append({
            "flight_id": flight_id,
            "expected_icao": expected_icao,
            "predicted_icao": predicted_icao,
            "correct": expected_icao == predicted_icao,
            "missing_prediction": expected_icao is not None and predicted_icao is None,
            "extra_prediction": expected_icao is None and predicted_icao is not None,
        })

    df_mapping_comparison = pl.DataFrame(comparison_rows)

    expected_count = len(gold_dict)
    predicted_count = len(mapping_dict)
    correct_count = df_mapping_comparison.filter(pl.col("correct")).height

    stats = {
        "expected_mappings": expected_count,
        "predicted_mappings": predicted_count,
        "correct_mappings": correct_count,
        "missing_predictions": df_mapping_comparison.filter(pl.col("missing_prediction")).height,
        "extra_predictions": df_mapping_comparison.filter(pl.col("extra_prediction")).height,
        "accuracy": correct_count / expected_count if expected_count else 0,
    }
    return stats

if __name__ == "__main__":
    main()
