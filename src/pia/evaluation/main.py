from datetime import datetime, timedelta
import os
from pathlib import Path
from data_engineering.flights.flight_type import TestFlight
from pia.pia import build_graph, process_graph


os.environ.setdefault("OUTPUT_DIR", str(Path("output").resolve()))
os.environ.setdefault("MPLCONFIGDIR", str(Path("tmp/matplotlib").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx


def visualize_graph(G: nx.DiGraph, output_path: Path) -> None:
    def is_pia(flight: TestFlight) -> bool:
        return flight.icao.startswith("pia")

    def flight_label(flight: TestFlight) -> str:
        return (
            f"{flight.flight_id}\n"
            f"{flight.takeoff_airport_ident} -> {flight.landing_airport_ident}\n"
            f"{flight.takeoff_time:%H:%M}-{flight.landing_time:%H:%M}"
        )

    def node_color(flight: TestFlight) -> str:
        if is_pia(flight):
            return "#2563eb"
        return "#64748b"

    def node_border_color(node: str) -> str:
        is_source = G.in_degree(node) == 0
        is_sink = G.out_degree(node) == 0
        if is_source and is_sink:
            return "#7c3aed"
        if is_source:
            return "#16a34a"
        if is_sink:
            return "#dc2626"
        return "white"

    labels = {
        node: flight_label(attrs["flight"])
        for node, attrs in G.nodes(data=True)
    }
    colors = [
        node_color(attrs["flight"])
        for _, attrs in G.nodes(data=True)
    ]
    border_colors = [node_border_color(node) for node in G.nodes]

    ordered_nodes = sorted(
        G.nodes,
        key=lambda node: (
            G.nodes[node]["flight"].takeoff_time,
            G.nodes[node]["flight"].icao,
        ),
    )
    time_groups: dict[datetime, list[str]] = {}
    for node in ordered_nodes:
        time_groups.setdefault(G.nodes[node]["flight"].takeoff_time, []).append(node)

    pos = {}
    for x_index, takeoff_time in enumerate(sorted(time_groups)):
        group = time_groups[takeoff_time]
        y_start = (len(group) - 1) / 2
        for y_index, node in enumerate(group):
            pos[node] = (x_index * 3.0, y_start - y_index)

    fig, ax = plt.subplots(figsize=(13, 8))
    nx.draw_networkx_edges(
        G,
        pos,
        edge_color="#374151",
        arrows=True,
        arrowsize=22,
        connectionstyle="arc3,rad=0.08",
        min_source_margin=34,
        min_target_margin=34,
        width=2.2,
        ax=ax,
    )
    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=colors,
        node_size=250,
        edgecolors=border_colors,
        linewidths=2.5,
        ax=ax,
    )
    for node, (x, y) in pos.items():
        flight = G.nodes[node]["flight"]
        ax.text(
            x,
            y,
            labels[node],
            color="white",
            fontsize=8.5,
            fontweight="bold",
            ha="center",
            va="center",
            linespacing=1.25,
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": node_color(flight),
                "edgecolor": node_border_color(node),
                "linewidth": 2.5,
            },
            zorder=3,
        )
    ax.set_title("PIA Flight Continuity Graph", fontsize=16, pad=16)
    ax.text(
        0.5,
        -0.04,
        "Blue = PIA ICAO, gray = regular ICAO. Green border = source, red border = sink. Edge means landing airport/time can feed next takeoff.",
        transform=ax.transAxes,
        fontsize=10,
        ha="center",
    )
    ax.margins(0.35)
    ax.axis("off")
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

from pia.evaluation.pia_graph_0 import flights_0, test_mapping_dict_0
def main():
    G = build_graph(flights_0)
    output_path = Path("tmp/pia_graph_0.png")
    visualize_graph(G, output_path)
    G, mapping_dict = process_graph(G)
    print(mapping_dict == test_mapping_dict_0)
    
    source_nodes = [node for node in G.nodes if G.in_degree(node) == 0]
    sink_nodes = [node for node in G.nodes if G.out_degree(node) == 0]
    output_path = Path("tmp/pia_graph_1.png")
    visualize_graph(G, output_path)
    print(f"\nWrote visualization to {output_path}")

if __name__ == "__main__":
    main()
