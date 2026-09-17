"""Export a concise figure for the held-out broad airport comparison."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot(report_dir):
    report_dir = Path(report_dir)
    report = json.loads((report_dir / "test_metrics.json").read_text())
    paired = json.loads((report_dir / "paired_analysis.json").read_text())
    groups = ["all", "large_airport", "medium_airport", "small_airport"]
    x = np.arange(len(groups))
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), layout="constrained")
    plt.rcParams.update(
        {"axes.spines.top": False, "axes.spines.right": False, "font.size": 10}
    )
    for offset, key, label, color in [
        (-0.25, "nearest", "Nearest airport", "#e7973e"),
        (0, "original_network", "Original NN", "#8293a1"),
        (0.25, "new_network", "Broadly trained NN", "#256bb1"),
    ]:
        values = [report["metrics"][key][g]["accuracy"] * 100 for g in groups]
        axes[0].bar(x + offset, values, width=0.24, label=label, color=color)
    axes[0].set_xticks(
        x,
        [
            f"{g.replace('_', ' ')}\nn={report['metrics']['nearest'][g]['endpoints']:,}"
            for g in groups
        ],
    )
    axes[0].set_ylim(0, 105)
    axes[0].set_ylabel("Agreement with FAA airport label (%)")
    axes[0].set_title("Fresh June 5–6 holdout")
    axes[0].legend(loc="lower right", fontsize=8)
    gain = [paired["paired_results"][g]["difference"] * 100 for g in groups]
    low = [
        (
            paired["paired_results"][g]["difference"]
            - paired["paired_results"][g]["paired_aircraft_bootstrap_95_ci"][0]
        )
        * 100
        for g in groups
    ]
    high = [
        (
            paired["paired_results"][g]["paired_aircraft_bootstrap_95_ci"][1]
            - paired["paired_results"][g]["difference"]
        )
        * 100
        for g in groups
    ]
    axes[1].barh(x, gain, color="#256bb1")
    axes[1].errorbar(gain, x, xerr=[low, high], fmt="none", color="#193047", capsize=4)
    axes[1].set_yticks(x, [g.replace("_", " ") for g in groups])
    axes[1].invert_yaxis()
    axes[1].axvline(0, color="#555", lw=0.8)
    axes[1].set_xlim(-1, 14)
    axes[1].set_xlabel("New network minus nearest (percentage points)")
    axes[1].set_title("Paired aircraft bootstrap, 95% interval")
    fig.suptitle(
        "Airport prediction after adding small-airport training examples\nFull-coverage FAA source test, 66,098 endpoints",
        fontsize=13,
    )
    fig.savefig(report_dir / "heldout-performance.png", dpi=180)
    fig.savefig(report_dir / "heldout-performance.svg")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, required=True)
    plot(parser.parse_args().report_dir)
