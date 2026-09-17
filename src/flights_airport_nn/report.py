"""Render saved experiment results without selecting or changing a model."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl
import torch
from matplotlib.ticker import MaxNLocator


def make_report(run_dir, audit_dir=None):
    run_dir = Path(run_dir)
    config = json.loads((run_dir / "config.json").read_text())
    history = json.loads((run_dir / "history.json").read_text())
    metrics = json.loads((run_dir / "test_metrics.json").read_text())
    checkpoint = torch.load(run_dir / "model.pt", map_location="cpu", weights_only=True)
    audit = (
        json.loads((Path(audit_dir) / "unseen_airport_metrics.json").read_text())
        if audit_dir
        else None
    )
    predictions = pl.read_parquet(run_dir / "test_predictions.parquet")
    predictions.filter(~pl.col("correct")).sort(
        "confidence", descending=True
    ).write_csv(run_dir / "test_errors.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    epochs = [r["epoch"] for r in history]
    axes[0].plot(
        epochs,
        [r["train_loss"] for r in history],
        marker="o",
        label="Training (augmented)",
    )
    axes[0].plot(
        epochs, [r["validation_loss"] for r in history], marker="o", label="Validation"
    )
    axes[0].axvline(
        checkpoint["epoch"], color="grey", ls="--", label="Selected checkpoint"
    )
    axes[0].set(
        yscale="log",
        xlabel="Epoch",
        ylabel="Cross-entropy loss",
        title="Training and independent validation dates",
    )
    axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[0].legend()
    keys = ["overall", "takeoff", "landing"]
    labels = ["All test endpoints", "Takeoff", "Landing"]
    error = [(1 - metrics[k]["accuracy"]) * 100 for k in keys]
    baseline = [(1 - metrics[k]["nearest_accuracy"]) * 100 for k in keys]
    positions = list(range(3))
    a = axes[1].bar(
        [p - 0.18 for p in positions],
        baseline,
        width=0.36,
        label="Nearest airport",
        color="#9ca3af",
    )
    b = axes[1].bar(
        [p + 0.18 for p in positions],
        error,
        width=0.36,
        label="Neural network",
        color="#2563eb",
    )
    axes[1].bar_label(a, fmt="%.3f%%", padding=3)
    axes[1].bar_label(b, fmt="%.3f%%", padding=3)
    axes[1].set(
        xticks=positions,
        xticklabels=labels,
        ylabel="Incorrect predictions (%)",
        title="Untouched test dates: filtered US commercial flights",
    )
    axes[1].set_ylim(0, max(baseline + error) * 1.2 + 0.01)
    axes[1].legend()
    fig.savefig(run_dir / "evaluation.png", dpi=180)
    fig.savefig(run_dir / "evaluation.svg")
    plt.close(fig)
    lines = [
        "# Airport neural network: measured results",
        "",
        (
            "These results apply to the existing BTS/SFDPS agreement dataset of US commercial flights, "
            "after its ADS-B proximity/continuity filters and temporal matching to segmented flights. "
            "They do not establish accuracy on all raw segments, general aviation, or worldwide traffic."
        ),
        "",
        f"Training: {config['train_dates'][0]}–{config['train_dates'][-1]} ({len(config['train_dates'])} days).",
        f"Validation: {config['validation_dates'][0]}–{config['validation_dates'][-1]} ({len(config['validation_dates'])} days).",
        f"Test: {metrics['test_dates'][0]}–{metrics['test_dates'][-1]} ({len(metrics['test_dates'])} days).",
        "",
        (
            f"Selected epoch: **{checkpoint['epoch']}** of {len(history)} completed epochs; "
            f"seed {config['seed']}; confidence temperature {checkpoint['temperature']:.4f}. "
            "The checkpoint and temperature were selected without using test dates."
        ),
        "",
        "| Test group | Endpoints | NN accuracy | Nearest accuracy | Macro airport accuracy |",
        "|---|---:|---:|---:|---:|",
    ]
    for key in [
        "overall",
        "takeoff",
        "landing",
        "unseen_aircraft",
        "unseen_airports",
        "distant_endpoint",
    ]:
        row = metrics[key]
        if row["endpoints"]:
            lines.append(
                f"| {key.replace('_', ' ')} | {row['endpoints']:,} | {row['accuracy']:.4%} | "
                f"{row['nearest_accuracy']:.4%} | {row['mean_airport_accuracy']:.4%} |"
            )
        else:
            lines.append(
                f"| {key.replace('_', ' ')} | 0 | Not measured | Not measured | Not measured |"
            )
    overall = metrics["overall"]
    low, high = overall["accuracy_interval_95"]
    lines += [
        "",
        (
            f"Test errors: **{predictions.filter(~pl.col('correct')).height:,}**. "
            f"Top-three accuracy: {overall['top3_accuracy']:.4%}. Candidate recall: {overall['candidate_recall']:.4%}. "
            f"Expected calibration error: {metrics['ece']:.6f}."
        ),
        "",
        (
            f"Endpoint-level 95% Wilson interval: {low:.4%}–{high:.4%}. "
            "This interval treats endpoints as independent and does not account for repeated aircraft/routes or paired flight endpoints."
        ),
        "",
        "## Generalization controls",
        "",
        "- No airport ID, aircraft ID, callsign, route, or absolute coordinate inputs.",
        "- Candidate retrieval never injects the labeled airport; missing candidates count as errors.",
        "- Chronologically separated training, validation, and final test dates.",
        "- Weight decay, dropout, noisy altitude/coverage augmentation, learning-rate reduction, and early stopping.",
        "- Both endpoint types share one scorer; ranking loss compares competing airports within each endpoint.",
        "",
    ]
    if audit:
        row = audit["overall"]
        lines += [
            "## Separate airport holdout audit",
            "",
            (
                f"A separate model excluded {len(audit['excluded_airports'])} airport identities from positive training labels, "
                "validation checkpoint selection, and confidence calibration. Candidate retrieval still included those airports."
            ),
            "",
            (
                f"On {row['endpoints']:,} held-out-airport endpoints: **{row['accuracy']:.4%} accuracy**, "
                f"versus {row['nearest_accuracy']:.4%} for nearest-airport assignment. "
                "This tests transfer within the same commercial-flight population; it does not prove transfer to general aviation."
            ),
            "",
        ]
    lines += [
        "## Files",
        "",
        "- `model.pt`: trained checkpoint, feature version, selected epoch, and temperature.",
        "- `config.json`: dates, hyperparameters, source fingerprints, and code digest.",
        "- `history.json`: training/validation learning curve.",
        "- `test_metrics.json`: subgroup metrics, per-day results, and calibration bins.",
        "- `test_predictions.parquet`: every labeled test endpoint, including mistakes.",
        "- `test_errors.csv`: incorrect predictions sorted by confidence.",
        "- `evaluation.png` / `evaluation.svg`: exportable figures.",
        "",
        "![Learning curves and held-out error rates](evaluation.png)",
        "",
    ]
    (run_dir / "results.md").write_text("\n".join(lines))
    return run_dir / "results.md"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path)
    args = parser.parse_args(argv)
    print(make_report(args.run_dir, args.audit_dir))


if __name__ == "__main__":
    main()
