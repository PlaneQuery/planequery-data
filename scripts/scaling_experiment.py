"""
Scaling experiment: train with increasing num_days and evaluate on a fixed held-out date.

Usage:
    python scripts/scaling_experiment.py \
        --start-date 2025-10-01 \
        --eval-date  2025-09-01 \
        --num-days-list 1 3 7 \
        --epochs 15

Outputs a plot saved to data/model/output/scaling_experiment.png.
"""
import argparse
import subprocess
from datetime import datetime
from pathlib import Path
import os
import matplotlib.pyplot as plt
from flights_ai_model.train import train
from flights_ai_model.inference import save_inference_results
import polars as pl
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "./"))


def run_experiment(
    start_date: datetime,
    eval_date: datetime,
    num_days_list: list[int],
    epochs: int | None = None,
    val_date: datetime | None = None,
    hidden_dim: int = 128,
):

    results = []

    for num_days in num_days_list:
        print(f"\n{'='*60}")
        print(f"Training with num_days={num_days}")
        print(f"{'='*60}")

        model_path, df_flights, stats = train(
            start_date=start_date,
            num_days=num_days,
            epochs=epochs,
            val_date=val_date,
            hidden_dim=hidden_dim
        )

        print(f"\nRunning inference on eval date {eval_date:%Y-%m-%d}…")
        save_inference_results(model_path, eval_date.date(), df_flights, pl.DataFrame([stats]))

        result = {
            "num_days": num_days,
            "precision": stats["precision"],
            "recall": stats["recall"],
            "f1": stats["f1"],
            "n_correctly_matched": stats["n_correctly_matched"],
            "n_predicted": stats["n_predicted"],
            "n_gold": stats["n_gold"],
        }
        results.append(result)
        print(f"  num_days={num_days}  precision={stats['precision']:.3f}  recall={stats['recall']:.3f}  f1={stats['f1']:.3f}")

    return results


def plot_results(results: list[dict], output_path: Path):
    num_days    = [r["num_days"]     for r in results]
    f1s         = [r["f1"]           for r in results]
    precisions  = [r["precision"]    for r in results]
    recalls     = [r["recall"]       for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Model performance vs. training data size", fontsize=13)

    axes[0].plot(num_days, f1s, marker="o", color="steelblue")
    axes[0].set_xlabel("Training days")
    axes[0].set_ylabel("Best val event F1")
    axes[0].set_title("Val F1 (FIRST/LAST)")
    axes[0].set_ylim(0, 1)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(num_days, precisions, marker="o", label="Precision", color="green")
    axes[1].plot(num_days, recalls,    marker="s", label="Recall",    color="orange")
    axes[1].plot(num_days, f1s,        marker="^", label="F1",        color="purple")
    axes[1].set_xlabel("Training days")
    axes[1].set_ylabel("Score")
    axes[1].set_title("Precision / Recall / F1")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    n_predicted = [r["n_predicted"]        for r in results]
    n_matched   = [r["n_correctly_matched"] for r in results]
    n_gold      = [r["n_gold"]             for r in results]
    axes[2].plot(num_days, n_predicted, marker="o", label="Predicted", color="steelblue")
    axes[2].plot(num_days, n_matched,   marker="s", label="Matched",   color="green")
    axes[2].axhline(n_gold[0], color="gray", linestyle="--", label=f"Gold ({n_gold[0]})")
    axes[2].set_xlabel("Training days")
    axes[2].set_ylabel("Flight count")
    axes[2].set_title("Flight counts on eval day")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print(f"\nPlot saved to {output_path}")
    plt.show()

def main(args):
    start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    eval_date  = datetime.strptime(args.eval_date,  "%Y-%m-%d")
    commit_id = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    
    val_date = datetime.strptime(args.val_date, "%Y-%m-%d") if args.val_date else None
    hidden_dim = args.hidden_dim
    results = run_experiment(
        start_date=start_date,
        eval_date=eval_date,
        num_days_list=args.num_days_list,
        epochs=args.epochs,
        val_date=val_date,
        hidden_dim=hidden_dim,
    )
    label = args.label
    # Save raw results as CSV alongside the plot
    index = 0
    out_dir = OUTPUT_DIR / "data/experiments/scaling" / f"{commit_id}_{label}_{hidden_dim}_{index}"
    while out_dir.exists():
        index += 1
        out_dir = OUTPUT_DIR / "data/experiments/scaling" / f"{commit_id}_{label}_{hidden_dim}_{index}"
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.csv"
    pl.DataFrame(results).select(["num_days", "precision", "recall", "f1", "n_correctly_matched", "n_predicted", "n_gold"]).write_csv(results_path)
    print(f"Raw results saved to {results_path}")

    plot_results(results, out_dir / f"results.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scaling experiment: performance vs. num_days")
    parser.add_argument("--start-date",     default="2025-09-01", help="First training day (YYYY-MM-DD)")
    parser.add_argument("--eval-date",      default="2025-08-01", help="Fixed held-out eval day (YYYY-MM-DD)")
    parser.add_argument("--num-days-list",  nargs="+", type=int, default=[1, 3, 7, 15, 30])
    parser.add_argument("--epochs",         type=int,   default=None, help="Epochs (default: auto-scale with data size)")
    parser.add_argument("--val-date",        default=None, help="Fixed validation date YYYY-MM-DD (default: last training day)")
    parser.add_argument("--label",           default="", help="Label for the experiment")
    parser.add_argument("--hidden-dim-list", nargs="+", type=int, default=[128], help="Hidden dimensions to try (default: [128])")
    args = parser.parse_args()
    # TODO: Add learning rates and add so that we go through all combinations of hyperparameters. Also add going through combinations of features.
    hidden_dim_list = args.hidden_dim_list
    for hidden_dim in hidden_dim_list:
        print(f"\n{'#'*80}\nRunning experiment with hidden_dim={hidden_dim}\n{'#'*80}\n")
        args.hidden_dim = hidden_dim
        main(args)
