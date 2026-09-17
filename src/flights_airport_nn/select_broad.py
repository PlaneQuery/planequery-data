"""Select a broad checkpoint by validation airport-group gains, before test scoring."""

import argparse
import json
from datetime import date
from pathlib import Path

import torch

from flights_airport_nn.benchmark import fit_temperature
from flights_airport_nn.inference import load_model
from flights_airport_nn.train_broad import TYPE_NAMES, score


def select(run_dir):
    run_dir = Path(run_dir)
    history = json.loads((run_dir / "history.json").read_text())
    if not history:
        raise ValueError("No validation epochs")

    def gain(row):
        return [row["gains"][name] for name in TYPE_NAMES]

    # Prefer all three groups above nearest, with retained large-airport quality.
    feasible = [
        r
        for r in history
        if min(gain(r)) > 0 and r["validation"]["large_airport"]["accuracy"] >= 0.99
    ]
    if feasible:
        chosen = max(
            feasible,
            key=lambda r: (
                sum(gain(r)) / 3,
                r["validation"]["all"]["accuracy"],
                -r["epoch"],
            ),
        )
        policy = "highest macro group gain among epochs beating nearest in every size group and scoring >=99% on large airports"
    else:
        chosen = max(
            history, key=lambda r: (min(gain(r)), sum(gain(r)) / 3, -r["epoch"])
        )
        policy = "highest weakest-group gain; no epoch met all group constraints"
    src = run_dir / f"candidate_epoch_{chosen['epoch']:02d}.pt"
    checkpoint = torch.load(src, map_location="cpu", weights_only=True)
    model, _ = load_model(src)
    val_dates = [date.fromisoformat(d) for d in checkpoint["validation_dates"]]
    result, (logits, targets) = score(model, val_dates, return_logits=True)
    temperature = fit_temperature(logits, targets)
    checkpoint["temperature"] = temperature
    checkpoint["broad_selection"] = {
        "policy": policy,
        "validation_epoch": chosen["epoch"],
    }
    dest = run_dir / "selected_model.pt"
    if dest.exists():
        raise FileExistsError(dest)
    torch.save(checkpoint, dest)
    details = {
        "policy": policy,
        "selected_epoch": chosen["epoch"],
        "validation": result["metrics"],
        "gains": chosen["gains"],
        "temperature": temperature,
        "source_checkpoint": str(src),
        "selected_checkpoint": str(dest),
    }
    (run_dir / "selection.json").write_text(json.dumps(details, indent=2) + "\n")
    print(json.dumps(details, indent=2), flush=True)
    return dest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    select(parser.parse_args().run_dir)
