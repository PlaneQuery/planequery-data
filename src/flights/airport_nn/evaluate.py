"""Score frozen cached endpoints, retaining retrieval misses and label quality."""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import torch

from flights.airport_nn.data import CACHE
from flights.airport_nn.model import load_model


def iter_parts(files):
    """Read one bounded part ahead while the GPU processes the current part."""

    def read(path):
        with np.load(path, allow_pickle=False) as z:
            arrays = tuple(z[k] for k in ("x", "mask", "target", "eligible"))
        return (
            path,
            *arrays,
            pl.read_parquet(path.with_suffix(".parquet")),
        )

    paths = iter(files)
    first = next(paths, None)
    if first is None:
        return
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(read, first)
        for path in paths:
            part = pending.result()
            pending = pool.submit(read, path)
            yield part
        yield pending.result()


def files_for(root, days):
    files = []
    for day in days:
        folder = Path(root) / str(day)
        manifest = json.loads((folder / "manifest.json").read_text())
        if (
            manifest.get("date") != str(day)
            or manifest.get("network") != "adsblol"
            or manifest.get("airport_identity") != "ourairports_id"
        ):
            raise ValueError(
                f"Incompatible training cache: {folder}. Expected ADSB.lol data "
                "with stable OurAirports IDs. Rebuild using "
                "python -m flights.airport_nn.data --output <new-cache-directory> "
                "and pass that directory with --cache."
            )
        parts = [folder / f"part-{i:04d}.npz" for i in range(manifest["parts"])]
        if any(not p.exists() or not p.with_suffix(".parquet").exists() for p in parts):
            raise FileNotFoundError(f"Incomplete data preparation: {folder}")
        files.extend(parts)
    return files


def score(model, files, batch_size=128):
    device = next(model.parameters()).device
    frames, loss_sum, sample_count = [], 0.0, 0
    model.eval()
    with torch.inference_mode():
        for _, x, mask, targets, eligible, metadata in iter_parts(files):
            if metadata.is_empty():
                continue
            predicted, confidence = [], []
            for start in range(0, len(x), batch_size):
                values = torch.as_tensor(
                    x[start : start + batch_size], device=device, dtype=torch.float32
                )
                valid = torch.as_tensor(mask[start : start + batch_size], device=device)
                logits = model(values, valid)
                probability = logits.softmax(-1)
                conf, index = probability.max(-1)
                index = index.cpu().numpy()
                available = mask[start : start + batch_size].any((1, 2))
                predicted.extend(np.where(available, index, -1).tolist())
                confidence.extend(np.where(available, conf.cpu().numpy(), 0).tolist())
                keep = eligible[start : start + batch_size] & (
                    targets[start : start + batch_size] >= 0
                )
                if keep.any():
                    y = torch.as_tensor(
                        targets[start : start + batch_size][keep],
                        device=device,
                        dtype=torch.long,
                    )
                    loss_sum += float(
                        torch.nn.functional.cross_entropy(
                            logits[keep], y, reduction="sum"
                        )
                    )
                    sample_count += int(keep.sum())
            candidate_ids = metadata["candidate_ids"].to_list()
            candidate_idents = metadata["candidate_idents"].to_list()
            nn_ids = [
                airports[i] if i >= 0 else None
                for airports, i in zip(candidate_ids, predicted)
            ]
            nn_idents = [
                airports[i] if i >= 0 else ""
                for airports, i in zip(candidate_idents, predicted)
            ]
            frames.append(
                metadata.with_columns(
                    pl.Series("nn_id", nn_ids, dtype=pl.Int64),
                    pl.Series("nn_ident", nn_idents, dtype=pl.String),
                    pl.Series("confidence", confidence),
                ).with_columns(
                    (
                        (pl.col("target") >= 0)
                        & (pl.col("nn_id") == pl.col("truth_id"))
                    ).alias("nn_correct"),
                    (
                        pl.col("truth_id").is_not_null()
                        & (pl.col("nearest_id") == pl.col("truth_id"))
                    ).alias("nearest_correct"),
                )
            )
    return pl.concat(
        frames, how="diagonal_relaxed"
    ), loss_sum / sample_count if sample_count else float("inf")


def metrics(data):
    def rates(frame):
        return {
            "endpoints": frame.height,
            "airports": frame["truth_id"].n_unique(),
            "aircraft": frame["icao"].n_unique(),
            "nn_accuracy": frame["nn_correct"].mean(),
            "nearest_accuracy": frame["nearest_correct"].mean(),
            "candidate_recall": frame.select((pl.col("target") >= 0).mean()).item(),
            "nn_only_correct": frame.filter(
                pl.col("nn_correct") & ~pl.col("nearest_correct")
            ).height,
            "nearest_only_correct": frame.filter(
                ~pl.col("nn_correct") & pl.col("nearest_correct")
            ).height,
        }

    eligible = data.filter(pl.col("eligible"))
    return {"primary": rates(eligible)}


def write_report(data, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    data.write_parquet(output / "endpoints.parquet")
    summary = metrics(data)
    (output / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "metadata.json").write_text(
        json.dumps(
            {
                "airport_identity": "ourairports_id",
                "truth_field": "truth_id",
                "prediction_field": "nn_id",
                "nearest_field": "nearest_id",
                "candidate_field": "candidate_ids",
                "target_definition": "index of truth_id in candidate_ids",
                "display_only_fields": [
                    "truth_ident",
                    "nn_ident",
                    "nearest_ident",
                    "candidate_idents",
                ],
            },
            indent=2,
        )
        + "\n"
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dates", nargs="+", type=date.fromisoformat, required=True)
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    paths = files_for(args.cache, args.dates)
    data, _ = score(load_model(args.model), paths)
    print(json.dumps(write_report(data, args.output), indent=2))
