"""Train a new ranker from scratch using February 2026 ADSB.lol only."""

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path

import numpy as np
import torch

from flights.airport_nn.data import CACHE, EUROPE_CACHE, TYPES, fingerprint
from flights.airport_nn.evaluate import (
    files_for,
    iter_parts,
    metrics,
    score,
    write_report,
)
from flights.airport_nn.features import ALTITUDE, ELAPSED, GAP, VERSION
from flights.airport_nn.model import AirportRanker, load_model
from flights.airport_nn.source import DEFAULT_MODEL_PATH

TRAIN_DAYS = [date(2026, 2, i) for i in range(1, 21)]
VALIDATION_DAYS = [date(2026, 2, i) for i in range(21, 28)]
EUROPE_VALIDATION_DAYS = [date(2024, 6, 12), date(2024, 6, 15)]
BUFFER_DAYS = [date(2026, 2, 28)]
TEST_DAYS = [date(2026, 3, 1), date(2024, 6, 1)]


def selection_loss(losses):
    """Equal regions, then equal days: high US traffic cannot swamp Europe."""
    us = np.mean([losses[str(d)] for d in VALIDATION_DAYS])
    europe = np.mean([losses[str(d)] for d in EUROPE_VALIDATION_DAYS])
    loss = float((us + europe) / 2)
    if not np.isfinite(loss):
        raise ValueError("Every validation day must have usable labels and finite loss")
    return loss


def train(
    cache=CACHE,
    output=DEFAULT_MODEL_PATH.parent,
    epochs=10,
    batch_size=128,
    seed=42,
    europe_cache=EUROPE_CACHE,
    train_days=TRAIN_DAYS,
    validation_days=VALIDATION_DAYS,
    europe_validation_days=EUROPE_VALIDATION_DAYS,
):
    if epochs < 1 or batch_size < 1:
        raise ValueError("Epochs and batch size must be positive")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "best.pt").exists() or (output / "model.pt").exists():
        raise FileExistsError(
            "Choose a new output directory; training starts from scratch"
        )
    train_days = list(train_days)
    validation_days = list(validation_days)
    europe_validation_days = list(europe_validation_days)
    default_split = (
        train_days == TRAIN_DAYS
        and validation_days == VALIDATION_DAYS
        and europe_validation_days == EUROPE_VALIDATION_DAYS
    )
    buffer_days = BUFFER_DAYS if default_split else []
    test_days = TEST_DAYS if default_split else []
    splits = (
        train_days
        + validation_days
        + europe_validation_days
        + buffer_days
        + test_days
    )
    if len(splits) != len(set(splits)):
        raise ValueError(
            "Training, validation, buffer and final tests must be disjoint"
        )
    paths = {"train": files_for(cache, train_days)}
    validation_paths = {str(d): files_for(cache, [d]) for d in validation_days} | {
        str(d): files_for(europe_cache, [d]) for d in europe_validation_days
    }
    paths.update(validation_paths)
    # Final tests are run separately only after publishing the selected checkpoint.
    spec = {
        "network": "adsblol",
        "label_sources": ["SFDPS", "BTS"],
        "initialization": "random (fixed nearest-distance prior)",
        "train_dates": list(map(str, train_days)),
        "validation_dates": list(
            map(str, validation_days + europe_validation_days)
        ),
        "validation_regions": {
            "US": list(map(str, validation_days)),
            "Europe": list(map(str, europe_validation_days)),
        },
        "validation_label_sources": ["SFDPS", "BTS", "EUROCONTROL"],
        "buffer_dates": list(map(str, buffer_days)),
        "test_dates": list(map(str, test_days)),
        "selection": "0.5 * mean US daily weighted CE + 0.5 * mean Europe daily weighted CE; weights large=1 medium=2 small=4; retrieval misses retained in accuracy, excluded from CE",
        "test_exposure": "Both final benchmark dates were evaluated in earlier experiments; fixed benchmarks, not untouched tests",
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": 0.001,
        "class_weights": dict(zip(TYPES, (1, 2, 4))),
        "inputs": {
            key: [
                fingerprint(p)
                for path in files
                for p in (
                    path,
                    path.with_suffix(".parquet"),
                )
            ]
            for key, files in paths.items()
        },
        "code_sha256": hashlib.sha256(
            b"".join(p.read_bytes() for p in sorted(Path(__file__).parent.glob("*.py")))
        ).hexdigest(),
    }
    (output / "training.json").write_text(json.dumps(spec, indent=2) + "\n")
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    rng = np.random.default_rng(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AirportRanker().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=1, factor=0.5
    )
    best, stale, history = float("inf"), 0, []
    print(
        f"Training from scratch on {device}; {len(paths['train'])} training parts",
        flush=True,
    )
    for epoch in range(1, epochs + 1):
        model.train()
        shuffled = list(paths["train"])
        rng.shuffle(shuffled)
        total, count = 0.0, 0
        for part_index, (_, x, mask, target, eligible, metadata) in enumerate(
            iter_parts(shuffled), 1
        ):
            types = metadata["airport_type"].to_list()
            if not set(metadata["date"].to_list()) <= set(spec["train_dates"]):
                raise ValueError(
                    "Optimizer input contains a date outside the training split"
                )
            weights = np.array(
                [spec["class_weights"].get(t, 1) for t in types], np.float32
            )
            order = np.flatnonzero(eligible & (target >= 0))
            rng.shuffle(order)
            for start in range(0, len(order), batch_size):
                ix = order[start : start + batch_size]
                values = torch.as_tensor(x[ix], device=device, dtype=torch.float32)
                valid = torch.as_tensor(mask[ix], device=device)
                lengths = valid.sum(-1)[valid.any(-1)]
                if rng.random() < 0.2 and int(lengths.min()) > 8:
                    cut = int(rng.integers(1, 7))
                    values, valid = values[:, :, cut:].clone(), valid[:, :, cut:]
                    elapsed = values[:, :, :1, ELAPSED].clone()
                    values[:, :, :, ELAPSED] -= elapsed
                    values[:, :, :, GAP] += elapsed
                    values[:, :, 0, 2:4] = 0  # no preceding sample after a prefix cut
                noise = torch.randn(len(ix), 1, 1, device=device) * 0.0075
                values[..., ALTITUDE] += noise * (values[..., 18] == 0)
                y = torch.as_tensor(target[ix], device=device, dtype=torch.long)
                w = torch.as_tensor(weights[ix], device=device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(values, valid)
                loss = (
                    torch.nn.functional.cross_entropy(logits, y, reduction="none") * w
                ).sum() / w.sum()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total += float(loss.detach()) * len(ix)
                count += len(ix)
            if part_index % 50 == 0:
                print(
                    f"Epoch {epoch}: {part_index}/{len(shuffled)} parts, {count:,} endpoints",
                    flush=True,
                )
        if not count:
            raise ValueError("No eligible training labels were retrieved")
        losses, summary = {}, {}
        for day, files in validation_paths.items():
            validation, losses[day] = score(model, files, batch_size)
            summary[day] = metrics(validation)
            print(f"Epoch {epoch}: validation {day} loss={losses[day]:.6f}", flush=True)
        train_loss = total / count
        if validation_days and europe_validation_days:
            loss = float(
                (
                    np.mean([losses[str(day)] for day in validation_days])
                    + np.mean(
                        [losses[str(day)] for day in europe_validation_days]
                    )
                )
                / 2
            )
        elif losses:
            loss = float(np.mean(list(losses.values())))
        else:
            loss = train_loss
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": loss,
            "validation": summary,
            "validation_daily_losses": losses,
        }
        history.append(record)
        (output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "validation_loss": loss,
                }
            ),
            flush=True,
        )
        scheduler.step(loss)
        if loss < best:
            best, stale = loss, 0
            checkpoint = {
                "weights": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "hidden_size": model.hidden_size,
                "feature_version": VERSION,
                "epoch": epoch,
                "training": spec,
                "validation": summary,
            }
            temporary = output / "checkpoint.tmp"
            torch.save(checkpoint, temporary)
            temporary.replace(output / "best.pt")
        else:
            stale += 1
            if stale >= 3:
                break
    model = load_model(output / "best.pt", device)
    for day, files in validation_paths.items():
        validation, _ = score(model, files, batch_size)
        write_report(validation, output / "validation" / day)
    (output / "best.pt").replace(output / "model.pt")
    print(
        "Frozen checkpoint published; final benchmark dates were not loaded.",
        flush=True,
    )
    return output / "model.pt"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL_PATH.parent)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--europe-cache", type=Path, default=EUROPE_CACHE)
    parser.add_argument(
        "--train-dates",
        nargs="+",
        type=date.fromisoformat,
        default=TRAIN_DAYS,
    )
    parser.add_argument(
        "--validation-dates",
        nargs="*",
        type=date.fromisoformat,
        default=VALIDATION_DAYS,
    )
    parser.add_argument(
        "--europe-validation-dates",
        nargs="*",
        type=date.fromisoformat,
        default=EUROPE_VALIDATION_DAYS,
    )
    args = parser.parse_args()
    print(
        train(
            args.cache,
            args.output,
            args.epochs,
            europe_cache=args.europe_cache,
            train_days=args.train_dates,
            validation_days=args.validation_dates,
            europe_validation_days=args.europe_validation_dates,
        )
    )
