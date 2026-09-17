"""Train a new ranker from scratch using February 2026 ADSB.lol only."""

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path

import numpy as np
import torch

from flights_airport_nn.data import CACHE, TYPES, fingerprint
from flights_airport_nn.derive_day import DEFAULT_MODEL_PATH
from flights_airport_nn.evaluate import (
    files_for,
    iter_parts,
    metrics,
    score,
    write_report,
)
from flights_airport_nn.features import ALTITUDE, ELAPSED, GAP, VERSION
from flights_airport_nn.model import AirportRanker, load_model

TRAIN_DAYS = [date(2026, 2, i) for i in range(1, 21)]
VALIDATION_DAYS = [date(2026, 2, i) for i in range(21, 25)]
TEST_DAYS = [date(2026, 2, i) for i in range(25, 29)]


def train(
    cache=CACHE, output=DEFAULT_MODEL_PATH.parent, epochs=10, batch_size=128, seed=42
):
    if epochs < 1 or batch_size < 1:
        raise ValueError("Epochs and batch size must be positive")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "best.pt").exists() or (output / "model.pt").exists():
        raise FileExistsError(
            "Choose a new output directory; training starts from scratch"
        )
    paths = {
        "train": files_for(cache, TRAIN_DAYS),
        "validation": files_for(cache, VALIDATION_DAYS),
    }
    # The test data are only scored after checkpoint selection; never loaded into
    # optimizer batches or used for early stopping or parameter selection.
    spec = {
        "network": "adsblol",
        "label_sources": ["SFDPS", "BTS"],
        "initialization": "random (fixed nearest-distance prior)",
        "train_dates": list(map(str, TRAIN_DAYS)),
        "validation_dates": list(map(str, VALIDATION_DAYS)),
        "test_dates": list(map(str, TEST_DAYS)),
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": 0.001,
        "class_weights": dict(zip(TYPES, (1, 2, 4))),
        "inputs": {
            key: [fingerprint(p) for p in files] for key, files in paths.items()
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
        validation, loss = score(model, paths["validation"], batch_size)
        summary = metrics(validation)
        record = {
            "epoch": epoch,
            "train_loss": total / count,
            "validation_loss": loss,
            "validation": summary,
        }
        history.append(record)
        (output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "train_loss": total / count,
                    "validation_loss": loss,
                    **summary["primary"],
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
    validation, _ = score(model, paths["validation"], batch_size)
    write_report(validation, paths["validation"], output / "validation")
    test_files = files_for(cache, TEST_DAYS)
    test, _ = score(model, test_files, batch_size)
    summary = write_report(test, test_files, output / "test")
    (output / "best.pt").replace(output / "model.pt")
    print("Held-out February 25–28:", json.dumps(summary), flush=True)
    return output / "model.pt"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL_PATH.parent)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    print(train(args.cache, args.output, args.epochs))
