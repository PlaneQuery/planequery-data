"""Fine-tune the airport ranker with broad FAA supervision and fresh date splits."""

import argparse
import hashlib
import json
import random
from datetime import date
from pathlib import Path

import numpy as np
import torch

from flights_airport_nn.benchmark import fit_temperature
from flights_airport_nn.broad_cache import CACHE_ROOT
from flights_airport_nn.inference import load_model
from flights_airport_nn.utils import device

TYPE_NAMES = ("large_airport", "medium_airport", "small_airport")


def files_for(dates):
    files = []
    for day in dates:
        folder = CACHE_ROOT / day.isoformat()
        parts = sorted(folder.glob("part-*.npz"))
        if not parts:
            raise FileNotFoundError(f"Missing prepared candidates: {day}")
        files.extend(parts)
    return files


def score(model, dates, temperature=1.0, return_logits=False):
    result = {
        "truth_type": [],
        "truth": [],
        "target": [],
        "nearest": [],
        "predicted": [],
        "nearest_type": [],
        "predicted_type": [],
        "confidence": [],
        "flight_id": [],
        "endpoint": [],
        "date": [],
    }
    all_logits = []
    all_targets = []
    model.eval()
    with torch.inference_mode():
        for path in files_for(dates):
            with np.load(path, allow_pickle=False) as z:
                x = z["x"]
                length = z["length"]
                count = z["count"]
                target = z["target"]
                truth_type = z["truth_type"]
                nearest = z["nearest"]
                ids = z["flight_id"]
                endpoints = z["endpoint"]
                truth = z["truth"]
            scores = []
            for start in range(0, len(x), 32):
                values = torch.from_numpy(x[start : start + 32].astype(np.float32)).to(
                    device
                )
                mask = (
                    torch.arange(x.shape[2], device=device)[None, None, :]
                    < torch.from_numpy(length[start : start + 32].astype(np.int64)).to(
                        device
                    )[:, None, None]
                )
                mask = mask & (
                    torch.arange(x.shape[1], device=device)[None, :, None]
                    < torch.from_numpy(count[start : start + 32].astype(np.int64)).to(
                        device
                    )[:, None, None]
                )
                scores.append(model.rank(values, mask).cpu().numpy())
            scores = np.concatenate(scores)
            valid = count > 0
            order = np.argsort(-scores, axis=1)
            prediction = np.where(valid, order[:, 0], -1)
            # Feature columns are small, medium, large; report types as large=0,
            # medium=1, small=2, matching TYPE_NAMES and the truth labels.
            candidate_type = 2 - np.argmax(x[:, :, 0, 13:16], axis=-1)
            nearest_type = np.where(
                nearest >= 0,
                candidate_type[np.arange(len(x)), np.maximum(nearest, 0)],
                -1,
            )
            predicted_type = np.where(
                valid,
                candidate_type[np.arange(len(x)), np.maximum(prediction, 0)],
                -1,
            )
            scaled = scores / temperature
            exp = np.exp(scaled - scaled.max(axis=1, keepdims=True))
            confidence = (exp / exp.sum(axis=1, keepdims=True))[
                np.arange(len(scores)), order[:, 0]
            ]
            confidence[~valid] = 0
            for name, arr in [
                ("truth_type", truth_type),
                ("truth", truth),
                ("target", target),
                ("nearest", nearest),
                ("predicted", prediction),
                ("nearest_type", nearest_type),
                ("predicted_type", predicted_type),
                ("confidence", confidence),
                ("flight_id", ids),
                ("endpoint", endpoints),
            ]:
                result[name].append(arr)
            result["date"].append(np.full(len(x), path.parent.name, dtype="U10"))
            retrieved = (target >= 0) & (truth_type >= 0)
            if return_logits and retrieved.any():
                all_logits.append(scores[retrieved])
                all_targets.append(target[retrieved])
    result = {k: np.concatenate(v) for k, v in result.items()}
    keep = result["truth_type"] >= 0
    for k in result:
        result[k] = result[k][keep]
    result["correct"] = (result["target"] >= 0) & (
        result["predicted"] == result["target"]
    )
    result["nearest_correct"] = (result["target"] >= 0) & (
        result["nearest"] == result["target"]
    )
    result["retrieved"] = result["target"] >= 0

    def metrics(selection):
        if selection.sum() == 0:
            return {"endpoints": 0}
        return {
            "endpoints": int(selection.sum()),
            "accuracy": float(result["correct"][selection].mean()),
            "nearest_accuracy": float(result["nearest_correct"][selection].mean()),
            "candidate_recall": float(result["retrieved"][selection].mean()),
        }

    result["metrics"] = {"all": metrics(np.ones(len(result["truth_type"]), dtype=bool))}
    for i, name in enumerate(TYPE_NAMES):
        result["metrics"][name] = metrics(result["truth_type"] == i)
    if return_logits:
        return result, (np.concatenate(all_logits), np.concatenate(all_targets))
    return result


def train(
    train_dates,
    validation_dates,
    initial_model,
    output,
    epochs=8,
    learning_rate=5e-4,
    seed=42,
    class_weights=(1.0, 2.0, 6.0),
):
    if set(train_dates) & set(validation_dates):
        raise ValueError("Dates must be disjoint")
    if len(class_weights) != 3 or any(weight <= 0 for weight in class_weights):
        raise ValueError("Require positive large, medium, and small airport weights")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.set_num_threads(4)
    model, config = load_model(initial_model)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=0.01
    )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "model.pt").exists():
        raise FileExistsError(output / "model.pt")
    spec = {
        "train_dates": [d.isoformat() for d in train_dates],
        "validation_dates": [d.isoformat() for d in validation_dates],
        "initial_model": str(initial_model),
        "initial_sha256": hashlib.sha256(Path(initial_model).read_bytes()).hexdigest(),
        "epochs": epochs,
        "learning_rate": learning_rate,
        "seed": seed,
        "class_weights": list(class_weights),
        "cache_signatures": {
            d.isoformat(): json.loads(
                (CACHE_ROOT / d.isoformat() / "manifest.json").read_text()
            )["signature"]
            for d in train_dates + validation_dates
        },
    }
    (output / "config.json").write_text(json.dumps(spec, indent=2) + "\n")
    weights = torch.tensor(class_weights, device=device)
    rng = np.random.default_rng(seed)
    best = -float("inf")
    history = []
    stale = 0
    for epoch in range(epochs):
        model.train()
        parts = files_for(train_dates)
        rng.shuffle(parts)
        loss_sum = 0.0
        n = 0
        for part in parts:
            with np.load(part, allow_pickle=False) as z:
                x = z["x"]
                length = z["length"]
                count = z["count"]
                target = z["target"]
                typ = z["truth_type"]
            eligible = np.flatnonzero((target >= 0) & (typ >= 0))
            rng.shuffle(eligible)
            for ix in np.array_split(eligible, max(1, (len(eligible) + 31) // 32)):
                if len(ix) == 0:
                    continue
                values = torch.from_numpy(x[ix].astype(np.float32)).to(device)
                lengths = torch.from_numpy(length[ix].astype(np.int64)).to(device)
                counts = torch.from_numpy(count[ix].astype(np.int64)).to(device)
                mask = (
                    torch.arange(x.shape[2], device=device)[None, None, :]
                    < lengths[:, None, None]
                )
                mask = mask & (
                    torch.arange(x.shape[1], device=device)[None, :, None]
                    < counts[:, None, None]
                )
                if random.random() < 0.15 and lengths.min() > 8:
                    cut = random.randint(1, 6)
                    values = values[:, :, cut:, :]
                    mask = mask[:, :, cut:]
                    # Keep elapsed and coverage-gap features consistent with the cut.
                    elapsed = config.enabled.index("elapsed")
                    gap = config.enabled.index("coverage_gap")
                    removed = values[:, :, :1, elapsed].clone()
                    values[:, :, :, elapsed] -= removed
                    values[:, :, :, gap] += removed
                y = torch.from_numpy(target[ix].astype(np.int64)).to(device)
                w = weights[torch.from_numpy(typ[ix].astype(np.int64)).to(device)]
                logits = model.rank(values, mask)
                loss = (
                    torch.nn.functional.cross_entropy(logits, y, reduction="none") * w
                ).sum() / w.sum()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                loss_sum += float(loss.detach()) * len(ix)
                n += len(ix)
        val = score(model, validation_dates)
        m = val["metrics"]
        gains = [m[t]["accuracy"] - m[t]["nearest_accuracy"] for t in TYPE_NAMES]
        # A balanced date-level objective prevents commercial volume from hiding small-airport failures.
        objective = min(gains) + 0.2 * float(np.mean(gains))
        record = {
            "epoch": epoch + 1,
            "train_loss": loss_sum / max(n, 1),
            "validation": m,
            "gains": dict(zip(TYPE_NAMES, gains)),
            "objective": objective,
        }
        history.append(record)
        (output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        print(json.dumps(record), flush=True)
        old = torch.load(initial_model, map_location="cpu", weights_only=True)
        checkpoint = {
            **old,
            "model_state_dict": {
                k: v.detach().cpu() for k, v in model.state_dict().items()
            },
            "epoch": epoch + 1,
            "temperature": 1.0,
            "train_dates": old.get("train_dates", []) + spec["train_dates"],
            "validation_dates": spec["validation_dates"],
            "broad_training": spec,
        }
        torch.save(checkpoint, output / f"candidate_epoch_{epoch + 1:02d}.pt")
        if objective > best + 1e-4:
            best = objective
            stale = 0
            torch.save(checkpoint, output / "model.pt")
        else:
            stale += 1
            if stale >= 3:
                break
    best_checkpoint = torch.load(
        output / "model.pt", map_location=device, weights_only=True
    )
    model.load_state_dict(best_checkpoint["model_state_dict"])
    val, (logits, targets) = score(model, validation_dates, return_logits=True)
    temperature = fit_temperature(logits, targets)
    best_checkpoint["temperature"] = temperature
    torch.save(best_checkpoint, output / "model.pt")
    (output / "validation_metrics.json").write_text(
        json.dumps({"metrics": val["metrics"], "temperature": temperature}, indent=2)
        + "\n"
    )
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-dates", nargs="+", type=date.fromisoformat, required=True
    )
    parser.add_argument(
        "--validation-dates", nargs="+", type=date.fromisoformat, required=True
    )
    parser.add_argument("--initial-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--class-weights", nargs=3, type=float, default=(1.0, 2.0, 6.0))
    args = parser.parse_args()
    train(
        args.train_dates,
        args.validation_dates,
        args.initial_model,
        args.output_dir,
        epochs=args.epochs,
        class_weights=args.class_weights,
    )
