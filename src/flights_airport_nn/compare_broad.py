"""Select a conservative nearest-airport fallback on validation, then test fresh dates."""

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from flights_airport_nn.broad_cache import CACHE_ROOT
from flights_airport_nn.inference import load_model
from flights_airport_nn.train_broad import TYPE_NAMES, score

THRESHOLDS = (0.0, 0.5, 0.8, 0.9, 0.99, 0.999, 1.01)


def aligned(*results):
    first = results[0]
    for other in results[1:]:
        for k in (
            "date",
            "flight_id",
            "endpoint",
            "truth_type",
            "truth",
            "target",
            "nearest",
        ):
            if not np.array_equal(first[k], other[k]):
                raise ValueError(f"Model results are not aligned: {k}")


def choices(old, new):
    aligned(old, new)
    nearest = old["nearest"]
    valid = old["nearest_type"]
    rules = {}
    for typ in (-1, 0, 1, 2):
        subset = valid == typ
        if not subset.any():
            continue
        baseline = (nearest[subset] == old["target"][subset]).mean()
        best = (0.0, "nearest", 1.01, baseline)
        for name, result in [("old", old), ("new", new)]:
            for threshold in THRESHOLDS:
                pick = np.where(
                    result["confidence"][subset] >= threshold,
                    result["predicted"][subset],
                    nearest[subset],
                )
                accuracy = (pick == old["target"][subset]).mean()
                # Require a useful improvement before replacing a simple baseline.
                gain = accuracy - baseline
                if gain > best[0] + 0.001:
                    best = (gain, name, threshold, accuracy)
        rules[str(typ)] = {
            "model": best[1],
            "confidence_threshold": best[2],
            "validation_endpoints": int(subset.sum()),
            "validation_accuracy": float(best[3]),
            "nearest_accuracy": float(baseline),
            "gain": float(best[0]),
        }
    return rules


def combine(old, new, rules):
    aligned(old, new)
    selected = old["nearest"].copy()
    source = np.full(len(selected), "nearest", dtype="U7")
    for typ, rule in rules.items():
        result = {"old": old, "new": new}.get(rule["model"])
        if result is None:
            continue
        use = (old["nearest_type"] == int(typ)) & (
            result["confidence"] >= rule["confidence_threshold"]
        )
        selected[use] = result["predicted"][use]
        source[use] = rule["model"]
    return selected, source


def metrics(base, prediction, selection_mask=None):
    valid = base["target"] >= 0
    out = {}
    selection_mask = (
        np.ones(len(valid), bool) if selection_mask is None else selection_mask
    )
    for key, subset in [("all", selection_mask)] + [
        (name, selection_mask & (base["truth_type"] == i))
        for i, name in enumerate(TYPE_NAMES)
    ]:
        n = int(subset.sum())
        out[key] = {
            "endpoints": n,
            "accuracy": float(((prediction == base["target"]) & valid)[subset].mean())
            if n
            else None,
            "nearest_accuracy": float(
                ((base["nearest"] == base["target"]) & valid)[subset].mean()
            )
            if n
            else None,
            "candidate_recall": float(valid[subset].mean()) if n else None,
        }
    return out


def summarize_comparison(old, new, hybrid):
    selected, source = combine(old, new, hybrid)
    return (
        {
            "nearest": metrics(old, old["nearest"]),
            "original_network": metrics(old, old["predicted"]),
            "new_network": metrics(new, new["predicted"]),
            "selected_hybrid": metrics(new, selected),
        },
        selected,
        source,
    )


def evaluate(original, new_model, validation_dates, test_dates, output):
    if set(validation_dates) & set(test_dates):
        raise ValueError("Dates must be disjoint")
    original, new_model_path = Path(original), Path(new_model)
    original_hash = hashlib.sha256(original.read_bytes()).hexdigest()
    new_hash = hashlib.sha256(new_model_path.read_bytes()).hexdigest()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    old_model, _ = load_model(original)
    new_model, _ = load_model(new_model_path)
    used = set(new_model.training_dates) | set(new_model.validation_dates)
    if used & set(test_dates) or tuple(new_model.validation_dates) != tuple(
        validation_dates
    ):
        raise ValueError("Test dates must be new and validation dates must match")
    old_val = score(old_model, validation_dates, old_model.temperature)
    new_val = score(new_model, validation_dates, new_model.temperature)
    rules = choices(old_val, new_val)
    validation, _, _ = summarize_comparison(old_val, new_val, rules)
    print("validation rules", json.dumps(rules, indent=2), flush=True)
    print("validation", json.dumps(validation, indent=2), flush=True)
    (output / "validation_selection.json").write_text(
        json.dumps(
            {
                "dates": [d.isoformat() for d in validation_dates],
                "rules": rules,
                "metrics": validation,
            },
            indent=2,
        )
        + "\n"
    )
    old_test = score(old_model, test_dates, old_model.temperature)
    new_test = score(new_model, test_dates, new_model.temperature)
    test, selected, source = summarize_comparison(old_test, new_test, rules)
    training_dates = json.loads((new_model_path.parent / "config.json").read_text())[
        "train_dates"
    ]
    seen_airports, seen_aircraft = set(), set()
    for day_text in training_dates:
        for path in (CACHE_ROOT / day_text).glob("part-*.npz"):
            with np.load(path, allow_pickle=False) as data:
                seen_airports.update(data["truth"][data["target"] >= 0])
                seen_aircraft.update(
                    flight.split("_")[0] for flight in data["flight_id"]
                )
    unseen_airport = ~np.isin(old_test["truth"], list(seen_airports))
    unseen_aircraft = ~np.isin(
        np.array([flight.split("_")[0] for flight in old_test["flight_id"]]),
        list(seen_aircraft),
    )
    test["unseen_airports"] = {
        "nearest": metrics(old_test, old_test["nearest"], unseen_airport),
        "new_network": metrics(new_test, new_test["predicted"], unseen_airport),
        "selected_hybrid": metrics(new_test, selected, unseen_airport),
    }
    test["unseen_aircraft"] = {
        "nearest": metrics(old_test, old_test["nearest"], unseen_aircraft),
        "new_network": metrics(new_test, new_test["predicted"], unseen_aircraft),
        "selected_hybrid": metrics(new_test, selected, unseen_aircraft),
    }
    frame = pl.DataFrame(
        {
            "date": old_test["date"],
            "flight_id": old_test["flight_id"],
            "endpoint": old_test["endpoint"],
            "truth": old_test["truth"],
            "truth_type": [TYPE_NAMES[t] for t in old_test["truth_type"]],
            "seen_airport": ~unseen_airport,
            "seen_aircraft": ~unseen_aircraft,
            "target": old_test["target"],
            "nearest": old_test["nearest"],
            "original": old_test["predicted"],
            "new": new_test["predicted"],
            "hybrid": selected,
            "selection_source": source,
            "original_confidence": old_test["confidence"],
            "new_confidence": new_test["confidence"],
        }
    )
    frame.write_parquet(output / "test_predictions.parquet")
    by_day = {}
    for day in sorted(set(old_test["date"])):
        b = old_test["date"] == day
        by_day[day] = {
            "nearest": metrics(
                {k: v[b] for k, v in old_test.items() if isinstance(v, np.ndarray)},
                old_test["nearest"][b],
            )["all"],
            "new_network": metrics(
                {k: v[b] for k, v in new_test.items() if isinstance(v, np.ndarray)},
                new_test["predicted"][b],
            )["all"],
            "selected_hybrid": metrics(
                {k: v[b] for k, v in new_test.items() if isinstance(v, np.ndarray)},
                selected[b],
            )["all"],
        }
    report = {
        "dates": [d.isoformat() for d in test_dates],
        "original_model_sha256": original_hash,
        "new_model_sha256": new_hash,
        "validation_rules": rules,
        "by_date": by_day,
        "metrics": test,
        "limitation": "FAA single-source labels, same source type as broad training. Airport identities and aircraft can recur across dates. Dates were withheld from training and checkpoint/fallback selection.",
    }
    if (
        hashlib.sha256(original.read_bytes()).hexdigest() != original_hash
        or hashlib.sha256(new_model_path.read_bytes()).hexdigest() != new_hash
    ):
        raise RuntimeError("A checkpoint changed during evaluation")
    report["models_unchanged"] = True
    (output / "test_metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = [
        "# Fresh-date broad airport comparison",
        "",
        f"Test dates: {', '.join(report['dates'])}. Rules and checkpoint were selected using validation dates only.",
        "",
        "| Truth cohort | Endpoints | Nearest | Original NN | New NN | Validation-selected hybrid |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in ("all",) + TYPE_NAMES:
        n = test["nearest"][name]["endpoints"]
        lines.append(
            f"| {name.replace('_', ' ')} | {n:,} | {test['nearest'][name]['accuracy']:.2%} | {test['original_network'][name]['accuracy']:.2%} | {test['new_network'][name]['accuracy']:.2%} | {test['selected_hybrid'][name]['accuracy']:.2%} |"
        )
    lines += ["", "## Airport identities absent from broad training", ""]
    for name in ("all",) + TYPE_NAMES:
        nearest = test["unseen_airports"]["nearest"][name]
        if nearest["endpoints"]:
            new = test["unseen_airports"]["new_network"][name]
            lines.append(
                f"- {name.replace('_', ' ')}: {nearest['endpoints']:,} endpoints; "
                f"nearest {nearest['accuracy']:.2%}, new network {new['accuracy']:.2%}."
            )
    lines += ["", "## Individual dates", ""]
    for day, rows in by_day.items():
        lines.append(
            f"- {day}: {rows['nearest']['endpoints']:,} endpoints; "
            f"nearest {rows['nearest']['accuracy']:.2%}, "
            f"new network {rows['new_network']['accuracy']:.2%}, "
            f"selected hybrid {rows['selected_hybrid']['accuracy']:.2%}."
        )
    lines += [
        "",
        "Selection rules were frozen on validation data before reading test scores. See `validation_selection.json` for exact thresholds.",
        "",
        report["limitation"],
        "",
    ]
    (output / "results.md").write_text("\n".join(lines))
    print("test", json.dumps(test, indent=2), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-model", type=Path, required=True)
    parser.add_argument("--new-model", type=Path, required=True)
    parser.add_argument(
        "--validation-dates", nargs="+", type=date.fromisoformat, required=True
    )
    parser.add_argument(
        "--test-dates", nargs="+", type=date.fromisoformat, required=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    evaluate(
        args.original_model,
        args.new_model,
        args.validation_dates,
        args.test_dates,
        args.output_dir,
    )
