"""Endpoint accuracy with retrieval failures in the denominator."""

import numpy as np
import polars as pl
import torch

from flights_airport_nn.dataset import cache_path


def evaluate_dates(
    model, dates, feature_config, batch_size=64, temperature=1.0, excluded_airports=()
):
    records, all_logits, all_targets = [], [], []
    device = next(model.parameters()).device
    model.eval()
    with torch.inference_mode():
        for day in dates:
            with np.load(cache_path(day), allow_pickle=False) as z:
                x, lengths, counts, targets = (
                    z[k] for k in ("x", "lengths", "counts", "target")
                )
                nearest, airports, truth, endpoints, flight_ids = (
                    z[k]
                    for k in ("nearest", "airports", "truth", "endpoint", "flight_id")
                )
                for start in range(0, len(x), batch_size):
                    stop = min(start + batch_size, len(x))
                    values = torch.tensor(
                        x[start:stop, :, :, feature_config.column_indices],
                        dtype=torch.float32,
                        device=device,
                    )
                    mask = (
                        torch.arange(x.shape[2], device=device)[None, None, :]
                        < torch.tensor(lengths[start:stop], device=device)[
                            :, None, None
                        ]
                    )
                    mask = mask & (
                        torch.arange(x.shape[1], device=device)[None, :, None]
                        < torch.tensor(counts[start:stop], device=device)[:, None, None]
                    )
                    logits = model.rank(values, mask).cpu().numpy()
                    valid = (targets[start:stop] >= 0) & ~np.isin(
                        truth[start:stop], list(excluded_airports)
                    )
                    all_logits.append(logits[valid])
                    all_targets.append(targets[start:stop][valid])
                    probability = (
                        torch.from_numpy(logits / temperature).softmax(1).numpy()
                    )
                    order = np.argsort(-logits, axis=1)
                    for j, i in enumerate(range(start, stop)):
                        if not truth[i] or truth[i] in excluded_airports:
                            continue
                        available = counts[i] > 0
                        pred = airports[i, order[j, 0]] if available else ""
                        records.append(
                            {
                                "date": day.isoformat(),
                                "flight_id": flight_ids[i],
                                "icao": flight_ids[i].split("_")[0],
                                "endpoint": endpoints[i],
                                "truth": truth[i],
                                "prediction": pred,
                                "retrieved": bool(targets[i] >= 0),
                                "correct": bool(pred == truth[i]),
                                "top3": bool(
                                    targets[i] >= 0 and targets[i] in order[j, :3]
                                ),
                                "nearest_correct": bool(
                                    available and airports[i, nearest[i]] == truth[i]
                                ),
                                "confidence": float(probability[j].max())
                                if available
                                else 0.0,
                                "distance_km": float(
                                    np.expm1(float(x[i, nearest[i], 0, 4]) * 5)
                                )
                                if available
                                else None,
                            }
                        )
    if not records:
        raise ValueError("Evaluation dates contain no labeled endpoints")
    return (
        pl.DataFrame(records),
        np.concatenate(all_logits),
        np.concatenate(all_targets),
    )


def summarize(frame):
    def stats(df):
        n = df.height
        if not n:
            return {"endpoints": 0}
        correct = df["correct"].sum()
        p = correct / n
        # Wilson binomial interval, reported per endpoint (flights are correlated).
        z = 1.96
        center = (p + z * z / (2 * n)) / (1 + z * z / n)
        margin = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
        return {
            "endpoints": n,
            "accuracy": p,
            "accuracy_interval_95": [center - margin, center + margin],
            "candidate_recall": df["retrieved"].mean(),
            "top3_accuracy": df["top3"].mean(),
            "nearest_accuracy": df["nearest_correct"].mean(),
            "mean_airport_accuracy": df.group_by("truth")
            .agg(pl.col("correct").mean())["correct"]
            .mean(),
        }

    result = {"overall": stats(frame)}
    for endpoint in ("takeoff", "landing"):
        result[endpoint] = stats(frame.filter(pl.col("endpoint") == endpoint))
    result["by_date"] = {
        str(key[0]): stats(df)
        for key, df in frame.partition_by("date", as_dict=True).items()
    }
    if "seen_aircraft" in frame.columns:
        result["unseen_aircraft"] = stats(frame.filter(~pl.col("seen_aircraft")))
        result["unseen_airports"] = stats(frame.filter(~pl.col("seen_airport")))
    result["distant_endpoint"] = stats(frame.filter(pl.col("distance_km") > 10))
    bins = []
    for index in range(10):
        sub = frame.filter((pl.col("confidence") * 10).floor().clip(0, 9) == index)
        if sub.height:
            bins.append(
                {
                    "count": sub.height,
                    "confidence": sub["confidence"].mean(),
                    "accuracy": sub["correct"].mean(),
                }
            )
    result["calibration_bins"] = bins
    result["ece"] = sum(
        b["count"] * abs(b["confidence"] - b["accuracy"]) for b in bins
    ) / max(1, frame.height)
    return result


def fit_temperature(logits, targets):
    if len(targets) == 0:
        raise ValueError("Calibration requires at least one retrieved validation label")
    x = torch.tensor(logits, dtype=torch.float32)
    y = torch.tensor(targets, dtype=torch.long)
    log_t = torch.zeros((), requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=40)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(x / log_t.exp().clamp(0.1, 10), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.detach().exp().clamp(0.1, 10))
